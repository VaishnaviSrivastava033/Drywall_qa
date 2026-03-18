"""
Fine-tune CLIPSeg for prompted drywall segmentation.

Usage:
    python src/train.py \
        --train_manifest data/combined_train.jsonl \
        --val_manifest   data/combined_valid.jsonl \
        --output_dir     checkpoints/ \
        --epochs         20 \
        --batch_size     8 \
        --lr             5e-5 \
        --seed           42

CLIPSeg (CIDAS/clipseg-rd64-refined) is a CLIP-based model that conditions
on a text prompt to produce a segmentation logit map. We fine-tune the full
model (decoder + CLIP backbone) with a combined BCE + Dice loss.

Loss  = 0.5 * BCE(pred, gt) + 0.5 * DiceLoss(pred, gt)

Training saves:
    checkpoints/best_model/   -- best val mIoU checkpoint
    checkpoints/last_model/   -- last-epoch checkpoint
    checkpoints/train_log.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# Local
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import DrywallDataset


# ── Reproducibility ────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Loss ──────────────────────────────────────────────────────────────────────
def dice_loss(pred_sigmoid, target, eps=1e-6):
    pred   = pred_sigmoid.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)


def combined_loss(logits, target, bce_weight=0.5):
    pred_sigmoid = torch.sigmoid(logits)
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(pred_sigmoid, target)
    return bce_weight * bce + (1 - bce_weight) * dice


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds_bin, targets_bin, eps=1e-6):
    """preds_bin, targets_bin: bool tensors on CPU."""
    inter = (preds_bin & targets_bin).float().sum()
    union = (preds_bin | targets_bin).float().sum()
    iou   = (inter + eps) / (union + eps)
    dice  = (2 * inter + eps) / (preds_bin.float().sum() + targets_bin.float().sum() + eps)
    return iou.item(), dice.item()


# ── Collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_padded, attention_mask_padded = [], []
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        input_ids_padded.append(torch.nn.functional.pad(b["input_ids"], (0, pad), value=0))
        attention_mask_padded.append(torch.nn.functional.pad(b["attention_mask"], (0, pad), value=0))
    return {
        "pixel_values":   torch.stack([b["pixel_values"] for b in batch]),
        "input_ids":      torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "mask":           torch.stack([b["mask"]         for b in batch]),
    }


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for step, batch in enumerate(loader):
        pv   = batch["pixel_values"].to(device)
        iids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask = batch["mask"].to(device)         # (B,1,H,W)

        # CLIPSeg forward – outputs logits (B, H, W) at 64x64 resolution
        out    = model(pixel_values=pv, input_ids=iids, attention_mask=attn)
        logits = out.logits                      # (B, H, W)

        # Upsample mask to logit resolution for loss
        logit_h, logit_w = logits.shape[-2], logits.shape[-1]
        mask_ds = F.interpolate(mask, size=(logit_h, logit_w), mode="nearest")
        mask_ds = mask_ds.squeeze(1)             # (B, H, W)

        loss = combined_loss(logits, mask_ds)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1} step {step+1}/{len(loader)} "
                  f"loss={loss.item():.4f} elapsed={elapsed:.1f}s")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_iou, all_dice = [], []
    total_loss = 0.0
    for batch in loader:
        pv   = batch["pixel_values"].to(device)
        iids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask = batch["mask"].to(device)

        out    = model(pixel_values=pv, input_ids=iids, attention_mask=attn)
        logits = out.logits

        logit_h, logit_w = logits.shape[-2], logits.shape[-1]
        mask_ds = F.interpolate(mask, size=(logit_h, logit_w), mode="nearest").squeeze(1)

        loss = combined_loss(logits, mask_ds)
        total_loss += loss.item()

        pred_sigmoid = torch.sigmoid(logits)
        pred_bin     = (pred_sigmoid > threshold).cpu()
        gt_bin       = (mask_ds > 0.5).cpu()

        for p, g in zip(pred_bin, gt_bin):
            iou, dice = compute_metrics(p, g)
            all_iou.append(iou)
            all_dice.append(dice)

    return {
        "loss": total_loss / len(loader),
        "mIoU": float(np.mean(all_iou)),
        "Dice": float(np.mean(all_dice)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", default="data/combined_train.jsonl")
    ap.add_argument("--val_manifest",   default="data/combined_valid.jsonl")
    ap.add_argument("--output_dir",     default="checkpoints")
    ap.add_argument("--model_name",     default="CIDAS/clipseg-rd64-refined")
    ap.add_argument("--epochs",         type=int,   default=20)
    ap.add_argument("--batch_size",     type=int,   default=8)
    ap.add_argument("--lr",             type=float, default=5e-5)
    ap.add_argument("--weight_decay",   type=float, default=1e-4)
    ap.add_argument("--warmup_steps",   type=int,   default=50)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--num_workers",    type=int,   default=4)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model & processor ────────────────────────────────────────────────────
    processor = CLIPSegProcessor.from_pretrained(args.model_name)
    model     = CLIPSegForImageSegmentation.from_pretrained(args.model_name)
    model     = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = DrywallDataset(args.train_manifest, processor=processor,
                              augment=True, prompt_aug=True, seed=args.seed)
    val_ds   = DrywallDataset(args.val_manifest,   processor=processor,
                              augment=False, prompt_aug=False, seed=args.seed)

    # num_workers=0 required on Windows; pin_memory only useful with GPU
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=pin)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────────
    log_path  = output_dir / "train_log.jsonl"
    best_miou = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        t_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time  = time.time() - t_start

        scheduler.step()

        log_entry = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_metrics["loss"], 4),
            "val_mIoU":   round(val_metrics["mIoU"], 4),
            "val_Dice":   round(val_metrics["Dice"], 4),
            "epoch_time": round(epoch_time, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_mIoU={val_metrics['mIoU']:.4f} | "
              f"val_Dice={val_metrics['Dice']:.4f} | "
              f"time={epoch_time:.0f}s")

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            model.save_pretrained(str(output_dir / "best_model"))
            processor.save_pretrained(str(output_dir / "best_model"))
            print(f"  ** New best mIoU: {best_miou:.4f} — checkpoint saved")

    model.save_pretrained(str(output_dir / "last_model"))
    processor.save_pretrained(str(output_dir / "last_model"))
    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()