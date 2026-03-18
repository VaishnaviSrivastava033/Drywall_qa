"""
Evaluate predicted masks against ground-truth masks.

Usage:
    python src/evaluate.py \
        --manifest  data/combined_test.jsonl \
        --pred_dir  outputs/ \
        --report    reports/eval_report.json

Computes per-prompt and overall mIoU and Dice on the test split.
Also generates visual side-by-sides (orig | GT | pred) for the report.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_pred(pred_dir, image_id, prompt):
    prompt_tag = prompt.replace(" ", "_")
    path = Path(pred_dir) / f"{image_id}__{prompt_tag}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.float32)


def load_gt(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.float32)


def iou_dice(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    iou   = (inter + eps) / (union + eps)
    dice  = (2 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps)
    return float(iou), float(dice)


def make_visual(image_path, gt_mask, pred_mask, out_path, prompt, iou, dice):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H, W = image.shape[:2]
    gt_up   = cv2.resize(gt_mask.astype(np.uint8),   (W, H), interpolation=cv2.INTER_NEAREST)
    pred_up = cv2.resize(pred_mask.astype(np.uint8),  (W, H), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Prompt: "{prompt}" | IoU={iou:.3f} | Dice={dice:.3f}',
                 fontsize=11, y=1.02)

    axes[0].imshow(image);         axes[0].set_title("Original",    fontsize=10)
    axes[1].imshow(gt_up,  cmap="gray"); axes[1].set_title("Ground Truth", fontsize=10)
    axes[2].imshow(pred_up, cmap="gray"); axes[2].set_title("Prediction",   fontsize=10)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--report",   default="reports/eval_report.json")
    ap.add_argument("--visual_dir", default="reports/visuals")
    ap.add_argument("--n_visuals",  type=int, default=4,
                    help="Number of visual examples to save per prompt")
    args = ap.parse_args()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.visual_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for line in Path(args.manifest).read_text().strip().splitlines():
        if line.strip():
            records.append(json.loads(line))

    results_by_prompt = {}
    visual_count = {}

    for rec in records:
        prompt   = rec["prompt"]
        image_id = rec["image_id"]

        pred_bin = load_pred(args.pred_dir, image_id, prompt)
        if pred_bin is None:
            print(f"  [warn] No prediction for {image_id} / '{prompt}'")
            continue

        gt_bin  = load_gt(rec["mask_path"])

        # Resize to same size for fair comparison
        h_pred, w_pred = pred_bin.shape
        h_gt,   w_gt   = gt_bin.shape
        if (h_pred, w_pred) != (h_gt, w_gt):
            pred_bin = cv2.resize(pred_bin, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)

        iou, dice = iou_dice(pred_bin, gt_bin)

        results_by_prompt.setdefault(prompt, {"iou": [], "dice": []})
        results_by_prompt[prompt]["iou"].append(iou)
        results_by_prompt[prompt]["dice"].append(dice)

        # Save visual examples
        vc = visual_count.get(prompt, 0)
        if vc < args.n_visuals:
            vis_path = Path(args.visual_dir) / f"{prompt.replace(' ','_')}_{vc+1}.png"
            make_visual(rec["image_path"], gt_bin, pred_bin,
                        str(vis_path), prompt, iou, dice)
            visual_count[prompt] = vc + 1

    # Aggregate
    summary = {}
    all_iou, all_dice = [], []
    print("\n=== Evaluation Results ===")
    for prompt, vals in results_by_prompt.items():
        m_iou  = float(np.mean(vals["iou"]))
        m_dice = float(np.mean(vals["dice"]))
        summary[prompt] = {"mIoU": round(m_iou, 4), "Dice": round(m_dice, 4),
                            "n": len(vals["iou"])}
        all_iou.extend(vals["iou"])
        all_dice.extend(vals["dice"])
        print(f"  '{prompt}':  mIoU={m_iou:.4f}  Dice={m_dice:.4f}  n={len(vals['iou'])}")

    overall = {"mIoU": round(float(np.mean(all_iou)), 4),
               "Dice": round(float(np.mean(all_dice)), 4),
               "n":    len(all_iou)}
    summary["overall"] = overall
    print(f"\n  Overall:         mIoU={overall['mIoU']:.4f}  Dice={overall['Dice']:.4f}")

    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved: {args.report}")
    print(f"Visuals in:   {args.visual_dir}/")


if __name__ == "__main__":
    main()
