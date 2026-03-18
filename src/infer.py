"""
Run inference on a directory of images or a manifest file.

Usage:
    # Single image
    python src/infer.py \
        --input path/to/image.jpg \
        --prompt "segment crack" \
        --model_dir checkpoints/best_model \
        --output_dir outputs/

    # Whole manifest (generates one mask per record)
    python src/infer.py \
        --manifest data/combined_test.jsonl \
        --model_dir checkpoints/best_model \
        --output_dir outputs/

    # Image directory with multiple prompts
    python src/infer.py \
        --image_dir path/to/images/ \
        --prompts "segment crack" "segment taping area" \
        --model_dir checkpoints/best_model \
        --output_dir outputs/

Output masks:
    PNG, single-channel, same spatial size as source, values {0,255}.
    Filename format: <image_id>__<prompt_with_underscores>.png
    e.g.  img_001__segment_crack.png
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

THRESHOLD = 0.5
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_model(model_dir, device):
    processor = CLIPSegProcessor.from_pretrained(model_dir)
    model     = CLIPSegForImageSegmentation.from_pretrained(model_dir)
    model.eval().to(device)
    return model, processor


@torch.no_grad()
def predict_mask(model, processor, image_bgr, prompt, device, threshold=THRESHOLD):
    """
    Returns binary mask (np.uint8 {0,255}) at the original image resolution.
    """
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rs  = cv2.resize(image_rgb, (352, 352), interpolation=cv2.INTER_LINEAR)

    enc = processor(text=[prompt], images=image_rs,
                    return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    out    = model(**enc)
    logits = out.logits   # (1, H', W')
    prob   = torch.sigmoid(logits[0])   # (H', W')

    # Upsample to original size
    prob_up = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    mask = (prob_up > threshold).astype(np.uint8) * 255
    return mask, prob_up


def save_mask(mask, out_dir, image_id, prompt):
    prompt_tag = prompt.replace(" ", "_")
    out_path   = Path(out_dir) / f"{image_id}__{prompt_tag}.png"
    cv2.imwrite(str(out_path), mask)
    return out_path


def process_manifest(manifest_path, model, processor, device, output_dir):
    records = []
    for line in Path(manifest_path).read_text().strip().splitlines():
        if line.strip():
            records.append(json.loads(line))

    times = []
    for rec in records:
        image_bgr = cv2.imread(rec["image_path"])
        if image_bgr is None:
            print(f"  [warn] Cannot read: {rec['image_path']}")
            continue

        t0 = time.perf_counter()
        mask, _ = predict_mask(model, processor, image_bgr,
                               rec["prompt"], device)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        out = save_mask(mask, output_dir, rec["image_id"], rec["prompt"])
        print(f"  {out.name}  ({elapsed*1000:.1f} ms)")

    if times:
        avg = sum(times) / len(times)
        print(f"\nDone. {len(times)} masks | avg inference: {avg*1000:.1f} ms/image")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",  default="checkpoints/best_model")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--threshold",  type=float, default=THRESHOLD)
    # Input options (mutually exclusive)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--manifest",  help="JSONL manifest (uses prompt per record)")
    grp.add_argument("--input",     help="Single image path")
    grp.add_argument("--image_dir", help="Directory of images")
    # Prompt for single / dir modes
    ap.add_argument("--prompt",  default="segment crack")
    ap.add_argument("--prompts", nargs="+",
                    default=["segment crack", "segment taping area"])
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_dir} on {device}...")
    model, processor = load_model(args.model_dir, device)

    if args.manifest:
        process_manifest(args.manifest, model, processor, device, args.output_dir)

    elif args.input:
        image_bgr = cv2.imread(args.input)
        if image_bgr is None:
            raise FileNotFoundError(args.input)
        image_id = Path(args.input).stem
        t0   = time.perf_counter()
        mask, prob = predict_mask(model, processor, image_bgr,
                                  args.prompt, device, args.threshold)
        elapsed = time.perf_counter() - t0
        out = save_mask(mask, args.output_dir, image_id, args.prompt)
        print(f"Saved: {out}  ({elapsed*1000:.1f} ms)")

    elif args.image_dir:
        img_paths = [p for p in sorted(Path(args.image_dir).iterdir())
                     if p.suffix.lower() in IMG_EXTS]
        for img_path in img_paths:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            for prompt in args.prompts:
                t0 = time.perf_counter()
                mask, _ = predict_mask(model, processor, image_bgr,
                                       prompt, device, args.threshold)
                elapsed = time.perf_counter() - t0
                out = save_mask(mask, args.output_dir, img_path.stem, prompt)
                print(f"  {out.name}  ({elapsed*1000:.1f} ms)")


if __name__ == "__main__":
    main()
