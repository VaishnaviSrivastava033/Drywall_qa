"""
Download and prepare datasets from Roboflow.

Usage:
    python src/download_data.py --api_key YOUR_ROBOFLOW_API_KEY

Datasets:
    Dataset 1 - Drywall-Join-Detect (taping areas):
        workspace: objectdetect-pu6rn / project: drywall-join-detect
    Dataset 2 - Cracks-3ii36 (cracks):
        workspace: fyp-ny1jt / project: cracks-3ii36

Both are downloaded in COCO Segmentation format so we can extract
polygon masks to binary PNG masks.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def download_roboflow(workspace, project, version, api_key, out_dir, fmt="coco"):
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("Install roboflow: pip install roboflow")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt, location=str(out_dir))
    return dataset


def coco_to_binary_masks(coco_json, image_dir, mask_dir, prompt_tag):
    """
    Converts COCO annotations to binary masks.
    Handles both segmentation polygons AND bounding boxes (object-detection datasets).
    For bbox-only annotations, the filled rectangle becomes the mask region.
    """
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(coco_json).read_text())

    ann_by_img = {}
    for ann in data.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    records = []
    for img_info in data["images"]:
        iid   = img_info["id"]
        fname = img_info["file_name"]
        H     = img_info["height"]
        W     = img_info["width"]

        img_path = Path(image_dir) / fname
        if not img_path.exists():
            img_path = Path(image_dir) / Path(fname).name
        if not img_path.exists():
            print(f"  [warn] image not found: {fname}")
            continue

        mask = np.zeros((H, W), dtype=np.uint8)
        for ann in ann_by_img.get(iid, []):
            segs = ann.get("segmentation", [])

            # --- Try polygon segmentation first ---
            if isinstance(segs, list) and len(segs) > 0:
                for seg in segs:
                    if len(seg) < 6:
                        continue
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

            # --- RLE segmentation ---
            elif isinstance(segs, dict):
                try:
                    from pycocotools import mask as coco_mask
                    rle = coco_mask.frPyObjects(segs, H, W)
                    m   = coco_mask.decode(rle)
                    mask = np.maximum(mask, (m * 255).astype(np.uint8))
                except ImportError:
                    print("  [warn] RLE mask found but pycocotools not installed")

            # --- Fallback: use bounding box as mask region ---
            bbox = ann.get("bbox")
            if bbox and mask.max() == 0:
                x, y, bw, bh = [int(v) for v in bbox]
                x2 = min(x + bw, W)
                y2 = min(y + bh, H)
                mask[y:y2, x:x2] = 255

        stem = Path(fname).stem
        mask_name = f"{stem}__{prompt_tag}.png"
        cv2.imwrite(str(mask_dir / mask_name), mask)

        records.append({
            "image_path": str(img_path),
            "mask_path":  str(mask_dir / mask_name),
            "prompt":     prompt_tag.replace("_", " "),
            "image_id":   stem,
        })

    return records


def build_manifest(records, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Manifest: {out_path}  ({len(records)} records)")


def process_dataset(raw_dir, out_dir, primary_prompt, split="train"):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    coco_json = None
    image_dir = None

    for s in [split, split.capitalize()]:
        candidate = raw_dir / s / "_annotations.coco.json"
        if candidate.exists():
            coco_json = candidate
            image_dir = candidate.parent
            break

    if coco_json is None:
        hits = list(raw_dir.rglob("_annotations.coco.json"))
        for h in hits:
            if split in str(h):
                coco_json = h
                image_dir = h.parent
                break
        else:
            if hits:
                coco_json = hits[0]
                image_dir = coco_json.parent

    if coco_json is None:
        print(f"  [warn] No COCO JSON found under {raw_dir} for split={split}")
        return []

    prompt_tag = primary_prompt.replace(" ", "_")
    mask_dir   = out_dir / "masks" / split
    records    = coco_to_binary_masks(coco_json, image_dir, mask_dir, prompt_tag)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True, help="Roboflow API key")
    ap.add_argument("--skip_download", action="store_true")
    args = ap.parse_args()

    d1_raw = DATA_DIR / "dataset1" / "raw"
    if not args.skip_download:
        print("Downloading Dataset 1 (taping areas)...")
        download_roboflow("objectdetect-pu6rn", "drywall-join-detect", 1,
                          args.api_key, d1_raw)

    for split in ["train", "valid", "test"]:
        recs = process_dataset(d1_raw, DATA_DIR / "dataset1", "segment taping area", split)
        build_manifest(recs, DATA_DIR / "dataset1" / f"manifest_{split}.jsonl")

    d2_raw = DATA_DIR / "dataset2" / "raw"
    if not args.skip_download:
        print("Downloading Dataset 2 (cracks)...")
        download_roboflow("university-bswxt", "crack-bphdr", 2,
                          args.api_key, d2_raw, fmt="coco-segmentation")

    for split in ["train", "valid", "test"]:
        recs = process_dataset(d2_raw, DATA_DIR / "dataset2", "segment crack", split)
        build_manifest(recs, DATA_DIR / "dataset2" / f"manifest_{split}.jsonl")

    for split in ["train", "valid", "test"]:
        m1 = DATA_DIR / "dataset1" / f"manifest_{split}.jsonl"
        m2 = DATA_DIR / "dataset2" / f"manifest_{split}.jsonl"
        combined = []
        for mf in [m1, m2]:
            if mf.exists():
                combined.extend(mf.read_text().strip().splitlines())
        out = DATA_DIR / f"combined_{split}.jsonl"
        out.write_text("\n".join(combined))
        print(f"Combined {split}: {len(combined)} records")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()