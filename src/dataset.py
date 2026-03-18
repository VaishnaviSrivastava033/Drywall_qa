"""
PyTorch Dataset for prompted segmentation.

Each sample returns:
    image       : Tensor (3, H, W)  -- CLIPSeg expects 352x352
    mask        : Tensor (1, H, W)  -- float32 {0,1}
    prompt      : str
    image_id    : str
    image_path  : str
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPSegProcessor

# Fixed seed helpers are in train.py; this module is seed-agnostic.

IMG_SIZE = 352  # CLIPSeg native resolution

# Multiple prompt variants for augmentation during training
PROMPT_VARIANTS = {
    "segment taping area": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment drywall joint",
        "tape joint on drywall",
    ],
    "segment crack": [
        "segment crack",
        "segment wall crack",
        "segment surface crack",
        "crack in wall",
        "crack defect",
    ],
}


def get_variants(prompt):
    # Normalise: find which canonical key this prompt belongs to
    for canon, variants in PROMPT_VARIANTS.items():
        if prompt in variants or prompt == canon:
            return variants
    return [prompt]


class DrywallDataset(Dataset):
    def __init__(self, manifest_path, processor=None,
                 augment=False, prompt_aug=False, seed=42):
        self.records     = self._load(manifest_path)
        self.processor   = processor or CLIPSegProcessor.from_pretrained(
                                "CIDAS/clipseg-rd64-refined")
        self.augment     = augment
        self.prompt_aug  = prompt_aug
        self.rng         = random.Random(seed)

    @staticmethod
    def _load(path):
        records = []
        for line in Path(path).read_text().strip().splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = cv2.imread(rec["image_path"])
        if image is None:
            raise FileNotFoundError(rec["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(rec["mask_path"])
        mask = (mask > 127).astype(np.float32)  # {0,1}

        # ── Augmentations ─────────────────────────────────────────────────────
        if self.augment:
            image, mask = self._augment(image, mask)

        # ── Resize to CLIPSeg native resolution ───────────────────────────────
        image_res = cv2.resize(image, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)
        mask_res  = cv2.resize(mask,  (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_NEAREST)

        # ── Prompt selection ──────────────────────────────────────────────────
        prompt = rec["prompt"]
        if self.prompt_aug:
            variants = get_variants(prompt)
            prompt   = self.rng.choice(variants)

        # ── Processor: handles CLIP image normalisation + text tokenisation ───
        enc = self.processor(
            text=[prompt],
            images=image_res,
            return_tensors="pt",
            padding=True,
        )
        # enc keys: pixel_values (1,3,352,352), input_ids (1,L), attention_mask (1,L)

        return {
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "mask":           torch.tensor(mask_res, dtype=torch.float32).unsqueeze(0),
            "prompt":         prompt,
            "image_id":       rec["image_id"],
            "image_path":     rec["image_path"],
            "orig_h":         image.shape[0],
            "orig_w":         image.shape[1],
        }

    def _augment(self, image, mask):
        # Horizontal flip
        if self.rng.random() < 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()

        # Vertical flip
        if self.rng.random() < 0.3:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()

        # Random 90-degree rotation
        if self.rng.random() < 0.3:
            k = self.rng.randint(1, 3)
            image = np.rot90(image, k).copy()
            mask  = np.rot90(mask, k).copy()

        # Colour jitter (brightness + contrast)
        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.7, 1.3)   # contrast
            beta  = self.rng.randint(-20, 20)     # brightness
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        return image, mask
