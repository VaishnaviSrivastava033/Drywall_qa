"""
Generates training curve plots from train_log.jsonl for the final report.

Usage:
    python src/generate_report_figures.py \
        --log  checkpoints/train_log.jsonl \
        --out  reports/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path):
    entries = []
    for line in Path(log_path).read_text().strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def plot_training_curves(entries, out_dir):
    epochs     = [e["epoch"]      for e in entries]
    train_loss = [e["train_loss"] for e in entries]
    val_loss   = [e["val_loss"]   for e in entries]
    val_miou   = [e["val_mIoU"]   for e in entries]
    val_dice   = [e["val_Dice"]   for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, train_loss, label="Train loss", color="#2563eb", linewidth=2)
    axes[0].plot(epochs, val_loss,   label="Val loss",   color="#dc2626", linewidth=2,
                 linestyle="--")
    axes[0].set_title("Loss (BCE + Dice)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Metrics
    axes[1].plot(epochs, val_miou, label="Val mIoU", color="#059669", linewidth=2)
    axes[1].plot(epochs, val_dice, label="Val Dice", color="#7c3aed", linewidth=2,
                 linestyle="--")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="checkpoints/train_log.jsonl")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    entries = load_log(args.log)
    plot_training_curves(entries, args.out)
    print("Figures generated.")


if __name__ == "__main__":
    main()
