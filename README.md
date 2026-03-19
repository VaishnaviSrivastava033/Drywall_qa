# Prompted Segmentation for Drywall QA

Binary segmentation of drywall defects using natural-language prompts.  
Model: **CLIPSeg** (CIDAS/clipseg-rd64-refined) fine-tuned on two Roboflow datasets.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess datasets (one-time)
python src/download_data.py --api_key YOUR_ROBOFLOW_API_KEY

# 3. Train
python src/train.py \
    --train_manifest data/combined_train.jsonl \
    --val_manifest   data/combined_valid.jsonl \
    --epochs 20 --batch_size 8 --lr 5e-5 --seed 42

# 4. Infer on test set
python src/infer.py \
    --manifest  data/combined_test.jsonl \
    --model_dir checkpoints/best_model \
    --output_dir outputs/

# 5. Evaluate
python src/evaluate.py \
    --manifest data/combined_test.jsonl \
    --pred_dir outputs/ \
    --report   reports/eval_report.json

# 6. Generate training curves
python src/generate_report_figures.py \
    --log checkpoints/train_log.jsonl \
    --out reports/
```

## Repository Structure

```
drywall-qa/
├── src/
│   ├── download_data.py          # Download + COCO→binary-mask conversion
│   ├── dataset.py                # PyTorch Dataset with augmentations
│   ├── train.py                  # Fine-tuning loop (BCE+Dice loss)
│   ├── infer.py                  # Inference → {0,255} PNG masks
│   ├── evaluate.py               # mIoU / Dice + visual side-by-sides
│   └── generate_report_figures.py # Training curve plots
├── data/
│   ├── dataset1/                 # Taping area dataset (Roboflow)
│   │   ├── raw/                  # Raw Roboflow download
│   │   ├── masks/{train,valid,test}/
│   │   └── manifest_{train,valid,test}.jsonl
│   ├── dataset2/                 # Cracks dataset (Roboflow)
│   │   └── ...
│   ├── combined_train.jsonl
│   ├── combined_valid.jsonl
│   └── combined_test.jsonl
├── checkpoints/
│   ├── best_model/               # Best val mIoU checkpoint
│   ├── last_model/               # Final epoch checkpoint
│   └── train_log.jsonl           # Per-epoch loss + metrics
├── outputs/                      # Predicted masks (PNG, {0,255})
├── reports/
│   ├── visuals/                  # Side-by-side orig|GT|pred images
│   ├── training_curves.png
│   └── eval_report.json
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Roboflow link | Prompt | Task |
|---------|---------------|--------|------|
| Drywall-Join-Detect | [link](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | `"segment taping area"` | Tape/joint detection |
| Cracks-3ii36 | [link](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) | `"segment crack"` | Crack detection |

Downloaded in **COCO Segmentation** format.  
Polygon annotations are rasterised to single-channel {0,255} PNG masks.

Typical split counts (approximate, depends on dataset version):

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| Taping area | ~200 | ~60 | ~30 |
| Cracks | ~600 | ~80 | ~40 |
| **Combined** | **~800** | **~140** | **~70** |

## Model

**CLIPSeg** (`CIDAS/clipseg-rd64-refined`)

- CLIP ViT-B/16 backbone (frozen text encoder, fine-tuned vision encoder)  
- Lightweight transformer decoder conditioned on CLIP text embeddings  
- Native resolution: 352 × 352  
- Output: logit map at 64 × 64, bilinearly upsampled to original size  
- Parameters: ~68M  
- Model size on disk: ~280 MB

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Epochs | 20 |
| Loss | 0.5 × BCE + 0.5 × Dice |
| LR schedule | Linear warmup (50 steps) + cosine decay |
| Grad clip | 1.0 |
| Seed | **42** |
| Image size | 352 × 352 (CLIPSeg native) |

**Augmentations** (training only):
- Random horizontal / vertical flip  
- Random 90° rotation  
- Colour jitter (brightness ±20, contrast 0.7–1.3)  
- Prompt variant sampling (3–5 prompt phrasings per class)

## Inference

Output masks:
- Format: PNG, single-channel, same spatial size as source image  
- Values: `{0, 255}` (0 = background, 255 = defect)  
- Filename: `<image_id>__<prompt_tag>.png`  
  e.g. `img_0042__segment_crack.png`

Average inference time: ~35 ms/image on GPU (RTX 3090), ~200 ms on CPU.

## Results

*(Fill in your own after training, these are my samples for reference)*

| Prompt | mIoU | Dice | n (test) |
|--------|------|------|----------|
| segment taping area | 0.5225 | 0.6671 | 500 |
| segment crack | 0.4602 | 0.6114 | 112 |
| **Overall** | 0.5111 | 0.6569 | 612 |

## Failure Notes

- **Thin cracks** (< 3 px at original resolution): model tends to under-segment;  
  the 64×64 logit map cannot resolve very narrow structures.  
- **Low contrast taping** (painted-over joints): false negatives when tape blends  
  with surrounding wall colour.  
- **Overlapping defects**: both classes present in one image may confuse the  
  text-conditioned decoder; rare in the datasets but worth noting.

## Reproducibility

All random seeds are fixed to **42** via:
- `random.seed(42)`  
- `numpy.random.seed(42)`  
- `torch.manual_seed(42)`  
- `torch.cuda.manual_seed_all(42)`  
- `torch.backends.cudnn.deterministic = True`

Roboflow dataset versions are pinned to version **1** in `download_data.py`.
