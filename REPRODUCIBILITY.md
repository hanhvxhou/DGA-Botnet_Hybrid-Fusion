# Reproducibility Guide

Step-by-step instructions to reproduce all results reported in the paper and `results/`.

## 1. Environment Setup

### Prerequisites

| Component | Required Version |
|-----------|------------------|
| OS | Windows 10 / Ubuntu 22.04+ |
| Python | 3.10.9 |
| CUDA | 12.8 |
| GPU | NVIDIA with >= 12 GB VRAM (RTX 3080 / 4070 Ti / 5070 Ti / A5000+) |
| RAM | >= 32 GB (64 GB recommended for M5 Dual Transformer) |
| Disk space | ~30 GB (datasets + checkpoints) |

### Install

```bash
# Clone
git clone https://github.com/hanhvxhou/DGA-Botnet_Hybrid-Fusion.git
cd DGA-Botnet_Hybrid-Fusion

# Virtual env
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# PyTorch with CUDA 12.8
pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Rest
pip install -r requirements.txt

# NLTK corpus
python -c "import nltk; nltk.download('words')"
```

### Verify environment

```bash
python systemInfo.py
```

Expected output (should match hardware section of `results/README.md`).

## 2. Dataset Preparation

### Download sources (see `linkDataset.txt`)

| Dataset | URL | Expected location |
|---------|-----|-------------------|
| UTL_DGA22 | https://data.mendeley.com/datasets/y8ph45msv8 (and related) | `data/raw/UTL_DGA22/` |
| Alexa Top 1M | https://www.kaggle.com/datasets/cheedcheed/top1m | `data/raw/alexa_top_1m.csv` |
| UMUDGA | https://data.mendeley.com/datasets/y8ph45msv8 | `data/raw/UMUDGA/` |
| Netlab360 | https://github.com/360netlab/DGA | `data/raw/Netlab360/` |
| DGArchive | https://dgarchive.caad.fkie.fraunhofer.de/ (academic access required) | `data/raw/DGArchive/` |

### Pre-split DataNew

```bash
python splitDataUTL.py
```

This creates:
- `DataNew/train.csv` — 212,800 samples (70%)
- `DataNew/val.csv` — 45,600 samples (15%)
- `DataNew/test.csv` — 45,600 samples (15%)

**Critical:** random seed is fixed at 42 inside `splitDataUTL.py`. Do not re-split per experiment.

### Verify split

```bash
python -c "import pandas as pd; print(pd.read_csv('DataNew/train.csv').shape, pd.read_csv('DataNew/val.csv').shape, pd.read_csv('DataNew/test.csv').shape)"
```

Expected: `(212800, N) (45600, N) (45600, N)` where N is the number of columns.

## 3. Run Experiments

### Step 1 — Baselines M1-M5

Run each baseline (can be parallel on different GPU). Order within M1-M5 does not matter.

```bash
python M1.py       # ~20 sec    - Random Forest + n-gram + LSA
python M2.py       # ~2 min     - TF-IDF + Deep MLP
python M3.py       # ~2 min     - RCNN-SPP
python M4.py       # ~2 min     - ATT-CNN-BiLSTM
python M5.py       # ~30 min    - Dual-Embedding Transformer
```

Each writes to `out_M1/`, `out_M2/`, etc. Compare `summary.txt` against `results/baselines/*/summary.txt`.

**Expected F1 range (within +/- 0.3 pp of):**
- M1: 93.24 %
- M2: 95.99 %
- M3: 96.49 %
- M4: 96.14 %
- M5: 95.65 %

### Step 2 — Embedding Selection (selectAlgorithm)

```bash
python selectAlgorithm.py       # ~3-4 hours (trains CNN + BiLSTM + 3 PLMs)
```

Output in `out_select/`. Winner should be **BERT** (F1 ~97.45%).

### Step 3 — M6 Hybrid Late Fusion

```bash
# Main configuration (Markov features, no w_min constraint)
python M6_v3_Markov.py          # ~15-20 min (grid search + evaluation)

# Variant with w_min = 0.05
python M6_v3_Markov_w2min.py    # ~15-20 min

# Baseline M6 (without Markov)
python M6_v3.py                 # ~10 min
```

**Expected M6 (default, Markov) results:**
- F1 = 97.73 %
- AUC = 0.9967
- DGA-W F1 = 93.02 %
- Raw weights (w1, w2, w3) = (0.100, 0.100, 0.000)

### Step 4 — Ablation (Markov vs. Jaccard)

```bash
python Test_M6_v3_Markov.py     # Standalone handcraft branch with Markov
python Test_M6_v3_Jaccard.py    # Standalone handcraft branch with Jaccard
```

**Expected:** Markov F1 = 91.36 %, Jaccard F1 = 88.10 %.

### Step 5 — External Evaluation

External evaluation is run as part of `M6_v3_Markov.py` and writes to `test_external/`.

**Expected aggregate DR (over 272,605 samples):** 94.62 %.

## 4. Full Pipeline (One Command)

```bash
python splitDataUTL.py && \
python M1.py && python M2.py && python M3.py && python M4.py && python M5.py && \
python selectAlgorithm.py && \
python M6_v3_Markov.py && \
python Test_M6_v3_Markov.py && \
python Test_M6_v3_Jaccard.py
```

Total runtime: **~5-6 hours** on RTX 5070 Ti (dominated by M5 and selectAlgorithm PLM fine-tuning).

## 5. Comparing Your Results

For each experiment, open the corresponding file under `results/` and compare key metrics:

| Your output | Reference |
|-------------|-----------|
| `out_M1/summary.txt` | `results/baselines/M1/summary.txt` |
| `out_M2/summary.txt` | `results/baselines/M2/summary.txt` |
| ... | ... |
| `out_M6_v3_Markov/summary.txt` | `results/m6_markov/summary.txt` |
| `test_external/summary_markov.txt` | `results/external_eval/summary_markov.txt` |

### Tolerance

| Metric | Acceptable deviation |
|--------|----------------------|
| Binary F1 | +/- 0.3 percentage points |
| Per-family F1 (DGA-C/P/W) | +/- 0.5 percentage points |
| AUC | +/- 0.002 |
| External DR | +/- 0.5 percentage points |
| Soft voting weights | May differ (grid search picks any (w, w, 0) raw tuple due to internal normalization — effective weights still equal to (0.5, 0.5, 0.0)) |

Larger deviations suggest a reproducibility issue — see Section 6.

## 6. Troubleshooting

### My F1 is much lower than reported

- Verify random seed = 42 (search `seed` in `splitDataUTL.py` and model files)
- Ensure DataNew is **not** being re-split by each model
- Check CUDA version matches 12.8
- Check `transformers`, `torch`, `xgboost` versions match `requirements.txt`

### OOM (Out of Memory) on GPU

- M5 Dual Transformer is the heaviest. Reduce `batch_size` from 256 to 128 (expect slower training, same final metric within tolerance)
- BERT fine-tuning in `selectAlgorithm.py` needs ~12 GB VRAM. Use `distilbert-only` mode if VRAM < 12 GB (manual edit)

### `nltk.corpus.words` not found

```bash
python -c "import nltk; nltk.download('words')"
```

### `test.csv not found` when running M1-M5

Run `python splitDataUTL.py` first to generate `DataNew/`.

### External datasets not evaluated

Confirm `data/raw/UMUDGA/`, `data/raw/Netlab360/`, `data/raw/DGArchive/` exist with the expected file structure described in `linkDataset.txt`.

## 7. Hardware Variations

Results reported in the paper use RTX 5070 Ti. On other hardware:

| GPU | Expected training time M5 | Expected training time selectAlgorithm |
|-----|---------------------------|----------------------------------------|
| RTX 5070 Ti | ~30 min | ~3-4 hours |
| RTX 4090 | ~25 min | ~3 hours |
| RTX 3080 10 GB | ~45 min | ~5 hours (reduce batch size) |
| A100 40 GB | ~15 min | ~1.5 hours |
| CPU only | Not recommended (>12 hours for M5) | Not recommended (>48 hours) |

Final F1 metrics should be reproducible across GPUs within the tolerance above.

## 8. Contact

For reproducibility issues not covered here, open an issue:
https://github.com/hanhvxhou/DGA-Botnet_Hybrid-Fusion/issues
