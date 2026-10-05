# DGA-Botnet Hybrid Fusion (DGA-HLF)

Detecting Domain Generation Algorithm (DGA) botnet domains using a **Hybrid Late Fusion** architecture that combines **BiLSTM** (String-Level), **DistilBERT** (Semantic-Level), and **XGBoost** over 20 handcrafted features (Statistical-Level) through **soft voting** with internally normalized weights.

The proposed **M6** model achieves F1 = **97.73%**, AUC = **0.9967**, and DGA-W F1 = **93.02%** on the in-distribution test set (UTL_DGA22 + Tranco Top 1M, 45,600 samples), surpassing standalone BERT (97.45%, 109M parameters). External evaluation on **124 DGA families** from UMUDGA, Netlab360, and DGArchive (272,605 samples) yields DR = **94.62%**, with **90.40% on 63 out-of-distribution families**.

---

## Table of Contents

- [Contributions](#contributions)
- [M6 Architecture](#m6-architecture)
- [Repository Structure](#repository-structure)
- [Environment Requirements](#environment-requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Running the Experiments](#running-the-experiments)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Contact](#contact)

---

## Contributions

1. **Information Representation Framework** — a four-category taxonomy (String / Semantic / Context / Hybrid) that classifies DGA detection methods by the source of information they exploit, rather than by the underlying architecture (ML / DL / Transformer / GNN).
2. **M6 Hybrid Late Fusion** — three independent classification branches (BiLSTM, DistilBERT, XGBoost) integrated through soft voting with optimized weights; surpasses every single-branch model including standalone BERT.
3. **Controlled Experimental Evaluation** — DataNew with 304,000 samples pre-split 70:15:15 + three external datasets covering 124 DGA families; within/out-of-distribution analysis.

## M6 Architecture

```
Domain input
     │
     ├─────────────┬──────────────────┬───────────────────┐
     ▼             ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────┐
│ Branch 1 │  │   Branch 2   │  │     Branch 3     │
│  String  │  │   Semantic   │  │    Statistical   │
├──────────┤  ├──────────────┤  ├──────────────────┤
│ Char emb │  │ WordPiece    │  │ 20 features      │
│ BiLSTM   │  │ DistilBERT   │  │ (17 + 1 + 2)     │
│ MLP head │  │ Classifier   │  │ XGBoost          │
└────┬─────┘  └──────┬───────┘  └─────────┬────────┘
     │               │                    │
     ▼               ▼                    ▼
  P_string       P_semantic           P_handcraft
     │               │                    │
     └───────────────┼────────────────────┘
                     ▼
            Soft voting (w1, w2, w3)
         Internally normalized before fusion
                     ▼
              Final: DGA / Benign
```

See Section 4 of the paper for the fusion equation and weight optimization details.

## Repository Structure

```
DGA-HLF/
├── M1.py                       # Baseline M1 — Random Forest + n-gram + TF-IDF/LSA
├── M2.py                       # Baseline M2 — TF-IDF + Deep MLP
├── M3.py                       # Baseline M3 — RCNN-SPP (BiLSTM + CNN + SPP)
├── M4.py                       # Baseline M4 — ATT-CNN-BiLSTM
├── M5.py                       # Baseline M5 — Dual-Embedding Transformer
│
├── selectAlgorithm.py          # Embedding selection (CNN / BiLSTM / BERT / DistilBERT / SecureBERT)
│
├── M6_v3.py                    # M6 Hybrid Late Fusion
├── M6_v3_Markov.py             # M6 with Markov chain perplexity features
├── M6_v3_Markov_w2min.py       # M6 with minimum-weight constraint
│
├── Test_M6_v3_Markov.py        # Ablation: Markov perplexity features
├── Test_M6_v3_Jaccard.py       # Ablation: Jaccard similarity (compared with Markov)
│
├── splitDataUTL.py             # Pre-split DataNew 70:15:15 after deduplication
├── systemInfo.py               # Collect hardware information
│
├── linkDataset.txt             # Quick dataset links (see DATASET.md for details)
├── DATASET.md                  # Detailed dataset download instructions
├── REPRODUCIBILITY.md          # Step-by-step reproduction guide
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # This file
│
└── results/                    # Reference outputs for reviewer comparison
    ├── README.md
    ├── baselines/              # M1-M5 outputs
    ├── select_algorithm/       # Embedding comparison results
    ├── m6_markov/              # M6 Hybrid Late Fusion outputs
    └── external_eval/          # UMUDGA + Netlab360 + DGArchive detection rates
```

The following directories are NOT committed (via `.gitignore`):

```
DataNew/                        # train.csv / val.csv / test.csv after pre-split
ngrams/                         # Markov bigram/trigram counts from benign training set
out_M1/ ... out_M5/             # Baseline outputs (full version with model weights)
out_select/                     # selectAlgorithm outputs (candidate comparison)
out_M6_v3/                      # M6 Early Fusion output (baseline)
out_M6_v3_Markov/               # Main M6 output (Markov perplexity)
out_M6_v3_Markov_w2min/         # M6 output with w_min constraint
test_external/                  # Evaluation results on UMUDGA / Netlab360 / DGArchive
```

## Environment Requirements

### Hardware (recommended)

- CPU: Intel Core, 24 threads
- RAM: 64 GB
- GPU: NVIDIA RTX 5070 Ti 16 GB VRAM (or equivalent, minimum 12 GB VRAM for DistilBERT/BERT)
- Storage: ~30 GB for datasets and checkpoints

### Software

- OS: Windows 10 / Ubuntu 22.04+
- Python 3.10.9
- CUDA 12.8
- PyTorch 2.9.0+cu128

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/hanhvxhou/DGA-Botnet_Hybrid-Fusion.git
cd DGA-Botnet_Hybrid-Fusion

# 2. Create a virtual environment
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Install PyTorch with CUDA 12.8 first (for GPU users)
pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 4. Install the remaining dependencies
pip install -r requirements.txt

# 5. Download the NLTK corpus (for the meaning_ratio feature)
python -c "import nltk; nltk.download('words')"
```

CPU-only users can skip step 3; step 4 will install the CPU wheel automatically.

## Datasets

Datasets are NOT committed to this repository (large size, third-party licenses). See [DATASET.md](DATASET.md) for detailed download instructions, directory structure, and troubleshooting.

| Dataset | Role | Families | Samples |
|---------|------|----------|---------|
| UTL_DGA22 | DGA training | 76 | 152,000 |
| Tranco Top 1M | Benign training | — | 152,000 |
| UMUDGA | External evaluation | 50 | 100,000 |
| Netlab360 | External evaluation | 43 | 50,013 |
| DGArchive | External evaluation | 79 | 122,592 |

### Preparing DataNew

After downloading UTL_DGA22 and Tranco Top 1M, run:

```bash
python splitDataUTL.py
```

This produces three files: `DataNew/train.csv` (212,800 samples), `DataNew/val.csv` (45,600), and `DataNew/test.csv` (45,600) — a 70:15:15 ratio after deduplication.

## Running the Experiments

### Collect system information

```bash
python systemInfo.py
```

### Baselines M1–M5

```bash
python M1.py       # Random Forest + n-gram + LSA (~20 s)
python M2.py       # TF-IDF + Deep MLP (~2 min)
python M3.py       # RCNN-SPP (~2 min)
python M4.py       # ATT-CNN-BiLSTM (~2 min)
python M5.py       # Dual-Embedding Transformer (~30 min)
```

Outputs are written to `out_M1/`, `out_M2/`, ..., `out_M5/`.

### Embedding selection (selectAlgorithm)

```bash
python selectAlgorithm.py
```

Compares five candidates: character CNN, character BiLSTM, DistilBERT, BERT, SecureBERT. Output in `out_select/`.

### M6 Hybrid Late Fusion

```bash
# Main M6 with Markov perplexity (default configuration)
python M6_v3_Markov.py

# Variant with minimum-weight constraint (w_min >= 0.05)
python M6_v3_Markov_w2min.py

# Original version (without Markov)
python M6_v3.py
```

### Ablation: Markov vs. Jaccard

```bash
python Test_M6_v3_Markov.py       # Standalone handcraft branch with Markov features
python Test_M6_v3_Jaccard.py      # Standalone handcraft branch with Jaccard similarity
```

### Full pipeline (from scratch)

```bash
python splitDataUTL.py              # Step 1: prepare DataNew
python M1.py && python M2.py && python M3.py && python M4.py && python M5.py   # Step 2: baselines
python selectAlgorithm.py           # Step 3: embedding selection
python M6_v3_Markov.py              # Step 4: train and evaluate M6
```

Total runtime: ~5-6 hours on an RTX 5070 Ti.

## Results

### Baselines M1-M5 (in-distribution test set, 45,600 samples)

| Model | F1 (%) | DGA-C | DGA-P | DGA-W | Infer (ms) | Size (MB) |
|-------|--------|-------|-------|-------|------------|-----------|
| M1 — RF + n-gram | 93.24 | 95.01 | 86.34 | 81.14 | 0.002 | 67.56 |
| M2 — TF-IDF + MLP | 95.99 | 95.20 | 90.65 | 87.61 | 0.007 | 20.25 |
| **M3 — RCNN-SPP** | **96.49** | 95.94 | 91.87 | 89.19 | 0.011 | 5.54 |
| M4 — ATT-CNN-BiLSTM | 96.14 | 95.86 | 91.35 | 87.85 | 0.007 | 1.60 |
| M5 — Dual Transformer | 95.65 | 95.08 | 90.12 | 86.11 | 0.092 | 38.28 |

### Embedding selection

| Candidate | F1 (%) | DGA-W | Train (s) | Infer (ms) | Size (MB) |
|-----------|--------|-------|-----------|------------|-----------|
| CNN (char) | 95.31 | 85.09 | 76.2 | 0.003 | 1.40 |
| BiLSTM (char) | 96.45 | 88.84 | 151.6 | 0.010 | 2.79 |
| DistilBERT | 97.33 | 91.45 | 2,183.7 | 0.349 | 253.19 |
| BERT | **97.45** | 92.31 | 4,154.3 | 0.642 | 417.71 |
| SecureBERT | 97.28 | 92.07 | 4,167.3 | 0.628 | 475.56 |

**DistilBERT** is selected for M6 (Green AI tradeoff).

### M6 Hybrid Late Fusion

| Model | F1 (%) | DGA-C | DGA-P | DGA-W | Infer (ms) | Size (MB) |
|-------|--------|-------|-------|-------|------------|-----------|
| M3-RCNN-SPP | 96.49 | 95.94 | 91.87 | 89.19 | 0.011 | 5.54 |
| DistilBERT | 97.33 | 96.74 | 93.69 | 91.45 | 0.349 | 253.19 |
| BERT | 97.45 | 97.03 | 94.22 | 92.31 | 0.642 | 417.71 |
| **M6 Hybrid** | **97.73** | **97.47** | **94.99** | **93.02** | ~1.001 | 256.32 |

### External evaluation (3 datasets, 124 families, 272,605 samples)

| Dataset | Families | Samples | DR (%) | Families in UTL_DGA22 |
|---------|----------|---------|--------|------------------------|
| UMUDGA | 50 | 100,000 | 97.26 | 46/50 |
| Netlab360 | 43 | 50,013 | 93.98 | 24/43 |
| DGArchive | 79 | 122,592 | 92.72 | 31/79 |
| **Combined** | **124** | **272,605** | **94.62** | **61/124** |
| Within-distribution | 61 | 183,746 | 96.65 | — |
| Out-of-distribution | 63 | 88,859 | 90.40 | — |

The gap of only **6.25 pp** between within- and out-of-distribution families demonstrates the generalization capability of Hybrid Late Fusion.

## Reproducibility

Reference outputs for reviewers are provided in the `results/` folder. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for a complete step-by-step guide including:

- Environment setup and verification
- Dataset preparation
- Full pipeline execution
- Expected metrics with tolerance ranges
- Troubleshooting common issues
- Hardware variation notes

Reviewers can compare their reproduction against the files under `results/`:
- `results/baselines/M1/` through `results/baselines/M5/` — baseline outputs
- `results/select_algorithm/` — embedding comparison
- `results/m6_markov/` — M6 Hybrid Late Fusion
- `results/external_eval/` — external evaluation

Minor differences within +/- 0.3 percentage points on F1 are expected due to non-deterministic CUDA operations and hardware variations.

## License

Source code is released under the **MIT License** — see the `LICENSE` file.

Datasets are used under their respective original licenses:
- UTL_DGA22 — Tuan et al. 2023
- Tranco Top 1M — tranco-list.eu (research use)
- UMUDGA — University of Murcia (CC-BY 4.0)
- Netlab360 — Qihoo 360 (open)
- DGArchive — Fraunhofer FKIE (academic access)

## Contact

**Dr. Vu Xuan Hanh**
Faculty of Information Technology, Hanoi Open University
Email: hanhvx@hou.edu.vn
GitHub: [@hanhvxhou](https://github.com/hanhvxhou)

For code-related issues, please open an [issue](https://github.com/hanhvxhou/DGA-Botnet_Hybrid-Fusion/issues) on this repository.
