# DGA-Botnet Hybrid Fusion (DGA-HLF)

Code and reproducibility material for the paper *“DGA Botnet Detection using Hybrid Late Fusion: An Evaluation of BiLSTM, DistilBERT and XGBoost Branches Combined through Soft Voting”* (under review).

The task is the **classification of individual domain strings** as DGA-generated or benign. It is not the detection of compromised hosts, active botnets or DNS traffic patterns.

M6 has three candidate branches:

- a character-level **BiLSTM** with an MLP head (string level);
- a fine-tuned **DistilBERT** classifier (semantic level);
- **XGBoost** over 20 handcrafted features.

Their probabilities are combined by **soft voting**. The weights and the threshold are selected on the validation partition. In all five seeds the search assigned **zero weight to the XGBoost branch**, so the evaluated model is a **two-branch BiLSTM + DistilBERT ensemble**.

## Main results (revised manuscript)

All numbers are means ± standard deviations over five independently seeded training runs (seeds 42–46). Every model uses the same validation-based threshold selection.

| Model | F1 (%) | AUC-ROC | DGA-W F1 (%) |
|---|---|---|---|
| M3 RCNN-SPP (best baseline) | 96.47 ± 0.10 | 0.9937 | 89.21 ± 0.24 |
| BiLSTM | 96.39 ± 0.05 | 0.9935 | 88.48 ± 0.44 |
| DistilBERT | 97.40 ± 0.03 | 0.9953 | 91.91 ± 0.16 |
| BERT | 97.43 ± 0.05 | 0.9950 | 92.30 ± 0.54 |
| **M6 late fusion (BiLSTM + DistilBERT)** | **97.71 ± 0.02** | **0.9965** | **93.14 ± 0.17** |

- **M6 vs BERT:** +0.28 F1 points. Paired t-test p = 3.8×10⁻⁴; family-cluster bootstrap 95% CI [+0.11, +0.46]; McNemar p = 1.4×10⁻⁵.
- **Early fusion with the same inputs:** 97.39–97.43% F1, i.e. late fusion is better by 0.28–0.32 points.
- **Three-branch and minimum-weight variants (w_min = 0.05–0.20):** 97.54–97.68% F1. Adding the handcrafted branch never helps.
- **Pretraining ablation:** a DistilBERT trained from random initialisation, with the same architecture and tokenizer, reaches 97.34 ± 0.09% F1 (p = 0.21 vs pretrained).
- **External DGA collections** (UMUDGA, Netlab360, DGArchive; 268,751 domains after removing exact overlaps with the training corpus): detection rate 94.39%. It is 96.43% for families seen in training and 86.82% for unseen families.
- **Independent benign sets** (Tranco, Cisco Umbrella): the false-positive rate is 4.6–6.3%, compared with 1.95% on the in-distribution test set.
- **Measured cost (RTX 5070 Ti, end-to-end from raw strings):** M6 has a 3.06 ms median latency for a single domain and processes 3,201 domains/s at batch size 256, using about 95 mJ per domain.

The original submission reported a single run (F1 97.73%). It also reported an inference time of 1.001 ms, which was not a measurement of M6. See `revision/README.md` for all corrections.

## Repository structure

```
├── M1.py … M5.py               # Baselines: RF + n-grams/LSA, TF-IDF + MLP, RCNN-SPP, ATT-CNN-BiLSTM, Dual-Embedding Transformer
├── selectAlgorithm.py          # Candidate backbones (char CNN, char BiLSTM, DistilBERT, BERT, SecureBERT);
│                               #   env DGA_SEED, DGA_CANDIDATES, DGA_SELECT_ROOT, DGA_BERT_RANDOM_INIT=1 (pretraining ablation)
├── M6_v3_Markov.py             # M6 hybrid late fusion (main model); env DGA_SEED, DGA_OUTPUT_DIR, DGA_SELECT_ROOT
├── M6_v3_Markov_w2min.py       # M6 with a minimum-weight constraint
├── M6_v3.py                    # Earlier M6 version without Markov features
├── Test_M6_v3_Markov.py        # External evaluation (UMUDGA / Netlab360 / DGArchive)
├── Test_M6_v3_Jaccard.py       # Ablation: Jaccard similarity instead of Markov perplexity
├── splitDataUTL.py             # Builds DataNew (70:15:15 split per family, after deduplication)
├── systemInfo.py               # Hardware information
├── run_multiseed_v3.ps1        # Five-seed runs of M1–M6 (fixed backbones)
├── run_fullseed.ps1            # Five-seed runs with backbones retrained per seed
├── MULTISEED.md                # How to run the multi-seed experiments
├── DATASET.md, linkDataset.txt # Dataset download instructions
├── REPRODUCIBILITY.md          # Step-by-step reproduction guide
├── results/                    # Reference outputs of the original single-seed runs
└── revision/                   # Material for the revised manuscript (code, manifests, labels, results, predictions)
```

Datasets, model checkpoints and experiment outputs (`DataNew/`, `test_external/`, `out_*`, `rev_out/`) are not committed.

## Installation

```bash
git clone <repository-url>
cd DGA-Botnet_Hybrid-Fusion
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128   # GPU users
pip install -r requirements.txt
python -c "import nltk; nltk.download('words')"
```

Tested with Python 3.10.9, PyTorch 2.9.0+cu128, transformers 4.57.1, XGBoost 3.2.0 and NLTK 3.9.4 on Windows 10, with an Intel Core 24-thread CPU, 64 GB RAM and an NVIDIA RTX 5070 Ti (16 GB).

## Datasets

| Dataset | Role | Families | Domains |
|---|---|---|---|
| UTL_DGA22 | DGA training/validation/test | 76 | 152,000 |
| Tranco Top 1M | Benign training/validation/test | — | 152,000 |
| UMUDGA | External DGA evaluation | 50 files | 100,000 sampled |
| Netlab360 | External DGA evaluation | 43 files | 50,013 sampled |
| DGArchive | External DGA evaluation | 79 files | 122,592 sampled |
| Cisco Umbrella Top 1M | Independent benign evaluation | — | 50,000 sampled |

See [DATASET.md](DATASET.md) for download instructions. Run `python splitDataUTL.py` to build `DataNew/train.csv` (212,800), `val.csv` (45,600) and `test.csv` (45,600). The exact split manifests used in the paper are provided in `revision/manifests/`.

All learned preprocessing is fitted on the training partition only. This covers TF-IDF vocabularies, SVD, scalers, the Markov models (106,400 benign training domains) and fine-tuning. The validation partition is used for early stopping and for selecting fusion weights and thresholds. The test partition is used only for reporting.

## Running the experiments

```bash
python splitDataUTL.py                       # 1. DataNew
python M1.py; python M2.py; python M3.py; python M4.py; python M5.py   # 2. baselines
python selectAlgorithm.py                    # 3. candidate backbones
python M6_v3_Markov.py                       # 4. M6
python Test_M6_v3_Markov.py                  # 5. external evaluation
```

- **Five seeds:** see [MULTISEED.md](MULTISEED.md).
- **Additional analyses of the revision:** see [revision/README.md](revision/README.md). These cover early fusion, minimum-weight variants, the pretraining ablation, calibration, independent benign sets, suffix perturbations, latency and the corrected handcrafted features.

## DGA subtypes

DGA-C, DGA-P and DGA-W are heuristic, per-domain labels used only to stratify the evaluation. They are computed on the main part *m* of the domain (leading `www.` and suffix removed, inner dots dropped):

1. **DGA-W** if |m| ≥ 12, vowel ratio ≥ 0.30 and entropy ≤ 3.6.
2. Otherwise **DGA-C** if entropy ≥ 3.8 or digit ratio ≥ 0.15.
3. Otherwise **DGA-P** if 0.20 ≤ vowel ratio ≤ 0.45.
4. Otherwise **DGA-C**.

Subtype F1 is computed on the domains of that subtype plus all benign test domains. (`DGA-R` in older output files is the same category as DGA-C.)

## License

The source code is released under the MIT License (see `LICENSE`). Datasets remain under the licenses of their providers: UTL_DGA22 (Tuan et al., 2023), Tranco, UMUDGA (CC-BY 4.0), 360 Netlab, DGArchive (Fraunhofer FKIE, academic access) and Cisco Umbrella.

## Contact

Author information is withheld during double-blind review. Please use the issue tracker of the repository.
