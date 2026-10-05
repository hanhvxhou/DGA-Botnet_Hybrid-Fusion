# DGA Botnet Detection — Hybrid Late Fusion

Phát hiện tên miền do thuật toán sinh (Domain Generation Algorithm, DGA) bằng kiến trúc **Hybrid Late Fusion** kết hợp BiLSTM (String-Level), DistilBERT (Semantic-Level) và XGBoost trên 20 đặc trưng thủ công (Statistical-Level) qua **soft voting**.

Mô hình đề xuất M6 đạt **F1 = 97.73%**, **AUC = 0.9967**, **DGA-W = 93.02%** trên tập kiểm thử nội bộ (UTL_DGA22 + Alexa Top 1M, 45.600 mẫu), vượt BERT đơn lẻ (97.45%, 109 triệu tham số). Đánh giá ngoại vi trên **124 họ DGA** từ UMUDGA, Netlab360, DGArchive (272.605 mẫu) đạt **DR = 94.62%**, trong đó **90.40% trên 63 họ out-of-distribution**.

---

## Mục lục

- [Đóng góp khoa học](#đóng-góp-khoa-học)
- [Kiến trúc M6](#kiến-trúc-m6)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Dữ liệu](#dữ-liệu)
- [Chạy thực nghiệm](#chạy-thực-nghiệm)
- [Kết quả chính](#kết-quả-chính)
- [Tái hiện bài báo](#tái-hiện-bài-báo)
- [Trích dẫn](#trích-dẫn)
- [Giấy phép](#giấy-phép)
- [Liên hệ](#liên-hệ)

---

## Đóng góp khoa học

1. **Framework biểu diễn thông tin** — phân loại các phương pháp DGA detection theo bốn nhóm: String-Level, Semantic-Level, Context-Level và Hybrid, thay cho phân loại truyền thống theo kiến trúc (ML/DL/Transformer/GNN).
2. **M6 Hybrid Late Fusion** — ba nhánh phân loại độc lập kết hợp qua soft voting, vượt mọi mô hình đơn lẻ (bao gồm BERT standalone 109M params).
3. **Đánh giá có kiểm soát** — DataNew (304.000 mẫu, chia trước 70:15:15) + ba tập ngoại vi 124 họ DGA, cho phân tích within/out-of-distribution.

## Kiến trúc M6

```
Domain input
     │
     ├─────────────┬─────────────────┬──────────────────┐
     ▼             ▼                 ▼                  │
┌──────────┐  ┌──────────────┐  ┌────────────────┐
│ Branch 1 │  │  Branch 2    │  │   Branch 3     │
│ String   │  │  Semantic    │  │  Statistical   │
├──────────┤  ├──────────────┤  ├────────────────┤
│ Char emb │  │ WordPiece    │  │ 20 features    │
│ BiLSTM   │  │ DistilBERT   │  │ (17+1+2)       │
│ MLP head │  │ Classifier   │  │ XGBoost        │
└────┬─────┘  └──────┬───────┘  └────────┬───────┘
     │               │                   │
     ▼               ▼                   ▼
  P_string       P_semantic         P_handcraft
     │               │                   │
     └───────────────┼───────────────────┘
                     ▼
              Soft voting
         P = Σ wᵢ · Pᵢ, τ = 0.5
                     ▼
              DGA / Benign
```

Chi tiết công thức fusion, chọn trọng số và từng nhánh — xem bài báo (`docs/paper/`) hoặc code (`src/m6/`).

## Cấu trúc thư mục

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                  # Dữ liệu gốc (không commit)
│   │   ├── UTL_DGA22/
│   │   ├── alexa_top_1m.csv
│   │   ├── UMUDGA/
│   │   ├── Netlab360/
│   │   └── DGArchive/
│   └── DataNew/              # Pre-split train/val/test (không commit)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── split_dataset.py            # Chia 70:15:15 + loại trùng lặp
│   │   ├── feature_extraction.py       # 20 đặc trưng thủ công
│   │   └── markov_perplexity.py        # Markov bigram/trigram
│   │
│   ├── baselines/
│   │   ├── m1_rf_ngram_pca.py          # Random Forest + n-gram + PCA
│   │   ├── m2_tfidf_mlp.py             # TF-IDF + Deep MLP
│   │   ├── m3_rcnn_spp.py              # RCNN với Spatial Pyramid Pooling
│   │   ├── m4_att_cnn_bilstm.py        # Attention + CNN + BiLSTM
│   │   └── m5_dual_transformer.py      # Dual-Embedding Transformer
│   │
│   ├── select_algorithm/
│   │   ├── cnn_char.py
│   │   ├── bilstm_char.py
│   │   ├── distilbert_finetune.py
│   │   ├── bert_finetune.py
│   │   └── securebert_finetune.py
│   │
│   ├── m6/
│   │   ├── train_branch_string.py      # BiLSTM + MLP head
│   │   ├── train_branch_semantic.py    # DistilBERT classifier head
│   │   ├── train_branch_statistical.py # XGBoost + 20 features
│   │   ├── grid_search_weights.py      # Tìm (w1, w2, w3) và τ
│   │   └── predict.py                  # Inference pipeline
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # Acc, Prec, Rec, F1, AUC
│   │   ├── per_family.py               # F1 theo DGA-R/P/W
│   │   ├── external_eval.py            # DR trên UMUDGA/Netlab360/DGArchive
│   │   └── cost_metrics.py             # Train time, inference, size, năng lượng
│   │
│   └── utils/
│       ├── config.py
│       ├── seeds.py
│       └── logging.py
│
├── experiments/
│   ├── configs/                        # YAML config cho từng mô hình
│   ├── logs/                           # Log huấn luyện (không commit)
│   └── results/                        # CSV kết quả (không commit)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_m1_m5_comparison.ipynb
│   ├── 04_select_algorithm.ipynb
│   ├── 05_m6_results.ipynb
│   └── 06_external_evaluation.ipynb
│
├── checkpoints/                        # Model weights (không commit)
│   ├── m1_rf.pkl
│   ├── m3_rcnn_spp.pt
│   ├── bilstm_backbone.pt
│   ├── distilbert_finetuned/
│   └── xgboost_handcraft.json
│
├── docs/
│   ├── paper/                          # Bản thảo bài báo
│   │   ├── ieee_access_submission.docx
│   │   └── figures/
│   ├── framework.md                    # Framework biểu diễn thông tin
│   └── reproducibility.md
│
└── scripts/
    ├── run_all_baselines.sh
    ├── run_m6.sh
    └── run_external_eval.sh
```

## Yêu cầu môi trường

### Phần cứng (khuyến nghị)

- CPU: Intel Core 24 luồng trở lên
- RAM: 64 GB
- GPU: NVIDIA RTX 5070 Ti 16 GB VRAM (hoặc tương đương, ≥ 12 GB VRAM cho DistilBERT/BERT)
- Dung lượng: ~50 GB cho dữ liệu + checkpoints

### Phần mềm

- OS: Windows 10 / Ubuntu 22.04+
- Python 3.10.9
- CUDA 12.8
- PyTorch 2.9.0+cu128

## Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/hanhvxhou/dga-botnet-hybrid-fusion.git
cd dga-botnet-hybrid-fusion

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Tải NLTK corpus (cho meaning_ratio feature)
python -c "import nltk; nltk.download('words')"
```

### requirements.txt

```
torch==2.9.0
transformers==4.40.0
xgboost==3.2.0
scikit-learn==1.4.0
numpy==1.26.4
pandas==2.2.0
nltk==3.8.1
tqdm==4.66.0
matplotlib==3.8.0
seaborn==0.13.0
codecarbon==2.3.4
pyyaml==6.0.1
jupyter==1.0.0
```

## Dữ liệu

### Nguồn

| Tập dữ liệu | Số họ | Số mẫu | Mục đích | Nguồn |
|-------------|-------|--------|----------|-------|
| UTL_DGA22 | 76 | 152.000 | DGA huấn luyện | [Kaggle](https://www.kaggle.com/datasets/utldga22) |
| Alexa Top 1M | — | 152.000 | Benign | [AWS](https://www.alexa.com/topsites) |
| UMUDGA | 50 | 100.000 | Đánh giá ngoại vi | Zago et al. 2020 |
| Netlab360 | 43 | 50.013 | Đánh giá ngoại vi | [Qihoo360](https://data.netlab.360.com/dga/) |
| DGArchive | 79 | 122.592 | Đánh giá ngoại vi | [DGArchive](https://dgarchive.caad.fkie.fraunhofer.de/) |

### Pre-split (DataNew)

Chia một lần duy nhất 70:15:15 sau khi loại trùng lặp:

```bash
python src/preprocessing/split_dataset.py \
  --input data/raw/combined.csv \
  --output data/DataNew/ \
  --ratio 0.7 0.15 0.15 \
  --seed 42
```

Kết quả: 212.800 train / 45.600 val / 45.600 test.

## Chạy thực nghiệm

### Baselines M1–M5

```bash
# Chạy từng mô hình
python src/baselines/m1_rf_ngram_pca.py --config experiments/configs/m1.yaml
python src/baselines/m2_tfidf_mlp.py    --config experiments/configs/m2.yaml
python src/baselines/m3_rcnn_spp.py     --config experiments/configs/m3.yaml
python src/baselines/m4_att_cnn_bilstm.py --config experiments/configs/m4.yaml
python src/baselines/m5_dual_transformer.py --config experiments/configs/m5.yaml

# Hoặc chạy tất cả
bash scripts/run_all_baselines.sh
```

### selectAlgorithm

```bash
python src/select_algorithm/cnn_char.py
python src/select_algorithm/bilstm_char.py
python src/select_algorithm/distilbert_finetune.py
python src/select_algorithm/bert_finetune.py
python src/select_algorithm/securebert_finetune.py
```

### M6 Hybrid Late Fusion

```bash
# Bước 1: Huấn luyện từng nhánh (sau selectAlgorithm)
python src/m6/train_branch_string.py       # ~5s (MLP head)
python src/m6/train_branch_semantic.py     # Dùng DistilBERT đã fine-tune
python src/m6/train_branch_statistical.py  # XGBoost, ~1.2s

# Bước 2: Grid-search trọng số và threshold trên VAL
python src/m6/grid_search_weights.py \
  --val-probs experiments/results/val_probs.npz \
  --output experiments/results/weight_search.csv

# Bước 3: Inference với cấu hình tối ưu
python src/m6/predict.py \
  --weights-file experiments/results/best_weights.json \
  --input data/DataNew/test.csv
```

### Đánh giá ngoại vi

```bash
python src/evaluation/external_eval.py \
  --checkpoints checkpoints/ \
  --external-datasets data/raw/UMUDGA data/raw/Netlab360 data/raw/DGArchive \
  --output experiments/results/external_eval.csv
```

## Kết quả chính

### Baselines M1–M5 (tập kiểm thử nội bộ, 45.600 mẫu)

| Mô hình | F1 (%) | DGA-R | DGA-P | DGA-W | Train (phút) | Infer (ms) | Size (MB) |
|---------|--------|-------|-------|-------|--------------|------------|-----------|
| M1 — RF + n-gram + PCA | 93.24 | 95.01 | 86.34 | 81.14 | 0.31 | 0.002 | 67.56 |
| M2 — TF-IDF + MLP | 95.99 | 95.20 | 90.65 | 87.61 | 1.78 | 0.007 | 20.25 |
| **M3 — RCNN-SPP** | **96.49** | 95.94 | 91.87 | 89.19 | 1.55 | 0.011 | 5.54 |
| M4 — ATT-CNN-BiLSTM | 96.14 | 95.86 | 91.35 | 87.85 | 1.66 | 0.007 | 1.60 |
| M5 — Dual Transformer | 95.65 | 95.08 | 90.12 | 86.11 | 32.10 | 0.092 | 38.28 |

### selectAlgorithm

| Candidate | F1 (%) | DGA-W | Train (s) | Infer (ms) | Size (MB) |
|-----------|--------|-------|-----------|------------|-----------|
| CNN (char) | 95.31 | 85.09 | 76.2 | 0.003 | 1.40 |
| BiLSTM (char) | 96.45 | 88.84 | 151.6 | 0.010 | 2.79 |
| DistilBERT | 97.33 | 91.45 | 2,183.7 | 0.349 | 253.19 |
| **BERT (winner)** | **97.45** | 92.31 | 4,154.3 | 0.642 | 417.71 |
| SecureBERT | 97.28 | 92.07 | 4,167.3 | 0.628 | 475.56 |

→ Chọn **DistilBERT** cho M6 (Green AI: nhỏ hơn 1.88×, nhanh hơn 1.76×, chỉ mất 0.05 pp F1).

### M6 Hybrid Late Fusion

| Mô hình | F1 (%) | DGA-R | DGA-P | DGA-W | Infer (ms) | Size (MB) |
|---------|--------|-------|-------|-------|------------|-----------|
| M3-RCNN-SPP | 96.49 | 95.94 | 91.87 | 89.19 | 0.011 | 5.54 |
| DistilBERT | 97.33 | 96.74 | 93.69 | 91.45 | 0.349 | 253.19 |
| BERT | 97.45 | 97.03 | 94.22 | 92.31 | 0.642 | 417.71 |
| **M6 Hybrid** | **97.73** | **97.47** | **94.99** | **93.02** | ~1.001 | 256.32 |

### Đánh giá ngoại vi (3 tập, 124 họ, 272.605 mẫu)

| Dataset | Số họ | Số mẫu | DR (%) | Họ ∈ UTL_DGA22 |
|---------|-------|--------|--------|-----------------|
| UMUDGA | 50 | 100.000 | 97.26 | 46/50 |
| Netlab360 | 43 | 50.013 | 93.98 | 24/43 |
| DGArchive | 79 | 122.592 | 92.72 | 31/79 |
| **Hợp** | **124** | **272.605** | **94.62** | **61/124** |
| Within-distribution | 61 | 183.746 | 96.65 | — |
| Out-of-distribution | 63 | 88.859 | 90.40 | — |

Chênh lệch within vs out-of-distribution chỉ **6.25 pp** — minh chứng khả năng tổng quát hóa của Hybrid Late Fusion.

## Tái hiện bài báo

Để tái hiện toàn bộ kết quả bài báo IEEE Access:

```bash
# 1. Chuẩn bị dữ liệu
bash scripts/prepare_data.sh

# 2. Chạy toàn bộ pipeline (ước tính ~24 giờ trên RTX 5070 Ti)
bash scripts/run_full_pipeline.sh

# 3. Sinh bảng và hình cho bài báo
python scripts/generate_tables.py
python scripts/generate_figures.py
```

Chi tiết: `docs/reproducibility.md`.

## Trích dẫn

Nếu sử dụng mã nguồn hoặc kết quả của repository này, vui lòng trích dẫn:

```bibtex
@article{vu2026dga,
  title   = {DGA Botnet Detection via Hybrid Late Fusion:
             Combining BiLSTM, DistilBERT, and 20-Dimensional
             Handcrafted Features through Soft Voting},
  author  = {Vu, Xuan Hanh and Tran, Tien Dung},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

## Giấy phép

Mã nguồn phát hành theo giấy phép **MIT License** — xem `LICENSE`.

Dữ liệu được sử dụng theo giấy phép của từng nguồn gốc:
- UTL_DGA22, Alexa Top 1M, UMUDGA, Netlab360, DGArchive — theo điều khoản của chủ sở hữu.

## Liên hệ

**TS. Vũ Xuân Hạnh**
Khoa Công nghệ Thông tin, Trường Đại học Mở Hà Nội
Email: hanhvx@hou.edu.vn
GitHub: [@hanhvxhou](https://github.com/hanhvxhou)

Mọi vấn đề về mã nguồn, vui lòng mở issue tại repository.
