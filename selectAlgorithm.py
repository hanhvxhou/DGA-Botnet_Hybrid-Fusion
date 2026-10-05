"""
selectAlgorithm.py — Huấn luyện và đánh giá 5 kỹ thuật để chọn ra mô hình tốt nhất
                     làm bộ trích xuất embedding cho M6.

5 CANDIDATE (đều self-contained, cùng protocol M1-M5):
  1. CNN   — Char-based parallel CNN (Woodbridge 2016, Yu-Koltun style)
  2. BiLSTM— Char-based BiLSTM với MaxPool-over-time
  3. DistilBERT (distilbert-base-uncased) — fine-tune toàn phần
  4. BERT       (bert-base-uncased)       — fine-tune toàn phần
  5. SecureBERT (ehsanaghaei/SecureBERT)  — fine-tune toàn phần

TÍNH CÔNG BẰNG CHO SO SÁNH VỚI M1-M5 VÀ CHUẨN BỊ M6:
- Dùng CHÍNH XÁC cùng split (seed=42, test_size=0.15, val_size=0.15) như M1-M5
  → train/val/test byte-to-byte giống nhau với M1-M5 → Bảng 5 so sánh được
- CÙNG heuristic phân loại DGA-R/DGA-P/DGA-W
- CÙNG cách đo năng lượng (codecarbon → fallback util-based)
- CÙNG cách đo inference (3 repeat, average, batch GPU)
- CÙNG cách đo kích thước mô hình (joblib/pt compressed)
- CÙNG format output: metrics.json, summary.txt, per_family_summary.txt,
  confusion_matrix.csv, test_predictions.csv

TIÊU CHÍ CHỌN WINNER: F1 cao nhất trên tập test (metric_selection = 'f1').

Input  : DataNew/{train.csv, val.csv, test.csv}  (đã chia sẵn để tránh leakage)
Output : out_select/
            cnn/ bilstm/ distilbert/ bert/ securebert/
                ├── model.pt
                ├── tokenizer/                  (chỉ cho BERT-family)
                ├── metrics.json
                ├── confusion_matrix.csv
                ├── test_predictions.csv
                ├── per_family_results.json
                ├── per_family_summary.txt
                └── summary.txt
            comparison.json             — tất cả 5 mô hình cạnh nhau
            comparison_table.txt        — bảng so sánh đẹp
            winner.txt                  — tên winner + lý do
            winner_model/               — copy của winner để M6 dùng

Cài đặt:
    pip install numpy pandas scikit-learn torch transformers
    pip install codecarbon  # tuỳ chọn - đo năng lượng chính xác

Chạy:
    python selectAlgorithm.py
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# CẤU HÌNH CHUNG (đồng bộ với M1-M5)
# ---------------------------------------------------------------------------
DATA_DIR    = "DataNew"      # Thư mục chứa train.csv, val.csv, test.csv
TRAIN_CSV   = "train.csv"
VAL_CSV     = "val.csv"
TEST_CSV    = "test.csv"
CSV_SEP     = ";"
DOMAIN_COL  = "domain"
LABEL_COL   = "label"
ROOT_OUT    = Path("out_select")
SEED        = 42
# Encode ký tự (giống M3/M4)
CHAR_VOCAB = ['<pad>'] + list("abcdefghijklmnopqrstuvwxyz0123456789-.")
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)
PAD_ID = 0
MAX_DOMAIN_LEN = 64

# Siêu tham số chung (CNN/BiLSTM) — đồng bộ M3-M5
BATCH_SIZE_DL   = 256
LEARNING_RATE_DL = 1e-3
NUM_EPOCHS_DL    = 30
EARLY_STOP_DL    = 5
WEIGHT_DECAY     = 1e-5
DROPOUT_DL       = 0.3
FC_HIDDEN_DL     = 256

# CNN
CNN_EMBED_DIM   = 128
CNN_CONV_KERNELS = [2, 3, 4, 5]
CNN_CONV_CHANNELS = 128

# BiLSTM
BILSTM_EMBED_DIM = 128
BILSTM_HIDDEN    = 128
BILSTM_LAYERS    = 2

# Siêu tham số cho BERT-family (fine-tune theo tiêu chuẩn HF)
# PHƯƠNG ÁN C: train 10 epochs như M3-M5 để so sánh công bằng.
#   - 10 epochs cho BERT đủ thời gian hội tụ trên domain ngẫu nhiên (không giống tiếng Anh)
#   - patience=3 để tránh dừng sớm khi có 1-2 epoch noise
#   - Chuẩn literature DGA với BERT: Yu et al. 2020 (10ep), Aghaei et al. 2022 (5ep),
#     Leyva et al. 2024 (5-8ep)
BERT_BATCH_SIZE   = 64
BERT_LEARNING_RATE = 2e-5
BERT_NUM_EPOCHS    = 10
BERT_EARLY_STOP    = 3
BERT_MAX_LEN       = 64
BERT_WARMUP_RATIO  = 0.1

# Các model BERT-family
BERT_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert":       "bert-base-uncased",
    "securebert": "ehsanaghaei/SecureBERT",
}

# Thiết bị
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# CPU profile (giống M1-M5)
CPU_MODEL              = "Intel 24-core (Xeon W / Core Ultra class)"
CPU_BASE_POWER_W       = 125.0
CPU_MAX_TURBO_POWER_W  = 280.0
CPU_AVG_MULTITHREAD_W  = 200.0
CPU_AVG_LIGHT_LOAD_W   = 60.0
CPU_LOGICAL_CORES      = 24

USE_CODECARBON = True

VOWELS = set("aeiou"); DIGITS = set("0123456789")

# Metric chọn winner
METRIC_SELECTION = "f1"   # "f1" hoặc "f1_dga_w" (F1 trên word-list DGA)


# ---------------------------------------------------------------------------
# ENERGY METER (giống hệt M1-M5)
# ---------------------------------------------------------------------------
def estimate_power_watts(cpu_seconds, wall_seconds):
    if wall_seconds <= 0: return CPU_AVG_LIGHT_LOAD_W
    util = min(1.0, cpu_seconds / (wall_seconds * CPU_LOGICAL_CORES))
    if util <= 0.05: return CPU_AVG_LIGHT_LOAD_W
    if util >= 1.0:  return CPU_AVG_MULTITHREAD_W
    return CPU_AVG_LIGHT_LOAD_W + (CPU_AVG_MULTITHREAD_W - CPU_AVG_LIGHT_LOAD_W) * (util - 0.05) / 0.95


class EnergyMeter:
    def __init__(self, label):
        self.label = label; self.method = "estimated_from_utilization"
        self.energy_joules = 0.0; self.power_watts = 0.0; self.utilization = 0.0
        self._tracker = None
        if USE_CODECARBON:
            try:
                from codecarbon import EmissionsTracker
                self._tracker = EmissionsTracker(project_name=label, measure_power_secs=1,
                                                  save_to_file=False, log_level="error")
                self.method = "codecarbon"
            except Exception:
                self._tracker = None

    def __enter__(self):
        self._t_wall = time.time(); self._t_cpu = time.process_time()
        if self._tracker is not None:
            try: self._tracker.start()
            except Exception: self._tracker = None
        return self

    def __exit__(self, *args):
        self.wall_seconds = time.time() - self._t_wall
        self.cpu_seconds = time.process_time() - self._t_cpu
        self.utilization = min(1.0, self.cpu_seconds / max(self.wall_seconds * CPU_LOGICAL_CORES, 1e-9))
        if self._tracker is not None:
            try:
                self._tracker.stop()
                kwh = getattr(self._tracker.final_emissions_data, "energy_consumed", None)
                if kwh is not None:
                    self.energy_joules = float(kwh) * 3_600_000.0
                    self.power_watts = self.energy_joules / max(self.wall_seconds, 1e-9)
                else: self._fallback()
            except Exception: self._fallback()
        else: self._fallback()

    def _fallback(self):
        self.power_watts = estimate_power_watts(self.cpu_seconds, self.wall_seconds)
        self.energy_joules = self.power_watts * self.wall_seconds


# ---------------------------------------------------------------------------
# DATA (giống hệt M1-M5)
# ---------------------------------------------------------------------------
def load_data_split(data_dir=DATA_DIR):
    """Đọc 3 file CSV đã chia sẵn (tránh leakage).
    Trả về: X_train, X_val, X_test, y_train, y_val, y_test, df_meta (dict)."""
    data_dir = Path(data_dir)
    paths = {
        "train": data_dir / TRAIN_CSV,
        "val":   data_dir / VAL_CSV,
        "test":  data_dir / TEST_CSV,
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy '{p}' (split={name}).")

    splits = {}
    for name, p in paths.items():
        df = pd.read_csv(p, sep=CSV_SEP, dtype=str, keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]
        if DOMAIN_COL not in df.columns or LABEL_COL not in df.columns:
            raise ValueError(f"[{name}] Cần '{DOMAIN_COL}' và '{LABEL_COL}'. "
                             f"Có: {list(df.columns)}")
        df[DOMAIN_COL] = df[DOMAIN_COL].astype(str).str.strip().str.lower()
        df[LABEL_COL]  = df[LABEL_COL].astype(int)
        df = df[df[DOMAIN_COL].str.len() > 0].reset_index(drop=True)
        splits[name] = df

    X_train = splits["train"][DOMAIN_COL].tolist()
    X_val   = splits["val"][DOMAIN_COL].tolist()
    X_test  = splits["test"][DOMAIN_COL].tolist()
    y_train = splits["train"][LABEL_COL].values
    y_val   = splits["val"][LABEL_COL].values
    y_test  = splits["test"][LABEL_COL].values

    meta = {
        "data_dir": str(data_dir),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_total": len(X_train) + len(X_val) + len(X_test),
        "n_train_dga":    int((y_train == 1).sum()),
        "n_train_benign": int((y_train == 0).sum()),
        "n_val_dga":      int((y_val   == 1).sum()),
        "n_val_benign":   int((y_val   == 0).sum()),
        "n_test_dga":     int((y_test  == 1).sum()),
        "n_test_benign":  int((y_test  == 0).sum()),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, meta


def encode_domain_char(domain, max_len=MAX_DOMAIN_LEN):
    d = domain.strip().lower()
    ids = [CHAR_TO_ID.get(c, PAD_ID) for c in d]
    if len(ids) > max_len: ids = ids[:max_len]
    else: ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def domains_to_char_tensor(domains, max_len=MAX_DOMAIN_LEN):
    arr = np.array([encode_domain_char(d, max_len) for d in domains], dtype=np.int64)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# HEURISTIC PHÂN LOẠI DGA (giống hệt M1-M5)
# ---------------------------------------------------------------------------
def shannon_entropy(s):
    if not s: return 0.0
    counts = Counter(s); n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())


def split_domain(domain):
    d = re.sub(r"^www\.", "", domain.strip().lower())
    parts = d.split(".")
    if len(parts) == 1: return parts[0], ""
    return ".".join(parts[:-1]), parts[-1]


def classify_dga_type(domain):
    main, _ = split_domain(domain)
    text = main.replace(".", "")
    n = max(len(text), 1)
    ent = shannon_entropy(text)
    vowel_ratio = sum(1 for c in text if c in VOWELS) / n
    digit_ratio = sum(1 for c in text if c in DIGITS) / n
    length = len(text)
    if length >= 12 and vowel_ratio >= 0.30 and ent <= 3.6: return "DGA-W"
    if ent >= 3.8 or digit_ratio >= 0.15: return "DGA-R"
    if 0.20 <= vowel_ratio <= 0.45 and ent < 3.8: return "DGA-P"
    return "DGA-R"


def classify_many(domains):
    return np.array([classify_dga_type(d) for d in domains])


# ---------------------------------------------------------------------------
# KIẾN TRÚC 1: CHAR-BASED CNN (Woodbridge 2016 style)
# Parallel Conv1D (k=2,3,4,5) → MaxPool-over-time → Concat → Dense → Softmax
# ---------------------------------------------------------------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=CNN_EMBED_DIM,
                  kernels=None, channels=CNN_CONV_CHANNELS,
                  fc_hidden=FC_HIDDEN_DL, dropout=DROPOUT_DL, pad_id=PAD_ID):
        super().__init__()
        kernels = kernels or CNN_CONV_KERNELS
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, channels, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.fc = nn.Sequential(
            nn.Linear(len(kernels) * channels, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x, return_embedding=False):
        emb = self.embedding(x)                 # (B, L, E)
        h = emb.transpose(1, 2)                  # (B, E, L)
        feats = [F.relu(conv(h)).max(dim=2)[0] for conv in self.convs]  # mỗi (B, C)
        features = torch.cat(feats, dim=1)       # (B, K*C)
        if return_embedding:
            return features
        return self.fc(features)


# ---------------------------------------------------------------------------
# KIẾN TRÚC 2: CHAR-BASED BILSTM (Woodbridge 2016 reference)
# Embed → BiLSTM(2 layers) → MaxPool-over-time → Dense → Softmax
# ---------------------------------------------------------------------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=BILSTM_EMBED_DIM,
                  hidden=BILSTM_HIDDEN, num_layers=BILSTM_LAYERS,
                  fc_hidden=FC_HIDDEN_DL, dropout=DROPOUT_DL, pad_id=PAD_ID):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden,
                             num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x, return_embedding=False):
        emb = self.embedding(x)                    # (B, L, E)
        lstm_out, _ = self.lstm(emb)                # (B, L, 2H)
        features = lstm_out.max(dim=1)[0]           # MaxPool over time (B, 2H)
        if return_embedding:
            return features
        return self.fc(features)


# ---------------------------------------------------------------------------
# KIẾN TRÚC 3-5: BERT-FAMILY (fine-tune)
# ---------------------------------------------------------------------------
class BertForDGA(nn.Module):
    """Wrapper để BERT/DistilBERT/SecureBERT nhận input_ids + attention_mask,
       trả về logits (B, 2). Có thể trả về embedding [CLS] nếu return_embedding=True."""
    def __init__(self, bert_model, hidden_size, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy [CLS] embedding — hidden state tại vị trí 0
        hidden = outputs.last_hidden_state[:, 0, :]   # (B, H)
        if return_embedding:
            return hidden
        logits = self.classifier(self.dropout(hidden))
        return logits


def load_bert_model_and_tokenizer(model_name):
    """Load model + tokenizer từ HuggingFace. Tự động phát hiện kiến trúc."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    # Hidden size có thể khác nhau: DistilBERT 768, BERT 768, SecureBERT 768
    hidden_size = bert_model.config.hidden_size
    return bert_model, tokenizer, hidden_size


# ---------------------------------------------------------------------------
# DATASET WRAPPERS
# ---------------------------------------------------------------------------
class BertDomainDataset(torch.utils.data.Dataset):
    """Dataset cho BERT-family: tokenize lazy, padded đến BERT_MAX_LEN."""
    def __init__(self, domains, labels, tokenizer, max_len=BERT_MAX_LEN):
        self.domains = domains
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.domains[idx],
            truncation=True, padding="max_length", max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# TRAIN CNN / BILSTM
# ---------------------------------------------------------------------------
def train_char_model(model, train_loader, val_loader, y_val,
                      num_epochs=NUM_EPOCHS_DL, lr=LEARNING_RATE_DL,
                      patience=EARLY_STOP_DL):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history, best_f1, best_state, best_epoch, no_imp = [], -1.0, None, 0, 0
    for epoch in range(1, num_epochs + 1):
        model.train(); rl, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
            rl += loss.item() * xb.size(0); n += xb.size(0)
        tl = rl / n
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
        vp = np.concatenate(preds)
        vf1 = f1_score(y_val, vp, zero_division=0); vacc = accuracy_score(y_val, vp)
        history.append({"epoch": epoch, "train_loss": tl, "val_f1": vf1, "val_acc": vacc})
        print(f"    Epoch {epoch:2d}/{num_epochs}  loss={tl:.4f}  "
              f"val_acc={vacc*100:.2f}%  val_F1={vf1*100:.2f}%"
              + ("  *" if vf1 > best_f1 else ""))
        if vf1 > best_f1 + 1e-5:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    → Early stop epoch {epoch}"); break
    return history, best_state, best_epoch, best_f1


def predict_char(model, X_tensor, batch_size=512):
    model.eval()
    preds, probas = [], []
    n = X_tensor.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X_tensor[i:i+batch_size].to(DEVICE)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probas.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probas)


def measure_inference_char(model, X_tensor_gpu, n_samples, n_repeats=3):
    rs = []
    for _ in range(n_repeats):
        with EnergyMeter("inference") as m:
            predict_char(model, X_tensor_gpu, batch_size=1024)
            if USE_CUDA: torch.cuda.synchronize()
        rs.append({"wall": m.wall_seconds, "cpu": m.cpu_seconds,
                   "energy": m.energy_joules, "power": m.power_watts,
                   "util": m.utilization, "method": m.method})
    return _summary_inference(rs, n_samples)


def _summary_inference(rs, n_samples):
    avg = lambda k: float(np.mean([r[k] for r in rs]))
    return {
        "n_repeats": len(rs), "n_samples": n_samples,
        "avg_wall_seconds": avg("wall"), "avg_cpu_seconds": avg("cpu"),
        "avg_energy_joules": avg("energy"), "avg_power_watts": avg("power"),
        "avg_cpu_utilization": avg("util"),
        "energy_per_sample_mJ": avg("energy") / n_samples * 1000.0,
        "wall_per_sample_ms":   avg("wall") / n_samples * 1000.0,
        "method": rs[0]["method"],
    }


# ---------------------------------------------------------------------------
# TRAIN BERT-FAMILY
# ---------------------------------------------------------------------------
def train_bert_model(model, train_loader, val_loader, y_val,
                      num_epochs=BERT_NUM_EPOCHS, lr=BERT_LEARNING_RATE,
                      patience=BERT_EARLY_STOP, warmup_ratio=BERT_WARMUP_RATIO):
    from transformers import get_linear_schedule_with_warmup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=int(total_steps * warmup_ratio),
                                                  num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    history, best_f1, best_state, best_epoch, no_imp = [], -1.0, None, 0, 0
    for epoch in range(1, num_epochs + 1):
        model.train(); rl, n = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()
            rl += loss.item() * labels.size(0); n += labels.size(0)
        tl = rl / n

        model.eval(); preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                logits = model(input_ids, attention_mask)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        vp = np.concatenate(preds)
        vf1 = f1_score(y_val, vp, zero_division=0); vacc = accuracy_score(y_val, vp)
        history.append({"epoch": epoch, "train_loss": tl, "val_f1": vf1, "val_acc": vacc})
        print(f"    Epoch {epoch:2d}/{num_epochs}  loss={tl:.4f}  "
              f"val_acc={vacc*100:.2f}%  val_F1={vf1*100:.2f}%"
              + ("  *" if vf1 > best_f1 else ""))
        if vf1 > best_f1 + 1e-5:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    → Early stop epoch {epoch}"); break
    return history, best_state, best_epoch, best_f1


def predict_bert(model, loader):
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            logits = model(input_ids, attention_mask)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probas.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probas)


def measure_inference_bert(model, loader, n_samples, n_repeats=3):
    rs = []
    for _ in range(n_repeats):
        with EnergyMeter("inference") as m:
            predict_bert(model, loader)
            if USE_CUDA: torch.cuda.synchronize()
        rs.append({"wall": m.wall_seconds, "cpu": m.cpu_seconds,
                   "energy": m.energy_joules, "power": m.power_watts,
                   "util": m.utilization, "method": m.method})
    return _summary_inference(rs, n_samples)


# ---------------------------------------------------------------------------
# ĐÁNH GIÁ CHUNG
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba, label="Test", silent=False):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try: auc = roc_auc_score(y_true, y_proba)
    except ValueError: auc = float("nan")
    if not silent:
        print(f"\n  === {label} ===")
    print(f"  Accuracy : {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  "
              f"Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%  |  AUC: {auc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": auc, "confusion_matrix": cm.tolist()}


def evaluate_per_family(y_true, y_pred, dga_types):
    results = {}
    benign_mask = (y_true == 0)
    for lbl in ["DGA-R", "DGA-P", "DGA-W"]:
        dga_mask = (y_true == 1) & (dga_types == lbl)
        n_dga = int(dga_mask.sum())
        if n_dga == 0:
            results[lbl] = {"n_dga": 0}; continue
        sub = benign_mask | dga_mask
        yt, yp = y_true[sub], y_pred[sub]
        results[lbl] = {
            "n_dga": n_dga, "n_benign": int(benign_mask.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        }
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# SAVE COMMON — lưu metrics/summary/confusion/predictions/per_family cho mỗi model
# ---------------------------------------------------------------------------
def save_all_outputs(model_name, out_dir, model_path_size_mb, n_params,
                      val_metrics, test_metrics, per_family, dga_dist,
                      em, energy_infer, feat_time,
                      data_info, config_info, test_df,
                      env_extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model":       model_name,
        "validation":  val_metrics,
        "test":        test_metrics,
        "per_family":  per_family,
        "dga_type_distribution": dga_dist,
        "model_size":  {"compressed_MB": model_path_size_mb, "num_parameters": n_params},
        "training_cost": {
            "wall_seconds": em.wall_seconds, "cpu_seconds": em.cpu_seconds,
            "cpu_utilization": em.utilization, "avg_power_watts": em.power_watts,
            "energy_joules": em.energy_joules, "energy_Wh": em.energy_joules/3600.0,
            "energy_method": em.method,
        },
        "inference_cost": energy_infer,
        "feature_extraction_seconds": feat_time,
        "data": data_info,
        "config": config_info,
        "heuristic_thresholds": {
            "DGA-W": "length >= 12 AND vowel_ratio >= 0.30 AND entropy <= 3.6",
            "DGA-R": "entropy >= 3.8 OR digit_ratio >= 0.15",
            "DGA-P": "0.20 <= vowel_ratio <= 0.45 AND entropy < 3.8",
        },
        "cpu_profile": {
            "model": CPU_MODEL, "base_power_W": CPU_BASE_POWER_W,
            "max_turbo_power_W": CPU_MAX_TURBO_POWER_W,
            "avg_multithread_W": CPU_AVG_MULTITHREAD_W,
            "avg_light_load_W": CPU_AVG_LIGHT_LOAD_W,
            "logical_cores": CPU_LOGICAL_CORES,
        },
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "cuda_available": USE_CUDA,
        },
    }
    if env_extra:
        metrics["environment"].update(env_extra)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    cm = np.array(test_metrics["confusion_matrix"])
    pd.DataFrame(cm, index=["true_benign", "true_DGA"],
                 columns=["pred_benign", "pred_DGA"]).to_csv(out_dir / "confusion_matrix.csv")
    test_df.to_csv(out_dir / "test_predictions.csv", index=False, sep=CSV_SEP)

    # per_family_results.json
    with open(out_dir / "per_family_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "dga_type_distribution": dga_dist,
            "overall": {"accuracy": test_metrics["accuracy"], "f1": test_metrics["f1"]},
            "per_family": per_family,
            "heuristic_thresholds": metrics["heuristic_thresholds"],
        }, f, indent=2, ensure_ascii=False, default=float)

    # per_family_summary.txt
    pf_lines = [f"====== {model_name} — F1 theo loại DGA ======",
                 f"Test samples : {data_info['n_test']:,}", ""]
    pf_lines.append("Phân bố loại DGA (heuristic):")
    for k in ["DGA-R", "DGA-P", "DGA-W", "benign"]:
        cnt = dga_dist.get(k, 0)
        pf_lines.append(f"  {k:<8} : {cnt:>8,}  ({cnt/data_info['n_test']*100:5.2f}%)")
    pf_lines.append("")
    pf_lines.append(f"Toàn test: Accuracy = {test_metrics['accuracy']*100:.2f}%   "
                     f"F1 = {test_metrics['f1']*100:.2f}%")
    pf_lines.append("")
    pf_lines.append(f"{'Loại':<8} {'n_dga':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    for lbl in ["DGA-R", "DGA-P", "DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            pf_lines.append(f"{lbl:<8} {r['n_dga']:>8,} "
                            f"{r['accuracy']*100:>7.2f} {r['precision']*100:>7.2f} "
                            f"{r['recall']*100:>7.2f} {r['f1']*100:>7.2f}")
    (out_dir / "per_family_summary.txt").write_text("\n".join(pf_lines), encoding="utf-8")

    # summary.txt
    lines = [
        f"================ {model_name} ================",
        f"Data dir: {DATA_DIR}/ (train.csv, val.csv, test.csv)",
        f"Tổng mẫu           : {data_info['total_samples']:,}",
        f"Chia                : Train={data_info['n_train']:,} | Val={data_info['n_val']:,} | Test={data_info['n_test']:,}",
        "",
        "[Hiệu năng — Test]",
        f"  Accuracy         : {test_metrics['accuracy']*100:.2f}%",
        f"  Precision        : {test_metrics['precision']*100:.2f}%",
        f"  Recall           : {test_metrics['recall']*100:.2f}%",
        f"  F1-score         : {test_metrics['f1']*100:.2f}%",
        f"  AUC-ROC          : {test_metrics['auc']:.4f}",
        "",
        "[Chi phí huấn luyện]",
        f"  Wall time        : {em.wall_seconds:.2f} s  ({em.wall_seconds/60:.2f} phút)",
        f"  CPU utilization  : {em.utilization*100:.1f}%",
        f"  Avg power        : {em.power_watts:.1f} W",
        f"  Energy           : {em.energy_joules:.2f} J  ({em.energy_joules/3600:.4f} Wh)",
        f"  Đo bằng          : {em.method}",
        "",
        "[Chi phí suy diễn / mẫu]",
        f"  Wall             : {energy_infer['wall_per_sample_ms']:.4f} ms/mẫu",
        f"  Energy           : {energy_infer['energy_per_sample_mJ']:.4f} mJ/mẫu",
        f"  Avg power        : {energy_infer['avg_power_watts']:.1f} W",
        f"  Đo bằng          : {energy_infer['method']}",
        "",
        "[Kích thước mô hình]",
        f"  File (nén)       : {model_path_size_mb:.2f} MB",
        f"  Số tham số       : {n_params:,}",
        "",
        "[Bảng 5 — F1 theo loại DGA]",
        f"  {'Loại':<8} {'n_dga':>8} {'F1 (%)':>8} {'Prec':>8} {'Recall':>8}",
    ]
    for lbl in ["DGA-R", "DGA-P", "DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            lines.append(f"  {lbl:<8} {r['n_dga']:>8,} "
                         f"{r['f1']*100:>7.2f} {r['precision']*100:>7.2f} "
                         f"{r['recall']*100:>7.2f}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# PIPELINE 1: CNN
# ---------------------------------------------------------------------------
def run_cnn(X_train, X_val, X_test, y_train, y_val, y_test, dga_types, data_info):
    print("\n" + "="*70)
    print(" [1/5] CHAR-BASED CNN (Woodbridge 2016 / parallel Conv1D)")
    print("="*70)
    out_dir = ROOT_OUT / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    Xtr = domains_to_char_tensor(X_train); Xva = domains_to_char_tensor(X_val); Xte = domains_to_char_tensor(X_test)
    feat_time = time.time() - t0

    model = CharCNN().to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Tham số: {n_params:,}  |  Device: {DEVICE}")

    train_ds = TensorDataset(Xtr, torch.from_numpy(y_train.astype(np.int64)))
    val_ds   = TensorDataset(Xva, torch.from_numpy(y_val.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_DL, shuffle=True,
                               num_workers=0, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_DL*4, shuffle=False,
                             num_workers=0, pin_memory=USE_CUDA)

    print("  → Huấn luyện ...")
    with EnergyMeter("training_cnn") as em:
        history, best_state, best_epoch, best_f1_val = train_char_model(
            model, train_loader, val_loader, y_val
        )
    print(f"  Wall={em.wall_seconds:.1f}s  Energy={em.energy_joules:.1f}J  ({em.method})")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_proba = predict_char(model, Xva)
    val_metrics = evaluate(y_val, val_pred, val_proba, label="Validation")

    Xte_gpu = Xte.to(DEVICE)
    energy_infer = measure_inference_char(model, Xte_gpu, len(X_test))
    test_pred, test_proba = predict_char(model, Xte)
    test_metrics = evaluate(y_test, test_pred, test_proba, label="Test")

    per_family = evaluate_per_family(y_test, test_pred, dga_types)
    dga_dist = {k: int(v) for k, v in Counter(np.where(y_test==1, dga_types, "benign")).items()}

    # Lưu mô hình
    model_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(),
                "arch": "CharCNN",
                "vocab_size": VOCAB_SIZE, "embed_dim": CNN_EMBED_DIM,
                "kernels": CNN_CONV_KERNELS, "channels": CNN_CONV_CHANNELS,
                "fc_hidden": FC_HIDDEN_DL, "dropout": DROPOUT_DL,
                "max_domain_len": MAX_DOMAIN_LEN,
                "best_epoch": best_epoch, "best_val_f1": best_f1_val,
                "training_history": history}, model_path)
    size_mb = get_size_mb(model_path)

    test_df = pd.DataFrame({"domain": X_test, "true_label": y_test,
                             "pred_label": test_pred, "prob_DGA": test_proba})
    metrics = save_all_outputs(
        "CNN (Char-based)", out_dir, size_mb, n_params,
        val_metrics, test_metrics, per_family, dga_dist,
        em, energy_infer, feat_time, data_info,
        config_info={"arch": "CharCNN", "embed_dim": CNN_EMBED_DIM,
                      "kernels": CNN_CONV_KERNELS, "channels": CNN_CONV_CHANNELS,
                      "fc_hidden": FC_HIDDEN_DL, "dropout": DROPOUT_DL,
                      "batch_size": BATCH_SIZE_DL, "lr": LEARNING_RATE_DL,
                      "num_epochs": NUM_EPOCHS_DL, "early_stop": EARLY_STOP_DL,
                      "weight_decay": WEIGHT_DECAY, "seed": SEED},
        test_df=test_df,
    )
    return metrics, model, "char", None  # encoder_type, tokenizer


# ---------------------------------------------------------------------------
# PIPELINE 2: BiLSTM
# ---------------------------------------------------------------------------
def run_bilstm(X_train, X_val, X_test, y_train, y_val, y_test, dga_types, data_info):
    print("\n" + "="*70)
    print(" [2/5] CHAR-BASED BiLSTM (2 layers)")
    print("="*70)
    out_dir = ROOT_OUT / "bilstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    Xtr = domains_to_char_tensor(X_train); Xva = domains_to_char_tensor(X_val); Xte = domains_to_char_tensor(X_test)
    feat_time = time.time() - t0

    model = CharBiLSTM().to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Tham số: {n_params:,}  |  Device: {DEVICE}")

    train_ds = TensorDataset(Xtr, torch.from_numpy(y_train.astype(np.int64)))
    val_ds   = TensorDataset(Xva, torch.from_numpy(y_val.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_DL, shuffle=True,
                               num_workers=0, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_DL*4, shuffle=False,
                             num_workers=0, pin_memory=USE_CUDA)

    print("  → Huấn luyện ...")
    with EnergyMeter("training_bilstm") as em:
        history, best_state, best_epoch, best_f1_val = train_char_model(
            model, train_loader, val_loader, y_val
        )
    print(f"  Wall={em.wall_seconds:.1f}s  Energy={em.energy_joules:.1f}J  ({em.method})")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_proba = predict_char(model, Xva)
    val_metrics = evaluate(y_val, val_pred, val_proba, label="Validation")

    Xte_gpu = Xte.to(DEVICE)
    energy_infer = measure_inference_char(model, Xte_gpu, len(X_test))
    test_pred, test_proba = predict_char(model, Xte)
    test_metrics = evaluate(y_test, test_pred, test_proba, label="Test")

    per_family = evaluate_per_family(y_test, test_pred, dga_types)
    dga_dist = {k: int(v) for k, v in Counter(np.where(y_test==1, dga_types, "benign")).items()}

    model_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(),
                "arch": "CharBiLSTM",
                "vocab_size": VOCAB_SIZE, "embed_dim": BILSTM_EMBED_DIM,
                "hidden": BILSTM_HIDDEN, "num_layers": BILSTM_LAYERS,
                "fc_hidden": FC_HIDDEN_DL, "dropout": DROPOUT_DL,
                "max_domain_len": MAX_DOMAIN_LEN,
                "best_epoch": best_epoch, "best_val_f1": best_f1_val,
                "training_history": history}, model_path)
    size_mb = get_size_mb(model_path)

    test_df = pd.DataFrame({"domain": X_test, "true_label": y_test,
                             "pred_label": test_pred, "prob_DGA": test_proba})
    metrics = save_all_outputs(
        "BiLSTM (Char-based)", out_dir, size_mb, n_params,
        val_metrics, test_metrics, per_family, dga_dist,
        em, energy_infer, feat_time, data_info,
        config_info={"arch": "CharBiLSTM", "embed_dim": BILSTM_EMBED_DIM,
                      "hidden": BILSTM_HIDDEN, "num_layers": BILSTM_LAYERS,
                      "fc_hidden": FC_HIDDEN_DL, "dropout": DROPOUT_DL,
                      "batch_size": BATCH_SIZE_DL, "lr": LEARNING_RATE_DL,
                      "num_epochs": NUM_EPOCHS_DL, "early_stop": EARLY_STOP_DL,
                      "weight_decay": WEIGHT_DECAY, "seed": SEED},
        test_df=test_df,
    )
    return metrics, model, "char", None


# ---------------------------------------------------------------------------
# PIPELINE 3-5: BERT-FAMILY
# ---------------------------------------------------------------------------
def run_bert(key, model_name,
              X_train, X_val, X_test, y_train, y_val, y_test,
              dga_types, data_info, idx, total):
    print("\n" + "="*70)
    print(f" [{idx}/{total}] {key.upper()} — fine-tune {model_name}")
    print("="*70)
    out_dir = ROOT_OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  → Load model + tokenizer từ {model_name} ...")
    t0 = time.time()
    try:
        bert_model, tokenizer, hidden_size = load_bert_model_and_tokenizer(model_name)
    except Exception as e:
        print(f"  !! LỖI: Không load được {model_name}: {e}")
        print(f"  !! Bỏ qua mô hình này.")
        return None, None, None, None
    feat_time = time.time() - t0

    model = BertForDGA(bert_model, hidden_size).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Tham số: {n_params:,}  |  Hidden size: {hidden_size}  |  Device: {DEVICE}")

    train_ds = BertDomainDataset(X_train, y_train, tokenizer)
    val_ds   = BertDomainDataset(X_val, y_val, tokenizer)
    test_ds  = BertDomainDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BERT_BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_ds, batch_size=BERT_BATCH_SIZE*2, shuffle=False,
                             num_workers=0, pin_memory=USE_CUDA)
    test_loader = DataLoader(test_ds, batch_size=BERT_BATCH_SIZE*2, shuffle=False,
                              num_workers=0, pin_memory=USE_CUDA)

    print(f"  → Huấn luyện ({BERT_NUM_EPOCHS} epochs, batch {BERT_BATCH_SIZE}, "
          f"lr={BERT_LEARNING_RATE}) ...")
    with EnergyMeter(f"training_{key}") as em:
        history, best_state, best_epoch, best_f1_val = train_bert_model(
            model, train_loader, val_loader, y_val
        )
    print(f"  Wall={em.wall_seconds:.1f}s ({em.wall_seconds/60:.1f} phút)  "
          f"Energy={em.energy_joules:.1f}J  ({em.method})")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_proba = predict_bert(model, val_loader)
    val_metrics = evaluate(y_val, val_pred, val_proba, label="Validation")

    energy_infer = measure_inference_bert(model, test_loader, len(X_test))
    test_pred, test_proba = predict_bert(model, test_loader)
    test_metrics = evaluate(y_test, test_pred, test_proba, label="Test")

    per_family = evaluate_per_family(y_test, test_pred, dga_types)
    dga_dist = {k: int(v) for k, v in Counter(np.where(y_test==1, dga_types, "benign")).items()}

    # Lưu model + tokenizer
    model_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(),
                "arch": "BertForDGA",
                "model_name": model_name, "hidden_size": hidden_size,
                "max_len": BERT_MAX_LEN,
                "best_epoch": best_epoch, "best_val_f1": best_f1_val,
                "training_history": history}, model_path)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    size_mb = get_size_mb(model_path)

    test_df = pd.DataFrame({"domain": X_test, "true_label": y_test,
                             "pred_label": test_pred, "prob_DGA": test_proba})
    metrics = save_all_outputs(
        f"{key.title()} ({model_name})", out_dir, size_mb, n_params,
        val_metrics, test_metrics, per_family, dga_dist,
        em, energy_infer, feat_time, data_info,
        config_info={"arch": "BertForDGA", "model_name": model_name,
                      "hidden_size": hidden_size, "max_len": BERT_MAX_LEN,
                      "batch_size": BERT_BATCH_SIZE, "lr": BERT_LEARNING_RATE,
                      "num_epochs": BERT_NUM_EPOCHS, "early_stop": BERT_EARLY_STOP,
                      "warmup_ratio": BERT_WARMUP_RATIO, "seed": SEED},
        test_df=test_df,
        env_extra={"transformers_model": model_name, "hidden_size": hidden_size},
    )
    return metrics, model, "bert", tokenizer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    if USE_CUDA: torch.cuda.manual_seed_all(SEED)

    print(f"[CPU] {CPU_MODEL}")
    print(f"[GPU] CUDA={USE_CUDA}  |  Device={DEVICE}")
    if USE_CUDA:
        print(f"      GPU name: {torch.cuda.get_device_name(0)}")
    print(f"      GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # -- 1. Đọc data + chia tập (GIỐNG HỆT M1-M5) --
    print(f"\n[Data] Đọc 3 file CSV từ {DATA_DIR}/ ...")
    X_train, X_val, X_test, y_train, y_val, y_test, data_meta = load_data_split()
    n_total = data_meta["n_total"]
    n_pos   = data_meta["n_train_dga"]    + data_meta["n_val_dga"]    + data_meta["n_test_dga"]
    n_neg   = data_meta["n_train_benign"] + data_meta["n_val_benign"] + data_meta["n_test_benign"]
    print(f"       Tổng: {n_total:,}  |  Train: {len(X_train):,}  "
          f"Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"       DGA: {n_pos:,}  |  benign: {n_neg:,}")

    # -- 2. Phân loại heuristic trên tập test --
    print("[Heuristic] Phân loại tập test thành DGA-R/DGA-P/DGA-W ...")
    dga_types = classify_many(X_test)
    dga_display = np.where(y_test == 1, dga_types, "benign")
    c = Counter(dga_display)
    for k in ["DGA-R", "DGA-P", "DGA-W", "benign"]:
        print(f"           {k:<8} : {c[k]:>8,}  ({c[k]/len(X_test)*100:5.2f}%)")

    data_info = {"data_dir": DATA_DIR, "splits": {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}, "total_samples": n_total,
                  "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                  "n_positive_total": n_pos, "n_negative_total": n_neg}

    results = {}

    # -- 3. CHẠY 5 CANDIDATE --
    results["cnn"],        _, _, _ = run_cnn(X_train, X_val, X_test, y_train, y_val, y_test,
                                              dga_types, data_info)
    results["bilstm"],     _, _, _ = run_bilstm(X_train, X_val, X_test, y_train, y_val, y_test,
                                                 dga_types, data_info)
    for idx, (key, model_name) in enumerate(BERT_MODELS.items(), start=3):
        res = run_bert(key, model_name, X_train, X_val, X_test, y_train, y_val, y_test,
                        dga_types, data_info, idx=idx, total=5)
        if res is not None and res[0] is not None:
            results[key] = res[0]

    # -- 4. CHỌN WINNER + SO SÁNH --
    print("\n" + "="*70)
    print(" SO SÁNH 5 CANDIDATE")
    print("="*70)

    comparison = {}
    for key, m in results.items():
        if m is None: continue
        comparison[key] = {
            "model":              m["model"],
            "test_accuracy":      m["test"]["accuracy"],
            "test_precision":     m["test"]["precision"],
            "test_recall":        m["test"]["recall"],
            "test_f1":            m["test"]["f1"],
            "test_auc":           m["test"]["auc"],
            "train_seconds":      m["training_cost"]["wall_seconds"],
            "train_energy_J":     m["training_cost"]["energy_joules"],
            "inference_ms_per_sample": m["inference_cost"]["wall_per_sample_ms"],
            "inference_energy_per_sample_mJ": m["inference_cost"]["energy_per_sample_mJ"],
            "model_size_MB":      m["model_size"]["compressed_MB"],
            "num_parameters":     m["model_size"]["num_parameters"],
            "f1_dga_R":           m["per_family"].get("DGA-R", {}).get("f1", None),
            "f1_dga_P":           m["per_family"].get("DGA-P", {}).get("f1", None),
            "f1_dga_W":           m["per_family"].get("DGA-W", {}).get("f1", None),
        }

    with open(ROOT_OUT / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=float)

    # Bảng text
    header = ("| {:<18} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>12} |"
              .format("Model", "Acc %", "F1 %", "AUC", "DGA-R", "DGA-P", "DGA-W",
                      "Train (s)", "Infer ms", "Size (MB)"))
    sep = "|" + "-" * (len(header) - 2) + "|"
    table_lines = [header, sep]
    for key, c_ in comparison.items():
        def pct(v):
            return f"{v*100:.2f}" if v is not None else "—"
        table_lines.append("| {:<18} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10.1f} | {:>10.4f} | {:>12.2f} |".format(
            key,
            pct(c_["test_accuracy"]), pct(c_["test_f1"]),
            f"{c_['test_auc']:.4f}" if c_["test_auc"] else "—",
            pct(c_["f1_dga_R"]), pct(c_["f1_dga_P"]), pct(c_["f1_dga_W"]),
            c_["train_seconds"],
            c_["inference_ms_per_sample"],
            c_["model_size_MB"],
        ))
    table_text = "\n".join(table_lines)
    print(table_text)

    # Xác định winner
    valid = {k: c_ for k, c_ in comparison.items() if c_["test_f1"] is not None}
    if METRIC_SELECTION == "f1":
        winner_key = max(valid, key=lambda k: valid[k]["test_f1"])
    elif METRIC_SELECTION == "f1_dga_w":
        winner_key = max(valid, key=lambda k: valid[k].get("f1_dga_W", 0) or 0)
    else:
        winner_key = max(valid, key=lambda k: valid[k]["test_f1"])

    winner_info = comparison[winner_key]
    winner_msg = (f"\n=== WINNER: {winner_key.upper()} — {winner_info['model']} ===\n"
                   f"  Test F1      : {winner_info['test_f1']*100:.2f}%\n"
                   f"  Test Acc     : {winner_info['test_accuracy']*100:.2f}%\n"
                   f"  F1 DGA-R     : {winner_info['f1_dga_R']*100:.2f}%\n"
                   f"  F1 DGA-P     : {winner_info['f1_dga_P']*100:.2f}%\n"
                   f"  F1 DGA-W     : {winner_info['f1_dga_W']*100:.2f}%\n"
                   f"  Train time   : {winner_info['train_seconds']:.1f} s\n"
                   f"  Inference    : {winner_info['inference_ms_per_sample']:.4f} ms/mẫu\n"
                   f"  Model size   : {winner_info['model_size_MB']:.2f} MB\n"
                   f"  Tham số      : {winner_info['num_parameters']:,}\n"
                   f"\n  Tiêu chí chọn: {METRIC_SELECTION}")
    print(winner_msg)

    (ROOT_OUT / "comparison_table.txt").write_text(table_text + winner_msg, encoding="utf-8")
    (ROOT_OUT / "winner.txt").write_text(
        f"winner_key={winner_key}\n"
        f"model={winner_info['model']}\n"
        f"test_f1={winner_info['test_f1']*100:.2f}\n"
        f"metric_selection={METRIC_SELECTION}\n",
        encoding="utf-8",
    )

    # -- 5. Copy thư mục winner thành winner_model/ để M6 dùng --
    winner_src = ROOT_OUT / winner_key
    winner_dst = ROOT_OUT / "winner_model"
    if winner_dst.exists():
        shutil.rmtree(winner_dst)
    shutil.copytree(winner_src, winner_dst)
    print(f"\n  → Đã copy {winner_src} → {winner_dst}")
    print(f"  → Dùng thư mục này làm embedding extractor cho M6.")


if __name__ == "__main__":
    main()
