"""
M5 — Dual-Embedding Transformer: hai luồng Char + Bigram (Ding et al., 2023)
Self-contained.

Kiến trúc: hai luồng Transformer song song, mỗi luồng 6 tầng encoder
(d_model=256, nhead=8). Luồng 1 xử lý ký tự, luồng 2 xử lý bigram. Mỗi luồng
có 1 token CLS đứng đầu; CLS của hai luồng được nối lại (Fusion) rồi đưa
qua MLP + Softmax.

Input  : DataNew/{train.csv, val.csv, test.csv}  (đã chia sẵn để tránh leakage)
Output : out_M5/
            - model.pt, metrics.json, confusion_matrix.csv, test_predictions.csv,
              per_family_results.json, per_family_summary.txt, summary.txt

Cài đặt:
    pip install numpy pandas scikit-learn torch
    pip install codecarbon  # tuỳ chọn

Chạy:
    python M5.py
"""
from __future__ import annotations

import json, math, os, re, sys, time, platform
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
# CẤU HÌNH
# ---------------------------------------------------------------------------
DATA_DIR    = "DataNew"      # Thư mục chứa train.csv, val.csv, test.csv
TRAIN_CSV   = "train.csv"
VAL_CSV     = "val.csv"
TEST_CSV    = "test.csv"
CSV_SEP     = ";"
DOMAIN_COL  = "domain"
LABEL_COL   = "label"
OUTPUT_DIR  = Path("out_M5")
SEED        = 42
# Encode ký tự
CHAR_VOCAB = ['<pad>', '<cls>'] + list("abcdefghijklmnopqrstuvwxyz0123456789-.")
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)
PAD_ID = 0
CLS_ID = 1
MAX_DOMAIN_LEN = 64        # bao gồm [CLS] ở đầu

# Encode bigram — dùng character set ký tự cơ bản
_BIGRAM_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789-.")
BIGRAM_VOCAB = ['<pad>', '<cls>', '<unk>'] + [c1 + c2 for c1 in _BIGRAM_CHARS for c2 in _BIGRAM_CHARS]
BIGRAM_TO_ID = {b: i for i, b in enumerate(BIGRAM_VOCAB)}
BIGRAM_VOCAB_SIZE = len(BIGRAM_VOCAB)
BIGRAM_PAD_ID = 0
BIGRAM_CLS_ID = 1
BIGRAM_UNK_ID = 2
MAX_BIGRAM_LEN = MAX_DOMAIN_LEN    # tương đương chiều dài character sequence

# Kiến trúc M5 — Dual-Embedding Transformer (Ding et al., 2023)
D_MODEL        = 256
N_HEADS        = 8
N_LAYERS       = 6
FFN_DIM        = 1024
FC_HIDDEN      = 256
DROPOUT        = 0.1

BATCH_SIZE    = 256
# Transformer cần lr nhỏ hơn CNN/BiLSTM: lr=5e-4 thay vì 1e-3 (Vaswani et al. 2017,
# Devlin et al. 2019). Kết hợp với AdamW (decoupled weight decay), tránh divergence sớm
# do Transformer nhạy cảm với learning rate.
LEARNING_RATE = 5e-4
NUM_EPOCHS    = 30
EARLY_STOP_PATIENCE = 5
WEIGHT_DECAY  = 1e-5

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

CPU_MODEL              = "Intel 24-core (Xeon W / Core Ultra class)"
CPU_BASE_POWER_W       = 125.0
CPU_MAX_TURBO_POWER_W  = 280.0
CPU_AVG_MULTITHREAD_W  = 200.0
CPU_AVG_LIGHT_LOAD_W   = 60.0
CPU_LOGICAL_CORES      = 24

USE_CODECARBON = True

VOWELS = set("aeiou"); DIGITS = set("0123456789")


# ---------------------------------------------------------------------------
# ENERGY METER
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
# DATA
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


def encode_domain(domain, max_len=MAX_DOMAIN_LEN):
    """Encode chuỗi ký tự, thêm [CLS] ở đầu."""
    d = domain.strip().lower()
    ids = [CLS_ID] + [CHAR_TO_ID.get(c, PAD_ID) for c in d]
    if len(ids) > max_len: ids = ids[:max_len]
    else: ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def domains_to_tensor(domains, max_len=MAX_DOMAIN_LEN):
    arr = np.array([encode_domain(d, max_len) for d in domains], dtype=np.int64)
    return torch.from_numpy(arr)


def encode_domain_bigram(domain, max_len=MAX_BIGRAM_LEN):
    """Encode chuỗi bigram, thêm [CLS] ở đầu."""
    d = domain.strip().lower()
    ids = [BIGRAM_CLS_ID]
    for i in range(len(d) - 1):
        bg = d[i:i+2]
        ids.append(BIGRAM_TO_ID.get(bg, BIGRAM_UNK_ID))
    if len(ids) > max_len: ids = ids[:max_len]
    else: ids = ids + [BIGRAM_PAD_ID] * (max_len - len(ids))
    return ids


def domains_to_bigram_tensor(domains, max_len=MAX_BIGRAM_LEN):
    arr = np.array([encode_domain_bigram(d, max_len) for d in domains], dtype=np.int64)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# HEURISTIC PHÂN LOẠI DGA
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
# KIẾN TRÚC M5: DUAL-EMBEDDING TRANSFORMER
# Hai luồng Transformer Encoder song song:
#   - Luồng character: [CLS] + char tokens → Char Embedding + Positional
#                      → TransformerEncoder ×6 (d=256, heads=8) → h_cls_char
#   - Luồng bigram   : [CLS] + bigram tokens → Bigram Embedding + Positional
#                      → TransformerEncoder ×6 → h_cls_bigram
# Fusion: concat([h_cls_char, h_cls_bigram]) → MLP + Softmax
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBranch(nn.Module):
    """Một luồng: Embedding + PosEnc + Transformer Encoder ×N_LAYERS."""
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                  n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
                  pad_id=0, max_len=MAX_DOMAIN_LEN):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (B, L) long; vị trí 0 là [CLS]
        pad_mask = (x == self.pad_id)                 # (B, L) True ở pad
        emb = self.embedding(x)                        # (B, L, D)
        emb = self.pos_enc(emb)
        emb = self.embed_dropout(emb)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)   # (B, L, D)
        h_cls = out[:, 0, :]                           # token CLS
        return h_cls


class DualEmbeddingTransformer(nn.Module):
    def __init__(self, char_vocab_size=VOCAB_SIZE, bigram_vocab_size=BIGRAM_VOCAB_SIZE,
                  d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                  ffn_dim=FFN_DIM, fc_hidden=FC_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.char_branch = TransformerBranch(
            char_vocab_size, d_model, n_heads, n_layers, ffn_dim, dropout,
            pad_id=PAD_ID, max_len=MAX_DOMAIN_LEN,
        )
        self.bigram_branch = TransformerBranch(
            bigram_vocab_size, d_model, n_heads, n_layers, ffn_dim, dropout,
            pad_id=BIGRAM_PAD_ID, max_len=MAX_BIGRAM_LEN,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x_char, x_bigram):
        h_char = self.char_branch(x_char)              # (B, D)
        h_bi   = self.bigram_branch(x_bigram)           # (B, D)
        fused = torch.cat([h_char, h_bi], dim=1)        # (B, 2D)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# TRAIN + PREDICT
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, y_val):
    # AdamW: weight decay decoupled, chuẩn cho Transformer (Loshchilov & Hutter, 2019)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history, best_f1, best_state, best_epoch, no_imp = [], -1.0, None, 0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        rl, n = 0.0, 0
        for xc, xb, yb in train_loader:
            xc = xc.to(DEVICE, non_blocking=True)
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xc, xb), yb); loss.backward(); optimizer.step()
            rl += loss.item() * yb.size(0); n += yb.size(0)
        tl = rl / n
        model.eval(); preds = []
        with torch.no_grad():
            for xc, xb, _ in val_loader:
                xc = xc.to(DEVICE, non_blocking=True)
                xb = xb.to(DEVICE, non_blocking=True)
                preds.append(model(xc, xb).argmax(dim=1).cpu().numpy())
        vp = np.concatenate(preds)
        vf1 = f1_score(y_val, vp, zero_division=0); vacc = accuracy_score(y_val, vp)
        history.append({"epoch": epoch, "train_loss": tl, "val_f1": vf1, "val_acc": vacc})
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}  loss={tl:.4f}  "
              f"val_acc={vacc*100:.2f}%  val_F1={vf1*100:.2f}%" + ("  *" if vf1 > best_f1 else ""))
        if vf1 > best_f1 + 1e-5:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_PATIENCE:
                print(f"  → Early stop epoch {epoch}"); break
    return history, best_state, best_epoch, best_f1


def predict(model, Xc_tensor, Xb_tensor, batch_size=256):
    model.eval()
    preds, probas = [], []
    n = Xc_tensor.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xc = Xc_tensor[i:i+batch_size].to(DEVICE)
            xb = Xb_tensor[i:i+batch_size].to(DEVICE)
            logits = model(xc, xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probas.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probas)


def measure_inference_energy(model, Xc_gpu, Xb_gpu, n_samples, n_repeats=3):
    rs = []
    for _ in range(n_repeats):
        with EnergyMeter("inference") as m:
            predict(model, Xc_gpu, Xb_gpu, batch_size=256)
            if USE_CUDA: torch.cuda.synchronize()
        rs.append({"wall": m.wall_seconds, "cpu": m.cpu_seconds,
                   "energy": m.energy_joules, "power": m.power_watts,
                   "util": m.utilization, "method": m.method})
    avg = lambda k: float(np.mean([r[k] for r in rs]))
    return {
        "n_repeats": n_repeats, "n_samples": n_samples,
        "avg_wall_seconds": avg("wall"), "avg_cpu_seconds": avg("cpu"),
        "avg_energy_joules": avg("energy"), "avg_power_watts": avg("power"),
        "avg_cpu_utilization": avg("util"),
        "energy_per_sample_mJ": avg("energy") / n_samples * 1000.0,
        "wall_per_sample_ms":   avg("wall") / n_samples * 1000.0,
        "method": rs[0]["method"],
    }


# ---------------------------------------------------------------------------
# ĐÁNH GIÁ
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba, label="Test"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try: auc = roc_auc_score(y_true, y_proba)
    except ValueError: auc = float("nan")
    print(f"\n=== {label} ===")
    print(f"Accuracy : {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%  |  AUC-ROC : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["benign","DGA"], digits=4))
    cm = confusion_matrix(y_true, y_pred); print("Confusion matrix:"); print(cm)
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
# MAIN
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    if USE_CUDA: torch.cuda.manual_seed_all(SEED)

    print(f"[CPU] {CPU_MODEL}  |  [GPU] CUDA={USE_CUDA}  |  Device={DEVICE}")
    print(f"\n[1/7] Đọc 3 file CSV từ {DATA_DIR}/ ...")
    print("[2/7] Chia tập ...")
    X_train, X_val, X_test, y_train, y_val, y_test, data_meta = load_data_split()
    n_total = data_meta["n_total"]
    n_pos   = data_meta["n_train_dga"]    + data_meta["n_val_dga"]    + data_meta["n_test_dga"]
    n_neg   = data_meta["n_train_benign"] + data_meta["n_val_benign"] + data_meta["n_test_benign"]
    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print("[3/7] Encode ký tự + bigram ...")
    t0 = time.time()
    Xtr_c = domains_to_tensor(X_train)
    Xva_c = domains_to_tensor(X_val)
    Xte_c = domains_to_tensor(X_test)
    Xtr_b = domains_to_bigram_tensor(X_train)
    Xva_b = domains_to_bigram_tensor(X_val)
    Xte_b = domains_to_bigram_tensor(X_test)
    feat_time = time.time() - t0
    print(f"      Xtr char={tuple(Xtr_c.shape)}  bigram={tuple(Xtr_b.shape)}  "
          f"(char_vocab={VOCAB_SIZE}, bigram_vocab={BIGRAM_VOCAB_SIZE})")
    print(f"      Thời gian: {feat_time:.2f}s")
    print(f"[4/7] Build Dual-Embedding Transformer "
          f"(d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, ffn={FFN_DIM}) ...")
    model = DualEmbeddingTransformer().to(DEVICE)
    n_params = count_parameters(model)
    print(f"      Số tham số: {n_params:,}")

    train_ds = TensorDataset(Xtr_c, Xtr_b, torch.from_numpy(y_train.astype(np.int64)))
    val_ds   = TensorDataset(Xva_c, Xva_b, torch.from_numpy(y_val.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=USE_CUDA)

    print("[5/7] Huấn luyện — đo năng lượng ...")
    with EnergyMeter("training") as em:
        history, best_state, best_epoch, best_f1_val = train_model(model, train_loader, val_loader, y_val)
    print(f"      Best epoch={best_epoch}  val F1={best_f1_val*100:.2f}%")
    print(f"      Wall={em.wall_seconds:.2f}s  CPU={em.cpu_seconds:.2f}s  "
          f"Util={em.utilization*100:.1f}%  Power={em.power_watts:.1f}W  "
          f"Energy={em.energy_joules:.2f}J  ({em.method})")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_proba = predict(model, Xva_c, Xva_b)
    val_metrics = evaluate(y_val, val_pred, val_proba, label="Validation")

    print("\n[6/7] Đánh giá test + đo năng lượng suy diễn ...")
    Xte_c_gpu = Xte_c.to(DEVICE)
    Xte_b_gpu = Xte_b.to(DEVICE)
    energy_infer = measure_inference_energy(model, Xte_c_gpu, Xte_b_gpu, len(X_test), n_repeats=3)
    print(f"      Wall={energy_infer['wall_per_sample_ms']:.4f} ms/mẫu  "
          f"Energy={energy_infer['energy_per_sample_mJ']:.4f} mJ/mẫu  "
          f"Power={energy_infer['avg_power_watts']:.1f}W")

    test_pred, test_proba = predict(model, Xte_c, Xte_b)
    test_metrics = evaluate(y_test, test_pred, test_proba, label="Test")

    print("\n[7/7] Per-family + lưu kết quả ...")
    dga_types = classify_many(X_test)
    dga_display = np.where(y_test == 1, dga_types, "benign")
    c = Counter(dga_display)
    for k in ["DGA-R", "DGA-P", "DGA-W", "benign"]:
        print(f"      {k:<8} : {c[k]:>8,}  ({c[k]/len(X_test)*100:5.2f}%)")
    per_family = evaluate_per_family(y_test, test_pred, dga_types)
    for lbl in ["DGA-R", "DGA-P", "DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            print(f"      {lbl}: F1={r['f1']*100:.2f}%  Prec={r['precision']*100:.2f}%  "
                  f"Recall={r['recall']*100:.2f}%  (n={r['n_dga']:,})")

    # Lưu
    model_path = OUTPUT_DIR / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "char_vocab_size": VOCAB_SIZE, "bigram_vocab_size": BIGRAM_VOCAB_SIZE,
        "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "ffn_dim": FFN_DIM,
        "fc_hidden": FC_HIDDEN, "dropout": DROPOUT,
        "max_domain_len": MAX_DOMAIN_LEN, "max_bigram_len": MAX_BIGRAM_LEN,
        "best_epoch": best_epoch, "best_val_f1": best_f1_val, "training_history": history,
    }, model_path)
    model_size_mb = get_size_mb(model_path)
    print(f"      Đã lưu model → {model_path} ({model_size_mb:.2f} MB)")

    metrics = {
        "validation": val_metrics, "test": test_metrics,
        "per_family": per_family, "dga_type_distribution": dict(c),
        "model_size": {"compressed_MB": model_size_mb, "num_parameters": n_params},
        "training_cost": {
            "wall_seconds": em.wall_seconds, "cpu_seconds": em.cpu_seconds,
            "cpu_utilization": em.utilization, "avg_power_watts": em.power_watts,
            "energy_joules": em.energy_joules, "energy_Wh": em.energy_joules/3600.0,
            "energy_method": em.method, "best_epoch": best_epoch, "best_val_f1": best_f1_val,
        },
        "inference_cost": energy_infer,
        "feature_extraction_seconds": feat_time,
        "training_history": history,
        "data": {"data_dir": DATA_DIR, "splits": {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}, "total_samples": n_total,
                 "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                 "n_positive_total": n_pos, "n_negative_total": n_neg},
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                    "ffn_dim": FFN_DIM, "fc_hidden": FC_HIDDEN, "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
                    "weight_decay": WEIGHT_DECAY, "max_domain_len": MAX_DOMAIN_LEN,
                    "max_bigram_len": MAX_BIGRAM_LEN, "seed": SEED, "device": str(DEVICE)},
        "heuristic_thresholds": {
            "DGA-W": "length >= 12 AND vowel_ratio >= 0.30 AND entropy <= 3.6",
            "DGA-R": "entropy >= 3.8 OR digit_ratio >= 0.15",
            "DGA-P": "0.20 <= vowel_ratio <= 0.45 AND entropy < 3.8",
        },
        "cpu_profile": {"model": CPU_MODEL, "base_power_W": CPU_BASE_POWER_W,
                         "max_turbo_power_W": CPU_MAX_TURBO_POWER_W,
                         "avg_multithread_W": CPU_AVG_MULTITHREAD_W,
                         "avg_light_load_W": CPU_AVG_LIGHT_LOAD_W,
                         "logical_cores": CPU_LOGICAL_CORES},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                         "processor": platform.processor(), "cpu_count": os.cpu_count(),
                         "torch": torch.__version__, "cuda_available": USE_CUDA},
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    cm = np.array(test_metrics["confusion_matrix"])
    pd.DataFrame(cm, index=["true_benign","true_DGA"],
                 columns=["pred_benign","pred_DGA"]).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    pd.DataFrame({"domain": X_test, "true_label": y_test, "pred_label": test_pred,
                   "prob_DGA": test_proba}).to_csv(OUTPUT_DIR / "test_predictions.csv",
                                                    index=False, sep=CSV_SEP)

    with open(OUTPUT_DIR / "per_family_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "M5 — Dual-Embedding Transformer",
            "dga_type_distribution": dict(c),
            "overall": {"accuracy": test_metrics["accuracy"], "f1": test_metrics["f1"]},
            "per_family": per_family,
            "heuristic_thresholds": metrics["heuristic_thresholds"],
        }, f, indent=2, ensure_ascii=False, default=float)

    pf_lines = ["====== M5 — F1 theo loại DGA ======",
                 f"Test samples : {len(X_test):,}", f"Device       : {DEVICE}", "",
                 "Phân bố loại DGA (heuristic):"]
    for k in ["DGA-R","DGA-P","DGA-W","benign"]:
        pf_lines.append(f"  {k:<8} : {c[k]:>8,}  ({c[k]/len(X_test)*100:5.2f}%)")
    pf_lines += ["", f"Toàn test: Accuracy = {test_metrics['accuracy']*100:.2f}%   "
                     f"F1 = {test_metrics['f1']*100:.2f}%", "",
                  f"{'Loại':<8} {'n_dga':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"]
    for lbl in ["DGA-R","DGA-P","DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            pf_lines.append(f"{lbl:<8} {r['n_dga']:>8,} "
                            f"{r['accuracy']*100:>7.2f} {r['precision']*100:>7.2f} "
                            f"{r['recall']*100:>7.2f} {r['f1']*100:>7.2f}")
    (OUTPUT_DIR / "per_family_summary.txt").write_text("\n".join(pf_lines), encoding="utf-8")

    lines = [
        "================ M5 — Dual-Embedding Transformer (Char + Bigram) ================",
        f"Data dir: {DATA_DIR}/ (train.csv, val.csv, test.csv)",
        f"Tổng mẫu           : {n_total:,}  (DGA={n_pos:,}, benign={n_neg:,})",
        f"Chia                : Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}",
        "",
        "[Kiến trúc]",
        f"  Char vocab       : {VOCAB_SIZE}  |  Bigram vocab: {BIGRAM_VOCAB_SIZE}",
        f"  d_model          : {D_MODEL}",
        f"  Số heads         : {N_HEADS}",
        f"  Số layers        : {N_LAYERS}  (mỗi luồng)",
        f"  FFN dim          : {FFN_DIM}",
        f"  FC hidden        : {FC_HIDDEN}",
        f"  Tham số          : {n_params:,}",
        f"  Device           : {DEVICE}",
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
        f"  Best epoch       : {best_epoch}  (val F1={best_f1_val*100:.2f}%)",
        "",
        "[Chi phí suy diễn / mẫu]",
        f"  Wall             : {energy_infer['wall_per_sample_ms']:.4f} ms/mẫu",
        f"  Energy           : {energy_infer['energy_per_sample_mJ']:.4f} mJ/mẫu",
        f"  Avg power        : {energy_infer['avg_power_watts']:.1f} W",
        f"  Đo bằng          : {energy_infer['method']}",
        "",
        "[Kích thước mô hình]",
        f"  File tổng        : {model_size_mb:.2f} MB  ({n_params:,} tham số)",
        "",
        "[Bảng 5 — F1 theo loại DGA]",
        f"  {'Loại':<8} {'n_dga':>8} {'F1 (%)':>8} {'Prec':>8} {'Recall':>8}",
    ]
    for lbl in ["DGA-R","DGA-P","DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            lines.append(f"  {lbl:<8} {r['n_dga']:>8,} "
                         f"{r['f1']*100:>7.2f} {r['precision']*100:>7.2f} "
                         f"{r['recall']*100:>7.2f}")
    lines += ["", "[Môi trường]",
              f"  Python           : {sys.version.split()[0]}",
              f"  PyTorch          : {torch.__version__}",
              f"  CUDA available   : {USE_CUDA}"]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print("\n" + "="*60 + "\nTÓM TẮT M5\n" + "="*60)
    print(summary)

    def v(lbl):
        return f"{per_family[lbl]['f1']*100:.1f}" if per_family[lbl].get("n_dga",0)>0 else "—"
    print(f"\nDòng Bảng 5 (copy-paste):\n"
          f"  M5 — Dual-Embedding Transformer  |  {v('DGA-R')}  |  {v('DGA-P')}  |  {v('DGA-W')}\n")


if __name__ == "__main__":
    main()
