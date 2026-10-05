"""
M4 — ATT-CNN-BiLSTM: CNN + BiLSTM + Attention (Ren et al., 2020)
Self-contained.

Kiến trúc: Character Embedding → Conv1D → BiLSTM → Attention layer
           → Dense + Dropout → Softmax.

Input  : DataNew/{train.csv, val.csv, test.csv}  (đã chia sẵn để tránh leakage)
Output : out_M4/
            - model.pt, metrics.json, confusion_matrix.csv, test_predictions.csv,
              per_family_results.json, per_family_summary.txt, summary.txt

Cài đặt:
    pip install numpy pandas scikit-learn torch
    pip install codecarbon  # tuỳ chọn

Chạy:
    python M4.py
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["USE_TF"]                     = "NO"
os.environ["USE_TORCH"]                  = "YES"
os.environ["TRANSFORMERS_VERBOSITY"]    = "error"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"

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
OUTPUT_DIR  = Path("out_M4")
SEED        = 42
# Encode ký tự
CHAR_VOCAB = ['<pad>'] + list("abcdefghijklmnopqrstuvwxyz0123456789-.")
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)
PAD_ID = 0
MAX_DOMAIN_LEN = 64

# Kiến trúc M4 — ATT-CNN-BiLSTM (Ren et al., 2020)
EMBED_DIM     = 128
CONV_KERNEL   = 3
CONV_CHANNELS = 128
LSTM_HIDDEN   = 128    # 2 chiều → 256 khi concat
ATTN_DIM      = 128    # Chiều projection trong attention
FC_HIDDEN     = 256
DROPOUT       = 0.3

BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
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
    d = domain.strip().lower()
    ids = [CHAR_TO_ID.get(c, PAD_ID) for c in d]
    if len(ids) > max_len: ids = ids[:max_len]
    else: ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def domains_to_tensor(domains, max_len=MAX_DOMAIN_LEN):
    arr = np.array([encode_domain(d, max_len) for d in domains], dtype=np.int64)
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
# KIẾN TRÚC M4: ATT-CNN-BiLSTM
# Character Embedding → Conv1D (k=3) → BiLSTM → Attention → Dense → Softmax
# Attention theo Ren et al. 2020:
#   u_i = tanh(W h_i + b)
#   α_i = softmax(u_i^T u_w)     (u_w là context vector học được)
#   c   = Σ α_i * h_i
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Attention pooling theo kiểu Yang et al. (hierarchical attention)."""
    def __init__(self, hidden_dim, attn_dim=ATTN_DIM):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.u_w = nn.Parameter(torch.randn(attn_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.u_w.unsqueeze(0))

    def forward(self, H, mask=None):
        # H: (B, L, D)
        u = torch.tanh(self.W(H))                  # (B, L, attn_dim)
        scores = torch.matmul(u, self.u_w)          # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=1)             # (B, L)
        c = torch.sum(alpha.unsqueeze(-1) * H, dim=1)  # (B, D)
        return c, alpha


class ATT_CNN_BiLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                  conv_kernel=CONV_KERNEL, conv_channels=CONV_CHANNELS,
                  lstm_hidden=LSTM_HIDDEN, attn_dim=ATTN_DIM,
                  fc_hidden=FC_HIDDEN, dropout=DROPOUT, pad_id=PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        self.conv = nn.Conv1d(embed_dim, conv_channels,
                               kernel_size=conv_kernel, padding=conv_kernel // 2)

        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden,
                             num_layers=1, batch_first=True, bidirectional=True)
        lstm_out_dim = lstm_hidden * 2

        self.attention = AttentionPooling(hidden_dim=lstm_out_dim, attn_dim=attn_dim)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x):
        mask = (x != self.pad_id)                     # (B, L)
        emb = self.embedding(x)                        # (B, L, E)
        h = emb.transpose(1, 2)                         # (B, E, L)
        h = F.relu(self.conv(h))                        # (B, C, L)
        h = h.transpose(1, 2)                           # (B, L, C)
        lstm_out, _ = self.lstm(h)                      # (B, L, 2H)
        context, _ = self.attention(lstm_out, mask=mask)
        return self.fc(context)


# ---------------------------------------------------------------------------
# TRAIN + PREDICT
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, y_val):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history, best_f1, best_state, best_epoch, no_imp = [], -1.0, None, 0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        rl, n = 0.0, 0
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


def predict(model, X_tensor, batch_size=512):
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


def measure_inference_energy(model, X_tensor_gpu, n_samples, n_repeats=3):
    rs = []
    for _ in range(n_repeats):
        with EnergyMeter("inference") as m:
            predict(model, X_tensor_gpu, batch_size=1024)
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
    print("[3/7] Encode ký tự ...")
    t0 = time.time()
    Xtr = domains_to_tensor(X_train); Xva = domains_to_tensor(X_val); Xte = domains_to_tensor(X_test)
    feat_time = time.time() - t0
    print(f"      Xtr shape: {tuple(Xtr.shape)}, time={feat_time:.2f}s")
    print(f"[4/7] Build ATT-CNN-BiLSTM (embed={EMBED_DIM}, conv_k={CONV_KERNEL}, "
          f"channels={CONV_CHANNELS}, LSTM={LSTM_HIDDEN}, attn={ATTN_DIM}) ...")
    model = ATT_CNN_BiLSTM().to(DEVICE)
    n_params = count_parameters(model)
    print(f"      Số tham số: {n_params:,}")

    train_ds = TensorDataset(Xtr, torch.from_numpy(y_train.astype(np.int64)))
    val_ds   = TensorDataset(Xva, torch.from_numpy(y_val.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*4, shuffle=False,
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

    val_pred, val_proba = predict(model, Xva)
    val_metrics = evaluate(y_val, val_pred, val_proba, label="Validation")

    print("\n[6/7] Đánh giá test + đo năng lượng suy diễn ...")
    Xte_gpu = Xte.to(DEVICE)
    energy_infer = measure_inference_energy(model, Xte_gpu, len(X_test), n_repeats=3)
    print(f"      Wall={energy_infer['wall_per_sample_ms']:.4f} ms/mẫu  "
          f"Energy={energy_infer['energy_per_sample_mJ']:.4f} mJ/mẫu  "
          f"Power={energy_infer['avg_power_watts']:.1f}W")

    test_pred, test_proba = predict(model, Xte)
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
        "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM,
        "conv_kernel": CONV_KERNEL, "conv_channels": CONV_CHANNELS,
        "lstm_hidden": LSTM_HIDDEN, "attn_dim": ATTN_DIM,
        "fc_hidden": FC_HIDDEN, "dropout": DROPOUT, "max_domain_len": MAX_DOMAIN_LEN,
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
        "config": {"embed_dim": EMBED_DIM, "conv_kernel": CONV_KERNEL,
                    "conv_channels": CONV_CHANNELS, "lstm_hidden": LSTM_HIDDEN,
                    "attn_dim": ATTN_DIM, "fc_hidden": FC_HIDDEN, "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
                    "weight_decay": WEIGHT_DECAY, "max_domain_len": MAX_DOMAIN_LEN,
                    "seed": SEED, "device": str(DEVICE)},
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
            "model": "M4 — ATT-CNN-BiLSTM",
            "dga_type_distribution": dict(c),
            "overall": {"accuracy": test_metrics["accuracy"], "f1": test_metrics["f1"]},
            "per_family": per_family,
            "heuristic_thresholds": metrics["heuristic_thresholds"],
        }, f, indent=2, ensure_ascii=False, default=float)

    pf_lines = ["====== M4 — F1 theo loại DGA ======",
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
        "================ M4 — ATT-CNN-BiLSTM ================",
        f"Data dir: {DATA_DIR}/ (train.csv, val.csv, test.csv)",
        f"Tổng mẫu           : {n_total:,}  (DGA={n_pos:,}, benign={n_neg:,})",
        f"Chia                : Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}",
        "",
        "[Kiến trúc]",
        f"  Char embedding   : {EMBED_DIM}-d",
        f"  Conv1D           : kernel={CONV_KERNEL}, channels={CONV_CHANNELS}",
        f"  BiLSTM           : {LSTM_HIDDEN} đơn vị (2 chiều → {LSTM_HIDDEN*2})",
        f"  Attention        : hidden_dim={LSTM_HIDDEN*2}, attn_dim={ATTN_DIM}",
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

    print("\n" + "="*60 + "\nTÓM TẮT M4\n" + "="*60)
    print(summary)

    def v(lbl):
        return f"{per_family[lbl]['f1']*100:.1f}" if per_family[lbl].get("n_dga",0)>0 else "—"
    print(f"\nDòng Bảng 5 (copy-paste):\n"
          f"  M4 — ATT-CNN-BiLSTM  |  {v('DGA-R')}  |  {v('DGA-P')}  |  {v('DGA-W')}\n")


if __name__ == "__main__":
    main()
