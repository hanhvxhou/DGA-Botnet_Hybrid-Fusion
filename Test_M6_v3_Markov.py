"""
Test_M6_v3_Markov.py
====================
Test M6-v3 (phiên bản Markov, 20 handcraft features) trên tập DGA bên ngoài
(UMUDGA, Netlab360).

Cấu trúc thư mục input:
    test_external/
    ├── UMUDGA/
    │   ├── conficker.txt        (mỗi dòng 1 domain, không header)
    │   ├── kraken.txt
    │   └── ...
    └── Netlab360/
        ├── banjori.txt
        └── ...

Output:
    test_external/summary_Markov.txt    — báo cáo Detection Rate
    test_external/per_family_Markov.csv — số liệu chi tiết

Detection Rate (DR):
    DR = N_DGA / N_TEST
    với N_TEST là số mẫu lấy ra từ mỗi file (bốc ngẫu nhiên),
    N_DGA là số mẫu được M6-v3 dự đoán là DGA.
    Vì toàn bộ file đều là DGA, DR = Recall = True Positive Rate.

Mô hình M6-v3 (Late Fusion, 20 handcraft, Markov variant):
    P = w_string  · P_string    (BiLSTM emb → MLP head)
      + w_semantic · P_semantic (DistilBERT [CLS] → fine-tuned classifier)
      + w_handcraft· P_handcraft(20 features → XGBoost)
    pred = 1 nếu P >= threshold

20 handcraft features (Markov variant):
    - 17 base statistical (entropy, ratio, longest_run, TLD, ...)
    - 1  meaning_ratio (Soleymani & Arabgol 2021)
    - 2  Markov perplexity:
          • markov_bi_perplexity:  bigram char-level + Laplace smoothing
          • markov_tri_perplexity: trigram char-level + Laplace smoothing

Yêu cầu file đã có sẵn:
    out_select/bilstm/model.pt
    out_select/distilbert/{model.pt, tokenizer/}
    out_M6_v3_Markov/models/{string_head.pt, handcraft_xgb.joblib, scaler.joblib}
    out_M6_v3_Markov/metrics.json       (chứa weights + threshold)
    out_M6_v3_Markov/benign_markov.npz  (cache Markov bigram/trigram models)

Cách chạy:
    python Test_M6_v3_Markov.py

Để thay đổi tham số: sửa trực tiếp các biến TEST_ROOT, N_TEST, SEED, OUTPUT_TXT
ở phần CẤU HÌNH bên dưới.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# CẤU HÌNH — CHỈNH TRỰC TIẾP TẠI ĐÂY
# ---------------------------------------------------------------------------
# Thư mục chứa các sub-folder dataset (UMUDGA, Netlab360, ...)
TEST_ROOT  = "test_external"

# Số mẫu test mỗi file (lấy ngẫu nhiên; nếu file < N_TEST thì dùng toàn bộ)
N_TEST     = 2000

# Random seed cho việc sampling — cố định để có thể tái hiện
SEED       = 42

# File output (None = dùng <TEST_ROOT>/summary_<VARIANT>.txt)
OUTPUT_TXT = None

# ---------------------------------------------------------------------------
# CHỌN BIẾN THỂ MÔ HÌNH ĐỂ TEST
# ---------------------------------------------------------------------------
#   "markov"        -> out_M6_v3_Markov/        (bản gốc, w_min=0, w3 có thể =0)
#   "markov_w2min"  -> out_M6_v3_Markov_w2min/  (bản ràng buộc w_min, w3>0)
# Đổi biến này rồi chạy lại để so sánh 2 phương án trên cùng tập external.
VARIANT = "markov_w2min"

_VARIANT_DIRS = {
    "markov":       Path("out_M6_v3_Markov"),
    "markov_w2min": Path("out_M6_v3_Markov_w2min"),
}
if VARIANT not in _VARIANT_DIRS:
    raise ValueError(f"VARIANT='{VARIANT}' không hợp lệ. "
                     f"Chọn: {list(_VARIANT_DIRS)}")

# Tối ưu: nếu một nhánh có weight = 0 thì BỎ QUA hoàn toàn việc tính nhánh đó
# (kết quả KHÔNG đổi vì nhánh w=0 không vào công thức fusion, nhưng tiết
#  kiệm thời gian — đặc biệt nhánh handcraft với 2 đặc trưng Markov tốn kém).
# Đặt False nếu muốn vẫn tính đủ 3 nhánh để đo chi phí inference đầy đủ.
SKIP_ZERO_WEIGHT_BRANCH = True

# Đường dẫn các thành phần M6-v3 đã lưu (theo VARIANT)
M6V3_DIR        = _VARIANT_DIRS[VARIANT]
M6V3_MODELS_DIR = M6V3_DIR / "models"
M6V3_METRICS    = M6V3_DIR / "metrics.json"

SELECT_ROOT     = Path("out_select")
BILSTM_DIR      = SELECT_ROOT / "bilstm"
DISTILBERT_DIR  = SELECT_ROOT / "distilbert"

# Inference batch sizes (giống M6_v3.py)
EMBED_BATCH  = 256
BERT_MAX_LEN = 64
MAX_DOM_LEN  = 64

# Char vocab cho BiLSTM (giống selectAlgorithm/M6_v3)
CHAR_VOCAB  = ['<pad>'] + list("abcdefghijklmnopqrstuvwxyz0123456789-.")
CHAR_TO_ID  = {c: i for i, c in enumerate(CHAR_VOCAB)}
PAD_ID      = 0

# String head MLP (giống M6_v3)
STRING_HEAD_HIDDEN  = 128
STRING_HEAD_DROPOUT = 0.3

# Handcraft constants (giống M6_v3)
VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnopqrstvwxyz")
DIGITS     = set("0123456789")
HEX_CHARS  = set("0123456789abcdef")
COMMON_TLDS = {
    "com","net","org","info","biz","co","io","me","us","uk","vn","cn","ru","de",
    "fr","jp","in","br","tv","cc","name","online","site","shop","app","dev",
    "club","xyz","top","live","store","tech","art","mobi","asia","edu","gov",
    "mil","int","pro",
}

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")


# ===========================================================================
# 1. CLASS / FUNCTION ĐỊNH NGHĨA LẠI (giống M6_v3.py để load checkpoint)
# ===========================================================================
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, num_layers,
                 fc_hidden, dropout, pad_id=PAD_ID):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden,
                             num_layers=num_layers, batch_first=True,
                             bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x, return_embedding=False):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        features = lstm_out.max(dim=1)[0]
        if return_embedding:
            return features
        return self.fc(features)


class CharDataset(Dataset):
    def __init__(self, domains, max_len=MAX_DOM_LEN):
        self.domains = domains; self.max_len = max_len

    def __len__(self): return len(self.domains)

    def __getitem__(self, idx):
        d = self.domains[idx].strip().lower()
        ids = [CHAR_TO_ID.get(c, PAD_ID) for c in d]
        if len(ids) > self.max_len: ids = ids[:self.max_len]
        else: ids = ids + [PAD_ID] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class BertForDGA(nn.Module):
    def __init__(self, bert_model, hidden_size, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state[:, 0, :]
        if return_embedding:
            return hidden
        return self.classifier(self.dropout(hidden))


class BertDomainDataset(Dataset):
    def __init__(self, domains, tokenizer, max_len=BERT_MAX_LEN):
        self.domains = domains; self.tok = tokenizer; self.max_len = max_len

    def __len__(self): return len(self.domains)

    def __getitem__(self, idx):
        enc = self.tok(self.domains[idx], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0))


class StringHead(nn.Module):
    def __init__(self, in_dim, hidden=STRING_HEAD_HIDDEN, dropout=STRING_HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# 2. HANDCRAFT FEATURES (20-d: 17 base + meaning_ratio + markov_bi + markov_tri)
#    Phải KHỚP HỆT pipeline trong M6_v3_Markov.py để load đúng XGBoost + scaler.
# ===========================================================================

# [2A] Dictionary tiếng Anh cho meaning_ratio
ENGLISH_WORDS_SET = None


def _load_english_words():
    """Load từ điển tiếng Anh cho meaning_ratio (giống M6_v3_Markov.py)."""
    global ENGLISH_WORDS_SET
    if ENGLISH_WORDS_SET is not None: return
    try:
        import nltk
        from nltk.corpus import words as nltk_words
        try:
            ENGLISH_WORDS_SET = set(w.lower() for w in nltk_words.words()
                                     if w.isalpha() and 3 <= len(w) <= 12)
        except LookupError:
            nltk.download('words', quiet=True)
            ENGLISH_WORDS_SET = set(w.lower() for w in nltk_words.words()
                                     if w.isalpha() and 3 <= len(w) <= 12)
        print(f"    [meaning_ratio] Loaded {len(ENGLISH_WORDS_SET):,} English words from nltk")
    except Exception as e:
        print(f"    [meaning_ratio] nltk unavailable ({e}), dùng fallback ~500 từ")
        ENGLISH_WORDS_SET = set("""
        the and that have for not with you this but his from they she her him
        one all would there their what about which when make like time just
        know take person into year your good some could them see other than
        then now look only come its over think also back after use two how
        our work first well way even new want because any these give day most
        service love life home page contact about news blog site web online
        shop store buy sell price deal sale free quick best top easy fast safe
        secure trust bank pay card money cash gold silver green blue red black
        white light dark sun moon star sky cloud water fire earth wind rain
        food drink eat cook kitchen bread milk apple orange banana cherry
        cloud bridge tower garden forest mountain river ocean beach desert
        happy angry tired sleepy busy ready bright dark smart wise kind friend
        family mother father brother sister parent child baby young old
        system network server data file user admin password login signup register
        book paper pen desk chair table floor wall door window house room
        school college study learn teach class room course science english math
        music song play game sport team win lose score point goal player
        car road drive train plane ship boat travel trip tour ride walk run
        movie film show story actor star stage scene cast voice light news
        health doctor nurse hospital patient medicine cure treat surgery dental
        tech digital mobile phone call text message mail email reply send
        google amazon facebook apple microsoft netflix youtube twitter github
        """.split())


# [2B] Markov Chain models (load từ cache benign_markov.npz)
_MARKOV_BI_LOG_PROB  = None
_MARKOV_TRI_LOG_PROB = None
_MARKOV_BI_DEFAULT  = None
_MARKOV_TRI_DEFAULT = None
_MARKOV_VOCAB_SIZE  = 40


def load_markov_cache(cache_path: Path):
    """Load Markov bigram/trigram models đã được M6_v3_Markov.py tạo sẵn."""
    global _MARKOV_BI_LOG_PROB, _MARKOV_TRI_LOG_PROB
    global _MARKOV_BI_DEFAULT, _MARKOV_TRI_DEFAULT
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {cache_path}. Phải chạy M6_v3_Markov.py trước "
            f"để tạo cache này (cũng đảm bảo train pipeline khớp).")
    data = np.load(cache_path, allow_pickle=True)
    _MARKOV_BI_LOG_PROB  = data["bi_log_prob"].item()
    _MARKOV_TRI_LOG_PROB = data["tri_log_prob"].item()
    _MARKOV_BI_DEFAULT   = float(data["bi_default"])
    _MARKOV_TRI_DEFAULT  = float(data["tri_default"])
    print(f"  ✓ Markov models: |bigram|={len(_MARKOV_BI_LOG_PROB):,}, "
          f"|trigram|={len(_MARKOV_TRI_LOG_PROB):,}")


# [2C] Helper functions
def shannon_entropy(s):
    if not s: return 0.0
    counts = Counter(s); n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())


def longest_run(s, charset):
    best = cur = 0
    for ch in s:
        if ch in charset: cur += 1; best = max(best, cur)
        else: cur = 0
    return best


def split_domain(domain):
    d = re.sub(r"^www\.", "", domain.strip().lower())
    parts = d.split(".")
    if len(parts) == 1: return parts[0], ""
    return ".".join(parts[:-1]), parts[-1]


def meaning_ratio(text):
    """ut_meaning_ratio (Soleymani & Arabgol 2021): % ký tự thuộc từ tiếng Anh."""
    if not text or ENGLISH_WORDS_SET is None: return 0.0
    n = len(text); i = 0; covered = 0
    while i < n:
        matched = False
        for length in range(min(12, n - i), 2, -1):
            if text[i:i+length] in ENGLISH_WORDS_SET:
                covered += length; i += length; matched = True
                break
        if not matched: i += 1
    return covered / max(n, 1)


def markov_bi_perplexity(text):
    """Perplexity bigram char-level theo Markov model + Laplace smoothing."""
    if _MARKOV_BI_LOG_PROB is None or len(text) < 2:
        return 0.0
    default = _MARKOV_BI_DEFAULT
    log_sum = 0.0; n_eval = 0
    for i in range(len(text) - 1):
        bi = text[i:i+2]
        log_sum += _MARKOV_BI_LOG_PROB.get(bi, default)
        n_eval += 1
    if n_eval == 0: return 0.0
    avg_neg_log = -log_sum / n_eval
    if avg_neg_log > 50: avg_neg_log = 50.0
    return math.exp(avg_neg_log)


def markov_tri_perplexity(text):
    """Perplexity trigram char-level theo Markov model + Laplace smoothing."""
    if _MARKOV_TRI_LOG_PROB is None or len(text) < 3:
        return 0.0
    default = _MARKOV_TRI_DEFAULT
    log_sum = 0.0; n_eval = 0
    for i in range(len(text) - 2):
        tri = text[i:i+3]
        log_sum += _MARKOV_TRI_LOG_PROB.get(tri, default)
        n_eval += 1
    if n_eval == 0: return 0.0
    avg_neg_log = -log_sum / n_eval
    if avg_neg_log > 50: avg_neg_log = 50.0
    return math.exp(avg_neg_log)


def hand_crafted_features(domain: str) -> np.ndarray:
    """20 features = 17 base + meaning_ratio + markov_bi_perplexity + markov_tri_perplexity."""
    d = domain.strip().lower()
    main, tld = split_domain(d)
    text = main.replace(".", "")
    text_clean = text.replace("-", "")
    n = max(len(text), 1)
    nv = sum(1 for c in text if c in VOWELS)
    nc = sum(1 for c in text if c in CONSONANTS)
    nd = sum(1 for c in text if c in DIGITS)
    nh = sum(1 for c in text if c in HEX_CHARS)
    ns = sum(1 for c in text if c not in VOWELS and c not in CONSONANTS and c not in DIGITS)
    nu = len(set(text))
    return np.asarray([
        len(d), len(main), shannon_entropy(text),
        nv/n, nc/n, nd/n, nh/n, ns/n, nu/n,
        longest_run(text, VOWELS), longest_run(text, CONSONANTS),
        longest_run(text, DIGITS),
        d.count("."), len(tld),
        int(any(c in DIGITS for c in text)),
        int(tld in COMMON_TLDS),
        n - nu,
        meaning_ratio(text_clean),
        markov_bi_perplexity(text_clean),
        markov_tri_perplexity(text_clean),
    ], dtype=np.float32)


def extract_handcrafted_matrix(domains):
    return np.vstack([hand_crafted_features(d) for d in domains])


# ===========================================================================
# 3. LOAD ALL M6-v3 COMPONENTS
# ===========================================================================
def load_bilstm(bilstm_dir: Path):
    model_pt = bilstm_dir / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"Không tìm thấy {model_pt}")
    ckpt = torch.load(model_pt, map_location=DEVICE, weights_only=False)
    model = CharBiLSTM(
        vocab_size=ckpt["vocab_size"], embed_dim=ckpt["embed_dim"],
        hidden=ckpt["hidden"], num_layers=ckpt["num_layers"],
        fc_hidden=ckpt["fc_hidden"], dropout=ckpt["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    embed_dim = ckpt["hidden"] * 2
    print(f"  ✓ BiLSTM:     hidden={ckpt['hidden']}, embed_dim={embed_dim}, "
          f"val_F1={ckpt.get('best_val_f1',0)*100:.2f}%")
    return model, embed_dim


def load_string_head(path: Path, in_dim: int):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    head = StringHead(
        in_dim=ckpt.get("in_dim", in_dim),
        hidden=ckpt.get("hidden", STRING_HEAD_HIDDEN),
        dropout=ckpt.get("dropout", STRING_HEAD_DROPOUT),
    ).to(DEVICE)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    print(f"  ✓ String head: in_dim={ckpt.get('in_dim', in_dim)}, "
          f"val_F1={ckpt.get('best_val_f1', 0)*100:.2f}%")
    return head


def load_distilbert(distilbert_dir: Path):
    from transformers import AutoModel, AutoTokenizer
    model_pt = distilbert_dir / "model.pt"
    tok_dir  = distilbert_dir / "tokenizer"
    if not model_pt.exists():
        raise FileNotFoundError(f"Không tìm thấy {model_pt}")

    ckpt = torch.load(model_pt, map_location=DEVICE, weights_only=False)
    model_name  = ckpt["model_name"]
    hidden_size = ckpt["hidden_size"]
    bert_base = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    model = BertForDGA(bert_base, hidden_size).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  ✓ DistilBERT: {model_name}, hidden={hidden_size}, "
          f"val_F1={ckpt.get('best_val_f1', 0)*100:.2f}%")
    return model, tokenizer


def load_handcraft(models_dir: Path):
    xgb_path    = models_dir / "handcraft_xgb.joblib"
    scaler_path = models_dir / "scaler.joblib"
    if not xgb_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {xgb_path} hoặc {scaler_path}")
    xgb = joblib.load(xgb_path)
    scaler = joblib.load(scaler_path)
    print(f"  ✓ Handcraft:  XGBoost (best_iter={xgb.best_iteration}), "
          f"scaler dim={scaler.n_features_in_}")
    return xgb, scaler


def load_fusion_config(metrics_path: Path):
    """Load weights + threshold đã tìm trên val set từ metrics.json."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    fusion = metrics.get("fusion", {})
    weights   = fusion.get("weights")
    threshold = fusion.get("threshold")
    if weights is None or threshold is None:
        raise ValueError(f"Không tìm thấy weights/threshold trong {metrics_path}")
    print(f"  ✓ Fusion config:")
    print(f"      weights   = {weights}")
    print(f"      threshold = {threshold:.4f}")
    return weights, threshold


# ===========================================================================
# 4. INFERENCE — EXTRACT P FROM 3 BRANCHES
# ===========================================================================
def extract_bilstm_embeddings(model, domains, batch_size=EMBED_BATCH):
    ds = CharDataset(domains)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=USE_CUDA)
    embs = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            emb = model(xb, return_embedding=True)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def predict_string_head(head, emb, batch_size=2048):
    head.eval()
    n = len(emb)
    probs = []
    X = torch.from_numpy(emb).to(DEVICE)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            logits = head(xb)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs).astype(np.float32)


def extract_distilbert_probs(model, domains, tokenizer, batch_size=EMBED_BATCH):
    ds = BertDomainDataset(domains, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=USE_CUDA)
    probs = []
    model.eval()
    with torch.no_grad():
        for ids, mask in loader:
            ids = ids.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            logits = model(ids, mask, return_embedding=False)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0).astype(np.float32)


def predict_handcraft(xgb, scaler, domains):
    hc = extract_handcrafted_matrix(domains)
    hc_s = scaler.transform(hc)
    return xgb.predict_proba(hc_s)[:, 1].astype(np.float32)


def fuse(P_string, P_semantic, P_handcraft, weights):
    """Soft voting: P = sum(w_i * P_i) / sum(w_i)."""
    used = {k: v for k, v in weights.items() if v > 0}
    s = sum(used.values()) + 1e-12
    used = {k: v/s for k, v in used.items()}
    P = np.zeros_like(P_string)
    for nm, p_arr in [("string", P_string), ("semantic", P_semantic),
                       ("handcraft", P_handcraft)]:
        if nm in used:
            P = P + used[nm] * p_arr
    return P


# ===========================================================================
# 5. ĐỌC DOMAIN FILE
# ===========================================================================
def load_domain_file(path: Path) -> list[str]:
    """Đọc file .txt — mỗi dòng 1 domain, không header."""
    domains = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            d = line.strip().lower()
            # Bỏ khoảng trắng, comment
            if not d or d.startswith("#"): continue
            # Loại http://, https://, www.
            d = re.sub(r"^https?://", "", d)
            d = re.sub(r"^www\.", "", d)
            # Bỏ trailing slash hoặc path
            d = d.split("/")[0].split("?")[0].strip()
            if d and not d.startswith("."):
                domains.append(d)
    return domains


def sample_domains(domains: list[str], n_test: int, seed: int) -> tuple[list[str], int]:
    """Lấy n_test mẫu ngẫu nhiên (hoặc toàn bộ nếu file < n_test)."""
    n_total = len(domains)
    if n_total == 0: return [], 0
    if n_total <= n_test:
        return domains, n_total
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_total, size=n_test, replace=False)
    return [domains[i] for i in idx], n_test


# ===========================================================================
# 6. PIPELINE: TEST 1 FAMILY
# ===========================================================================
def test_family(domains, models_pack, weights, threshold):
    """Test M6-v3 trên một list domain. Trả về (n_test, n_dga_pred, dr, mean_prob)."""
    n_test = len(domains)
    if n_test == 0:
        return 0, 0, 0.0, 0.0
    bilstm_model, string_head, dbert_model, dbert_tok, xgb, scaler = models_pack

    # Trọng số mỗi nhánh (mặc định 0 nếu thiếu key)
    w_str = float(weights.get("string", 0) or 0)
    w_sem = float(weights.get("semantic", 0) or 0)
    w_hc  = float(weights.get("handcraft", 0) or 0)

    zeros = np.zeros(n_test, dtype=np.float32)

    # Branch 1 — String (BiLSTM emb → MLP head)
    if w_str > 0 or not SKIP_ZERO_WEIGHT_BRANCH:
        emb_bl = extract_bilstm_embeddings(bilstm_model, domains)
        P_str  = predict_string_head(string_head, emb_bl)
    else:
        P_str  = zeros

    # Branch 2 — Semantic (DistilBERT [CLS] → classifier)
    if w_sem > 0 or not SKIP_ZERO_WEIGHT_BRANCH:
        P_sem  = extract_distilbert_probs(dbert_model, domains, dbert_tok)
    else:
        P_sem  = zeros

    # Branch 3 — Handcraft (20 features → XGBoost). Bỏ qua nếu w_hc=0
    # (kết quả KHÔNG đổi vì fuse() loại nhánh w=0; chỉ tiết kiệm thời gian).
    if w_hc > 0 or not SKIP_ZERO_WEIGHT_BRANCH:
        P_hc   = predict_handcraft(xgb, scaler, domains)
    else:
        P_hc   = zeros

    # Fusion (fuse() tự loại nhánh w<=0 và chuẩn hóa phần còn lại)
    P = fuse(P_str, P_sem, P_hc, weights)
    pred = (P >= threshold).astype(int)
    n_dga = int(pred.sum())
    dr = n_dga / n_test
    return n_test, n_dga, dr, float(P.mean())


# ===========================================================================
# 7. MAIN
# ===========================================================================
def main():
    test_root = Path(TEST_ROOT)
    n_test    = int(N_TEST)
    seed      = int(SEED)
    out_path  = Path(OUTPUT_TXT) if OUTPUT_TXT else (
        test_root / f"summary_{VARIANT}.txt")
    csv_path  = test_root / f"per_family_{VARIANT}.csv"

    if not test_root.exists() or not test_root.is_dir():
        print(f"ERROR: thư mục '{test_root}' không tồn tại")
        sys.exit(1)
    if n_test < 1:
        print(f"ERROR: N_TEST phải ≥ 1 (đang là {n_test})")
        sys.exit(1)

    print("="*70)
    print("M6-v3 (Markov) — External Test (UMUDGA, Netlab360, ...)")
    print("="*70)
    print(f"Root      : {test_root}")
    print(f"N_TEST    : {n_test:,} mẫu/file (lấy ngẫu nhiên, seed={seed})")
    print(f"Output    : {out_path}")
    print(f"Device    : {DEVICE}")

    # ------------------------------------------------------------------
    # [1] Load tất cả thành phần M6-v3
    # ------------------------------------------------------------------
    print("\n[1/4] Load M6-v3 components ...")
    bilstm_model, bilstm_dim = load_bilstm(BILSTM_DIR)
    string_head = load_string_head(M6V3_MODELS_DIR / "string_head.pt", bilstm_dim)
    dbert_model, dbert_tok = load_distilbert(DISTILBERT_DIR)
    xgb, scaler = load_handcraft(M6V3_MODELS_DIR)
    weights, threshold = load_fusion_config(M6V3_METRICS)

    # Load resources cho 20 handcraft features (PHẢI KHỚP với pipeline train trong M6_v3_Markov.py)
    print("\n  Load resources cho 20 handcraft features:")
    _load_english_words()
    markov_cache = M6V3_DIR / "benign_markov.npz"
    load_markov_cache(markov_cache)

    # Sanity check trên 1 domain mẫu
    sample = "google.com"
    fv = hand_crafted_features(sample)
    assert fv.shape[0] == 20, f"Feature dim mismatch: {fv.shape[0]} != 20"
    print(f"  Sanity check '{sample}': dim={fv.shape[0]}, "
          f"meaning={fv[17]:.3f}, mk_bi={fv[18]:.2f}, mk_tri={fv[19]:.2f}")

    models_pack = (bilstm_model, string_head, dbert_model, dbert_tok, xgb, scaler)

    # ------------------------------------------------------------------
    # [2] Quét cấu trúc thư mục
    # ------------------------------------------------------------------
    print(f"\n[2/4] Quét thư mục {test_root}/ ...")
    datasets = {}
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir(): continue
        files = sorted(sub.glob("*.txt"))
        if not files: continue
        datasets[sub.name] = files
        print(f"  ✓ {sub.name:<20}: {len(files)} họ DGA")

    if not datasets:
        print(f"ERROR: không tìm thấy thư mục dataset nào trong '{test_root}'")
        sys.exit(1)

    # ------------------------------------------------------------------
    # [3] Test từng họ DGA trong từng dataset
    # ------------------------------------------------------------------
    print(f"\n[3/4] Test từng họ DGA (N_TEST={n_test:,}/file) ...")
    all_rows = []
    grand_test = 0; grand_dga = 0
    dataset_summary = {}
    t0 = time.time()

    for ds_name, files in datasets.items():
        ds_test = 0; ds_dga = 0
        ds_rows = []
        print(f"\n  === {ds_name} ===")
        print(f"  {'Family':<28} {'N_total':>9} {'N_test':>8} {'N_DGA':>8} "
              f"{'DR(%)':>8} {'mean_p':>8}")
        print(f"  {'-'*28} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for fpath in files:
            fname = fpath.stem
            try:
                all_domains = load_domain_file(fpath)
                n_total = len(all_domains)
                if n_total == 0:
                    print(f"  {fname:<28} {'EMPTY':>9}")
                    continue
                # Sample (giới hạn n_test)
                domains, n_used = sample_domains(all_domains, n_test, seed)
                # Test
                _, n_dga, dr, mean_p = test_family(
                    domains, models_pack, weights, threshold)

                row = {
                    "dataset":  ds_name,
                    "family":   fname,
                    "n_total":  n_total,
                    "n_test":   n_used,
                    "n_dga":    n_dga,
                    "dr":       dr,
                    "mean_prob": mean_p,
                }
                ds_rows.append(row); all_rows.append(row)
                ds_test += n_used; ds_dga += n_dga

                flag = "" if dr >= 0.90 else (" *" if dr >= 0.80 else " !!")
                print(f"  {fname:<28} {n_total:>9,} {n_used:>8,} "
                      f"{n_dga:>8,} {dr*100:>7.2f}{flag} {mean_p:>8.4f}")
            except Exception as e:
                print(f"  {fname:<28} ERROR: {e}")

        # Dataset summary
        ds_dr = ds_dga / ds_test if ds_test > 0 else 0.0
        dataset_summary[ds_name] = {
            "n_test_total": ds_test, "n_dga_total": ds_dga,
            "dr_total": ds_dr, "n_families": len(ds_rows),
        }
        grand_test += ds_test; grand_dga += ds_dga
        print(f"  {'-'*28} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'TOTAL ' + ds_name:<28} {'':>9} {ds_test:>8,} "
              f"{ds_dga:>8,} {ds_dr*100:>7.2f}")

    elapsed = time.time() - t0
    grand_dr = grand_dga / grand_test if grand_test > 0 else 0.0
    print(f"\n  ===== GRAND TOTAL =====")
    print(f"  N_test = {grand_test:,}, N_DGA = {grand_dga:,}, "
          f"DR = {grand_dr*100:.2f}%, time = {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # [4] Lưu kết quả
    # ------------------------------------------------------------------
    print(f"\n[4/4] Ghi kết quả ...")

    # CSV chi tiết
    df_csv = pd.DataFrame(all_rows)
    df_csv["dr"] = df_csv["dr"].apply(lambda x: round(x*100, 4))  # %
    df_csv = df_csv.rename(columns={"dr": "dr_percent"})
    df_csv.to_csv(csv_path, index=False, sep=";")
    print(f"  ✓ {csv_path}")

    # summary.txt
    # Trọng số chuẩn hóa (cái thực sự dùng trong fuse)
    _pos = {k: weights.get(k, 0) for k in ("string", "semantic", "handcraft")
            if weights.get(k, 0) > 0}
    _s = sum(_pos.values()) or 1.0
    wn = {k: (weights.get(k, 0)/_s if weights.get(k, 0) > 0 else 0.0)
          for k in ("string", "semantic", "handcraft")}

    lines = [
        "="*70,
        f"M6-v3 External Test Summary  [VARIANT = {VARIANT}]",
        "="*70,
        f"Model dir   : {M6V3_DIR}",
        f"Model       : M6-v3 (Late Fusion: BiLSTM + DistilBERT + Handcraft)",
        f"Weights RAW : string={weights.get('string',0):.3f}, "
        f"semantic={weights.get('semantic',0):.3f}, "
        f"handcraft={weights.get('handcraft',0):.3f}",
        f"Weights NORM: string={wn['string']:.3f}, "
        f"semantic={wn['semantic']:.3f}, "
        f"handcraft={wn['handcraft']:.3f}  (cái thực sự dùng)",
        f"Handcraft   : {'BỎ QUA (w=0, skip để tiết kiệm)' if (weights.get('handcraft',0)==0 and SKIP_ZERO_WEIGHT_BRANCH) else 'CÓ tính'}",
        f"Threshold   : {threshold:.4f}",
        f"N_TEST      : {n_test:,} mẫu/file (lấy ngẫu nhiên, seed={seed})",
        f"Inference   : {elapsed:.1f}s, device={DEVICE}",
        "",
    ]
    for ds_name, files in datasets.items():
        ds_rows = [r for r in all_rows if r["dataset"] == ds_name]
        if not ds_rows: continue
        lines.append(f"[{ds_name}]")
        lines.append(f"  {'Family':<28} {'N_total':>10} {'N_test':>9} "
                     f"{'N_DGA':>9} {'DR(%)':>8}")
        lines.append("  " + "─" * 66)
        for r in ds_rows:
            lines.append(f"  {r['family']:<28} {r['n_total']:>10,} "
                         f"{r['n_test']:>9,} {r['n_dga']:>9,} {r['dr']*100:>7.2f}")
        s = dataset_summary[ds_name]
        lines.append("  " + "─" * 66)
        lines.append(f"  {'TOTAL ' + ds_name:<28} {'':>10} "
                     f"{s['n_test_total']:>9,} {s['n_dga_total']:>9,} "
                     f"{s['dr_total']*100:>7.2f}")
        lines.append("")

    lines += [
        "="*70,
        f"GRAND TOTAL                  N_test={grand_test:,}  "
        f"N_DGA={grand_dga:,}  DR={grand_dr*100:.2f}%",
        "="*70,
        "",
        "Ghi chú:",
        "  - DR (Detection Rate) = N_DGA / N_TEST = Recall (vì các file đều là DGA)",
        "  - N_total: số domain trong file gốc",
        "  - N_test:  số domain thực tế dùng (≤ N_TEST input)",
        "  - N_DGA:   số domain được M6-v3 dự đoán là DGA",
        "  - Khi N_total < N_TEST, dùng toàn bộ file. Khi N_total ≥ N_TEST,",
        "    lấy ngẫu nhiên N_TEST mẫu (cố định seed để có thể tái hiện).",
    ]
    summary = "\n".join(lines)
    out_path.write_text(summary, encoding="utf-8")
    print(f"  ✓ {out_path}")

    # In tóm tắt cuối
    print("\n" + "="*70)
    print("KẾT QUẢ TÓM TẮT")
    print("="*70)
    for ds_name, s in dataset_summary.items():
        print(f"  {ds_name:<20}: {s['n_families']} họ, "
              f"{s['n_test_total']:,} mẫu, DR={s['dr_total']*100:.2f}%")
    print(f"  {'GRAND TOTAL':<20}: {grand_test:,} mẫu, DR={grand_dr*100:.2f}%")


if __name__ == "__main__":
    main()
