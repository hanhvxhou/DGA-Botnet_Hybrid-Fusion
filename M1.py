"""
M1 — Random Forest + n-gram + PCA (Soleymani & Arabgol, 2021) — self-contained

Input  : DataNew/{train.csv, val.csv, test.csv}  (đã chia sẵn để tránh leakage)
Output : out_M1/
            - model.joblib         — pipeline đã huấn luyện
            - metrics.json         — chỉ số hiệu năng + chi phí
            - confusion_matrix.csv — ma trận nhầm lẫn trên test
            - test_predictions.csv — dự đoán chi tiết trên tập test
            - per_family_results.json / per_family_summary.txt  — Bảng 5
            - summary.txt          — tóm tắt dễ đọc

Cài đặt:
    pip install numpy pandas scikit-learn scipy joblib
    pip install codecarbon  # tuỳ chọn — đo năng lượng chính xác (đọc RAPL)

Chạy:
    python M1.py
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["USE_TF"]                     = "NO"
os.environ["USE_TORCH"]                  = "YES"
os.environ["TRANSFORMERS_VERBOSITY"]    = "error"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"

import json
import math
import re
import sys
import time
import platform
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR  = Path("out_M1")
SEED        = 42
# Trích đặc trưng
NGRAM_RANGE = (1, 3)
MAX_NGRAM_FEATURES = 5000
PCA_DIM = 50

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 25
RF_N_JOBS       = -1
RF_CLASS_WEIGHT = "balanced"

# CPU profile — máy Windows / Intel 24-core / RTX 5070 Ti
CPU_MODEL              = "Intel 24-core (Xeon W / Core Ultra class)"
CPU_BASE_POWER_W       = 125.0      # TDP base workstation-class
CPU_MAX_TURBO_POWER_W  = 280.0      # MTP boost
CPU_AVG_MULTITHREAD_W  = 200.0      # trung bình khi full load
CPU_AVG_LIGHT_LOAD_W   = 60.0       # tải nhẹ
CPU_LOGICAL_CORES      = 24

USE_CODECARBON = True

# ---------------------------------------------------------------------------
# ĐỌC DỮ LIỆU
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


def estimate_power_watts(cpu_seconds, wall_seconds):
    if wall_seconds <= 0:
        return CPU_AVG_LIGHT_LOAD_W
    util = min(1.0, cpu_seconds / (wall_seconds * CPU_LOGICAL_CORES))
    util_low, util_high = 0.05, 1.0
    p_low, p_high = CPU_AVG_LIGHT_LOAD_W, CPU_AVG_MULTITHREAD_W
    if util <= util_low:
        return p_low
    if util >= util_high:
        return p_high
    return p_low + (p_high - p_low) * (util - util_low) / (util_high - util_low)


class EnergyMeter:
    def __init__(self, label):
        self.label = label
        self.method = "estimated_from_utilization"
        self.energy_joules = 0.0
        self.power_watts = 0.0
        self.utilization = 0.0
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
        self._t_wall = time.time()
        self._t_cpu = time.process_time()
        if self._tracker is not None:
            try:
                self._tracker.start()
            except Exception:
                self._tracker = None
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
                else:
                    self._fallback()
            except Exception:
                self._fallback()
        else:
            self._fallback()

    def _fallback(self):
        self.power_watts = estimate_power_watts(self.cpu_seconds, self.wall_seconds)
        self.energy_joules = self.power_watts * self.wall_seconds


def measure_energy_per_sample(rf, X_test, n_repeats=3):
    n = len(X_test)
    rs = []
    for _ in range(n_repeats):
        with EnergyMeter("inference") as m:
            rf.predict(X_test)
        rs.append({"wall": m.wall_seconds, "cpu": m.cpu_seconds,
                   "energy": m.energy_joules, "power": m.power_watts,
                   "util": m.utilization, "method": m.method})
    avg = lambda k: float(np.mean([r[k] for r in rs]))
    return {
        "n_repeats": n_repeats, "n_samples": n,
        "avg_wall_seconds":    avg("wall"),
        "avg_cpu_seconds":     avg("cpu"),
        "avg_energy_joules":   avg("energy"),
        "avg_power_watts":     avg("power"),
        "avg_cpu_utilization": avg("util"),
        "energy_per_sample_mJ": avg("energy") / n * 1000.0,
        "wall_per_sample_ms":   avg("wall") / n * 1000.0,
        "method": rs[0]["method"],
    }


# ---------------------------------------------------------------------------
# ĐẶC TRƯNG THỦ CÔNG
# ---------------------------------------------------------------------------
VOWELS = set("aeiou"); CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
DIGITS = set("0123456789"); HEX_CHARS = set("0123456789abcdef")

# Danh sách từ tiếng Anh phổ biến cho meaning_ratio (theo Soleymani & Arabgol 2021)
# Lazy-load: ưu tiên nltk.corpus.words; fallback dùng bộ ~500 từ phổ biến
ENGLISH_WORDS_SET = None

def _load_english_words():
    """Load từ điển tiếng Anh cho meaning_ratio."""
    global ENGLISH_WORDS_SET
    if ENGLISH_WORDS_SET is not None:
        return
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


def meaning_ratio(text):
    """ut_meaning_ratio (Soleymani & Arabgol 2021): % ký tự thuộc từ tiếng Anh
    được phát hiện qua tham lam segmentation."""
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

COMMON_TLDS = {"com","net","org","info","biz","co","io","me","us","uk","vn","cn","ru","de","fr",
               "jp","in","br","tv","cc","name","online","site","shop","app","dev","club","xyz",
               "top","live","store","tech","art","mobi","asia","edu","gov","mil","int","pro"}


def shannon_entropy(s):
    if not s: return 0.0
    counts = Counter(s); n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())


def longest_run(s, charset):
    best = cur = 0
    for ch in s:
        if ch in charset:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def split_domain(domain):
    d = re.sub(r"^www\.", "", domain.strip().lower())
    parts = d.split(".")
    if len(parts) == 1: return parts[0], ""
    return ".".join(parts[:-1]), parts[-1]


def hand_crafted_features(domain):
    d = domain.strip().lower()
    main, tld = split_domain(d)
    text = main.replace(".", "")
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
        longest_run(text, VOWELS), longest_run(text, CONSONANTS), longest_run(text, DIGITS),
        d.count("."), len(tld), int(any(c in DIGITS for c in text)),
        int(tld in COMMON_TLDS), n - nu,
        meaning_ratio(text),                  # 18: ut_meaning_ratio (Soleymani & Arabgol 2021)
    ], dtype=np.float32)


def extract_handcrafted_matrix(domains):
    return np.vstack([hand_crafted_features(d) for d in domains])


# ---------------------------------------------------------------------------
# HEURISTIC PHÂN LOẠI DGA (cho Bảng 5)
# ---------------------------------------------------------------------------
def classify_dga_type(domain):
    main, _ = split_domain(domain)
    text = main.replace(".", "")
    n = max(len(text), 1)
    ent = shannon_entropy(text)
    vowel_ratio = sum(1 for c in text if c in VOWELS) / n
    digit_ratio = sum(1 for c in text if c in DIGITS) / n
    length = len(text)
    if length >= 12 and vowel_ratio >= 0.30 and ent <= 3.6:
        return "DGA-W"
    if ent >= 3.8 or digit_ratio >= 0.15:
        return "DGA-R"
    if 0.20 <= vowel_ratio <= 0.45 and ent < 3.8:
        return "DGA-P"
    return "DGA-R"


def classify_many(domains):
    return np.array([classify_dga_type(d) for d in domains])


# ---------------------------------------------------------------------------
# PIPELINE ĐẶC TRƯNG
# ---------------------------------------------------------------------------
def build_features(X_train, X_val, X_test):
    print(f"[Features] TF-IDF char n-gram {NGRAM_RANGE}, max_features={MAX_NGRAM_FEATURES} ...")
    vec = TfidfVectorizer(analyzer="char", ngram_range=NGRAM_RANGE,
                           max_features=MAX_NGRAM_FEATURES, lowercase=True,
                           min_df=2, sublinear_tf=True)
    Xtr_t = vec.fit_transform(X_train); Xva_t = vec.transform(X_val); Xte_t = vec.transform(X_test)

    print(f"[Features] TruncatedSVD → {PCA_DIM}-d ...")
    svd = TruncatedSVD(n_components=PCA_DIM, random_state=SEED)
    Xtr_s = svd.fit_transform(Xtr_t); Xva_s = svd.transform(Xva_t); Xte_s = svd.transform(Xte_t)
    print(f"           Phương sai giải thích: {svd.explained_variance_ratio_.sum():.4f}")
    print("[Features] Đặc trưng thủ công (18d, gồm meaning_ratio) ...")
    Xtr_h = extract_handcrafted_matrix(X_train)
    Xva_h = extract_handcrafted_matrix(X_val)
    Xte_h = extract_handcrafted_matrix(X_test)

    scaler = StandardScaler()
    Xtr_h = scaler.fit_transform(Xtr_h); Xva_h = scaler.transform(Xva_h); Xte_h = scaler.transform(Xte_h)

    Xtr = np.hstack([Xtr_s, Xtr_h]); Xva = np.hstack([Xva_s, Xva_h]); Xte = np.hstack([Xte_s, Xte_h])
    print(f"           Kích thước cuối: {Xtr.shape[1]}")
    return Xtr, Xva, Xte, {"vectorizer": vec, "svd": svd, "scaler": scaler}


# ---------------------------------------------------------------------------
# ĐÁNH GIÁ
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba, label="Test"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    print(f"\n=== {label} ===")
    print(f"Accuracy : {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%  |  AUC-ROC : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["benign","DGA"], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:"); print(cm)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": auc, "confusion_matrix": cm.tolist()}


def evaluate_per_family(y_true, y_pred, dga_types):
    results = {}
    benign_mask = (y_true == 0)
    for lbl in ["DGA-R", "DGA-P", "DGA-W"]:
        dga_mask = (y_true == 1) & (dga_types == lbl)
        n_dga = int(dga_mask.sum())
        if n_dga == 0:
            results[lbl] = {"n_dga": 0, "warning": "không có mẫu"}
            continue
        sub = benign_mask | dga_mask
        yt, yp = y_true[sub], y_pred[sub]
        acc = accuracy_score(yt, yp); prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0); f1 = f1_score(yt, yp, zero_division=0)
        cm = confusion_matrix(yt, yp)
        results[lbl] = {
            "n_dga": n_dga, "n_benign": int(benign_mask.sum()),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm.tolist(),
        }
    return results


# ---------------------------------------------------------------------------
# KÍCH THƯỚC MÔ HÌNH
# ---------------------------------------------------------------------------
def get_size_mb(path): return path.stat().st_size / (1024 * 1024)


def get_components_size(payload):
    sizes = {}
    probe = Path("_size_probe.joblib")
    for k in ("vectorizer", "svd", "scaler", "rf"):
        if k in payload:
            try:
                joblib.dump(payload[k], probe, compress=0)
                sizes[k + "_MB"] = get_size_mb(probe)
                probe.unlink(missing_ok=True)
            except Exception:
                sizes[k + "_MB"] = None
    return sizes


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print(f"[CPU] {CPU_MODEL}  |  logical cores={CPU_LOGICAL_CORES}")
    print(f"\n[1/7] Đọc 3 file CSV từ {DATA_DIR}/ ...")
    print("      Load English words cho meaning_ratio (Soleymani & Arabgol 2021) ...")
    _load_english_words()
    print("[2/7] Chia train/val/test (stratified) ...")
    X_train, X_val, X_test, y_train, y_val, y_test, data_meta = load_data_split()
    n_total = data_meta["n_total"]
    n_pos   = data_meta["n_train_dga"]    + data_meta["n_val_dga"]    + data_meta["n_test_dga"]
    n_neg   = data_meta["n_train_benign"] + data_meta["n_val_benign"] + data_meta["n_test_benign"]
    print(f"      Train: {len(X_train):,}   Val: {len(X_val):,}   Test: {len(X_test):,}")
    print("[3/7] Trích đặc trưng ...")
    t0 = time.time()
    Xtr, Xva, Xte, arts = build_features(X_train, X_val, X_test)
    feat_time = time.time() - t0
    print(f"      Thời gian: {feat_time:.2f}s")
    print(f"[4/7] Huấn luyện Random Forest (n_estimators={RF_N_ESTIMATORS}, "
          f"max_depth={RF_MAX_DEPTH}) — đo năng lượng ...")
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                 n_jobs=RF_N_JOBS, random_state=SEED,
                                 class_weight=RF_CLASS_WEIGHT, criterion="gini")
    with EnergyMeter("training") as em:
        rf.fit(Xtr, y_train)
    print(f"      Wall={em.wall_seconds:.2f}s  CPU={em.cpu_seconds:.2f}s  "
          f"Util={em.utilization*100:.1f}%  Power={em.power_watts:.1f}W  "
          f"Energy={em.energy_joules:.2f}J  ({em.method})")

    val_pred = rf.predict(Xva); val_proba = rf.predict_proba(Xva)[:, 1]
    val_metrics = evaluate(y_val, val_pred, val_proba, label="Validation")

    print("\n[5/7] Đánh giá test + đo năng lượng suy diễn (3 lần) ...")
    energy_infer = measure_energy_per_sample(rf, Xte, n_repeats=3)
    print(f"      Wall={energy_infer['wall_per_sample_ms']:.4f} ms/mẫu  "
          f"Energy={energy_infer['energy_per_sample_mJ']:.4f} mJ/mẫu  "
          f"Power={energy_infer['avg_power_watts']:.1f}W")

    test_pred = rf.predict(Xte); test_proba = rf.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_proba, label="Test")

    # -- PER-FAMILY --
    print("\n[6/7] Phân loại per-family bằng heuristic + tính F1 theo loại ...")
    dga_types = classify_many(X_test)
    dga_display = np.where(y_test == 1, dga_types, "benign")
    c = Counter(dga_display)
    for k in ["DGA-R", "DGA-P", "DGA-W", "benign"]:
        print(f"      {k:<8} : {c[k]:>8,}  ({c[k]/len(X_test)*100:5.2f}%)")
    per_family = evaluate_per_family(y_test, test_pred, dga_types)
    print("      Kết quả per-family:")
    for lbl in ["DGA-R", "DGA-P", "DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            print(f"        {lbl}: F1={r['f1']*100:.2f}%  Prec={r['precision']*100:.2f}%  "
                  f"Recall={r['recall']*100:.2f}%  (n={r['n_dga']:,})")

    # -- LƯU --
    print("\n[7/7] Lưu mô hình + kết quả ...")
    payload = {"vectorizer": arts["vectorizer"], "svd": arts["svd"],
               "scaler": arts["scaler"], "rf": rf,
               "config": {"ngram_range": NGRAM_RANGE, "max_ngram_features": MAX_NGRAM_FEATURES,
                           "pca_dim": PCA_DIM, "rf_n_estimators": RF_N_ESTIMATORS,
                           "rf_max_depth": RF_MAX_DEPTH, "seed": SEED}}
    model_path = OUTPUT_DIR / "model.joblib"
    joblib.dump(payload, model_path, compress=3)
    model_size_mb = get_size_mb(model_path)
    print(f"      Đã lưu pipeline → {model_path}  ({model_size_mb:.2f} MB)")

    comp_sizes = get_components_size(payload)
    for k, v in comp_sizes.items():
        if v is not None:
            print(f"        {k}: {v:.2f} MB")

    # metrics.json
    metrics = {
        "validation": val_metrics,
        "test":       test_metrics,
        "per_family": per_family,
        "dga_type_distribution": dict(c),
        "model_size": {"compressed_MB": model_size_mb, "components_MB": comp_sizes},
        "training_cost": {
            "wall_seconds": em.wall_seconds, "cpu_seconds": em.cpu_seconds,
            "cpu_utilization": em.utilization, "avg_power_watts": em.power_watts,
            "energy_joules": em.energy_joules, "energy_Wh": em.energy_joules/3600.0,
            "energy_method": em.method,
        },
        "inference_cost": energy_infer,
        "feature_extraction_seconds": feat_time,
        "data": {"data_dir": DATA_DIR, "splits": {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV}, "total_samples": n_total,
                 "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                 "n_positive_total": n_pos, "n_negative_total": n_neg},
        "config": {
            "ngram_range": list(NGRAM_RANGE), "max_ngram_features": MAX_NGRAM_FEATURES,
            "pca_dim": PCA_DIM, "rf_n_estimators": RF_N_ESTIMATORS,
            "rf_max_depth": RF_MAX_DEPTH, "rf_class_weight": RF_CLASS_WEIGHT,
            "seed": SEED,
        },
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
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
        },
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=float)

    # confusion matrix + predictions
    cm = np.array(test_metrics["confusion_matrix"])
    pd.DataFrame(cm, index=["true_benign","true_DGA"], columns=["pred_benign","pred_DGA"]).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv")
    pd.DataFrame({"domain": X_test, "true_label": y_test, "pred_label": test_pred,
                   "prob_DGA": test_proba}).to_csv(OUTPUT_DIR / "test_predictions.csv",
                                                    index=False, sep=CSV_SEP)

    # per_family_results.json
    with open(OUTPUT_DIR / "per_family_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "M1 — RF + n-gram + PCA",
            "dga_type_distribution": dict(c),
            "overall": {"accuracy": test_metrics["accuracy"], "f1": test_metrics["f1"]},
            "per_family": per_family,
            "heuristic_thresholds": metrics["heuristic_thresholds"],
        }, f, indent=2, ensure_ascii=False, default=float)

    # per_family_summary.txt
    pf_lines = ["====== M1 — F1 theo loại DGA ======",
                 f"Model        : {model_path}",
                 f"Test samples : {len(X_test):,}", ""]
    pf_lines.append("Phân bố loại DGA (heuristic):")
    for k in ["DGA-R","DGA-P","DGA-W","benign"]:
        pf_lines.append(f"  {k:<8} : {c[k]:>8,}  ({c[k]/len(X_test)*100:5.2f}%)")
    pf_lines.append("")
    pf_lines.append(f"Toàn test: Accuracy = {test_metrics['accuracy']*100:.2f}%   "
                     f"F1 = {test_metrics['f1']*100:.2f}%")
    pf_lines.append("")
    pf_lines.append(f"{'Loại':<8} {'n_dga':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    for lbl in ["DGA-R","DGA-P","DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            pf_lines.append(f"{lbl:<8} {r['n_dga']:>8,} "
                            f"{r['accuracy']*100:>7.2f} {r['precision']*100:>7.2f} "
                            f"{r['recall']*100:>7.2f} {r['f1']*100:>7.2f}")
    (OUTPUT_DIR / "per_family_summary.txt").write_text("\n".join(pf_lines), encoding="utf-8")

    # summary.txt
    lines = [
        "================ M1 — Random Forest + n-gram + PCA ================",
        f"Data dir: {DATA_DIR}/ (train.csv, val.csv, test.csv)",
        f"Tổng mẫu           : {n_total:,}  (DGA={n_pos:,}, benign={n_neg:,})",
        f"Chia                : Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}",
        "",
        "[CPU profile]",
        f"  Model            : {CPU_MODEL}",
        f"  Logical cores    : {CPU_LOGICAL_CORES}",
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
        f"  CPU time         : {em.cpu_seconds:.2f} s",
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
        f"  File tổng (nén)  : {model_size_mb:.2f} MB",
    ]
    for k, v in comp_sizes.items():
        if v is not None:
            lines.append(f"  {k:<16} : {v:.2f} MB (chưa nén)")
    lines += ["",
              "[Bảng 5 — F1 theo loại DGA]",
              f"  {'Loại':<8} {'n_dga':>8} {'F1 (%)':>8} {'Prec':>8} {'Recall':>8}"]
    for lbl in ["DGA-R","DGA-P","DGA-W"]:
        r = per_family[lbl]
        if r.get("n_dga", 0) > 0:
            lines.append(f"  {lbl:<8} {r['n_dga']:>8,} "
                         f"{r['f1']*100:>7.2f} {r['precision']*100:>7.2f} "
                         f"{r['recall']*100:>7.2f}")
    lines += ["",
              "[Môi trường]",
              f"  Python           : {sys.version.split()[0]}",
              f"  Platform         : {platform.platform()}",
              f"  CPU count        : {os.cpu_count()}"]

    summary = "\n".join(lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print("\n" + "="*60 + "\nTÓM TẮT M1\n" + "="*60)
    print(summary)

    # dòng bảng 5 copy-paste
    def v(lbl):
        return f"{per_family[lbl]['f1']*100:.1f}" if per_family[lbl].get("n_dga",0)>0 else "—"
    print(f"\nDòng Bảng 5 (copy-paste):\n"
          f"  M1 — RF + n-gram + PCA  |  {v('DGA-R')}  |  {v('DGA-P')}  |  {v('DGA-W')}\n")


if __name__ == "__main__":
    main()
