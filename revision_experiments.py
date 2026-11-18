# -*- coding: utf-8 -*-
"""
revision_experiments.py - additional experiments for the IJICS major revision
(DGA-HLF, M6 Hybrid Late Fusion).

Put this file in the project root (same folder as M1.py ... M6_v3_Markov.py,
DataNew/, test_external/, out_select*/, out_M*_seed*/).

Sub-commands
------------
  python revision_experiments.py prep    --tranco top-1m.csv.zip --umbrella umbrella-top-1m.csv.zip
      (CPU, once) builds every evaluation set, handcraft feature caches
      (original Markov + corrected BOS/EOS Markov), tokenisation statistics,
      worked feature examples, Markov specification.
  python revision_experiments.py dump    --seed 42
      (GPU, once per seed) scores EVERY model on EVERY evaluation set and
      saves probabilities; trains early-fusion models and the corrected-Markov
      XGBoost for that seed; checks consistency with the saved M6 outputs.
  python revision_experiments.py latency --seed 42
      (GPU+CPU) end-to-end latency / throughput / tail latency / GPU energy.
  python revision_experiments.py collect
      copies small metadata files and zips rev_out/ -> rev_upload.zip

All outputs go to rev_out/. Nothing in the original out_* folders is modified.
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(".").resolve()
OUT = Path("rev_out")
SETS_DIR = OUT / "sets"
CACHE_DIR = OUT / "cache"
DATA_DIR = Path("DataNew")
TEST_FAMILY_DIR = DATA_DIR / "Test"
EXTERNAL_DIR = Path("test_external")
N_EXTERNAL = 2000          # identical to Test_M6_v3_Markov.py
EXTERNAL_SEED = 42         # identical to Test_M6_v3_Markov.py
N_BENIGN = 50000
BENIGN_SEED = 42
BERT_MAX_LEN = 64

SET_ORDER = ["val", "test", "test_tldcom", "test_notld", "test_www", "ext",
             "ben_tranco", "ben_umb_reg", "ben_umb_fqdn"]


def log(msg=""):
    print(msg, flush=True)


def m6mod():
    """Import the user's M6_v3_Markov.py (functions are reused verbatim)."""
    return importlib.import_module("M6_v3_Markov")


# ===========================================================================
# Paths of the trained artefacts for one seed
# ===========================================================================
def seed_paths(seed: int) -> dict:
    sel = Path("out_select") if seed == 42 else Path(f"out_select_seed{seed}")
    m6 = Path("out_M6_v3_Markov_seed42") if seed == 42 else Path(f"out_M6_v3_Markov_fullseed{seed}")
    if seed == 42 and not m6.exists() and Path("out_M6_v3_Markov").exists():
        m6 = Path("out_M6_v3_Markov")
    return {
        "select": sel,
        "bilstm": sel / "bilstm",
        "distilbert": sel / "distilbert",
        "bert": sel / "bert",
        "m6": m6,
        "rand_distilbert": Path(f"out_select_rand_seed{seed}") / "distilbert",
        **{f"M{k}": Path(f"out_M{k}_seed{seed}") for k in range(1, 6)},
    }


# ===========================================================================
# Corrected character Markov model (BOS/EOS, properly normalised)
# ===========================================================================
SIGMA = "abcdefghijklmnopqrstuvwxyz0123456789"
SIGMA_SET = set(SIGMA)
BOS, EOS = "^", "$"


def markov_text(domain: str, m6) -> str:
    """Same text as M6 (www. and TLD removed, '.' and '-' removed); additionally
    every character outside a-z0-9 is dropped (documented handling of
    unsupported characters)."""
    main, _ = m6.split_domain(domain)
    t = main.replace(".", "").replace("-", "")
    return "".join(c for c in t if c in SIGMA_SET)


class MarkovEOS:
    """Laplace-smoothed character bigram/trigram model with explicit BOS padding
    and an EOS symbol. Predicted alphabet = 36 symbols + EOS (V = 37), so every
    conditional distribution sums to one; context counts include only positions
    that are followed by a predicted symbol (including EOS)."""
    V = len(SIGMA) + 1

    def __init__(self):
        self.bi = Counter(); self.bi_h = Counter()
        self.tri = Counter(); self.tri_h = Counter()

    def fit(self, texts):
        for t in texts:
            s = BOS + t + EOS
            for i in range(1, len(s)):
                self.bi[s[i - 1:i + 1]] += 1
                self.bi_h[s[i - 1]] += 1
            s = BOS + BOS + t + EOS
            for i in range(2, len(s)):
                self.tri[s[i - 2:i + 1]] += 1
                self.tri_h[s[i - 2:i]] += 1
        return self

    def logp_bi(self, h, c):
        return math.log((self.bi.get(h + c, 0) + 1) / (self.bi_h.get(h, 0) + self.V))

    def logp_tri(self, h, c):
        return math.log((self.tri.get(h + c, 0) + 1) / (self.tri_h.get(h, 0) + self.V))

    def ppl_bi(self, t):
        s = BOS + t + EOS
        lp = sum(self.logp_bi(s[i - 1], s[i]) for i in range(1, len(s)))
        return math.exp(-lp / (len(s) - 1))

    def ppl_tri(self, t):
        s = BOS + BOS + t + EOS
        lp = sum(self.logp_tri(s[i - 2:i], s[i]) for i in range(2, len(s)))
        return math.exp(-lp / (len(s) - 2))

    def check_normalisation(self, n_hist=50):
        """max |sum_c P(c|h) - 1| over the most frequent histories (should be ~1e-15)."""
        out = 0.0
        for h, _ in self.bi_h.most_common(n_hist):
            s = sum(math.exp(self.logp_bi(h, c)) for c in SIGMA + EOS)
            out = max(out, abs(s - 1))
        for h, _ in self.tri_h.most_common(n_hist):
            s = sum(math.exp(self.logp_tri(h, c)) for c in SIGMA + EOS)
            out = max(out, abs(s - 1))
        return out

    def to_dict(self):
        return {"bi": dict(self.bi), "bi_h": dict(self.bi_h),
                "tri": dict(self.tri), "tri_h": dict(self.tri_h)}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.bi, m.bi_h = Counter(d["bi"]), Counter(d["bi_h"])
        m.tri, m.tri_h = Counter(d["tri"]), Counter(d["tri_h"])
        return m


def handcraft_matrix_eos(domains, m6, mk: MarkovEOS, hc_orig: np.ndarray):
    """20 features identical to M6 except features 19-20 (Markov perplexities)
    computed with the corrected estimator."""
    X = hc_orig.copy()
    for i, d in enumerate(domains):
        t = markov_text(d, m6)
        X[i, 18] = mk.ppl_bi(t)
        X[i, 19] = mk.ppl_tri(t)
    return X


# ===========================================================================
# Readers
# ===========================================================================
def load_domain_file_like_test_m6(path: Path) -> list:
    """Verbatim copy of load_domain_file() in Test_M6_v3_Markov.py."""
    domains = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            d = line.strip().lower()
            if not d or d.startswith("#"):
                continue
            d = re.sub(r"^https?://", "", d)
            d = re.sub(r"^www\.", "", d)
            d = d.split("/")[0].split("?")[0].strip()
            if d and not d.startswith("."):
                domains.append(d)
    return domains


def sample_like_test_m6(domains, n_test, seed):
    n_total = len(domains)
    if n_total == 0:
        return []
    if n_total <= n_test:
        return list(domains)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_total, size=n_test, replace=False)
    return [domains[i] for i in idx]


def read_top_list(path: Path) -> list:
    """Reads a Tranco/Umbrella list: .zip/.csv ('rank,domain') or .txt (one domain per line)."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if not n.endswith("/")][0]
            raw = z.read(name).decode("utf-8", errors="ignore")
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = line.split(",")[-1].strip().lower().rstrip(".")
        if d and d != "domain":
            out.append(d)
    return out


def registered_domain(fqdn: str, extractor) -> str:
    if extractor is not None:
        e = extractor(fqdn)
        if e.domain and e.suffix:
            return f"{e.domain}.{e.suffix}"
        return ""
    parts = fqdn.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else ""


def read_set(name: str) -> pd.DataFrame:
    return pd.read_csv(SETS_DIR / f"{name}.csv", sep=";", dtype=str,
                       keep_default_na=False).assign(label=lambda d: d["label"].astype(int))


def available_sets():
    return [s for s in SET_ORDER if (SETS_DIR / f"{s}.csv").exists()]


# ===========================================================================
# PREP
# ===========================================================================
def cmd_prep(args):
    m6 = m6mod()
    OUT.mkdir(exist_ok=True); SETS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log("[prep 1/7] DataNew ...")
    X_tr, X_va, X_te, y_tr, y_va, y_te, meta = m6.load_data_split(str(DATA_DIR))
    log(f"   train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")

    # family map from DataNew/Test/<family>.txt
    fam = {}
    if TEST_FAMILY_DIR.exists():
        for f in sorted(TEST_FAMILY_DIR.glob("*.txt")):
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                d = line.strip().lower()
                if d:
                    fam[d] = f.stem
    else:
        log(f"   [WARN] {TEST_FAMILY_DIR} not found - test families unknown")
    test_family = [("benign" if y == 0 else fam.get(d, "unknown")) for d, y in zip(X_te, y_te)]
    n_unknown = sum(1 for f in test_family if f == "unknown")
    log(f"   test DGA families mapped: {len(set(test_family)) - 1}  (unmapped DGA: {n_unknown})")

    def write_set(name, domains, labels, extra=None):
        df = pd.DataFrame({"domain": domains, "label": labels})
        for k, v in (extra or {}).items():
            df[k] = v
        df.to_csv(SETS_DIR / f"{name}.csv", sep=";", index=False)
        log(f"   set {name:<13} n={len(df):,}")

    write_set("val", X_va, y_va)
    write_set("test", X_te, y_te, {"family": test_family})

    # parsing / suffix robustness
    def swap_tld(d, new):
        main, tld = m6.split_domain(d)   # (also strips www.)
        if not tld:
            return d if new is None else d + new
        return main if new is None else main + new
    write_set("test_tldcom", [swap_tld(d, ".com") for d in X_te], y_te, {"family": test_family})
    write_set("test_notld", [swap_tld(d, None) for d in X_te], y_te, {"family": test_family})
    write_set("test_www", ["www." + d for d in X_te], y_te, {"family": test_family})

    log("[prep 2/7] external DGA sets (same sampling as Test_M6_v3_Markov.py) ...")
    all_dev = set(X_tr) | set(X_va) | set(X_te)
    ext_rows, overlap_rows = [], []
    if EXTERNAL_DIR.exists():
        for sub in sorted(EXTERNAL_DIR.iterdir()):
            if not sub.is_dir():
                continue
            for f in sorted(sub.glob("*.txt")):
                doms = load_domain_file_like_test_m6(f)
                uniq = set(doms)
                overlap_rows.append({
                    "dataset": sub.name, "family": f.stem, "n_lines": len(doms),
                    "n_unique": len(uniq),
                    "overlap_train": len(uniq & set(X_tr)), "overlap_val": len(uniq & set(X_va)),
                    "overlap_test": len(uniq & set(X_te)), "overlap_any_dev": len(uniq & all_dev),
                    "in_datanew_family_names": f.stem.lower() in {v.lower() for v in fam.values()},
                })
                for d in sample_like_test_m6(doms, N_EXTERNAL, EXTERNAL_SEED):
                    ext_rows.append((d, 1, sub.name, f.stem, int(d in all_dev)))
        ext = pd.DataFrame(ext_rows, columns=["domain", "label", "dataset", "family", "in_dev"])
        ext.to_csv(SETS_DIR / "ext.csv", sep=";", index=False)
        pd.DataFrame(overlap_rows).to_csv(OUT / "external_overlap.csv", sep=";", index=False)
        log(f"   set ext           n={len(ext):,}  families={len(overlap_rows)}")
    else:
        log(f"   [WARN] {EXTERNAL_DIR} not found - external set skipped")
        ext = pd.DataFrame(columns=["domain"])
    ext_all = set(ext["domain"]) if len(ext) else set()

    log("[prep 3/7] independent benign sets ...")
    rng = np.random.RandomState(BENIGN_SEED)
    if args.tranco and Path(args.tranco).exists():
        tr = read_top_list(Path(args.tranco))
        pool = [d for d in dict.fromkeys(tr) if d not in all_dev and d not in ext_all]
        pick = rng.choice(len(pool), size=min(N_BENIGN, len(pool)), replace=False)
        doms = [pool[i] for i in sorted(pick)]
        ranks = {d: i + 1 for i, d in enumerate(tr)}
        write_set("ben_tranco", doms, [0] * len(doms), {"rank": [ranks[d] for d in doms]})
        log(f"   Tranco list={len(tr):,}  pool after excluding DataNew/external={len(pool):,}")
    else:
        log("   [WARN] --tranco not given/not found - ben_tranco skipped")
    if args.umbrella and Path(args.umbrella).exists():
        try:
            import tldextract
            extractor = tldextract.TLDExtract(suffix_list_urls=())   # bundled PSL snapshot, offline
            psl = f"tldextract {tldextract.__version__} (bundled snapshot)"
        except Exception:
            extractor, psl = None, "last-two-labels fallback (tldextract not installed)"
        um = read_top_list(Path(args.umbrella))
        um = [d for d in dict.fromkeys(um) if "." in d and not d.endswith(".arpa")]
        pool = [d for d in um if d not in all_dev and d not in ext_all]
        pick = rng.choice(len(pool), size=min(N_BENIGN, len(pool)), replace=False)
        doms = [pool[i] for i in sorted(pick)]
        write_set("ben_umb_fqdn", doms, [0] * len(doms))
        regs = list(dict.fromkeys(r for r in (registered_domain(d, extractor) for d in um) if r))
        pool = [d for d in regs if d not in all_dev and d not in ext_all]
        pick = rng.choice(len(pool), size=min(N_BENIGN, len(pool)), replace=False)
        doms = [pool[i] for i in sorted(pick)]
        write_set("ben_umb_reg", doms, [0] * len(doms))
        log(f"   Umbrella FQDN={len(um):,}  registered={len(regs):,}  PSL: {psl}")
    else:
        log("   [WARN] --umbrella not given/not found - Umbrella sets skipped")

    log("[prep 4/7] handcraft features: original Markov (verbatim M6) + corrected BOS/EOS Markov ...")
    m6._load_english_words()
    benign_train = [d for d, y in zip(X_tr, y_tr) if y == 0]
    m6.build_markov_models(benign_train)
    mk = MarkovEOS().fit([markov_text(d, m6) for d in benign_train])
    norm_err_eos = mk.check_normalisation()
    # deficiency of the original estimator: sum over next symbols for frequent contexts
    orig_sums = []
    uni = Counter()
    for d in benign_train:
        main, _ = m6.split_domain(d)
        for c in main.replace(".", "").replace("-", ""):
            uni[c] += 1
    for h, _ in uni.most_common(10):
        s = sum(math.exp(m6._MARKOV_BI_LOG_PROB.get(h + c, m6._MARKOV_BI_DEFAULT)) for c in SIGMA)
        orig_sums.append({"context": h, "sum_P_next_over_36_symbols": s})
    with open(CACHE_DIR / "markov_eos.json", "w", encoding="utf-8") as f:
        json.dump(mk.to_dict(), f)
    sets = {"train": list(X_tr)}
    for s in available_sets():
        sets[s] = read_set(s)["domain"].tolist()
    for s, doms in sets.items():
        t0 = time.time()
        hc = m6.extract_handcrafted_matrix(doms).astype(np.float32)
        hce = handcraft_matrix_eos(doms, m6, mk, hc).astype(np.float32)
        np.save(CACHE_DIR / f"hc_orig_{s}.npy", hc)
        np.save(CACHE_DIR / f"hc_eos_{s}.npy", hce)
        log(f"   {s:<13} {len(doms):>8,}  {time.time() - t0:6.1f}s")
    try:
        import nltk
        nltk_ver = nltk.__version__
    except Exception:
        nltk_ver = "n/a"
    spec = {
        "char_vocab_models": ["<pad>"] + list("abcdefghijklmnopqrstuvwxyz0123456789-."),
        "unsupported_char_handling_models": "mapped to <pad> (id 0) by CHAR_TO_ID.get(c, PAD_ID)",
        "markov_orig": {"V": 40, "bigram_states": len(m6._MARKOV_BI_LOG_PROB),
                        "trigram_states": len(m6._MARKOV_TRI_LOG_PROB),
                        "unseen_logp": m6._MARKOV_BI_DEFAULT,
                        "sum_P_next_frequent_contexts": orig_sums},
        "markov_eos": {"alphabet": SIGMA, "V": MarkovEOS.V, "bos": BOS, "eos": EOS,
                       "bigram_states": len(mk.bi), "trigram_states": len(mk.tri),
                       "max_normalisation_error": norm_err_eos},
        "n_benign_train_for_markov": len(benign_train),
        "dictionary": {"source": "nltk.corpus.words", "nltk_version": nltk_ver,
                       "filter": "isalpha and 3<=len<=12, lower-cased",
                       "n_words": len(m6.ENGLISH_WORDS_SET or [])},
        "feature_names": m6.HANDCRAFT_FEATURE_NAMES,
    }
    with open(OUT / "markov_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=float)
    log(f"   EOS normalisation error = {norm_err_eos:.2e}; original sums (first contexts) = "
        f"{[round(r['sum_P_next_over_36_symbols'], 3) for r in orig_sums[:5]]}")

    log("[prep 5/7] worked feature examples ...")
    te = read_set("test")
    types = np.where(te["label"].values == 1, m6.classify_many(te["domain"].tolist()), "benign")
    ex = ["google.com", "wikipedia.org"]
    for t in ["benign", "DGA-R", "DGA-P", "DGA-W"]:
        ex += te["domain"][types == t].head(2).tolist()
    hc = m6.extract_handcrafted_matrix(ex)
    hce = handcraft_matrix_eos(ex, m6, mk, hc)
    rows = []
    for i, d in enumerate(ex):
        r = {"domain": d, "subtype_rule": m6.classify_dga_type(d),
             "markov_text": markov_text(d, m6)}
        r.update({n: float(v) for n, v in zip(m6.HANDCRAFT_FEATURE_NAMES, hc[i])})
        r["markov_bi_perplexity_eos"] = float(hce[i, 18])
        r["markov_tri_perplexity_eos"] = float(hce[i, 19])
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "feature_examples.csv", sep=";", index=False)

    log("[prep 6/7] tokenisation statistics (DistilBERT / BERT tokenizers saved by selectAlgorithm) ...")
    tokenization_stats(te, types, m6)

    log("[prep 7/7] subtype rule margins ...")
    rows = []
    for d, y, t, f in zip(te["domain"], te["label"], types, te["family"]):
        if y != 1:
            continue
        main, _ = m6.split_domain(d)
        text = main.replace(".", "")
        n = max(len(text), 1)
        rows.append({"domain": d, "family": f, "subtype": t, "len": len(text),
                     "entropy": m6.shannon_entropy(text),
                     "vowel_ratio": sum(c in m6.VOWELS for c in text) / n,
                     "digit_ratio": sum(c in m6.DIGITS for c in text) / n})
    pd.DataFrame(rows).to_csv(OUT / "test_dga_subtype_inputs.csv", sep=";", index=False)
    log("[prep] done -> rev_out/")


def tokenization_stats(te, types, m6):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        log(f"   [WARN] transformers unavailable: {e}"); return
    rows, examples = [], []
    for key in ["distilbert", "bert"]:
        tdir = Path("out_select") / key / "tokenizer"
        try:
            tok = AutoTokenizer.from_pretrained(str(tdir) if tdir.exists() else
                                                {"distilbert": "distilbert-base-uncased",
                                                 "bert": "bert-base-uncased"}[key])
        except Exception as e:
            log(f"   [WARN] tokenizer {key}: {e}"); continue
        vocab_words = m6.ENGLISH_WORDS_SET or set()
        enc = tok(te["domain"].tolist(), add_special_tokens=True, truncation=False)["input_ids"]
        unk = tok.unk_token_id
        for t in ["benign", "DGA-R", "DGA-P", "DGA-W"]:
            idx = np.where(types == t)[0]
            n_tok, n_unk, n_single, n_dict, n_pieces, n_trunc, chars = [], 0, 0, 0, 0, 0, []
            for i in idx:
                ids = enc[i][1:-1]   # strip [CLS]/[SEP]
                n_tok.append(len(ids))
                chars.append(len(te["domain"].iloc[i]))
                if len(enc[i]) > BERT_MAX_LEN:
                    n_trunc += 1
                pieces = tok.convert_ids_to_tokens(ids)
                for p in pieces:
                    if p in (".", "-"):
                        continue
                    n_pieces += 1
                    q = p[2:] if p.startswith("##") else p
                    if len(q) == 1:
                        n_single += 1
                    if len(q) >= 3 and q in vocab_words:
                        n_dict += 1
                n_unk += sum(1 for x in ids if x == unk)
            n_tok = np.array(n_tok)
            rows.append({"tokenizer": key, "subtype": t, "n_domains": len(idx),
                         "mean_tokens": n_tok.mean(), "median_tokens": float(np.median(n_tok)),
                         "p95_tokens": float(np.percentile(n_tok, 95)),
                         "tokens_per_char": float(n_tok.sum() / max(sum(chars), 1)),
                         "pct_single_char_pieces": 100 * n_single / max(n_pieces, 1),
                         "pct_dictionary_pieces": 100 * n_dict / max(n_pieces, 1),
                         "n_unk_tokens": n_unk, "pct_domains_truncated_at_64": 100 * n_trunc / max(len(idx), 1)})
            for i in idx[:3]:
                examples.append({"tokenizer": key, "subtype": t, "domain": te["domain"].iloc[i],
                                 "pieces": " ".join(tok.convert_ids_to_tokens(enc[i][1:-1]))})
        rows.append({"tokenizer": key, "subtype": "vocab", "n_domains": tok.vocab_size})
    pd.DataFrame(rows).to_csv(OUT / "tokenization_stats.csv", sep=";", index=False)
    pd.DataFrame(examples).to_csv(OUT / "tokenization_examples.csv", sep=";", index=False)
    same = None
    try:
        same = (AutoTokenizer.from_pretrained(str(Path("out_select/distilbert/tokenizer"))).get_vocab()
                == AutoTokenizer.from_pretrained(str(Path("out_select/bert/tokenizer"))).get_vocab())
    except Exception:
        pass
    log(f"   tokenisation stats written; DistilBERT and BERT vocabularies identical: {same}")


# ===========================================================================
# DUMP (per seed)
# ===========================================================================
def _torch():
    import torch
    return torch


def probs_char_model(model, domains, encode_batch, dev, bs=2048, two=False):
    torch = _torch()
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(domains), bs):
            chunk = domains[i:i + bs]
            if two:
                xc, xb = encode_batch(chunk)
                logits = model(xc.to(dev), xb.to(dev))
            else:
                logits = model(encode_batch(chunk).to(dev))
            out.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def bert_forward(model, tok, domains, dev, bs=512, want="prob"):
    torch = _torch()
    probs, cls = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(domains), bs):
            enc = tok(domains[i:i + bs], truncation=True, padding="max_length",
                      max_length=BERT_MAX_LEN, return_tensors="pt")
            ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            h = model(ids, mask, return_embedding=True)
            if want in ("prob", "both"):
                p = torch.softmax(model.classifier(model.dropout(h)), 1)[:, 1]
                probs.append(p.float().cpu().numpy())
            if want in ("cls", "both"):
                cls.append(h.float().cpu().numpy())
    P = np.concatenate(probs).astype(np.float32) if probs else None
    C = np.concatenate(cls).astype(np.float32) if cls else None
    return P, C


def bilstm_emb(model, domains, m6, dev, bs=4096):
    torch = _torch()
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(domains), bs):
            ids = [[m6.CHAR_TO_ID.get(c, m6.PAD_ID) for c in d.strip().lower()][:m6.MAX_DOM_LEN]
                   for d in domains[i:i + bs]]
            ids = [x + [m6.PAD_ID] * (m6.MAX_DOM_LEN - len(x)) for x in ids]
            xb = torch.tensor(ids, dtype=torch.long, device=dev)
            out.append(model(xb, return_embedding=True).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def bilstm_native_from_emb(model, emb, dev, bs=8192):
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(emb), bs):
            out.append(torch.softmax(model.fc(torch.from_numpy(emb[i:i + bs]).to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def head_probs(head, X, dev, bs=8192):
    torch = _torch()
    out = []
    head.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(head(torch.from_numpy(X[i:i + bs]).to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def make_xgb(seed, dev_cuda):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, n_jobs=-1, tree_method="hist", random_state=seed,
              eval_metric="logloss", early_stopping_rounds=20, verbosity=0)
    if dev_cuda:
        try:
            return XGBClassifier(device="cuda", **kw)
        except TypeError:
            pass
    return XGBClassifier(**kw)


def cmd_dump(args):
    torch = _torch()
    m6 = m6mod()
    seed = args.seed
    P = seed_paths(seed)
    odir = OUT / f"seed{seed}"
    odir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[dump] seed={seed} device={dev}  paths: " + json.dumps({k: str(v) for k, v in P.items()}))
    torch.manual_seed(seed); np.random.seed(seed)

    sets = {s: read_set(s) for s in available_sets()}
    doms = {s: df["domain"].tolist() for s, df in sets.items()}
    res_path = odir / "probs.npz"
    res = dict(np.load(res_path)) if res_path.exists() else {}
    errors, checks = [], {}

    def have(key):
        return all(f"{key}|{s}" in res for s in sets)

    def save():
        np.savez_compressed(res_path, **res)

    def run(key, fn):
        if have(key) and not args.force:
            log(f"   [skip] {key}"); return
        t0 = time.time()
        try:
            out = fn()
            for s, v in out.items():
                res[f"{key}|{s}"] = np.asarray(v, dtype=np.float32)
            save()
            log(f"   [ok] {key:<16} {time.time() - t0:7.1f}s")
        except Exception as e:
            msg = f"{key}: {e!r}\n{traceback.format_exc()}"
            errors.append(msg)
            log(f"   [FAIL] {key}: {e!r}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    X_tr, X_va, X_te, y_tr, y_va, y_te, _ = m6.load_data_split(str(DATA_DIR))

    # ---------- BiLSTM (selectAlgorithm) + M6 string head + early fusion ----------
    emb_cache = {}

    def get_bilstm():
        model, dim = m6.load_bilstm(P["bilstm"])
        return model, dim

    def f_bilstm():
        model, _ = get_bilstm()
        out = {}
        for s in sets:
            e = bilstm_emb(model, doms[s], m6, dev)
            emb_cache[s] = e
            out[s] = bilstm_native_from_emb(model, e, dev)
        emb_cache["train"] = bilstm_emb(model, list(X_tr), m6, dev)
        del model
        return out
    run("bilstm", f_bilstm)

    def ensure_emb():
        if all(s in emb_cache for s in list(sets) + ["train"]):
            return
        model, _ = get_bilstm()
        for s in list(sets) + ["train"]:
            if s not in emb_cache:
                emb_cache[s] = bilstm_emb(model, doms[s] if s != "train" else list(X_tr), m6, dev)
        del model

    def f_string():
        ensure_emb()
        ck = torch.load(P["m6"] / "models" / "string_head.pt", map_location=dev, weights_only=False)
        head = m6.StringHead(ck["in_dim"], hidden=ck.get("hidden", 128), dropout=ck.get("dropout", 0.3)).to(dev)
        head.load_state_dict(ck["state_dict"]); head.eval()
        return {s: head_probs(head, emb_cache[s], dev) for s in sets}
    run("string", f_string)

    # ---------- DistilBERT / BERT / random-init DistilBERT ----------
    cls_cache = {}

    def f_bert_family(dirkey, keep_cls=False):
        def f():
            model, tok, _ = m6.load_distilbert(P[dirkey])
            out = {}
            for s in sets:
                p, c = bert_forward(model, tok, doms[s], dev, want="both" if keep_cls else "prob")
                out[s] = p
                if keep_cls:
                    cls_cache[s] = c
            if keep_cls:
                _, cls_cache["train"] = bert_forward(model, tok, list(X_tr), dev, want="cls")
            del model
            return out
        return f
    run("distilbert", f_bert_family("distilbert", keep_cls=True))
    run("bert", f_bert_family("bert"))
    if P["rand_distilbert"].joinpath("model.pt").exists():
        run("distilbert_rand", f_bert_family("rand_distilbert"))
    else:
        log(f"   [info] {P['rand_distilbert']} not found - random-init DistilBERT skipped")

    # ---------- handcraft XGB: original (saved) + corrected Markov (retrained) ----------
    def hc(kind, s):
        return np.load(CACHE_DIR / f"hc_{kind}_{s}.npy")

    def f_hc_orig():
        import joblib
        xgb = joblib.load(P["m6"] / "models" / "handcraft_xgb.joblib")
        sc = joblib.load(P["m6"] / "models" / "scaler.joblib")
        return {s: xgb.predict_proba(sc.transform(hc("orig", s)))[:, 1] for s in sets}
    run("hc_orig", f_hc_orig)

    def f_hc_eos():
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(hc("eos", "train"))
        xgb = make_xgb(seed, dev.type == "cuda")
        xgb.fit(sc.transform(hc("eos", "train")), y_tr,
                eval_set=[(sc.transform(hc("eos", "val")), y_va)], verbose=False)
        return {s: xgb.predict_proba(sc.transform(hc("eos", s)))[:, 1] for s in sets}
    run("hc_eos", f_hc_eos)

    # ---------- early fusion with the same active inputs ----------
    def ensure_cls():
        if all(s in cls_cache for s in list(sets) + ["train"]):
            return
        model, tok, _ = m6.load_distilbert(P["distilbert"])
        for s in list(sets) + ["train"]:
            if s not in cls_cache:
                _, cls_cache[s] = bert_forward(model, tok, doms[s] if s != "train" else list(X_tr), dev, want="cls")
        del model

    def concat(s, with_hc=False):
        parts = [emb_cache[s], cls_cache[s]]
        if with_hc:
            parts.append(hc("orig", s))
        return np.hstack(parts).astype(np.float32)

    def f_ef_mlp():
        from sklearn.preprocessing import StandardScaler
        ensure_emb(); ensure_cls()
        torch.manual_seed(seed); np.random.seed(seed)
        sc = StandardScaler().fit(concat("train"))
        head, _ = m6.train_string_head(sc.transform(concat("train")).astype(np.float32), y_tr,
                                       sc.transform(concat("val")).astype(np.float32), y_va)
        return {s: head_probs(head, sc.transform(concat(s)).astype(np.float32), dev) for s in sets}
    run("ef_mlp2", f_ef_mlp)

    def f_ef_xgb(with_hc):
        def f():
            ensure_emb(); ensure_cls()
            xgb = make_xgb(seed, dev.type == "cuda")
            xgb.fit(concat("train", with_hc), y_tr, eval_set=[(concat("val", with_hc), y_va)], verbose=False)
            return {s: xgb.predict_proba(concat(s, with_hc))[:, 1] for s in sets}
        return f
    run("ef_xgb2", f_ef_xgb(False))
    run("ef_xgb3", f_ef_xgb(True))
    emb_cache.clear(); cls_cache.clear()

    # ---------- baselines M1-M5 (fixed-backbone multi-seed runs) ----------
    def f_m1():
        import joblib
        m1 = importlib.import_module("M1")
        m1._load_english_words()
        pay = joblib.load(P["M1"] / "model.joblib")
        out = {}
        for s in sets:
            d = doms[s]
            Xs = pay["svd"].transform(pay["vectorizer"].transform(d))
            Xh = pay["scaler"].transform(m1.extract_handcrafted_matrix(d))
            out[s] = pay["rf"].predict_proba(np.hstack([Xs, Xh]))[:, 1]
        return out

    def f_m2():
        import joblib
        m2 = importlib.import_module("M2")
        vec = joblib.load(P["M2"] / "vectorizer.joblib")
        ck = torch.load(P["M2"] / "model.pt", map_location=dev, weights_only=False)
        model = m2.DeepMLP(ck["input_dim"], ck["hidden_dims"], ck["dropout"]).to(dev)
        model.load_state_dict(ck["state_dict"]); model.eval()
        enc = lambda chunk: torch.from_numpy(np.asarray(vec.transform(chunk).todense(), dtype=np.float32))
        return {s: probs_char_model(model, doms[s], enc, dev, bs=2048) for s in sets}

    def f_char(modname, cls_name, two=False):
        def f():
            mm = importlib.import_module(modname)
            ck = torch.load(P[modname] / "model.pt", map_location=dev, weights_only=False)
            model = getattr(mm, cls_name)().to(dev)
            model.load_state_dict(ck["state_dict"]); model.eval()
            if two:
                enc = lambda chunk: (mm.domains_to_tensor(chunk), mm.domains_to_bigram_tensor(chunk))
            else:
                enc = mm.domains_to_tensor
            return {s: probs_char_model(model, doms[s], enc, dev, bs=1024, two=two) for s in sets}
        return f
    run("M1", f_m1)
    run("M2", f_m2)
    run("M3", f_char("M3", "RCNN_SPP"))
    run("M4", f_char("M4", "ATT_CNN_BiLSTM"))
    run("M5", f_char("M5", "DualEmbeddingTransformer", two=True))

    # ---------- consistency checks against saved outputs ----------
    try:
        pm6 = pd.read_csv(P["m6"] / "test_predictions.csv", sep=";")
        for col, key in [("P_string", "string"), ("P_semantic", "distilbert"), ("P_handcraft", "hc_orig")]:
            if f"{key}|test" in res and col in pm6.columns:
                checks[f"M6.{col}_maxabsdiff"] = float(np.max(np.abs(pm6[col].values - res[f"{key}|test"])))
    except Exception as e:
        checks["M6"] = repr(e)
    for key, d in [("bilstm", P["bilstm"]), ("distilbert", P["distilbert"]), ("bert", P["bert"]),
                   *[(f"M{k}", P[f"M{k}"]) for k in range(1, 6)]]:
        try:
            pp = pd.read_csv(d / "test_predictions.csv", sep=";")
            if f"{key}|test" in res:
                checks[f"{key}.prob_maxabsdiff"] = float(np.max(np.abs(pp["prob_DGA"].values - res[f"{key}|test"])))
        except Exception as e:
            checks[f"{key}"] = f"no test_predictions.csv ({type(e).__name__})"
    with open(odir / "checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
    # metadata of fusion config saved by M6
    for name in ["metrics.json", "summary.txt"]:
        if (P["m6"] / name).exists():
            shutil.copy(P["m6"] / name, odir / f"m6_{name}")
    for k in ["bilstm", "distilbert", "bert", "rand_distilbert", *[f"M{i}" for i in range(1, 6)]]:
        if (P[k] / "metrics.json").exists():
            shutil.copy(P[k] / "metrics.json", odir / f"{k}_metrics.json")
    (odir / "errors.txt").write_text("\n\n".join(errors) if errors else "none", encoding="utf-8")
    log(f"[dump] seed={seed} done. checks: {json.dumps(checks)}")
    log(f"[dump] errors: {len(errors)}")


# ===========================================================================
# LATENCY
# ===========================================================================
def cmd_latency(args):
    torch = _torch()
    import joblib
    m6 = m6mod()
    m6._load_english_words()
    P = seed_paths(args.seed)
    gpu = torch.cuda.is_available()
    te = read_set("test")["domain"].tolist()
    rng = np.random.RandomState(0)
    pool = [te[i] for i in rng.permutation(len(te))]
    X_tr, _, _, y_tr, _, _, _ = m6.load_data_split(str(DATA_DIR))
    m6.build_markov_models([d for d, y in zip(X_tr, y_tr) if y == 0])

    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
    except Exception as e:
        log(f"   [info] NVML energy unavailable ({e}); pip install nvidia-ml-py to enable")
        nvml = None

    rows = []

    def bench(name, fn, device, batch_sizes, n_req):
        for bs in batch_sizes:
            reps = n_req.get(bs, 50)
            batches = [pool[(i * bs) % (len(pool) - bs):(i * bs) % (len(pool) - bs) + bs] for i in range(reps + 20)]
            for b in batches[:20]:
                fn(b)                              # warm-up
            if device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            e0 = pynvml_energy(nvml) if device == "cuda" else None
            lat = []
            t_all = time.perf_counter()
            for b in batches[20:]:
                t0 = time.perf_counter()
                fn(b)
                if device == "cuda":
                    torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000)
            wall = time.perf_counter() - t_all
            e1 = pynvml_energy(nvml) if device == "cuda" else None
            lat = np.array(lat)
            row = {"model": name, "device": device, "batch": bs, "n_batches": reps,
                   "p50_ms_per_batch": np.percentile(lat, 50), "p95_ms_per_batch": np.percentile(lat, 95),
                   "p99_ms_per_batch": np.percentile(lat, 99), "mean_ms_per_domain": lat.mean() / bs,
                   "throughput_domains_per_s": reps * bs / wall,
                   "peak_gpu_mem_MB": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
                   "gpu_energy_mJ_per_domain": ((e1 - e0) / (reps * bs)) if (e0 is not None and e1 is not None) else None}
            rows.append(row)
            log(f"   {name:<22} {device:<4} bs={bs:<4} p50={row['p50_ms_per_batch']:.2f}ms "
                f"p99={row['p99_ms_per_batch']:.2f}ms  {row['throughput_domains_per_s']:.0f} dom/s")
            pd.DataFrame(rows).to_csv(OUT / "latency.csv", sep=";", index=False)

    devices = (["cuda"] if gpu else []) + (["cpu"] if args.cpu else [])
    for device in devices:
        dev = torch.device(device)
        bl, _ = m6.load_bilstm(P["bilstm"])
        bl = bl.to(dev).eval()
        ck = torch.load(P["m6"] / "models" / "string_head.pt", map_location=dev, weights_only=False)
        head = m6.StringHead(ck["in_dim"], hidden=ck.get("hidden", 128), dropout=ck.get("dropout", 0.3)).to(dev)
        head.load_state_dict(ck["state_dict"]); head.eval()
        db, dtok, _ = m6.load_distilbert(P["distilbert"]); db = db.to(dev).eval()
        bt, btok, _ = m6.load_distilbert(P["bert"]); bt = bt.to(dev).eval()
        xgb = joblib.load(P["m6"] / "models" / "handcraft_xgb.joblib")
        sc = joblib.load(P["m6"] / "models" / "scaler.joblib")
        m3 = importlib.import_module("M3")
        ck3 = torch.load(P["M3"] / "model.pt", map_location=dev, weights_only=False)
        mm3 = m3.RCNN_SPP().to(dev); mm3.load_state_dict(ck3["state_dict"]); mm3.eval()

        def enc_char(b):
            ids = [[m6.CHAR_TO_ID.get(c, m6.PAD_ID) for c in d.strip().lower()][:m6.MAX_DOM_LEN] for d in b]
            return torch.tensor([x + [m6.PAD_ID] * (m6.MAX_DOM_LEN - len(x)) for x in ids], dtype=torch.long, device=dev)

        def bert_p(model, tok, b):
            enc = tok(b, truncation=True, padding="max_length", max_length=BERT_MAX_LEN, return_tensors="pt")
            return torch.softmax(model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)), 1)[:, 1]

        @torch.no_grad()
        def f_bilstm(b):
            return torch.softmax(bl(enc_char(b)), 1)[:, 1].cpu().numpy()

        @torch.no_grad()
        def f_db(b):
            return bert_p(db, dtok, b).cpu().numpy()

        @torch.no_grad()
        def f_bert(b):
            return bert_p(bt, btok, b).cpu().numpy()

        @torch.no_grad()
        def f_m3(b):
            return torch.softmax(mm3(m3.domains_to_tensor(b).to(dev)), 1)[:, 1].cpu().numpy()

        @torch.no_grad()
        def f_m6(b):
            ps = torch.softmax(head(bl(enc_char(b), return_embedding=True)), 1)[:, 1]
            pd_ = bert_p(db, dtok, b)
            return (0.5 * ps + 0.5 * pd_).cpu().numpy()

        @torch.no_grad()
        def f_m6_hc(b):
            p2 = f_m6(b)
            ph = xgb.predict_proba(sc.transform(m6.extract_handcrafted_matrix(b)))[:, 1]
            return 0.5 * p2 + 0.0 * ph     # handcraft computed but weight 0 (as the deployed config)

        n_req = {1: 2000, 32: 300, 256: 100} if device == "cuda" else {1: 300, 32: 60, 256: 15}
        bss = [1, 32, 256]
        for name, fn in [("BiLSTM", f_bilstm), ("M3_RCNN_SPP", f_m3), ("DistilBERT", f_db),
                         ("BERT", f_bert), ("M6_2branch", f_m6), ("M6_with_handcraft_branch", f_m6_hc)]:
            bench(name, fn, device, bss, n_req)
        del bl, head, db, bt, mm3
        if gpu:
            torch.cuda.empty_cache()

    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if gpu else None,
           "cpu_threads": torch.get_num_threads(), "python": sys.version.split()[0]}
    try:
        import platform
        env["cpu"] = platform.processor()
    except Exception:
        pass
    with open(OUT / "latency_env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    log("[latency] done -> rev_out/latency.csv")


def pynvml_energy(h):
    if h is None:
        return None
    try:
        import pynvml
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
    except Exception:
        return None


# ===========================================================================
# COLLECT
# ===========================================================================
def cmd_collect(args):
    meta = OUT / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    pats = ["out_select*/comparison.json", "out_select*/comparison_table.txt", "out_select*/winner.txt",
            "out_select*/*/metrics.json", "out_select*/*/summary.txt",
            "out_M6_v3_Markov_w2min/metrics.json", "out_M6_v3_Markov_w2min/summary.txt",
            "out_M6_v3_Markov/metrics.json", "out_M6_v3_Markov/summary.txt",
            "test_external/summary_*.txt", "test_external/per_family_*.csv"]
    n = 0
    for p in pats:
        for f in ROOT.glob(p):
            dst = meta / str(f.relative_to(ROOT)).replace(os.sep, "__").replace("/", "__")
            shutil.copy(f, dst); n += 1
    log(f"   copied {n} metadata files")
    zpath = Path("rev_upload.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT.rglob("*"):
            if f.is_file() and "cache" not in f.parts:
                z.write(f, f.relative_to(OUT.parent))
        if args.with_data:
            for base in [DATA_DIR, EXTERNAL_DIR]:
                for f in base.rglob("*"):
                    if f.is_file() and f.suffix in (".csv", ".txt"):
                        z.write(f, f)
    log(f"[collect] {zpath} = {zpath.stat().st_size / 2**20:.1f} MB")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--tranco"); p.add_argument("--umbrella")
    p = sub.add_parser("dump"); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("latency"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="also benchmark on CPU (slow)")
    p = sub.add_parser("collect"); p.add_argument("--with-data", action="store_true")
    a = ap.parse_args()
    {"prep": cmd_prep, "dump": cmd_dump, "latency": cmd_latency, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
