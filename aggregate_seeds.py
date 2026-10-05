"""
aggregate_seeds.py — Gộp kết quả multi-seed thành bảng mean ± std + bootstrap CI.

Dùng để trả lời Reviewer 1 yêu cầu:
  "assess stability across independently seeded training runs"
  "Report paired confidence intervals for the differences"

Usage:
    cd D:\\job\\pycharm\\DGABotnet2k25\\DGA-HLF
    python aggregate_seeds.py

Expected input (sau khi chạy run_multiseed.ps1):
    out_M1_seed{42..46}/metrics.json
    out_M2_seed{42..46}/metrics.json
    out_M3_seed{42..46}/metrics.json
    out_M4_seed{42..46}/metrics.json
    out_M5_seed{42..46}/metrics.json
    out_M6_v3_Markov_seed{42..46}/metrics.json

Output:
    aggregate_results/
        table_binary_mean_std.csv    — Table 1 revised với mean ± std
        table_per_family_mean_std.csv — Table 2 revised với mean ± std
        paired_ci_m6_vs_others.txt   — Bootstrap CI for F1 differences
        aggregated_summary.md        — Markdown summary để copy vào response letter
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["M1", "M2", "M3", "M4", "M5", "M6_v3_Markov"]
MODEL_DISPLAY = {
    "M1": "M1 (RF + n-gram + PCA)",
    "M2": "M2 (TF-IDF + MLP)",
    "M3": "M3 (RCNN-SPP)",
    "M4": "M4 (ATT-CNN-BiLSTM)",
    "M5": "M5 (Dual Transformer)",
    "M6_v3_Markov": "M6 Hybrid Late Fusion",
}
OUT_ROOT = Path("aggregate_results")
OUT_ROOT.mkdir(exist_ok=True)


def load_metrics(model: str, seed: int) -> dict | None:
    """Load metrics.json for (model, seed). Return None if missing."""
    path = Path(f"out_{model}_seed{seed}") / "metrics.json"
    if not path.exists():
        print(f"  [MISS] {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_binary(m: dict) -> dict:
    """Extract binary metrics from M1-M5 structure."""
    t = m.get("test", {})
    return {
        "accuracy": t.get("accuracy", np.nan),
        "precision": t.get("precision", np.nan),
        "recall": t.get("recall", np.nan),
        "f1": t.get("f1", np.nan),
        "auc": t.get("auc", np.nan),
    }


def extract_per_family(m: dict) -> dict:
    """Extract per-family F1 from M1-M5 structure."""
    pf = m.get("per_family", {})
    out = {}
    for key in ["DGA-R", "DGA-P", "DGA-W", "DGA-C"]:
        if key in pf:
            out[key] = pf[key].get("f1", np.nan)
    # Normalise DGA-R → DGA-C mapping if needed
    if "DGA-R" in out and "DGA-C" not in out:
        out["DGA-C"] = out.pop("DGA-R")
    return out


def extract_predictions(model: str, seed: int) -> pd.DataFrame | None:
    """Load test_predictions.csv for paired bootstrap."""
    path = Path(f"out_{model}_seed{seed}") / "test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=";")


def compute_mean_std(values: list) -> tuple:
    """Return (mean, std) ignoring NaN."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0)


def fmt(mean: float, std: float, pct: bool = True) -> str:
    """Format mean ± std for publication."""
    if np.isnan(mean):
        return "—"
    if pct:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def bootstrap_paired_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000,
                         ci: float = 0.95, seed: int = 42) -> dict:
    """
    Paired bootstrap confidence interval for F1 difference (x - y).
    x, y: binary correct/wrong arrays of same length (same test samples).
    Returns dict with mean_diff, ci_low, ci_high, p_value.
    """
    assert len(x) == len(y), "Arrays must be same length"
    rng = np.random.default_rng(seed)
    n = len(x)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        f1x = 2 * xb.sum() / (2 * xb.sum() + (n - xb.sum()) + (n - yb.sum()) + 1e-9)
        f1y = 2 * yb.sum() / (2 * yb.sum() + (n - xb.sum()) + (n - yb.sum()) + 1e-9)
        diffs.append(f1x - f1y)
    diffs = np.array(diffs)
    alpha = 1 - ci
    return {
        "mean_diff": diffs.mean(),
        "ci_low": np.quantile(diffs, alpha / 2),
        "ci_high": np.quantile(diffs, 1 - alpha / 2),
        "p_value_sign_flip": (diffs <= 0).mean() if diffs.mean() > 0 else (diffs >= 0).mean(),
    }


def main():
    print("=" * 70)
    print("Aggregate multi-seed results")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODELS}\n")

    # --- 1. Load all metrics ---
    all_metrics: dict[str, dict[int, dict]] = {}
    for model in MODELS:
        all_metrics[model] = {}
        for seed in SEEDS:
            m = load_metrics(model, seed)
            if m is not None:
                all_metrics[model][seed] = m

    # --- 2. Binary metrics table ---
    print("\n[1/4] Building binary metrics table (mean ± std)...")
    rows = []
    for model in MODELS:
        if not all_metrics[model]:
            print(f"  [SKIP] No data for {model}")
            continue
        binaries = [extract_binary(all_metrics[model][s]) for s in SEEDS if s in all_metrics[model]]
        n_seeds = len(binaries)
        acc = compute_mean_std([b["accuracy"] for b in binaries])
        prec = compute_mean_std([b["precision"] for b in binaries])
        rec = compute_mean_std([b["recall"] for b in binaries])
        f1 = compute_mean_std([b["f1"] for b in binaries])
        auc = compute_mean_std([b["auc"] for b in binaries])
        rows.append({
            "Model": MODEL_DISPLAY[model],
            "n_seeds": n_seeds,
            "Acc (%)": fmt(*acc),
            "Pre (%)": fmt(*prec),
            "Rec (%)": fmt(*rec),
            "F1 (%)": fmt(*f1),
            "AUC-ROC": fmt(*auc, pct=False),
        })
    df_bin = pd.DataFrame(rows)
    df_bin.to_csv(OUT_ROOT / "table_binary_mean_std.csv", index=False)
    print(df_bin.to_string(index=False))

    # --- 3. Per-family F1 table ---
    print("\n[2/4] Building per-family F1 table (mean ± std)...")
    rows_pf = []
    for model in MODELS:
        if not all_metrics[model]:
            continue
        pfs = [extract_per_family(all_metrics[model][s]) for s in SEEDS if s in all_metrics[model]]
        dgac = compute_mean_std([pf.get("DGA-C", pf.get("DGA-R", np.nan)) for pf in pfs])
        dgap = compute_mean_std([pf.get("DGA-P", np.nan) for pf in pfs])
        dgaw = compute_mean_std([pf.get("DGA-W", np.nan) for pf in pfs])
        rows_pf.append({
            "Model": MODEL_DISPLAY[model],
            "DGA-C F1 (%)": fmt(*dgac),
            "DGA-P F1 (%)": fmt(*dgap),
            "DGA-W F1 (%)": fmt(*dgaw),
        })
    df_pf = pd.DataFrame(rows_pf)
    df_pf.to_csv(OUT_ROOT / "table_per_family_mean_std.csv", index=False)
    print(df_pf.to_string(index=False))

    # --- 4. Paired bootstrap CI for M6 vs each baseline ---
    print("\n[3/4] Computing paired bootstrap CI (M6 vs baselines, seed 42 only)...")
    lines = ["Paired bootstrap 95% CI for F1 difference (M6 − baseline)",
             "Based on test set predictions of seed 42 (same test samples for all).",
             "", f"{'Baseline':<25} {'Mean diff':>12} {'95% CI':>25} {'p(sign-flip)':>14}",
             "-" * 80]

    m6_pred = extract_predictions("M6_v3_Markov", 42)
    if m6_pred is None:
        print("  [WARN] No M6 predictions found for seed 42. Skipping CI.")
    else:
        y_true = m6_pred["true_label"].values
        m6_correct = (m6_pred["pred_label"].values == y_true).astype(int)

        for model in ["M1", "M2", "M3", "M4", "M5"]:
            pred = extract_predictions(model, 42)
            if pred is None:
                continue
            # Ensure same ordering (both sorted by domain)
            pred = pred.sort_values("domain").reset_index(drop=True)
            m6_pred_sorted = m6_pred.sort_values("domain").reset_index(drop=True)
            y_true2 = m6_pred_sorted["true_label"].values
            m6_correct2 = (m6_pred_sorted["pred_label"].values == y_true2).astype(int)
            base_correct = (pred["pred_label"].values == pred["true_label"].values).astype(int)

            if len(base_correct) != len(m6_correct2):
                print(f"  [WARN] {model}: length mismatch, skipping")
                continue

            ci = bootstrap_paired_ci(m6_correct2, base_correct, n_boot=1000, seed=42)
            line = (f"{MODEL_DISPLAY[model]:<25} "
                    f"{ci['mean_diff']*100:>+10.3f}pp  "
                    f"[{ci['ci_low']*100:>+6.3f}, {ci['ci_high']*100:>+6.3f}]pp  "
                    f"{ci['p_value_sign_flip']:>13.4f}")
            lines.append(line)
            print(line)

    with open(OUT_ROOT / "paired_ci_m6_vs_others.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- 5. Markdown summary for response letter ---
    print("\n[4/4] Writing aggregated_summary.md...")
    md = []
    md.append("# Multi-seed aggregated results (5 seeds: 42-46)\n")
    md.append("## Binary detection performance\n")
    md.append(df_bin.to_markdown(index=False))
    md.append("\n\n## Per-family F1 score\n")
    md.append(df_pf.to_markdown(index=False))
    md.append("\n\n## Paired bootstrap 95% CI (M6 - baseline, seed 42)\n```")
    md.append("\n".join(lines))
    md.append("```")
    with open(OUT_ROOT / "aggregated_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[DONE] Outputs saved to: {OUT_ROOT.absolute()}")
    print("  - table_binary_mean_std.csv")
    print("  - table_per_family_mean_std.csv")
    print("  - paired_ci_m6_vs_others.txt")
    print("  - aggregated_summary.md  ← paste this into response letter")


if __name__ == "__main__":
    main()
