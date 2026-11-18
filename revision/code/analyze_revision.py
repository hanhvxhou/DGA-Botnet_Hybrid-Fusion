# -*- coding: utf-8 -*-
"""Offline analysis of rev_out/ (produced by revision_experiments.py).
usage: python analyze_revision.py <root containing rev_out> <out_dir>"""
import json, sys, math, itertools
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve

ROOT = Path(sys.argv[1]); RO = ROOT / "rev_out"; OUTD = Path(sys.argv[2]); OUTD.mkdir(parents=True, exist_ok=True)
REPORT = []
def rep(s=""): REPORT.append(s); print(s)

VOWELS = set("aeiou"); DIGITS = set("0123456789")
def entropy(s):
    if not s: return 0.0
    c = Counter(s); n = len(s); return -sum(v/n*math.log2(v/n) for v in c.values())
def split_domain(d):
    import re
    d = re.sub(r"^www\.", "", d.strip().lower()); p = d.split(".")
    return (p[0], "") if len(p) == 1 else (".".join(p[:-1]), p[-1])
def subtype(d):
    m, _ = split_domain(d); t = m.replace(".", ""); n = max(len(t), 1)
    e = entropy(t); vr = sum(c in VOWELS for c in t)/n; dr = sum(c in DIGITS for c in t)/n
    if len(t) >= 12 and vr >= 0.30 and e <= 3.6: return "DGA-W"
    if e >= 3.8 or dr >= 0.15: return "DGA-C"
    if 0.20 <= vr <= 0.45 and e < 3.8: return "DGA-P"
    return "DGA-C"

def rd(name): return pd.read_csv(RO/"sets"/f"{name}.csv", sep=";", dtype=str, keep_default_na=False).assign(label=lambda d: d.label.astype(int))
SETS = {p.stem: rd(p.stem) for p in sorted((RO/"sets").glob("*.csv"))}
y = {s: df.label.values for s, df in SETS.items()}
test = SETS["test"]; yv, yt = y["val"], y["test"]
types = np.array(["benign" if l == 0 else subtype(d) for d, l in zip(test.domain, test.label)])
fam = test["family"].values.copy() if "family" in test else np.array(["unknown"]*len(test), dtype=object)
_fm = RO/"test_family_map.csv"
if _fm.exists():
    _m = dict(pd.read_csv(_fm, sep=";", dtype=str).values)
    fam = np.array([("benign" if l == 0 else _m.get(d, f if f not in ("unknown", "?") else "unknown")) for d, l, f in zip(test.domain, test.label, fam)], dtype=object)
SEEDS = sorted(int(p.name[4:]) for p in RO.glob("seed*") if (p/"probs.npz").exists())
PR = {s: dict(np.load(RO/f"seed{s}"/"probs.npz")) for s in SEEDS}
rep(f"# Revision analysis  seeds={SEEDS}  sets={list(SETS)}")
rep(f"test subtype counts: {dict(Counter(types))}")

def _f1(yy, pr):
    yy = np.asarray(yy); pr = np.asarray(pr)
    tp = np.sum(pr & (yy == 1)); fp = np.sum(pr & (yy == 0)); fn = np.sum((~pr) & (yy == 1))
    return 2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0
def thr_grid(p, yy, lo=0.30, hi=0.70, step=0.005):
    best_t, best = 0.5, -1
    for t in np.arange(lo, hi + 1e-9, step):
        f = _f1(yy, p >= t)
        if f > best: best, best_t = f, float(t)
    return best_t
def metrics(yy, p, t):
    pr = (p >= t).astype(int)
    out = dict(acc=accuracy_score(yy, pr), pre=precision_score(yy, pr, zero_division=0),
               rec=recall_score(yy, pr, zero_division=0), f1=f1_score(yy, pr, zero_division=0),
               auc=roc_auc_score(yy, p) if len(set(yy)) > 1 else np.nan)
    for st in ["DGA-C", "DGA-P", "DGA-W"]:
        m = (types == "benign") | (types == st)
        out[f"f1_{st}"] = f1_score(yy[m], pr[m], zero_division=0)
        out[f"rec_{st}"] = recall_score(yy[types == st], pr[types == st], zero_division=0)
    return out
def wprob(P, w):
    used = {k: v for k, v in w.items() if v > 0}; s = sum(used.values())
    return sum(v/s*P[k] for k, v in used.items())
def grid_weights(Pv, yy, names, w_min=0.0, thr=0.5):
    def ok(c):
        if w_min <= 0: return True
        s = sum(v for v in c if v > 0)
        return s > 0 and all((v/s if v > 0 else 0) >= w_min - 1e-9 for v in c)
    vals = [round(x*0.1, 3) for x in range(11)]
    best, bw = -1, None
    for c in itertools.product(vals, repeat=len(names)):
        if sum(c) == 0 or not ok(c): continue
        w = dict(zip(names, c)); f = _f1(yy, wprob(Pv, w) >= thr)
        if f > best: best, bw = f, w
    grids = {}
    for n in names:
        cand = set()
        for d in [-0.1, -0.05, 0, 0.05, 0.1]:
            v = max(0, min(1, bw[n] + d)); cand.add(round(round(v/0.05)*0.05, 3))
        grids[n] = sorted(cand)
    for c in itertools.product(*[grids[n] for n in names]):
        if sum(c) == 0 or not ok(c): continue
        w = dict(zip(names, c)); f = _f1(yy, wprob(Pv, w) >= thr)
        if f > best: best, bw = f, w
    return bw
def norm(w):
    s = sum(v for v in w.values() if v > 0); return {k: round(v/s, 3) if v > 0 else 0 for k, v in w.items()}

# ---------------------------------------------------------------- models & variants
BASE = [m for m in ["M1","M2","M3","M4","M5","bilstm","string","distilbert","bert","distilbert_rand","hc_orig","hc_eos","ef_mlp2","ef_xgb2","ef_xgb3"]
        if all(f"{m}|test" in PR[s] for s in SEEDS)]
rows, FUSED = [], defaultdict(dict)   # FUSED[variant][seed] = dict(set -> prob), thr, w
for s in SEEDS:
    P = PR[s]
    get = lambda m, st: P[f"{m}|{st}"]
    for m in BASE:
        for mode in ["0.5", "val"]:
            t = 0.5 if mode == "0.5" else thr_grid(get(m, "val"), yv)
            r = metrics(yt, get(m, "test"), t); r.update(model=m, seed=s, thr_mode=mode, thr=t); rows.append(r)
    variants = {"M6_LF_search3_orig": (["string","distilbert","hc_orig"], 0.0),
                "M6_LF_search3_eos": (["string","distilbert","hc_eos"], 0.0),
                **{f"LF3_wmin{wm:.2f}_orig": (["string","distilbert","hc_orig"], wm) for wm in [0.05, 0.10, 0.15, 0.20]},
                **{f"LF3_wmin{wm:.2f}_eos": (["string","distilbert","hc_eos"], wm) for wm in [0.05, 0.10, 0.20]},
                "LF2_string_bert": (["string","bert"], 0.0)}
    fixed = {"LF2_equal": {"string": .5, "distilbert": .5}, "LF3_equal_orig": {"string": 1/3, "distilbert": 1/3, "hc_orig": 1/3},
             "LF2_bilstmnative_distilbert": {"bilstm": .5, "distilbert": .5}}
    for v, (names, wm) in variants.items():
        if not all(f"{n}|val" in P for n in names): continue
        Pv = {n: get(n, "val") for n in names}; w = grid_weights(Pv, yv, names, wm)
        fixed[v] = w
    for v, w in fixed.items():
        if not all(f"{n}|val" in P for n in w): continue
        pv = wprob({n: get(n, "val") for n in w}, w); t = thr_grid(pv, yv)
        FUSED[v][s] = dict(w=norm(w), thr=t, P={st: wprob({n: get(n, st) for n in w}, w) for st in SETS if all(f"{n}|{st}" in P for n in w)})
        r = metrics(yt, FUSED[v][s]["P"]["test"], t); r.update(model=v, seed=s, thr_mode="val", thr=t, weights=json.dumps(norm(w))); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(OUTD/"all_runs.csv", sep=";", index=False)

def ms(x, pct=True, d=2):
    x = np.asarray(x, float); k = 100 if pct else 1
    return f"{x.mean()*k:.{d}f} ± {x.std(ddof=1)*k:.{d}f}" if len(x) > 1 else f"{x.mean()*k:.{d}f}"
rep("\n## A. Test performance (mean ± sd over seeds); thr_mode=val -> threshold tuned on validation (0.30-0.70, step 0.005)")
summ = []
for (m, mode), g in R.groupby(["model", "thr_mode"], sort=False):
    summ.append(dict(model=m, thr=mode, n=len(g), F1=ms(g.f1), Acc=ms(g.acc), Pre=ms(g.pre), Rec=ms(g.rec),
                     AUC=ms(g.auc, False, 4), C=ms(g["f1_DGA-C"]), P=ms(g["f1_DGA-P"]), W=ms(g["f1_DGA-W"]),
                     thr_mean=f"{g.thr.mean():.3f}", weights=g.weights.iloc[0] if "weights" in g and isinstance(g.weights.iloc[0], str) else ""))
S = pd.DataFrame(summ); S.to_csv(OUTD/"summary_table.csv", sep=";", index=False); rep(S.to_string(index=False))

# ---------------------------------------------------------------- reproduction check
rep("\n## B. Reproduction of the original M6 selection from dumped probabilities")
for s in SEEDS:
    f = RO/f"seed{s}"/"m6_metrics.json"
    sel = FUSED["M6_LF_search3_orig"].get(s)
    msg = f"seed {s}: re-searched weights={sel['w'] if sel else None} thr={sel['thr'] if sel else None}"
    if f.exists():
        mj = json.load(open(f, encoding="utf-8")); msg += f" | saved test F1={mj.get('test',{}).get('f1')}"
    if sel: msg += f" | recomputed F1={f1_score(yt,(sel['P']['test']>=sel['thr']).astype(int)):.5f}"
    rep(msg)
    ck = RO/f"seed{s}"/"checks.json"
    if ck.exists(): rep(f"   checks: {json.load(open(ck))}")

# ---------------------------------------------------------------- paired comparisons
REF = "M6_LF_search3_orig"
def preds(model, s, st="test"):
    if model in FUSED: return (FUSED[model][s]["P"][st] >= FUSED[model][s]["thr"]).astype(int)
    t = R[(R.model == model) & (R.seed == s) & (R.thr_mode == "val")].thr.iloc[0]
    return (PR[s][f"{model}|{st}"] >= t).astype(int)
def f1fast(yy, pr):
    tp = np.sum((pr == 1) & (yy == 1)); fp = np.sum((pr == 1) & (yy == 0)); fn = np.sum((pr == 0) & (yy == 1))
    return 2*tp/(2*tp+fp+fn)
rng = np.random.default_rng(0)
dga_idx = {f: np.where((fam == f) & (yt == 1))[0] for f in set(fam[yt == 1])}
ben_idx = np.where(yt == 0)[0]; fams = sorted(dga_idx)
def boot_ci(yy, a, b, B=10000, cluster=False):
    n = len(yy); d = np.empty(B)
    for i in range(B):
        if cluster:
            fs = rng.choice(fams, size=len(fams), replace=True)
            idx = np.concatenate([dga_idx[f] for f in fs] + [rng.choice(ben_idx, size=len(ben_idx), replace=True)])
        else:
            idx = rng.integers(0, n, n)
        d[i] = f1fast(yy[idx], a[idx]) - f1fast(yy[idx], b[idx])
    return np.percentile(d, [2.5, 97.5])
rep(f"\n## C. Paired comparison {REF} vs others (val-tuned thresholds for all)")
cmp_rows = []
others = [m for m in ["bert","distilbert","bilstm","string","M3","M4","M2","M5","M1","ef_mlp2","ef_xgb2","ef_xgb3","LF2_string_bert"] if m in BASE or m in FUSED]
for m in others:
    d = []
    for s in SEEDS:
        a, b = preds(REF, s), preds(m, s); d.append(f1fast(yt, a) - f1fast(yt, b))
    s0 = SEEDS[0]; a, b = preds(REF, s0), preds(m, s0)
    ci = boot_ci(yt, a, b, B=2000); cci = boot_ci(yt, a, b, B=2000, cluster=True) if len(fams) > 1 else [np.nan, np.nan]
    n01 = np.sum((a == yt) & (b != yt)); n10 = np.sum((a != yt) & (b == yt))
    mc = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    tt = stats.ttest_rel([f1fast(yt, preds(REF, s)) for s in SEEDS], [f1fast(yt, preds(m, s)) for s in SEEDS]).pvalue if len(SEEDS) > 1 else np.nan
    cmp_rows.append(dict(model=m, dF1_per_seed=" ".join(f"{x*100:+.2f}" for x in d), dF1_mean=f"{np.mean(d)*100:+.2f}",
                         t_p=f"{tt:.2e}", boot95_seed0=f"[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]",
                         famcluster95_seed0=f"[{cci[0]*100:+.2f},{cci[1]*100:+.2f}]", mcnemar_p_seed0=f"{mc:.2e}",
                         n_only_M6_correct=int(n01), n_only_other_correct=int(n10)))
C = pd.DataFrame(cmp_rows); C.to_csv(OUTD/"paired.csv", sep=";", index=False); rep(C.to_string(index=False))

# ---------------------------------------------------------------- calibration
def ece(yy, p, nb=15):
    b = np.minimum((p*nb).astype(int), nb-1); e = 0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean()*abs(p[m].mean() - yy[m].mean())
    return e
rep("\n## D. Calibration on test (ECE 15 equal-width bins, Brier); mean over seeds")
cal = []
for m in [x for x in ["string","bilstm","distilbert","bert","hc_orig","M3"] if x in BASE] + [REF]:
    E, Bs = [], []
    for s in SEEDS:
        p = FUSED[m][s]["P"]["test"] if m in FUSED else PR[s][f"{m}|test"]
        E.append(ece(yt, p)); Bs.append(np.mean((p - yt)**2))
    cal.append(dict(model=m, ECE=ms(E, False, 4), Brier=ms(Bs, False, 4)))
# temperature scaling of the two branches on val -> does the fused decision change?
def logit(p): p = np.clip(p, 1e-7, 1-1e-7); return np.log(p/(1-p))
def fitT(p, yy):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.1), np.log(10), 200)):
        q = 1/(1+np.exp(-logit(p)/T)); nll = -np.mean(yy*np.log(q+1e-12) + (1-yy)*np.log(1-q+1e-12))
        best = min(best, (nll, T))
    return best[1]
ts = []
for s in SEEDS:
    P = PR[s]; Ts = {n: fitT(P[f"{n}|val"], yv) for n in ["string","distilbert"]}
    cal_t = {n: 1/(1+np.exp(-logit(P[f"{n}|{st}"])/Ts[n])) for n in Ts for st in ["val"]}
    pv = 0.5*(1/(1+np.exp(-logit(P["string|val"])/Ts["string"]))) + 0.5*(1/(1+np.exp(-logit(P["distilbert|val"])/Ts["distilbert"])))
    pt = 0.5*(1/(1+np.exp(-logit(P["string|test"])/Ts["string"]))) + 0.5*(1/(1+np.exp(-logit(P["distilbert|test"])/Ts["distilbert"])))
    t = thr_grid(pv, yv); f_ts = f1fast(yt, (pt >= t).astype(int)); f0 = f1fast(yt, preds("LF2_equal", s))
    ts.append(dict(seed=s, T_string=round(Ts["string"], 3), T_distilbert=round(Ts["distilbert"], 3), F1_fused_raw=round(f0*100, 3), F1_fused_tempscaled=round(f_ts*100, 3),
                   ECE_fused_tempscaled=round(ece(yt, pt), 4)))
rep(pd.DataFrame(cal).to_string(index=False)); rep(pd.DataFrame(ts).to_string(index=False))
pd.DataFrame(cal).to_csv(OUTD/"calibration.csv", sep=";", index=False); pd.DataFrame(ts).to_csv(OUTD/"temperature_scaling.csv", sep=";", index=False)

# ---------------------------------------------------------------- disagreement / per family
rep("\n## E. Branch disagreement and per-family effect (string vs distilbert vs fused), mean over seeds")
fam_rows = []; dis_rows = []
for s in SEEDS:
    a = preds("string", s); b = preds("distilbert", s); f = preds(REF, s); bt = preds("bert", s) if "bert" in BASE else None
    dis = a != b
    dis_rows.append(dict(seed=s, disagree=int(dis.sum()), pct=dis.mean()*100, string_right=int((dis & (a == yt)).sum()),
                         distil_right=int((dis & (b == yt)).sum()), fused_right_on_dis=int((dis & (f == yt)).sum()),
                         err_string=int((a != yt).sum()), err_distil=int((b != yt).sum()), err_fused=int((f != yt).sum()),
                         both_wrong=int(((a != yt) & (b != yt)).sum()), both_right_fused_wrong=int(((a == yt) & (b == yt) & (f != yt)).sum()),
                         both_wrong_fused_right=int(((a != yt) & (b != yt) & (f == yt)).sum())))
    for g in sorted(set(fam)):
        m = fam == g
        fam_rows.append(dict(seed=s, family=g, n=int(m.sum()), subtype=Counter(types[m]).most_common(1)[0][0],
                             err_string=int((a[m] != yt[m]).sum()), err_distil=int((b[m] != yt[m]).sum()),
                             err_bert=int((bt[m] != yt[m]).sum()) if bt is not None else np.nan, err_fused=int((f[m] != yt[m]).sum()),
                             disagree=int(dis[m].sum())))
D = pd.DataFrame(dis_rows); rep(D.to_string(index=False)); D.to_csv(OUTD/"disagreement.csv", sep=";", index=False)
F = pd.DataFrame(fam_rows).groupby(["family"]).agg(n=("n","first"), subtype=("subtype","first"), err_string=("err_string","mean"),
        err_distil=("err_distil","mean"), err_bert=("err_bert","mean"), err_fused=("err_fused","mean"), disagree=("disagree","mean")).reset_index()
F["gain_vs_distil"] = F.err_distil - F.err_fused; F["gain_vs_bert"] = F.err_bert - F.err_fused
F.to_csv(OUTD/"per_family_errors.csv", sep=";", index=False)
dg = F[F.family != "benign"]
for col in ["gain_vs_distil", "gain_vs_bert"]:
    tot = dg[col].sum(); pos = dg[dg[col] > 0][col].sort_values(ascending=False)
    rep(f"{col}: total={tot:.1f} errors; families improved={int((dg[col]>0).sum())}, worse={int((dg[col]<0).sum())}, equal={int((dg[col]==0).sum())} of {len(dg)}; "
        f"share of positive gain from top-5 families={pos.head(5).sum()/max(pos.sum(),1e-9)*100:.1f}%; benign gain={float(F[F.family=='benign'][col].sum()):.1f}")

# ---------------------------------------------------------------- robustness
rep("\n## F. Suffix / parsing robustness (F1 on transformed test set, val-tuned thresholds), mean over seeds")
rob = []
for m in [x for x in ["bilstm","string","distilbert","bert","M3","hc_orig"] if x in BASE] + [REF]:
    r = dict(model=m)
    for st in ["test","test_tldcom","test_notld","test_www"]:
        if st not in SETS: continue
        r[st] = ms([f1fast(y[st], preds(m, s, st)) for s in SEEDS])
    rob.append(r)
rep(pd.DataFrame(rob).to_string(index=False)); pd.DataFrame(rob).to_csv(OUTD/"robustness.csv", sep=";", index=False)

# ---------------------------------------------------------------- external
if "ext" in SETS:
    rep("\n## G. External DGA sets: per-family detection rate; macro = equal weight per family")
    ext = SETS["ext"]; ext_rows = []
    for m in [x for x in ["bilstm","string","distilbert","bert","M3","M4","hc_orig","ef_mlp2"] if x in BASE] + [REF]:
        for s in SEEDS:
            for mode in ["0.5", "val"]:
                if m in FUSED: p = FUSED[m][s]["P"]["ext"]; t = 0.5 if mode == "0.5" else FUSED[m][s]["thr"]
                else: p = PR[s][f"{m}|ext"]; t = 0.5 if mode == "0.5" else R[(R.model==m)&(R.seed==s)&(R.thr_mode=="val")].thr.iloc[0]
                pr = (p >= t)
                for (ds, fm), g in ext.assign(pr=pr).groupby(["dataset","family"]):
                    ext_rows.append(dict(model=m, seed=s, thr=mode, dataset=ds, family=fm, n=len(g), dr=g.pr.mean()))
    E = pd.DataFrame(ext_rows); E.to_csv(OUTD/"external_per_family.csv", sep=";", index=False)
    agg = []
    for (m, mode), g in E.groupby(["model","thr"], sort=False):
        per_seed = g.groupby("seed").apply(lambda h: pd.Series(dict(micro=np.average(h.dr, weights=h.n), macro=h.dr.mean(),
                    worst=h.dr.min(), p10=np.percentile(h.dr, 10), n_lt50=(h.dr < .5).sum())))
        agg.append(dict(model=m, thr=mode, micro=ms(per_seed.micro), macro=ms(per_seed.macro), worst=ms(per_seed.worst),
                        p10=ms(per_seed.p10), fam_lt50=f"{per_seed.n_lt50.mean():.1f}"))
    rep(pd.DataFrame(agg).to_string(index=False)); pd.DataFrame(agg).to_csv(OUTD/"external_summary.csv", sep=";", index=False)

# ---------------------------------------------------------------- independent benign
bsets = [b for b in ["ben_tranco","ben_umb_reg","ben_umb_fqdn"] if b in SETS]
if bsets:
    rep("\n## H. Independent benign sets: FPR at val threshold; recall (DataNew test DGA) at fixed FPR")
    br = []
    for m in [x for x in ["bilstm","string","distilbert","bert","M3","hc_orig"] if x in BASE] + [REF]:
        for b in bsets:
            vals = defaultdict(list)
            for s in SEEDS:
                if m in FUSED: pb = FUSED[m][s]["P"][b]; pp = FUSED[m][s]["P"]["test"][yt == 1]; t = FUSED[m][s]["thr"]
                else: pb = PR[s][f"{m}|{b}"]; pp = PR[s][f"{m}|test"][yt == 1]; t = R[(R.model==m)&(R.seed==s)&(R.thr_mode=="val")].thr.iloc[0]
                vals["fpr"].append((pb >= t).mean())
                for target in [1e-3, 1e-2]:
                    tq = np.quantile(pb, 1 - target); vals[f"rec@{target}"].append((pp > tq).mean())
            br.append(dict(model=m, benign_set=b, FPR_at_val_thr=ms(vals["fpr"], True, 2), alerts_per_1M=f"{np.mean(vals['fpr'])*1e6:,.0f}",
                           recall_at_FPR_0p1=ms(vals["rec@0.001"]), recall_at_FPR_1=ms(vals["rec@0.01"])))
    rep(pd.DataFrame(br).to_string(index=False)); pd.DataFrame(br).to_csv(OUTD/"benign_independent.csv", sep=";", index=False)

# ---------------------------------------------------------------- copies
for f in ["tokenization_stats.csv","tokenization_examples.csv","feature_examples.csv","markov_spec.json","external_overlap.csv","latency.csv","latency_env.json"]:
    if (RO/f).exists():
        rep(f"\n## file {f}"); rep((RO/f).read_text(encoding="utf-8")[:6000])
(OUTD/"report.md").write_text("\n".join(REPORT), encoding="utf-8")
