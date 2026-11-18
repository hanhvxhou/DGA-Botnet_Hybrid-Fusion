"""External-set analysis (Table 5): excludes the UTLDGA22 folder, removes exact overlaps,
reconciles family aliases, separates seen/unseen families.
usage: python ext_analysis.py <root containing rev_out> <analysis dir>"""
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd
RO = Path(sys.argv[1]) / "rev_out"; OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True, parents=True)
runs = pd.read_csv(OUT / "all_runs.csv", sep=";")
SEEDS = sorted(int(p.name[4:]) for p in RO.glob("seed*"))
ext = pd.read_csv(RO / "sets/ext.csv", sep=";", dtype=str, keep_default_na=False)
ext["in_dev"] = ext.in_dev.astype(int)
ext = ext[ext.dataset != "UTLDGA22"].reset_index(drop=True)
mask_all = None
dev_fams = sorted(pd.read_csv(RO / "test_family_map.csv", sep=";").family.unique())
ALIAS = {"cyyptolocker": "cryptolocker", "zeus_newgoz": "newgoz", "gameover": "newgoz", "pykspa2": "pykspa2", "pykspa2s": "pykspa2",
         "mydoomtid": "mydoom", "vidrotid": "vidro", "pushdotid": "pushdo", "tempedrevetdd": "tempedreve", "nymaim2": "nymaim2",
         "orchardgenesis": "orchard", "mirainomi": "mirai"}
def base(n):
    n = n.lower(); n = ALIAS.get(n, n)
    if n.startswith("pykspa") and n not in ("pykspa2",): return "pykspa"
    n = re.sub(r"_(v?\d+|improved.*|precursor|noise|gpl|luther|nasa|rfc4343)$", "", n)
    n = re.sub(r"(weekly)$", "", n)
    return n
dev_bases = {base(f) for f in dev_fams}
ext["base"] = ext.family.map(base)
ext["seen"] = ext.base.isin(dev_bases)
fam_tbl = ext.groupby(["dataset", "family"]).agg(base=("base", "first"), seen=("seen", "first"), n=("domain", "count"), in_dev=("in_dev", "sum")).reset_index()
fam_tbl.to_csv(OUT / "external_family_alias.csv", sep=";", index=False)
idx_keep = np.where(ext.index >= 0)[0]
# indices into original ext order (probs aligned to original incl UTLDGA22)
ext_full = pd.read_csv(RO / "sets/ext.csv", sep=";", dtype=str, keep_default_na=False)
orig_idx = np.where(ext_full.dataset.values != "UTLDGA22")[0]
models = ["M6", "bert", "distilbert", "string", "bilstm", "M3", "M4", "hc_orig"]
rows = []
for s in SEEDS:
    P = np.load(RO / f"seed{s}/probs.npz")
    for m in models:
        if m == "M6":
            r = runs[(runs.model == "M6_LF_search3_orig") & (runs.seed == s)].iloc[0]
            w = json.loads(r.weights); t = r.thr
            p = sum(w[k] * P[f"{k}|ext"] for k in w if w[k] > 0) / sum(v for v in w.values() if v > 0)
        else:
            t = runs[(runs.model == m) & (runs.seed == s) & (runs.thr_mode == "val")].thr.iloc[0]; p = P[f"{m}|ext"]
        for mode, tt in [("0.5", 0.5), ("val", t)]:
            pr = (p[orig_idx] >= tt)
            d = ext.assign(pr=pr)
            for subset, dd in [("all", d), ("novel_domains", d[d.in_dev == 0])]:
                g = dd.groupby(["dataset", "family"]).agg(n=("pr", "size"), dr=("pr", "mean"), seen=("seen", "first")).reset_index()
                for grp, gg in [("all", g), ("seen", g[g.seen]), ("unseen", g[~g.seen])]:
                    if len(gg) == 0: continue
                    rows.append(dict(model=m, seed=s, thr=mode, subset=subset, group=grp, n_fam=len(gg), n=int(gg.n.sum()),
                                     micro=float(np.average(gg.dr, weights=gg.n)), macro=float(gg.dr.mean()),
                                     worst=float(gg.dr.min()), n_lt50=int((gg.dr < .5).sum())))
                    for ds, g2 in gg.groupby("dataset"):
                        rows.append(dict(model=m, seed=s, thr=mode, subset=subset, group=f"{grp}|{ds}", n_fam=len(g2), n=int(g2.n.sum()),
                                         micro=float(np.average(g2.dr, weights=g2.n)), macro=float(g2.dr.mean()), worst=float(g2.dr.min()), n_lt50=int((g2.dr < .5).sum())))
                if m == "M6" and mode == "val" and subset == "novel_domains" and s == SEEDS[0]:
                    g.sort_values("dr").to_csv(OUT / "external_M6_perfamily_seed42_novel.csv", sep=";", index=False)
E = pd.DataFrame(rows); E.to_csv(OUT / "external_v2_runs.csv", sep=";", index=False)
def ms(x): x = np.asarray(x) * 100; return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
agg = E.groupby(["model", "thr", "subset", "group"]).agg(n_fam=("n_fam", "first"), n=("n", "first"), micro=("micro", ms), macro=("macro", ms),
                                                          worst=("worst", ms), fam_lt50=("n_lt50", "mean")).reset_index()
agg.to_csv(OUT / "external_v2_summary.csv", sep=";", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"external (excl. UTLDGA22): n={len(ext)}, in_dev={ext.in_dev.sum()}, families={len(fam_tbl)}, seen fams={fam_tbl.seen.sum()}, unique bases={ext.base.nunique()}, unique seen bases={ext[ext.seen].base.nunique()}")
print(agg[(agg.thr == "val") & (agg.subset == "novel_domains") & (agg.group.isin(["all", "seen", "unseen"]))].to_string(index=False))
print(agg[(agg.thr == "0.5") & (agg.subset == "all") & (agg.group.isin(["all", "seen", "unseen", "all|UMUDGA", "all|Netlab360", "all|Dgarchive"]))&(agg.model=="M6")].to_string(index=False))
