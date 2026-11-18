# -*- coding: utf-8 -*-
"""make_family_map.py - gan ho DGA cho tung domain trong DataNew/test.csv
(thay cho thu muc DataNew/Test bi thieu). Chi doc file, khong ghi de gi.

Usage (chay o thu muc goc du an, co the chay song song khi run_revision.ps1 dang chay):
    python make_family_map.py
    python make_family_map.py --dga-dir "duong\\dan\\toi\\DGA_Botnets_Domains"
Output: rev_out/test_family_map.csv  (domain;family)
"""
import argparse, re
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dga-dir", type=Path, default=None)
a = ap.parse_args()

cands = [a.dga_dir] if a.dga_dir else [
    Path("data/raw/UTL_DGA22/DGA_Botnets_Domains"), Path("raw/UTL_DGA22/DGA_Botnets_Domains"),
    Path("UTL_DGA22/DGA_Botnets_Domains"), Path("DGA_Botnets_Domains"), Path("../UTL_DGA22/DGA_Botnets_Domains")]
dga_dir = next((p for p in cands if p and p.exists()), None)
if dga_dir is None:
    hits = [p for p in Path(".").rglob("DGA_Botnets_Domains") if p.is_dir()]
    dga_dir = hits[0] if hits else None
if dga_dir is None:
    raise SystemExit("Khong tim thay thu muc DGA_Botnets_Domains. Chay lai voi --dga-dir <duong dan>.")
print(f"DGA dir: {dga_dir}")

te = pd.read_csv("DataNew/test.csv", sep=None, engine="python", dtype=str)
dcol = [c for c in te.columns if "domain" in c.lower()][0]
lcol = [c for c in te.columns if "label" in c.lower()][0]
test_dga = set(te.loc[te[lcol].astype(str).isin(["1", "dga", "DGA"]), dcol].str.strip().str.lower())
print(f"test DGA domains: {len(test_dga):,}")

fam, multi = {}, 0
for f in sorted(dga_dir.rglob("*.txt")):
    name = re.sub(r"-\d+", "", f.stem)
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        d = line.strip().lower()
        if d in test_dga:
            if d in fam and fam[d] != name:
                multi += 1
            fam.setdefault(d, name)
out = pd.DataFrame(sorted(fam.items()), columns=["domain", "family"])
Path("rev_out").mkdir(exist_ok=True)
out.to_csv("rev_out/test_family_map.csv", sep=";", index=False)
print(f"mapped {len(out):,}/{len(test_dga):,}  families={out.family.nunique()}  "
      f"domains in >1 family file={multi}  -> rev_out/test_family_map.csv")
