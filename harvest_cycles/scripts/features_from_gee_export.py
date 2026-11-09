#!/usr/bin/env python3
"""
Recompute the harvest-cycle features from the Earth Engine export of
harvest_cycles/gee/landtrendr_harvest_features.js (MODE = 'EXPORT'), with exactly the same code as
the offline analysis (harvest_lib.py), so GEE-based and offline results can be compared.

Usage:
  python features_from_gee_export.py harvest_features_val_2020.csv [harvest_features_qc_2020.csv] --out features_gee.csv

The export holds surface-reflectance NBR (nbr_1985 ... nbr_2020), which is higher than the
top-of-atmosphere NBR used offline, so the closed-canopy / open-land levels are re-calibrated here,
label-free, on the exported rows that Hansen calls closed canopy (tree cover >= 90% in 2000, no loss
2001-2020, no gain), using their 2000-2020 values: closed canopy = 5th percentile. Open land = 25th percentile of the
1999-2001 values of rows with <= 10% tree cover in 2000 and no gain, at most closed canopy minus half a cut.
The LandTrendr features computed inside Earth Engine are kept with a 'gee_' prefix.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--out", default="features_gee.csv")
ap.add_argument("--drop", type=float, default=0.20)
args = ap.parse_args()

frames = []
for f in args.csv:
    d = pd.read_csv(f)
    d["source_file"] = os.path.basename(f)
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
ycols = [f"nbr_{y}" for y in hl.YEARS if f"nbr_{y}" in d.columns]
assert ycols, "no nbr_YYYY columns found - was the export made with MODE = 'EXPORT'?"
S = d[ycols].copy()
S.columns = [int(c[4:]) for c in ycols]
S = S.reindex(columns=hl.YEARS)

# closed canopy per Hansen using information up to the cutoff only; tree cover is known from 2000 on
closed = (d["hansen_tc2000"] >= 90) & ~d["hansen_lossyear"].between(1, hl.CUTOFF - 2000) & (d["hansen_gain"] == 0)
v = S.loc[closed, [y for y in hl.YEARS if y >= 2000]].to_numpy().ravel()
v = v[np.isfinite(v)]
forest_level = float(np.percentile(v, 5))
# open land: rows Hansen calls <= 10% tree cover in 2000 (no gain), their 1999-2001 values, 25th percentile,
# kept at least half a cut below closed canopy (falls back to closed canopy minus one cut if too few rows)
opn = (d["hansen_tc2000"] <= 10) & (d["hansen_gain"] == 0)
vo = S.loc[opn, [y for y in (1999, 2000, 2001) if y in S.columns]].to_numpy().ravel()
vo = vo[np.isfinite(vo)]
open_level = float(min(np.percentile(vo, 25), forest_level - args.drop / 2)) if len(vo) >= 30 else forest_level - args.drop
print(f"calibration on {int(closed.sum())} closed-canopy rows: forest_level={forest_level:.3f}, open_level={open_level:.3f}")
p = hl.HarvestParams(drop=args.drop, forest_level=forest_level, open_level=open_level)

rows = []
for k in range(len(d)):
    row = S.iloc[k]
    r = {}
    for method in ["S", "R"]:
        f = hl.harvest_features(row, p, method=method)
        r.update({f"{method}_{key}": val for key, val in f.items()})
    r.update(hl.summary_features(row, p))
    rows.append(r)
F = pd.DataFrame(rows)
GEE_FEATURES = ["n_cuts", "n_cycles", "last_cut_year", "max_drop", "forest_since_1985", "abrupt_start",
                "years_below_forest", "n_vertices"]
keep = [c for c in d.columns if c not in ycols]
out = pd.concat([d[keep].rename(columns={c: f"gee_{c}" for c in GEE_FEATURES}), F], axis=1)
out.to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(out)} rows")
