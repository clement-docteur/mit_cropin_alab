#!/usr/bin/env python3
"""
Step 7: harvest features from the yearly series (data up to 2020 only).

Usage: python harvest_features.py <data_dir>

1. Label-free calibration, using only Hansen information up to 2020 (never the plantation/forest labels):
   closed canopy = 5th percentile of the 2000-2020 values of points with >= 90% tree cover in 2000, no gain and
   no loss 2001-2020 (pixel and 3x3); open land = 25th percentile of the 1999-2001 values of points with <= 10%
   tree cover in 2000 and no gain, kept at least half a cut below closed canopy.
2. For every series (NBR: yearly median, dry-season median, wet-season median, yearly max; NDVI: median) and
   method (R = rules on the yearly values, S = LandTrendr-style segmentation, <= 18 segments): the harvest
   features and continuous summaries. "S6" = segmentation capped at 6 segments (as first specified), NBR median.
3. Threshold sensitivity for NBR median: cut counts for drops 0.10-0.30, rules and segmentation (cap 18 and 6).

Writes features_main.csv.gz, features_grid.csv.gz and calibration_levels.csv to <data_dir>.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

D = sys.argv[1]
SERIES = [("nbr", a) for a in hl.AGGREGATIONS] + [("ndvi", "median")]
DROPS = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_DROP = {"nbr": 0.20, "ndvi": 0.15}   # NDVI saturates over canopy, so its falls are smaller

pts = pd.read_csv(f"{D}/points.csv")
han = pd.read_csv(f"{D}/hansen_gfc.csv")
series = {(i, a): pd.read_csv(f"{D}/series_{i}_{a}.csv.gz", index_col=0).reindex(pts["pid"]) for i, a in SERIES}
for s in series.values():
    s.columns = s.columns.astype(int)

# ------------------------------------------------------------------ 1. label-free calibration
closed = han[(han.tc2000 >= 90) & ~han.lossyear.between(1, hl.CUTOFF - 2000) & (han.gain == 0)
             & (han.loss_3x3_le2020_n == 0)]["pid"]
open_pts = han[(han.tc2000 <= 10) & (han.gain == 0)]["pid"]
CAL_YEARS = [y for y in hl.YEARS if y >= 2000]          # tree cover is only known from 2000 on
cal_rows, LEVELS = [], {}
for (i, a), s in series.items():
    v_closed = s.loc[s.index.isin(closed), CAL_YEARS].to_numpy().ravel()
    v_closed = v_closed[np.isfinite(v_closed)]
    v_open = s.loc[s.index.isin(open_pts), [1999, 2000, 2001]].to_numpy().ravel()
    v_open = v_open[np.isfinite(v_open)]
    forest_level = float(np.percentile(v_closed, 5))
    open_level = float(min(np.percentile(v_open, 25), forest_level - DEFAULT_DROP[i] / 2))
    LEVELS[(i, a)] = (forest_level, open_level)
    cal_rows.append(dict(index=i, agg=a, n_closed_points=int(s.index.isin(closed).sum()),
                         closed_median=float(np.median(v_closed)), closed_p5=forest_level,
                         n_open_points=int(s.index.isin(open_pts).sum()), open_median=float(np.median(v_open)),
                         open_p25=float(np.percentile(v_open, 25)), open_level=open_level))
cal = pd.DataFrame(cal_rows)
cal.to_csv(f"{D}/calibration_levels.csv", index=False)
print(cal.round(3).to_string(index=False))


def params_for(i, a, drop=None):
    fl, ol = LEVELS[(i, a)]
    return hl.HarvestParams(drop=DEFAULT_DROP[i] if drop is None else drop, forest_level=fl, open_level=ol)


def work(pid):
    rows, grid = [], []
    for (i, a), s in series.items():
        row = s.loc[pid]
        p0 = params_for(i, a)
        summ = hl.summary_features(row, p0)
        methods = [("R", None), ("S", {})] + ([("S6", {"max_segments": 6})] if (i, a) == ("nbr", "median") else [])
        for name, kw in methods:
            f = hl.harvest_features(row, p0, method="R" if name == "R" else "S", seg_kw=kw)
            rows.append(dict(pid=pid, index=i, agg=a, method=name, **f, **summ))
        t, y = hl.valid_tv(row)
        if (i, a) != ("nbr", "median") or len(t) < 8:
            continue
        for d in DROPS:
            ev = hl.cycles_from_rules(t, y, params_for(i, a, d))
            grid.append(dict(pid=pid, method="R", max_segments=np.nan, drop=d, n_cuts=len(ev)))
        for ms in [18, 6]:
            tv, vv, fit = hl.segment(t, y, max_segments=ms)
            for d in DROPS:
                ev = hl.cycles_from_segments(tv, vv, t, fit, params_for(i, a, d))
                grid.append(dict(pid=pid, method="S", max_segments=ms, drop=d, n_cuts=len(ev)))
    return rows, grid


if __name__ == "__main__":
    with Pool(os.cpu_count()) as pool:
        res = pool.map(work, list(pts["pid"]), chunksize=20)
    main = pd.DataFrame([r for rows, _ in res for r in rows])
    grid = pd.DataFrame([g for _, gr in res for g in gr])
    main.to_csv(f"{D}/features_main.csv.gz", index=False)
    grid.to_csv(f"{D}/features_grid.csv.gz", index=False)
    m = main.join(pts.set_index("pid")[["dataset", "label"]], on="pid")
    print(m[(m["index"] == "nbr") & (m["agg"] == "median")].groupby(["method", "dataset", "label"])
          [["n_cuts", "n_cycles", "forest_since_1985", "abrupt_start"]].mean().round(3))
    print("rows:", len(main), "grid rows:", len(grid))
