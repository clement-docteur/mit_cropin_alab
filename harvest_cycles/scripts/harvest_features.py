#!/usr/bin/env python3
"""
Step 5: harvest-cycle features from the annual series (only data up to the 2020 cutoff).

Usage: python harvest_features.py <data_dir>

1. Label-free calibration of the NBR/NDVI 'closed canopy' and 'open land' levels for every series type
   (never uses the plantation/forest labels, and only Hansen information up to 2020):
   closed canopy = 5th percentile of the 2000-2020 values of pixels with >= 90% tree cover in 2000,
   no gain and no loss 2001-2020 (pixel and 3x3); open land = 25th percentile of the 1999-2001 values
   of pixels with <= 10% tree cover in 2000 and no gain, kept at least half a cut below closed canopy.
2. For every index (NBR, NDVI) x annual aggregation (median, max, p75, dry-season median,
   wet-season median) x method (R = rules on annual values, S = LandTrendr-style segmentation):
   the four interpretable features + continuous series summaries.
3. A sensitivity grid (drop threshold x BIC penalty, and segment cap 6 / 12 / 18 for NBR / median) for the
   cut / cycle counts. Method "S6" = the segmentation as first specified (cap 6), NBR / median only.

Writes <data_dir>/features_main.csv.gz (wide, one row per point and configuration),
<data_dir>/features_grid.csv.gz and <data_dir>/calibration_levels.csv.
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

D = sys.argv[1]
INDICES = ["nbr", "ndvi"]
AGGS = list(hl.AGGREGATIONS)
DROPS = [0.10, 0.15, 0.20, 0.25, 0.30]
PENS = [0.5, 1.0, 2.0]
DEFAULT_DROP = {"nbr": 0.20, "ndvi": 0.15}   # NDVI saturates over canopy, so its drops are smaller
DEFAULT_PEN = 1.0

pts = pd.read_csv(f"{D}/points.csv")
han = pd.read_csv(f"{D}/hansen_gfc.csv")
series = {(i, a): pd.read_csv(f"{D}/series_{i}_{a}.csv.gz", index_col=0).reindex(pts["pid"])
          for i in INDICES for a in AGGS}
for s in series.values():
    s.columns = s.columns.astype(int)

# ------------------------------------------------------------------ 1. label-free calibration
# closed canopy per Hansen, using only information up to the cutoff: >= 90% tree cover in 2000, no gain
# (2000-2012), no loss 2001-2020 at the pixel or its 3x3 neighbours. Only the years 2000-2020 are used,
# because tree cover is only known from 2000 on.
closed = han[(han.tc2000 >= 90) & ~han.lossyear.between(1, hl.CUTOFF - 2000) & (han.gain == 0)
             & (han.loss_3x3_le2020_n == 0)]["pid"]
CAL_YEARS = [y for y in hl.YEARS if y >= 2000]
open_pts = han[(han.tc2000 <= 10) & (han.gain == 0)]["pid"]
cal_rows, LEVELS = [], {}
for (i, a), s in series.items():
    v_closed = s.loc[s.index.isin(closed), CAL_YEARS].to_numpy().ravel()
    v_closed = v_closed[np.isfinite(v_closed)]
    v_all = s.to_numpy().ravel()
    v_all = v_all[np.isfinite(v_all)]
    forest_level = float(np.percentile(v_closed, 5))
    # open land (pasture, crops, bare soil): pixels Hansen calls <= 10% tree cover in 2000 with no gain,
    # their 1999-2001 values; 25th percentile, kept at least half a cut below closed canopy
    v_open = s.loc[s.index.isin(open_pts), [1999, 2000, 2001]].to_numpy().ravel()
    v_open = v_open[np.isfinite(v_open)]
    open_level = float(min(np.percentile(v_open, 25), forest_level - DEFAULT_DROP[i] / 2))
    LEVELS[(i, a)] = (forest_level, open_level)
    cal_rows.append(dict(index=i, agg=a, n_closed_points=int(s.index.isin(closed).sum()),
                         n_open_points=int(s.index.isin(open_pts).sum()), open_p25=float(np.percentile(v_open, 25)),
                         open_median=float(np.median(v_open)),
                         closed_median=float(np.median(v_closed)), closed_p5=forest_level,
                         closed_p1=float(np.percentile(v_closed, 1)), open_level=open_level,
                         all_p5=float(np.percentile(v_all, 5)), all_median=float(np.median(v_all)),
                         share_all_below_forest=float(np.mean(v_all < forest_level)),
                         share_all_below_open=float(np.mean(v_all <= open_level))))
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
        t, y = hl.valid_tv(row)
        p0 = params_for(i, a)
        summ = hl.summary_features(row, p0)
        for method in ["R", "S"]:
            f = hl.harvest_features(row, p0, method=method, seg_kw=dict(bic_penalty=DEFAULT_PEN))
            rows.append(dict(pid=pid, index=i, agg=a, method=method, **f, **summ))
        if (i, a) == ("nbr", "median"):
            # the segmentation as first specified (6-segment cap), kept as a reported sensitivity
            f = hl.harvest_features(row, p0, method="S", seg_kw=dict(bic_penalty=DEFAULT_PEN, max_segments=6))
            rows.append(dict(pid=pid, index=i, agg=a, method="S6", **f, **summ))
        if len(t) < 8:
            continue
        # sensitivity grid: R for every drop; S for every drop x BIC penalty (segment once per penalty)
        for d in DROPS:
            ev = hl.cycles_from_rules(t, y, params_for(i, a, d))
            grid.append(dict(pid=pid, index=i, agg=a, method="R", drop=d, pen=np.nan, max_segments=np.nan,
                             n_cuts=len(ev), n_cycles=sum(e["cycle"] for e in ev)))
        for pen in PENS:
            for ms in ([6, 12, 18] if (i, a) == ("nbr", "median") else [18]):
                tv, vv, fit = hl.segment(t, y, max_segments=ms, bic_penalty=pen)
                for d in DROPS:
                    ev = hl.cycles_from_segments(tv, vv, t, fit, params_for(i, a, d))
                    grid.append(dict(pid=pid, index=i, agg=a, method="S", drop=d, pen=pen, max_segments=ms,
                                     n_cuts=len(ev), n_cycles=sum(e["cycle"] for e in ev)))
    return rows, grid


if __name__ == "__main__":
    with Pool(os.cpu_count()) as pool:
        res = pool.map(work, list(pts["pid"]), chunksize=20)
    main = pd.DataFrame([r for rows, _ in res for r in rows])
    grid = pd.DataFrame([g for _, gr in res for g in gr])
    main.to_csv(f"{D}/features_main.csv.gz", index=False)
    grid.to_csv(f"{D}/features_grid.csv.gz", index=False)
    lab = pts.set_index("pid")[["dataset", "label"]]
    m = main.join(lab, on="pid")
    show = m[(m["index"] == "nbr") & (m["agg"] == "median")]
    print(show.groupby(["method", "dataset", "label"])[["n_cuts", "n_cycles", "forest_since_1985", "abrupt_start",
                                                         "regrow_years"]].mean().round(3))
    print("rows:", len(main), "grid rows:", len(grid))
