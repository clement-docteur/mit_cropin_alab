#!/usr/bin/env python3
"""
Step 6: per-observation NBR/NDVI -> sensor harmonisation -> one value per point and year (1985-2020).

Usage: python build_annual_series.py <data_dir>
  <data_dir> holds points.csv, scenes_selected.csv and landsat_parts/ (steps 2-4).
Writes series_<index>_<agg>.csv.gz (rows = points, columns = 1985..2020), series_obs_counts.csv.gz,
series_sensor_offsets.csv and observations_indices.csv.gz.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

D = sys.argv[1]
SERIES = [("nbr", a) for a in hl.AGGREGATIONS] + [("ndvi", "median")]
pts = pd.read_csv(f"{D}/points.csv")
obs = hl.load_observations(f"{D}/landsat_parts", f"{D}/scenes_selected.csv")
obs = obs[obs["pid"].isin(pts["pid"])]
print(f"observations: {len(obs):,}  clear: {int(obs['clear'].sum()):,}  points: {obs['pid'].nunique()}")
obs = hl.add_indices(obs)

# Put Landsat 5 TM and 8 OLI on the Landsat 7 ETM+ scale (offsets from same-point pairs <= 8 days apart).
# Landsat 4 TM has no ETM+ overlap; it is the same instrument design as Landsat 5 TM.
report = []
for idx in ["nbr", "ndvi"]:
    oli, n8, ci8 = hl.estimate_sensor_offsets(obs, idx, "LC08", "LE07", years=(2013, hl.CUTOFF))
    tm, n5, ci5 = hl.estimate_sensor_offsets(obs, idx, "LT05", "LE07", years=(1999, 2011))
    print(f"{idx}: OLI - ETM+ = {oli:+.4f} ({n8} pairs), TM - ETM+ = {tm:+.4f} ({n5} pairs)")
    report.append(dict(index=idx, oli_minus_etm=oli, n_pairs_oli=n8, oli_ci_low=ci8[0], oli_ci_high=ci8[1],
                       tm_minus_etm=tm, n_pairs_tm=n5, tm_ci_low=ci5[0], tm_ci_high=ci5[1]))
    obs = hl.harmonise(obs, idx, {"LC08": oli, "LT05": tm, "LT04": tm})
pd.DataFrame(report).to_csv(f"{D}/series_sensor_offsets.csv", index=False)

n0 = int(obs["nbr"].notna().sum())
obs = hl.dedupe_same_pass(obs)
print(f"same-pass duplicates merged: {n0 - int(obs['nbr'].notna().sum()):,} of {n0:,} valid observations")
cnt = (obs[obs["nbr"].notna()].groupby(["pid", "year"]).size().unstack("year")
       .reindex(index=pts["pid"], columns=hl.YEARS).fillna(0))
cnt.to_csv(f"{D}/series_obs_counts.csv.gz")
print(f"clear observations per point-year: median {np.median(cnt.values):.0f}; "
      f"point-years with none: {float((cnt.values == 0).mean()):.3f}")

for idx, agg in SERIES:
    s = hl.annual_series(obs, idx, agg, pids=pts["pid"])
    s.to_csv(f"{D}/series_{idx}_{agg}.csv.gz")
    print(f"series_{idx}_{agg}: missing point-years {float(s.isna().mean().mean()):.3f}")
obs[["pid", "product_id", "sensor", "date", "year", "month", "nbr", "ndvi"]].to_csv(
    f"{D}/observations_indices.csv.gz", index=False)
