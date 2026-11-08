#!/usr/bin/env python3
"""
Step 7b: harvest features from the 3 x 3 Landsat window around each point (90 m x 90 m), to test whether the
neighbourhood helps for small patches (Cropin's QC clearings are mostly smaller than one 30 m pixel).

Usage: python neighbourhood_features.py <data_dir>
Needs landsat_parts_win3/ (extract_landsat_window.py <data_dir> <workers> 3), series_sensor_offsets.csv and
calibration_levels.csv from the single-pixel run. Primary configuration only (NBR, yearly median).

Writes <data_dir>/features_win3.csv.gz with, per point:
  c_*   the centre pixel recomputed from the window data (sanity check against the single-pixel run)
  nb_*  aggregates over the up-to-9 pixels (any cut, share of pixels with a cut, max drop, mean variability,
        'forest since 1985' in all pixels, ...)
  mean_* features of the 90 m mean series (yearly NBR averaged over the window first)
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

D = sys.argv[1]
pts = pd.read_csv(f"{D}/points.csv")
cal = pd.read_csv(f"{D}/calibration_levels.csv")
offs = pd.read_csv(f"{D}/series_sensor_offsets.csv").set_index("index")
c = cal[(cal["index"] == "nbr") & (cal["agg"] == "median")].iloc[0]
P = hl.HarvestParams(drop=0.20, forest_level=float(c.closed_p5), open_level=float(c.open_level))
print("levels:", P.forest_level, P.open_level)

sc = pd.read_csv(f"{D}/scenes_selected.csv", dtype={"pr": str})[["product_id", "sensor", "date"]]
cols = ["pid", "product_id", "dr", "dc", "clear", "red", "nir", "swir2"]
obs = pd.concat((pd.read_csv(f, usecols=cols) for f in glob.glob(f"{D}/landsat_parts_win3/*.csv")), ignore_index=True)
obs = obs.merge(sc, on="product_id", how="left")
obs["date"] = pd.to_datetime(obs["date"])
obs["year"] = obs["date"].dt.year
obs["month"] = obs["date"].dt.month
obs = hl.add_indices(obs)
o = offs.loc["nbr"]
obs = hl.harmonise(obs, "nbr", {"LC08": o.oli_minus_etm, "LT05": o.tm_minus_etm, "LT04": o.tm_minus_etm})
obs["sub"] = obs["pid"] + "|" + obs["dr"].astype(str) + "|" + obs["dc"].astype(str)
obs = hl.dedupe_same_pass(obs, cols=("nbr",), key="sub")
print(f"window observations (same-pass duplicates merged): {len(obs):,}, sub-pixels: {obs['sub'].nunique():,}")
# annual_series groups by 'pid': give it the sub-pixel id
S = hl.annual_series(obs.rename(columns={"pid": "pid_point"}).rename(columns={"sub": "pid"}), "nbr", "median")
S.index.name = "sub"
meta = S.index.to_series().str.split("|", expand=True)
meta.columns = ["pid", "dr", "dc"]
meta[["dr", "dc"]] = meta[["dr", "dc"]].astype(int)


def per_sub(sub):
    row = S.loc[sub]
    out = {"sub": sub}
    for m in ["R", "S"]:
        f = hl.harvest_features(row, P, method=m)
        out.update({f"{m}_{k}": v for k, v in f.items()})
    out.update(hl.summary_features(row, P))
    return out


def per_mean(pid):
    subs = meta.index[meta["pid"] == pid]
    row = S.loc[subs].mean(axis=0)          # 90 m mean series (missing years stay missing only if all 9 are)
    out = {"pid": pid}
    for m in ["R", "S"]:
        f = hl.harvest_features(row, P, method=m)
        out.update({f"mean_{m}_{k}": v for k, v in f.items()})
    out.update({f"mean_{k}": v for k, v in hl.summary_features(row, P).items()})
    return out


if __name__ == "__main__":
    with Pool(os.cpu_count()) as pool:
        subf = pd.DataFrame(pool.map(per_sub, list(S.index), chunksize=50))
        meanf = pd.DataFrame(pool.map(per_mean, list(pts["pid"]), chunksize=20))
    subf = subf.join(meta, on="sub")
    centre = subf[(subf.dr == 0) & (subf.dc == 0)].drop(columns=["sub", "dr", "dc"]).set_index("pid").add_prefix("c_")
    g = subf.groupby("pid")
    nb = pd.DataFrame({
        "nb_n_pixels": g.size(),
        "nb_any_cut_R": g["R_n_cuts"].apply(lambda x: int((x.fillna(0) >= 1).any())),
        "nb_share_cut_R": g["R_n_cuts"].apply(lambda x: float((x.fillna(0) >= 1).mean())),
        "nb_any_cut_S": g["S_n_cuts"].apply(lambda x: int((x.fillna(0) >= 1).any())),
        "nb_share_cut_S": g["S_n_cuts"].apply(lambda x: float((x.fillna(0) >= 1).mean())),
        "nb_max_n_cuts_S": g["S_n_cuts"].max(),
        "nb_all_forest_since_1985_S": g["S_forest_since_1985"].apply(lambda x: int((x.fillna(0) == 1).all())),
        "nb_share_forest_since_1985_S": g["S_forest_since_1985"].apply(lambda x: float((x.fillna(0) == 1).mean())),
        "nb_any_abrupt_start_S": g["S_abrupt_start"].apply(lambda x: int((x.fillna(0) == 1).any())),
        "nb_max_s_max_drop3": g["s_max_drop3"].max(),
        "nb_mean_s_std": g["s_std"].mean(),
        "nb_max_s_std": g["s_std"].max(),
        "nb_mean_s_frac_below_forest": g["s_frac_below_forest"].mean(),
        "nb_min_s_min": g["s_min"].min(),
    })
    out = pts[["pid"]].set_index("pid").join(centre).join(nb).join(meanf.set_index("pid"))
    out.to_csv(f"{D}/features_win3.csv.gz")
    print("wrote", len(out), "points;", out.shape[1], "columns")
