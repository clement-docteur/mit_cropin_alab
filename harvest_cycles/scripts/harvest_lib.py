"""
Shared functions for the harvest-cycle analysis. Only data up to the end of CUTOFF (2020, EUDR) is used.

observations (point x Landsat scene, cloud-masked TOA reflectance)
  -> NBR / NDVI per observation -> sensor harmonisation (offsets measured on same-point pairs)
  -> one value per point and year
  -> harvest features, by LandTrendr-style segmentation ("S") or simple rules on the yearly values ("R"),
     plus continuous summaries of the series and the raw series itself.
NBR is in TOA units: closed canopy ~0.6-0.75 in southern Brazil; a fresh clear-cut ~0.0-0.3.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

CUTOFF = 2020
FIRST_YEAR = 1985
YEARS = np.arange(FIRST_YEAR, CUTOFF + 1)
DRY_MONTHS = (4, 5, 6, 7, 8, 9)      # Apr-Sep: clearest skies, but deciduous forest is leafless
WET_MONTHS = (1, 2, 3, 10, 11, 12)   # Oct-Mar: full leaf, cloudier
AGGREGATIONS = ("median", "dry_median", "wet_median", "max")


# --------------------------------------------------------------------------- observations
def load_observations(parts_dir, scenes_csv):
    """Concatenate the per-scene CSVs written by extract_landsat.py and attach scene metadata."""
    import glob
    import os
    obs = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(parts_dir, "*.csv"))), ignore_index=True)
    sc = pd.read_csv(scenes_csv, dtype={"pr": str})[["product_id", "sensor", "date"]]
    obs = obs.merge(sc, on="product_id", how="left")
    obs["date"] = pd.to_datetime(obs["date"])
    obs["year"] = obs["date"].dt.year
    obs["month"] = obs["date"].dt.month
    return obs


def add_indices(obs):
    """NBR and NDVI from TOA reflectance; physically impossible values are set to NaN."""
    o = obs
    ok = (o["clear"] == 1) & (o[["red", "nir", "swir2"]] > 0).all(axis=1) & (o[["red", "nir", "swir2"]] < 1.2).all(axis=1)
    o["nbr"] = np.where(ok, (o["nir"] - o["swir2"]) / (o["nir"] + o["swir2"]), np.nan)
    o["ndvi"] = np.where(ok, (o["nir"] - o["red"]) / (o["nir"] + o["red"]), np.nan)
    # water / deep shadow / residual cloud: NIR very low or index outside the vegetated-land range
    bad = (o["nir"] < 0.05) | (o["nbr"] < -0.6) | (o["ndvi"] < -0.3)
    o.loc[bad, ["nbr", "ndvi"]] = np.nan
    return o


def estimate_sensor_offsets(obs, index, sensor="LC08", ref="LE07", max_days=8, years=(2013, CUTOFF)):
    """Median (sensor - ref) difference of `index` for the same point seen by both sensors within `max_days`.
    Returns (offset, n_pairs, bootstrap 95% CI)."""
    o = obs[obs[index].notna() & obs["year"].between(*years)]
    m = o[o.sensor == sensor][["pid", "date", index]].merge(
        o[o.sensor == ref][["pid", "date", index]], on="pid", suffixes=("_s", "_r"))
    m = m[(m["date_s"] - m["date_r"]).abs().dt.days <= max_days]
    d = (m[f"{index}_s"] - m[f"{index}_r"]).to_numpy()
    if len(d) < 30:
        return 0.0, len(d), (np.nan, np.nan)
    rng = np.random.default_rng(0)
    boots = [np.median(rng.choice(d, len(d))) for _ in range(500)]
    return float(np.median(d)), len(d), tuple(np.percentile(boots, [2.5, 97.5]))


def harmonise(obs, index, offsets):
    """Put every sensor on the Landsat 7 ETM+ scale: offsets = {sensor_code: offset to remove}."""
    for sensor, off in offsets.items():
        obs.loc[obs.sensor == sensor, index] = obs.loc[obs.sensor == sensor, index] - off
    return obs


def dedupe_same_pass(obs, cols=("nbr", "ndvi")):
    """A point in the overlap of two rows of one path is seen twice per pass (two products of one acquisition):
    keep one observation per point, sensor and date (mean of the two)."""
    keys = ["pid", "sensor", "date"]
    agg = {c: "mean" for c in cols if c in obs.columns}
    agg.update({c: "first" for c in obs.columns if c not in agg and c not in keys})
    return obs.groupby(keys, as_index=False).agg(agg)


def annual_series(obs, index="nbr", agg="median", pids=None):
    """One value per point and year (columns = YEARS, NaN = no clear observation)."""
    o = obs[obs[index].notna() & obs["year"].between(FIRST_YEAR, CUTOFF)]
    if agg == "dry_median":
        o = o[o["month"].isin(DRY_MONTHS)]
    elif agg == "wet_median":
        o = o[o["month"].isin(WET_MONTHS)]
    elif agg not in ("median", "max"):
        raise ValueError(agg)
    g = o.groupby(["pid", "year"])[index]
    s = (g.max() if agg == "max" else g.median()).unstack("year").reindex(columns=YEARS)
    return s if pids is None else s.reindex(pids)


# --------------------------------------------------------------------------- series helpers
def despike(y, thr=0.9, min_spike=0.05):
    """LandTrendr-style spike dampening: a value differing from the mean of its two neighbours by |s|, while
    the neighbours differ by less than (1-thr)*|s|, is replaced by that mean (largest spike first, repeatedly)."""
    y = np.asarray(y, float).copy()
    if len(y) < 3:
        return y
    for _ in range(len(y)):
        mid = (y[:-2] + y[2:]) / 2
        s = y[1:-1] - mid
        cand = (np.abs(s) >= min_spike) & (np.abs(y[2:] - y[:-2]) < (1 - thr) * np.abs(s))
        if not cand.any():
            break
        k = int(np.argmax(np.where(cand, np.abs(s), -1)))
        y[k + 1] = mid[k]
    return y


def valid_tv(row):
    """(years, values) of the non-missing yearly values of one point."""
    v = np.asarray(row, float)
    m = np.isfinite(v)
    return YEARS[m].astype(float), v[m]


# --------------------------------------------------------------------------- segmentation (S)
def _p2p_costs(t, y):
    n = len(t)
    C = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            pred = y[i] + (y[j] - y[i]) * (t[i:j + 1] - t[i]) / (t[j] - t[i])
            C[i, j] = np.sum((y[i:j + 1] - pred) ** 2)
    return C


def _dp_vertices(C, kmax):
    """Optimal point-to-point vertex sets for 1..kmax segments (dynamic programming)."""
    n = C.shape[0]
    F = np.full((kmax + 1, n), np.inf)
    B = np.full((kmax + 1, n), -1, int)
    F[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(1, n):
            cand = F[k - 1, :j] + C[:j, j]
            i = int(np.argmin(cand))
            F[k, j], B[k, j] = cand[i], i
    out = {}
    for k in range(1, min(kmax, n - 1) + 1):
        if np.isfinite(F[k, n - 1]):
            v, j = [n - 1], n - 1
            for kk in range(k, 0, -1):
                j = B[kk, j]
                v.append(j)
            out[k] = sorted(v)
    return out


def _refit(t, y, vidx):
    """Least-squares vertex values for fixed vertex years (LandTrendr's 'anchored' fit)."""
    eye = np.eye(len(vidx))
    A = np.column_stack([np.interp(t, t[vidx], eye[m]) for m in range(len(vidx))])
    vals = np.linalg.lstsq(A, y, rcond=None)[0]
    return vals, A @ vals


def segment(t, y, max_segments=18, bic_penalty=1.0, spike_thr=0.9):
    """LandTrendr-like segmentation of one yearly series; model size chosen by BIC.
    Returns vertex years, fitted vertex values and the fitted series.
    max_segments must allow ~3 segments per rotation (cut, regrowth, plateau). With 6 (LandTrendr's usual
    value) a pixel holds at most ~2 cuts, and on regular 5-7-year rotations the BIC returns a straight line
    unless ~15 segments are allowed. Do not make it unbounded: near n-2 the BIC saturates."""
    y = despike(y, spike_thr)
    n = len(t)
    if n < 2:
        return t, y, y
    if n < 6:
        return t[[0, -1]], y[[0, -1]], np.interp(t, t[[0, -1]], y[[0, -1]])
    best = None
    for k, vidx in _dp_vertices(_p2p_costs(t, y), min(max_segments, n - 2)).items():
        vals, fit = _refit(t, y, vidx)
        bic = n * np.log(float(np.sum((y - fit) ** 2)) / n + 1e-6) + bic_penalty * 2 * k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, vidx, vals, fit)
    _, vidx, vals, fit = best
    return t[vidx], vals, fit


@dataclass
class HarvestParams:
    drop: float = 0.20            # minimum NBR fall of a cut
    max_cut_years: int = 3        # a cut must happen within this many years (abrupt, not decline)
    rec_frac: float = 0.5         # share of the fall that must grow back to call it a cycle
    max_rec_years: int = 10       # ... within this many years after the low point
    regrow_frac: float = 0.8      # regrowth time = years to recover this share of the fall
    forest_level: float = 0.50    # closed canopy at or above this (calibrated in harvest_features.py)
    open_level: float = 0.40      # open land at or below this (calibrated in harvest_features.py)
    min_open_years: int = 3       # an 'abrupt start' needs at least this many open years before it
    fast_rise_years: int = 5      # ... and closed canopy within this many years of the last open year
    stable_tolerance: int = 0     # years allowed below forest_level for 'forest since 1985'


def _time_to_reach(t, f, start_t, target):
    """First year >= start_t at which curve f reaches `target` (linear interpolation), or None."""
    idx = np.where(t >= start_t)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if f[a] >= target:
            return t[a]
        if f[b] >= target:
            return t[a] + (target - f[a]) / (f[b] - f[a]) * (t[b] - t[a])
    if len(idx) and f[idx[-1]] >= target:
        return t[idx[-1]]
    return None


def cycles_from_segments(tv, vv, t, fit, p: HarvestParams):
    """Cut events (fast NBR falls between vertices) and whether each regrows (along the fitted curve)."""
    events = []
    for i in range(len(tv) - 1):
        t0, t1, v0, v1 = tv[i], tv[i + 1], vv[i], vv[i + 1]
        if (v0 - v1) >= p.drop and (t1 - t0) <= p.max_cut_years:
            drop = v0 - v1
            r50 = _time_to_reach(t, fit, t1, v1 + p.rec_frac * drop)
            r80 = _time_to_reach(t, fit, t1, v1 + p.regrow_frac * drop)
            events.append(dict(cut_year=t0 + 1, low_year=t1, drop=drop,
                               cycle=r50 is not None and (r50 - t1) <= p.max_rec_years,
                               regrow_years=None if r80 is None else r80 - t1))
    return events


def cycles_from_rules(t, y, p: HarvestParams, ref_years=3):
    """Cut events from the yearly values: NBR falls >= drop below the max of the previous `ref_years` values
    (never reaching back past the previous event's low) and stays down the next year, or falls >= 2*drop."""
    y = despike(y)
    n = len(y)
    events, i, last_low = [], 1, 0
    while i < n:
        lo = max(0, i - ref_years, last_low)
        prev = y[lo:i][(t[i] - t[lo:i]) <= ref_years + 2]     # do not compare across long data gaps
        if len(prev):
            pre = prev.max()
            if pre - y[i] >= p.drop and (i == n - 1 or y[i + 1] <= pre - p.drop / 2 or pre - y[i] >= 2 * p.drop):
                j = i + int(np.argmin(y[i:min(i + 3, n)]))
                drop = pre - y[j]
                r50 = _time_to_reach(t, y, t[j], y[j] + p.rec_frac * drop)
                r80 = _time_to_reach(t, y, t[j], y[j] + p.regrow_frac * drop)
                cyc = r50 is not None and (r50 - t[j]) <= p.max_rec_years
                events.append(dict(cut_year=t[i], low_year=t[j], drop=drop, cycle=cyc,
                                   regrow_years=None if r80 is None else r80 - t[j]))
                last_low = j
                i = int(np.searchsorted(t, r50)) + 1 if cyc else j + 2
                continue
        i += 1
    return events


# --------------------------------------------------------------------------- features
def establishment(t, y, p: HarvestParams):
    """'Abrupt start': the first year at closed canopy after an open spell (>= min_open_years open values in the
    8 values ending at the last open one); abrupt if that last open year is <= fast_rise_years earlier.
    Returns (abrupt 0/1, rise years, year) or (0, NaN, NaN)."""
    y = despike(y)
    low = y <= p.open_level
    for e in range(len(y)):
        if y[e] < p.forest_level:
            continue
        before = np.where(low[:e])[0]
        if len(before) < p.min_open_years:
            continue
        last_open = before[-1]
        if np.sum(low[max(0, last_open - 7):last_open + 1]) < p.min_open_years:
            continue
        rise = t[e] - t[last_open]
        return int(rise <= p.fast_rise_years), float(rise), float(t[e])
    return 0, np.nan, np.nan


def harvest_features(row, p: HarvestParams, method="S", seg_kw=None):
    """The harvest features of one yearly series (pd.Series indexed by YEARS)."""
    t, y = valid_tv(row)
    out = dict(n_years=len(t))
    if len(t) < 8:
        return out
    if method == "S":
        tv, vv, fit = segment(t, y, **(seg_kw or {}))
        ev = cycles_from_segments(tv, vv, t, fit, p)
        out["n_vertices"] = len(tv)
    else:
        ev = cycles_from_rules(t, y, p)
    cyc = [e for e in ev if e["cycle"]]
    rg = [e["regrow_years"] for e in cyc if e["regrow_years"] is not None]
    below = int(np.sum(despike(y) < p.forest_level))
    a, rise, yr = establishment(t, y, p)
    out.update(n_cuts=len(ev), n_cycles=len(cyc),
               last_cut_year=max((e["cut_year"] for e in ev), default=np.nan),
               max_drop=max((e["drop"] for e in ev), default=0.0),
               regrow_years=float(np.min(rg)) if rg else np.nan,
               years_below_forest=below, forest_since_1985=int(len(ev) == 0 and below <= p.stable_tolerance),
               abrupt_start=a, establish_rise_years=rise, establish_year=yr)
    return out


def summary_features(row, p: HarvestParams):
    """Continuous summaries of the yearly series: variability, lowest value, largest 3-year fall, share of
    years below closed canopy."""
    t, y = valid_tv(row)
    if len(t) < 8:
        return {}
    ys = despike(y)
    drop3 = [ys[max(0, i - 3):i].max() - ys[i] for i in range(1, len(ys))]
    return dict(s_std=ys.std(), s_min=ys.min(), s_max_drop3=float(max(drop3)) if drop3 else 0.0,
                s_frac_below_forest=float(np.mean(ys < p.forest_level)))


def full_series_matrix(series):
    """The 36 yearly values (linearly interpolated across gaps, edge-filled) plus first differences, for the
    black-box benchmark."""
    s = series.interpolate(axis=1, limit_direction="both")
    d = s.diff(axis=1).iloc[:, 1:]
    d.columns = [f"d{c}" for c in d.columns]
    s.columns = [f"y{c}" for c in s.columns]
    return pd.concat([s, d], axis=1)
