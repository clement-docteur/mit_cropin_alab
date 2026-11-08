"""
Shared functions for the harvest-cycle analysis.

Everything here only ever looks at data up to the end of the cutoff year (CUTOFF = 2020, EUDR).

Pipeline
  observations (one row per point x Landsat scene, cloud-masked TOA reflectance)
    -> spectral index per observation (NBR, NDVI)
    -> sensor harmonisation (Landsat 8 OLI offset estimated from near-simultaneous Landsat 7 pairs)
    -> one value per point and year (several aggregation rules, compared in the analysis)
    -> harvest-cycle features, two ways:
         R = simple rules on the annual values (drop, then regrowth)
         S = LandTrendr-style piecewise-linear segmentation, then rules on the segments
       plus continuous summaries of the series (C) and the raw series itself (F).

Units: NBR and NDVI are unitless (-1..1). Thresholds below are in TOA NBR units. TOA NBR of closed
canopy in southern Brazil is ~0.6-0.75; a fresh clear-cut drops it to ~0.0-0.3.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CUTOFF = 2020
FIRST_YEAR = 1985
YEARS = np.arange(FIRST_YEAR, CUTOFF + 1)

DRY_MONTHS = (4, 5, 6, 7, 8, 9)          # Apr-Sep: clearest skies in the south, but deciduous forest is leafless
WET_MONTHS = (1, 2, 3, 10, 11, 12)       # Oct-Mar: full leaf, cloudier


# --------------------------------------------------------------------------- observations
def load_observations(parts_dir, scenes_csv):
    """Concatenate the per-scene CSVs written by extract_landsat.py and attach scene metadata."""
    import glob
    import os
    files = glob.glob(os.path.join(parts_dir, "*.csv"))
    obs = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    sc = pd.read_csv(scenes_csv, dtype={"pr": str})[["product_id", "sensor", "pr", "date", "cc"]]
    obs = obs.merge(sc, on="product_id", how="left")
    obs["date"] = pd.to_datetime(obs["date"])
    obs["year"] = obs["date"].dt.year
    obs["month"] = obs["date"].dt.month
    obs["doy"] = obs["date"].dt.dayofyear
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
    """Median (sensor - ref) difference of `index` for the same point observed by both sensors within
    `max_days` days. Default: Landsat 8 OLI vs Landsat 7 ETM+ in 2013-2020; for Landsat 5 TM use
    sensor="LT05", years=(1999, 2011). Returns (offset, n_pairs, bootstrap 95% CI)."""
    o = obs[obs[index].notna() & obs["year"].between(*years)]
    a = o[o.sensor == sensor][["pid", "date", index]]
    b = o[o.sensor == ref][["pid", "date", index]]
    m = a.merge(b, on="pid", suffixes=("_s", "_r"))
    m = m[(m["date_s"] - m["date_r"]).abs().dt.days <= max_days]
    d = (m[f"{index}_s"] - m[f"{index}_r"]).to_numpy()
    if len(d) < 30:
        return 0.0, len(d), (np.nan, np.nan)
    rng = np.random.default_rng(0)
    boots = [np.median(rng.choice(d, len(d))) for _ in range(500)]
    return float(np.median(d)), len(d), tuple(np.percentile(boots, [2.5, 97.5]))


def harmonise(obs, index, offsets):
    """Put every sensor on the Landsat 7 ETM+ scale by removing its estimated offset.
    offsets: {sensor_code: offset}, e.g. {"LC08": -0.003, "LT05": -0.011, "LT04": -0.011}."""
    for sensor, off in offsets.items():
        obs.loc[obs.sensor == sensor, index] = obs.loc[obs.sensor == sensor, index] - off
    return obs


def dedupe_same_pass(obs, cols=("nbr", "ndvi"), key="pid"):
    """A point in the overlap of two rows of the same path is seen twice in one pass (two products of one
    acquisition). Keep one observation per point, sensor and date (mean of the two)."""
    keys = [key, "sensor", "date"]
    agg = {c: "mean" for c in cols if c in obs.columns}
    rest = [c for c in obs.columns if c not in agg and c not in keys]
    agg.update({c: "first" for c in rest})
    return obs.groupby(keys, as_index=False).agg(agg)


AGGREGATIONS = ("median", "max", "p75", "dry_median", "wet_median")


def annual_series(obs, index="nbr", agg="median", pids=None):
    """One value per point and year (columns = YEARS, NaN = no clear observation)."""
    o = obs[obs[index].notna() & obs["year"].between(FIRST_YEAR, CUTOFF)]
    if agg == "dry_median":
        o = o[o["month"].isin(DRY_MONTHS)]
        g = o.groupby(["pid", "year"])[index].median()
    elif agg == "wet_median":
        o = o[o["month"].isin(WET_MONTHS)]
        g = o.groupby(["pid", "year"])[index].median()
    elif agg == "median":
        g = o.groupby(["pid", "year"])[index].median()
    elif agg == "max":
        g = o.groupby(["pid", "year"])[index].max()
    elif agg == "p75":
        g = o.groupby(["pid", "year"])[index].quantile(0.75)
    else:
        raise ValueError(agg)
    s = g.unstack("year").reindex(columns=YEARS)
    if pids is not None:
        s = s.reindex(pids)
    return s


# --------------------------------------------------------------------------- series helpers
def despike(y, thr=0.9, min_spike=0.05):
    """LandTrendr-style spike dampening. A value is a one-year spike when it differs from the mean of
    its two neighbours by |s| while the neighbours differ from each other by less than (1-thr)*|s|.
    The largest spike is replaced by the neighbour mean, repeatedly."""
    y = np.asarray(y, float).copy()
    n = len(y)
    if n < 3:
        return y
    for _ in range(n):
        mid = (y[:-2] + y[2:]) / 2
        s = y[1:-1] - mid
        d = np.abs(y[2:] - y[:-2])
        cand = (np.abs(s) >= min_spike) & (d < (1 - thr) * np.abs(s))
        if not cand.any():
            break
        k = int(np.argmax(np.where(cand, np.abs(s), -1)))
        y[k + 1] = mid[k]
    return y


def valid_tv(row):
    """(years, values) of the non-missing annual values of one point."""
    v = np.asarray(row, float)
    m = np.isfinite(v)
    return YEARS[m].astype(float), v[m]


# --------------------------------------------------------------------------- segmentation (S)
def _p2p_costs(t, y):
    n = len(t)
    C = np.zeros((n, n))
    for i in range(n - 1):
        tt = t[i:]
        for j in range(i + 1, n):
            pred = y[i] + (y[j] - y[i]) * (tt[: j - i + 1] - t[i]) / (t[j] - t[i])
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
    tv = t[vidx]
    eye = np.eye(len(vidx))
    A = np.column_stack([np.interp(t, tv, eye[m]) for m in range(len(vidx))])
    vals = np.linalg.lstsq(A, y, rcond=None)[0]
    return vals, A @ vals


def segment(t, y, max_segments=18, bic_penalty=1.0, spike_thr=0.9):
    """LandTrendr-like temporal segmentation of one annual series.
    Returns vertex years, fitted vertex values, and the fitted series. Model size is chosen by BIC
    (penalty multiplier `bic_penalty`; larger = fewer segments).
    max_segments must allow ~3 segments per rotation (cut, regrowth, plateau): with 6 (LandTrendr's usual
    value) a pixel holds at most 2 cuts, and on regular 5-7-year rotations the BIC is all-or-nothing, so a
    cap below ~15 can return a straight line with 0 cuts. 18 recovers them (forest false alarms +0.6 points
    vs 12 on the MapBiomas points). Do not make it unbounded: near n-2 the BIC saturates."""
    y = despike(y, spike_thr)
    n = len(t)
    if n < 2:
        return t, y, y
    if n < 6:
        return t[[0, -1]], y[[0, -1]], np.interp(t, t[[0, -1]], y[[0, -1]])
    C = _p2p_costs(t, y)
    cands = _dp_vertices(C, min(max_segments, n - 2))
    best = None
    for k, vidx in cands.items():
        vals, fit = _refit(t, y, vidx)
        sse = float(np.sum((y - fit) ** 2))
        bic = n * np.log(sse / n + 1e-6) + bic_penalty * 2 * k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, vidx, vals, fit)
    _, vidx, vals, fit = best
    return t[vidx], vals, fit


@dataclass
class HarvestParams:
    drop: float = 0.20            # minimum NBR decrease of a cut (TOA NBR units)
    max_cut_years: int = 3        # a cut must happen within this many years (abrupt, not decline)
    rec_frac: float = 0.5         # share of the drop that must grow back to call it a cycle
    max_rec_years: int = 10       # ... within this many years after the low point
    regrow_frac: float = 0.8      # regrowth time = years to recover this share of the drop
    forest_level: float = 0.50    # NBR at or above this = closed canopy (calibrated on stable forest)
    open_level: float = 0.40      # NBR at or below this = open land (pasture, crop, bare soil)
    min_open_years: int = 3       # an establishment needs at least this many open years before it
    fast_rise_years: int = 5      # 'abrupt start' if open -> forest level within this many years
    stable_tolerance: int = 0     # years allowed below forest_level for 'forest since 1985'
    extra: dict = field(default_factory=dict)


def _time_to_reach(t, f, start_t, target):
    """First year >= start_t at which the fitted/annual curve f reaches `target` (linear interpolation)."""
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
    """Cut events (fast NBR drops) and cut-and-regrow cycles from LandTrendr-style vertices."""
    events = []
    for i in range(len(tv) - 1):
        t0, t1, v0, v1 = tv[i], tv[i + 1], vv[i], vv[i + 1]
        if (v0 - v1) >= p.drop and (t1 - t0) <= p.max_cut_years:
            drop = v0 - v1
            r50 = _time_to_reach(t, fit, t1, v1 + p.rec_frac * drop)
            r80 = _time_to_reach(t, fit, t1, v1 + p.regrow_frac * drop)
            events.append(dict(cut_year=t0 + 1, low_year=t1, pre=v0, low=v1, drop=drop,
                               cycle=r50 is not None and (r50 - t1) <= p.max_rec_years,
                               regrow_years=None if r80 is None else r80 - t1))
    return events


# --------------------------------------------------------------------------- rule-based (R)
def cycles_from_rules(t, y, p: HarvestParams, ref_years=3):
    """Cut events from the annual values directly: NBR falls >= drop below the max of the previous
    `ref_years` values and stays down the next year (or falls >= 2*drop). A cycle = the cut is followed
    by >= rec_frac recovery within max_rec_years."""
    y = despike(y)
    n = len(y)
    events, i, last_low = [], 1, 0
    while i < n:
        lo = max(0, i - ref_years, last_low)            # never reach back past the previous event's low
        prev = y[lo:i]
        prev_t = t[lo:i]
        prev = prev[(t[i] - prev_t) <= ref_years + 2]   # do not compare across long data gaps
        if len(prev):
            pre = prev.max()
            if pre - y[i] >= p.drop and (i == n - 1 or y[i + 1] <= pre - p.drop / 2 or pre - y[i] >= 2 * p.drop):
                j = i + int(np.argmin(y[i:min(i + 3, n)]))
                drop = pre - y[j]
                r50 = _time_to_reach(t, y, t[j], y[j] + p.rec_frac * drop)
                r80 = _time_to_reach(t, y, t[j], y[j] + p.regrow_frac * drop)
                cyc = r50 is not None and (r50 - t[j]) <= p.max_rec_years
                events.append(dict(cut_year=t[i], low_year=t[j], pre=pre, low=y[j], drop=drop,
                                   cycle=cyc, regrow_years=None if r80 is None else r80 - t[j]))
                last_low = j
                if cyc:
                    i = int(np.searchsorted(t, r50)) + 1
                else:
                    i = j + 2
                continue
        i += 1
    return events


# --------------------------------------------------------------------------- the four features
def establishment(t, y, p: HarvestParams):
    """'Abrupt start': >= min_open_years at open-land level, then forest level reached within
    fast_rise_years of the last open year. Returns (abrupt_start 0/1, rise_years or NaN, year)."""
    y = despike(y)
    low = y <= p.open_level
    best = (0, np.nan, np.nan)
    for e in range(len(y)):
        if y[e] < p.forest_level:
            continue
        before = np.where(low[:e])[0]
        if len(before) < p.min_open_years:
            continue
        last_open = before[-1]
        # the open spell must be a real spell, not scattered noisy years: >= min_open_years
        # open values among the years last_open-7 .. last_open
        spell = np.sum(low[max(0, last_open - 7):last_open + 1])
        if spell < p.min_open_years:
            continue
        rise = t[e] - t[last_open]
        fast = int(rise <= p.fast_rise_years)
        if np.isnan(best[1]) or rise < best[1]:
            best = (fast, float(rise), float(t[e]))
        break  # first time the series reaches forest level after an open spell
    return best


def harvest_features(row, p: HarvestParams, method="S", seg_kw=None):
    """The four interpretable harvest features for one annual series (pd.Series indexed by YEARS)."""
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
    out["n_cuts"] = len(ev)
    out["n_cycles"] = len(cyc)
    out["last_cut_year"] = max((e["cut_year"] for e in ev), default=np.nan)
    out["max_drop"] = max((e["drop"] for e in ev), default=0.0)
    rg = [e["regrow_years"] for e in cyc if e["regrow_years"] is not None]
    out["regrow_years"] = float(np.min(rg)) if rg else np.nan
    ys = despike(y)
    below = int(np.sum(ys < p.forest_level))
    out["years_below_forest"] = below
    out["forest_since_1985"] = int(len(ev) == 0 and below <= p.stable_tolerance)
    a, rise, yr = establishment(t, y, p)
    out["abrupt_start"] = a
    out["establish_rise_years"] = rise
    out["establish_year"] = yr
    return out


def summary_features(row, p: HarvestParams):
    """Continuous summaries of the annual series up to the cutoff (C)."""
    t, y = valid_tv(row)
    if len(t) < 8:
        return {}
    ys = despike(y)
    d1 = np.diff(ys)
    drop3 = [ys[max(0, i - 3):i].max() - ys[i] for i in range(1, len(ys))]
    k = int(np.argmin(ys))
    recent = ys[t >= CUTOFF - 4]
    early = ys[t <= FIRST_YEAR + 5]
    try:
        from scipy.stats import theilslopes
        slope = theilslopes(ys, t)[0]
    except Exception:
        slope = np.polyfit(t, ys, 1)[0]
    return dict(s_min=ys.min(), s_p10=np.percentile(ys, 10), s_median=np.median(ys), s_std=ys.std(),
                s_range=ys.max() - ys.min(), s_max_drop1=float(-d1.min()) if len(d1) else 0.0,
                s_max_drop3=float(max(drop3)) if drop3 else 0.0, s_max_rise1=float(d1.max()) if len(d1) else 0.0,
                s_years_since_min=CUTOFF - t[k], s_recent_mean=recent.mean() if len(recent) else np.nan,
                s_early_mean=early.mean() if len(early) else np.nan, s_slope=slope,
                s_frac_below_forest=float(np.mean(ys < p.forest_level)))


def full_series_matrix(series):
    """Raw annual values (linearly interpolated across missing years, edge-filled) for the
    black-box benchmark (F), plus first differences."""
    s = series.interpolate(axis=1, limit_direction="both")
    d = s.diff(axis=1).iloc[:, 1:]
    d.columns = [f"d{c}" for c in d.columns]
    s.columns = [f"y{c}" for c in s.columns]
    return pd.concat([s, d], axis=1)
