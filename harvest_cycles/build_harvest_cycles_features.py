#!/usr/bin/env python3
"""
Harvest-cycle features from yearly Landsat NBR series (step 2 of 2; step 1 is preprocess_landsat.py).

Each point's yearly NBR series (1985-2020) is read on its own: one-year spikes are removed, straight-line
segments are fitted through the yearly values (LandTrendr-like, number of segments chosen by BIC), and cuts
(fast NBR falls), cycles (cuts that grow back) and summaries of the series are derived from them. The seasonal
features (wet minus dry season) come from the dated observations. All thresholds are in config.py.

Usage:
    python build_harvest_cycles_features.py --data DIR --points POINTS.csv --out FEATURES.csv
        [--label-column COLUMN] [--no-seasonal] [--calibrate-levels HANSEN.csv] [--set name=value ...]

    DIR holds the outputs of preprocess_landsat.py: series_nbr_median.csv.gz and, for the seasonal features,
    observations.csv.gz. POINTS.csv has a `pid` column and, optionally, the MapBiomas reference class
    (1 = plantation, 0 = natural forest) in the column named by --label-column.

Output: FEATURES.csv with one row per point: pid, label_mapbiomas (empty where unknown), then the 15
FEATURE_COLUMNS and the 4 seasonal columns (described in README.md), and FEATURES.settings.json with the command
line, settings, inputs and software versions used. Cropin's QC labels are confidential: the script refuses a
label for any point whose `dataset` column says "qc".

The functions below are also imported by harvest_cycles_findings.ipynb, so the analysis and this script always
compute the same features.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FeatureSettings, with_overrides, write_run_record  # noqa: E402

log = logging.getLogger("harvest_cycles.features")

DEFAULT_SETTINGS = FeatureSettings()

# Output columns, in order. The first four are the core features (H4); with the next six they form H-all.
FEATURE_COLUMNS = [
    "n_cycles", "regrow_speed", "forest_since_1985", "abrupt_start",
    "n_cuts", "years_since_last_cut", "max_drop", "s_frac_below_forest", "s_min", "s_max_drop3",
    "s_std",
    "has_cut", "has_cycle", "two_plus_cycles", "fast_regrowth",
]
BINARY_COLUMNS = ["forest_since_1985", "abrupt_start", "has_cut", "has_cycle", "two_plus_cycles", "fast_regrowth"]
INTEGER_COLUMNS = BINARY_COLUMNS + ["n_cycles", "n_cuts"]      # written as integers, empty where missing


def years_of(s: FeatureSettings = DEFAULT_SETTINGS) -> np.ndarray:
    return np.arange(s.first_year, s.last_year + 1)


def seasonal_columns(s: FeatureSettings = DEFAULT_SETTINGS) -> list[str]:
    return [f"season_diff_{idx}_{y0}_{y1}" for idx in s.seasonal_indices for y0, y1, _ in s.seasonal_windows]


# ----------------------------------------------------------------------------------------------- one pixel
def despike(y, s: FeatureSettings = DEFAULT_SETTINGS) -> np.ndarray:
    """Replace one-year spikes by the mean of their two neighbours, largest first."""
    y = np.asarray(y, float).copy()
    if len(y) < 3:
        return y
    for _ in range(len(y)):
        mid = (y[:-2] + y[2:]) / 2
        dev = y[1:-1] - mid
        spike = (np.abs(dev) >= s.despike_min_spike) & (np.abs(y[2:] - y[:-2]) < (1 - s.despike_threshold) * np.abs(dev))
        if not spike.any():
            break
        k = int(np.argmax(np.where(spike, np.abs(dev), -1)))
        y[k + 1] = mid[k]
    return y


def segment(t, y, s: FeatureSettings = DEFAULT_SETTINGS):
    """Best straight-line fit through the yearly values for 1..max_segments segments; keeps the lowest BIC.
    Returns (segment-end years, fitted segment-end values, fitted series)."""
    y = despike(y, s)
    n = len(t)
    if n < 6:
        return t[[0, -1]], y[[0, -1]], np.interp(t, t[[0, -1]], y[[0, -1]])
    cost = np.zeros((n, n))                          # error of one straight line from year i to year j
    for i in range(n - 1):
        for j in range(i + 1, n):
            line = y[i] + (y[j] - y[i]) * (t[i:j + 1] - t[i]) / (t[j] - t[i])
            cost[i, j] = np.sum((y[i:j + 1] - line) ** 2)
    kmax = min(s.max_segments, n - 2)
    best_cost = np.full((kmax + 1, n), np.inf)
    back = np.full((kmax + 1, n), -1)
    best_cost[0, 0] = 0.0
    for k in range(1, kmax + 1):                     # dynamic programming over segment ends
        for j in range(1, n):
            cand = best_cost[k - 1, :j] + cost[:j, j]
            i = int(np.argmin(cand))
            best_cost[k, j], back[k, j] = cand[i], i
    best = None
    for k in range(1, kmax + 1):
        if not np.isfinite(best_cost[k, n - 1]):
            continue
        ends, j = [n - 1], n - 1
        for kk in range(k, 0, -1):
            j = back[kk, j]
            ends.append(j)
        ends = sorted(ends)
        basis = np.column_stack([np.interp(t, t[ends], np.eye(len(ends))[m]) for m in range(len(ends))])
        values = np.linalg.lstsq(basis, y, rcond=None)[0]      # refit the segment-end values
        fit = basis @ values
        bic = n * np.log(np.sum((y - fit) ** 2) / n + 1e-6) + s.bic_penalty * 2 * k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, ends, values, fit)
    _, ends, values, fit = best
    return t[ends], values, fit


def time_to_reach(t, fit, start, target):
    """First year >= start at which the fitted series reaches target (linear interpolation), or None."""
    idx = np.where(t >= start)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if fit[a] >= target:
            return t[a]
        if fit[b] >= target:
            return t[a] + (target - fit[a]) / (fit[b] - fit[a]) * (t[b] - t[a])
    if len(idx) and fit[idx[-1]] >= target:
        return t[idx[-1]]
    return None


def find_cuts(tv, vv, t, fit, s: FeatureSettings = DEFAULT_SETTINGS) -> list[dict]:
    """Cuts: falls >= drop between consecutive segment ends at most max_cut_years apart. A cut is a cycle if the
    fitted series grows back cycle_regrow_fraction of the fall within max_cycle_years."""
    events = []
    for t0, t1, v0, v1 in zip(tv[:-1], tv[1:], vv[:-1], vv[1:]):
        if v0 - v1 >= s.drop and t1 - t0 <= s.max_cut_years:
            fall = v0 - v1
            half = time_to_reach(t, fit, t1, v1 + s.cycle_regrow_fraction * fall)
            most = time_to_reach(t, fit, t1, v1 + s.regrow_time_fraction * fall)
            events.append(dict(cut_year=t0 + 1, low_year=t1, drop=fall,
                               cycle=half is not None and half - t1 <= s.max_cycle_years,
                               regrow_years=None if most is None else most - t1))
    return events


def abrupt_start(t, y, s: FeatureSettings = DEFAULT_SETTINGS):
    """New stand on open ground: >= min_open_years at open-land level, then closed canopy within fast_rise_years.
    Returns (0 or 1, year closed canopy is reached)."""
    y = despike(y, s)
    low = y <= s.open_level
    for e in range(len(y)):
        if y[e] < s.canopy_level:
            continue
        before = np.where(low[:e])[0]
        if len(before) < s.min_open_years:
            continue
        last_open = before[-1]
        if np.sum(low[max(0, last_open - (s.open_lookback_values - 1)):last_open + 1]) < s.min_open_years:
            continue
        return int(t[e] - t[last_open] <= s.fast_rise_years), float(t[e])
    return 0, np.nan


# Keys of the dictionary pixel_features returns (a point with too few years gets only the first three).
PIXEL_KEYS = ["n_years", "events", "seg", "n_cuts", "n_cycles", "last_cut_year", "max_drop", "regrow_years",
              "years_below_forest", "forest_since_1985", "abrupt_start", "canopy_year",
              "s_std", "s_min", "s_max_drop3", "s_frac_below_forest"]


def pixel_features(values, s: FeatureSettings = DEFAULT_SETTINGS) -> dict:
    """Everything read from one yearly series (values aligned with first_year..last_year, NaN = no data).
    Besides the features, returns the cut events, the fitted segments and supporting values (n_years,
    last_cut_year, regrow_years, years_below_forest, canopy_year) used by the analysis."""
    v = np.asarray(values, float)
    t, y = years_of(s)[np.isfinite(v)].astype(float), v[np.isfinite(v)]
    if len(t) < s.min_years:                         # too few years to read a history
        return dict(n_years=len(t), events=[], seg=None)
    tv, vv, fit = segment(t, y, s)
    events = find_cuts(tv, vv, t, fit, s)
    cycles = [e for e in events if e["cycle"]]
    regrow = [e["regrow_years"] for e in cycles if e["regrow_years"] is not None]
    ys = despike(y, s)
    below = int(np.sum(ys < s.canopy_level))
    abrupt, canopy_year = abrupt_start(t, y, s)
    w = s.fall_window_years
    fall = [ys[max(0, i - w):i].max() - ys[i] for i in range(1, len(ys))]
    return dict(n_years=len(t), events=events, seg=(tv, vv, t, fit),
                n_cuts=len(events), n_cycles=len(cycles),
                last_cut_year=max((e["cut_year"] for e in events), default=np.nan),
                max_drop=max((e["drop"] for e in events), default=0.0),
                regrow_years=min(regrow) if regrow else np.nan,
                years_below_forest=below, forest_since_1985=int(not events and below == 0),
                abrupt_start=abrupt, canopy_year=canopy_year,
                s_std=ys.std(), s_min=ys.min(), s_max_drop3=max(fall), s_frac_below_forest=np.mean(ys < s.canopy_level))


# ----------------------------------------------------------------------------------------------- many pixels
def series_features(series: pd.DataFrame, s: FeatureSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """pixel_features for every row of a yearly series (index = pid, columns = years), plus the yes/no features,
    regrowth speed and years since the last cut. Points with fewer than min_years of data get empty features."""
    series = series.reindex(columns=years_of(s))
    rows = [dict(pid=pid, **pixel_features(row, s)) for pid, row in zip(series.index, series.to_numpy(float))]
    f = pd.DataFrame(rows, columns=["pid", *PIXEL_KEYS]).set_index("pid")   # every column, even if no point has data
    ok = f.n_years >= s.min_years

    def flag(condition):
        return condition.astype(int).where(ok)

    f["has_cut"] = flag(f.n_cuts >= 1)
    f["has_cycle"] = flag(f.n_cycles >= 1)
    f["two_plus_cycles"] = flag(f.n_cycles >= 2)
    f["fast_regrowth"] = flag((f.n_cycles >= 1) & (f.regrow_years <= s.fast_regrowth_years))
    f["regrow_speed"] = pd.Series(np.where((f.n_cycles >= 1) & (f.regrow_years > 0),
                                           1 / f.regrow_years.clip(lower=s.min_regrow_years), 0.0), index=f.index).where(ok)
    f["years_since_last_cut"] = (s.last_year - f.last_cut_year).fillna(s.years_since_last_cut_if_none).where(ok)
    return f


def seasonal_features(observations: pd.DataFrame, s: FeatureSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Median wet-season minus median dry-season value of each index, per point and calendar year, then the median
    over the years of each window that have both seasons (empty if fewer than the window's min_years).
    `observations`: one row per clear observation with pid, year, month and the index columns."""
    o = observations[observations["year"].between(s.first_year, s.last_year)].copy()
    o["season"] = np.where(o["month"].isin(s.seasonal_dry_months), "dry", "wet")
    out = {}
    for idx in s.seasonal_indices:
        oi = o[o[idx].notna()]
        for y0, y1, min_years in s.seasonal_windows:
            d = (oi[oi["year"].between(y0, y1)].groupby(["pid", "year", "season"])[idx].median()
                 .unstack("season").reindex(columns=["dry", "wet"]).dropna())
            diff = (d["wet"] - d["dry"]).groupby(level="pid")
            out[f"season_diff_{idx}_{y0}_{y1}"] = diff.median().where(diff.size() >= min_years)
    return pd.DataFrame(out)


def calibrate_levels(series: pd.DataFrame, hansen: pd.DataFrame, s: FeatureSettings = DEFAULT_SETTINGS):
    """Closed-canopy and open-land levels of a yearly series, from Hansen tree cover only (never from labels).
    `hansen`: pid, tc2000, gain, lossyear (1..24 = 2001..2024) and loss_3x3_le2020_n (neighbours with a loss up to
    2020). Returns (canopy_level, open_level, number of closed-canopy pixels, number of open-land pixels), counting
    only pixels of the series. Raises ValueError if either kind has fewer than calibration_min_pixels pixels.
    Levels recalibrated on a batch differ from batch to batch (and from the defaults the models were trained with)."""
    last_loss = s.last_year - 2000
    closed = hansen[(hansen.tc2000 >= s.closed_min_tree_cover) & ~hansen.lossyear.between(1, last_loss)
                    & (hansen.gain == 0) & (hansen.loss_3x3_le2020_n == 0)].pid
    open_land = hansen[(hansen.tc2000 <= s.open_max_tree_cover) & (hansen.gain == 0)].pid
    in_closed, in_open = series.index.isin(closed), series.index.isin(open_land)
    n_closed, n_open = int(in_closed.sum()), int(in_open.sum())
    if min(n_closed, n_open) < s.calibration_min_pixels:
        raise ValueError(f"too few pixels to calibrate the levels: {n_closed} closed-canopy and {n_open} open-land "
                         f"pixels (at least {s.calibration_min_pixels} of each needed); keep the default levels")
    v_closed = series.loc[in_closed, [y for y in years_of(s) if y >= s.closed_first_year]].to_numpy().ravel()
    v_open = series.loc[in_open, list(s.open_years)].to_numpy().ravel()
    v_closed, v_open = v_closed[np.isfinite(v_closed)], v_open[np.isfinite(v_open)]
    if not (len(v_closed) and len(v_open)):
        raise ValueError("no yearly values at the calibration pixels in the calibration years")
    canopy = np.percentile(v_closed, s.closed_percentile)
    open_level = min(np.percentile(v_open, s.open_percentile), canopy - s.drop / 2)
    return float(canopy), float(open_level), n_closed, n_open


def build_features(series: pd.DataFrame, observations: pd.DataFrame | None,
                   s: FeatureSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """The output table: FEATURE_COLUMNS from the yearly NBR series, plus the seasonal features when observations
    are given. Index = pid, in the order of `series`."""
    f = series_features(series, s)[FEATURE_COLUMNS]
    if observations is not None:
        f = f.join(seasonal_features(observations, s).reindex(f.index))
    return f


# ----------------------------------------------------------------------------------------------- input / output
def read_series(path, s: FeatureSettings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """A yearly series written by preprocess_landsat.py: index = pid, one column per year."""
    series = pd.read_csv(path, index_col="pid", dtype={"pid": str})      # pids stay text: "007" is not 7
    series.columns = series.columns.astype(int)
    missing = [y for y in years_of(s) if y not in series.columns]
    if missing:
        raise SystemExit(f"{path}: no column for years {[int(y) for y in missing]}; first_year and last_year must "
                         "lie within the years of the series")
    return series.reindex(columns=years_of(s))


CALIBRATION_SETTINGS = ["closed_min_tree_cover", "closed_percentile", "closed_first_year", "open_max_tree_cover",
                        "open_percentile", "open_years", "calibration_min_pixels"]


def read_labels(points: pd.DataFrame, column: str | None) -> pd.Series:
    """The MapBiomas class per pid (Int64: 1 plantation, 0 natural forest, <NA> unknown). Refuses values other than
    0, 1 and empty, and any label on a Cropin QC point (dataset == "qc"): those labels are confidential."""
    index = pd.Index(points["pid"], name="pid")
    if column is None:
        return pd.Series(pd.NA, index=index, dtype="Int64")
    if column not in points:
        raise SystemExit(f"--label-column {column!r}: no such column in the points file "
                         f"(columns: {', '.join(points.columns)})")
    raw = points[column]
    text = raw.astype("string").str.strip()
    bad = raw.notna() & ~text.isin(["0", "1", "0.0", "1.0"])
    if bad.any():
        raise SystemExit(f"--label-column {column!r}: labels must be 0, 1 or empty; found "
                         f"{sorted(text[bad].unique().tolist())[:5]}")
    if "dataset" in points and (raw.notna() & (points["dataset"].astype("string").str.strip().str.lower() == "qc")).any():
        raise SystemExit("the points file has labels for Cropin QC points (dataset == 'qc'). Those labels are "
                         "confidential and must not be written: use public_data/harvest_cycles/points.csv, whose QC "
                         "labels are empty.")
    labels = pd.to_numeric(raw).astype("Int64")
    labels.index = index
    return labels


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="folder with series_nbr_median.csv.gz (and observations.csv.gz) from preprocess_landsat.py")
    p.add_argument("--points", type=Path, required=True, help="points CSV with a pid column")
    p.add_argument("--out", type=Path, required=True, help="output CSV")
    p.add_argument("--label-column", metavar="COLUMN",
                   help="column of the points CSV with the MapBiomas class, 1 = plantation, 0 = natural forest "
                        "(label in public_data/harvest_cycles/points.csv); without it the label column is empty")
    p.add_argument("--calibrate-levels", type=Path, metavar="HANSEN.csv",
                   help="recalibrate canopy_level and open_level from Hansen values at these points "
                        "(hansen_gfc.csv from preprocess_landsat.py --hansen)")
    p.add_argument("--no-seasonal", action="store_true", help="skip the seasonal features (no observations file)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a FeatureSettings value from config.py for this run (repeatable)")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        s = with_overrides(FeatureSettings(), args.overrides)
    except ValueError as e:
        p.error(str(e))
    overridden = {o.partition("=")[0].strip() for o in args.overrides}
    if args.calibrate_levels and overridden & {"canopy_level", "open_level"}:
        p.error("--calibrate-levels computes canopy_level and open_level: do not also --set them")
    if not args.calibrate_levels and overridden & set(CALIBRATION_SETTINGS):
        log.warning("%s: calibration settings act only through --calibrate-levels; without it the levels stay at "
                    "canopy_level and open_level",
                    ", ".join(sorted(overridden & set(CALIBRATION_SETTINGS))))
    if args.out.is_dir():
        p.error(f"--out {args.out} is a folder; give a file name such as {args.out / 'harvest_cycles_features.csv'}")
    points = pd.read_csv(args.points, dtype={"pid": str})
    if "pid" not in points or points["pid"].isna().any() or points["pid"].duplicated().any():
        raise SystemExit(f"{args.points}: needs a pid column with a unique, non-empty id on every row")
    labels = read_labels(points, args.label_column)
    series_path = args.data / "series_nbr_median.csv.gz"
    obs_path = None if args.no_seasonal else args.data / "observations.csv.gz"
    if not series_path.exists():
        raise SystemExit(f"not found: {series_path} (run preprocess_landsat.py first, or check --data)")
    if obs_path is not None and not obs_path.exists():
        raise SystemExit(f"not found: {obs_path} (run preprocess_landsat.py first, or pass --no-seasonal)")

    series = read_series(series_path, s)
    absent = ~points["pid"].isin(series.index)
    if absent.all() and len(points):
        raise SystemExit(f"none of the {len(points)} points is in {series_path}: wrong --data folder, or a different "
                         "points file than the one given to preprocess_landsat.py?")
    if absent.any():
        log.warning("%d points are not in %s and get empty features (e.g. %s)", int(absent.sum()), series_path.name,
                    ", ".join(points["pid"][absent].head(5)))
    series = series.reindex(points["pid"])
    extra, inputs = {}, dict(series=series_path, observations=obs_path, points=args.points)
    if args.calibrate_levels:
        hansen = pd.read_csv(args.calibrate_levels, dtype={"pid": str})
        try:
            canopy, open_level, n_closed, n_open = calibrate_levels(series, hansen, s)
        except ValueError as e:
            raise SystemExit(f"--calibrate-levels: {e}") from None
        s = replace(s, canopy_level=canopy, open_level=open_level)
        extra["calibration"] = dict(closed_canopy_pixels=n_closed, open_land_pixels=n_open)
        inputs["calibration_hansen"] = args.calibrate_levels
        log.info("calibrated levels: closed canopy %.4f (%d pixels), open land %.4f (%d pixels)",
                 canopy, n_closed, open_level, n_open)
    observations = None
    if obs_path is not None:
        observations = pd.read_csv(obs_path, usecols=["pid", "year", "month", *s.seasonal_indices], dtype={"pid": str})
        observations = observations[observations["pid"].isin(points["pid"])]

    log.info("computing features for %d points (about 15 seconds per 2,000 points)", len(points))
    features = build_features(series, observations, s)
    for c in INTEGER_COLUMNS:
        features[c] = features[c].astype("Int64")
    out = features.reindex(points["pid"])
    out.insert(0, "label_mapbiomas", labels.reindex(out.index))
    out.index.name = "pid"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)
    settings_path = write_run_record(args.out, s, inputs, extra, script=Path(__file__).name, command=argv)
    n_short = int(out[FEATURE_COLUMNS[0]].isna().sum()) - int(absent.sum())
    log.info("wrote %s (%d points x %d features; %d with fewer than %d years of data, %d not in the data) and %s",
             args.out, len(out), out.shape[1] - 1, n_short, s.min_years, int(absent.sum()), settings_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
