"""Reproduce the per-point predictions of harvest_cycles/harvest_cycles_findings.ipynb on the MapBiomas points.

The code below is copied from the notebook (sections 1, 2 and 3.4) with two changes: Cropin's QC labels are not read
(they are not needed for the MapBiomas results), and only the first cross-validation split (seed 42) is run, the split
the notebook uses for its "plantations caught at <= 5% forest flagged" figures. Output: the out-of-fold score of every
MapBiomas point and its call at that threshold, in species_split/results/model_predictions_val.csv.

Also writes every detected cut at those points to species_split/results/cut_events_val.csv.

Run from anywhere:  python3 species_split/scripts/notebook_predictions.py   (about a minute)
"""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

INPUTS = ["points.csv", "hansen_gfc.csv", "val_label_history.csv"] + [
    f"series_{name}.csv.gz" for name in ["nbr_median", "ndvi_median", "nbr_dry_median", "nbr_wet_median", "nbr_max"]]


def find_up(rel):
    # `rel` in the current folder or the nearest parent folder that has it (the notebook runs from anywhere in the repository)
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / rel).exists():
            return d / rel
    return Path(rel)


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "species_split" / "results"
DATA = REPO / "public_data/harvest_cycles"
TEAM_TABLE = REPO / "data_exploration/cropin_model_features_2022.csv"
YEARS = np.arange(1985, 2021)        # 36 years, up to the EUDR cutoff
SEED = 42
PLANT, FOREST, CUT, GREY = "#c8800d", "#197037", "#b8358a", "#7d8a83"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e8e4", "font.size": 10, "legend.frameon": False})


def pct(x):
    return "–" if pd.isna(x) else (f"{100 * x:.1f}%" if 0 < x < 0.1 else f"{100 * x:.0f}%")


def num(x):
    # two significant digits, without scientific notation for ordinary sizes
    return f"{x:.0f}" if x >= 10 else (f"{x:.1f}" if x >= 1 else f"{x:.2g}")


def pval(p):
    return "–" if pd.isna(p) else f"{p:.2g}"


pts = pd.read_csv(DATA / "points.csv")
pts["label"] = pts["label"].fillna(-1).astype(int)   # QC labels are not available here (-1)              # dataset: val (MapBiomas) or qc (Cropin); label: 1 = plantation

def read_series(name):
    s = pd.read_csv(DATA / f"series_{name}.csv.gz", index_col=0).reindex(pts["pid"])
    s.columns = s.columns.astype(int)
    return s                                        # one row per point, one column per year 1985..2020


nbr = read_series("nbr_median")
hansen = pd.read_csv(DATA / "hansen_gfc.csv")        # Hansen Global Forest Change: tree cover 2000, gain, loss year
history = pd.read_csv(DATA / "val_label_history.csv")  # MapBiomas reference class per year (MapBiomas points only)
team = pd.read_csv(TEAM_TABLE)                      # the team's features and MapBiomas 10 m map calls
team["ever_farm"] = (team["ever_farm_mean"] >= 0.5).astype(int)


def levels(series, drop):
    # closed-canopy and open-land levels of a yearly series, from Hansen tree cover only
    closed = hansen[(hansen.tc2000 >= 90) & ~hansen.lossyear.between(1, 20) & (hansen.gain == 0)
                    & (hansen.loss_3x3_le2020_n == 0)].pid
    open_land = hansen[(hansen.tc2000 <= 10) & (hansen.gain == 0)].pid
    v_closed = series.loc[series.index.isin(closed), [y for y in YEARS if y >= 2000]].to_numpy().ravel()
    v_open = series.loc[series.index.isin(open_land), [1999, 2000, 2001]].to_numpy().ravel()
    canopy = np.percentile(v_closed[np.isfinite(v_closed)], 5)
    return canopy, min(np.percentile(v_open[np.isfinite(v_open)], 25), canopy - drop / 2), len(closed), len(open_land)


CANOPY, OPEN, n_closed, n_open = levels(nbr, drop=0.20)
print(f"Closed-canopy level (5th percentile at stable closed-canopy pixels): NBR {CANOPY:.3f}  (from {n_closed:,} pixels)")
print(f"Open-land level (25th percentile at open-land pixels):              NBR {OPEN:.3f}  (from {n_open} pixels)")


DROP, MAX_CUT_YEARS, MAX_REGROW_YEARS = 0.20, 3, 10


def despike(y, thr=0.9, min_spike=0.05):
    # replace one-year spikes by the mean of their two neighbours, largest first
    y = np.asarray(y, float).copy()
    if len(y) < 3:
        return y
    for _ in range(len(y)):
        mid = (y[:-2] + y[2:]) / 2
        s = y[1:-1] - mid
        spike = (np.abs(s) >= min_spike) & (np.abs(y[2:] - y[:-2]) < (1 - thr) * np.abs(s))
        if not spike.any():
            break
        k = int(np.argmax(np.where(spike, np.abs(s), -1)))
        y[k + 1] = mid[k]
    return y


def segment(t, y, max_segments=18):
    # best straight-line fit through yearly points for 1..max_segments segments; keep the one with the lowest BIC
    y = despike(y)
    n = len(t)
    if n < 6:
        return t[[0, -1]], y[[0, -1]], np.interp(t, t[[0, -1]], y[[0, -1]])
    cost = np.zeros((n, n))                         # error of one straight line from year i to year j
    for i in range(n - 1):
        for j in range(i + 1, n):
            line = y[i] + (y[j] - y[i]) * (t[i:j + 1] - t[i]) / (t[j] - t[i])
            cost[i, j] = np.sum((y[i:j + 1] - line) ** 2)
    kmax = min(max_segments, n - 2)
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
        bic = n * np.log(np.sum((y - fit) ** 2) / n + 1e-6) + 2 * k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, ends, values, fit)
    _, ends, values, fit = best
    return t[ends], values, fit


def time_to_reach(t, fit, start, target):
    # first year >= start at which the fitted line reaches target (linear interpolation), or None
    idx = np.where(t >= start)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if fit[a] >= target:
            return t[a]
        if fit[b] >= target:
            return t[a] + (target - fit[a]) / (fit[b] - fit[a]) * (t[b] - t[a])
    if len(idx) and fit[idx[-1]] >= target:
        return t[idx[-1]]
    return None


def find_cuts(tv, vv, t, fit, drop=DROP):
    # cuts = falls >= drop between consecutive segment ends at most 3 years apart; cycle = regrows half within 10 years
    events = []
    for t0, t1, v0, v1 in zip(tv[:-1], tv[1:], vv[:-1], vv[1:]):
        if v0 - v1 >= drop and t1 - t0 <= MAX_CUT_YEARS:
            fall = v0 - v1
            half = time_to_reach(t, fit, t1, v1 + 0.5 * fall)
            most = time_to_reach(t, fit, t1, v1 + 0.8 * fall)
            events.append(dict(cut_year=t0 + 1, low_year=t1, drop=fall,
                               cycle=half is not None and half - t1 <= MAX_REGROW_YEARS,
                               regrow_years=None if most is None else most - t1))
    return events


def abrupt_start(t, y, canopy, open_level, min_open_years=3, fast_rise_years=5):
    # planting on open land: >= 3 open-land years, then closed canopy within 5 years. Returns (0/1, canopy year)
    y = despike(y)
    low = y <= open_level
    for e in range(len(y)):
        if y[e] < canopy:
            continue
        before = np.where(low[:e])[0]
        if len(before) < min_open_years:
            continue
        last_open = before[-1]
        if np.sum(low[max(0, last_open - 7):last_open + 1]) < min_open_years:
            continue
        return int(t[e] - t[last_open] <= fast_rise_years), float(t[e])
    return 0, np.nan


def pixel_features(row, canopy, open_level, drop=DROP, max_segments=18):
    v = row.to_numpy(float)
    t, y = YEARS[np.isfinite(v)].astype(float), v[np.isfinite(v)]
    if len(t) < 8:                                   # too few years to read a history
        return dict(n_years=len(t), events=[], seg=None)
    tv, vv, fit = segment(t, y, max_segments)
    events = find_cuts(tv, vv, t, fit, drop)
    cycles = [e for e in events if e["cycle"]]
    regrow = [e["regrow_years"] for e in cycles if e["regrow_years"] is not None]
    ys = despike(y)
    below = int(np.sum(ys < canopy))
    abrupt, canopy_year = abrupt_start(t, y, canopy, open_level)
    fall3 = [ys[max(0, i - 3):i].max() - ys[i] for i in range(1, len(ys))]
    return dict(n_years=len(t), events=events, seg=(tv, vv, t, fit),
                n_cuts=len(events), n_cycles=len(cycles),
                last_cut_year=max((e["cut_year"] for e in events), default=np.nan),
                max_drop=max((e["drop"] for e in events), default=0.0),
                regrow_years=min(regrow) if regrow else np.nan,
                years_below_forest=below, forest_since_1985=int(not events and below == 0),
                abrupt_start=abrupt, canopy_year=canopy_year,
                s_std=ys.std(), s_min=ys.min(), s_max_drop3=max(fall3), s_frac_below_forest=np.mean(ys < canopy))


def all_features(series, canopy, open_level, **kw):
    f = pd.DataFrame([dict(pid=pid, **pixel_features(row, canopy, open_level, **kw)) for pid, row in series.iterrows()])
    return f.set_index("pid")


feat = all_features(nbr, CANOPY, OPEN)              # about a minute
feat["has_cut"] = (feat.n_cuts >= 1).astype(int)
feat["has_cycle"] = (feat.n_cycles >= 1).astype(int)
feat["two_plus_cycles"] = (feat.n_cycles >= 2).astype(int)
feat["fast_regrowth"] = ((feat.n_cycles >= 1) & (feat.regrow_years <= 4)).astype(int)
feat["regrow_speed"] = np.where((feat.n_cycles >= 1) & (feat.regrow_years > 0), 1 / feat.regrow_years.clip(lower=1), 0.0)
feat["years_since_last_cut"] = (2020 - feat.last_cut_year).fillna(40)

df = (pts.merge(feat.drop(columns=["events", "seg"]), left_on="pid", right_index=True, how="left")
         .merge(hansen[["pid", "tc2000", "gain", "lossyear"]], on="pid", how="left"))
df["hansen_loss"] = df.lossyear.between(1, 20).astype(int)        # Hansen losses after 2020 are ignored
df["hansen_gain"] = df.gain.astype(int)
df["tc2000"] = df.tc2000.astype(float)
# clusters for honest p-values: municipalities (MapBiomas points) and 0.1-degree cells (QC clearings)
cell = "g" + (df.lon / 0.1).round().astype(int).astype(str) + "_" + (df.lat / 0.1).round().astype(int).astype(str)
df["cluster"] = np.where(df.dataset == "val", df.CD_MUN.astype(str), cell)
val = (df[df.dataset == "val"].merge(team.drop(columns=["lat", "lon", "CD_MUN", "plantation"]), on="pt_id", how="left")
         .merge(history[["pid"] + [f"CLASS_{y}" for y in range(1985, 2023)]], on="pid", how="left"))
qc = df[df.dataset == "qc"].copy()
print(f"Harvest features computed for {feat.n_cuts.notna().sum():,} of {len(feat):,} points "
      f"(the rest have fewer than 8 years of data: {(feat.n_years < 8).sum()})")
first_cut = feat.events.map(lambda ev: min((e["cut_year"] for e in ev), default=np.nan))
ab = val[(val.label == 1) & (val.abrupt_start == 1)]
print(f"Plantation abrupt starts that come after a detected cut (a site left bare after a harvest): "
      f"{(first_cut.reindex(ab.pid).to_numpy() < ab.canopy_year.to_numpy()).mean():.0%} of {len(ab)}")


def full_series(series):
    # the 36 yearly values (gaps interpolated) and their 35 year-to-year changes
    s = series.interpolate(axis=1, limit_direction="both")
    d = s.diff(axis=1).iloc[:, 1:]
    d.columns, s.columns = [f"d{c}" for c in d.columns], [f"y{c}" for c in s.columns]
    return pd.concat([s, d], axis=1)


full = full_series(nbr)
H4 = ["n_cycles", "regrow_speed", "forest_since_1985", "abrupt_start"]
HALL = H4 + ["n_cuts", "years_since_last_cut", "max_drop", "s_frac_below_forest", "s_min", "s_max_drop3"]
HANSEN = ["hansen_loss", "hansen_gain", "tc2000"]
TEAM_FEATURES = ["height_stdDev", "ndvi_aug_oct_mean", "sag_oct_nov_mean", "sag_swing_mean", "ever_farm"]
FULL = list(full.columns)
SETS = {"H4: four harvest features": H4, "H-all: ten harvest features": HALL,
        "H-all + variability (post hoc)": HALL + ["s_std"], "36 yearly values + changes": FULL,
        "Hansen (comparison)": HANSEN, "H-all + Hansen": HALL + HANSEN, "Team's features (reference)": TEAM_FEATURES,
        "Team's features + H-all": TEAM_FEATURES + HALL}

V = val.dropna(subset=TEAM_FEATURES).merge(full, left_on="pid", right_index=True, how="left").reset_index(drop=True)
for f in HALL + HANSEN + ["s_std"]:
    V[f] = V[f].astype(float).fillna(V[f].median())
V[FULL] = V[FULL].fillna(V[FULL].median())
print(f"Model table: {len(V):,} of {len(val):,} MapBiomas points ({len(val) - len(V)} lacks a team feature); "
      f"plantations the MapBiomas 10 m map calls forest among them: {(V.group == 'missed plantation').sum()}")


def make_model(kind, seed=SEED):
    if kind == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1)



y, municipality = V.label.to_numpy(), V.CD_MUN.to_numpy()
rep = SEED
folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(V, y, municipality))
oof = {}
for name, cols, kind in [("H-all: ten harvest features", HALL, "RF"), ("H-all: ten harvest features", HALL, "LR"),
                         ("H4: four harvest features", H4, "RF"), ("36 yearly values + changes", FULL, "RF"),
                         ("H-all + variability (post hoc)", HALL + ["s_std"], "RF")]:
    X = V[cols].to_numpy(float)
    o = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(folds):
        o[te] = make_model(kind, rep).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof[(name, kind)] = o


def recall_at_fpr(y, s, max_fpr=0.05):
    fpr, tpr, thr = roc_curve(y, s)
    k = np.where(fpr <= max_fpr)[0]
    j = k[np.argmax(tpr[k])]
    return tpr[j], fpr[j], thr[j]


out = V[["pid", "pt_id", "lon", "lat", "label", "CD_MUN", "state", "group", "mb_pred", "n_cuts", "n_cycles", "has_cut", "has_cycle",
         "two_plus_cycles", "abrupt_start", "forest_since_1985", "last_cut_year", "canopy_year", "s_std", "regrow_years",
         "hansen_loss", "hansen_gain", "tc2000"]].copy()
for (name, kind), o in oof.items():
    key = {"H-all: ten harvest features": "hall", "H4: four harvest features": "h4", "36 yearly values + changes": "full",
           "H-all + variability (post hoc)": "hallv"}[name] + "_" + kind.lower()
    rec, fp, thr = recall_at_fpr(y, o)
    out[key + "_score"] = o
    out[key + "_pred"] = (o >= thr).astype(int)
    print(f"{name} {kind}: AUC {roc_auc_score(y, o):.3f}, recall {rec:.1%} at FPR {fp:.1%}, threshold {thr:.3f}")
out.to_csv(RESULTS / "model_predictions_val.csv", index=False)
# every detected cut at the MapBiomas points (for the cut intervals and regrowth times by species)
pd.DataFrame([dict(pid=pid, **e) for pid, ev in feat.events.items() if pid.startswith("V") for e in ev]).to_csv(
    RESULTS / "cut_events_val.csv", index=False)
