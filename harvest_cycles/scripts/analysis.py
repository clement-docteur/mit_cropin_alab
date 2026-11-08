#!/usr/bin/env python3
"""
Step 6: can harvest cycles tell plantations from natural forest, and how much do they add?

Usage: python analysis.py <data_dir> <repo_harvest_cycles_dir>

Primary configuration (chosen before looking at results): NBR, annual median of all clear observations,
LandTrendr-style segmentation ("S"), default thresholds in harvest_lib.HarvestParams. Every other index /
aggregation / method / threshold is reported as a comparison or sensitivity analysis.
Revised after an independent code review (the first results had been seen, so these are disclosed as post hoc):
  - segment cap 6 -> 18: with 6 segments a pixel can hold at most ~2 cuts, and on regular 5-7-year
    rotations the BIC returns a straight line unless ~15 segments are allowed. The cap-6 version is kept
    as method "S6" (and caps 6/12/18 in the sensitivity grid).
  - calibration: closed canopy now uses only information up to 2020 (Hansen loss 2001-2020) and only the
    2000-2020 values (tree cover is known from 2000). Open land is now calibrated on open land itself
    (pixels with <= 10% Hansen tree cover in 2000, their 1999-2001 values, 25th percentile); the first
    definition (closed-canopy median minus two cut depths) sat at bare-soil level and missed pasture.
  - rules (R): a cut can no longer be counted twice; same-pass duplicate observations are merged.

Datasets
  VAL = MapBiomas validation points in PR/SC/RS (2022 label forest(3) or plantation(9), no edge
        points); the same 1,831 points as data_exploration/cropin_model_features_2022.csv.
        Groups vs MapBiomas 10 m (collection 4) come from that file.
  QC  = Cropin's 433 hand-checked clearing polygons (310 plantation, 123 forest), one Landsat pixel
        at each polygon's representative point. Confidential: only aggregate numbers are written
        to the repo; per-polygon tables stay in <data_dir>.

All harvest features use Landsat data up to the end of 2020 only.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

D, OUT = sys.argv[1], sys.argv[2]
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
PRIV = os.path.join(D, "private_results")          # per-polygon QC outputs (git-ignored folder)
for p in (RES, FIG, PRIV):
    os.makedirs(p, exist_ok=True)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SEED = 42
PRIMARY = dict(index="nbr", agg="median", method="S")
LOG = open(os.path.join(RES, "analysis_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ----------------------------------------------------------------------------- style (dataviz palette)
C_FOREST, C_PLANT, C_MISSED = "#1baf7a", "#eb6834", "#2a78d6"     # validated set, fixed per entity
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- stats helpers
def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def cluster_boot(y, s, cl, fn=roc_auc_score, n_boot=2000, seed=SEED):
    """95% CI of fn(y, s) resampling whole clusters (municipalities / grid cells)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(cl)
    idx_by = [np.where(codes == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_by[p] for p in rng.integers(0, len(uniq), len(uniq))])
        if y[ii].min() == y[ii].max():
            continue
        vals.append(fn(y[ii], s[ii]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def odds_ratio(a, b, c, d):
    """a = plantation with feature, b = plantation without, c = forest with, d = forest without."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)


def clustered_logit(X, y, groups, names):
    """Logit with municipality-clustered (sandwich) SEs: p-values that respect spatial clustering."""
    import statsmodels.api as sm
    Xc = sm.add_constant(np.asarray(X, float), has_constant="add")
    try:
        r = sm.GLM(y, Xc, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    except Exception as e:  # perfect separation etc.
        return None, str(e)
    tab = pd.DataFrame({"coef": r.params, "se_cluster": r.bse, "p_cluster": r.pvalues,
                        "OR": np.exp(r.params), "OR_lo": np.exp(r.params - 1.96 * r.bse),
                        "OR_hi": np.exp(r.params + 1.96 * r.bse)}, index=["const"] + list(names))
    return tab, r


def recall_at_fpr(y, s, max_fpr):
    fpr, tpr, thr = roc_curve(y, s)
    k = np.where(fpr <= max_fpr)[0]
    j = k[np.argmax(tpr[k])]
    return tpr[j], fpr[j], thr[j]


# ----------------------------------------------------------------------------- load
pts = pd.read_csv(f"{D}/points.csv")
feat_all = pd.read_csv(f"{D}/features_main.csv.gz")
grid = pd.read_csv(f"{D}/features_grid.csv.gz")
han = pd.read_csv(f"{D}/hansen_gfc.csv")
cal = pd.read_csv(f"{D}/calibration_levels.csv")
offs = pd.read_csv(f"{D}/series_sensor_offsets.csv")
cnt = pd.read_csv(f"{D}/series_obs_counts.csv.gz", index_col=0)
hist = pd.read_csv(f"{D}/val_label_history.csv")
team = pd.read_csv(os.path.join(REPO, "data_exploration", "cropin_model_features_2022.csv"))
team["ever_farm"] = (team["ever_farm_mean"] >= 0.5).astype(int)
series = {a: pd.read_csv(f"{D}/series_nbr_{a}.csv.gz", index_col=0) for a in hl.AGGREGATIONS}
for s in series.values():
    s.columns = s.columns.astype(int)

han["hansen_loss_le2020"] = han["lossyear"].between(1, 20).astype(int)        # cutoff: losses after 2020 ignored
han["hansen_gain"] = han["gain"].astype(int)
han["hansen_loss3x3_le2020"] = (han["loss_3x3_le2020_n"] > 0).astype(int)
han["tc2000"] = han["tc2000"].astype(float)


def features_for(index, agg, method):
    f = feat_all[(feat_all["index"] == index) & (feat_all["agg"] == agg) & (feat_all["method"] == method)].copy()
    f["has_cut"] = (f["n_cuts"] >= 1).astype(int)
    f["has_cycle"] = (f["n_cycles"] >= 1).astype(int)
    f["two_plus_cycles"] = (f["n_cycles"] >= 2).astype(int)
    f["fast_regrowth"] = ((f["n_cycles"] >= 1) & (f["regrow_years"] <= 4)).astype(int)
    f["regrow_speed"] = np.where((f["n_cycles"] >= 1) & (f["regrow_years"] > 0), 1 / f["regrow_years"].clip(lower=1), 0.0)
    f["years_since_last_cut"] = (hl.CUTOFF - f["last_cut_year"]).fillna(40)
    return f


F = features_for(**PRIMARY)
F6 = features_for("nbr", "median", "S6")
F6 = F6[["pid", "n_cycles", "regrow_speed", "forest_since_1985", "abrupt_start", "n_cuts", "years_since_last_cut",
         "max_drop"]].set_index("pid").add_suffix("_s6").reset_index()
df = pts.merge(F, on="pid", how="left").merge(F6, on="pid", how="left").merge(han, on="pid", how="left")
df["cluster"] = np.where(df["dataset"] == "val", df["CD_MUN"].astype(str),
                         "g" + (df["lon"] / 0.1).round().astype(int).astype(str) + "_" + (df["lat"] / 0.1).round().astype(int).astype(str))
val = df[df.dataset == "val"].merge(team.drop(columns=["lat", "lon", "CD_MUN", "plantation"]), on="pt_id", how="left")
val = val.merge(hist[["pid"] + [f"CLASS_{y}" for y in range(1985, 2023)]], on="pid", how="left")
qc = df[df.dataset == "qc"].copy()
log(f"VAL n={len(val)} (plantation {val.label.sum()}), QC n={len(qc)} (plantation {qc.label.sum()})")
log("VAL groups:", val["group"].value_counts().to_dict())

# ============================================================================= 1. data quality
log("\n=== 1. Landsat coverage and calibration")
c = cnt.reindex(pts["pid"])
log(f"clear obs per point-year: median {np.median(c.values):.0f}, mean {c.values.mean():.2f}; "
    f"point-years with none: {(c.values == 0).mean():.3f}")
by_year = (c == 0).mean()
log("share of points with no clear observation, worst years:", by_year.sort_values(ascending=False).head(6).round(3).to_dict())
log("points with < 25 of 36 years observed:", int(((c > 0).sum(axis=1) < 25).sum()))
log("sensor offsets (subtracted):\n" + offs.round(4).to_string(index=False))
log("calibration (label-free, Hansen closed canopy):\n" + cal.round(3).to_string(index=False))
qrow = dict(n_points=len(pts), clear_obs_per_point_year_median=float(np.median(c.values)),
            share_point_years_missing=float((c.values == 0).mean()))
pd.DataFrame([qrow]).to_csv(os.path.join(RES, "data_coverage.csv"), index=False)
cal.to_csv(os.path.join(RES, "calibration_levels.csv"), index=False)
offs.to_csv(os.path.join(RES, "sensor_offsets.csv"), index=False)

# ============================================================================= 2. method comparison
log("\n=== 2. Which way of using the time series separates best? (all configurations, default thresholds)")
rows = []
for (i, a, m), f in feat_all.groupby(["index", "agg", "method"]):
    g = pts[["pid", "dataset", "label", "CD_MUN"]].merge(f, on="pid")
    for ds, gg in g.groupby("dataset"):
        y = gg["label"].to_numpy()
        nc = gg["n_cuts"].fillna(0).to_numpy()
        ncy = gg["n_cycles"].fillna(0).to_numpy()
        fs = gg["forest_since_1985"].fillna(0).to_numpy()
        md = gg["s_max_drop3"].fillna(0).to_numpy()
        rows.append(dict(index=i, agg=a, method=m, dataset=ds, n=len(gg),
                         tpr_cut=(nc[y == 1] >= 1).mean(), fpr_cut=(nc[y == 0] >= 1).mean(),
                         tpr_cycle=(ncy[y == 1] >= 1).mean(), fpr_cycle=(ncy[y == 0] >= 1).mean(),
                         auc_n_cuts=roc_auc_score(y, nc), auc_n_cycles=roc_auc_score(y, ncy),
                         auc_max_drop=roc_auc_score(y, md),
                         stable_forest_share=(fs[y == 0] == 1).mean(), stable_plant_share=(fs[y == 1] == 1).mean()))
mc = pd.DataFrame(rows)
mc["youden_cut"] = mc["tpr_cut"] - mc["fpr_cut"]
mc.to_csv(os.path.join(RES, "method_comparison.csv"), index=False)
for ds in ["val", "qc"]:
    log(f"\n[{ds}] sorted by AUC of n_cuts")
    log(mc[mc.dataset == ds].sort_values("auc_n_cuts", ascending=False)
        [["index", "agg", "method", "tpr_cut", "fpr_cut", "tpr_cycle", "fpr_cycle", "auc_n_cuts", "auc_max_drop",
          "stable_forest_share", "stable_plant_share"]].round(3).to_string(index=False))
# paired cluster-bootstrap CIs for the differences behind the recommendations
def _auc_diff(yy, ss):
    return roc_auc_score(yy, ss[:, 0]) - roc_auc_score(yy, ss[:, 1])


def _score(i, a, m, col):
    f = feat_all[(feat_all["index"] == i) & (feat_all["agg"] == a) & (feat_all["method"] == m)].set_index("pid")[col]
    return f.reindex(pts["pid"]).fillna(0).to_numpy()


cmp_rows = []
CMPS = [("NBR vs NDVI (median, rules)", ("nbr", "median", "R"), ("ndvi", "median", "R"), "n_cuts"),
        ("NBR vs NDVI (median, segmentation)", ("nbr", "median", "S"), ("ndvi", "median", "S"), "n_cuts"),
        ("rules vs segmentation (NBR, median)", ("nbr", "median", "R"), ("nbr", "median", "S"), "n_cuts"),
        ("segmentation cap 18 vs 6 (NBR, median)", ("nbr", "median", "S"), ("nbr", "median", "S6"), "n_cuts"),
        ("median vs dry-season median (rules)", ("nbr", "median", "R"), ("nbr", "dry_median", "R"), "n_cuts"),
        ("median vs wet-season median (rules)", ("nbr", "median", "R"), ("nbr", "wet_median", "R"), "n_cuts"),
        ("median vs yearly max (rules)", ("nbr", "median", "R"), ("nbr", "max", "R"), "n_cuts"),
        ("largest 3-year drop vs cut count (NBR, median)", ("nbr", "median", "S"), ("nbr", "median", "S"), ("s_max_drop3", "n_cuts")),
        ("NBR variability vs cut count (NBR, median)", ("nbr", "median", "S"), ("nbr", "median", "S"), ("s_std", "n_cuts"))]
cl_all = np.where(pts["dataset"] == "val", pts["CD_MUN"].astype(str),
                  "g" + (pts["lon"] / 0.1).round().astype(int).astype(str) + "_" + (pts["lat"] / 0.1).round().astype(int).astype(str))
for name, A, B, col in CMPS:
    ca, cb = (col, col) if isinstance(col, str) else col
    sa, sb = _score(*A, ca), _score(*B, cb)
    for ds in ["val", "qc"]:
        msk = (pts["dataset"] == ds).to_numpy()
        yy = pts["label"].to_numpy()[msk]
        ss = np.c_[sa[msk], sb[msk]]
        lo, hi = cluster_boot(yy, ss, cl_all[msk], fn=_auc_diff, n_boot=1000)
        cmp_rows.append(dict(comparison=name, dataset=ds, auc_a=roc_auc_score(yy, ss[:, 0]), auc_b=roc_auc_score(yy, ss[:, 1]),
                             diff=_auc_diff(yy, ss), ci_lo=lo, ci_hi=hi))
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(RES, "method_comparison_paired_ci.csv"), index=False)
log("\nPaired AUC differences (A - B), 95% CI resampling municipalities (VAL) / 0.1-degree cells (QC):\n" +
    cmp_df.round(3).to_string(index=False))
# Hansen reference rows
for ds, gg in df.groupby("dataset"):
    y = gg["label"].to_numpy()
    for col in ["hansen_loss_le2020", "hansen_gain", "hansen_loss3x3_le2020"]:
        v = gg[col].to_numpy()
        log(f"[{ds}] {col}: plantation {v[y == 1].mean():.3f}, forest {v[y == 0].mean():.3f}, AUC {roc_auc_score(y, v):.3f}")

# ============================================================================= 3. sensitivity
log("\n=== 3. Sensitivity of 'at least one cut' to the drop threshold and segmentation penalty (NBR, median)")
gs = grid[(grid["index"] == "nbr") & (grid["agg"] == "median")].merge(pts[["pid", "dataset", "label"]], on="pid")
gs["cut"] = (gs["n_cuts"] >= 1).astype(int)
sens = gs.groupby(["dataset", "method", "max_segments", "pen", "drop"], dropna=False).apply(
    lambda x: pd.Series(dict(tpr=x.loc[x.label == 1, "cut"].mean(), fpr=x.loc[x.label == 0, "cut"].mean(),
                             auc=roc_auc_score(x["label"], x["n_cuts"])))).reset_index()
sens.to_csv(os.path.join(RES, "sensitivity_drop_threshold.csv"), index=False)
log(sens.round(3).to_string(index=False))

# ============================================================================= 4. univariate significance
log("\n=== 4. Univariate significance, primary configuration (NBR, annual median, segmentation)")
BIN = ["has_cut", "has_cycle", "two_plus_cycles", "fast_regrowth", "abrupt_start", "forest_since_1985",
       "hansen_loss_le2020", "hansen_gain"]
CONT = ["n_cuts", "n_cycles", "years_since_last_cut", "s_max_drop3", "s_min", "s_frac_below_forest", "s_std"]
uni = []
for name, d in [("val", val), ("qc", qc)]:
    y = d["label"].to_numpy()
    cl = d["cluster"].to_numpy()
    for f in BIN:
        v = d[f].fillna(0).astype(int).to_numpy()
        a, b = int(((v == 1) & (y == 1)).sum()), int(((v == 0) & (y == 1)).sum())
        cc, dd = int(((v == 1) & (y == 0)).sum()), int(((v == 0) & (y == 0)).sum())
        orr, lo, hi = odds_ratio(a, b, cc, dd)
        p_f = stats.fisher_exact([[a, b], [cc, dd]])[1]
        # an empty cell = perfect separation: the clustered logit SE is meaningless there, keep only Fisher
        tab = None if min(a, b, cc, dd) == 0 else clustered_logit(v[:, None], y, cl, [f])[0]
        auc = roc_auc_score(y, v)
        alo, ahi = cluster_boot(y, v.astype(float), cl, n_boot=1000)
        uni.append(dict(dataset=name, feature=f, kind="binary", plant_rate=a / (a + b), forest_rate=cc / (cc + dd),
                        plant_n=a, forest_n=cc, OR=orr, OR_lo=lo, OR_hi=hi, p_fisher=p_f,
                        p_cluster=np.nan if tab is None else tab.loc[f, "p_cluster"], auc=auc, auc_lo=alo, auc_hi=ahi))
    for f in CONT:
        v = d[f].astype(float).to_numpy()
        ok = np.isfinite(v)
        p_mw = stats.mannwhitneyu(v[ok & (y == 1)], v[ok & (y == 0)]).pvalue
        auc = roc_auc_score(y[ok], v[ok])
        flip = auc < 0.5
        alo, ahi = cluster_boot(y[ok], v[ok], cl[ok], n_boot=1000)
        uni.append(dict(dataset=name, feature=f, kind="continuous", plant_median=np.median(v[ok & (y == 1)]),
                        forest_median=np.median(v[ok & (y == 0)]), p_mannwhitney=p_mw, auc=auc, auc_lo=alo, auc_hi=ahi,
                        direction="lower in plantation" if flip else "higher in plantation"))
    # regrowth time, among points with a complete cycle only
    sub = d[d["n_cycles"] >= 1]
    v, yy = sub["regrow_years"].to_numpy(float), sub["label"].to_numpy()
    ok = np.isfinite(v)
    if ok.sum() > 10 and len(set(yy[ok])) == 2:
        uni.append(dict(dataset=name, feature="regrow_years (points with a cycle)", kind="continuous",
                        plant_median=np.median(v[ok & (yy == 1)]), forest_median=np.median(v[ok & (yy == 0)]),
                        plant_n=int((ok & (yy == 1)).sum()), forest_n=int((ok & (yy == 0)).sum()),
                        p_mannwhitney=stats.mannwhitneyu(v[ok & (yy == 1)], v[ok & (yy == 0)]).pvalue,
                        auc=roc_auc_score(yy[ok], -v[ok]), direction="faster in plantation if auc>0.5"))
uni = pd.DataFrame(uni)
uni.to_csv(os.path.join(RES, "univariate_significance.csv"), index=False)
# abrupt start depends on where 'open land' is drawn: sensitivity over the open level
c0 = cal[(cal["index"] == "nbr") & (cal["agg"] == "median")].iloc[0]
ab_rows = []
for lvl in [0.30, 0.35, float(c0.open_level), 0.40, 0.45]:
    p_l = hl.HarvestParams(forest_level=float(c0.closed_p5), open_level=lvl)
    ab = {pid: hl.establishment(*hl.valid_tv(series["median"].loc[pid]), p_l)[0] for pid in pts["pid"]}
    for name, d in [("val", val), ("qc", qc)]:
        v = d["pid"].map(ab).to_numpy()
        yy = d["label"].to_numpy()
        a_, b_ = int(((v == 1) & (yy == 1)).sum()), int(((v == 0) & (yy == 1)).sum())
        c_, d_ = int(((v == 1) & (yy == 0)).sum()), int(((v == 0) & (yy == 0)).sum())
        ab_rows.append(dict(open_level=round(lvl, 3), calibrated=abs(lvl - float(c0.open_level)) < 1e-9, dataset=name,
                            plant_rate=a_ / (a_ + b_), forest_rate=c_ / (c_ + d_),
                            p_fisher=stats.fisher_exact([[a_, b_], [c_, d_]])[1]))
abs_df = pd.DataFrame(ab_rows)
abs_df.to_csv(os.path.join(RES, "abrupt_start_open_level_sensitivity.csv"), index=False)
log("Abrupt start vs the open-land level (NBR, yearly median):\n" + abs_df.round(4).to_string(index=False))
log(uni.round(4).to_string(index=False))

# ============================================================================= 5. groups vs MapBiomas 10 m
log("\n=== 5. VAL groups (relative to MapBiomas 10 m collection 4)")
G = ["real forest", "forest called plantation", "caught plantation", "missed plantation", "other"]
grp_rows = []
for g in G:
    d = val[val.group == g]
    r = dict(group=g, n=len(d), plantation_share=d.label.mean())
    for f in ["has_cut", "has_cycle", "fast_regrowth", "abrupt_start", "forest_since_1985", "hansen_loss_le2020", "hansen_gain"]:
        k = int(d[f].fillna(0).sum())
        lo, hi = wilson(k, len(d))
        r.update({f: k / len(d), f + "_lo": lo, f + "_hi": hi})
    grp_rows.append(r)
grp = pd.DataFrame(grp_rows)
grp.to_csv(os.path.join(RES, "rates_by_mapbiomas_group.csv"), index=False)
log(grp[["group", "n", "has_cut", "has_cycle", "fast_regrowth", "abrupt_start", "forest_since_1985", "hansen_loss_le2020",
         "hansen_gain"]].round(3).to_string(index=False))
for g in ["missed plantation", "forest called plantation"]:
    for f in ["has_cut", "has_cycle", "forest_since_1985", "hansen_loss_le2020"]:
        a = val[val.group == g][f].fillna(0)
        b = val[val.group == "real forest"][f].fillna(0)
        p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        log(f"  {g} vs real forest, {f}: {a.mean():.3f} vs {b.mean():.3f}, Fisher p={p:.2g}")

# ============================================================================= 6. added value in models (VAL, spatial CV)
log("\n=== 6. Added value on top of the team's features (VAL, spatial CV by municipality)")
BASE_TEAM = ["height_stdDev", "ndvi_aug_oct_mean", "sag_oct_nov_mean", "sag_swing_mean", "ever_farm"]
BASE_NOMB = [f for f in BASE_TEAM if f != "ever_farm"]   # ever_farm = MapBiomas history: excluded when testing on MapBiomas labels
H4 = ["n_cycles", "regrow_speed", "forest_since_1985", "abrupt_start"]
HALL = H4 + ["n_cuts", "years_since_last_cut", "max_drop", "s_frac_below_forest", "s_min", "s_max_drop3"]
HANS = ["hansen_loss_le2020", "hansen_gain", "tc2000"]
HALL6 = [f + "_s6" for f in H4 + ["n_cuts", "years_since_last_cut", "max_drop"]] + ["s_frac_below_forest", "s_min", "s_max_drop3"]
fullmat = hl.full_series_matrix(series["median"])
FULL = list(fullmat.columns)
V = val.dropna(subset=BASE_TEAM).merge(fullmat, left_on="pid", right_index=True, how="left").reset_index(drop=True)
for f in HALL + HANS + HALL6 + ["s_std"]:
    V[f] = V[f].astype(float).fillna(V[f].median())
n_nan_full = int(V[FULL].isna().any(axis=1).sum())
V[FULL] = V[FULL].fillna(V[FULL].median())
log(f"rows with a missing yearly value after interpolation (filled with the column median): {n_nan_full}")
log(f"model rows: {len(V)} (plantation {V.label.sum()}, missed plantations {(V.group == 'missed plantation').sum()})")


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))


def make_rf(seed=SEED):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1)


SETS = {
    "team (with ever_farm)": BASE_TEAM,
    "team + H4": BASE_TEAM + H4,
    "team + H-all": BASE_TEAM + HALL,
    "base (no MapBiomas history)": BASE_NOMB,
    "base + H4": BASE_NOMB + H4,
    "base + H-all": BASE_NOMB + HALL,
    "base + Hansen": BASE_NOMB + HANS,
    "base + H-all + Hansen": BASE_NOMB + HALL + HANS,
    "base + full series": BASE_NOMB + FULL,
    "H4 only": H4,
    "H-all only": HALL,
    "Hansen only": HANS,
    "H-all + Hansen only": HALL + HANS,
    "full series only": FULL,
    "base + H-all [segment cap 6, as first specified]": BASE_NOMB + HALL6,
    # post hoc (added after seeing that NBR variability is the strongest single summary): labelled as such
    "base + H-all + variability (post hoc)": BASE_NOMB + HALL + ["s_std"],
    "H-all + variability only (post hoc)": HALL + ["s_std"],
}
PAIRS = {"team + H4": "team (with ever_farm)", "team + H-all": "team (with ever_farm)",
         "base + H4": "base (no MapBiomas history)", "base + H-all": "base (no MapBiomas history)",
         "base + Hansen": "base (no MapBiomas history)", "base + H-all + Hansen": "base (no MapBiomas history)",
         "base + full series": "base (no MapBiomas history)", "H-all + Hansen only": "Hansen only",
         "base + H-all + variability (post hoc)": "base + H-all", "H-all + variability only (post hoc)": "H-all only",
         "base + H-all [segment cap 6, as first specified]": "base (no MapBiomas history)"}
REPEATS = [42, 43, 44, 45, 46]
y = V["label"].to_numpy()
groups = V["CD_MUN"].to_numpy()
cv_rows, oof = [], {}
for rep in REPEATS:
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
    folds = list(gkf.split(V, y, groups))
    for sname, cols in SETS.items():
        X = V[cols].to_numpy(float)
        for mname in ["LR", "RF"]:
            if sname.startswith("full") and mname == "LR":
                continue
            o = np.full(len(y), np.nan)
            for k, (tr, te) in enumerate(folds):
                mdl = make_lr() if mname == "LR" else make_rf(rep)
                mdl.fit(X[tr], y[tr])
                o[te] = mdl.predict_proba(X[te])[:, 1]
                cv_rows.append(dict(repeat=rep, fold=k, set=sname, model=mname, auc=roc_auc_score(y[te], o[te])))
            if rep == SEED:
                oof[(sname, mname)] = o
cvr = pd.DataFrame(cv_rows)
cvr.to_csv(os.path.join(RES, "cv_fold_auc.csv"), index=False)
summ = cvr.groupby(["set", "model"]).auc.agg(["mean", "std"]).reset_index()
delta_rows = []
for new, base in PAIRS.items():
    for mname in ["LR", "RF"]:
        a = cvr[(cvr.set == new) & (cvr.model == mname)].set_index(["repeat", "fold"]).auc
        b = cvr[(cvr.set == base) & (cvr.model == mname)].set_index(["repeat", "fold"]).auc
        if a.empty or b.empty:
            continue
        dlt = (a - b).dropna()
        o_new, o_base = oof[(new, mname)], oof[(base, mname)]
        lo, hi = cluster_boot(y, np.c_[o_new, o_base], groups,
                              fn=lambda yy, ss: roc_auc_score(yy, ss[:, 0]) - roc_auc_score(yy, ss[:, 1]), n_boot=1000)
        delta_rows.append(dict(model=mname, new=new, base=base, auc_new=a.mean(), auc_base=b.mean(),
                               delta_mean=dlt.mean(), delta_sd=dlt.std(ddof=1),
                               folds_improved=f"{int((dlt > 0).sum())}/{len(dlt)}",
                               delta_oof_seed42=roc_auc_score(y, o_new) - roc_auc_score(y, o_base),
                               delta_ci_lo_cluster_boot=lo, delta_ci_hi_cluster_boot=hi))
dl = pd.DataFrame(delta_rows)
dl.to_csv(os.path.join(RES, "added_value_delta_auc.csv"), index=False)
log("AUC by feature set (mean, sd over 5 repeats x 5 spatial folds):\n" + summ.round(3).to_string(index=False))
log("Paired gains:\n" + dl.round(3).to_string(index=False))

# operating points (seed 42 OOF; thresholds chosen on the same OOF -> optimistic, as in the team's run)
op_rows = []
gv = V["group"].to_numpy()
for (sname, mname), o in oof.items():
    for max_fpr in [0.05, 0.01]:
        tpr, fpr, thr = recall_at_fpr(y, o, max_fpr)
        flag = o >= thr
        op_rows.append(dict(set=sname, model=mname, max_fpr=max_fpr, threshold=thr, recall=tpr, fpr=fpr,
                            auc_oof=roc_auc_score(y, o), ap=average_precision_score(y, o),
                            missed_caught=int((flag & (gv == "missed plantation")).sum()),
                            missed_n=int((gv == "missed plantation").sum()),
                            realforest_flagged=int((flag & (gv == "real forest")).sum()),
                            realforest_n=int((gv == "real forest").sum()),
                            fcp_flagged=int((flag & (gv == "forest called plantation")).sum()),
                            fcp_n=int((gv == "forest called plantation").sum())))
ops = pd.DataFrame(op_rows)
ops.to_csv(os.path.join(RES, "operating_points.csv"), index=False)
log("Operating points (OOF, seed 42):\n" + ops.round(3).to_string(index=False))
mbp = (V["mb_pred"] == 2).astype(int).to_numpy()
log(f"MapBiomas 10 m on these rows: recall {mbp[y == 1].mean():.3f}, FPR {mbp[y == 0].mean():.3f}")

# clustered logit, base (no MapBiomas history) + H4, standardised continuous features
Xs = V[BASE_NOMB + H4].astype(float).copy()
for col in BASE_NOMB + ["n_cycles", "regrow_speed"]:
    Xs[col] = (Xs[col] - Xs[col].mean()) / Xs[col].std()
tab, res = clustered_logit(Xs.to_numpy(), y, groups, BASE_NOMB + H4)
if tab is not None:
    tab.to_csv(os.path.join(RES, "logit_base_plus_H4_clustered.csv"))
    log("Logit, municipality-clustered SEs (continuous features per SD; binary per unit):\n" + tab.round(4).to_string())
    import statsmodels.api as sm
    k = len(BASE_NOMB) + 1
    R = np.zeros((len(H4), len(tab)))
    for j in range(len(H4)):
        R[j, k + j] = 1
    w = res.wald_test(R, scalar=True)
    log(f"Joint Wald test that all 4 harvest features add nothing (clustered): chi2={float(w.statistic):.1f}, "
        f"df={len(H4)}, p={float(w.pvalue):.2g}")

# permutation importance in the RF base + H-all (seed 42 folds)
from sklearn.inspection import permutation_importance  # noqa: E402
gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
cols = BASE_NOMB + HALL
X = V[cols].to_numpy(float)
imp = []
for tr, te in gkf.split(V, y, groups):
    m = make_rf().fit(X[tr], y[tr])
    imp.append(permutation_importance(m, X[te], y[te], scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=-1).importances_mean)
imp = pd.DataFrame({"feature": cols, "perm_mean": np.mean(imp, 0), "perm_sd": np.std(imp, 0, ddof=1)}).sort_values("perm_mean", ascending=False)
imp.to_csv(os.path.join(RES, "permutation_importance_base_plus_Hall.csv"), index=False)
log("Permutation importance (drop in held-out AUC), RF base + H-all:\n" + imp.round(4).to_string(index=False))

# ============================================================================= 7. the safe veto
log("\n=== 7. 'Forest every year since 1985' as a veto on plantation calls")
veto_rows = []
for tol in [0, 1, 2]:
    fsv = ((val["n_cuts"].fillna(0) == 0) & (val["years_below_forest"].fillna(99) <= tol)).astype(int)
    fsq = ((qc["n_cuts"].fillna(0) == 0) & (qc["years_below_forest"].fillna(99) <= tol)).astype(int)
    mb = (val["mb_pred"] == 2)
    veto_rows.append(dict(tolerance_years=tol,
                          val_forest_protected=fsv[val.label == 0].mean(), val_plant_vetoed=fsv[val.label == 1].mean(),
                          mb_forest_calls_cancelled=int((fsv[mb & (val.label == 0)]).sum()), mb_forest_calls=int((mb & (val.label == 0)).sum()),
                          mb_correct_calls_lost=int((fsv[mb & (val.label == 1)]).sum()), mb_correct_calls=int((mb & (val.label == 1)).sum()),
                          qc_forest_protected=fsq[qc.label == 0].mean(), qc_plant_vetoed=fsq[qc.label == 1].mean()))
    # on top of the RF model flags at FPR<=5% (base without MapBiomas history)
    for sname in ["base (no MapBiomas history)", "base + H-all"]:
        o = oof[(sname, "RF")]
        _, _, thr = recall_at_fpr(y, o, 0.05)
        flag = o >= thr
        fv = ((V["n_cuts"].fillna(0) == 0) & (V["years_below_forest"].fillna(99) <= tol)).to_numpy()
        veto_rows[-1].update({f"{sname}: forest flags": int((flag & (y == 0)).sum()),
                              f"{sname}: forest flags vetoed": int((flag & (y == 0) & fv).sum()),
                              f"{sname}: plantation flags": int((flag & (y == 1)).sum()),
                              f"{sname}: plantation flags vetoed": int((flag & (y == 1) & fv).sum())})
veto = pd.DataFrame(veto_rows)
veto.to_csv(os.path.join(RES, "safe_veto.csv"), index=False)
log(veto.round(3).T.to_string())

# ============================================================================= 8. QC transfer
log("\n=== 8. Cropin QC polygons: train on VAL, test on QC (independent)")
Q = qc.merge(fullmat, left_on="pid", right_index=True, how="left").reset_index(drop=True)
for f in HALL + HANS + ["s_std"]:
    Q[f] = Q[f].astype(float).fillna(V[f].median())
Q[FULL] = Q[FULL].fillna(V[FULL].median())
yq = Q["label"].to_numpy()
qrows = []
for sname in ["H4 only", "H-all only", "H-all + variability only (post hoc)", "Hansen only", "H-all + Hansen only",
              "full series only"]:
    cols = SETS[sname]
    for mname in ["LR", "RF"]:
        if sname.startswith("full") and mname == "LR":
            continue
        mdl = (make_lr() if mname == "LR" else make_rf()).fit(V[cols].to_numpy(float), y)
        pq = mdl.predict_proba(Q[cols].to_numpy(float))[:, 1]
        thr = recall_at_fpr(y, oof[(sname, mname)], 0.05)[2]
        lo, hi = cluster_boot(yq, pq, Q["cluster"].to_numpy(), n_boot=1000)
        qrows.append(dict(set=sname, model=mname, qc_auc=roc_auc_score(yq, pq), qc_auc_lo=lo, qc_auc_hi=hi,
                          recall_at_val_thr=(pq[yq == 1] >= thr).mean(), forest_flagged_at_val_thr=(pq[yq == 0] >= thr).mean(),
                          recall_at_qc_fpr5=recall_at_fpr(yq, pq, 0.05)[0]))
qt = pd.DataFrame(qrows)
qt.to_csv(os.path.join(RES, "qc_transfer.csv"), index=False)
log(qt.round(3).to_string(index=False))
lc = qc[qc.n_cuts >= 1]
log("QC: last cut year (points with a cut) plantation:", lc[lc.label == 1]["last_cut_year"].value_counts().sort_index().to_dict())
log("QC: last cut year forest:", lc[lc.label == 0]["last_cut_year"].value_counts().sort_index().to_dict())
log(f"QC plantations with a cut 2010-2020: {(qc[qc.label == 1]['last_cut_year'].between(2010, 2020)).mean():.3f}; "
    f"forest: {(qc[qc.label == 0]['last_cut_year'].between(2010, 2020)).mean():.3f}")
by_size = qc.assign(size=pd.cut(qc["qc_area_m2"], [0, 300, 900, 2700, 1e9], labels=["<300 m2", "300-900", "900-2700", ">2700"]))
log("QC rates by polygon size:\n" + by_size.groupby(["size", "label"])[["has_cut", "has_cycle", "forest_since_1985", "hansen_loss_le2020"]]
    .mean().round(3).to_string())
qc[["pid", "label", "qc_area_m2", "n_cuts", "n_cycles", "last_cut_year", "regrow_years", "forest_since_1985", "abrupt_start",
    "hansen_loss_le2020", "hansen_gain"]].to_csv(os.path.join(PRIV, "qc_features_primary.csv"), index=False)

# ============================================================================= 9. stratified diagnostics (VAL)
log("\n=== 9. Where does it work and fail? (VAL; the reference label history is used only to stratify)")
yrs = range(1985, 2023)
H = val[[f"CLASS_{y}" for y in yrs]].to_numpy()
first9 = np.array([1985 + int(np.argmax(r == 9)) if (r == 9).any() else np.nan for r in H])
val["first_plantation_year"] = first9
val["est_period"] = pd.cut(val["first_plantation_year"].where(val["label"] == 1), [1984, 1985, 2000, 2013, 2022],
                           labels=["plantation in 1985", "1986-2000", "2001-2013", "2014-2020 (1st rotation)"]).astype(str).replace("nan", "")
always_forest = (H[:, :36] == 3).all(axis=1)
val["forest_history"] = np.where(val.label == 0, np.where(always_forest, "forest every year 1985-2020", "non-forest in some year"), "")
strat = []
for col in ["state", "est_period", "forest_history"]:
    for k, d in val[(val[col] != "") & val[col].notna()].groupby(col, observed=True):
        strat.append(dict(stratum=col, value=k, n=len(d), plantation=int(d.label.sum()),
                          has_cut=d.has_cut.mean(), has_cycle=d.has_cycle.mean(), forest_since_1985=d.forest_since_1985.mean(),
                          abrupt_start=d.abrupt_start.mean(), hansen_loss_le2020=d.hansen_loss_le2020.mean()))
# bracatinga zone: forest points within 80 km of Curitiba
dist_cwb = np.hypot((val.lon + 49.27) * 111 * np.cos(np.deg2rad(25.43)), (val.lat + 25.43) * 111)
for zone, d in [("forest within 80 km of Curitiba (bracatinga zone)", val[(dist_cwb <= 80) & (val.label == 0)]),
                ("forest elsewhere", val[(dist_cwb > 80) & (val.label == 0)])]:
    strat.append(dict(stratum="bracatinga", value=zone, n=len(d), plantation=0, has_cut=d.has_cut.mean(),
                      has_cycle=d.has_cycle.mean(), forest_since_1985=d.forest_since_1985.mean(),
                      abrupt_start=d.abrupt_start.mean(), hansen_loss_le2020=d.hansen_loss_le2020.mean()))
strat = pd.DataFrame(strat)
strat.to_csv(os.path.join(RES, "stratified_rates.csv"), index=False)
log(strat.round(3).to_string(index=False))
by_state = []
for st, d in val.groupby("state"):
    if d.label.nunique() == 2:
        by_state.append(dict(state=st, n=len(d), auc_n_cuts=roc_auc_score(d.label, d.n_cuts.fillna(0)),
                             tpr_cut=d[d.label == 1].has_cut.mean(), fpr_cut=d[d.label == 0].has_cut.mean()))
log(pd.DataFrame(by_state).round(3).to_string(index=False))
val[["pid", "pt_id", "label", "group", "state", "n_cuts", "n_cycles", "last_cut_year", "years_since_last_cut", "max_drop",
     "regrow_years", "regrow_speed", "forest_since_1985", "abrupt_start", "establish_rise_years", "years_below_forest",
     "s_min", "s_max_drop3", "s_frac_below_forest", "s_std", "hansen_loss_le2020", "hansen_loss3x3_le2020", "hansen_gain",
     "tc2000", "first_plantation_year"]].to_csv(os.path.join(RES, "val_harvest_features_primary.csv"), index=False)


# ============================================================================= 11. extra diagnostics
log("\n=== 11. Extra diagnostics")
# (a) does NBR variability still separate when no cut was detected?
sub_rows = []
for name, mask in [("all points", np.ones(len(val), bool)),
                   ("plantations with no detected cut vs all forest", ~((val.label == 1) & (val.n_cuts > 0)).to_numpy()),
                   ("plantations vs forest with a non-forest year", ((val.label == 1) | (val.forest_history == "non-forest in some year")).to_numpy()),
                   ("plantations vs forest every year 1985-2020", ((val.label == 1) | (val.forest_history == "forest every year 1985-2020")).to_numpy())]:
    d = val[mask]
    for f in ["s_std", "s_max_drop3", "s_frac_below_forest", "n_cuts"]:
        sub_rows.append(dict(subset=name, feature=f, n_plant=int(d.label.sum()), n_forest=int((d.label == 0).sum()),
                             auc=roc_auc_score(d.label, d[f].fillna(0))))
sub = pd.DataFrame(sub_rows)
sub.to_csv(os.path.join(RES, "subset_auc.csv"), index=False)
log(sub.round(3).to_string(index=False))

# (b) within-QC spatial CV (grid cells of 0.1 degree), harvest-only features
wq = []
yq2 = Q["label"].to_numpy()
gq = Q["cluster"].to_numpy()
for sname in ["H4 only", "H-all only", "Hansen only", "H-all + Hansen only", "full series only"]:
    cols = SETS[sname]
    X = Q[cols].to_numpy(float)
    for mname in ["LR", "RF"]:
        if sname.startswith("full") and mname == "LR":
            continue
        aucs = []
        for rep in REPEATS:
            o = np.full(len(yq2), np.nan)
            for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, yq2, gq):
                mdl = (make_lr() if mname == "LR" else make_rf(rep)).fit(X[tr], yq2[tr])
                o[te] = mdl.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(yq2, o))
        wq.append(dict(set=sname, model=mname, qc_cv_auc_mean=np.mean(aucs), qc_cv_auc_sd=np.std(aucs, ddof=1),
                       n_clusters=len(set(gq))))
wq = pd.DataFrame(wq)
wq.to_csv(os.path.join(RES, "qc_within_cv.csv"), index=False)
log("Within-QC spatial CV (5 repeats of 5-fold GroupKFold over 0.1-degree cells):\n" + wq.round(3).to_string(index=False))

# (c) simple rules Cropin could apply without a model
rules = {
    "at least one cut": lambda d: d.has_cut == 1,
    "cut or abrupt start": lambda d: (d.has_cut == 1) | (d.abrupt_start == 1),
    "Hansen loss<=2020 or gain": lambda d: (d.hansen_loss_le2020 == 1) | (d.hansen_gain == 1),
    "cut, abrupt start or Hansen": lambda d: (d.has_cut == 1) | (d.abrupt_start == 1) | (d.hansen_loss_le2020 == 1) | (d.hansen_gain == 1),
    "two or more cuts": lambda d: d.n_cuts >= 2,
    "NOT forest every year since 1985": lambda d: d.forest_since_1985 == 0,
}
rr = []
for rname, fn in rules.items():
    for name, d in [("val", val), ("qc", qc)]:
        f = fn(d).to_numpy()
        yy = d.label.to_numpy()
        rr.append(dict(rule=rname, dataset=name, plantation_flagged=f[yy == 1].mean(), forest_flagged=f[yy == 0].mean(),
                       n_plant=int((yy == 1).sum()), n_forest=int((yy == 0).sum())))
rr = pd.DataFrame(rr)
rr.to_csv(os.path.join(RES, "simple_rules.csv"), index=False)
log(rr.round(3).to_string(index=False))


# ============================================================================= 12. 3 x 3 neighbourhood vs centre pixel
WIN = f"{D}/features_win3.csv.gz"
if os.path.exists(WIN):
    log("\n=== 12. 3 x 3 neighbourhood (90 m) vs the centre pixel")
    W = pd.read_csv(WIN).set_index("pid")
    def derive(prefix):
        """H-all definitions applied to the S/R features with a given prefix (c_ = centre, mean_ = 90 m mean series)."""
        k = "S_" if prefix in ("c_", "mean_") else ""
        sp = "" if prefix == "c_" else prefix
        d = pd.DataFrame(index=W.index)
        d["n_cycles"] = W[f"{prefix}{k}n_cycles"]
        ry = W[f"{prefix}{k}regrow_years"]
        d["regrow_speed"] = np.where((d["n_cycles"] >= 1) & (ry > 0), 1 / ry.clip(lower=1), 0.0)
        d["forest_since_1985"] = W[f"{prefix}{k}forest_since_1985"]
        d["abrupt_start"] = W[f"{prefix}{k}abrupt_start"]
        d["n_cuts"] = W[f"{prefix}{k}n_cuts"]
        d["years_since_last_cut"] = (hl.CUTOFF - W[f"{prefix}{k}last_cut_year"]).fillna(40)
        d["max_drop"] = W[f"{prefix}{k}max_drop"]
        for f in ["s_frac_below_forest", "s_min", "s_max_drop3", "s_std"]:
            d[f] = W[f"{sp}{f}"] if prefix != "c_" else W[f"c_{f}"]
        return d.astype(float)
    CEN, MEA = derive("c_"), derive("mean_")
    NB = W[["nb_share_cut_S", "nb_max_n_cuts_S", "nb_share_forest_since_1985_S", "nb_any_abrupt_start_S",
            "nb_max_s_max_drop3", "nb_mean_s_std", "nb_mean_s_frac_below_forest", "nb_min_s_min"]].astype(float)
    # sanity: centre pixel recomputed from the window equals the single-pixel run
    chk = df.set_index("pid")[["n_cuts", "s_std"]].join(CEN[["n_cuts", "s_std"]], rsuffix="_w").dropna()
    log(f"centre-pixel check vs single-pixel run: n_cuts equal {float((chk.n_cuts == chk.n_cuts_w).mean()):.3f}, "
        f"s_std corr {chk.s_std.corr(chk.s_std_w):.4f}")
    rows12 = []
    for name, d in [("val", val), ("qc", qc)]:
        ids = d["pid"].to_numpy()
        yy = d["label"].to_numpy()
        for fam, X in [("centre pixel", CEN), ("90 m mean series", MEA)]:
            for f in ["n_cuts", "s_std", "s_max_drop3", "s_frac_below_forest"]:
                rows12.append(dict(dataset=name, version=fam, feature=f, auc=roc_auc_score(yy, X.loc[ids, f].fillna(0))))
            fs = X.loc[ids, "forest_since_1985"].fillna(0).to_numpy()
            rows12.append(dict(dataset=name, version=fam, feature="forest_since_1985: forest kept / plantation lost",
                               auc=np.nan, forest_kept=fs[yy == 0].mean(), plant_lost=fs[yy == 1].mean()))
        for f in ["nb_share_cut_S", "nb_mean_s_std", "nb_max_s_max_drop3", "nb_mean_s_frac_below_forest"]:
            rows12.append(dict(dataset=name, version="any of 9 pixels / mean of 9", feature=f,
                               auc=roc_auc_score(yy, NB.loc[ids, f].fillna(0))))
        fs = NB.loc[ids, "nb_share_forest_since_1985_S"].fillna(0).to_numpy() == 1
        rows12.append(dict(dataset=name, version="all 9 pixels", feature="forest_since_1985: forest kept / plantation lost",
                           auc=np.nan, forest_kept=fs[yy == 0].mean(), plant_lost=fs[yy == 1].mean()))
    r12 = pd.DataFrame(rows12)
    r12.to_csv(os.path.join(RES, "neighbourhood_univariate.csv"), index=False)
    log(r12.round(3).to_string(index=False))
    # transfer VAL -> QC and VAL spatial CV with centre vs neighbourhood versions of H-all (+ variability)
    t12 = []
    vids, qids = V["pid"].to_numpy(), Q["pid"].to_numpy()
    for fam, X in [("centre pixel", CEN), ("90 m mean series", MEA), ("centre + 9-pixel aggregates", CEN.join(NB))]:
        Xv = X.loc[vids].fillna(X.loc[vids].median()).to_numpy(float)
        Xq = X.loc[qids].fillna(X.loc[vids].median()).to_numpy(float)
        for mname in ["LR", "RF"]:
            o = np.full(len(y), np.nan)
            for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xv, y, groups):
                o[te] = (make_lr() if mname == "LR" else make_rf()).fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
            mdl = (make_lr() if mname == "LR" else make_rf()).fit(Xv, y)
            pq = mdl.predict_proba(Xq)[:, 1]
            thr = recall_at_fpr(y, o, 0.05)[2]
            lo, hi = cluster_boot(yq, pq, Q["cluster"].to_numpy(), n_boot=1000)
            t12.append(dict(features=f"H-all + variability, {fam}", model=mname, val_cv_auc=roc_auc_score(y, o),
                            qc_auc=roc_auc_score(yq, pq), qc_auc_lo=lo, qc_auc_hi=hi,
                            qc_recall_at_val_thr=(pq[yq == 1] >= thr).mean(), qc_forest_flagged_at_val_thr=(pq[yq == 0] >= thr).mean()))
    t12 = pd.DataFrame(t12)
    t12.to_csv(os.path.join(RES, "neighbourhood_models.csv"), index=False)
    log(t12.round(3).to_string(index=False))

# ============================================================================= 10. figures
log("\n=== 10. Figures")
S = series["median"]


def band(ax, ids, color, label, n_label=True):
    m = S.reindex(ids)
    med, q1, q3 = m.median(), m.quantile(0.25), m.quantile(0.75)
    ax.fill_between(hl.YEARS, q1, q3, color=color, alpha=0.10, lw=0)
    ax.plot(hl.YEARS, med, color=color, lw=2, label=f"{label} (n={len(ids)})" if n_label else label)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
ax = axes[0]
band(ax, val[val.group == "real forest"].pid, C_FOREST, "Natural forest, MapBiomas right")
band(ax, val[val.group == "caught plantation"].pid, C_PLANT, "Plantation, MapBiomas right")
band(ax, val[val.group == "missed plantation"].pid, C_MISSED, "Plantation MapBiomas calls forest")
ax.set_title("MapBiomas reference points", loc="left")
ax.set_ylabel("NBR (annual median, TOA)")
ax.legend(loc="lower left", fontsize=8)
ax = axes[1]
band(ax, qc[qc.label == 0].pid, C_FOREST, "Forest (real deforestation)")
band(ax, qc[qc.label == 1].pid, C_PLANT, "Plantation (false alarm)")
ax.set_title("Cropin QC clearings (cut in 2026)", loc="left")
ax.legend(loc="lower left", fontsize=8)
for ax in axes:
    ax.set_xlim(1985, 2020)
    ax.set_ylim(0.2, 0.85)
fig.suptitle("Yearly NBR, 1985-2020 (line = median, band = interquartile range)", fontsize=9, color=INK2, x=0.01, ha="left")
fig.tight_layout()
save(fig, "nbr_trajectories_by_group.png")

# examples
def example(ax, pid, title, color):
    row = S.loc[pid]
    t, yv = hl.valid_tv(row)
    tv, vv, fit = hl.segment(t, yv)
    p = hl.HarvestParams(forest_level=float(cal[(cal["index"] == "nbr") & (cal["agg"] == "median")].closed_p5.iloc[0]))
    ev = hl.cycles_from_segments(tv, vv, t, fit, p)
    for e in ev:
        ax.axvspan(e["cut_year"] - 0.5, e["low_year"] + 0.5, color=GRID, alpha=0.9, lw=0)
    ax.plot(t, yv, "o", ms=3.5, color=MUTED, mec=SURF, mew=0.8)
    ax.plot(tv, vv, "-", color=color, lw=2)
    ax.plot(tv, vv, "o", ms=5, color=color, mec=SURF, mew=1.5)
    ax.axhline(p.forest_level, color=AXIS, lw=0.8)
    nc = sum(e["cycle"] for e in ev)
    ax.set_title(f"{title}\n{len(ev)} cut(s), {nc} cycle(s)", loc="left", fontsize=9)
    ax.set_xlim(1984, 2021)
    ax.set_ylim(-0.2, 0.95)


rng = np.random.default_rng(3)
picks = []
for g, cond, title, col in [
        ("caught plantation", val.n_cycles >= 2, "Plantation, several cycles", C_PLANT),
        ("caught plantation", val.n_cycles == 1, "Plantation, one cycle", C_PLANT),
        ("caught plantation", val.has_cut == 0, "Plantation, no cut seen", C_PLANT),
        ("missed plantation", val.has_cut == 1, "Missed plantation, cut seen", C_MISSED),
        ("real forest", val.forest_since_1985 == 1, "Natural forest, stable", C_FOREST),
        ("real forest", val.has_cut == 1, "Natural forest with a 'cut' (false)", C_FOREST)]:
    cand = val[(val.group == g) & cond & (val.n_years >= 25)].pid.to_numpy()
    if len(cand):
        picks.append((rng.choice(cand), title, col))
fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
for ax, (pid, title, col) in zip(axes.ravel(), picks):
    example(ax, pid, title, col)
for ax in axes[:, 0]:
    ax.set_ylabel("NBR")
fig.suptitle("Examples (MapBiomas reference points): grey dots = yearly NBR, line = LandTrendr-style fit, "
             "shaded = detected cut, rule = closed-canopy level", fontsize=8.5, color=INK2, x=0.01, ha="left")
fig.tight_layout()
save(fig, "examples_segmentation.png")

# feature rates, VAL and QC
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
names = {"has_cut": "At least one cut", "has_cycle": "At least one cut-and-regrow cycle", "two_plus_cycles": "Two or more cycles",
         "fast_regrowth": "Cycle with regrowth <= 4 years", "abrupt_start": "Abrupt start (open land, fast greening)",
         "forest_since_1985": "Forest every year since 1985", "hansen_loss_le2020": "Hansen loss 2001-2020",
         "hansen_gain": "Hansen gain 2000-2012"}
order = list(names)
for ax, (nm, d) in zip(axes, [("MapBiomas reference points", val), ("Cropin QC clearings", qc)]):
    for j, f in enumerate(order):
        for k, (lab, col, off) in enumerate([(1, C_PLANT, -0.18), (0, C_FOREST, 0.18)]):
            dd = d[d.label == lab][f].fillna(0)
            r = dd.mean()
            lo, hi = wilson(dd.sum(), len(dd))
            yy = j + off
            ax.barh(yy, r, height=0.32, color=col, lw=0)
            ax.plot([lo, hi], [yy, yy], color=INK2, lw=1)
            ax.text(min(hi + 0.02, 0.98), yy, f"{100 * r:.0f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[f] for f in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of points (95% CI)")
    ax.set_title(nm, loc="left")
    ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch  # noqa: E402
axes[1].legend(handles=[Patch(color=C_PLANT, label="Plantation"), Patch(color=C_FOREST, label="Natural forest")],
               loc="center right", fontsize=8)
fig.tight_layout()
save(fig, "feature_rates.png")

# sensitivity (TPR vs FPR of 'at least one cut') for both datasets
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, ds in zip(axes, ["val", "qc"]):
    s = sens[sens.dataset == ds]
    for (m, ms, pen), sty in [(("R", np.nan, np.nan), dict(marker="s", color=INK, label="Rules on yearly values")),
                              (("S", 18, 0.5), dict(marker="^", color=MUTED, label="Segmentation, loose (BIC x0.5)")),
                              (("S", 18, 1.0), dict(marker="o", color=INK2, label="Segmentation, default (<= 18 segments)")),
                              (("S", 18, 2.0), dict(marker="D", color="#a8a79f", label="Segmentation, strict (BIC x2)")),
                              (("S", 6, 1.0), dict(marker="v", color="#c3c2b7", label="Segmentation capped at 6 segments"))]:
        ss = s[(s.method == m) & ((s.pen == pen) | (np.isnan(pen) & s.pen.isna()))
               & ((s.max_segments == ms) | (np.isnan(ms) & s.max_segments.isna()))].sort_values("drop")
        ax.plot(ss.fpr, ss.tpr, "-", lw=1.5, color=sty["color"])
        ax.plot(ss.fpr, ss.tpr, sty["marker"], ms=5, color=sty["color"], mec=SURF, mew=1, label=sty["label"])
        if m == "S" and pen == 1.0 and ms == 18:
            for r in ss.itertuples():
                ax.annotate(f"{r.drop:.2f}", (r.fpr, r.tpr), textcoords="offset points", xytext=(5, -3), fontsize=7, color=INK2)
    ax.set_xlabel("Natural forest with at least one 'cut' (false alarm share)")
    ax.set_ylabel("Plantations with at least one cut")
    ax.set_title({"val": "MapBiomas reference points", "qc": "Cropin QC clearings"}[ds] + ": NBR drop threshold", loc="left")
    ax.set_xlim(0, max(0.3, s.fpr.max() + 0.02))
    ax.set_ylim(0, 1)
axes[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
save(fig, "sensitivity_threshold.png")

# added value: AUC by feature set
fig, ax = plt.subplots(figsize=(7.5, 4.6))
order_sets = ["Hansen only", "H4 only", "H-all only", "H-all + Hansen only", "full series only",
              "base (no MapBiomas history)", "base + H4", "base + H-all", "base + Hansen", "base + H-all + Hansen",
              "base + full series", "team (with ever_farm)", "team + H4", "team + H-all"]
for j, sname in enumerate(order_sets):
    for mname, mk, col, dy in [("LR", "s", INK, -0.17), ("RF", "o", MUTED, 0.17)]:
        r = summ[(summ.set == sname) & (summ.model == mname)]
        if r.empty:
            continue
        ax.errorbar(r["mean"], j + dy, xerr=r["std"], fmt=mk, color=col, ms=5, mec=SURF, mew=1, elinewidth=1,
                    capsize=0, label={"LR": "Logistic regression", "RF": "Random forest"}[mname] if j == 0 else None)
ax.set_yticks(range(len(order_sets)))
ax.set_yticklabels(order_sets, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("AUC, spatial CV by municipality (mean +/- sd over 25 folds)")
ax.set_title("Plantation vs natural forest, MapBiomas reference points", loc="left")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
ax.text(0.0, -0.2, "base = team features without MapBiomas history; team = base + ever_farm; H4 = the four harvest "
        "features; H-all = H4 + six continuous NBR-history summaries; Hansen = loss 2001-2020, gain, tree cover 2000",
        transform=ax.transAxes, fontsize=7, color=INK2, va="top", wrap=True)
fig.tight_layout()
save(fig, "added_value_auc.png")
log("figures written to", FIG)
LOG.close()
