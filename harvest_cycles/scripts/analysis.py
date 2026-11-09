#!/usr/bin/env python3
"""
Step 8: can harvest cycles tell plantations from natural forest, and how much do they add?

Usage: python analysis.py <data_dir> <harvest_cycles_dir>

Primary configuration: NBR, yearly median of all clear observations, LandTrendr-style segmentation ("S",
<= 18 segments), thresholds in harvest_lib.HarvestParams. Other indices / aggregations / methods / thresholds
are comparisons. Changed after an independent review, once first results had been seen (disclosed as post hoc):
segment cap 6 -> 18 (the cap-6 version is kept as "S6"), calibration restricted to information up to 2020,
open-land level calibrated on open land, double-counted cuts and duplicate passes removed.

Datasets
  VAL = the 1,831 MapBiomas validation points of data_exploration/cropin_model_features_2022.csv (PR/SC/RS,
        2022 label forest or plantation, no edge points); groups vs MapBiomas 10 m come from that file.
  QC  = Cropin's 433 hand-checked clearings (one Landsat pixel each). Confidential: only aggregates are written
        to results/; per-polygon tables go to <data_dir>/private_results/.
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
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_lib as hl  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

D, OUT = sys.argv[1], sys.argv[2]
RES, FIG = os.path.join(OUT, "results"), os.path.join(OUT, "figures")
PRIV = os.path.join(D, "private_results")
for p in (RES, FIG, PRIV):
    os.makedirs(p, exist_ok=True)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SEED = 42
LOG = open(os.path.join(RES, "analysis_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ----------------------------------------------------------------------------- style
C_FOREST, C_PLANT, C_MISSED = "#1baf7a", "#eb6834", "#2a78d6"
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
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def cluster_boot(y, s, cl, fn=roc_auc_score, n_boot=1000, seed=SEED):
    """95% CI of fn(y, s), resampling whole clusters (municipalities or 0.1-degree cells)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(cl)
    idx_by = [np.where(codes == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_by[p] for p in rng.integers(0, len(uniq), len(uniq))])
        if y[ii].min() != y[ii].max():
            vals.append(fn(y[ii], s[ii]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def auc_diff(yy, ss):
    return roc_auc_score(yy, ss[:, 0]) - roc_auc_score(yy, ss[:, 1])


def odds_ratio(a, b, c, d):
    """Woolf CI; a/b = plantations with/without the feature, c/d = forest with/without (+0.5 if a cell is 0)."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)


def clustered_logit(X, y, groups, names):
    """Logit with cluster-robust (sandwich) standard errors."""
    import statsmodels.api as sm
    Xc = sm.add_constant(np.asarray(X, float), has_constant="add")
    r = sm.GLM(y, Xc, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                         cov_kwds={"groups": pd.factorize(groups)[0]})
    tab = pd.DataFrame({"coef": r.params, "se_cluster": r.bse, "p_cluster": r.pvalues, "OR": np.exp(r.params),
                        "OR_lo": np.exp(r.params - 1.96 * r.bse), "OR_hi": np.exp(r.params + 1.96 * r.bse)},
                       index=["const"] + list(names))
    return tab, r


def recall_at_fpr(y, s, max_fpr):
    fpr, tpr, thr = roc_curve(y, s)
    k = np.where(fpr <= max_fpr)[0]
    j = k[np.argmax(tpr[k])]
    return tpr[j], fpr[j], thr[j]


def grid_cell(d):
    return "g" + (d["lon"] / 0.1).round().astype(int).astype(str) + "_" + (d["lat"] / 0.1).round().astype(int).astype(str)


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
S = pd.read_csv(f"{D}/series_nbr_median.csv.gz", index_col=0)
S.columns = S.columns.astype(int)

han["hansen_loss_le2020"] = han["lossyear"].between(1, 20).astype(int)        # losses after 2020 ignored
han["hansen_gain"] = han["gain"].astype(int)
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


F6 = features_for("nbr", "median", "S6")
F6 = F6[["pid", "n_cycles", "regrow_speed", "forest_since_1985", "abrupt_start", "n_cuts", "years_since_last_cut",
         "max_drop"]].set_index("pid").add_suffix("_s6").reset_index()
df = (pts.merge(features_for("nbr", "median", "S"), on="pid", how="left").merge(F6, on="pid", how="left")
      .merge(han, on="pid", how="left"))
df["cluster"] = np.where(df["dataset"] == "val", df["CD_MUN"].astype(str), grid_cell(df))
val = df[df.dataset == "val"].merge(team.drop(columns=["lat", "lon", "CD_MUN", "plantation"]), on="pt_id", how="left")
val = val.merge(hist[["pid"] + [f"CLASS_{y}" for y in range(1985, 2023)]], on="pid", how="left")
qc = df[df.dataset == "qc"].copy()
log(f"VAL n={len(val)} (plantation {val.label.sum()}), QC n={len(qc)} (plantation {qc.label.sum()})")

# ============================================================================= 1. data and calibration
log("\n=== 1. Landsat coverage, sensor offsets, calibration")
c = cnt.reindex(pts["pid"])
log(f"clear observations per point-year: median {np.median(c.values):.0f}; point-years with none: "
    f"{(c.values == 0).mean():.3f}; worst years: {(c == 0).mean().sort_values(ascending=False).head(3).round(3).to_dict()}")
log("sensor offsets removed (to the Landsat 7 scale):\n" + offs.round(4).to_string(index=False))
log("calibration (label-free, from Hansen):\n" + cal.round(3).to_string(index=False))
log(f"Landsat scenes used: {len(pd.read_csv(f'{D}/scenes_selected.csv'))}; QC polygon area median "
    f"{qc['qc_area_m2'].median():.0f} m2")
cal.to_csv(os.path.join(RES, "calibration_levels.csv"), index=False)

# ============================================================================= 2. ways of using the series
log("\n=== 2. Which way of using the time series separates best? (default thresholds)")
rows = []
for (i, a, m), f in feat_all.groupby(["index", "agg", "method"]):
    g = pts[["pid", "dataset", "label"]].merge(f, on="pid")
    for ds, gg in g.groupby("dataset"):
        y = gg["label"].to_numpy()
        nc = gg["n_cuts"].fillna(0).to_numpy()
        rows.append(dict(index=i, agg=a, method=m, dataset=ds, tpr_cut=(nc[y == 1] >= 1).mean(),
                         fpr_cut=(nc[y == 0] >= 1).mean(), auc_n_cuts=roc_auc_score(y, nc)))
mc = pd.DataFrame(rows)
mc.to_csv(os.path.join(RES, "method_comparison.csv"), index=False)
log(mc.round(3).to_string(index=False))


def score(i, a, m, col):
    f = feat_all[(feat_all["index"] == i) & (feat_all["agg"] == a) & (feat_all["method"] == m)].set_index("pid")[col]
    return f.reindex(pts["pid"]).fillna(0).to_numpy()


NBR_S, NBR_R = ("nbr", "median", "S"), ("nbr", "median", "R")
CMPS = [("NBR vs NDVI (median, rules)", NBR_R, ("ndvi", "median", "R"), "n_cuts", "n_cuts"),
        ("NBR vs NDVI (median, segmentation)", NBR_S, ("ndvi", "median", "S"), "n_cuts", "n_cuts"),
        ("rules vs segmentation (NBR, median)", NBR_R, NBR_S, "n_cuts", "n_cuts"),
        ("segmentation cap 18 vs 6 (NBR, median)", NBR_S, ("nbr", "median", "S6"), "n_cuts", "n_cuts"),
        ("median vs dry-season median (rules)", NBR_R, ("nbr", "dry_median", "R"), "n_cuts", "n_cuts"),
        ("median vs wet-season median (rules)", NBR_R, ("nbr", "wet_median", "R"), "n_cuts", "n_cuts"),
        ("median vs yearly max (rules)", NBR_R, ("nbr", "max", "R"), "n_cuts", "n_cuts"),
        ("largest 3-year fall vs cut count", NBR_S, NBR_S, "s_max_drop3", "n_cuts"),
        ("NBR variability vs cut count", NBR_S, NBR_S, "s_std", "n_cuts")]
cl_all = np.where(pts["dataset"] == "val", pts["CD_MUN"].astype(str), grid_cell(pts))
cmp_rows = []
for name, A, B, ca, cb in CMPS:
    sa, sb = score(*A, ca), score(*B, cb)
    for ds in ["val", "qc"]:
        msk = (pts["dataset"] == ds).to_numpy()
        yy = pts["label"].to_numpy()[msk]
        ss = np.c_[sa[msk], sb[msk]]
        lo, hi = cluster_boot(yy, ss, cl_all[msk], fn=auc_diff)
        cmp_rows.append(dict(comparison=name, dataset=ds, auc_a=roc_auc_score(yy, ss[:, 0]),
                             auc_b=roc_auc_score(yy, ss[:, 1]), diff=auc_diff(yy, ss), ci_lo=lo, ci_hi=hi))
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(RES, "method_comparison_paired_ci.csv"), index=False)
log("\nPaired AUC differences (A - B), 95% CI resampling municipalities (VAL) / 0.1-degree cells (QC):\n" +
    cmp_df.round(3).to_string(index=False))

gs = grid.merge(pts[["pid", "dataset", "label"]], on="pid")
gs["cut"] = (gs["n_cuts"] >= 1).astype(int)
sens = gs.groupby(["dataset", "method", "max_segments", "drop"], dropna=False).apply(
    lambda x: pd.Series(dict(tpr=x.loc[x.label == 1, "cut"].mean(), fpr=x.loc[x.label == 0, "cut"].mean(),
                             auc=roc_auc_score(x["label"], x["n_cuts"])))).reset_index()
sens.to_csv(os.path.join(RES, "sensitivity_drop_threshold.csv"), index=False)
log("\nDrop threshold (NBR, median): share of plantations (tpr) and forest (fpr) with at least one cut:\n" +
    sens.round(3).to_string(index=False))

# ============================================================================= 3. univariate significance
log("\n=== 3. Univariate significance, primary configuration")
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
        # an empty cell = perfect separation: the clustered SE is meaningless, keep only Fisher
        p_cl = np.nan if min(a, b, cc, dd) == 0 else clustered_logit(v[:, None], y, cl, [f])[0].loc[f, "p_cluster"]
        alo, ahi = cluster_boot(y, v.astype(float), cl)
        uni.append(dict(dataset=name, feature=f, kind="binary", plant_rate=a / (a + b), forest_rate=cc / (cc + dd),
                        plant_n=a, forest_n=cc, OR=orr, OR_lo=lo, OR_hi=hi,
                        p_fisher=stats.fisher_exact([[a, b], [cc, dd]])[1], p_cluster=p_cl,
                        auc=roc_auc_score(y, v), auc_lo=alo, auc_hi=ahi))
    for f in CONT:
        v = d[f].astype(float).to_numpy()
        ok = np.isfinite(v)
        auc = roc_auc_score(y[ok], v[ok])
        alo, ahi = cluster_boot(y[ok], v[ok], cl[ok])
        uni.append(dict(dataset=name, feature=f, kind="continuous", plant_median=np.median(v[ok & (y == 1)]),
                        forest_median=np.median(v[ok & (y == 0)]),
                        p_mannwhitney=stats.mannwhitneyu(v[ok & (y == 1)], v[ok & (y == 0)]).pvalue,
                        auc=auc, auc_lo=alo, auc_hi=ahi,
                        direction="lower in plantation" if auc < 0.5 else "higher in plantation"))
    sub = d[d["n_cycles"] >= 1]                       # regrowth time, among points with a cycle only
    v, yy = sub["regrow_years"].to_numpy(float), sub["label"].to_numpy()
    ok = np.isfinite(v)
    uni.append(dict(dataset=name, feature="regrow_years (points with a cycle)", kind="continuous",
                    plant_median=np.median(v[ok & (yy == 1)]), forest_median=np.median(v[ok & (yy == 0)]),
                    plant_n=int((ok & (yy == 1)).sum()), forest_n=int((ok & (yy == 0)).sum()),
                    p_mannwhitney=stats.mannwhitneyu(v[ok & (yy == 1)], v[ok & (yy == 0)]).pvalue,
                    auc=roc_auc_score(yy[ok], -v[ok]), direction="faster in plantation if auc > 0.5"))
uni = pd.DataFrame(uni)
uni.to_csv(os.path.join(RES, "univariate_significance.csv"), index=False)
log(uni.round(4).to_string(index=False))

# ============================================================================= 4. groups vs MapBiomas 10 m
log("\n=== 4. VAL groups (MapBiomas 10 m collection 4 vs the reference)")
grp_rows = []
for g in ["real forest", "forest called plantation", "caught plantation", "missed plantation", "other"]:
    d = val[val.group == g]
    grp_rows.append(dict(group=g, n=len(d), **{f: d[f].fillna(0).mean() for f in
                    ["has_cut", "has_cycle", "fast_regrowth", "abrupt_start", "forest_since_1985", "hansen_loss_le2020",
                     "hansen_gain"]}))
grp = pd.DataFrame(grp_rows)
grp.to_csv(os.path.join(RES, "rates_by_mapbiomas_group.csv"), index=False)
log(grp.round(3).to_string(index=False))
for g in ["missed plantation", "forest called plantation"]:
    for f in ["has_cut", "forest_since_1985", "hansen_loss_le2020"]:
        a = val[val.group == g][f].fillna(0)
        b = val[val.group == "real forest"][f].fillna(0)
        p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        log(f"  {g} vs real forest, {f}: {a.mean():.3f} vs {b.mean():.3f}, Fisher p={p:.2g}")

# ============================================================================= 5. added value in models
log("\n=== 5. Added value on top of the team's features (VAL, spatial CV by municipality)")
BASE_TEAM = ["height_stdDev", "ndvi_aug_oct_mean", "sag_oct_nov_mean", "sag_swing_mean", "ever_farm"]
BASE = [f for f in BASE_TEAM if f != "ever_farm"]   # ever_farm = MapBiomas history: excluded when testing on MapBiomas
H4 = ["n_cycles", "regrow_speed", "forest_since_1985", "abrupt_start"]
HALL = H4 + ["n_cuts", "years_since_last_cut", "max_drop", "s_frac_below_forest", "s_min", "s_max_drop3"]
HANS = ["hansen_loss_le2020", "hansen_gain", "tc2000"]
HALL6 = [f + "_s6" for f in H4 + ["n_cuts", "years_since_last_cut", "max_drop"]] + HALL[-3:]
fullmat = hl.full_series_matrix(S)
FULL = list(fullmat.columns)
V = val.dropna(subset=BASE_TEAM).merge(fullmat, left_on="pid", right_index=True, how="left").reset_index(drop=True)
for f in HALL + HANS + HALL6 + ["s_std"]:
    V[f] = V[f].astype(float).fillna(V[f].median())
V[FULL] = V[FULL].fillna(V[FULL].median())
log(f"model rows: {len(V)} (plantation {V.label.sum()}, missed plantations {(V.group == 'missed plantation').sum()})")


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))


def make_rf(seed=SEED):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1)


SETS = {
    "team (with ever_farm)": BASE_TEAM, "team + H4": BASE_TEAM + H4, "team + H-all": BASE_TEAM + HALL,
    "base (no MapBiomas history)": BASE, "base + H4": BASE + H4, "base + H-all": BASE + HALL,
    "base + Hansen": BASE + HANS, "base + H-all + Hansen": BASE + HALL + HANS,
    "base + H-all [segment cap 6, as first specified]": BASE + HALL6,
    "H4 only": H4, "H-all only": HALL, "Hansen only": HANS, "H-all + Hansen only": HALL + HANS, "full series only": FULL,
    # post hoc, added after NBR variability turned out to be the strongest single summary
    "base + H-all + variability (post hoc)": BASE + HALL + ["s_std"],
    "H-all + variability only (post hoc)": HALL + ["s_std"],
}
PAIRS = {"team + H4": "team (with ever_farm)", "team + H-all": "team (with ever_farm)",
         "base + H4": "base (no MapBiomas history)", "base + H-all": "base (no MapBiomas history)",
         "base + Hansen": "base (no MapBiomas history)", "base + H-all + Hansen": "base (no MapBiomas history)",
         "base + H-all [segment cap 6, as first specified]": "base (no MapBiomas history)",
         "base + H-all + variability (post hoc)": "base + H-all", "H-all + variability only (post hoc)": "H-all only"}
REPEATS = [42, 43, 44, 45, 46]
y = V["label"].to_numpy()
groups = V["CD_MUN"].to_numpy()
cv_rows, oof = [], {}
for rep in REPEATS:
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(V, y, groups))
    for sname, cols in SETS.items():
        X = V[cols].to_numpy(float)
        for mname in ["LR", "RF"]:
            if sname.startswith("full") and mname == "LR":
                continue
            o = np.full(len(y), np.nan)
            for k, (tr, te) in enumerate(folds):
                mdl = (make_lr() if mname == "LR" else make_rf(rep)).fit(X[tr], y[tr])
                o[te] = mdl.predict_proba(X[te])[:, 1]
                cv_rows.append(dict(repeat=rep, fold=k, set=sname, model=mname, auc=roc_auc_score(y[te], o[te])))
            if rep == SEED:
                oof[(sname, mname)] = o
cvr = pd.DataFrame(cv_rows)
summ = cvr.groupby(["set", "model"]).auc.agg(["mean", "std"]).reset_index()
summ.to_csv(os.path.join(RES, "cv_auc_by_set.csv"), index=False)
delta_rows = []
for new, base in PAIRS.items():
    for mname in ["LR", "RF"]:
        a = cvr[(cvr.set == new) & (cvr.model == mname)].set_index(["repeat", "fold"]).auc
        b = cvr[(cvr.set == base) & (cvr.model == mname)].set_index(["repeat", "fold"]).auc
        dlt = (a - b).dropna()
        lo, hi = cluster_boot(y, np.c_[oof[(new, mname)], oof[(base, mname)]], groups, fn=auc_diff)
        delta_rows.append(dict(model=mname, new=new, base=base, auc_new=a.mean(), auc_base=b.mean(),
                               delta_mean=dlt.mean(), delta_sd=dlt.std(ddof=1),
                               folds_improved=f"{int((dlt > 0).sum())}/{len(dlt)}",
                               delta_oof_seed42=auc_diff(y, np.c_[oof[(new, mname)], oof[(base, mname)]]),
                               delta_ci_lo_cluster_boot=lo, delta_ci_hi_cluster_boot=hi))
dl = pd.DataFrame(delta_rows)
dl.to_csv(os.path.join(RES, "added_value_delta_auc.csv"), index=False)
log("AUC by feature set (mean, sd over 5 repeats x 5 spatial folds):\n" + summ.round(3).to_string(index=False))
log("Paired gains:\n" + dl.round(3).to_string(index=False))
loso = []                                     # stricter spatial test: train on two states, test on the third
for st in ["PR", "SC", "RS"]:
    tr, te = (V["state"] != st).to_numpy(), (V["state"] == st).to_numpy()
    for base_name, new_name in [("team (with ever_farm)", "team + H-all"), ("base (no MapBiomas history)", "base + H-all")]:
        for mname in ["LR", "RF"]:
            a_ = [roc_auc_score(y[te], (make_lr() if mname == "LR" else make_rf()).fit(V[SETS[n_]].to_numpy(float)[tr], y[tr])
                                .predict_proba(V[SETS[n_]].to_numpy(float)[te])[:, 1]) for n_ in (base_name, new_name)]
            loso.append(dict(held_out=st, sets=f"{base_name} -> {new_name}", model=mname, auc_base=a_[0], auc_new=a_[1],
                             gain=a_[1] - a_[0]))
loso = pd.DataFrame(loso)
log("Leave-one-state-out (train on two states, test on the third):\n" + loso.round(3).to_string(index=False))

# operating points (seed-42 OOF scores; thresholds picked on the same scores, so optimistic)
op_rows = []
gv = V["group"].to_numpy()
for (sname, mname), o in oof.items():
    for max_fpr in [0.05, 0.01]:
        tpr, fpr, thr = recall_at_fpr(y, o, max_fpr)
        flag = o >= thr
        op_rows.append(dict(set=sname, model=mname, max_fpr=max_fpr, threshold=thr, recall=tpr, fpr=fpr,
                            missed_caught=int((flag & (gv == "missed plantation")).sum()),
                            missed_n=int((gv == "missed plantation").sum()),
                            realforest_flagged=int((flag & (gv == "real forest")).sum()),
                            fcp_flagged=int((flag & (gv == "forest called plantation")).sum()),
                            fcp_n=int((gv == "forest called plantation").sum())))
ops = pd.DataFrame(op_rows)
ops.to_csv(os.path.join(RES, "operating_points.csv"), index=False)
log("Operating points (OOF, seed 42):\n" + ops.round(3).to_string(index=False))
mbp = (V["mb_pred"] == 2).astype(int).to_numpy()
log(f"MapBiomas 10 m on these rows: recall {mbp[y == 1].mean():.3f}, FPR {mbp[y == 0].mean():.3f}")
o = oof[("team + H-all", "RF")]
flag = o >= recall_at_fpr(y, o, 0.05)[2]
for lab, what in [(1, "plantations flagged"), (0, "forest flagged")]:
    m_ = y == lab
    only_model, only_mb = int((flag[m_] & (mbp[m_] == 0)).sum()), int((~flag[m_] & (mbp[m_] == 1)).sum())
    log(f"  team + H-all RF (FPR <= 5%) vs MapBiomas 10 m, {what}: {flag[m_].mean():.3f} vs {mbp[m_].mean():.3f}; "
        f"model only {only_model}, MapBiomas only {only_mb}, exact McNemar p={stats.binomtest(only_model, only_model + only_mb).pvalue:.2g}")

# municipality-clustered logit, base + H4 (continuous features standardised, binary not)
Xs = V[BASE + H4].astype(float).copy()
for col in BASE + ["n_cycles", "regrow_speed"]:
    Xs[col] = (Xs[col] - Xs[col].mean()) / Xs[col].std()
tab, res = clustered_logit(Xs.to_numpy(), y, groups, BASE + H4)
tab.to_csv(os.path.join(RES, "logit_base_plus_H4_clustered.csv"))
R = np.zeros((len(H4), len(tab)))
for j in range(len(H4)):
    R[j, len(BASE) + 1 + j] = 1
w = res.wald_test(R, scalar=True)
log("Logit, municipality-clustered SEs (continuous per SD, binary per unit):\n" + tab.round(4).to_string())
log(f"Joint Wald test of the 4 harvest features (clustered): chi2={float(w.statistic):.1f}, df={len(H4)}, "
    f"p={float(w.pvalue):.2g}")

# ============================================================================= 6. the safe veto
log("\n=== 6. 'Forest every year since 1985' as a veto on plantation calls")
cal0 = cal[(cal["index"] == "nbr") & (cal["agg"] == "median")].iloc[0]
P0 = hl.HarvestParams(forest_level=float(cal0.closed_p5), open_level=float(cal0.open_level))


def passes_since(pid, start):
    """'Forest since <start>' (tolerance 0): no cut from `start` on (segmentation of the full series) and no
    despiked yearly value below closed canopy from `start` on."""
    t, yv = hl.valid_tv(S.loc[pid])
    tv, vv, fit = hl.segment(t, yv)
    cuts = sum(e["cut_year"] >= start for e in hl.cycles_from_segments(tv, vv, t, fit, P0))
    return int(cuts == 0 and np.sum(hl.despike(yv)[t >= start] < P0.forest_level) == 0)


p93 = {pid: passes_since(pid, 1993) for pid in pts["pid"]}
veto_rows = []
for window, tol in [("1985-2020", 0), ("1985-2020", 1), ("1985-2020", 2), ("1993-2020", 0)]:
    if window == "1993-2020":
        fsv, fsq = val["pid"].map(p93), qc["pid"].map(p93)
    else:
        fsv = ((val["n_cuts"].fillna(0) == 0) & (val["years_below_forest"].fillna(99) <= tol)).astype(int)
        fsq = ((qc["n_cuts"].fillna(0) == 0) & (qc["years_below_forest"].fillna(99) <= tol)).astype(int)
    mb = (val["mb_pred"] == 2)
    row = dict(window=window, tolerance_years=tol, val_forest_protected=fsv[val.label == 0].mean(),
               val_plant_vetoed=fsv[val.label == 1].mean(),
               mb_forest_calls_cancelled=int(fsv[mb & (val.label == 0)].sum()), mb_forest_calls=int((mb & (val.label == 0)).sum()),
               mb_correct_calls_lost=int(fsv[mb & (val.label == 1)].sum()), mb_correct_calls=int((mb & (val.label == 1)).sum()),
               qc_forest_protected=fsq[qc.label == 0].mean(), qc_plant_vetoed=fsq[qc.label == 1].mean())
    fv = (V["pid"].map(p93).to_numpy() == 1 if window == "1993-2020" else
          ((V["n_cuts"].fillna(0) == 0) & (V["years_below_forest"].fillna(99) <= tol)).to_numpy())
    for sname in ["base (no MapBiomas history)", "base + H-all"]:      # on top of RF flags at FPR <= 5%
        flag = oof[(sname, "RF")] >= recall_at_fpr(y, oof[(sname, "RF")], 0.05)[2]
        row.update({f"{sname}: forest flags": int((flag & (y == 0)).sum()),
                    f"{sname}: forest flags vetoed": int((flag & (y == 0) & fv).sum()),
                    f"{sname}: plantation flags": int((flag & (y == 1)).sum()),
                    f"{sname}: plantation flags vetoed": int((flag & (y == 1) & fv).sum())})
    veto_rows.append(row)
veto = pd.DataFrame(veto_rows)
veto.to_csv(os.path.join(RES, "safe_veto.csv"), index=False)
log(veto.round(3).T.to_string())

# ============================================================================= 7. Cropin QC (independent test)
log("\n=== 7. Cropin QC clearings: train on VAL, test on QC")
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
        pq = (make_lr() if mname == "LR" else make_rf()).fit(V[cols].to_numpy(float), y).predict_proba(
            Q[cols].to_numpy(float))[:, 1]
        thr = recall_at_fpr(y, oof[(sname, mname)], 0.05)[2]
        lo, hi = cluster_boot(yq, pq, Q["cluster"].to_numpy())
        qrows.append(dict(set=sname, model=mname, qc_auc=roc_auc_score(yq, pq), qc_auc_lo=lo, qc_auc_hi=hi,
                          recall_at_val_thr=(pq[yq == 1] >= thr).mean(),
                          forest_flagged_at_val_thr=(pq[yq == 0] >= thr).mean()))
qt = pd.DataFrame(qrows)
qt.to_csv(os.path.join(RES, "qc_transfer.csv"), index=False)
log(qt.round(3).to_string(index=False))
log(f"QC with a cut in 2010-2020: plantation {(qc[qc.label == 1]['last_cut_year'].between(2010, 2020)).mean():.3f}, "
    f"forest {(qc[qc.label == 0]['last_cut_year'].between(2010, 2020)).mean():.3f}")
log("QC last cut year, plantations with a cut:",
    qc[(qc.label == 1) & (qc.n_cuts >= 1)]["last_cut_year"].value_counts().sort_index().to_dict())
by_size = qc.assign(size=pd.cut(qc["qc_area_m2"], [0, 300, 900, 2700, 1e9], labels=["<300 m2", "300-900", "900-2700", ">2700"]))
log("QC by polygon size:\n" + by_size.groupby(["size", "label"], observed=True)[["has_cut", "forest_since_1985"]]
    .agg(["mean", "size"]).round(3).to_string())
qc[["pid", "label", "qc_area_m2", "n_cuts", "n_cycles", "last_cut_year", "regrow_years", "forest_since_1985",
    "abrupt_start", "s_std", "hansen_loss_le2020", "hansen_gain"]].to_csv(os.path.join(PRIV, "qc_features_primary.csv"),
                                                                      index=False)

# ============================================================================= 8. where it works and fails
log("\n=== 8. Where does it work and fail? (VAL; the reference label history is used only to stratify)")
H = val[[f"CLASS_{y}" for y in range(1985, 2023)]].to_numpy()
val["first_plantation_year"] = np.array([1985 + int(np.argmax(r == 9)) if (r == 9).any() else np.nan for r in H])
val["est_period"] = pd.cut(val["first_plantation_year"].where(val["label"] == 1), [1984, 1985, 2000, 2013, 2022],
                           labels=["plantation in 1985", "1986-2000", "2001-2013", "2014-2020 (1st rotation)"]
                           ).astype(str).replace("nan", "")
val["forest_history"] = np.where(val.label == 0, np.where((H[:, :36] == 3).all(axis=1), "forest every year 1985-2020",
                                                          "non-forest in some year"), "")
dist_cwb = np.hypot((val.lon + 49.27) * 111 * np.cos(np.deg2rad(25.43)), (val.lat + 25.43) * 111)
val["bracatinga"] = np.where(val.label == 0, np.where(dist_cwb <= 80, "forest within 80 km of Curitiba", "forest elsewhere"), "")
strat = []
for col in ["est_period", "forest_history", "bracatinga"]:
    for k, d in val[val[col] != ""].groupby(col):
        strat.append(dict(stratum=col, value=k, n=len(d), has_cut=d.has_cut.mean(), forest_since_1985=d.forest_since_1985.mean(),
                          abrupt_start=d.abrupt_start.mean()))
strat = pd.DataFrame(strat)
strat.to_csv(os.path.join(RES, "stratified_rates.csv"), index=False)
log(strat.round(3).to_string(index=False))
val[["pid", "pt_id", "label", "group", "state", "n_cuts", "n_cycles", "last_cut_year", "years_since_last_cut", "max_drop",
     "regrow_years", "regrow_speed", "forest_since_1985", "abrupt_start", "years_below_forest", "s_min", "s_max_drop3",
     "s_frac_below_forest", "s_std", "hansen_loss_le2020", "hansen_gain", "tc2000", "first_plantation_year"]].to_csv(
    os.path.join(RES, "val_harvest_features_primary.csv"), index=False)

# ============================================================================= 9. agreement with independent sources
log("\n=== 9. Agreement with independent sources")
fl = float(cal0.closed_p5)
hp = df[df["lossyear"].between(1, 20)]
win = {0: [], 1: [], 2: []}
for r in hp.itertuples():
    t, yv = hl.valid_tv(S.loc[r.pid])                       # raw yearly values, simple detector
    falls = np.array([t[k] for k in range(1, len(t)) if yv[max(0, k - 3):k].max() - yv[k] >= 0.20])
    for k in win:
        win[k].append(bool(len(falls)) and bool(np.any(np.abs(falls - (2000 + r.lossyear)) <= k)))
log(f"Hansen loss 2001-2020 (n={len(hp)}): segmentation finds a cut {hp['has_cut'].mean():.3f}; an NBR fall >= 0.20 "
    f"within 0/1/2 years of Hansen's year: " + "/".join(f"{np.mean(v):.3f}" for v in win.values()))
coh = []
for r in val[val.label == 1].itertuples():
    h = np.array([getattr(r, f"CLASS_{yr}") for yr in range(1985, 2023)])
    k9 = int(np.argmax(h == 9))
    j = k9
    while j > 0 and h[j - 1] not in (3, 9):
        j -= 1
    if not (1988 <= 1985 + k9 <= 2018) or k9 - j < 3:       # planted on >= 3 years of open land
        continue
    t, yv = hl.valid_tv(S.loc[r.pid])
    low = t[(t >= 1985 + j) & (t < 1985 + k9) & (yv < fl)]
    closed_after = t[(t > low.max()) & (yv >= fl)] if len(low) else []
    coh.append(dict(abrupt=r.abrupt_start, lag=closed_after[0] - (1985 + k9) if len(closed_after) else np.nan))
coh = pd.DataFrame(coh)
lag = coh["lag"].dropna()
log(f"Plantations the reference shows planted on >= 3 years of open land (n={len(coh)}): first year at closed canopy "
    f"minus reference planting year: median {lag.median():.0f}, IQR {lag.quantile(.25):.0f} to {lag.quantile(.75):.0f}, "
    f"within -1..+3: {lag.between(-1, 3).mean():.3f} (n={len(lag)}); abrupt start detected: {coh['abrupt'].mean():.3f}")
fc = val[(val.label == 0) & (val.n_cuts >= 1)]["last_cut_year"]
p30 = hl.HarvestParams(forest_level=fl, open_level=0.30)
ab30 = val["pid"].map({pid: hl.establishment(*hl.valid_tv(S.loc[pid]), p30)[0] for pid in val["pid"]})
log(f"Abrupt start with the first-run open level 0.30 vs the calibrated {float(cal0.open_level):.3f}: plantations "
    f"{ab30[val.label == 1].mean():.3f} vs {val[val.label == 1]['abrupt_start'].mean():.3f}, forest "
    f"{ab30[val.label == 0].mean():.3f} vs {val[val.label == 0]['abrupt_start'].mean():.3f}")
for hist_, d in val[val.label == 0].groupby("forest_history"):
    log(f"Real forest, {hist_}: {int(d.has_cut.sum())} of {len(d)} with a cut")
log(f"Real-forest points with a cut (n={len(fc)}), last cut in 1986-1992 / 1993-2000 / 2001-2020: "
    f"{fc.between(1986, 1992).mean():.3f} / {fc.between(1993, 2000).mean():.3f} / {fc.between(2001, 2020).mean():.3f}")

# ============================================================================= 10. figures
log("\n=== 10. Figures")


def band(ax, ids, color, label):
    m = S.reindex(ids)
    ax.fill_between(hl.YEARS, m.quantile(0.25), m.quantile(0.75), color=color, alpha=0.10, lw=0)
    ax.plot(hl.YEARS, m.median(), color=color, lw=2, label=f"{label} (n={len(ids)})")


fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
band(axes[0], val[val.group == "real forest"].pid, C_FOREST, "Natural forest, MapBiomas right")
band(axes[0], val[val.group == "caught plantation"].pid, C_PLANT, "Plantation, MapBiomas right")
band(axes[0], val[val.group == "missed plantation"].pid, C_MISSED, "Plantation MapBiomas calls forest")
axes[0].set_title("MapBiomas reference points", loc="left")
axes[0].set_ylabel("NBR (annual median, TOA)")
band(axes[1], qc[qc.label == 0].pid, C_FOREST, "Forest (real deforestation)")
band(axes[1], qc[qc.label == 1].pid, C_PLANT, "Plantation (false alarm)")
axes[1].set_title("Cropin QC clearings (cut in 2026)", loc="left")
for ax in axes:
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(1985, 2020)
    ax.set_ylim(0.2, 0.85)
fig.suptitle("Yearly NBR, 1985-2020 (line = median, band = interquartile range)", fontsize=9, color=INK2, x=0.01, ha="left")
fig.tight_layout()
save(fig, "nbr_trajectories_by_group.png")

forest_level = float(cal[(cal["index"] == "nbr") & (cal["agg"] == "median")].closed_p5.iloc[0])


def example(ax, pid, title, color):
    t, yv = hl.valid_tv(S.loc[pid])
    tv, vv, fit = hl.segment(t, yv)
    ev = hl.cycles_from_segments(tv, vv, t, fit, hl.HarvestParams(forest_level=forest_level))
    for e in ev:
        ax.axvspan(e["cut_year"] - 0.5, e["low_year"] + 0.5, color=GRID, alpha=0.9, lw=0)
    ax.plot(t, yv, "o", ms=3.5, color=MUTED, mec=SURF, mew=0.8)
    ax.plot(tv, vv, "-", color=color, lw=2)
    ax.plot(tv, vv, "o", ms=5, color=color, mec=SURF, mew=1.5)
    ax.axhline(forest_level, color=AXIS, lw=0.8)
    ax.set_title(f"{title}\n{len(ev)} cut(s), {sum(e['cycle'] for e in ev)} cycle(s)", loc="left", fontsize=9)
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

names = {"has_cut": "At least one cut", "has_cycle": "At least one cut-and-regrow cycle", "two_plus_cycles": "Two or more cycles",
         "fast_regrowth": "Cycle with regrowth <= 4 years", "abrupt_start": "Abrupt start (open land, fast greening)",
         "forest_since_1985": "Forest every year since 1985", "hansen_loss_le2020": "Hansen loss 2001-2020",
         "hansen_gain": "Hansen gain 2000-2012"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (nm, d) in zip(axes, [("MapBiomas reference points", val), ("Cropin QC clearings", qc)]):
    for j, f in enumerate(names):
        for lab, col, off in [(1, C_PLANT, -0.18), (0, C_FOREST, 0.18)]:
            dd = d[d.label == lab][f].fillna(0)
            lo, hi = wilson(dd.sum(), len(dd))
            ax.barh(j + off, dd.mean(), height=0.32, color=col, lw=0)
            ax.plot([lo, hi], [j + off, j + off], color=INK2, lw=1)
            ax.text(min(hi + 0.02, 0.98), j + off, f"{100 * dd.mean():.0f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names.values()), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share of points (95% CI)")
    ax.set_title(nm, loc="left")
    ax.grid(axis="y", visible=False)
axes[1].legend(handles=[Patch(color=C_PLANT, label="Plantation"), Patch(color=C_FOREST, label="Natural forest")],
               loc="center right", fontsize=8)
fig.tight_layout()
save(fig, "feature_rates.png")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
order_sets = ["Hansen only", "H4 only", "H-all only", "H-all + Hansen only", "full series only",
              "base (no MapBiomas history)", "base + H4", "base + H-all", "base + Hansen", "base + H-all + Hansen",
              "team (with ever_farm)", "team + H4", "team + H-all"]
for j, sname in enumerate(order_sets):
    for mname, mk, col, dy in [("LR", "s", INK, -0.17), ("RF", "o", MUTED, 0.17)]:
        r = summ[(summ.set == sname) & (summ.model == mname)]
        if not r.empty:
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
