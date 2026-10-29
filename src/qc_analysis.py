"""Shared helpers for the QC error-analysis notebooks (01 and 02).

Kept in one place so both notebooks compute AUCs, tests, cross-validation and the
"plantations removed at <=5% forest hidden" metric in exactly the same way.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

# ---------------------------------------------------------------- palette & style
# MIT burgundy for plantation (the false alarms we want to remove), muted blue for forest,
# greys for context. Burgundy/blue pair checked for colour-blind separation (deltaE 23).
PAL = {
    "plantation": "#750014",
    "forest": "#3B6EA8",
    "grey": "#8C8C8C",
    "light": "#D0D0D0",
    "ink": "#333333",
}
LABELS = {"plantation": "Plantation (false alarm)", "forest": "Forest (real deforestation)"}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": "#999999", "axes.labelcolor": PAL["ink"], "text.color": PAL["ink"],
        "xtick.color": "#666666", "ytick.color": "#666666",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    })


def savefig(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")


def dist_plot(ax, d, col, xlabel, log=False, seed=0, label_col="qc_label"):
    """Box plot plus jittered dots for plantation vs forest on one horizontal axis."""
    rng = np.random.default_rng(seed)
    for i, lab in enumerate(["forest", "plantation"]):
        x = d.loc[d[label_col] == lab, col].dropna().values
        ax.scatter(x, i + rng.uniform(-0.18, 0.18, len(x)), s=9, alpha=0.45,
                   color=PAL[lab], linewidths=0)
        ax.boxplot(x, positions=[i], vert=False, widths=0.55, showfliers=False,
                   medianprops=dict(color=PAL["ink"], linewidth=2),
                   boxprops=dict(color=PAL["grey"]), whiskerprops=dict(color=PAL["grey"]),
                   capprops=dict(color=PAL["grey"]))
        ax.text(1.01, i, f"median {np.median(x):.3g}\n(n = {len(x)})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=PAL["ink"])
    ax.set_yticks([0, 1], [LABELS["forest"], LABELS["plantation"]])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")


# ---------------------------------------------------------------- data
MB_YEARS = list(range(1985, 2024))
MB_COLS = [f"mb_{y}" for y in MB_YEARS]
FARM_CODES = [21, 41, 19, 12, 15]   # mosaic, other temporary crops, other crops, grassland, pasture


def load_qc():
    """Load and merge the two GEE exports on id, with integrity checks."""
    a = pd.read_csv(DATA / "gee_exports/qc_error_analysis_A.csv")
    b = pd.read_csv(DATA / "gee_exports/qc_error_analysis_B_ndvi.csv")
    assert a["id"].is_unique and b["id"].is_unique, "duplicate ids"
    assert set(a["id"]) == set(b["id"]), "id sets differ between A and B"
    lab_b = b.set_index("id")["qc_label"]
    assert (a.set_index("id")["qc_label"] == lab_b.loc[a["id"]].values).all(), "labels disagree"
    d = a.merge(b.drop(columns="qc_label"), on="id", how="inner", validate="one_to_one")
    return d


def duplicate_key(d):
    """Key shared by repeated QC polygons. 12 locations appear 2-3 times in Cropin's file
    with IoU 1.0 (coordinates differ only by float noise), so centroid + area identify them."""
    return (d["lat"].round(6).astype(str) + "_" + d["lon"].round(6).astype(str)
            + "_" + d["area_m2"].round(2).astype(str))


def dedupe(d):
    """Keep the lowest id of each repeated polygon (labels always agree)."""
    k = duplicate_key(d)
    assert d.groupby(k)["qc_label"].nunique().max() == 1, "repeated polygons with conflicting labels"
    return d.loc[~k.duplicated()].copy()


def since_year(row_vals):
    """First year of the final unbroken run of class 3. 2024 if the polygon was not forest in 2023."""
    i = len(row_vals) - 1
    while i >= 0 and row_vals[i] == 3:
        i -= 1
    return MB_YEARS[0] + i + 1


def add_features(d, ndvi_years=range(2019, 2025)):
    d = d.copy()
    mb = d[MB_COLS]
    d["y"] = (d["qc_label"] == "plantation").astype(int)
    d["mb2020_group"] = d["mb_2020"].map({3: "called forest", 9: "called plantation"}).fillna("called other")
    d["group"] = np.where(d["qc_label"] == "forest", "forest",
                          d["mb_2020"].map({3: "plant_missed", 9: "plant_caught"}).fillna("plant_other"))
    # land history from MapBiomas C9 (30 m mode)
    d["n_yrs_forest"] = (mb == 3).sum(axis=1)
    d["n_class_changes"] = (mb.values[:, 1:] != mb.values[:, :-1]).sum(axis=1)
    d["forest_every_year"] = (mb == 3).all(axis=1).astype(int)
    d["since"] = [since_year(v) for v in mb.values]
    d["after2000"] = (d["since"] > 2000).astype(int)
    d["farm_any"] = mb.isin(FARM_CODES).any(axis=1).astype(int)
    d["ever_plant"] = (mb == 9).any(axis=1).astype(int)
    # NDVI summaries (default window 2019-2024: 2017-18 have too few clear views)
    def ndvi(prefix, yrs):
        d_cols = [f"{prefix}_{y}" for y in yrs]
        return d[d_cols]
    for tag, yrs in [("", ndvi_years), ("_17_24", range(2017, 2025))]:
        p10, med = ndvi("ndvi_p10", yrs), ndvi("ndvi_med", yrs)
        d[f"min_p10{tag}"] = p10.min(axis=1)
        d[f"n_dip_years{tag}"] = (p10 < 0.5).sum(axis=1)
        d[f"any_dip{tag}"] = (d[f"n_dip_years{tag}"] > 0).astype(int)
        d[f"mean_med{tag}"] = med.mean(axis=1)
        d[f"sd_med{tag}"] = med.std(axis=1)
    return d


# ---------------------------------------------------------------- statistics
def auc_oriented(y, x):
    """AUC for 'plantation' = 1. Returns (auc >= 0.5, direction) where direction is
    'higher in plantations' or 'lower in plantations'."""
    a = roc_auc_score(y, x)
    return (a, "higher in plantations") if a >= 0.5 else (1 - a, "lower in plantations")


def auc_ci(y, x, n_boot=2000, seed=0):
    """Point AUC (oriented) plus a percentile bootstrap 95% CI (resampling polygons)."""
    y, x = np.asarray(y), np.asarray(x)
    raw = roc_auc_score(y, x)
    flip = raw < 0.5
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if y[s].min() == y[s].max():
            continue
        bs.append(roc_auc_score(y[s], x[s]))
    bs = np.array(bs)
    if flip:
        raw, bs = 1 - raw, 1 - bs
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return raw, lo, hi


def mwu_test(x_pl, x_fo):
    return mannwhitneyu(x_pl, x_fo, alternative="two-sided").pvalue


def fisher_test(a, b):
    """a, b: 0/1 arrays for group A and group B. Returns dict with shares, odds ratio, p."""
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
    orr, p = fisher_exact(table)
    return dict(pct_a=100 * a.mean(), pct_b=100 * b.mean(), n_a=int(a.sum()), n_b=int(b.sum()),
                N_a=len(a), N_b=len(b), odds_ratio=orr, p=p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- modelling
def make_model():
    """Standardize, then plain L2 logistic regression (sklearn default C=1)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def spatial_groups(d, k=20, seed=0):
    """Spatial blocks: KMeans clusters of polygon centroids (lat/lon)."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(d[["lat", "lon"]].values)


def grid_groups(d, cell=0.25):
    return (np.floor(d["lat"] / cell).astype(int).astype(str) + "_"
            + np.floor(d["lon"] / cell).astype(int).astype(str)).values


def threshold_hiding_at_most(scores_forest, max_share=0.05):
    """Lowest cut t such that at most max_share of forests have score >= t
    (i.e. would be flagged as plantation and hidden). Flag rule everywhere: score >= t."""
    s = np.sort(np.asarray(scores_forest))[::-1]
    allowed = int(np.floor(max_share * len(s)))
    if allowed >= len(s):
        return -np.inf
    return np.nextafter(s[allowed], np.inf)


def _splitter(kind, seed, n_splits=5):
    if kind == "stratified":
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)


def _oof(X, y, groups, kind, seed, model_fn):
    p = np.zeros(len(y))
    for tr, te in _splitter(kind, seed).split(X, y, groups):
        m = model_fn().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def cv_evaluate(X, y, groups=None, kind="stratified", seeds=range(20), max_share=0.05,
                model_fn=make_model):
    """Cross-validated evaluation repeated over seeds.

    For each seed returns:
      auc            pooled out-of-fold AUC
      opt_caught/opt_hidden   threshold picked on the pooled out-of-fold predictions (optimistic,
                              the original method)
      nest_caught/nest_hidden threshold picked INSIDE each training fold, from inner-CV predictions
                              on the training data only, then applied to the untouched test fold
    Also returns pooled per-seed confusion counts for the nested rule.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rows, oofs = [], {}
    for seed in seeds:
        p = np.zeros(len(y))
        flag = np.zeros(len(y), dtype=bool)
        for tr, te in _splitter(kind, seed).split(X, y, groups):
            m = model_fn().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
            g_tr = None if groups is None else np.asarray(groups)[tr]
            p_in = _oof(X[tr], y[tr], g_tr, kind, seed + 1000, model_fn)
            t = threshold_hiding_at_most(p_in[y[tr] == 0], max_share)
            flag[te] = p[te] >= t
        assert np.isfinite(p).all(), "non-finite predictions"
        t_opt = threshold_hiding_at_most(p[y == 0], max_share)
        f_opt = p >= t_opt
        rows.append(dict(
            seed=seed, auc=roc_auc_score(y, p),
            opt_caught=100 * f_opt[y == 1].mean(), opt_hidden=100 * f_opt[y == 0].mean(),
            nest_caught=100 * flag[y == 1].mean(), nest_hidden=100 * flag[y == 0].mean(),
            TP=int((flag & (y == 1)).sum()), FP=int((flag & (y == 0)).sum()),
            FN=int((~flag & (y == 1)).sum()), TN=int((~flag & (y == 0)).sum()),
        ))
        oofs[seed] = p
    return pd.DataFrame(rows), oofs


def per_class_metrics(TP, FP, FN, TN):
    """Precision/recall for both classes when the model 'flags plantation'."""
    return pd.DataFrame({
        "precision": [TP / (TP + FP) if TP + FP else np.nan, TN / (TN + FN) if TN + FN else np.nan],
        "recall": [TP / (TP + FN), TN / (TN + FP)],
        "n_true": [TP + FN, TN + FP],
    }, index=["plantation (flagged)", "forest (kept)"])


def bootstrap_coefs(X, y, names, n_boot=2000, seed=0):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(seed)
    full = make_model().fit(X, y)[-1].coef_[0]
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        bs.append(make_model().fit(X[s], y[s])[-1].coef_[0])
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pd.DataFrame({"coef": full, "ci_low": lo, "ci_high": hi}, index=names)
