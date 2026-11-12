#!/usr/bin/env python3
"""
Proof of concept: which remote-sensing features separate plantation from natural
forest in southern Brazil (PR, SC, RS)? Labels = MapBiomas validation points (2022).

Usage:  python poc_plantation_vs_forest.py --data cropin_model_features_2022.csv --out outputs

Runs (same code, different row/feature sets):
  A_withVH_spec    : exactly the requested spec (6 primary features incl. Sentinel-1 VH).
                     Rows missing any primary/candidate feature are dropped (VH is missing
                     for ~72% of rows, so this keeps ~520 rows).
  A_noVH_ablation  : same rows as A, primary features WITHOUT VH. Isolates the effect of VH.
  B_noVH_allrows   : primary features without VH on all rows with data (~1830 rows).

Nothing is tuned. Fixed seed. Metadata columns are never used as features.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
N_SPLITS = 5
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

# ---------------------------------------------------------------- configuration
META = ["system:index", "pt_id", "lon", "lat", "CD_MUN", "group", "mb_pred", "mb_code",
        ".geo", "plantation"]
PRIMARY = ["height_stdDev", "vh_mean", "ndvi_aug_oct_mean", "sag_oct_nov_mean",
           "sag_swing_mean", "ever_farm"]
CANDIDATES = ["height_mean", "height_stdDev", "vh_mean", "vv_mean", "vhvv_mean",
              "ndvi_aug_oct_mean", "ndvi_year_mean", "ndvi_p10_jul_nov_mean",
              "ndmi_aug_oct_mean", "sag_oct_nov_mean", "sag_year_mean", "sag_swing_mean",
              "ever_farm", "farm_years_mean"]
S1 = ["vh_mean", "vv_mean", "vhvv_mean"]
GROUPS = {"missed plantation": "missed", "real forest": "realforest",
          "forest called plantation": "fcp"}


# ---------------------------------------------------------------- helpers
def make_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000,
                                            random_state=SEED))


def make_rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


MODELS = {"LR": make_lr, "RF": make_rf}


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def cv_oof(make_model, X, y, groups):
    """5-fold GroupKFold by municipality; returns OOF probabilities, fold ids, per-fold AUCs, models."""
    gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    fold = np.full(len(y), -1)
    aucs, models, tests = [], [], []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = make_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold[te] = k
        aucs.append(roc_auc_score(y[te], p))
        models.append(m)
        tests.append(te)
    return oof, fold, np.array(aucs), models, tests


def cluster_boot_auc(y, s, cl, n_boot=2000):
    """95% CI for AUC, resampling whole municipalities (respects spatial clustering)."""
    rng = np.random.default_rng(SEED)
    codes, uniq = pd.factorize(cl)
    idx_by = [np.where(codes == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by[p] for p in pick])
        if y[ii].min() == y[ii].max():
            continue
        vals.append(roc_auc_score(y[ii], s[ii]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def op_points(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))

    def at_fpr(t):
        idx = np.where(fpr <= t)[0]
        return thr[idx[np.argmax(tpr[idx])]]

    return {"t=0.5": 0.5, "Youden": float(thr[j]), "FPR<=5%": float(at_fpr(0.05)),
            "FPR<=1%": float(at_fpr(0.01))}


def binary_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    rec_lo, rec_hi = wilson(tp, tp + fn)
    fpr_lo, fpr_hi = wilson(fp, fp + tn)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tp + tn) / len(y),
                precision=div(tp, tp + fp), recall=div(tp, tp + fn),
                recall_lo=rec_lo, recall_hi=rec_hi, fpr=div(fp, fp + tn),
                fpr_lo=fpr_lo, fpr_hi=fpr_hi, fnr=div(fn, fn + tp))


def group_flags(pred, grp):
    """Share of each group flagged as plantation (Task 7)."""
    out = {}
    for gname, key in GROUPS.items():
        m = grp == gname
        n, k = int(m.sum()), int(pred[m].sum())
        lo, hi = wilson(k, n)
        out.update({f"{key}_n": n, f"{key}_flagged": k,
                    f"{key}_share": k / n if n else np.nan, f"{key}_lo": lo, f"{key}_hi": hi})
    return out


def vif_table(X, names):
    vifs = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        A = np.column_stack([np.ones(len(yj)), np.delete(X, j, axis=1)])
        beta = np.linalg.lstsq(A, yj, rcond=None)[0]
        r2 = 1 - (yj - A @ beta).var() / yj.var()
        vifs.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return pd.Series(vifs, index=names, name="VIF")


def irls_logit(X, y, tol=1e-10, maxit=100):
    """Unpenalised logistic regression by Newton/IRLS; Wald SEs. X must include the constant."""
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < tol:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    pv = 2 * stats.norm.sf(np.abs(beta / se))
    return beta, se, pv


def fit_logit(Xs, y, names):
    """statsmodels Logit if installed, else the equivalent IRLS implementation."""
    Xc = np.column_stack([np.ones(len(y)), Xs])
    try:
        import statsmodels.api as sm
        res = sm.Logit(y, Xc).fit(disp=0)
        beta, se, pv = np.asarray(res.params), np.asarray(res.bse), np.asarray(res.pvalues)
        engine = "statsmodels.Logit"
    except ImportError:
        beta, se, pv = irls_logit(Xc, y)
        engine = "IRLS fallback (statsmodels not installed; same MLE and Wald inference)"
        # sanity check against sklearn (practically unpenalised)
        sk = LogisticRegression(C=1e10, max_iter=5000).fit(Xs, y)
        diff = np.abs(np.r_[sk.intercept_, sk.coef_.ravel()] - beta).max()
        print(f"  [check] max |coef diff| IRLS vs sklearn(C=1e10): {diff:.2e}")
    z = stats.norm.ppf(0.975)
    tab = pd.DataFrame({"coef_per_SD": beta, "se": se,
                        "odds_ratio_per_SD": np.exp(beta),
                        "OR_ci_low": np.exp(beta - z * se), "OR_ci_high": np.exp(beta + z * se),
                        "p_value": pv}, index=["const"] + list(names))
    return tab, engine


# ---------------------------------------------------------------- plotting
def plot_roc(run, y, scores, mb, outdir):
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for name, (s, aucs) in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name}: AUC {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f} (folds)")
        t = op_points(y, s)["Youden"]
        pr = s >= t
        ax.plot((pr & (y == 0)).sum() / (y == 0).sum(), (pr & (y == 1)).sum() / (y == 1).sum(),
                "o", color=ax.lines[-1].get_color(), ms=6, mfc="white")
    ax.plot(mb["fpr"], mb["recall"], "k*", ms=13, label="MapBiomas (binary map)")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (forest called plantation)")
    ax.set_ylabel("True positive rate (plantation recall)")
    ax.set_title(f"ROC, out-of-fold, spatial CV\n{run}  (open circle = Youden point)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"roc_{run}.png"), dpi=150)
    plt.close(fig)


def plot_pr(run, y, scores, mb, outdir):
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for name, (s, _) in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name}: AP {average_precision_score(y, s):.3f}")
    ax.plot(mb["recall"], mb["precision"], "k*", ms=13, label="MapBiomas (binary map)")
    ax.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"prevalence {y.mean():.2f}")
    ax.set_xlabel("Recall (plantation)")
    ax.set_ylabel("Precision (plantation)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-recall, out-of-fold\n{run}", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"pr_{run}.png"), dpi=150)
    plt.close(fig)


def plot_importance(run, perm_df, imp_imp, outdir):
    order = perm_df["perm_mean"].sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    axes[0].barh(order, perm_df.loc[order, "perm_mean"], xerr=perm_df.loc[order, "perm_sd"],
                 color="tab:blue", capsize=3)
    axes[0].set_xlabel("Drop in AUC when permuted (held-out folds, mean ± sd)")
    axes[0].set_title("Permutation importance (primary)", fontsize=10)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[1].barh(order, imp_imp.loc[order], color="lightgrey")
    axes[1].set_xlabel("Impurity importance (reference only; biased to continuous features)")
    axes[1].set_title("RF impurity importance (fit on all data)", fontsize=10)
    fig.suptitle(run, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"importance_{run}.png"), dpi=150)
    plt.close(fig)


def plot_corr(run, corr, outdir):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Pearson correlation, primary features\n{run}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"corr_{run}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- one run
def run_one(name, d, prim, cand, outdir, store):
    print("\n" + "=" * 100 + f"\nRUN {name}: n={len(d)}, primary={prim}\n" + "=" * 100)
    assert not (set(prim) | set(cand or [])) & set(META), "metadata used as feature!"
    y = d["plantation"].to_numpy(int)
    grp = d["group"].to_numpy()
    cl = d["CD_MUN"].to_numpy()
    mb_pred = (d["mb_pred"].to_numpy() == 2).astype(int)
    print(f"n={len(y)}  plantation={y.sum()}  forest={(y == 0).sum()}  municipalities={len(set(cl))}")
    print("groups:", dict(pd.Series(grp).value_counts()))

    # --- Task 2: correlations, VIF
    X = d[prim].to_numpy(float)
    corr = d[prim].corr()
    vif = vif_table(X, prim)
    print("\n[Task 2] Correlation matrix (primary):\n", corr.round(2))
    hi = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(prim) for b in prim[i + 1:]
          if abs(corr.loc[a, b]) > 0.8]
    print("Pairs with |r|>0.8:", hi if hi else "none")
    print("VIF:\n", vif.round(2))
    corr.to_csv(os.path.join(outdir, f"corr_{name}.csv"))
    vif.to_csv(os.path.join(outdir, f"vif_{name}.csv"))
    plot_corr(name, corr, outdir)
    print("\nGroup medians of primary features:\n", d.groupby("group")[prim].median().round(3))

    # --- Tasks 3-4: CV with primary features
    res = {}
    for mname, mk in MODELS.items():
        oof, fold, aucs, models, tests = cv_oof(mk, X, y, cl)
        res[mname] = dict(oof=oof, fold=fold, aucs=aucs, models=models, tests=tests)
    folds_tab = pd.DataFrame({"fold": range(N_SPLITS),
                              "n_test": [len(t) for t in res["LR"]["tests"]],
                              "n_plantation": [int(y[t].sum()) for t in res["LR"]["tests"]],
                              "n_missed_plant": [int((grp[t] == "missed plantation").sum())
                                                 for t in res["LR"]["tests"]],
                              "AUC_LR": res["LR"]["aucs"], "AUC_RF": res["RF"]["aucs"]})
    print("\n[Task 4] Per-fold sizes and AUC:\n", folds_tab.round(3))

    # --- Tasks 5-7: metrics at thresholds + baseline + key result
    metric_rows, key_rows = [], []

    def add(model, fset, op, thr, pred, aucs, pooled, ap):
        m = binary_metrics(y, pred)
        metric_rows.append(dict(run=name, feature_set=fset, model=model, op_point=op,
                                threshold=thr, n=len(y), n_pos=int(y.sum()),
                                auc_fold_mean=np.nan if aucs is None else aucs.mean(),
                                auc_fold_sd=np.nan if aucs is None else aucs.std(ddof=1),
                                auc_pooled=pooled, avg_precision=ap, **m))
        key_rows.append(dict(run=name, feature_set=fset, model=model, op_point=op,
                             threshold=thr, **group_flags(pred, grp)))

    for mname in MODELS:
        s = res[mname]["oof"]
        pooled, ap = roc_auc_score(y, s), average_precision_score(y, s)
        for op, t in op_points(y, s).items():
            add(mname, "primary", op, t, (s >= t).astype(int), res[mname]["aucs"], pooled, ap)
    add("MapBiomas", "baseline", "binary map", np.nan, mb_pred, None, np.nan, np.nan)
    mb = binary_metrics(y, mb_pred)

    plot_roc(name, y, {m: (res[m]["oof"], res[m]["aucs"]) for m in MODELS}, mb, outdir)
    plot_pr(name, y, {m: (res[m]["oof"], None) for m in MODELS}, mb, outdir)

    # --- Task 8: significance
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    lr_tab, engine = fit_logit(Xs, y, prim)
    print(f"\n[Task 8] Logit on all data, standardised features ({engine}); n={len(y)}")
    print("NOTE: p-values likely optimistic (spatial autocorrelation; unweighted fit).")
    print(lr_tab.round(4))
    lr_tab.to_csv(os.path.join(outdir, f"logit_{name}.csv"))

    perm_runs = []
    for k in range(N_SPLITS):
        te = res["RF"]["tests"][k]
        pi = permutation_importance(res["RF"]["models"][k], X[te], y[te], scoring="roc_auc",
                                    n_repeats=20, random_state=SEED, n_jobs=-1)
        perm_runs.append(pi.importances_mean)
    perm_runs = np.array(perm_runs)
    perm_df = pd.DataFrame({"perm_mean": perm_runs.mean(0), "perm_sd": perm_runs.std(0, ddof=1)},
                           index=prim)
    rf_full = make_rf().fit(X, y)
    imp_imp = pd.Series(rf_full.feature_importances_, index=prim)
    feat = lr_tab.drop("const").join(perm_df).join(imp_imp.rename("rf_impurity"))
    feat["rank_LR(|z|)"] = (feat["coef_per_SD"] / feat["se"]).abs().rank(ascending=False).astype(int)
    feat["rank_RF(perm)"] = feat["perm_mean"].rank(ascending=False).astype(int)
    print("\nPermutation importance (drop in AUC on held-out folds) + impurity (reference):")
    print(feat[["perm_mean", "perm_sd", "rf_impurity", "rank_LR(|z|)", "rank_RF(perm)"]].round(4)
          .sort_values("perm_mean", ascending=False))
    feat.to_csv(os.path.join(outdir, f"feature_table_{name}.csv"))
    plot_importance(name, perm_df, imp_imp, outdir)

    # --- Task 9: secondary run with all candidates
    sec = None
    if cand:
        Xa = d[cand].to_numpy(float)
        sec = {}
        for mname, mk in MODELS.items():
            oof, fold, aucs, _, _ = cv_oof(mk, Xa, y, cl)
            assert (fold == res[mname]["fold"]).all(), "folds differ between primary/secondary"
            sec[mname] = dict(oof=oof, aucs=aucs)
            pooled, ap = roc_auc_score(y, oof), average_precision_score(y, oof)
            for op, t in op_points(y, oof).items():
                add(mname, "all_candidates", op, t, (oof >= t).astype(int), aucs, pooled, ap)
        print(f"\n[Task 9] Secondary run, {len(cand)} features: {cand}")
        rows9 = []
        for mname in MODELS:
            dl = sec[mname]["aucs"] - res[mname]["aucs"]
            rows9.append(dict(model=mname,
                              AUC_primary=f"{res[mname]['aucs'].mean():.3f} ± {res[mname]['aucs'].std(ddof=1):.3f}",
                              AUC_all=f"{sec[mname]['aucs'].mean():.3f} ± {sec[mname]['aucs'].std(ddof=1):.3f}",
                              delta_mean=dl.mean(), delta_sd=dl.std(ddof=1),
                              folds_improved=f"{(dl > 0).sum()}/{N_SPLITS}"))
        print(pd.DataFrame(rows9).round(3))

    metrics = pd.DataFrame(metric_rows)
    key = pd.DataFrame(key_rows)
    show = ["feature_set", "model", "op_point", "threshold", "auc_fold_mean", "auc_fold_sd",
            "accuracy", "precision", "recall", "fpr", "fnr", "tp", "fp", "fn", "tn"]
    print("\n[Tasks 5-6] Metrics (OOF). Threshold chosen on the same OOF scores -> optimistic.")
    print(metrics[metrics.feature_set.isin(["primary", "baseline"])][show].round(3).to_string(index=False))
    kshow = ["feature_set", "model", "op_point", "missed_n", "missed_flagged", "missed_share",
             "missed_lo", "missed_hi", "realforest_n", "realforest_flagged", "realforest_share",
             "fcp_n", "fcp_flagged", "fcp_share"]
    print("\n[Task 7] 'missed plantation' caught vs forest wrongly flagged:")
    print(key[key.feature_set.isin(["primary", "baseline"])][kshow].round(3).to_string(index=False))

    # --- Save final models (fit on ALL rows of this run) + thresholds + metadata
    import json
    import joblib
    import sklearn
    mdir = os.path.join(outdir, "Models")
    os.makedirs(mdir, exist_ok=True)
    meta = {"run": name, "n_rows": int(len(y)), "n_plantation": int(y.sum()),
            "features_primary": prim, "features_all_candidates": cand,
            "label": "plantation (1) vs natural forest (0)", "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "note": ("Fit on all rows; performance must be judged from the spatial-CV "
                     "out-of-fold results, not from these fitted models. Thresholds were "
                     "chosen on the same OOF scores (optimistic)."),
            "thresholds_from_oof": {m: op_points(y, res[m]["oof"]) for m in MODELS}}
    for mname, mk in MODELS.items():
        joblib.dump(mk().fit(X, y), os.path.join(mdir, f"{name}_{mname}_primary.joblib"))
        if cand:
            joblib.dump(mk().fit(d[cand].to_numpy(float), y),
                        os.path.join(mdir, f"{name}_{mname}_allfeat.joblib"))
    with open(os.path.join(mdir, f"{name}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # OOF predictions
    oofdf = d[["pt_id", "CD_MUN", "lat", "lon", "group", "plantation", "mb_pred"]].copy()
    oofdf.insert(0, "run", name)
    oofdf["fold"] = res["LR"]["fold"]
    oofdf["lr_prob"] = res["LR"]["oof"]
    oofdf["rf_prob"] = res["RF"]["oof"]
    if sec:
        oofdf["lr_prob_allfeat"] = sec["LR"]["oof"]
        oofdf["rf_prob_allfeat"] = sec["RF"]["oof"]
    store["oof"].append(oofdf)
    store["metrics"].append(metrics)
    store["key"].append(key)
    store["aucs"][name] = {m: res[m]["aucs"] for m in MODELS}
    store["folds"][name] = res["LR"]["fold"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cropin_model_features_2022.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    df["ever_farm"] = (df["ever_farm_mean"] >= 0.5).astype(int)
    df["state"] = (df["CD_MUN"] // 100000).map({41: "PR", 42: "SC", 43: "RS"})

    # --- Task 1
    print("[Task 1] rows:", len(df))
    print("By plantation label:\n", df["plantation"].value_counts().rename({0: "forest(0)", 1: "plantation(1)"}))
    print("By group:\n", df["group"].value_counts())
    print("Group x label:\n", pd.crosstab(df["group"], df["plantation"]))
    miss = df.isna().sum()
    print("Missing values per column (only columns with any):\n", miss[miss > 0])
    df["vh_missing"] = df["vh_mean"].isna()
    print("\nVH missing by state:\n", pd.crosstab(df["state"], df["vh_missing"], margins=True))
    print("VH missing by group:\n", pd.crosstab(df["group"], df["vh_missing"], margins=True))

    prim_noVH = [f for f in PRIMARY if f != "vh_mean"]
    cand_noS1 = [f for f in CANDIDATES if f not in S1]
    rows_A = df.dropna(subset=sorted(set(PRIMARY) | set(CANDIDATES))).reset_index(drop=True)
    rows_B = df.dropna(subset=sorted(set(prim_noVH) | set(cand_noS1))).reset_index(drop=True)
    n_prim_only = len(df.dropna(subset=PRIMARY))
    print(f"\nDropping rows missing PRIMARY features: {len(df) - n_prim_only} dropped, {n_prim_only} kept")
    print(f"Run A rows (also need all candidates): {len(rows_A)} kept "
          f"(missed plantations kept: {(rows_A.group == 'missed plantation').sum()} of "
          f"{(df.group == 'missed plantation').sum()})")
    print(f"Run B rows (no S1 features needed): {len(rows_B)} kept "
          f"(missed plantations kept: {(rows_B.group == 'missed plantation').sum()})")

    store = dict(oof=[], metrics=[], key=[], aucs={}, folds={})
    run_one("A_withVH_spec", rows_A, PRIMARY, CANDIDATES, args.out, store)
    run_one("A_noVH_ablation", rows_A, prim_noVH, None, args.out, store)
    run_one("B_noVH_allrows", rows_B, prim_noVH, cand_noS1, args.out, store)

    # VH effect on identical rows and identical folds
    assert (store["folds"]["A_withVH_spec"] == store["folds"]["A_noVH_ablation"]).all()
    print("\n" + "=" * 100 + "\nEffect of adding VH, same rows, same folds (run A vs A_noVH)")
    for m in MODELS:
        dl = store["aucs"]["A_withVH_spec"][m] - store["aucs"]["A_noVH_ablation"][m]
        print(f"  {m}: AUC with VH {store['aucs']['A_withVH_spec'][m].mean():.3f} vs without "
              f"{store['aucs']['A_noVH_ablation'][m].mean():.3f}; paired delta "
              f"{dl.mean():+.3f} ± {dl.std(ddof=1):.3f}; improved in {(dl > 0).sum()}/{N_SPLITS} folds")

    # --- Extra diagnostics (not in the original task list)
    oof_all = pd.concat(store["oof"])
    oof_all = oof_all.merge(df[["pt_id", "ever_farm", "state"]], on="pt_id", how="left")
    print("\n" + "=" * 100 + "\nExtra diagnostics")
    hard_rows, farm_rows = [], []
    for rn, dr in oof_all.groupby("run"):
        mp = dr[dr.group == "missed plantation"]
        print(f"\n{rn}: missed plantations n={len(mp)} in {mp.CD_MUN.nunique()} municipalities; "
              f"by state {mp.state.value_counts().to_dict()}; by CV fold {mp.fold.value_counts().sort_index().to_dict()}")
        h = dr[dr.group.isin(["missed plantation", "real forest"])]
        yh, clh = h["plantation"].to_numpy(int), h["CD_MUN"].to_numpy()
        for mname, col in [("LR", "lr_prob"), ("RF", "rf_prob")]:
            sh = h[col].to_numpy()
            lo, hi = cluster_boot_auc(yh, sh, clh)
            hard_rows.append(dict(run=rn, model=mname, n_missed=int(yh.sum()),
                                  n_real_forest=int((yh == 0).sum()),
                                  auc_missed_vs_realforest=roc_auc_score(yh, sh),
                                  ci_low=lo, ci_high=hi))
        t = pd.crosstab(h["group"], h["ever_farm"]).reindex(columns=[0, 1], fill_value=0)
        farm_rows.append(dict(run=rn, missed_with_farm=int(t.loc["missed plantation", 1]),
                              missed_n=int(t.loc["missed plantation"].sum()),
                              realforest_with_farm=int(t.loc["real forest", 1]),
                              realforest_n=int(t.loc["real forest"].sum()),
                              fisher_p=stats.fisher_exact(t.values)[1]))
    hard = pd.DataFrame(hard_rows)
    farm = pd.DataFrame(farm_rows)
    print("\nAUC on the hard subset only (missed plantation vs real forest), 95% CI = bootstrap over municipalities:")
    print(hard.round(3).to_string(index=False))
    print("\never_farm (majority of circle was farmland 1985-2021) in missed plantations vs real forest:")
    print(farm.round(4).to_string(index=False))
    # Do the other features add anything beyond SAG alone? (same folds as the main runs)
    sag_rows = []
    for rn, dr in [("A_withVH_spec", rows_A), ("B_noVH_allrows", rows_B)]:
        ys, cls = dr["plantation"].to_numpy(int), dr["CD_MUN"].to_numpy()
        for mname, mk in MODELS.items():
            _, _, a1, _, _ = cv_oof(mk, dr[["sag_oct_nov_mean"]].to_numpy(float), ys, cls)
            full = store["aucs"][rn][mname]
            dl = full - a1
            sag_rows.append(dict(run=rn, model=mname,
                                 AUC_sag_only=f"{a1.mean():.3f} ± {a1.std(ddof=1):.3f}",
                                 AUC_primary=f"{full.mean():.3f} ± {full.std(ddof=1):.3f}",
                                 paired_gain=dl.mean(), paired_gain_sd=dl.std(ddof=1),
                                 folds_improved=f"{(dl > 0).sum()}/{N_SPLITS}"))
    sag = pd.DataFrame(sag_rows)
    print("\nSAG-only baseline (sag_oct_nov_mean alone) vs primary feature set, AUC across folds:")
    print(sag.round(3).to_string(index=False))
    sag.to_csv(os.path.join(args.out, "sag_only_baseline.csv"), index=False)
    hard.to_csv(os.path.join(args.out, "hard_subset_auc.csv"), index=False)
    farm.to_csv(os.path.join(args.out, "ever_farm_test.csv"), index=False)

    pd.concat(store["oof"]).to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)
    pd.concat(store["metrics"]).to_csv(os.path.join(args.out, "metrics_all.csv"), index=False)
    pd.concat(store["key"]).to_csv(os.path.join(args.out, "key_result_missed_plantation.csv"), index=False)
    print("\nSaved outputs to", args.out)


if __name__ == "__main__":
    main()
