#!/usr/bin/env python3
"""
Run 2: updated radar (Sentinel-1 2021-2022, now ~100% coverage) + AlphaEarth embeddings.

Evaluation rule (as requested): FOREST is the positive class.
  - A plantation predicted as forest = false positive (FP) = Cropin's false deforestation alert.
  - Forest recall of a model is matched to MapBiomas's forest recall: inside every outer training
    fold the threshold is chosen on INNER out-of-fold scores so forest recall equals MapBiomas's
    forest recall on that training fold; it is then applied to the held-out fold.
  - MapBiomas "predicts forest" when mb_pred == 1 (mb_pred 0 = other, 2 = plantation count as not-forest).

CODING: 1 = forest everywhere (training labels, model scores = P(forest), logit outcome, output files).
Rules: group, mb_pred, mb_code, pt_id, system:index, lat, lon, CD_MUN are never features.
Spatial CV: 5-fold GroupKFold by CD_MUN. No hyperparameter tuning (except the L2 strength of the
regularized logistic regression, chosen by inner CV as requested).

Usage: python poc2_vh_embeddings.py --features <updated.csv> --embeddings <emb.csv> --out <dir> [--smoke]
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

SEED = 42
N_SPLITS = 5
INNER = 5
N_BOOT = 2000
N_REPEATS = 10          # permutation repeats
N_TREES = 500
HAND = ["height_stdDev", "vh_mean", "vv_mean", "ndvi_aug_oct_mean", "sag_oct_nov_mean",
        "sag_swing_mean", "ever_farm"]
EMB = [f"A{i:02d}" for i in range(64)]
FORBIDDEN = {"group", "mb_pred", "mb_code", "pt_id", "system:index", "lat", "lon", "CD_MUN",
             "plantation", ".geo"}
RADAR = ["vh_mean", "vv_mean"]


# =============================================================== small utilities
def wilson(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def vif_table(X, names):
    out = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        A = np.column_stack([np.ones(len(yj)), np.delete(X, j, axis=1)])
        beta = np.linalg.lstsq(A, yj, rcond=None)[0]
        r2 = 1 - (yj - A @ beta).var() / yj.var()
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return pd.Series(out, index=names, name="VIF")


def thr_for_recall(s, yf, target):
    """Largest threshold t such that forest recall of (s >= t) is >= target."""
    sf = np.sort(s[yf == 1])[::-1]
    k = int(np.ceil(target * len(sf) - 1e-9))
    k = max(1, min(k, len(sf)))
    return float(sf[k - 1])


def matched_fp(s, yf, target):
    """Ex-post matched evaluation on given scores: FP count at forest recall >= target."""
    t = thr_for_recall(s, yf, target)
    pf = s >= t
    return int((pf & (yf == 0)).sum()), float((pf & (yf == 1)).sum() / (yf == 1).sum()), t


# =============================================================== models
def make_rf():
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def make_lr_plain():
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10))


def make_lr_l2(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                              max_iter=5000))


def choose_C(Xtr, ytr, gtr, Cs):
    """L2 strength by inner grouped CV, maximizing mean AUC (transparent loop, no sklearn CV class)."""
    splits = list(GroupKFold(INNER, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr))
    best, best_auc = None, -1
    for C in Cs:
        aucs = []
        for a, b in splits:
            m = make_lr_l2(C).fit(Xtr[a], ytr[a])
            aucs.append(roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))
        if np.mean(aucs) > best_auc:
            best, best_auc = C, np.mean(aucs)
    return best


def run_fold(model, X, yf, g, mbf, tr, te, Cs):
    """Fit on outer-train (label 1 = forest), choose threshold on inner OOF scores, score the held-out fold."""
    Xtr, ytr, gtr = X[tr], yf[tr], g[tr]
    yf_tr = ytr
    extra = {}
    if model == "RF":
        mk = make_rf
    elif model == "LR-plain":
        mk = make_lr_plain
    else:
        C = choose_C(Xtr, ytr, gtr, Cs)
        extra["C"] = C
        mk = lambda: make_lr_l2(C)
    oof_in = np.zeros(len(tr))
    for a, b in GroupKFold(INNER, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr):
        oof_in[b] = mk().fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
    target = float(mbf[tr][yf_tr == 1].mean())          # MapBiomas forest recall on training fold
    t = thr_for_recall(oof_in, yf_tr, target)
    m = mk().fit(Xtr, ytr)
    s_te = m.predict_proba(X[te])[:, 1]
    return s_te, t, target, m, extra


# =============================================================== logistic regression inference
def irls(X, y, tol=1e-10, maxit=100):
    beta = np.zeros(X.shape[1])
    last = np.inf
    for it in range(1, maxit + 1):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        last = np.abs(step).max()
        if last < tol:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    W = p * (1 - p)
    H = X.T @ (X * W[:, None])
    return beta, p, H, it, last


def logit_inference(Xs, y, cl, names):
    n = len(y)
    Xc = np.column_stack([np.ones(n), Xs])
    beta, p, H, iters, last = irls(Xc, y)
    bread = np.linalg.inv(H)
    se_w = np.sqrt(np.diag(bread))
    # cluster-robust sandwich (CR1 small-sample factor G/(G-1) * (N-1)/(N-K))
    codes, uniq = pd.factorize(cl)
    G, K = len(uniq), Xc.shape[1]
    scores = Xc * (y - p)[:, None]
    S = np.zeros((G, K))
    np.add.at(S, codes, scores)
    meat = S.T @ S
    c = G / (G - 1) * (n - 1) / (n - K)
    V = c * bread @ meat @ bread
    se_r = np.sqrt(np.diag(V))
    z = stats.norm.ppf(0.975)
    pv_w = 2 * stats.norm.sf(np.abs(beta / se_w))
    pv_r = 2 * stats.norm.sf(np.abs(beta / se_r))
    tab = pd.DataFrame({
        "coef_per_SD": beta, "odds_ratio_per_SD": np.exp(beta),
        "wald_se": se_w, "wald_p": pv_w,
        "wald_OR_low": np.exp(beta - z * se_w), "wald_OR_high": np.exp(beta + z * se_w),
        "robust_se": se_r, "robust_p": pv_r,
        "robust_OR_low": np.exp(beta - z * se_r), "robust_OR_high": np.exp(beta + z * se_r),
        "se_ratio_robust/wald": se_r / se_w,
    }, index=["const"] + list(names))
    # compare to sklearn (unpenalised, unweighted)
    sk = LogisticRegression(C=np.inf, max_iter=100000, tol=1e-12).fit(Xs, y)
    diff = np.abs(np.r_[sk.intercept_, sk.coef_.ravel()] - beta).max()
    info = dict(iterations=iters, final_max_coef_change=float(last), converged=bool(last < 1e-10),
                max_abs_diff_vs_sklearn=float(diff), n=int(n), n_clusters=int(G),
                robust_factor=float(c), max_wald_se=float(se_w.max()),
                max_abs_coef=float(np.abs(beta).max()),
                possible_separation=bool((se_w > 5).any() or (np.abs(beta) > 10).any()))
    return tab, info


# =============================================================== grouped permutation importance
def handcrafted_clusters(df, cols, thr=0.7):
    """Complete-linkage clusters so every pair inside a cluster has |Spearman r| > thr."""
    rho = df[cols].corr(method="spearman").abs().to_numpy()
    if len(cols) == 1:
        return [[cols[0]]], pd.DataFrame(rho, index=cols, columns=cols)
    dist = 1 - rho
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="complete")
    lab = fcluster(Z, t=1 - thr, criterion="distance")
    groups = [[c for c, l in zip(cols, lab) if l == k] for k in sorted(set(lab))]
    return groups, pd.DataFrame(rho, index=cols, columns=cols)


def grouped_permutation(fold_models, tests, X, cols, yf, groups, mbf_target, s_base_pool,
                        rng_seed):
    """Permute each group of columns jointly inside each held-out fold."""
    rng = np.random.default_rng(rng_seed)
    colpos = {c: i for i, c in enumerate(cols)}
    n = len(yf)
    rows = []
    fp_base, _, _ = matched_fp(s_base_pool, yf, mbf_target)
    base_auc = [roc_auc_score(yf[te], m.predict_proba(X[te])[:, 1])
                for m, te in zip(fold_models, tests)]
    for gname, gcols in groups.items():
        pos = [colpos[c] for c in gcols]
        fold_auc_drop = np.zeros((N_SPLITS, N_REPEATS))
        pooled_fp = np.zeros(N_REPEATS)
        for r in range(N_REPEATS):
            s_pool = np.zeros(n)
            for k, (m, te) in enumerate(zip(fold_models, tests)):
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, pos] = Xp[perm][:, pos]                 # same row permutation for the whole group
                s_perm = m.predict_proba(Xp)[:, 1]
                s_pool[te] = s_perm
                fold_auc_drop[k, r] = base_auc[k] - roc_auc_score(yf[te], s_perm)
            pooled_fp[r] = matched_fp(s_pool, yf, mbf_target)[0]
        per_fold = fold_auc_drop.mean(axis=1)
        rows.append(dict(group=gname, n_features=len(gcols),
                         auc_drop_mean=per_fold.mean(), auc_drop_sd=per_fold.std(ddof=1),
                         fp_matched_permuted=pooled_fp.mean(),
                         delta_fp_reduction=fp_base - pooled_fp.mean(),
                         delta_fp_reduction_sd=pooled_fp.std(ddof=1)))
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


# =============================================================== bootstrap
def boot_indices(cl, n_boot):
    codes, uniq = pd.factorize(cl)
    idx_by = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    return [np.concatenate([idx_by[p] for p in row]) for row in picks]


def ci(x):
    return tuple(np.percentile(x, [2.5, 97.5]))


# =============================================================== plots
def plot_curves(res, mb, outdir, ref_keys):
    colors = {"RF|A": "tab:blue", "RF|B": "tab:orange", "RF|C": "tab:green",
              "LR-L2|B": "tab:red", "LR-L2|C": "tab:purple", "LR-plain|A": "tab:brown",
              "RF|A0": "grey"}
    styles = {"RF|A0": "--", "LR-L2|B": "-.", "LR-L2|C": "-.", "LR-plain|A": ":"}
    show = [k for k in colors if k in res]
    # ROC (forest positive)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for k in show:
        fpr, tpr, _ = roc_curve(res[k]["yf"], res[k]["s"])
        ax.plot(fpr, tpr, color=colors[k], ls=styles.get(k, "-"),
                label=f"{k} (AUC {res[k]['auc_pooled']:.3f})")
        ax.plot(res[k]["fp_rate"], res[k]["recall"], "o", color=colors[k], mfc="white", ms=6)
    ax.plot(mb["fp_rate"], mb["recall"], "k*", ms=14, label="MapBiomas")
    ax.plot([0, 1], [0, 1], color="lightgrey", lw=1)
    ax.set_xlabel("FP rate: plantations predicted as forest / all plantations")
    ax.set_ylabel("Forest recall")
    ax.set_title("ROC, forest = positive (out-of-fold, spatial CV)\nopen circle = fold-threshold result", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_run2.png"), dpi=150)
    plt.close(fig)
    # FP rate vs forest recall (zoom on recall >= 0.7)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for k in show:
        fpr, tpr, _ = roc_curve(res[k]["yf"], res[k]["s"])
        keep = tpr >= 0.7
        ax.plot(tpr[keep], fpr[keep], color=colors[k], ls=styles.get(k, "-"), label=k)
        ax.plot(res[k]["recall"], res[k]["fp_rate"], "o", color=colors[k], mfc="white", ms=6)
    ax.plot(mb["recall"], mb["fp_rate"], "k*", ms=14, label="MapBiomas")
    ax.axvline(mb["recall"], color="k", lw=0.8, ls=":")
    ax.set_xlabel("Forest recall")
    ax.set_ylabel("FP rate (plantation predicted as forest)")
    ax.set_title("FP rate vs forest recall (lower is better)\nopen circle = fold-threshold result; dotted line = MapBiomas recall", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fp_vs_recall_run2.png"), dpi=150)
    plt.close(fig)


def plot_importance(imps, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for j, (name, df) in enumerate(imps.items()):
        d = df.sort_values("auc_drop_mean")
        lab = [g if len(g) < 38 else g[:35] + "..." for g in d["group"]]
        axes[0, j].barh(lab, d["auc_drop_mean"], xerr=d["auc_drop_sd"], color="tab:blue", capsize=2)
        axes[0, j].set_title(f"RF on {name}: AUC drop when group permuted", fontsize=10)
        axes[0, j].axvline(0, color="k", lw=0.8)
        axes[1, j].barh(lab, d["delta_fp_reduction"], xerr=d["delta_fp_reduction_sd"], color="tab:red", capsize=2)
        axes[1, j].set_title(f"RF on {name}: change in FP reduction at matched recall (FP count)", fontsize=10)
        axes[1, j].axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "importance_run2.png"), dpi=150)
    plt.close(fig)


# =============================================================== main
def main():
    global N_BOOT, N_REPEATS, N_TREES
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/cropin_model_features_2022_updated.csv")
    ap.add_argument("--embeddings", default="data/cropin_embeddings_2022.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true", help="fast test run (few trees/bootstraps)")
    args = ap.parse_args()
    if args.smoke:
        N_BOOT, N_REPEATS, N_TREES = 60, 2, 40
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------ merge + missingness
    f = pd.read_csv(args.features)
    e = pd.read_csv(args.embeddings)
    print(f"[merge] features rows={len(f)}  embeddings rows={len(e)}")
    print(f"[merge] pt_id unique: features={f.pt_id.is_unique}, embeddings={e.pt_id.is_unique}")
    sf, se = set(f.pt_id), set(e.pt_id)
    print(f"[merge] matched={len(sf & se)}  only-in-features={len(sf - se)}  only-in-embeddings={len(se - sf)}")
    d = f.merge(e[["pt_id"] + EMB], on="pt_id", how="inner", validate="1:1")
    chk = f.merge(e[["pt_id", "CD_MUN", "plantation"]], on="pt_id", suffixes=("", "_e"))
    print(f"[merge] label agreement={np.mean(chk.plantation == chk.plantation_e):.4f}  "
          f"CD_MUN agreement={np.mean(chk.CD_MUN == chk.CD_MUN_e):.4f}")
    d["ever_farm"] = (d["ever_farm_mean"] >= 0.5).astype(int)
    miss = pd.DataFrame({"n_missing": d.isna().sum(), "pct_missing": d.isna().mean() * 100})
    miss.to_csv(os.path.join(args.out, "missing_share_by_column.csv"))
    print("[missing] columns with any missing values:\n", miss[miss.n_missing > 0].round(3))
    print(f"[missing] VH: vh_mean missing={d.vh_mean.isna().sum()} ({d.vh_mean.isna().mean()*100:.1f}%), "
          f"vv_mean missing={d.vv_mean.isna().sum()}, embeddings missing={int(d[EMB].isna().sum().sum())}")

    need = HAND + EMB
    assert not set(need) & FORBIDDEN
    dm = d.dropna(subset=need).reset_index(drop=True)
    print(f"[rows] dropped {len(d) - len(dm)} row(s) missing a model input -> {len(dm)} rows used for ALL models "
          f"(common row set so results are comparable)")

    yp = dm["plantation"].to_numpy(int)
    yf = 1 - yp
    grp = dm["group"].to_numpy()
    cl = dm["CD_MUN"].to_numpy()
    mbf = (dm["mb_pred"].to_numpy() == 1).astype(int)       # MapBiomas predicts forest
    is_missed = (yp == 1) & (mbf == 1)
    assert (is_missed == (grp == "missed plantation")).all()
    is_caught = grp == "caught plantation"
    is_other_pl = (grp == "other") & (yp == 1)
    n_pl, n_fo, n_missed = int(yp.sum()), int(yf.sum()), int(is_missed.sum())
    print(f"[rows] n={len(dm)}  plantation={n_pl}  forest={n_fo}  missed plantations (MapBiomas says forest)={n_missed} "
          f"in {len(set(cl[is_missed]))} municipalities; municipalities={len(set(cl))}")
    print("[rows] groups:", dict(pd.Series(grp).value_counts()))

    mb = dict(recall=float((mbf & yf).sum() / n_fo), fp=int((mbf & yp).sum()))
    mb["fp_rate"] = mb["fp"] / n_pl
    lo, hi = wilson(mb["fp"], n_pl)
    print(f"[MapBiomas] forest recall={mb['recall']:.4f} ({(mbf & yf).sum()}/{n_fo}); FP (plantation called forest)="
          f"{mb['fp']} -> FP rate {mb['fp_rate']:.4f} (Wilson CI {lo:.3f}-{hi:.3f})")

    # ------------------------------------------------ VIF on handcrafted set A
    cols_A = list(HAND)
    while True:
        vif = vif_table(dm[cols_A].to_numpy(float), cols_A)
        if vif.max() <= 5 or len(cols_A) <= 2:
            break
        drop = vif.idxmax()
        print(f"[VIF] dropping {drop} (VIF {vif.max():.2f} > 5)")
        cols_A.remove(drop)
    vif0 = vif_table(dm[HAND].to_numpy(float), HAND)
    vif0.to_csv(os.path.join(args.out, "vif_A.csv"))
    print("[VIF] initial VIFs (all 7 handcrafted features):\n", vif0.round(2).to_string())
    print("[VIF] VIFs of the kept set:\n", vif.round(2).to_string())
    print(f"[VIF] features kept in set A: {cols_A}  (dropped: {[c for c in HAND if c not in cols_A] or 'none'})")

    cols_A0 = [c for c in cols_A if c not in RADAR]
    cols_Avh = cols_A0 + (["vh_mean"] if "vh_mean" in cols_A else [])
    SPECS = {
        "RF|A": ("RF", cols_A), "RF|B": ("RF", EMB), "RF|C": ("RF", cols_A + EMB),
        "LR-L2|B": ("LR-L2", EMB), "LR-L2|C": ("LR-L2", cols_A + EMB),
        "LR-plain|A": ("LR-plain", cols_A),
        "RF|A0": ("RF", cols_A0), "RF|A_vh": ("RF", cols_Avh), "LR-plain|A0": ("LR-plain", cols_A0),
    }
    REF = {"RF|A0", "RF|A_vh", "LR-plain|A0", "LR-plain|A"}
    print("[sets] A0 = A without radar (the previous run's feature idea, same rows); A_vh = A0 + vh_mean "
          "(identical to A when vv_mean is dropped: kept only as a random-forest noise check)")

    # ------------------------------------------------ cross-validation
    outer = list(GroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(dm, yp, cl))
    fold_id = np.zeros(len(dm), int)
    for k, (_, te) in enumerate(outer):
        fold_id[te] = k
    print("[CV] outer fold sizes:", [len(te) for _, te in outer],
          " missed plantations per fold:", [int(is_missed[te].sum()) for _, te in outer])
    Cs = np.logspace(-4, 1, 11)
    res, fold_models, fold_rows = {}, {}, []
    for key, (model, cols) in SPECS.items():
        X = dm[cols].to_numpy(float)
        s = np.zeros(len(dm))
        thr_row = np.zeros(len(dm))
        tgt_row = np.zeros(len(dm))
        models, Cchosen = [], []
        tk = time.time()
        for k, (tr, te) in enumerate(outer):
            s_te, t, target, m, extra = run_fold(model, X, yf, cl, mbf, tr, te, Cs)
            s[te], thr_row[te], tgt_row[te] = s_te, t, target
            models.append(m)
            fold_rows.append(dict(spec=key, fold=k, threshold=t, mb_forest_recall_train=target,
                                  **{k2: v for k2, v in extra.items()}))
            if "C" in extra:
                Cchosen.append(extra["C"])
        pf = (s >= thr_row).astype(int)
        aucs = np.array([roc_auc_score(yf[te], s[te]) for _, te in outer])
        res[key] = dict(model=model, cols=cols, s=s, thr_row=thr_row, pf=pf, yf=yf, aucs=aucs,
                        auc_pooled=roc_auc_score(yf, s), models=models)
        r = res[key]
        r["recall"] = float((pf & yf).sum() / n_fo)
        r["fp"] = int((pf & yp).sum())
        r["fp_rate"] = r["fp"] / n_pl
        print(f"[CV] {key:<12} done in {time.time()-tk:5.0f}s  recall={r['recall']:.3f} FP={r['fp']} "
              f"AUC={r['auc_pooled']:.3f}" + (f"  C per fold={['%.3g' % c for c in Cchosen]}" if Cchosen else ""))
        if key not in ("RF|A", "RF|C"):
            r["models"] = None          # free memory; keep only what the importance step needs
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out, "fold_thresholds.csv"), index=False)

    # ------------------------------------------------ bootstrap (shared resamples across models)
    boots = boot_indices(cl, N_BOOT)
    yp_b, yf_b = yp.astype(bool), yf.astype(bool)
    mb_fp_flag = (mbf == 1) & yp_b
    mb_rec_flag = (mbf == 1) & yf_b
    bs = {}
    for key, r in res.items():
        pf_b = r["pf"].astype(bool)
        fp_flag, rec_flag = pf_b & yp_b, pf_b & yf_b
        fp_fixed, rec_fixed, mbfp_b, red_b = [], [], [], []
        fp_match, mbfp_match = [], []
        for idx in boots:
            nf = yf[idx].sum()
            fp_fixed.append(fp_flag[idx].sum())
            rec_fixed.append(rec_flag[idx].sum() / nf)
            mbfp_b.append(mb_fp_flag[idx].sum())
            target = mb_rec_flag[idx].sum() / nf
            fp_match.append(matched_fp(r["s"][idx], yf[idx], target)[0])
        bs[key] = dict(fp_fixed=np.array(fp_fixed), rec_fixed=np.array(rec_fixed),
                       fp_match=np.array(fp_match))
    mbfp_b = np.array([mb_fp_flag[idx].sum() for idx in boots])

    # ------------------------------------------------ metrics table
    rows = []
    for key, r in res.items():
        red = mb["fp"] - r["fp"]
        red_b = mbfp_b - bs[key]["fp_fixed"]
        with np.errstate(divide="ignore", invalid="ignore"):
            redpct_b = np.where(mbfp_b > 0, red_b / mbfp_b, np.nan)
        pf = r["pf"].astype(bool)
        fp_missed = int((pf & is_missed).sum())
        fp_new_caught = int((pf & is_caught).sum())
        fp_new_other = int((pf & is_other_pl).sum())
        fpm, recm, _ = matched_fp(r["s"], yf, mb["recall"])
        red_m_b = mbfp_b - bs[key]["fp_match"]
        rows.append(dict(
            spec=key, model=r["model"], n_features=len(r["cols"]),
            forest_recall=r["recall"], mb_forest_recall=mb["recall"],
            recall_minus_mb=r["recall"] - mb["recall"],
            fp=r["fp"], fp_rate=r["fp_rate"], mb_fp=mb["fp"], mb_fp_rate=mb["fp_rate"],
            fp_reduction=red, fp_reduction_pct=100 * red / mb["fp"],
            fp_red_ci_low=ci(red_b)[0], fp_red_ci_high=ci(red_b)[1],
            fp_red_pct_ci_low=100 * np.nanpercentile(redpct_b, 2.5),
            fp_red_pct_ci_high=100 * np.nanpercentile(redpct_b, 97.5),
            missed_plantations_n=n_missed, missed_rejected=n_missed - fp_missed,
            missed_still_forest=fp_missed, new_fp_from_caught_plantations=fp_new_caught,
            new_fp_from_other_plantations=fp_new_other,
            auc_pooled=r["auc_pooled"], auc_fold_mean=r["aucs"].mean(), auc_fold_sd=r["aucs"].std(ddof=1),
            exante_matched_fp=fpm, exante_matched_reduction=mb["fp"] - fpm,
            exante_matched_red_ci_low=ci(red_m_b)[0], exante_matched_red_ci_high=ci(red_m_b)[1],
            reference_only=key in REF))
    M = pd.DataFrame(rows)
    M.rename(columns={"exante_matched_fp": "expost_matched_fp", "exante_matched_reduction": "expost_matched_reduction",
                      "exante_matched_red_ci_low": "expost_matched_red_ci_low",
                      "exante_matched_red_ci_high": "expost_matched_red_ci_high"}, inplace=True)
    M.to_csv(os.path.join(args.out, "metrics_run2.csv"), index=False)
    show = ["spec", "forest_recall", "fp", "fp_rate", "fp_reduction", "fp_reduction_pct", "fp_red_pct_ci_low",
            "fp_red_pct_ci_high", "missed_rejected", "new_fp_from_caught_plantations",
            "new_fp_from_other_plantations", "auc_pooled", "auc_fold_mean", "auc_fold_sd"]
    print(f"\n[METRICS] MapBiomas forest recall={mb['recall']:.3f}, FP={mb['fp']} ({mb['fp_rate']:.3%}); "
          f"missed plantations n={n_missed}")
    print(M[show].round(3).to_string(index=False))
    print("\n[METRICS] ex-post exactly matched recall (threshold picked on the pooled held-out scores; "
          "a sanity check, optimistic):")
    print(M[["spec", "expost_matched_fp", "expost_matched_reduction", "expost_matched_red_ci_low",
             "expost_matched_red_ci_high"]].round(2).to_string(index=False))

    # ------------------------------------------------ paired comparisons
    pairs = [("RF|B", "RF|A", "embeddings only vs handcrafted (RF)"),
             ("RF|C", "RF|B", "A+embeddings vs embeddings (RF)"),
             ("RF|C", "RF|A", "A+embeddings vs handcrafted (RF)"),
             ("RF|A", "RF|A0", "adding VH to A0 (RF); vv_mean was dropped by the VIF rule"),
             ("RF|A_vh", "RF|A0", "adding VH to A0 (RF), same features as A with columns reordered"),
             ("RF|A_vh", "RF|A", "RF NOISE CHECK: identical features, only column order differs"),
             ("LR-plain|A", "LR-plain|A0", "adding VH to A0 (plain LR)"),
             ("LR-L2|C", "LR-L2|B", "A+embeddings vs embeddings (L2 LR)")]
    prow = []
    for a, b, desc in pairs:
        dfp = bs[a]["fp_fixed"] - bs[b]["fp_fixed"]
        dmt = bs[a]["fp_match"] - bs[b]["fp_match"]
        drec = bs[a]["rec_fixed"] - bs[b]["rec_fixed"]
        prow.append(dict(comparison=desc, first=a, second=b,
                         fp_diff=res[a]["fp"] - res[b]["fp"], fp_diff_ci_low=ci(dfp)[0], fp_diff_ci_high=ci(dfp)[1],
                         recall_diff=res[a]["recall"] - res[b]["recall"],
                         recall_diff_ci_low=ci(drec)[0], recall_diff_ci_high=ci(drec)[1],
                         expost_matched_fp_diff=matched_fp(res[a]["s"], yf, mb["recall"])[0]
                         - matched_fp(res[b]["s"], yf, mb["recall"])[0],
                         expost_diff_ci_low=ci(dmt)[0], expost_diff_ci_high=ci(dmt)[1]))
    P = pd.DataFrame(prow)
    P.to_csv(os.path.join(args.out, "paired_comparisons.csv"), index=False)
    print("\n[PAIRED] FP count difference (first - second; NEGATIVE = first has fewer false positives), "
          "95% cluster-bootstrap CI:")
    print(P.round(3).to_string(index=False))

    # ------------------------------------------------ plain LR significance on A
    XA = dm[cols_A].to_numpy(float)
    XAs = StandardScaler().fit_transform(XA)
    tab, info = logit_inference(XAs, yf, cl, cols_A)
    tab.to_csv(os.path.join(args.out, "coef_table_A_wald_vs_robust.csv"))
    print(f"\n[LOGIT] plain LR on A, outcome = FOREST (1 = forest; OR>1: more forest-like), standardized features")
    print("[LOGIT] diagnostics:", json.dumps(info))
    print(tab.round(4).to_string())
    with open(os.path.join(args.out, "logit_diagnostics.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    # ------------------------------------------------ grouped permutation importance
    hg, rho = handcrafted_clusters(dm, cols_A, 0.7)
    rho.round(2).to_csv(os.path.join(args.out, "spearman_abs_A.csv"))
    print("\n[IMPORTANCE] handcrafted clusters (|Spearman r| > 0.7, complete linkage):", hg)
    hand_groups = {"+".join(g): g for g in hg}
    imps = {}
    for set_name, key in [("A", "RF|A"), ("C", "RF|C")]:
        cols = res[key]["cols"]
        groups = dict(hand_groups)
        if set_name == "C":
            groups = {"embeddings (A00-A63, 64)": EMB, **groups}
        tk = time.time()
        imp = grouped_permutation(res[key]["models"], [te for _, te in outer], dm[cols].to_numpy(float), cols,
                                  yf, groups, mb["recall"], res[key]["s"], SEED + 1)
        imp.insert(0, "feature_set", set_name)
        imp["baseline_fp_matched"] = matched_fp(res[key]["s"], yf, mb["recall"])[0]
        imps[set_name] = imp
        print(f"\n[IMPORTANCE] RF on {set_name} ({time.time()-tk:.0f}s). AUC drop = mean±sd over folds; "
              f"delta_fp_reduction = FP reduction after permuting minus before (ex-post matched recall; "
              f"negative = the group was helping)")
        print(imp.round(4).to_string(index=False))
    pd.concat(imps.values()).to_csv(os.path.join(args.out, "grouped_importance.csv"), index=False)
    plot_importance({"A": imps["A"], "C": imps["C"]}, args.out)

    # ------------------------------------------------ plots + OOF predictions
    plot_curves(res, mb, args.out, REF)
    oof = dm[["pt_id", "CD_MUN", "group", "mb_pred"]].copy()
    oof["forest"] = yf                                      # 1 = forest
    oof["fold"] = fold_id
    oof["mb_pred_forest"] = mbf
    for key, r in res.items():
        tag = key.replace("|", "_").replace("-", "")
        oof[f"forest_score__{tag}"] = r["s"]
        oof[f"threshold__{tag}"] = r["thr_row"]
        oof[f"pred_forest__{tag}"] = r["pf"]
    oof.to_csv(os.path.join(args.out, "oof_predictions_run2.csv"), index=False)
    # ------------------------------------------------ exploratory: by state (NOT recall-matched within state)
    st = (dm["CD_MUN"] // 100000).map({41: "PR", 42: "SC", 43: "RS"}).to_numpy()
    srows = []
    for s_ in ["PR", "SC", "RS"]:
        m_ = st == s_
        for name, pf_ in [("MapBiomas", mbf)] + [(k, res[k]["pf"]) for k in
                                                  ["RF|A", "RF|B", "RF|C", "LR-L2|B", "LR-L2|C"]]:
            pf_ = np.asarray(pf_).astype(bool)
            srows.append(dict(state=s_, model=name, n=int(m_.sum()), n_plantation=int(yp[m_].sum()),
                              n_missed=int(is_missed[m_].sum()),
                              forest_recall=float((pf_ & yf_b)[m_].sum() / yf[m_].sum()),
                              fp=int((pf_ & yp_b)[m_].sum()),
                              fp_rate=float((pf_ & yp_b)[m_].sum() / yp[m_].sum())))
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(args.out, "metrics_by_state.csv"), index=False)
    print("\n[BY STATE] exploratory; thresholds were matched on pooled data, so recall differs by state:")
    print(S.pivot(index="model", columns="state", values="fp").loc[["MapBiomas", "RF|A", "RF|B", "RF|C", "LR-L2|B", "LR-L2|C"]].to_string())
    print(S.pivot(index="model", columns="state", values="forest_recall").round(3).loc[["MapBiomas", "RF|A", "RF|B", "RF|C", "LR-L2|B", "LR-L2|C"]].to_string())
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Outputs in {args.out}")


if __name__ == "__main__":
    main()
