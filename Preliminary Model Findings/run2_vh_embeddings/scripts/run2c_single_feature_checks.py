#!/usr/bin/env python3
"""
Run 2c: single-feature checks and "do the embeddings contain this feature?" (CODING: 1 = forest).

1) Each handcrafted feature ALONE: spatial-CV AUC (outcome = forest) and, for vh_mean, a univariate
   logit with cluster-robust SEs (OR per 1 SD; OR > 1 = more forest-like).
2) Ridge regression of each handcrafted feature on the 64 AlphaEarth embeddings, grouped CV by
   municipality: out-of-fold R2 (how much of the feature the embeddings can reproduce).

Usage: python run2c_single_feature_checks.py   (same folder as poc2_vh_embeddings.py)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import poc2_vh_embeddings as P

FEATS = "data/cropin_model_features_2022_updated.csv"
EMBS = "data/cropin_embeddings_2022.csv"
OUT = "results"


def univariate_logit_robust(x, y, cl):
    xs = (x - x.mean()) / x.std()
    Xc = np.column_stack([np.ones(len(y)), xs])
    beta, p, H, it, last = P.irls(Xc, y)
    bread = np.linalg.inv(H)
    codes, u = pd.factorize(cl)
    S = np.zeros((len(u), 2))
    np.add.at(S, codes, Xc * (y - p)[:, None])
    G, n, K = len(u), len(y), 2
    V = (G / (G - 1)) * ((n - 1) / (n - K)) * bread @ (S.T @ S) @ bread
    se = np.sqrt(np.diag(V))[1]
    z = stats.norm.ppf(0.975)
    return dict(odds_ratio_per_SD=float(np.exp(beta[1])), robust_or_low=float(np.exp(beta[1] - z * se)),
                robust_or_high=float(np.exp(beta[1] + z * se)),
                robust_p=float(2 * stats.norm.sf(abs(beta[1] / se))))


def main():
    f, e = pd.read_csv(FEATS), pd.read_csv(EMBS)
    d = f.merge(e[["pt_id"] + P.EMB], on="pt_id", how="inner", validate="1:1")
    d["ever_farm"] = (d["ever_farm_mean"] >= 0.5).astype(int)
    dm = d.dropna(subset=P.HAND + P.EMB).reset_index(drop=True)
    yf = 1 - dm["plantation"].to_numpy(int)          # 1 = forest
    cl = dm["CD_MUN"].to_numpy()
    cv = GroupKFold(P.N_SPLITS, shuffle=True, random_state=P.SEED)
    print(f"n={len(dm)}  forest={int(yf.sum())}  plantation={int((1 - yf).sum())}")

    rows = []
    for c in ["sag_oct_nov_mean", "sag_swing_mean", "vh_mean", "vv_mean", "height_stdDev",
              "ndvi_aug_oct_mean", "ever_farm"]:
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()),
                              dm[[c]].to_numpy(float), yf, groups=cl, cv=cv, method="predict_proba")[:, 1]
        r = dict(feature=c, alone_auc_spatial_cv=roc_auc_score(yf, p))
        r.update(univariate_logit_robust(dm[c].to_numpy(float), yf, cl))
        rows.append(r)
    S = pd.DataFrame(rows).sort_values("alone_auc_spatial_cv", ascending=False)
    S.to_csv(os.path.join(OUT, "single_feature_checks.csv"), index=False)
    print("\n[1] Each feature alone (outcome = forest; OR > 1 = more forest-like):")
    print(S.round(3).to_string(index=False))

    E = dm[P.EMB].to_numpy(float)
    rows = []
    for c in ["vh_mean", "vv_mean", "sag_oct_nov_mean", "sag_swing_mean", "ndvi_aug_oct_mean",
              "height_stdDev", "ever_farm_mean"]:
        pr = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
                               E, dm[c].to_numpy(float), groups=cl, cv=cv)
        rows.append(dict(feature=c, oof_r2_from_embeddings=r2_score(dm[c], pr)))
    R = pd.DataFrame(rows).sort_values("oof_r2_from_embeddings", ascending=False)
    R.to_csv(os.path.join(OUT, "embeddings_predict_features.csv"), index=False)
    print("\n[2] Can the 64 embeddings reproduce each feature? (ridge, held-out municipalities)")
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
