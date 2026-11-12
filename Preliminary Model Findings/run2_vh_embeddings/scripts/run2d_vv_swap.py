#!/usr/bin/env python3
"""
Run 2d: keep vv_mean instead of vh_mean (the VIF rule had dropped vv_mean although vv_mean alone is the
stronger single feature). CODING: 1 = forest. Same folds, same evaluation rule, same municipality-bootstrap
resamples as run 2, so every difference below is paired.

New models: RF and plain LR on A_vv (A with vh_mean -> vv_mean); RF and L2-LR on C_vv (A_vv + 64 embeddings).
Reference models are read from run 2's saved out-of-fold file (not refit).

Usage: python run2d_vv_swap.py   (same folder as poc2_vh_embeddings.py)
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import poc2_vh_embeddings as P

FEATS = "data/cropin_model_features_2022_updated.csv"
EMBS = "data/cropin_embeddings_2022.csv"
OUT = "results"
A_VH = ["height_stdDev", "vh_mean", "ndvi_aug_oct_mean", "sag_oct_nov_mean", "sag_swing_mean", "ever_farm"]
A_VV = ["vv_mean" if c == "vh_mean" else c for c in A_VH]


def main():
    t0 = time.time()
    f, e = pd.read_csv(FEATS), pd.read_csv(EMBS)
    d = f.merge(e[["pt_id"] + P.EMB], on="pt_id", how="inner", validate="1:1")
    d["ever_farm"] = (d["ever_farm_mean"] >= 0.5).astype(int)
    dm = d.dropna(subset=P.HAND + P.EMB).reset_index(drop=True)
    oof = pd.read_csv(os.path.join(OUT, "oof_predictions_run2.csv"), float_precision="round_trip")
    assert (oof["pt_id"].to_numpy() == dm["pt_id"].to_numpy()).all()

    yp = dm["plantation"].to_numpy(int)
    yf = 1 - yp
    cl = dm["CD_MUN"].to_numpy()
    grp = dm["group"].to_numpy()
    mbf = (dm["mb_pred"].to_numpy() == 1).astype(int)
    n_pl, n_fo = int(yp.sum()), int(yf.sum())
    is_missed, is_caught = grp == "missed plantation", grp == "caught plantation"
    is_other = (grp == "other") & (yp == 1)
    n_missed = int(is_missed.sum())
    yp_b, yf_b = yp.astype(bool), yf.astype(bool)
    mb_fp_flag, mb_rec_flag = (mbf == 1) & yp_b, (mbf == 1) & yf_b
    mb_fp, mb_rec = int(mb_fp_flag.sum()), float(mb_rec_flag.sum() / n_fo)
    outer = list(GroupKFold(P.N_SPLITS, shuffle=True, random_state=P.SEED).split(dm, yp, cl))
    fold_id = np.zeros(len(dm), int)
    for k, (_, te) in enumerate(outer):
        fold_id[te] = k
    assert (fold_id == oof["fold"].to_numpy()).all()
    tests = [te for _, te in outer]
    print(f"n={len(dm)}  missed plantations={n_missed}  MapBiomas: forest recall={mb_rec:.4f}, FP={mb_fp}")

    # ---- VIF / correlation of the swapped set
    vif = P.vif_table(dm[A_VV].to_numpy(float), A_VV)
    print("[VIF] set A_vv (vv_mean instead of vh_mean):\n", vif.round(2).to_string())
    hg, rho = P.handcrafted_clusters(dm, A_VV, 0.7)
    print("[clusters |Spearman|>0.7] A_vv:", hg)
    print("[corr] |Spearman| vv_mean with the other handcrafted features:",
          rho.loc["vv_mean"].drop("vv_mean").round(2).to_dict())

    # ---- reference models (run 2) + new fits
    res = {}
    for key, tag in [("RF|A0", "RF_A0"), ("RF|A", "RF_A"), ("RF|B", "RF_B"), ("RF|C", "RF_C"),
                     ("LR-plain|A0", "LRplain_A0"), ("LR-plain|A", "LRplain_A"),
                     ("LR-L2|B", "LRL2_B"), ("LR-L2|C", "LRL2_C")]:
        res[key] = dict(s=oof[f"forest_score__{tag}"].to_numpy(), pf=oof[f"pred_forest__{tag}"].to_numpy(int), new=False)
    new = {"RF|A_vv": ("RF", A_VV), "LR-plain|A_vv": ("LR-plain", A_VV),
           "RF|C_vv": ("RF", A_VV + P.EMB), "LR-L2|C_vv": ("LR-L2", A_VV + P.EMB)}
    Cs = np.logspace(-4, 1, 11)
    for key, (model, cols) in new.items():
        tk = time.time()
        X = dm[cols].to_numpy(float)
        s, thr, models = np.zeros(len(dm)), np.zeros(len(dm)), []
        for k, (tr, te) in enumerate(outer):
            s_te, t, _, m, _ = P.run_fold(model, X, yf, cl, mbf, tr, te, Cs)
            s[te], thr[te] = s_te, t
            models.append(m)
        res[key] = dict(s=s, pf=(s >= thr).astype(int), new=True, models=models, cols=cols, X=X)
        print(f"[fit] {key:<14} {time.time()-tk:4.0f}s  n_features={len(cols)}")

    # ---- metrics + paired bootstrap (same resamples as run 2)
    boots = P.boot_indices(cl, P.N_BOOT)
    mbfp_b = np.array([mb_fp_flag[idx].sum() for idx in boots])
    bs, rows = {}, []
    for key, r in res.items():
        pf = r["pf"].astype(bool)
        fp_flag, rec_flag = pf & yp_b, pf & yf_b
        fixed, rec, match = [], [], []
        for idx in boots:
            nf = yf[idx].sum()
            fixed.append(fp_flag[idx].sum())
            rec.append(rec_flag[idx].sum() / nf)
            match.append(P.matched_fp(r["s"][idx], yf[idx], mb_rec_flag[idx].sum() / nf)[0])
        bs[key] = dict(fixed=np.array(fixed), rec=np.array(rec), match=np.array(match))
        fa = [roc_auc_score(yf[te], r["s"][te]) for te in tests]
        red_b = mbfp_b - bs[key]["fixed"]
        redpct = np.where(mbfp_b > 0, red_b / np.maximum(mbfp_b, 1), np.nan)
        fp = int(fp_flag.sum())
        rows.append(dict(spec=key, new_in_run2d=r["new"], forest_recall=float(rec_flag.sum() / n_fo), fp=fp,
                         fp_rate=fp / n_pl, fp_reduction_vs_mb=mb_fp - fp, fp_reduction_pct=100 * (mb_fp - fp) / mb_fp,
                         red_pct_ci_low=100 * np.nanpercentile(redpct, 2.5), red_pct_ci_high=100 * np.nanpercentile(redpct, 97.5),
                         missed_rejected=n_missed - int((pf & is_missed).sum()),
                         new_fp_caught=int((pf & is_caught).sum()), new_fp_other=int((pf & is_other).sum()),
                         auc_pooled=roc_auc_score(yf, r["s"]), auc_fold_mean=np.mean(fa), auc_fold_sd=np.std(fa, ddof=1),
                         accuracy=float(((pf & yf_b).sum() + (~pf & ~yf_b).sum()) / len(yf)),
                         expost_matched_fp=P.matched_fp(r["s"], yf, mb_rec)[0]))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, "vv_swap_metrics.csv"), index=False)
    print(f"\n[METRICS] MapBiomas: forest recall {mb_rec:.3f}, FP {mb_fp}")
    print(M.drop(columns=["new_in_run2d"]).round(3).to_string(index=False))

    pairs = [("RF|A_vv", "RF|A0", "RF: VV added to A0 (effect of VV)"),
             ("RF|A", "RF|A0", "RF: VH added to A0 (effect of VH; run 2)"),
             ("RF|A_vv", "RF|A", "RF: A with VV minus A with VH"),
             ("LR-plain|A_vv", "LR-plain|A0", "plain LR: VV added to A0"),
             ("LR-plain|A", "LR-plain|A0", "plain LR: VH added to A0 (run 2)"),
             ("RF|C_vv", "RF|C", "RF: C with VV minus C with VH"),
             ("RF|C_vv", "RF|B", "RF: C with VV minus B (embeddings only)"),
             ("LR-L2|C_vv", "LR-L2|C", "L2-LR: C with VV minus C with VH"),
             ("LR-L2|C_vv", "LR-L2|B", "L2-LR: C with VV minus B (embeddings only)")]
    Mi = M.set_index("spec")
    prow = []
    for a, b, desc in pairs:
        dfp, drec, dmt = bs[a]["fixed"] - bs[b]["fixed"], bs[a]["rec"] - bs[b]["rec"], bs[a]["match"] - bs[b]["match"]
        prow.append(dict(comparison=desc, fp_a=int(Mi.loc[a, "fp"]), fp_b=int(Mi.loc[b, "fp"]),
                         fp_diff=int(Mi.loc[a, "fp"] - Mi.loc[b, "fp"]), ci_low=P.ci(dfp)[0], ci_high=P.ci(dfp)[1],
                         recall_diff=float(Mi.loc[a, "forest_recall"] - Mi.loc[b, "forest_recall"]),
                         auc_diff=float(Mi.loc[a, "auc_pooled"] - Mi.loc[b, "auc_pooled"]),
                         expost_fp_diff=int(Mi.loc[a, "expost_matched_fp"] - Mi.loc[b, "expost_matched_fp"]),
                         expost_ci_low=P.ci(dmt)[0], expost_ci_high=P.ci(dmt)[1]))
    Pp = pd.DataFrame(prow)
    Pp.to_csv(os.path.join(OUT, "vv_swap_paired.csv"), index=False)
    print("\n[PAIRED] FP count difference (first - second; NEGATIVE = fewer FP for the first), 95% bootstrap CI over "
          "municipalities. RF noise reference from run 2: identical features reordered gave +1 to +3 FP.")
    print(Pp.round(3).to_string(index=False))

    # ---- logit on A_vv (forest = 1)
    XAs = StandardScaler().fit_transform(dm[A_VV].to_numpy(float))
    tab, info = P.logit_inference(XAs, yf, cl, A_VV)
    tab.to_csv(os.path.join(OUT, "coef_table_A_vv_wald_vs_robust.csv"))
    print("\n[LOGIT] plain LR on A_vv, outcome = FOREST (OR>1: more forest-like), per 1 SD")
    print("[LOGIT] diagnostics:", info)
    print(tab[["odds_ratio_per_SD", "wald_OR_low", "wald_OR_high", "wald_p", "robust_OR_low", "robust_OR_high",
               "robust_p", "robust_se", "se_ratio_robust/wald"]].round(3).to_string())

    # ---- grouped permutation importance (RF), A_vv and C_vv
    hand_groups = {"+".join(g): g for g in hg}
    imps = []
    for set_name, key in [("A_vv", "RF|A_vv"), ("C_vv", "RF|C_vv")]:
        groups = dict(hand_groups)
        if set_name == "C_vv":
            groups = {"embeddings (A00-A63, 64)": P.EMB, **groups}
        imp = P.grouped_permutation(res[key]["models"], tests, res[key]["X"], res[key]["cols"], yf, groups,
                                    mb_rec, res[key]["s"], P.SEED + 1)
        imp.insert(0, "feature_set", set_name)
        imp["baseline_fp_matched"] = P.matched_fp(res[key]["s"], yf, mb_rec)[0]
        imps.append(imp)
        print(f"\n[IMPORTANCE] RF on {set_name}")
        print(imp.round(4).to_string(index=False))
    pd.concat(imps).to_csv(os.path.join(OUT, "grouped_importance_vv.csv"), index=False)

    o = dm[["pt_id", "CD_MUN", "group", "mb_pred"]].copy()
    o["forest"] = yf
    o["fold"] = fold_id
    for key in new:
        tag = key.replace("|", "_").replace("-", "")
        o[f"forest_score__{tag}"], o[f"pred_forest__{tag}"] = res[key]["s"], res[key]["pf"]
    o.to_csv(os.path.join(OUT, "oof_predictions_run2d.csv"), index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
