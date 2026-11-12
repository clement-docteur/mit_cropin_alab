#!/usr/bin/env python3
"""
Run 2b: do height_stdDev and ever_farm add anything to model C (A + embeddings)?
Retrain WITHOUT both features (RF and L2-LR) and compare false positives at matched forest recall
with the SAME folds and the SAME municipality-bootstrap resamples as run 2 (paired differences).

Reuses poc2_vh_embeddings.py (same functions, same seed, same evaluation rule). Reference models
(RF/L2-LR on B and C) are read from the saved out-of-fold file of run 2, not refit.
Noise calibration: RF on C with the columns reversed (identical features) shows how much FP moves
from random-forest feature-sampling noise alone.

Usage: python run2b_drop_height_farm.py   (run from the folder that contains poc2_vh_embeddings.py)
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import poc2_vh_embeddings as P

FEATS = "data/cropin_model_features_2022_updated.csv"
EMBS = "data/cropin_embeddings_2022.csv"
OUT = "results"
A_COLS = ["height_stdDev", "vh_mean", "ndvi_aug_oct_mean", "sag_oct_nov_mean", "sag_swing_mean", "ever_farm"]
KEEP = [c for c in A_COLS if c not in ("height_stdDev", "ever_farm")]   # 4 handcrafted features


def main():
    t0 = time.time()
    f, e = pd.read_csv(FEATS), pd.read_csv(EMBS)
    d = f.merge(e[["pt_id"] + P.EMB], on="pt_id", how="inner", validate="1:1")
    d["ever_farm"] = (d["ever_farm_mean"] >= 0.5).astype(int)
    dm = d.dropna(subset=P.HAND + P.EMB).reset_index(drop=True)
    oof = pd.read_csv(os.path.join(OUT, "oof_predictions_run2.csv"), float_precision="round_trip")
    assert (oof["pt_id"].to_numpy() == dm["pt_id"].to_numpy()).all(), "row order differs from run 2"

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
    assert (fold_id == oof["fold"].to_numpy()).all(), "folds differ from run 2"
    print(f"n={len(dm)}  plantations={n_pl}  forest={n_fo}  missed plantations={n_missed}  "
          f"MapBiomas: forest recall={mb_rec:.4f}, FP={mb_fp}")
    print(f"handcrafted kept in 'C_drop2': {KEEP}  (dropped: height_stdDev, ever_farm)")

    res = {}
    # ---- reference models from run 2 (not refit)
    for key, tag in [("RF|B", "RF_B"), ("RF|C", "RF_C"), ("LR-L2|B", "LRL2_B"), ("LR-L2|C", "LRL2_C")]:
        res[key] = dict(s=oof[f"forest_score__{tag}"].to_numpy(), pf=oof[f"pred_forest__{tag}"].to_numpy(int))
    # ---- new models
    new = {"RF|C_drop2": ("RF", KEEP + P.EMB),
           "LR-L2|C_drop2": ("LR-L2", KEEP + P.EMB),
           "RF|C_reordered (noise check)": ("RF", (A_COLS + P.EMB)[::-1])}
    Cs = np.logspace(-4, 1, 11)
    for key, (model, cols) in new.items():
        tk = time.time()
        X = dm[cols].to_numpy(float)
        s, thr = np.zeros(len(dm)), np.zeros(len(dm))
        Cch = []
        for k, (tr, te) in enumerate(outer):
            s_te, t, target, _, extra = P.run_fold(model, X, yf, cl, mbf, tr, te, Cs)
            s[te], thr[te] = s_te, t
            if "C" in extra:
                Cch.append(extra["C"])
        res[key] = dict(s=s, pf=(s >= thr).astype(int))
        print(f"[fit] {key:<30} {time.time()-tk:4.0f}s  n_features={len(cols)}"
              + (f"  C per fold={['%.3g' % c for c in Cch]}" if Cch else ""))

    # ---- metrics + paired bootstrap (same resamples as run 2)
    boots = P.boot_indices(cl, P.N_BOOT)
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
        fa = [roc_auc_score(yf[te], r["s"][te]) for _, te in outer]
        rows.append(dict(spec=key, forest_recall=float(rec_flag.sum() / n_fo), fp=int(fp_flag.sum()),
                         fp_rate=float(fp_flag.sum() / n_pl), mb_fp=mb_fp,
                         fp_reduction_vs_mb=mb_fp - int(fp_flag.sum()),
                         missed_rejected=n_missed - int((pf & is_missed).sum()),
                         new_fp_caught=int((pf & is_caught).sum()), new_fp_other=int((pf & is_other).sum()),
                         auc_pooled=roc_auc_score(yf, r["s"]), auc_fold_mean=np.mean(fa), auc_fold_sd=np.std(fa, ddof=1),
                         expost_matched_fp=P.matched_fp(r["s"], yf, mb_rec)[0]))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, "drop_height_farm_metrics.csv"), index=False)
    print("\n[METRICS] (MapBiomas: forest recall %.3f, FP %d)" % (mb_rec, mb_fp))
    print(M.round(3).to_string(index=False))

    pairs = [("RF|C_drop2", "RF|C", "RF: C without height+farm  minus  C"),
             ("RF|C_reordered (noise check)", "RF|C", "RF NOISE CHECK: C reordered minus C (identical features)"),
             ("LR-L2|C_drop2", "LR-L2|C", "L2-LR: C without height+farm  minus  C"),
             ("RF|C_drop2", "RF|B", "RF: C without height+farm  minus  B (embeddings only)"),
             ("LR-L2|C_drop2", "LR-L2|B", "L2-LR: C without height+farm  minus  B (embeddings only)")]
    prow = []
    for a, b, desc in pairs:
        dfp, drec, dm_ = bs[a]["fixed"] - bs[b]["fixed"], bs[a]["rec"] - bs[b]["rec"], bs[a]["match"] - bs[b]["match"]
        ra, rb = M.set_index("spec").loc[a], M.set_index("spec").loc[b]
        prow.append(dict(comparison=desc, fp_a=int(ra.fp), fp_b=int(rb.fp), fp_diff=int(ra.fp - rb.fp),
                         ci_low=P.ci(dfp)[0], ci_high=P.ci(dfp)[1],
                         recall_diff=float(ra.forest_recall - rb.forest_recall),
                         recall_ci_low=P.ci(drec)[0], recall_ci_high=P.ci(drec)[1],
                         missed_rejected_diff=int(ra.missed_rejected - rb.missed_rejected),
                         auc_diff=float(ra.auc_pooled - rb.auc_pooled),
                         expost_fp_diff=int(ra.expost_matched_fp - rb.expost_matched_fp),
                         expost_ci_low=P.ci(dm_)[0], expost_ci_high=P.ci(dm_)[1]))
    Pp = pd.DataFrame(prow)
    Pp.to_csv(os.path.join(OUT, "drop_height_farm_paired.csv"), index=False)
    print("\n[PAIRED] FP count difference (first - second; NEGATIVE = fewer FP for the first), "
          "95% bootstrap CI over municipalities, same resamples as run 2:")
    print(Pp.round(3).to_string(index=False))

    o = dm[["pt_id", "CD_MUN", "group", "mb_pred"]].copy()
    o["forest"] = yf                                         # 1 = forest
    o["fold"] = fold_id
    for key in new:
        tag = key.split(" ")[0].replace("|", "_").replace("-", "")
        o[f"forest_score__{tag}"], o[f"pred_forest__{tag}"] = res[key]["s"], res[key]["pf"]
    o.to_csv(os.path.join(OUT, "oof_predictions_run2b.csv"), index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
