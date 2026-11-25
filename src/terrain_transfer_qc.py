"""Should slope be in the model? Train on the MapBiomas reference points, test on Cropin's QC polygons.

On the reference points, adding slope and winter shadow to the 6-feature model hardly moved the AUC. But slope
mostly works as a land-use prior ("plantations are on flatter land") and pushed more flat native forest towards
"plantation", which is the direction that hides deforestation. The QC polygons are the alerts we care about, so this
compares three models there:
  B          the 6-feature short list (notebook 04)
  B+shadow   B plus winter hillshade (a physical correction for low winter sun on slopes)
  C          B plus winter hillshade plus slope (adds the land-use prior)
Each model is fitted on all reference points. Its cutoff hides <= 5% of reference forest (out-of-fold, spatial CV, seed 0,
as in notebook 04). It's then scored on the QC polygons: AUC with a bootstrap 95% CI, the paired AUC difference from B,
the share of plantations caught and the share of forest hidden. Hidden is also split by QC slope (flat < 6 deg, moderate 6-14,
steep > 14, the reference-point thirds). QC-only spatial CV with a nested cutoff is shown as a second view.
Inputs: gee/05 exports, data/gee_exports/ref_terrain.csv and qc_terrain.csv (gee/07). Output is aggregate only.
Run:  .venv/bin/python src/terrain_transfer_qc.py
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_analysis import (DATA, auc_ci, cv_evaluate, dedupe, load_qc, make_model, spatial_groups,  # noqa: E402
                         threshold_hiding_at_most)
from map_model_weights import OCTJAN, SHORT, WINTER, col, load_ref  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS matmul warnings (see CLAUDE.md)
TERR = ["slope_deg", "hs_winter"]
REF_T = Path(os.environ.get("REF_TERRAIN_CSV", DATA / "gee_exports/ref_terrain.csv"))
QC_T = Path(os.environ.get("QC_TERRAIN_CSV", DATA / "gee_exports/qc_terrain.csv"))
MODELS = {"B: 6 features": SHORT, "B + shadow": SHORT + ["hs_winter"], "C: B + shadow + slope": SHORT + TERR}
SLOPE_BINS, SLOPE_LAB = [-1, 6, 14, 90], ["flat (<6 deg)", "moderate (6-14)", "steep (>14)"]
SEED, N_BOOT = 0, 2000


def load_qc_features():
    raw = pd.read_csv(DATA / "gee_exports/qc_s2_bands_by_month.csv")
    qc = load_qc()[["id", "qc_label", "lat", "lon", "area_m2"]].merge(
        raw.drop(columns="qc_label"), on="id", how="inner", validate="one_to_one")
    qc = dedupe(qc).set_index("id")
    for name, (a, b) in {"NDVI": ("B8", "B4"), "NDMI": ("B8", "B11")}.items():
        for m in range(1, 13):
            qc[col(name, m)] = (qc[col(a, m)] - qc[col(b, m)]) / (qc[col(a, m)] + qc[col(b, m)])
    f = pd.DataFrame(index=qc.index)
    for b in ["SAG", "B6", "B8A", "NDMI", "NDVI"]:
        f[f"{b}_octjan"] = qc[[col(b, m) for m in OCTJAN]].mean(axis=1)
        f[f"{b}_winter"] = qc[[col(b, m) for m in WINTER]].mean(axis=1)
    f["y"] = (qc.qc_label == "plantation").astype(int)
    return f.join(qc[["lat", "lon"]])


def paired_diff_ci(y, pa, pb):
    rng, n, d = np.random.default_rng(SEED), len(y), []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if y[s].min() != y[s].max():
            d.append(roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]))
    return np.percentile(d, [2.5, 97.5])


def main():
    need = SHORT + TERR
    fr = load_ref().join(pd.read_csv(REF_T).set_index("id")[TERR], how="inner").dropna(subset=need)
    fq = load_qc_features().join(pd.read_csv(QC_T).set_index("id")[TERR], how="inner").dropna(subset=need)
    yq = fq.y.values
    nf = int((yq == 0).sum())
    print(f"reference: {len(fr)} ({fr.y.sum()} plantation) | QC: {len(fq)} ({yq.sum()} plantation, {nf} forest); "
          f"one QC forest = {100 / nf:.1f} points of 'hidden'")
    print("median slope (deg): reference plantation %.0f, forest %.0f | QC plantation %.0f, forest %.0f" % (
        fr[fr.y == 1].slope_deg.median(), fr[fr.y == 0].slope_deg.median(),
        fq[fq.y == 1].slope_deg.median(), fq[fq.y == 0].slope_deg.median()))
    sb = pd.cut(fq.slope_deg, SLOPE_BINS, labels=SLOPE_LAB)
    print("QC forest per slope band:", sb[yq == 0].value_counts().reindex(SLOPE_LAB).to_dict(),
          "| QC plantation:", sb[yq == 1].value_counts().reindex(SLOPE_LAB).to_dict())

    g_r, g_q = spatial_groups(fr, k=20, seed=SEED), spatial_groups(fq, k=20, seed=SEED)
    P, rows = {}, []
    for name, cols in MODELS.items():
        _, oof = cv_evaluate(fr[cols].values, fr.y.values, groups=g_r, kind="group", seeds=[SEED])
        t = threshold_hiding_at_most(oof[SEED][fr.y.values == 0], 0.05)
        p = make_model().fit(fr[cols].values, fr.y.values).predict_proba(fq[cols].values)[:, 1]
        P[name] = p
        a, lo, hi = auc_ci(yq, p, n_boot=N_BOOT, seed=SEED)
        flag = p >= t
        r = {"model": name, "ref->QC AUC": f"{a:.3f} [{lo:.2f}-{hi:.2f}]",
             "caught %": round(100 * flag[yq == 1].mean(), 1),
             "forest hidden": f"{int(flag[yq == 0].sum())}/{nf} ({100 * flag[yq == 0].mean():.1f}%)"}
        for lab in SLOPE_LAB:
            m = ((sb == lab).values) & (yq == 0)
            r[f"hidden {lab.split()[0]}"] = f"{int(flag[m].sum())}/{int(m.sum())}"
        rq, _ = cv_evaluate(fq[cols].values, yq, groups=g_q, kind="group", seeds=range(20))
        r["QC-only CV AUC"] = round(rq.auc.mean(), 3)
        r["QC-only caught % / hidden %"] = f"{rq.nest_caught.mean():.0f} / {rq.nest_hidden.mean():.1f}"
        rows.append(r)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\n" + pd.DataFrame(rows).set_index("model").to_string())

    base = P["B: 6 features"]
    print("\nAUC difference from B on QC (paired bootstrap 95% CI):")
    for name in list(MODELS)[1:]:
        lo, hi = paired_diff_ci(yq, base, P[name])
        print(f"  {name:22s} {roc_auc_score(yq, P[name]) - roc_auc_score(yq, base):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nRead: 'hidden' = QC forest the model would call plantation, i.e. real clearing that would be dismissed.")


if __name__ == "__main__":
    main()
