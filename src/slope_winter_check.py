"""Does the winter gain depend on terrain? (follow-up to the blind map check, gee/06)

In the blind map check, winter features helped in PR and RS. In the steep SC test area, almost every point where
winter added "plantation" was native forest, and winter SAG there looked like a hillshade.
The SC labels were the least reliable, so this script tests the same idea on the 2,290 MapBiomas
reference points instead, where the labels don't depend on us.

  1. Shadow test: within native forest only, does winter SAG follow the winter hillshade? If shadow
     drives the signal, shaded forest should look more "plantation-like".
  2. Gain by terrain: out-of-fold scores (spatial CV, 20 blocks, averaged over seeds 0-19) for
       A = SAG Oct-Jan only, B = the 6-feature short list, C = B plus slope and winter hillshade.
     Split the points into thirds by slope, and again by winter hillshade. In each third, report the AUC of A, B and C,
     B's gain over A (paired bootstrap 95% CI), and, at one cutoff hiding 5% of all forest, the share of forest
     flagged in that third. If shadow fools the winter features, B's forest-flagged share should climb on steep or
     shaded ground.
Inputs: data/gee_exports/ref_s2_bands_by_month_*.csv (gee/05), data/gee_exports/ref_terrain.csv (gee/07).
Output is aggregate only, plus figures/05a_winter_gain_by_terrain.png.
Run:  .venv/bin/python src/slope_winter_check.py
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_analysis import DATA, PAL, make_model, savefig, set_style, spatial_groups, threshold_hiding_at_most  # noqa: E402
from map_model_weights import SHORT, load_ref  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS matmul warnings (see CLAUDE.md)
TERRAIN = Path(os.environ.get("TERRAIN_CSV", DATA / "gee_exports/ref_terrain.csv"))
MODELS = {"A: SAG Oct-Jan": ["SAG_octjan"], "B: 6 features": SHORT,
          "C: 6 features + terrain": SHORT + ["slope_deg", "hs_winter"]}
SEEDS = range(20)


def oof(f, cols, seeds=SEEDS):
    X, y = f[cols].values, f.y.values
    p = np.zeros(len(y))
    for s in seeds:
        g = spatial_groups(f, k=20, seed=s)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            p[te] += make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    p /= len(seeds)
    assert np.isfinite(p).all()
    return p


def gain_ci(y, pa, pb, n_boot=2000, seed=0):
    rng, n, d = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if y[s].min() != y[s].max():
            d.append(roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]))
    return np.percentile(d, [2.5, 97.5])


def main():
    f = load_ref()
    t = pd.read_csv(TERRAIN).set_index("id")[["slope_deg", "hs_winter", "hs_summer", "relief_m"]]
    f = f.join(t, how="inner").dropna(subset=["slope_deg", "hs_winter"])
    print(f"reference points with terrain: {len(f)} ({f.y.sum()} plantation, {(f.y == 0).sum()} forest)")
    print("median slope by state (deg):", f.groupby("state").slope_deg.median().round(1).to_dict(),
          "| share steeper than 15 deg:", f.groupby("state").slope_deg.apply(lambda s: (s > 15).mean()).round(2).to_dict())

    print("\n1. SHADOW TEST, native forest only (Spearman rho; rho > 0 means sunlit forest has a higher index value)")
    fo = f[f.y == 0]
    for col, hs in [("SAG_winter", "hs_winter"), ("SAG_octjan", "hs_summer"), ("NDVI_winter", "hs_winter"),
                    ("B8A_winter", "hs_winter"), ("B6_winter", "hs_winter")]:
        r, p = spearmanr(fo[col], fo[hs])
        print(f"   {col:12s} vs {hs:9s}: rho {r:+.2f} (p {p:.1g})")
    print("   For reference, plantation vs forest gap: SAG_winter AUC "
          f"{roc_auc_score(f.y, f.SAG_winter):.2f} (above 0.5 means plantations higher)")

    scores = {m: oof(f, cols) for m, cols in MODELS.items()}
    cuts = {m: threshold_hiding_at_most(p[f.y.values == 0], 0.05) for m, p in scores.items()}
    y = f.y.values
    print("\n   overall AUC: " + ", ".join(f"{m} {roc_auc_score(y, p):.3f}" for m, p in scores.items()))

    rows = []
    for split, col in [("slope", "slope_deg"), ("winter hillshade", "hs_winter")]:
        bins = pd.qcut(f[col], 3, labels=["low", "mid", "high"])
        edges = f[col].quantile([0, 1 / 3, 2 / 3, 1]).round(1).tolist()
        print(f"\n2. BY {split.upper()} (thirds; edges {edges})")
        for b in ["low", "mid", "high"]:
            m = (bins == b).values
            yb = y[m]
            r = {"split": split, "bin": b, "n": int(m.sum()), "n_plant": int(yb.sum())}
            for name, p in scores.items():
                k = name[0]
                r[f"auc_{k}"] = roc_auc_score(yb, p[m])
                r[f"forest_flag_{k}"] = (p[m][yb == 0] >= cuts[name]).mean()
                r[f"caught_{k}"] = (p[m][yb == 1] >= cuts[name]).mean()
            pa, pb = scores["A: SAG Oct-Jan"][m], scores["B: 6 features"][m]
            r["gain_lo"], r["gain_hi"] = gain_ci(yb, pa, pb)
            rows.append(r)
            print(f"   {b:4s} n={r['n']:4d} ({r['n_plant']} plant) | AUC A {r['auc_A']:.3f}  B {r['auc_B']:.3f}  "
                  f"C {r['auc_C']:.3f} | B-A {r['auc_B'] - r['auc_A']:+.3f} [{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}] | "
                  f"forest flagged A {r['forest_flag_A']:.0%} B {r['forest_flag_B']:.0%} C {r['forest_flag_C']:.0%} | "
                  f"caught A {r['caught_A']:.0%} B {r['caught_B']:.0%} C {r['caught_C']:.0%}")
    res = pd.DataFrame(rows)

    import matplotlib.pyplot as plt
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
    colors = {"A": "#9A9A9A", "B": PAL["plantation"], "C": "#4A4A4A"}
    for j, split in enumerate(["slope", "winter hillshade"]):
        d = res[res.split == split]
        x = np.arange(3)
        for i, k in enumerate("ABC"):
            axes[0, j].bar(x + (i - 1) * 0.27, d[f"auc_{k}"], 0.27, color=colors[k], label=list(MODELS)[i])
            axes[1, j].bar(x + (i - 1) * 0.27, 100 * d[f"forest_flag_{k}"], 0.27, color=colors[k])
        lab = ["flat", "moderate", "steep"] if split == "slope" else ["most shaded", "middle", "most sunlit"]
        for ax in axes[:, j]:
            ax.set_xticks(x, lab)
        axes[1, j].set_xlabel(f"Reference points by {split} (thirds)")
        axes[1, j].axhline(5, color="#BBBBBB", lw=1, ls="--")
    axes[0, 0].set_ylim(0.5, 1)
    axes[0, 0].set_ylabel("AUC (0.5 = coin flip, 1 = perfect)")
    axes[1, 0].set_ylabel("Native forest called plantation (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Does winter imagery still help on steep or shaded ground? (2,290 MapBiomas reference points)")
    savefig(fig, "05a_winter_gain_by_terrain")
    print("\nsaved figures/05a_winter_gain_by_terrain.png")


if __name__ == "__main__":
    main()
