"""Weights and test areas for the map-view check of the winter signal (gee/06).

Deepak (11/12 meeting) asked us to check the winter result on a map, not only in numbers. This script:
  1. picks three ~10 km test areas (one per state) where plantation and native forest reference points
     sit close together;
  2. refits the two notebook-04 models (SAG Oct-Jan only, and the 6-feature short list) on the reference
     points OUTSIDE those areas, so the map is drawn by a model that never saw them;
  3. sets each model's cutoff so it hides <= 5% of reference forest (out-of-fold, spatial CV);
  4. writes the weights in raw units (no scaler needed in Earth Engine) into gee/06 between the
     "// <<WEIGHTS" and "// WEIGHTS>>" markers.

Output is aggregate only: three bounding boxes, point counts, and model weights.
Run:  .venv/bin/python src/map_model_weights.py
"""
import glob
import json
import warnings
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_analysis import DATA, ROOT, make_model, spatial_groups, threshold_hiding_at_most  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

OCTJAN, WINTER = [10, 11, 12, 1], [4, 5, 6, 7, 8, 9]          # same seasons as notebook 04
SHORT = ["SAG_octjan", "B6_winter", "B8A_winter", "SAG_winter", "NDMI_winter", "NDVI_winter"]
MODELS = {"sag": ["SAG_octjan"], "six": SHORT}
CELL = 0.1          # test-area size in degrees (about 10 km)
MARGIN = 0.05       # also hold out points this close to an area, so neighbours can't leak in
SEED = 0
warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS matmul warnings (see CLAUDE.md)
col = lambda b, m: f"{b}_{m:02d}"


def load_ref():
    files = sorted(glob.glob(str(DATA / "gee_exports/ref_s2_bands_by_month_*.csv")))
    assert len(files) == 3, "run gee/05 first"
    ref = pd.concat([pd.read_csv(f) for f in files]).set_index("id")
    pts = pd.read_csv(DATA / "processed/mapbiomas_ref_south.csv").set_index("id")
    ref = ref.join(pts[["lat", "lon"]])
    for name, (a, b) in {"NDVI": ("B8", "B4"), "NDMI": ("B8", "B11")}.items():
        for m in range(1, 13):
            ref[col(name, m)] = (ref[col(a, m)] - ref[col(b, m)]) / (ref[col(a, m)] + ref[col(b, m)])
    f = pd.DataFrame(index=ref.index)
    for b in ["SAG", "B6", "B8A", "NDMI", "NDVI"]:
        f[f"{b}_octjan"] = ref[[col(b, m) for m in OCTJAN]].mean(axis=1)
        f[f"{b}_winter"] = ref[[col(b, m) for m in WINTER]].mean(axis=1)
    f["y"] = (ref.label == "plantation").astype(int)
    f["state"] = ref.state
    return f.join(ref[["lat", "lon"]]).dropna(subset=SHORT)


def pick_areas(f):
    """One CELL x CELL box per state with the most balanced mix of plantation and forest points."""
    g = f.assign(cy=np.floor(f.lat / CELL), cx=np.floor(f.lon / CELL))
    c = g.groupby(["state", "cy", "cx"]).y.agg(plant="sum", n="size").reset_index()
    c["forest"] = c.n - c.plant
    c["mix"] = c[["plant", "forest"]].min(axis=1)
    best = c.sort_values(["mix", "n"], ascending=False).groupby("state").head(1)
    areas = []
    for _, r in best.sort_values("state").iterrows():
        areas.append({"name": r.state, "west": round(r.cx * CELL, 2), "south": round(r.cy * CELL, 2),
                      "east": round((r.cx + 1) * CELL, 2), "north": round((r.cy + 1) * CELL, 2),
                      "n_plantation_points": int(r.plant), "n_forest_points": int(r.forest)})
    return areas


def in_any(f, areas, margin):
    m = np.zeros(len(f), bool)
    for a in areas:
        m |= ((f.lon >= a["west"] - margin) & (f.lon <= a["east"] + margin)
              & (f.lat >= a["south"] - margin) & (f.lat <= a["north"] + margin)).values
    return m


def oof_scores(f, cols):
    X, y = f[cols].values, f.y.values
    groups = spatial_groups(f, k=20, seed=SEED)
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        p[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def raw_weights(model, cols):
    """Fold the StandardScaler into the logistic weights: logit = b0 + sum w_i * x_i (raw units)."""
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    w = lr.coef_[0] / sc.scale_
    b0 = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
    return dict(zip(cols, w.round(8).tolist())), float(round(b0, 8))


def main():
    f = load_ref()
    areas = pick_areas(f)
    held = in_any(f, areas, MARGIN)
    train = f[~held]
    print(f"reference points: {len(f)}; held out in or near the test areas: {held.sum()}; training: {len(train)}")
    for a in areas:
        print(f"  {a['name']}: box W{a['west']} S{a['south']} E{a['east']} N{a['north']} | "
              f"{a['n_plantation_points']} plantation, {a['n_forest_points']} forest points")

    out = {"areas": areas, "models": {}}
    for key, cols in MODELS.items():
        p = oof_scores(train, cols)
        cut = threshold_hiding_at_most(p[train.y.values == 0], 0.05)
        caught = (p[train.y.values == 1] >= cut).mean()
        m = make_model().fit(train[cols].values, train.y.values)
        w, b0 = raw_weights(m, cols)
        x = train[cols].values                       # check the folded weights reproduce the pipeline
        assert np.allclose(1 / (1 + np.exp(-(b0 + x @ np.array(list(w.values()))))), m.predict_proba(x)[:, 1], atol=1e-6)
        out["models"][key] = {"intercept": b0, "weights": w, "cutoff": round(float(cut), 4)}
        print(f"  model {key}: cutoff {cut:.3f} (hides <= 5% of training forest, catches {caught:.0%} of plantations)")

    js = "var CONFIG = " + json.dumps(out, indent=2) + ";"
    target = ROOT / "gee/06_map_view_winter_check.js"
    text = target.read_text()
    new = re.sub(r"// <<WEIGHTS.*?// WEIGHTS>>", "// <<WEIGHTS (written by src/map_model_weights.py)\n" + js + "\n// WEIGHTS>>",
                 text, flags=re.S)
    assert new != text or js in text, "markers not found in gee/06"
    target.write_text(new)
    print(f"wrote weights and areas into {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
