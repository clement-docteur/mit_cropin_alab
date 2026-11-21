"""Draws the photo-interpretation ("gold") sample and writes blind labelling sheets.

    python pine_detection/src/make_gold_sample.py

Only MapBiomas reference points (public locations) are sampled, never Cropin QC polygons. The strata are
fixed in advance (seed 0). Batch 1 is labelled first, because it answers "how often is SDPT's species
right?".

| Stratum | Points | Batch | Why |
|---|---|---|---|
| A_pine        SDPT pine, MapBiomas plantation          | 50  | 1 | Measures SDPT accuracy (pine) |
| A_eucalyptus  SDPT eucalyptus, MapBiomas plantation    | 50  | 1 | Measures SDPT accuracy (eucalyptus) |
| C_forest_in_sdpt  MapBiomas forest inside SDPT polygon | all | 1 | MapBiomas error, SDPT error, or change since 2014? |
| B_unmatched   MapBiomas plantation, no SDPT polygon    | 80  | 2 | Newer plantations: the labels SDPT can't give |
| D_other       SDPT acacia or unknown species           | all | 2 | Third class (black wattle) |
| E_control     MapBiomas forest, no SDPT polygon        | 15  | 2 | Blind check that interpreters don't over-call plantations |

The A and B strata are drawn proportionally by state, from both the dev and lockbox splits (labels aren't
features, so gold labels on lockbox points don't leak into training).

Outputs go to data/processed/gold_sample/ (gitignored):
- gold_key.csv: sample_id plus stratum, SDPT species, MapBiomas label, uid, split. **Not for the
  interpreters.**
- gold_sheet_batch{1,2}.csv: sample_id, lon, lat, then blank columns to fill in. Rows are shuffled, and
  there is no stratum or label.
- gold_batch{1,2}.kml: the same points, for Google Earth Pro (name = sample_id only).
- gold_points_gee.csv: sample_id, batch, lon, lat, to upload as the GEE asset
  projects/cropin-510901/assets/species_gold_sample.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import species_features as sf

OUT = sf.ROOT / "data/processed/gold_sample"
SEED = 0
N = {"A_pine": 50, "A_eucalyptus": 50, "B_unmatched": 80, "E_control": 15}
BATCH = {"A_pine": 1, "A_eucalyptus": 1, "C_forest_in_sdpt": 1, "B_unmatched": 2, "D_other": 2, "E_control": 2}
SHEET_COLS = ["species", "confidence", "imagery_date", "species_2014", "evidence", "notes", "interpreter"]


def by_state(pool, n, rng):
    """Proportional-by-state sample of n rows (largest remainder), without replacement."""
    if len(pool) <= n:
        return pool
    share = pool.state.value_counts() / len(pool) * n
    k = np.floor(share).astype(int)
    for st in (share - k).sort_values(ascending=False).index[: n - k.sum()]:
        k[st] += 1
    parts = [pool[pool.state == st].sample(k[st], random_state=rng.integers(1 << 31)) for st in k.index]
    return pd.concat(parts)


def kml(df, title):
    marks = "\n".join(
        f"  <Placemark><name>{r.sample_id}</name><Point><coordinates>{r.lon:.6f},{r.lat:.6f},0"
        f"</coordinates></Point></Placemark>" for r in df.itertuples())
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{title}</name>\n{marks}\n</Document></kml>\n")


def main():
    rng = np.random.default_rng(SEED)
    m = sf.add_sdpt(sf.load_master())
    plant, forest = m[m.label == 1], m[m.label == 0]
    strata = {
        "A_pine": by_state(plant[plant.species == "pine"], N["A_pine"], rng),
        "A_eucalyptus": by_state(plant[plant.species == "eucalyptus"], N["A_eucalyptus"], rng),
        "C_forest_in_sdpt": forest[forest.species != "none"],
        "B_unmatched": by_state(plant[plant.species == "none"], N["B_unmatched"], rng),
        "D_other": plant[plant.species.isin(["acacia", "unknown"])],
        "E_control": by_state(forest[forest.species == "none"], N["E_control"], rng),
    }
    key = pd.concat([s.assign(stratum=k, batch=BATCH[k]) for k, s in strata.items()])
    assert key.uid.is_unique
    key = key.sample(frac=1, random_state=SEED).sort_values("batch", kind="stable").reset_index(drop=True)
    key.insert(0, "sample_id", [f"G{i + 1:03d}" for i in range(len(key))])

    OUT.mkdir(parents=True, exist_ok=True)
    key[["sample_id", "batch", "stratum", "species", "sdpt_in", "sdpt_near", "label", "uid", "team_pid",
         "split", "state", "lon", "lat"]].rename(columns={"species": "sdpt_species", "label": "mapbiomas_label"}) \
        .to_csv(OUT / "gold_key.csv", index=False)
    for b in (1, 2):
        s = key[key.batch == b][["sample_id", "lon", "lat"]].round(6)
        s.assign(**{c: "" for c in SHEET_COLS}).to_csv(OUT / f"gold_sheet_batch{b}.csv", index=False)
        (OUT / f"gold_batch{b}.kml").write_text(kml(s, f"Species gold sample, batch {b}"), encoding="utf-8")
    key[["sample_id", "batch", "lon", "lat"]].round(6).to_csv(OUT / "gold_points_gee.csv", index=False)

    print(f"{len(key)} points written to {OUT.relative_to(sf.ROOT).as_posix()}/")
    print(pd.crosstab(key.stratum, key.batch, margins=True))
    print(pd.crosstab(key.stratum, key.state))
    return key


if __name__ == "__main__":
    main()
