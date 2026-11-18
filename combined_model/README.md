# Combined model: plantation vs natural forest (southern Brazil)

Nathan Chu (with Claude), 13 November 2026. This folder holds the shared feature table's build code, the combined-model search, and its results. It brings together features explored separately by Clement, Meera, Julia, and Nathan.

**Start here:** [`docs/Combined Model Search Protocol.md`](docs/Combined%20Model%20Search%20Protocol.md). It fixes the models, metrics, and selection rule before any model was fit, and records the finalists before the held-back test.

## Headline result (QC held-back half: 59 forest, 148 plantation; scored once)

| Model | AUC [95% CI] | Forest errors | Plantation recall |
|---|---|---|---|
| SAG (Cropin's index, reference) | 0.83 [0.75–0.89] | 3/59 | 53% |
| I2: sparse logistic regression, 14 features | 0.87 [0.82–0.92] | 2/59 | 46% |
| B5: random forest, 32 features + AlphaEarth | 0.93 [0.89–0.96] | 2/59 | 62% |

B5 vs SAG: AUC +0.10 [+0.04, +0.17], p < 0.001. I2 vs SAG: +0.05 [−0.03, +0.12], p = 0.12. Forest errors at a threshold set for 5% forest error on MapBiomas points. The held-back half was already seen during teammates' feature screening, so treat these numbers as optimistic. Cropin has no labels for forest it misses, so no number here shows the overall forest error.

## Contents

| Path | What it is |
|---|---|
| `docs/Results Summary.md` | Team summary of the search and the held-back result |
| `docs/Combined Model Search Protocol.md` | Locked protocol, addendum, and finalist record |
| `docs/Master feature table v3 - Data Dictionary.md` | **Current** shared feature table: every column, roles, and known issues (exactly the data the search used) |
| `docs/v3_columns.csv` | Each v3 column's role, MapBiomas AUC, and missing counts |
| `docs/archive/` | v2 dictionary and the v2 keep/drop decisions (how duplicates were resolved) |
| `scripts/build_master.py`, `build_master_v2.py`, `build_master_v3.py`, `harvest_features.py`, `meera_features.py` | Build the shared feature table (v1 merge, v2 deduplication, v3 = data used by the search; Meera's six measurements) |
| `scripts/rebuild_harvest.py`, `land_history.py` | Teammates' harvest-cycle and land-history ideas, rebuilt on our points and as-of windows |
| `scripts/search_v1.py`, `search_v1_extra.py`, `search_v2.py`, `heldback_v2.py` | Preliminary search, final search, and single held-back scoring |
| `scripts/sag_plus.py` | Follow-up: Cropin's SAG + forward-selected features (development half only) |
| `scripts/gee/` | GEE templates and generators for batch M1 (winter B6) and M2 (MapBiomas Collection 9 history) |
| `results/` | Aggregate results only (metrics, comparisons, model settings) |

## Conventions

- **Label 1 = plantation, 0 = natural forest** (run 2 used forest = 1).
- **Forest error** = real forest called plantation. It is not Cropin's "false positive".
- Training: 1,700 MapBiomas points (stable class 3/9 over 2013–2022, at least 2 interpreters agree, summer NDVI > 0.6). Spatial 5-fold cross-validation by 0.5° block.
- Prefixes: `nc_` Nathan + Claude · `mm_` Meera's measurements (rebuilt) · `rh_` harvest cycles (rebuilt) · `lh_` land history (rebuilt, MapBiomas-derived) · `ae_` AlphaEarth.

## Not in this repo (on purpose)

- **Data files:** the master feature table, AlphaEarth table, and per-point features contain or link to Cropin's confidential QC labels. They are in the team's shared Drive.
- **Generated GEE scripts** (`*_gee.js`): they embed point coordinates, including Cropin's QC locations. Use the templates and generators instead.

File paths in the scripts point to where they were run, so adjust them before re-running. Packages: pandas, numpy, scipy, scikit-learn.
