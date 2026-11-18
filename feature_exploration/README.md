# Feature Exploration: plantation vs natural forest (southern Brazil)

Nathan Chu (with Claude), November 2026. This folder contains the method, the results, and all the code for testing which satellite features separate **forest plantations from natural forest** in Paraná, Santa Catarina, and Rio Grande do Sul, for Cropin's EUDR deforestation alerts.

**Start here:** [`docs/Feature Exploration - Summary of Findings.md`](docs/Feature%20Exploration%20-%20Summary%20of%20Findings.md). It explains every hypothesis tested, how it was tested, and the results.

## Contents

| Path | What it is |
|---|---|
| `docs/00 Workflow Plan.md` | The method: validity rules, decision rule, phases |
| `docs/Feature Exploration - Summary of Findings.md` | One-document summary of Phases 0–7, including the final test |
| `docs/Phase 7 Protocol.md` | The final test, locked before the sealed data were opened |
| `docs/Canopy Height and Wetness - Team Summary.md` | Team-facing summary of the canopy height and wetness (canopy moisture) hypotheses: reasoning, method, problems, and results |
| `docs/Wave 1 Report.md`, `Wave 2 Report.md`, `Wave 3 Report.md` | Detailed results by feature family |
| `docs/Run Log.md` | Every run, decision, fix, and independent review, in order |
| `scripts/` | All analysis code (Python) and Google Earth Engine templates (JavaScript) |
| `scripts/review/` | Code the independent reviewer agents wrote to re-check the numbers |

## Scripts by phase

| Phase | Scripts |
|---|---|
| 0 Setup: points, splits, clearing dates, baselines | `p0_dbf.py`, `p0_prepare.py`, `p0_qc_split.py`, `p0_make_gee_script.py` + `p0_batch0_template.js`, `p0_clearing_dates.py`, `p0_build_tables.py`, `p0_baselines.py` |
| Wave 1: spectral (A), texture (C), history (D) | `w1_make_gee_script.py` + `w1_batch1_template.js`, `w1b_*`, `w1c_*` (refit), `w1_features.py`, `w1_common.py`, `w1_screen.py`, `w1_family_models.py`, `w1_nested.py`, `w1_clear_obs.py` |
| Wave 2: seasonal (B), radar (F) | `w2_*` |
| Wave 3: structure (E), terrain/context (G), public maps (H) | `w3_*`, `w3b_*` (GEDI re-export) |
| Phase 4: combine families | `p4_combine.py`, `p4_lean.py` |
| Phase 5: robustness | `p5_year_features.py`, `p5_robust.py` |
| Phase 6: error breakdown | `p6_errors.py` |
| Phase 7: final sealed test | `p7_make_gee_script.py` + `p7_alphaearth_template.js` (AlphaEarth benchmark), `p7_final.py` |
| Post-hoc (after Phase 7): winter NDMI | `p8_winter_ndmi.py`, `p8_common.py`. Pre-registered in the Run Log before running, but **post-hoc**: it re-used the already-opened sealed sets, so it is not an independent test |

## How the pipeline runs

1. **Python** (`*_make_gee_script.py`) reads the labelled points and writes a Google Earth Engine script from a `*_template.js`, with the points embedded.
2. The script is **run in the GEE Code Editor**. It exports CSVs to Google Drive (folder `cropin_texture`).
3. **Python** builds features from those CSVs, screens them (`*_screen.py`), fits models (`*_family_models.py`), and runs nested cross-validation (`*_nested.py`).

File paths in the scripts point to where they were run (a local folder and a cloud workspace), so adjust them before re-running. Python packages used: pandas, numpy, scipy, scikit-learn.

## What is deliberately NOT in this repo

- **Data files.** The MapBiomas validation points, Cropin's QC polygons, GEE exports, and feature/result tables are not here. They can be regenerated from the scripts and the source data.
- **Generated GEE scripts** (`*_gee.js`). They embed point coordinates and labels, including Cropin's QC data. Only the templates are included.
- **The sealed test sets.** A 20% MapBiomas "lockbox" and half of Cropin's QC points were kept unopened until the single final test (Phase 7, 11 Nov 2026). They have now been used once, so any new result on them is no longer an independent test. The files are named `*DO_NOT_OPEN*`.

## Key definitions

- **Forest error** = real forest called plantation. This is our most important error, and it is not Cropin's "false positive".
- **Plantation recall at 5% forest error** = the share of real plantations caught when the threshold is set so that 5% of real forest is called plantation.
- Cropin has no labels for forest it misses, so no number here shows the overall forest error.
