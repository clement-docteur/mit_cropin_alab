# Feature Exploration: plantation vs natural forest (southern Brazil)

Nathan Chu (with Claude), November 2026. This folder contains the method, the results, and all the code for testing which satellite features separate **forest plantations from natural forest** in Paraná, Santa Catarina, and Rio Grande do Sul, for Cropin's EUDR deforestation alerts.

**Start here:** [`docs/Feature Exploration - Summary of Findings.md`](docs/Feature%20Exploration%20-%20Summary%20of%20Findings.md). It explains every hypothesis tested, how it was tested, and the results.

## Contents

| Path | What it is |
|---|---|
| `docs/00 Workflow Plan.md` | The method: validity rules, decision rule, phases |
| `docs/Feature Exploration - Summary of Findings.md` | One-document summary of Phases 0–5 |
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

## How the pipeline runs

1. **Python** (`*_make_gee_script.py`) reads the labelled points and writes a Google Earth Engine script from a `*_template.js`, with the points embedded.
2. The script is **run in the GEE Code Editor**. It exports CSVs to Google Drive (folder `cropin_texture`).
3. **Python** builds features from those CSVs, screens them (`*_screen.py`), fits models (`*_family_models.py`), and runs nested cross-validation (`*_nested.py`).

File paths in the scripts point to where they were run (a local folder and a cloud workspace), so adjust them before re-running. Python packages used: pandas, numpy, scipy, scikit-learn.

## What is deliberately NOT in this repo

- **Data files.** The MapBiomas validation points, Cropin's QC polygons, GEE exports, and feature/result tables are not here. They can be regenerated from the scripts and the source data.
- **Generated GEE scripts** (`*_gee.js`). They embed point coordinates and labels, including Cropin's QC data. Only the templates are included.
- **The sealed test sets.** A 20% MapBiomas "lockbox" and half of Cropin's QC points are kept unopened for the single final test (Phase 7). Scripts that create them write files named `*DO_NOT_OPEN*`. **Please don't open or use those files**, or the final test is no longer independent.

## Key definitions

- **Forest error** = real forest called plantation. This is our most important error, and it is not Cropin's "false positive".
- **Plantation recall at 5% forest error** = the share of real plantations caught when the threshold is set so that 5% of real forest is called plantation.
- Cropin has no labels for forest it misses, so no number here shows the overall forest error.
