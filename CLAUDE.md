# CLAUDE.md: shared context for Claude Code sessions in this repo

## Project
MIT Analytics Lab × Cropin (fall 2026). Under the EUDR, land that was forest in Dec 2020 and is
cleared later is banned from EU supply chains. Cropin's forest layer raises clearing alerts, but
many are **plantation harvests** (eucalyptus, pine), not deforestation. Goal: an interpretable
**10 m plantation-probability layer** for southern Brazil (PR, SC, RS) that cuts false alarms
**without hiding real deforestation**.
- **Team:** Meera Mehta and Julia Hansen (data/EDA), Clement Docteur (plantation types, time
  series), Nathan Chu (structure, texture).
- **Mentor:** Raphael Raux.
- **Cropin:** Prakhyath Hegde, Deepak Murugan (technical lead), Uddesh (geospatial).
- **Deliverables:** to Cropin on 12/1, final on 12/10–11.

## Repo map
| Path | Contents |
|---|---|
| `data_exploration/` | Teammates' GEE scripts, `qc_map.html`, first notebook. **Don't edit others' files.** |
| `gee/01_*.js`, `gee/02_*.js` | Meera's exploration, and the export that makes `qc_error_analysis_A/B` |
| `gee/03_*.js` | Applies Cropin's forest layer and SAG to the QC polygons → `qc_cropin_layers.csv`, `qc_sag_timeseries.csv` |
| `gee/cropin/` | Cropin's own scripts, **verbatim** (forest layer, SAG posterior, SAG explorer). Plain-language guide in its README |
| `gee/archive/` | Superseded GEE versions (memory errors) |
| `data/` | Local-only data. `data/README.md` is the full data dictionary |
| `src/qc_analysis.py` | Shared helpers: loading, feature definitions, AUC/bootstrap, Fisher, Holm, spatial CV, nested threshold, palette |
| `src/build_mapbiomas_ref_south.py` | Rebuilds and checks the 2,290-point MapBiomas reference subset |
| `notebooks/01_qc_error_analysis.ipynb` | Benchmark, findings 1–6, combined model, reproducibility check |
| `notebooks/02_cropin_layers_and_SAG.ipynb` | Cropin layer and MapBiomas 10 m benchmark, SAG validation, Meta 1 m, SAG in the model |
| `figures/` | PNGs written by the notebooks (`01*`, `02*`) |
| `docs/eda_findings.md` | One-page team summary with current numbers |
| `docs/old_analysis/` | Aggregate stats from the old quick scripts (for the reproducibility check) |

## Data sources (all local; see `data/README.md`)
- **Cropin QC polygons** (sponsor):
  - file: `data/raw/cropin_qc/cropin_brazil_qc_forest_plantation_1.geojson`;
  - 433 polygons (310 plantation, 123 forest), plus `id` 0–432 in the GEE asset;
  - 12 locations are repeated (13 redundant rows).
- **MapBiomas 85k validation points** (shared by Deepak):
  - files: `data/raw/mapbiomas_validation/` (209 MB `.dbf`);
  - the LON/LAT columns are swapped, so use the geometry.
- **GEE exports:** `data/gee_exports/*.csv` (four files).

**GEE project `cropin-510901`** assets:
- `projects/cropin-510901/assets/cropin_qc`
- `projects/cropin-510901/assets/mapbiomas_ref_south`
- Julia's: `projects/cropin-510901/assets/cropin_brazil_qc_forest_plantation_1`, `.../mapbiomas_85k_points_validation`

**Public layers used:**
| Layer | Asset |
|---|---|
| MapBiomas C9 30 m | `projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1` |
| MapBiomas C10 30 m | `.../lulc/collection10/mapbiomas_brazil_collection10_coverage_v2` |
| MapBiomas 10 m C2 | `.../lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1` |
| MapBiomas 10 m S2-beta | `.../lulc/collection_S2_beta/collection_LULC_S2_beta` |
| MapBiomas 10 m C4 (Julia) | `.../lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1` |
| ETH canopy height | `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` |
| Meta canopy height | `projects/sat-io/open-datasets/facebook/meta-canopy-height` |
| Google Natural Forests | `projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection` (B0 / 250; threshold raw ≥ 125, not 0.5) |
| Hansen | `UMD/hansen/global_forest_change_2024_v1_12` |
| JRC GFC2020 | `JRC/GFC2020/V2` |
| TMF | `projects/JRC/TMF/v1_2024/AnnualChanges` |
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` |
| Cloud Score+ | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |

## Conventions
- **Never commit sponsor data, large files or recordings.**
  - The repo is **public**, so no Cropin data or anything derived per polygon: geojson, zips,
    exported CSVs, `data/processed/`.
  - No file > 50 MB.
  - No meeting transcripts or recordings, no teammates' decks, nothing with personal contact details.
  - `.gitignore` enforces the data folders. Check `git status` before every commit.
- **Notebook outputs** must stay aggregate. Never print raw rows or polygon coordinates.
- **Plain language** for a non-specialist sponsor. Explain each metric (AUC, odds ratio, p-value,
  CI) where it first appears. Be honest about caveats and don't oversell. Everything on the QC set
  is exploratory.
- **Statistics:**
  - Mann–Whitney U for continuous features, Fisher's exact test with odds ratios for binary ones;
  - bootstrap (2,000) 95% CIs for AUC;
  - Holm correction across a family of tests;
  - **spatially blocked CV** (20 KMeans blocks on lat/lon) with a **nested** ≤ 5%-forest threshold;
  - fixed seeds (0; seeds 0–19 for repeated CV).
  - Reuse `src/qc_analysis.py`; don't re-implement.
- **Charts:**
  - colours: MIT burgundy `#750014` for plantation and highlights, muted blue `#3B6EA8` for forest,
    greys for context, **no bright red** (`#FF1423`);
  - axis labels with units;
  - save PNGs to `figures/` via `savefig()`.
- **Git:** work on branches (`<name>/<topic>`), never commit to `main` directly, keep commits small,
  and don't modify teammates' files.

## How to rerun everything
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 1. GEE (only if the exports need refreshing): run gee/02 and gee/03 in the Code Editor
#    (project cropin-510901), RUN the tasks, download from Drive/cropin_exports to data/gee_exports/
# 2. optional check of the reference subset
.venv/bin/python src/build_mapbiomas_ref_south.py
# 3. notebooks (about 30 s each)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_qc_error_analysis.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_cropin_layers_and_SAG.ipynb
```
**macOS note:** NumPy 2 with Accelerate prints spurious "matmul" RuntimeWarnings from sklearn. The
notebooks filter them, and `cv_evaluate` asserts that predictions are finite.

## Current headline numbers (keep in sync with `docs/eda_findings.md`)
- **MapBiomas misses:**
  - C9 30 m 2020: 91% of QC plantations called forest (3.5% caught);
  - 10 m C2 / beta 2020: about 80%;
  - Julia's 10 m C4 2025: 82%.
- **Notebook-01 model:** spatial CV AUC 0.81. Removes about 30% of false alarms at ≤ 5% forest
  hidden. The midterm's 36.5% was a lucky split.
- **SAG Oct–Jan mean:** AUC 0.87. Adding it to the model gives AUC 0.89 and about 49% removed.
- **Cropin's posterior ≥ 0.8:** 68% caught, but 16% of forest flagged.
