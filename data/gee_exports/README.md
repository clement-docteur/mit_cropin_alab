# Earth Engine exports: not in git (derived from sponsor polygons)

| File | Made by |
|---|---|
| `qc_error_analysis_A.csv`, `qc_error_analysis_B_ndvi.csv` | `gee/02_qc_error_analysis_export.js` |
| `qc_cropin_layers.csv`, `qc_sag_timeseries.csv` | `gee/03_cropin_layers_and_SAG_qc_export.js` |
| `ref_sag_timeseries_PR.csv`, `_SC.csv`, `_RS.csv` | `gee/04_sag_mapbiomas_ref_export.js` |
| `ref_s2_bands_by_month_PR.csv`, `_SC.csv`, `_RS.csv`, `qc_s2_bands_by_month.csv` | `gee/05_s2_bands_by_month_export.js` |

The scripts write to Google Drive, folder `cropin_exports`. Download the files into this
folder. Every column is described in `data/README.md`.
