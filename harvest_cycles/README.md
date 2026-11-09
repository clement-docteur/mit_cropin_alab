# Harvest cycles: plantation vs natural forest from the Landsat record (1985–2020)

Does a pixel's history of clear-cuts and regrowth tell a plantation from natural forest in Paraná, Santa Catarina and
Rio Grande do Sul, using only imagery up to the EUDR cutoff (end of 2020)? Findings: **[`docs/summary.md`](docs/summary.md)**.

## Layout
| Path | What it is |
|---|---|
| `docs/summary.md` | Findings, significance, recommendations, caveats |
| `scripts/` | The pipeline (below) |
| `gee/landtrendr_harvest_features.js` | Earth Engine version for production and the 10 m map: real LandTrendr on Landsat Collection 2 surface reflectance; exports the features and the yearly NBR |
| `results/` | Tables and the run log from `analysis.py` (point level only for the public MapBiomas points; Cropin QC only as aggregates) |
| `figures/` | Figures from `analysis.py` |

## Pipeline (`scripts/`)
| Step | Script | Output, in the data folder |
|---|---|---|
| 1 | `filter_index.py <index.csv>` | Landsat scenes over the three states from Google's public scene index |
| 2 | `build_points.py --mb-shp <points.shp> --qc <qc.geojson> --index <index.csv> --out <data>` | `points.csv`: the 1,831 MapBiomas validation points of the team's table + 433 Cropin QC polygons, with Landsat path/rows |
| 3 | `list_scenes.py <index.csv> <data>` | `scenes_selected.csv`: ~6 low-cloud scenes per path/row and year, listed from the bucket itself (the index misses 1997 and Landsat 8 in 2018) |
| 4 | `extract_landsat.py <data> <workers>` | `landsat_parts/`: cloud-masked top-of-atmosphere red/NIR/SWIR2 at every point for every scene (~30 min) |
| 5 | `sample_hansen.py <data>` | `hansen_gfc.csv`: Hansen GFC tree cover 2000, gain, loss year |
| 6 | `build_annual_series.py <data>` | `series_*.csv.gz`: one NBR/NDVI value per point and year, sensors harmonised; `series_obs_counts.csv.gz`, `series_sensor_offsets.csv` |
| 7 | `harvest_features.py <data>` | `features_main.csv.gz`, `features_grid.csv.gz`, `calibration_levels.csv` |
| 8 | `analysis.py <data> <this folder>` | `results/`, `figures/`; per-polygon QC tables go to `<data>/private_results/` |
| – | `features_from_gee_export.py <val.csv> [<qc.csv>]` | The same features from the Earth Engine export, to compare real LandTrendr with this analysis |

- Shared code is in `harvest_lib.py`.
- Requirements: those in `../Preliminary Model Exploration/requirements.txt`, plus `geopandas`, `rasterio`, `pyproj`,
  `requests` and `statsmodels`.
- Steps 1–5 only need access to `storage.googleapis.com` (public buckets, no Earth Engine account).
- Steps 6–8 take about 15 minutes on 4 CPUs.

## Data handling
- Inputs and intermediate files live in `CROPIN-MIT-DATA/harvest_cycles/`, which is git-ignored.
- Cropin's QC polygons are confidential, so nothing per-polygon is committed.
- All harvest and Hansen features use data up to 31 Dec 2020 only (Hansen losses after 2020 are ignored). The labels are
  the 2022 MapBiomas reference classes, which differ from the 2020 classes at only 2 points.

## Using the features in the team's model
```python
import pandas as pd
team = pd.read_csv("data_exploration/cropin_model_features_2022.csv")
hc = pd.read_csv("harvest_cycles/results/val_harvest_features_primary.csv")
df = team.merge(hc.drop(columns=["label", "group"]), on="pt_id")
```
The file holds the harvest features, the continuous NBR-history summaries (`s_*`; `s_std` and `s_min` are the
strongest) and Hansen, all from data up to 2020. `first_plantation_year` comes from the MapBiomas reference history:
use it to stratify, never as a feature.
