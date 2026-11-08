# Harvest cycles: plantation vs natural forest from the Landsat record (1985–2020)

Does a pixel's history of clear-cuts and regrowth tell a plantation from natural forest in Paraná, Santa Catarina and
Rio Grande do Sul, using only imagery up to the EUDR cutoff (end of 2020)?
Findings, significance tests, recommendations and caveats: **[`docs/summary.md`](docs/summary.md)**.

## Layout
| Path | What it is |
|---|---|
| `docs/summary.md` | The write-up |
| `scripts/` | Full pipeline in Python (steps below) |
| `gee/landtrendr_harvest_features.js` | Earth Engine version for production and the 10 m map. It uses the real LandTrendr on Landsat Collection 2 surface reflectance and exports the same features plus the yearly NBR |
| `results/` | Tables and the run log written by `analysis.py`: point-level only for the public MapBiomas points, aggregates only for Cropin QC |
| `figures/` | Figures written by `analysis.py` |

## Pipeline (`scripts/`)
| Step | Script | Output, in the data folder |
|---|---|---|
| 1 | `filter_index.py <index.csv>` | Landsat scenes over the three states from Google's public index |
| 2 | `build_points.py --mb-shp <points.shp> --qc <qc.geojson> --index <index.csv> --out <data>` | `points.csv`: 1,831 MapBiomas validation points + 433 Cropin QC polygons (one representative point each), with Landsat path/rows |
| 3 | `list_scenes.py <index.csv> <data>` | `scenes_selected.csv`: ~6 low-cloud L1TP scenes per path/row and year, split dry/wet season. The bucket is listed directly, because the index misses 1997 and Landsat 8 in 2018 |
| 4 | `extract_landsat.py <data> <workers>` | `landsat_parts/`: cloud-masked TOA red/NIR/SWIR2 at every point for every scene, via HTTP range requests (~30 min) |
| 4b | `extract_landsat_window.py <data> <workers> 3` | `landsat_parts_win3/`: the same for the 3×3 window around each point (~35 min); only needed for the neighbourhood test |
| 5 | `sample_hansen.py <data>` | `hansen_gfc.csv`: Hansen GFC v1.12 tree cover 2000, gain and loss year (pixel and 3×3) |
| 6 | `build_annual_series.py <data>` | `series_<index>_<agg>.csv.gz`: one NBR/NDVI value per point and year. Sensors are harmonised and duplicate passes merged |
| 7 | `harvest_features.py <data>` | `features_main.csv.gz`, `features_grid.csv.gz`, `calibration_levels.csv` |
| 7b | `neighbourhood_features.py <data>` | `features_win3.csv.gz`: centre pixel vs 3×3 aggregates vs the 90 m mean series |
| 8 | `analysis.py <data> <this folder>` | `results/`, `figures/`; per-polygon QC tables go to `<data>/private_results/` |
| – | `features_from_gee_export.py <val.csv> [<qc.csv>]` | The same features from the Earth Engine export, to compare GEE LandTrendr with this analysis |

- `harvest_lib.py` holds the shared code: compositing, LandTrendr-style segmentation, the rule-based detector, and the
  four features plus continuous summaries.
- Requirements: the packages in `../Preliminary Model Exploration/requirements.txt`, plus `geopandas`, `rasterio`,
  `pyproj`, `requests` and `statsmodels`.
- Steps 1–5 only need internet access to `storage.googleapis.com` (public buckets; no Earth Engine account).
- A full rerun of steps 6–8 takes about 45 minutes on 4 CPUs.

## Definitions (primary configuration)
- **Yearly value:** median NBR of all clear observations in the calendar year.
- **Closed-canopy level:** 5th percentile of the 2000–2020 values at pixels Hansen calls ≥ 90% tree cover in 2000,
  with no gain and no loss 2001–2020.
- **Open-land level:** 25th percentile of the 1999–2001 values at pixels with ≤ 10% tree cover in 2000, kept at least
  half a cut below closed canopy.
- **Neither level uses the plantation/forest labels.**
- **Cut:** NBR falls ≥ 0.20 within ≤ 3 years. Found by LandTrendr-style segmentation (≤ 18 segments, BIC) or by
  simple rules on the yearly values.
- **Cycle:** a cut that regrows ≥ 50% of the fall within 10 years.
- **Forest since 1985:** no cut, and no yearly value below closed canopy.
- **Abrupt start:** a year at closed canopy with ≥ 3 open years in the 8 years before it, the last one at most 5
  years earlier.
- Only imagery up to 31 Dec 2020 enters any feature; Hansen losses after 2020 are ignored.

## Data handling
- Inputs and intermediate files live in `CROPIN-MIT-DATA/harvest_cycles/`, which is git-ignored.
- Cropin's QC polygons are confidential, so nothing per-polygon is committed.
- MapBiomas validation points are public. `results/val_harvest_features_primary.csv` keeps the team's `pt_id`, so it
  joins to `data_exploration/cropin_model_features_2022.csv`.

## Using the features in the team's model
```python
import pandas as pd
team = pd.read_csv("data_exploration/cropin_model_features_2022.csv")
hc = pd.read_csv("harvest_cycles/results/val_harvest_features_primary.csv")
df = team.merge(hc.drop(columns=["label", "group"]), on="pt_id")
```
- The file holds:
  - the four harvest features
  - the continuous NBR-history summaries (`s_*`; `s_std` and `s_max_drop3` are the strongest)
  - Hansen
- Everything is computed from data up to 2020.
- `first_plantation_year` comes from the MapBiomas reference history. Use it only to stratify, never as a feature.
