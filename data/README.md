# Data

> **The GitHub repo is public. No sponsor (Cropin) data is committed.** Every data folder
> is gitignored except its README. To rerun the notebooks, get the files from Meera or the
> team's shared drive and put them in the paths below.

| Folder | Contents | Source | In git? |
|---|---|---|---|
| `raw/cropin_qc/` | `cropin_brazil_qc_forest_plantation_1.geojson`: 433 QC polygons, property `qc_label` (310 plantation, 123 forest) | Cropin (sponsor) | No |
| `raw/mapbiomas_validation/` | MapBiomas 85k validation points shapefile (209 MB `.dbf`) | MapBiomas, shared by Deepak (Cropin) | No (too large) |
| `gee_assets/` | `cropin_qc.zip` (QC polygons with an added `id` 0–432) and `mapbiomas_ref_south.zip` (2,290 southern reference points), as uploaded to Earth Engine | Built by us | No |
| `gee_exports/` | The four CSVs exported from Earth Engine (described below) | `gee/02_*.js`, `gee/03_*.js` | No |
| `processed/` | `qc_features.csv` (notebook 01), `mapbiomas_ref_south.{gpkg,csv}` (`src/build_mapbiomas_ref_south.py`) | Derived | No |

Earth Engine assets, project `cropin-510901`:
- `projects/cropin-510901/assets/cropin_qc`
- `projects/cropin-510901/assets/mapbiomas_ref_south`

## What the QC polygons are (and why they are biased)
Each polygon is a place where **Cropin's forest layer said "forest"** and a **clearing was
detected in 2026**. Cropin staff then checked historical imagery by hand:
- `plantation` means a routine plantation harvest, i.e. a **false alarm**;
- `forest` means **real deforestation**.

So the sample is pre-filtered: every case was already called forest and was already cleared.
It cannot estimate accuracy in general. The polygons are tiny (median ≈ 270 m², about 3
Sentinel-2 pixels and smaller than one 30 m MapBiomas pixel).

## Data dictionary

All per-polygon CSVs have one row per QC polygon (433 rows), joined on `id`. "Polygon mean"
means Earth Engine's area-weighted mean of the pixels touching the polygon.

### `gee_exports/qc_error_analysis_A.csv` (from `gee/02_qc_error_analysis_export.js`, task A)
| Column | Unit | Definition |
|---|---|---|
| `id` | – | Polygon id 0–432 (added when converting the geojson to the GEE asset) |
| `qc_label` | – | `plantation` (false alarm) or `forest` (real deforestation), Cropin's hand label |
| `area_m2` | m² | Polygon area (geodesic, max error 1 m) |
| `lon`, `lat` | degrees (WGS84) | Polygon centroid |
| `height_mean` | m | Polygon mean of ETH Global Canopy Height 2020 (10 m), `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` |
| `height_sd_30m` | m | Std dev of ETH height in a **30 m-radius circle around the centroid**, at 10 m. ⚠️ Not inside the polygon, so it includes the surroundings |
| `nf_prob` | 0–1 | Polygon mean of Google "Natural Forests of the World" 2020 probability (raw `B0` / 250) |
| `ring_forest_frac` | 0–1 | Share of MapBiomas C9 2020 class 3 (forest) pixels in a 30–150 m ring around the polygon |
| `ring_plant_frac` | 0–1 | Same for class 9 (plantation) |
| `mb_1985` … `mb_2023` | class code | **Mode** of the MapBiomas Collection 9 (30 m) class inside the polygon each year. Polygons are sub-pixel, so this is effectively the overlapping pixel's class. Codes in the data: 0 no data, 3 forest formation, 9 forest plantation, 12 grassland, 15 pasture, 21 agriculture/pasture mosaic, 33 river/lake, 39 soybean, 40 rice, 41 other temporary crops |

### `gee_exports/qc_error_analysis_B_ndvi.csv` (from `gee/02_*.js`, task B)
Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`), scenes with `CLOUDY_PIXEL_PERCENTAGE < 60`,
pixels kept where SCL ∈ {4 vegetation, 5 bare soil}. NDVI = (B8 − B4)/(B8 + B4).
| Column | Unit | Definition |
|---|---|---|
| `id`, `qc_label` | – | As above |
| `ndvi_med_YYYY` | NDVI (−1…1) | Per-pixel median NDVI over calendar year YYYY, then polygon mean (2017–2025) |
| `ndvi_p10_YYYY` | NDVI | Per-pixel 10th percentile over the year (catches harvest dips), then polygon mean |
| `n_obs_YYYY` | count | Clear observations per pixel in the year, polygon mean. Medians are 5 (2017) and 10 (2018), and 25–37 from 2019 on. **Use 2019 onward for NDVI.** |

### `gee_exports/qc_cropin_layers.csv` (from `gee/03_cropin_layers_and_SAG_qc_export.js`, task C)
Cropin's forest-layer recipe (`gee/cropin/cropin_forest_layer.js`), evaluated for 2020 at 10 m.
`cf_*` columns are **polygon means of 0/1 pixel layers**, i.e. the share of the polygon that passes.
| Column | Unit | Definition |
|---|---|---|
| `id`, `qc_label` | – | As above |
| `cf_score` | 0–1 | Ensemble = (2·Hansen + 2·JRC + 3·MapBiomas C10 30 m forest + 3·TMF)/10, polygon mean |
| `cf_hansen` | share | Hansen v1.12 `treecover2000 ≥ 30` and no loss 2001–2020 |
| `cf_jrc` | share | JRC GFC2020 V2 > 0.5 |
| `cf_mb30c10_forest` | share | MapBiomas Collection 10 (30 m) 2020 class 3 |
| `cf_tmf` | share | JRC TMF v1_2024 `Dec2020` ≤ 2 or = 4 |
| `cf_mb10_plant_any` | share | MapBiomas 10 m Collection 2 **or** S2-beta 2020 class 9 (the plantation exclusion mask) |
| `cf_mb10beta_forest` | share | MapBiomas 10 m S2-beta 2020 class 3 (required by the final layer) |
| `cf_meta_ge4` | share | Meta 1 m canopy height ≥ 4 m (sampled at 10 m) |
| `cf_forest_final` | share | Final Cropin forest: score ≥ 0.75 AND not 10 m plantation AND 10 m beta forest AND Meta ≥ 4 m |
| `mb10c2_2016` … `mb10c2_2023` | class code | Mode of MapBiomas **10 m Collection 2** class inside the polygon |
| `mb10beta_2016` … `mb10beta_2022` | class code | Mode of MapBiomas **10 m S2-beta** class |
| `mb30c10_1985` … `mb30c10_2024` | class code | Mode of MapBiomas **Collection 10 (30 m)** class |
| `mean`, `stdDev`, `p10`, `p90` | m | Meta 1 m canopy height **inside the polygon** at 1 m: mean, std dev, 10th/90th percentile. Generic names come from the GEE reducer; notebook 02 renames them `meta_*` |
| `count` | pixels | Number of 1 m Meta pixels in the polygon |
| `system:index`, `.geo` | – | GEE export artefacts (empty geometry); dropped |

### `gee_exports/qc_sag_timeseries.csv` (from `gee/03_*.js`, task D)
One row per (polygon, Sentinel-2 image) with at least one valid pixel, 2019-01-01 to 2026-01-31,
all months (97,201 rows). Scenes with `CLOUDY_PIXEL_PERCENTAGE < 90`; Cloud Score+ `cs ≥ 0.6`;
pixels kept only if B2, B3 and B4 reflectance are all ≤ 0.05 (dark vegetation, Cropin's rule).
| Column | Unit | Definition |
|---|---|---|
| `id`, `qc_label` | – | As above |
| `date` | YYYY-MM-DD | Image date |
| `tile` | – | Sentinel-2 MGRS tile. Polygons in tile overlaps appear **twice on the same date** (10,837 duplicate id-date pairs) |
| `mean` | degrees | SAG (Spectral Angle at Green): angle at the green band between vectors to blue and red, polygon mean of valid pixels. Higher = flatter visible spectrum = darker green |
| `count` | pixels | Number of valid 10 m pixels used |

### `processed/qc_features.csv` (written by `notebooks/01_qc_error_analysis.ipynb`)
A merged with B plus the derived features defined in notebook 01, section b.
