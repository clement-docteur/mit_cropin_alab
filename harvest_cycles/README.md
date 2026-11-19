# Harvest cycles

Does a pixel's Landsat history of clear-cuts and regrowth (1985–2020, before the EUDR cutoff) tell a plantation from
natural forest? Findings: `harvest_cycles_findings.ipynb`. Project context, decisions and status: `HANDOFF.md`.

| File | What it is |
|---|---|
| `harvest_cycles_findings.ipynb` | The analysis and all findings, computed from `public_data/harvest_cycles/` |
| `preprocess_landsat.py` | Step 1: points → Landsat scenes → cloud-masked dated observations → yearly NBR/NDVI series |
| `build_harvest_cycles_features.py` | Step 2: yearly series + dated observations → one row of features per point; also imported by the notebook |
| `config.py` | Every threshold and setting of both steps; the defaults reproduce the analysis |
| `gee/cropin_layers_at_points.js` | Earth Engine export of Cropin's forest layer and SAG observations at the points (notebook section 8) |
| `HANDOFF.md` | Everything needed to pick up the work |

## Features for new points

Two steps, so the slow download runs once and the features can be recomputed with other settings in seconds.

```bash
python harvest_cycles/preprocess_landsat.py --points my_points.csv --out CROPIN-MIT-DATA/my_run --workers 32
python harvest_cycles/build_harvest_cycles_features.py --data CROPIN-MIT-DATA/my_run --points my_points.csv \
    --out CROPIN-MIT-DATA/my_run/harvest_cycles_features.csv [--label-column label]
```

- **Points file:** one row per point with `pid` (unique id, kept as text), `lon`, `lat` (WGS84 decimal degrees).
  To fill the output's `label_mapbiomas` column, add the MapBiomas class (1 = plantation, 0 = natural forest, empty if
  unknown) and name its column with `--label-column` (it is `label` in `public_data/harvest_cycles/points.csv`).
  Cropin's QC labels are confidential: the script refuses a label on any point whose `dataset` column says `qc`.
- **Where to write:** inside the git-ignored `CROPIN-MIT-DATA/` folder. The raw downloads (`cache/`, about 100 MB per
  2,000 points) must never be committed.
- **Cost:** the preprocessing reads single pixels over HTTP. Its cost follows the Landsat path/rows the points fall in
  (about 215 scenes each) more than the number of points: about 2 minutes for 12 points in 6 path/rows, 25 minutes
  for 2,264 points in 38 path/rows (64 workers, fast connection), plus a minute to stream the scene index once. It is
  resumable: after an interruption (Ctrl+C is fine), run the same command again. A folder is tied to its points and
  download settings (`manifest.json`); use a new `--out` for other points. Points outside every Landsat scene (for
  example with lon and lat swapped) are refused. Scenes that cannot be read are listed in `failed_scenes.csv` and left
  out (the run stops instead if they are more than 1% of the scenes); a re-run retries them.
- **Output:** `harvest_cycles_features.csv` (`pid`, `label_mapbiomas`, then the features below; the series features
  are empty where a point has fewer than 8 years of data or is not in the data, the seasonal ones have their own
  minimum) and `harvest_cycles_features.csv.settings.json` (command line, settings, input checksums, software
  versions, git commit).
- **Requirements:** Python 3.10+, numpy, pandas, requests, rasterio and pyproj (preprocessing only).

The features for the 2,264 analysis points are in `public_data/harvest_cycles/harvest_cycles_features.csv`.

### The features

| Column | Meaning |
|---|---|
| `n_cycles` | Number of cut-and-regrow cycles: cuts that grow back half of the fall within 10 years |
| `regrow_speed` | 1 / years for the quickest cycle to grow back 80% of its fall, counting at least 1 year (so at most 1; 0 if none does) |
| `forest_since_1985` | 1 if no cut and NBR never below the closed-canopy level in any year 1985–2020 |
| `abrupt_start` | A new stand on open ground: 1 if the first closed-canopy year preceded by at least 3 open-land years among the 8 yearly values ending at the last open year comes within 5 years of that last open year |
| `n_cuts` | Number of cuts: NBR falls of at least 0.20 within at most 3 years between fitted segment ends |
| `years_since_last_cut` | 2020 minus the year of the latest cut (40 if no cut) |
| `max_drop` | Largest cut, in NBR (0 if no cut) |
| `s_frac_below_forest` | Share of years with NBR below the closed-canopy level |
| `s_min` | Lowest yearly NBR |
| `s_max_drop3` | Largest fall of yearly NBR from the highest of the previous 3 values (years with data) |
| `s_std` | Variability: standard deviation of the yearly NBR (best single feature; chosen after first results) |
| `has_cut`, `has_cycle`, `two_plus_cycles` | 1 if at least one cut, at least one cycle, at least two cycles |
| `fast_regrowth` | 1 if a cycle grows back 80% of its fall within 4 years |
| `season_diff_nbr_1985_2020`, `season_diff_nbr_2013_2020`, `season_diff_ndvi_1985_2020`, `season_diff_ndvi_2013_2020` | Seasonality: median over the years of (wet-season median − dry-season median), dry season Apr–Sep; needs at least 5 (1985–2020) or 3 (2013–2020) years with both seasons. Natural forest swings more than plantations. |

The first four are the core set (H4); with the next six they form H-all, the analysis' main model features. Yearly NBR
values are first cleaned of one-year spikes; cuts come from a LandTrendr-like straight-line fit (up to 18 segments,
number chosen by BIC). The closed-canopy (0.584) and open-land (0.405) levels were calibrated from Hansen tree cover,
never from labels. `forest_since_1985` and `s_max_drop3` keep their names when `--set` changes first_year or fall_window_years (the
seasonal column names follow seasonal_indices and seasonal_windows); the `.settings.json` file records what was used.

### Changing the settings

All thresholds are in `config.py`, with comments. Change a default there, or for one run only:

```bash
python harvest_cycles/build_harvest_cycles_features.py ... --set drop=0.15 --set max_segments=12
```

`--set` values are checked (type and range) before anything runs. Other things to know:

- **NBR levels.** `--calibrate-levels hansen_gfc.csv` recalibrates the two levels on the batch's own pixels
  (`preprocess_landsat.py --hansen` writes that file; at least 30 closed-canopy and 30 open-land pixels are needed).
  Levels then differ from batch to batch, so features scored with a model trained on the default levels should keep
  the defaults; recalibrate only for a new imagery source or region, and retrain. The calibration settings in
  `config.py` act only through this option.
- **Sensor offsets.** New runs apply the Landsat 8 and Landsat 4/5 offsets measured at the 2,264 analysis points.
  `preprocess_landsat.py --set estimate_offsets=true` measures them on the new points instead, which is noisy for
  small batches (12 points moved the Landsat 8 NBR offset by 0.014).
- **Tuning.** Run the feature step over a grid of settings and choose with nested spatial cross-validation: tuning on
  the MapBiomas points alone risks fitting their labels, which are drawn from Landsat too.

## Before production

- **Move to Google Earth Engine with Landsat Collection 2.** This pipeline reads Landsat **Collection 1** from Google's
  public bucket. USGS has deprecated Collection 1; the bucket still serves it but may stop. Collection 2 also has
  better geolocation and surface reflectance. The switch needs every threshold re-validated (NBR levels, sensor
  offsets) and the features compared with these on the same points. Earth Engine also scales to whole states, which
  this point-by-point download does not.
- Tune the thresholds as above, and confirm the variability and seasonal features on new data (both were picked after
  looking at results).
- Small things left as they are: step 5 holds all observations in memory (about 0.2 MB per point, fine for thousands
  of points); the scripts import their settings module under the generic name `config`, which would clash only with
  another module of that name loaded first in the same Python session.
