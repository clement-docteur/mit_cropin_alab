# MapBiomas validation points (85,152 points, Brazil): not in git

**What:** the MapBiomas accuracy-assessment sample. Each point has an interpreted land-cover
class for every year 1985–2022 (`CLASS_yyyy`, MapBiomas codes), the number of the 3
interpreters who agreed (`COUNT_yyyy`), an edge flag (`BORDA_yyyy`), the state (`SIGLA`),
the biome (`BIOMA_250K`) and the sampling probability and weight (`PROB_AMOS`, `PESO_AMOS`).

**Where it came from:** shared with the team by Deepak Murugan (Cropin) in October 2026.
The original source is MapBiomas' public validation dataset.

**Why it isn't committed:** `mapbiomas_85k_points_validation.dbf` is 209 MB, over GitHub's
100 MB limit. This folder is gitignored except this README.

**Files expected here:** `mapbiomas_85k_points_validation.{shp,shx,dbf,prj,cpg,qmd}`

**⚠️ Gotcha:** the `LON` and `LAT` columns are **swapped** (LON holds latitude). Use the geometry.

**Derived subset:** `python src/build_mapbiomas_ref_south.py` rebuilds the 2,290-point
southern subset (PR/SC/RS, class 3 or 9 in 2020, stable 2016 = 2020 = 2022). It checks the
result against the Earth Engine asset `projects/cropin-510901/assets/mapbiomas_ref_south`.
