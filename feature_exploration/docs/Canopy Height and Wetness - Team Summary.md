# Canopy Height and Wetness – Team Summary

*Prepared by Nathan Chu (with Claude) · 12 November 2026 · Covers all canopy-height and wetness work in Feature Exploration, Phases 0–7 · Analyses are our own (Google Earth Engine + Python) unless a source is given · **[inference]** = our interpretation, not tested · Full detail: `Feature Exploration - Summary of Findings`, `Wave 1–3 Reports`, `Phase 7 Protocol`, `Run Log.md` (all in `Feature Exploration/`)*

---

## 1. Bottom line

| Topic | Result | Used in the final model? |
|---|---|---|
| **Canopy height** | Mostly **not useful**. Plantations are *not* more uniform in height (H12 not supported). Height-for-age can't be tested (H13). Lidar is too sparse at Cropin's alerts. The one signal that carries over is that plantations are **taller on average** (ETH: AUC 0.68 on MapBiomas, 0.74 at Cropin's alerts), but it adds little once other features are included. | No (Meta height and height spread stay in the **baseline** only, because Cropin already uses Meta) |
| **Canopy wetness** (shortwave-infrared moisture) | **Useful.** Plantation canopies look "wetter": higher NDMI and darker in shortwave infrared. Summer NDMI is one of the four features in the final model. | **Yes:** summer NDMI is in Lean 4 (main) and P4 (challenger) |
| **Drought response** (H23) | **Not supported** (wrong direction, weak), and untestable at Cropin's alerts | No |
| **Site wetness** (near rivers, H32; surface temperature, H27) | **Not supported** (AUC ≤ 0.59) | No |

"AUC" is how well a feature ranks plantations above forest: 0.5 = coin flip, 1.0 = perfect. Numbers below are shown as **MapBiomas points / Cropin QC points**.

---

## 2. How everything was trained and evaluated (read this first)

Every feature in this document went through the same pipeline. The choices below explain why some data was used for training and other data only for checking.

### 2.1 Training data: MapBiomas validation points

| Choice | Why |
|---|---|
| **MapBiomas validation points** in PR/SC/RS ([MapBiomas, Validation points](https://brasil.mapbiomas.org/en/?p=951)) | The only large, free, expert-labelled sample of forest vs plantation in our states: 3 interpreters per point, labelled every year |
| Keep only points labelled **forest (class 3) or plantation (class 9) in every year 2013–2022** | A stable label means the point really is that class, not something changing mid-period. Both classes use the same 10-year window so neither is "cleaner" than the other. |
| Drop points where only **1 of 3 interpreters agreed** from training (kept in evaluation) | Low-agreement labels are often wrong (47% of them are on borders). Training on them teaches the model noise, but excluding them from evaluation would hide hard cases. |
| Keep only points with **tree cover** (summer NDVI > 0.6) | Our layer only classifies pixels with trees, so recently harvested points would be the wrong question |
| Result: **1,700 training points** (1,365 forest, 335 plantation) | |

### 2.2 Evaluation data and why it is separate

| Set | Used for | Why it matters |
|---|---|---|
| **Spatially blocked cross-validation** on MapBiomas points (0.5° blocks, ~50 km) | Development scores | Nearby points look alike. If a neighbour of a test point is in training, scores look better than they are. Blocking keeps whole areas out of training. |
| **Cropin QC points, development half** (58 forest, 149 plantation) | Checking that results carry over to real alerts. **Never used for training.** | These are the cases Cropin actually cares about: small patches where Cropin's layer said "forest" and a clearing was detected. But they are a **biased sample** (mostly plantation, only at alerts) and small. |
| **Sealed sets:** MapBiomas "lockbox" (20% of blocks) and the **other half of the QC points** | One final test (Phase 7), opened once, after the model and thresholds were locked in writing | Every choice we made while looking at development data makes those data slightly "used up". Only data we never looked at gives an honest final score. |

### 2.3 Rules that protect the results

| Rule | Why |
|---|---|
| **As-of rule:** features only from before each point's clearing (MapBiomas: up to 2022; QC: up to the year before the estimated clearing) | After clearing the land is bare. Features from then would "see the answer". |
| **Forest error first:** we always fix the share of **real forest called plantation** (e.g., 5%) and then compare how many plantations each model catches (**recall**) | Missing real deforestation is the worst error (Cropin: tolerance "technically zero"). This is *our* dangerous error, not Cropin's "false positive" (a harvest flagged as deforestation). |
| **Decision rule for each feature:** significant after correction for many tests; AUC lower confidence bound > 0.60; same direction in all three states and as the hypothesis predicted; adds ≥ 5 points of recall over the baseline | Stops us from keeping features that look good by chance or only in one place |
| **Baseline** = Cropin's SAG + Meta canopy height + height spread | New features must beat what Cropin already uses |
| **Directions written down before seeing the data** (Waves 2 and 3) | Stops us from explaining away results after the fact |
| **Independent reviewer** (a separate AI agent given code and data but not our conclusions) after every wave | Caught several real problems (§3.4, §4.4). Every reported number was reproduced. |

### 2.4 How to read the results

- **MapBiomas vs Cropin points can disagree.** Many features score well on MapBiomas but poorly at Cropin's alerts, because the plantations at alerts look more like forest. A feature only matters for Cropin if it works on **both**.
- **Small samples.** The QC halves have about 58–59 forest points each, so an AUC gap of 0.02–0.04 cannot be distinguished from noise, and a forest error of 3/59 has a 95% CI of about 2–14%.
- **Cropin has no labels for forest it misses.** No number here measures the overall forest error, only the error at places Cropin already flagged.
- **Final test outcome (all features combined):** a significant gain on held-out MapBiomas points (AUC 0.84 → 0.92), but only a small, non-significant gain at Cropin's held-back alerts (0.81 → 0.83).

---

## 3. Canopy height

### 3.1 Hypotheses and reasoning

| # | Hypothesis | Reasoning |
|---|---|---|
| **H12** | Plantations have **more uniform height** | Plantations are even-aged, one or two species, evenly spaced, which is part of the EUDR plantation definition ([Regulation (EU) 2023/1115, Art. 2](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A02023R1115-20251226)). Natural forest mixes ages and species, so its height should vary more over short distances. (Detail: `Deep Dives/Deep Dive - H12 Canopy Height Uniformity`.) |
| **H13** | Plantations are **taller for their age** | Eucalyptus and pine are bred and managed for fast growth, so for the same years since the last clearing they should be taller than natural regrowth |
| **H14** | Plantations have **one canopy layer**; natural forest has several | Managed stands have little understorey; natural forest has emergent trees, a canopy, and sub-canopy |
| **H17** | Height **level** differs by forest type | E.g., the araucaria canopy exceeds 30 m (source in the master list, H17); plantations are harvested before they get very tall |

### 3.2 Data used

| Dataset | What it is | How we used it |
|---|---|---|
| **Meta canopy height** (1 m, global) | Height predicted from very-high-resolution imagery by a deep-learning model | Mean height and height spread within 15 m of each point. Part of the **baseline**, because Cropin's forest layer already uses it. |
| **ETH canopy height 2020** (10 m) | Height predicted from Sentinel-2 by a deep-learning model trained on GEDI lidar (Lang et al. 2022, [arXiv:2204.08322](https://arxiv.org/abs/2204.08322)) | Height at the point; mean and spread within 15 m and 50 m; the model's own uncertainty |
| **GEDI lidar** (from the space station) | Real laser measurements of canopy height, cover, and vertical structure in ~25 m footprints ([GEE catalog, L2A](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY); [L2B](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_B_002_MONTHLY)) | Quality-filtered shots within 100 m and 250 m, in the 3 years before each point's cutoff: top height (rh98), its spread, foliage height diversity (layers, H14), cover |

### 3.3 Implementation

- Exported from Google Earth Engine at every point (Phase 0 batch for Meta; Wave 3 batch for ETH and GEDI), then analysed in Python.
- **Height spread** = standard deviation of height within a 15 m or 50 m circle (H12). We also used the coefficient of variation (spread ÷ mean) to remove the effect of tall stands simply varying more in absolute metres.
- **Height-for-age** (H13) = ETH height ÷ years since the last Landsat-detected disturbance.

### 3.4 Problems we had to work around

| Problem | What we did |
|---|---|
| **The three height products disagree.** Median height of MapBiomas forest vs plantation: Meta 9.8 m vs 8.8 m (forest taller); ETH 21 m vs 26 m (plantations taller). | We report each separately. Meta is known to underestimate tall canopies (in São Paulo, mean error ~5 m, saturating above ~30 m; source in master list, H12), so we trust ETH more for height level **[inference]**. |
| **GEDI is sparse.** Only 44% of points have a shot within 100 m, and almost none of Cropin's points, because GEDI was switched off from March 2023 to April 2024. | Added a 250 m radius (74% coverage on MapBiomas points). GEDI could not be judged at Cropin's alerts. |
| **GEDI export lost its main columns.** The first rows had no shots, and Earth Engine CSV exports keep only the first rows' columns. | Re-exported with missing values written as −9999. |
| **H13 can't be tested.** 86% of points have no disturbance in 25 years, so their age is unknown. The ratio mostly re-expressed "years since disturbance", and it doubled forest error on disturbed natural forest (10% → 20%). | The reviewer flagged it; we **dropped** the feature and recorded H13 as "not testable". |
| **ETH model uncertainty** separates the classes well (AUC 0.80) but tracks height, region, and cloudiness. | Kept only as an optional add-on, labelled "uninterpretable". |
| **Dates.** ETH is a 2020 snapshot; Meta's image dates are unknown. | ETH 2020 is before all clearings (as-of rule OK). Meta is tagged "date unknown". |

### 3.5 Results

| Hypothesis | Feature | AUC (MapBiomas / QC) | Verdict |
|---|---|---|---|
| H12 uniform height | ETH spread within 50 m | 0.67 / 0.67, but **plantations vary more** | **Not supported** (opposite direction) |
| H12 | Meta spread within 15 m | 0.54 / 0.71, direction **flips** between sets | Not supported |
| H12 | GEDI height spread | ≤ 0.56 | Not supported |
| H13 height for age | Height ÷ years since disturbance | — | **Not testable** |
| H14 one canopy layer | GEDI foliage height diversity | ≤ 0.55 | Not supported |
| H17 height level | ETH height | **0.68 / 0.74** (plantations taller) | Plantations taller on average |
| H17 | Meta mean height | 0.52 / 0.52 | No signal |

**In models (development data):**

| Model | MapBiomas AUC | QC AUC | QC forest error |
|---|---|---|---|
| Baseline (includes Meta height and spread) | 0.84 | 0.78 | 17% |
| Baseline + ETH height + ETH uncertainty | 0.87 | 0.83 | 16% |
| 10-feature model + ETH height + uncertainty | 0.94 | 0.89 | 8.6% (vs 6.9% without) |

**Why plantations aren't more uniform** **[inference]:** at 10 m, a pixel's neighbourhood often mixes a plantation with roads, firebreaks, and harvested blocks of different ages, and plantation canopies are tall and sharp-edged. All of that adds height variation that has nothing to do with "even age" inside the stand.

---

## 4. Wetness

"Wetness" covers three different ideas. Only the first one helped.

### 4.1 Hypotheses and reasoning

| Idea | Hypothesis | Reasoning |
|---|---|---|
| **Canopy water content** | Plantation and natural canopies differ in shortwave-infrared (SWIR) moisture signals (exploratory; related to H20, pine vs broadleaf spectra) | Leaf water absorbs SWIR light (~1.6 and ~2.2 µm), while near-infrared (NIR) light is reflected by leaf structure. Moisture indices compare the two: NDMI = (NIR − SWIR1) / (NIR + SWIR1) (Gao 1996, [doi:10.1016/S0034-4257(96)00067-3](https://doi.org/10.1016/S0034-4257(96)00067-3)). Dense, uniform, fast-growing plantation canopies might hold more water per pixel and show less soil or dry litter. |
| **Drought response** (H23) | Eucalyptus **stays greener in drought** than natural forest | Eucalyptus has deep roots and high water use (an unverified lead in the master list), so it might suffer less in dry months |
| **Site wetness** (H32, H27) | Natural forest sits closer to rivers (H32), and plantations are warmer or cooler (H27) | Brazilian law protects riparian strips (APPs) at least 30 m wide along small rivers ([Lei 12.651/2012, Art. 4](https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2012/lei/l12651.htm)), where native forest must remain. Surface temperature reflects evapotranspiration. |

### 4.2 Data used

| Data | Use |
|---|---|
| Sentinel-2 surface reflectance, cloud-masked with Cloud Score+ (≥ 0.6) | Summer (Dec–Mar) and Oct–Jan composites: SWIR bands B11 and B12, NDMI, NBR (NIR vs SWIR2), NDWI (a green-vs-NIR open-water index), SWIR ratio. Monthly composites for the seasonal cycle of NDMI. |
| CHIRPS rainfall and its 1991–2020 monthly average ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)) | Define drought months: 2-month rain < 60% of normal |
| Landsat 8/9 surface temperature | Summer and winter temperature (H27) |
| JRC Global Surface Water ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater)) and MERIT Hydro height above drainage ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1)) | Distance to water and height above the nearest stream (H32) |

### 4.3 Implementation

- Moisture indices were computed from the same composites as all other spectral features (Wave 1).
- **Drought response:** for each point, a yearly curve (harmonic fit) was fitted to monthly NDVI and NDMI over 3 years. In drought months we measured how far the observed value fell below that curve, in standard units.
- Distances to water and temperatures were sampled at each point (Waves 2–3).

### 4.4 Problems we had to work around

| Problem | What we did |
|---|---|
| **MapBiomas and QC windows differ in weather.** MapBiomas features come from 2020–22, which were dry (75–97% of normal rain, median 8 drought months per point). QC features come from 2023–25, which were wet (100–114%, median 1 drought month). | Drought response could only be tested on MapBiomas points. We flag that moisture features may shift between dry and wet years. |
| **Single-year features are unstable** (Phase 5). Using one year instead of three moved forest error at Cropin's points from ~5% to 23% in the 2024 flood year. | Recommend **3-year windows** for any moisture-related feature in production |
| **Moisture overlaps with darkness.** Plantations are darker in SWIR *and* in red-edge/NIR, which SAG already captures. | The decision rule requires each feature to add recall **on top of** the baseline with SAG. Raw SWIR bands failed that test; NDMI passed. |
| **Surface-temperature export lost the 1 km comparison** (same CSV column issue as GEDI). | Point temperature only was tested |
| **"Wet" indices mean different things.** NDWI (green vs NIR) measures open water, not leaf water. | Treated separately; it was weak and flipped direction |

### 4.5 Results

| Feature | Direction | AUC (MapBiomas / QC) | Verdict |
|---|---|---|---|
| **Summer NDMI** | **higher = plantation** | **0.68 / 0.70** | **Passes all rules; in the final model** |
| Summer NBR (NIR vs SWIR2) | higher = plantation | 0.64 / 0.67 | Passes; overlaps with NDMI, so not selected |
| SWIR1 (B11), summer / Oct–Jan | lower = plantation | 0.71 / 0.73 and 0.69 / 0.75 | Strong on both sets, but adds < 5 points of recall over SAG |
| SWIR2 (B12), summer | lower = plantation | 0.68 / 0.70 | Same as B11 |
| NDMI 3-year mean and seasonal swing | higher = plantation | 0.65 / 0.77 and 0.68 / 0.72 | Good at alerts, but not selected over summer NDMI |
| Oct–Jan NDMI | higher = plantation | 0.59 / 0.60 | Weak |
| NDWI (open-water index) | flips | 0.53 / 0.71 (summer); 0.51 / 0.29 (Oct–Jan) | Not useful |
| H23 drought response (NDVI, NDMI) | expected: stays greener | 0.54, **opposite direction** | **Not supported**; untestable on QC |
| H27 surface temperature | — | ≤ 0.59 | Not supported |
| H32 distance to water / height above drainage | expected: farther from water | ≤ 0.52, partly opposite | **Not supported** |

**What drives NDMI:** in summer, MapBiomas plantations have a median NDMI of 0.40 vs 0.33 for forest. That is mostly because plantations are much darker in SWIR (B11 reflectance 0.13 vs 0.16) while their NIR is only slightly lower. **[inference]** A dense, closed, evergreen canopy with more leaf water and less visible dry litter or soil fits this.

**In the final model:** summer NDMI is one of the four Lean 4 features. Its standardized coefficient is about +1.1, the second-largest after spring SAG (+1.5), and it points the same way as when tested alone.

| Lean 4 result | MapBiomas | Cropin QC |
|---|---|---|
| Development (blocked CV) | AUC 0.91 vs 0.84 baseline | 0.85 vs 0.78 |
| **Final sealed test** | **0.92 vs 0.84 (significant)** | **0.83 vs 0.81 (not significant)** |

---

## 5. What this means for the team

1. **Drop canopy height as a main lead,** at least at 10 m. The idea of uniform plantation height doesn't hold up, and lidar can't reach most of Cropin's alerts. Height *level* (plantations taller) is a weak supporting signal.
2. **Keep canopy moisture (NDMI).** It's cheap (one index from Sentinel-2), interpretable, and points the same way on both MapBiomas and Cropin's points. It should be computed over **multi-year** windows because of the dry/wet-year shifts.
3. **Drought and riparian ideas didn't work here.** Drought response needs a dataset with real drought years at alert locations to test properly.
4. **Region:** all of this is for southern Brazil. Moisture contrasts depend on climate and species, so they must be retested before any global layer.
5. **The main open question** is not height or wetness specifically. It's why plantations at Cropin's alerts look more like forest than MapBiomas plantations do. More labelled alerts from Cropin are needed to answer it (a request has been drafted).

## 6. Files

All in `Cropin - A Lab/Feature Exploration/` (scripts also in the team GitHub upload folder):

| What | Where |
|---|---|
| Height and wetness features | `scripts/p0_batch0_template.js` (Meta height, Sentinel-2 composites), `w1_features.py` (moisture indices), `w2_features.py` (monthly NDMI, drought, temperature), `w3_batch3_template.js`, `w3b_gedi_template.js`, `w3_features.py` (ETH, GEDI, water distance) |
| Screens (every feature's AUC and decision-rule result) | `results/wave1/w1_screen.csv`, `results/wave2/w2_screen.csv`, `results/wave3/w3_screen.csv` |
| Final test | `Phase 7 Protocol`, `results/phase7/` |
| Hypothesis verdicts | *Task 1.2 – Hypotheses Master List* ("Test results so far") |
