# Feature Exploration – Wave 2 Report (Families B seasonal, F radar)

*Prepared by Nathan Chu (with Claude) · 6 November 2026 · Analyses are our own (Google Earth Engine + Python) unless a source is given · **[inference]** = our interpretation, not tested · Plan: `00 Workflow Plan.md` (v4.1) · Every number traces to `Run Log.md` and `results/wave2/` · Read with the Wave 1 Report*

---

## 1. Summary

- **Seasonal features (Family B) are the first whose threshold carries over to Cropin's points.** A model tuned to 5% forest error on MapBiomas gives **5.2% (3 of 58)** forest error on Cropin's QC points, against 17.2% for the baseline. In a paired comparison on the same 58 forest points, the seasonal model fixed 7 baseline errors and made none of its own (exact McNemar p = 0.016).
- **Cropin's SAG signal works better in spring.** Measured over September–November, SAG reaches AUC 0.884 on MapBiomas, against 0.831 for the December–January window Cropin uses. On QC the two are equal (0.810 vs 0.809).
- **When the seasonal cycle peaks matters.** The phase of the yearly cycle in SAG, red-edge, and NDVI separates the classes. It holds within every state and is not a stand-in for latitude.
- **Radar adds little.** The Sentinel-1 VH−VV ratio passes the screen but adds nothing on QC. ALOS L-band radar and surface temperature are weak.
- **Two pre-registered hypotheses were not supported:** H21 (plantations have a smaller seasonal swing) and H23 (plantations stay greener in drought).

"Forest error" always means **real forest called plantation** (our dangerous error), not Cropin's "false positive" (a plantation harvest flagged as deforestation).

## 2. Data and method (short)

Same points, splits, evaluation, and decision rule as Wave 1 (see Wave 1 Report §2). New inputs, each limited to the **3 calendar years ending at the point's cutoff** (MapBiomas 2020–2022; QC mostly 2023–2025):

| Input | Use |
|---|---|
| Sentinel-2 monthly medians (Cloud Score+ ≥ 0.6) | Harmonic fit per index (mean, amplitude, phase as cos/sin), SAG by season |
| CHIRPS monthly rainfall vs its 1991–2020 mean | Drought months: 2-month rain below 60% of normal (H23) |
| Landsat 8/9 surface temperature | Summer and winter medians (H27) |
| Sentinel-1 C-band (VV, VH) | Medians, ranges, and spread over time (H25) |
| ALOS-2 PALSAR-2 L-band (HH, HV) | Latest yearly mosaic before the cutoff (H26) |

The expected directions for H21 and H23 were written in `Run Log.md` before any Wave 2 data were seen.

SAG follows Cropin's formula, the angle at green between the blue–green and red–green vectors of the reflectance spectrum.

## 3. Results

### 3.1 Models

| Model | MapBiomas AUC | MapBiomas recall @1% / 2% / 5% | QC AUC [95% CI] | QC forest error at MapBiomas 5% threshold | QC recall |
|---|---|---|---|---|---|
| Baseline (SAG + height + height SD) | 0.841 | 19% / 27% / 42% | 0.782 [0.70–0.84] | 17.2% (10/58) | 64% |
| **B alone*** | 0.915 | 48% / 55% / 70% | **0.853** [0.80–0.90] | **5.2%** (3/58; Wilson CI 1.8–14.1%) | 56% |
| Baseline + B | 0.916 | 49% / 58% / 71% | 0.850 [0.79–0.89] | 6.9% (4/58) | 59% |
| Baseline + F (VH−VV, VV) | 0.868 | 33% / 46% / 59% | 0.775 [0.71–0.84] | 15.5% (9/58) | 57% |
| Wave 1 preview + B | 0.969 | 72% / 80% / 89% | **0.884** [0.84–0.92] | 13.8% (8/58); 6.9% with alert-like threshold | 75% (71%) |
| All families A–F (preview) | 0.969 | 76% / 82% / 89% | 0.871 [0.82–0.92] | 12.1% (7/58) | 71% |

\*SAG Sep–Nov, SAG Dec–Jan, SAG 3-year mean, and the seasonal phase of SAG, red-edge (NDRE), and SWIR.

Nested cross-validation gives the same MapBiomas AUCs (Baseline + B: 0.918), so selecting features on the same data did not inflate them.

### 3.2 SAG by season (single feature)

| Season | MapBiomas AUC | QC AUC |
|---|---|---|
| Dec–Jan (Cropin's window) | 0.831 | 0.809 |
| Mar–May | 0.709 | 0.728 |
| Jun–Aug | 0.714 | 0.713 |
| **Sep–Nov** | **0.884** | **0.810** |

**[inference]** Spring is when native trees flush new leaves, so native forest and plantations may differ most in colour then. We have not checked this against phenology studies.

### 3.3 Hypotheses tested (from *Task 1.2 – Hypotheses Master List*)

| Hypothesis | Expected | Result |
|---|---|---|
| H21: seasonal swing | Plantations swing less (evergreen) | **Not supported:** opposite direction, weak (AUC 0.58) |
| H22: timing of the seasonal cycle | No stated direction | **Supported:** NDVI phase AUC 0.73 on MapBiomas, 0.76 on QC; SAG phase 0.85 / 0.64 |
| H23: response to drought | Plantations stay greener | **Not supported:** opposite direction, weak (0.54). The QC window had almost no drought months (median 1), so QC can't test it |
| H25: Sentinel-1 C-band | No stated direction | **Partly supported:** VH−VV ratio 0.77 on MapBiomas, 0.67 on QC; adds nothing on QC once the baseline is included |
| H26: ALOS PALSAR L-band | No stated direction | **Not supported:** AUC ≤ 0.58 |
| H27: surface temperature | No stated direction | **Not supported:** AUC ≤ 0.59 |

### 3.4 Where errors happen (Baseline + B, MapBiomas)

| Group | Forest error |
|---|---|
| Interior | 4.4% |
| Border | 7.2% |
| 3 of 3 interpreters agree | 3.6% |
| 2 of 3 agree | 10.3% |

Unlike the history features in Wave 1, the seasonal features do not make border errors worse (baseline border error: 7.9%).

## 4. Checks and limits

- **Independent review:** a separate agent (not the Wave 1 reviewer) recomputed every model number to 4 decimals, and four seasonal features from the raw monthly data (largest difference 2e-11). It confirmed that every month used falls inside each point's as-of window.
  - **Not a location proxy:** the phase features barely correlate with latitude (|ρ| ≤ 0.10 for SAG and NDRE phase), and within-state AUCs are 0.88–0.94.
  - **Fixed:** one radar feature used 5 years instead of 3. After the fix and a re-run, no result changed.
- **Small QC sample:** 58 forest points in 30 blocks can't show "technically zero" forest error. The 5.2% has an upper bound of about 14%, and more given the clustering.
- **Different years:** MapBiomas uses 2020–22 (dry, 83% of normal rainfall) and QC uses 2023–25 (wet, 107%). SAG runs 1.5–4° higher on QC, and Sentinel-1 image counts fall from ~85 to ~35 after Sentinel-1B was lost in Dec 2021. Some MapBiomas–QC differences may be year effects, not forest differences.
- **Biased QC sample:** QC forest is only forest where Cropin already detected a clearing. **Cropin has no labels for forest it misses.**
- **Lost data:** the 1 km temperature means were lost in export, so only point temperature was tested.

## 5. What this means for Cropin

- A **spring (Sep–Nov) SAG window**, possibly alongside the current Dec–Jan window, is a cheap change to an existing method. It is worth testing on their own data.
- **Seasonal timing features** are the best candidates so far for keeping forest error at alerts close to target. They still need checking on the sealed QC half and, ideally, on Cropin's blind holdout.

## 6. Files

- `results/wave2/`: screen, models, nested CV, selected features, feature tables
- `scripts/`: `w2_*.py` and the GEE script `w2_batch2_gee.js`
