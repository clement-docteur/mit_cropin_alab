# Feature Exploration – Wave 1 Report (Families A, C, D)

*Prepared by Nathan Chu (with Claude) · 6 November 2026 · Analyses are our own (Google Earth Engine + Python) unless a source is given · **[inference]** = our interpretation, not tested · Plan: `00 Workflow Plan.md` (v4.1) · Every number traces to `Run Log.md` and `results/wave1/`*

---

## 1. Summary

- **Spectral features (Family A) give the most reliable gain.** Adding summer moisture (NDMI) and spring greenness to Cropin's SAG signal raises AUC on Cropin's QC points from 0.78 to 0.83.
- **Land history (Family D) is very strong on MapBiomas points but much weaker on Cropin's points.** Most of Cropin's plantation alerts are first harvests of long-standing stands, which have no earlier clear-cut to detect.
- **Texture (Family C) does not help at Cropin's alerts.** The texture signal flips direction between MapBiomas and Cropin's points, and the planting-row test fails. This supports Deepak's caution about texture at 10 m (Cropin meeting transcript, 21 Oct 2026, ~00:34:21–00:35:35).
- **Thresholds tuned on MapBiomas do not carry over to Cropin's points.** A threshold that gives 5% forest error on MapBiomas gives 10–17% on Cropin's QC points. Weighting MapBiomas points to look like alert locations does not fix this, so the cause is not landscape context.

"Forest error" in this report always means **real forest called plantation** (our dangerous error), not Cropin's "false positive" (a plantation harvest flagged as deforestation).

## 2. Data and method (short)

| Item | Detail |
|---|---|
| Training and cross-validation | 1,753 MapBiomas validation points in PR/SC/RS (1,386 forest, 367 plantation), stable class 2013–2022 in both classes ([MapBiomas, Validation points](https://brasil.mapbiomas.org/en/?p=951)); points with only 1 of 3 interpreters agreeing are left out of training |
| Cropin check | QC dev half: 207 points (58 forest, 149 plantation). The other half and a 20% MapBiomas "lockbox" stay sealed until the final test |
| As-of rule | Features only from before each point's clearing (QC clearing months estimated from monthly NDVI; 91% fall in Apr–Sep 2026) |
| Evaluation | Spatially blocked 5-fold cross-validation (0.5° blocks); AUC with block-bootstrap 95% CI; recall at 1%, 2%, and 5% forest error; threshold-transfer check on QC |
| Decision-rule baseline | SAG + Meta canopy height + height spread. NFW is excluded because its training labels may resemble MapBiomas (unverified) |
| Models | Logistic regression only, at most ~10 features, chosen by stability selection |

## 3. Results

### 3.1 Models

| Model | MapBiomas AUC | MapBiomas recall @1% / 2% / 5% | QC AUC [95% CI] | QC forest error at MapBiomas 5% threshold | QC recall |
|---|---|---|---|---|---|
| Baseline | 0.841 | 19% / 27% / 42% | 0.782 [0.70–0.84] | 17.2% (10/58) | 64% |
| + A: SAG, summer NDMI, Oct–Jan NDVI | 0.888 | 37% / 48% / 67% | 0.834 [0.78–0.89] | 17.2% (10/58) | 74% |
| + C: NIR homogeneity (3×3) | 0.846 | 23% / 29% / 50% | 0.794 [0.73–0.85] | 17.2% (10/58) | 60% |
| + D: six history features* | 0.955 | 68% / 73% / 82% | 0.802 [0.73–0.86] | 10.3% (6/58) | 56% |
| Preview A + C + D | 0.965 | 72% / 78% / 88% | 0.856 [0.81–0.91] | 13.8% (8/58) | 69% |
| Comparison: baseline + NFW | 0.957 | 63% / 74% / 83% | 0.780 | 13.8% (8/58) | 43% |

\*Largest NBR drop, number of CCDC breaks, years of continuous forest, Hansen loss pixels within 150 m, 1985–89 NDVI, LandTrendr vertex count.

### 3.2 Hypotheses tested (from *Task 1.2 – Hypotheses Master List*)

| Hypothesis | Result |
|---|---|
| H1–H5, H11: disturbance history (earlier clear-cuts, rotation, regrowth, continuous forest, abrupt loss, stability) | **Supported on MapBiomas** (AUC up to 0.89), **weak on QC** (0.62–0.68) |
| H6: neighbours cleared together (Hansen) | Supported on MapBiomas (0.79), no signal on QC (0.53) |
| H7: land before forest (1985–89 NDVI) | Supported on both (MapBiomas 0.70, QC 0.64) |
| H18/H24: lower texture, more uniform stands | **Context-dependent:** direction flips between MapBiomas and QC |
| H16: planting rows (directional texture) | **Not supported** (AUC 0.59); rows of ~3 m are below what 10 m pixels can resolve **[inference]** |
| H24 with SNIC segments | **Not supported**, and the most expensive feature tried (a failed run used 513,844 EECU-seconds) |

### 3.3 Where errors happen

| Group (MapBiomas, model with history) | Forest error |
|---|---|
| Interior points | 3.5% |
| Border points | 10.7% |
| 3 of 3 interpreters agree | 2.2% |
| 2 of 3 agree | 14.6% |

History features raise forest error at borders and on low-agreement labels. Edges need separate handling (hard case HC9).

## 4. Why the thresholds don't transfer

- MapBiomas points that look most like alert locations (top 25% by surrounding tree cover, patch size, and distance to the forest edge) show only 6–8% forest error, versus 14–17% on QC.
- **[inference]** The difference is about the forest itself at Cropin's alerts (for example younger or degraded forest), or about QC labels. A visual check of the misclassified QC forest points would tell which.

## 5. Checks and limits

- **Independent review:** a separate agent reproduced every model number to 4 decimals and recomputed three history features for all 2,024 points with no mismatches. It raised 7 issues:
  - Fixed: unequal history windows, a mislabelled Hansen "share", and LandTrendr/CCDC fits that used data after the cutoff (refitted in GEE).
  - Checked: selection optimism (nested CV gives the same AUC, 0.954 vs 0.955).
  - Remaining: 4 smaller items, listed in `Run Log.md`.
- **Small QC sample:** 58 forest points. A forest error of 6/58 has a 95% CI of about 4–21%.
- **Biased QC sample:** QC forest is only forest where Cropin already detected a clearing. **Cropin has no labels for forest it misses**, so these numbers do not show overall forest error.
- **QC dev is not fully independent:** it was used to flag context-dependent features and to fit the alert-like weights. The sealed QC half is the independent test.

## 6. Files

- `results/wave1/`: screens, models, nested CV, selected features, feature tables
- `scripts/`: `w1_*.py` and the GEE scripts `w1_batch1_gee.js`, `w1_batch1b_gee.js`, `w1c_refit_gee.js`
