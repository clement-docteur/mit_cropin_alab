# Master feature table v2 – data dictionary

*Built by Nathan Chu (with Claude), 13 Nov 2026 · Script `build_master.py` + `build_master_v2.py` · Repo commit `05bc61e` (12 Nov) · Files: `master_features_v2.csv` (2,667 rows × 56 columns), `master_alphaearth_v2.csv` (64 AlphaEarth dimensions, as-of year), `feature_decisions_v2.csv` (every column kept or dropped, with the reason)*

**Confidential:** contains Cropin's QC labels. Keep in the shared Drive, never on GitHub.

## 1. Rows

One row per point. Our point set (decision 13 Nov): MapBiomas points with stable class 3/9 over 2013–2022, and Cropin QC polygons after removing duplicates within 100 m.

| set | split | rows | plantations | run 2 features | harvest features |
|---|---|---|---|---|---|
| mb | dev | 1817 | 411 | 1428 | 1428 |
| mb | lockbox | 436 | 102 | 357 | 357 |
| qc | dev | 207 | 149 | 0 | 207 |
| qc | heldback | 207 | 148 | 0 | 207 |

## 2. Key columns and rules

| Column | Meaning |
|---|---|
| `uid` | Unique key: `mb_<TARGETID>` or `qc_<GeoJSON index>` |
| `label` | **1 = plantation, 0 = natural forest** (run 2 used forest = 1; converted) |
| `split` | `dev` = use freely · `lockbox` = MapBiomas points sealed in Phase 7 (**357 of them were already used by run 2**) · `heldback` = QC final judge (**the harvest-cycle analysis used all 433 QC polygons, and Meera's 420-alert analysis probably did too; report as "seen during feature screening"**) |
| `train_eligible` | MapBiomas dev, ≥ 2 interpreters agree, summer NDVI > 0.6 (1,700 points). Train only on these. |
| `block` | Spatial block for cross-validation (0.5° MapBiomas, 0.25° QC). Use it for folds and bootstrap. |
| `asof_year` | Last year features may use: 2022 for MapBiomas; year before the estimated clearing for QC |
| `border` | MapBiomas border flag. **Run 2 and harvest features are missing for every border point** (teammates excluded them), so edges cannot be scored with those features |
| `team_pid` | Teammates' ID (`V<TARGETID>`, `Q<index>`) |
| `r2_group_INFO_ONLY` | Run 2 group (uses the MapBiomas map). **Never a feature or target.** |
| `nc_wc_dist_nontree_m` | Distance to non-tree land: for the **edge error category**, not a model feature |

## 3. Features kept

Prefixes: `nc_` Nathan + Claude · `r2_` run 2 table (Clement), MapBiomas points only · `hc_` harvest cycles, recomputed from the notebook · `mm_` Meera's six measurements, rebuilt from our Sentinel-2 monthly medians. AUC = single-feature AUC on the 1,700 MapBiomas training points (folded to ≥ 0.5).

**Roles:** `feature` = model input · `veto` = forest-protection rule, not a model input · `pending_QC` = no QC values yet, so leave out of models scored at alerts until extracted · `benchmark` = needed for Cropin's baseline (SAG + Meta height + height spread), not a recommended feature.

| Column | Role | Definition | AUC MapBiomas | Coverage |
|---|---|---|---|---|
| `nc_B_SAG_SON` | feature | SAG median Sep–Nov, 3 yrs | 0.88 | both |
| `nc_A_SAG` | benchmark | SAG Oct–Jan composite, ONE season (Cropin-style) | 0.85 | both |
| `nc_B_SAG_phase_sin` | feature | SAG seasonal timing (sine of harmonic phase), monthly S2, 3 yrs to as-of year | 0.85 | both |
| `mm_SAG_win` | feature | SAG Apr–Sep, median of 3 yearly medians (Meera) | 0.73 | both |
| `r2_sag_swing_mean` | pending_QC | SAG seasonal swing 2022 — mean in 15 m buffer | 0.77 | MapBiomas only |
| `nc_wi3_NDMI` | feature | NDMI winter Apr–Sep, median of 3 yearly medians (3-yr window) | 0.71 | both |
| `nc_su3_NDMI` | feature | NDMI summer Dec–Mar, median of 3 yearly medians (3-yr window) | 0.68 | both |
| `nc_sg3_NDVI` | feature | NDVI Oct–Jan, median of 3 yearly medians | 0.66 | both |
| `mm_NDVI_win` | feature | NDVI Apr–Sep, median of 3 yearly medians (Meera) | 0.55 | both |
| `mm_B6_win` | feature | B6 red-edge reflectance Apr–Sep, 3 years (Meera). **Empty until GEE batch M1 runs** | — | both |
| `mm_B8A_win` | feature | B8A near-infrared reflectance Apr–Sep, 3 years (Meera) | 0.51 | both |
| `nc_B_NDRE_phase_sin` | feature | NDRE seasonal timing (sine), 3 yrs | 0.67 | both |
| `nc_B_SWIR_phase_sin` | feature | SWIR1 seasonal timing (sine), 3 yrs | 0.67 | both |
| `nc_B_SWIR_phase_cos` | feature | SWIR1 seasonal timing (cosine), 3 yrs | 0.7 | both |
| `nc_E_eth_h` | feature | ETH canopy height 2020 at point | 0.68 | both |
| `nc_chm_mean_r15` | benchmark | Meta canopy height mean, 15 m radius (~2019-20) | 0.52 | both |
| `nc_E_eth_sd50` | feature | ETH canopy height SD within 50 m | 0.67 | both |
| `nc_chm_stdDev_r15` | benchmark | Meta canopy height SD, 15 m radius | 0.54 | both |
| `nc_F_VHminusVV_median` | feature | Sentinel-1 VH−VV median dB, 3 yrs | 0.77 | both |
| `nc_F_VH_median` | feature | Sentinel-1 VH median dB, 3 yrs | 0.67 | both |
| `hc_s_std` | feature | SD of yearly NBR 1985–2020 (chosen after first results) | 0.97 | both (border points missing) |
| `nc_D_max_nbr_drop` | feature | Landsat: largest NBR drop, 25 yrs | 0.88 | both |
| `nc_D_cc_n_breaks` | feature | CCDC breaks (Landsat, fit to as-of year) | 0.88 | both |
| `hc_s_frac_below_forest` | feature | Share of years below closed-canopy NBR | 0.86 | both (border points missing) |
| `hc_max_drop` | feature | Largest cut (NBR) | 0.86 | both (border points missing) |
| `nc_D_years_continuous_forest` | feature | Landsat: years of continuous closed forest before as-of year | 0.85 | both |
| `hc_forest_since_1985` | veto | No cut and NBR never below closed canopy (veto feature) | 0.72 | both (border points missing) |
| `hc_abrupt_start` | feature | ≥3 yrs open land then closed canopy within 5 yrs | 0.71 | both (border points missing) |
| `nc_D_lt_n_vertices` | feature | LandTrendr vertices (fit to as-of year) | 0.71 | both |
| `nc_D_ndvi_1985_89` | feature | Landsat NDVI 1985–89 | 0.7 | both |
| `hc_tc2000` | feature | Hansen tree cover 2000 | 0.79 | both (border points missing) |
| `nc_D_h_total_loss_px` | feature | Hansen loss pixels within 150 m, to as-of year | 0.79 | both |
| `hc_hansen_gain` | feature | Hansen gain 2000–2012 | 0.76 | both (border points missing) |
| `nc_nfw` | feature | Natural Forests of the World 2020 probability (B0/250) | 0.96 | both |
| `nc_H_mapbiomas10m_plantation` | feature | MapBiomas 10 m map = plantation in as-of year (MAPBIOMAS-DERIVED) | 0.89 | both · **MapBiomas-derived** |
| `r2_ever_farm_mean` | pending_QC | Share of pixels ever farmland in MapBiomas Collection 9 history (MAPBIOMAS-DERIVED) — mean in 15 m buffer | 0.68 | MapBiomas only · **MapBiomas-derived** |

**Meera's six measurements** in this table: `nc_B_SAG_SON` (stands in for her SAG Oct–Jan: rho 0.90), `mm_B6_win`, `mm_B8A_win`, `mm_SAG_win`, `nc_wi3_NDMI` (identical to her winter NDMI), `mm_NDVI_win`. They are rebuilt with our as-of windows (MapBiomas 2020–2022; QC: 3 years before clearing), not her 2019–2022, so values will differ slightly from hers.

## 4. How duplicates were resolved

The rule was fixed before looking at QC results and uses MapBiomas training points only:

1. Group columns that measure the same idea (for example, four versions of SAG).
2. Rank inside a group: covers both MapBiomas and QC points, then 3-year window, then one season, then single-year (2022) table; ties broken by MapBiomas AUC.
3. Keep the top one. Drop any other member with |Spearman ρ| ≥ 0.85 to a kept member, or that measures the same thing from the same source. Keep the rest.
4. Keep MapBiomas-only columns only when no column covering both sets measures that idea.

| Dropped column | Reason |
|---|---|
| `mm_SAG_OctJan` | rho 0.90 with nc_B_SAG_SON (Meera's SAG Oct–Jan base) |
| `nc_B_SAG_DJ` | rho 0.92 with SAG Oct–Jan and 0.86 with SAG mean (same index, Dec–Jan window) |
| `nc_B_SAG_mean` | rho 0.87 with nc_B_SAG_SON |
| `r2_sag_oct_nov_mean` | rho 0.87 with nc_B_SAG_SON; MapBiomas only |
| `r2_sag_year_mean` | Same index (SAG), single year, MapBiomas only; rho 0.86 with SAG mean |
| `mm_NDMI_win` | Identical to nc_wi3_NDMI |
| `nc_A_su_NDMI` | One-season version of nc_su3_NDMI (rho 0.90) |
| `r2_ndmi_aug_oct_mean` | Winter/spring NDMI, single year, MapBiomas only; covered by nc_wi3_NDMI |
| `nc_A_sg_NDVI` | One-season version of nc_sg3_NDVI |
| `r2_ndvi_aug_oct_mean` | Single year, MapBiomas only, AUC 0.54; covered by winter NDVI |
| `r2_ndvi_p10_jul_nov_mean` | Single year, MapBiomas only, AUC 0.51 |
| `r2_ndvi_year_mean` | Single year, MapBiomas only, AUC 0.54 |
| `r2_height_mean` | rho 1.00 with nc_E_eth_h: run 2 height is the ETH map, not Meta |
| `r2_height_stdDev` | ETH height SD in 15 m: rho 0.75 with nc_E_eth_sd50, same source, MapBiomas only |
| `r2_vhvv_mean` | Cross-ratio, 2021–22, MapBiomas only; rho 0.78 with nc version |
| `r2_vh_mean` | rho 0.98 with nc_F_VH_median |
| `r2_vv_mean` | rho 0.95 with nc_F_VH_median |
| `hc_s_min` | rho -0.93 with hc_s_std |
| `hc_s_max_drop3` | rho 0.86 with hc_s_std |
| `hc_years_since_last_cut` | rho -0.99 with hc_max_drop |
| `hc_n_cuts` | rho 0.98 with hc_max_drop |
| `hc_n_cycles` | rho 0.96 with hc_max_drop |
| `hc_regrow_speed` | Teammate finding: poor predictor (regrowth equally fast in both classes) |
| `hc_has_cut` | Thresholded copy of cut count |
| `hc_has_cycle` | Thresholded copy of cycle count |
| `hc_two_plus_cycles` | Thresholded copy of cycle count |
| `hc_n_years` | Data-availability count, not a feature (AUC 0.51) |
| `hc_hansen_loss` | Same source and idea as nc_D_h_total_loss_px (pixel only, to 2020; lower AUC) |
| `r2_farm_years_mean` | rho 0.99 with r2_ever_farm_mean |
| `r2_ever_farm_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_farm_years_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_ndmi_aug_oct_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_ndvi_aug_oct_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_ndvi_p10_jul_nov_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_ndvi_year_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_sag_oct_nov_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_sag_swing_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_sag_year_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_vh_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_vhvv_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |
| `r2_vv_stdDev` | Within-buffer SD from the run 2 table: not in run 2's own model, MapBiomas only |

## 5. Known issues

1. **`mm_B6_win` is empty.** B6 was not in our monthly export. Run `m1_b6_winter_gee.js` (2 tasks, labels not included), then rebuild.
2. **Run 2 features have no QC values** (`r2_sag_swing_mean`, `r2_ever_farm_mean`). Clement needs to run his GEE script on the QC polygons with imagery from before each clearing.
3. **Border points lack all `hc_` features** (20% of training points). Models using them treat edges as missing.
4. **Run 2 "height" is the ETH map, not Meta** (ρ = 1.00 with `nc_E_eth_h`, 0.40 with Meta). Cropin was told it was Meta at the 12 Nov meeting; worth correcting.
5. **MapBiomas-derived features** (`nc_H_mapbiomas10m_plantation`, `r2_ever_farm_mean`) are allowed but must be reported with and without, and are circular on MapBiomas points.
6. **Selection after looking:** `hc_s_std` was chosen after first results; the harvest analysis screened features on all 433 QC polygons.
7. **History features drop sharply from MapBiomas points to QC alerts** (e.g., `hc_s_std` 0.97 → 0.77, CCDC breaks 0.88 → 0.61). Inference: MapBiomas interpreters label partly from Landsat history.
8. **Mapping cost differs:** `hc_` features use a custom segmentation on yearly Landsat series that GEE does not provide; mapping them needs per-pixel time-series exports. `nc_D_` history uses LandTrendr/CCDC, which run in GEE.
9. **Multipart QC polygons:** our QC features use the first part of 25 multipart polygons.
