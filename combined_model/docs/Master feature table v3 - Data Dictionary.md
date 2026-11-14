# Master feature table v3 – data dictionary

*Built by Nathan Chu (with Claude), 13 Nov 2026 · Script `build_master_v3.py` · Files: `master_features_v3.csv` (2,667 rows × 55 columns), `master_alphaearth_v3.csv` (64 AlphaEarth dimensions, as-of year; join on `uid`), `v3_columns.csv`*

**This is exactly the data the combined-model search used.** Compared with v2: Meera's winter B6 is filled; harvest-cycle features are **rebuilt on our own Landsat data** (`rh_`), and land history is **rebuilt from MapBiomas Collection 9** (`lh_`). Teammates' original data columns (`hc_`, `r2_`) are removed, because every idea they carried is now rebuilt on our points and windows.

**Confidential:** contains Cropin's QC labels. Share through the team Drive, never on GitHub.

## 1. Rows

| set | split | rows | plantations |
|---|---|---|---|
| mb | dev | 1817 | 411 |
| mb | lockbox | 436 | 102 |
| qc | dev | 207 | 149 |
| qc | heldback | 207 | 148 |

## 2. Key columns and rules

| Column | Meaning |
|---|---|
| `uid` | Unique key: `mb_<TARGETID>` or `qc_<GeoJSON index>` |
| `label` | **1 = plantation, 0 = natural forest** |
| `split` | `dev` = use freely · `lockbox` = MapBiomas points sealed in Phase 7 · `heldback` = QC final-test half, **already used once** for the combined-model finalists; do not tune on it |
| `train_eligible` | MapBiomas dev, ≥ 2 interpreters agree, summer NDVI > 0.6 (1,700 points). **Train only on these.** |
| `block` | Spatial block (0.5° MapBiomas, 0.25° QC). Use it for cross-validation folds and bootstrap. |
| `asof_year` | Last year features may use: 2022 for MapBiomas points; year before the estimated clearing for QC |
| `border` | MapBiomas border flag |
| `team_pid` | Teammates' ID (`V<TARGETID>`, `Q<index>`), to link to their files |
| `nc_wc_dist_nontree_m` | Distance to non-tree land: for the **edge error category**, not a model feature |

## 3. Columns

Prefixes: `nc_` Nathan + Claude · `mm_` Meera's measurements (rebuilt) · `rh_` harvest cycles (rebuilt on our Landsat) · `lh_` land history (rebuilt from MapBiomas). AUC = single-feature AUC on the 1,700 training points (folded to ≥ 0.5). **Roles:** `feature` = in the search's feature pool · `benchmark` = Cropin's baseline inputs (SAG + Meta height + height spread) · `veto` = tested as a rule, did not help.

| Column | Role | Definition | AUC MapBiomas | MapBiomas-derived |
|---|---|---|---|---|
| `nc_B_SAG_SON` | feature | SAG median Sep–Nov, 3 yrs | 0.88 | |
| `nc_B_SAG_phase_sin` | feature | SAG seasonal timing (sine of harmonic phase), monthly S2, 3 yrs to as-of year | 0.85 | |
| `mm_SAG_win` | feature | SAG Apr–Sep, median of 3 yearly medians (Meera) | 0.73 | |
| `nc_wi3_NDMI` | feature | NDMI winter Apr–Sep, median of 3 yearly medians (3-yr window) | 0.71 | |
| `nc_su3_NDMI` | feature | NDMI summer Dec–Mar, median of 3 yearly medians (3-yr window) | 0.68 | |
| `nc_sg3_NDVI` | feature | NDVI Oct–Jan, median of 3 yearly medians | 0.66 | |
| `mm_NDVI_win` | feature | NDVI Apr–Sep, median of 3 yearly medians (Meera) | 0.55 | |
| `mm_B6_win` | feature | B6 red-edge reflectance Apr–Sep, median of 3 yearly medians (Meera; batch M1) | 0.5 | |
| `mm_B8A_win` | feature | B8A near-infrared reflectance Apr–Sep, 3 years (Meera) | 0.51 | |
| `nc_B_NDRE_phase_sin` | feature | NDRE seasonal timing (sine), 3 yrs | 0.67 | |
| `nc_B_SWIR_phase_sin` | feature | SWIR1 seasonal timing (sine), 3 yrs | 0.67 | |
| `nc_B_SWIR_phase_cos` | feature | SWIR1 seasonal timing (cosine), 3 yrs | 0.7 | |
| `nc_E_eth_h` | feature | ETH canopy height 2020 at point | 0.68 | |
| `nc_E_eth_sd50` | feature | ETH canopy height SD within 50 m | 0.67 | |
| `nc_F_VHminusVV_median` | feature | Sentinel-1 VH−VV median dB, 3 yrs | 0.77 | |
| `nc_F_VH_median` | feature | Sentinel-1 VH median dB, 3 yrs | 0.67 | |
| `nc_D_max_nbr_drop` | feature | Landsat: largest NBR drop, 25 yrs | 0.88 | |
| `nc_D_cc_n_breaks` | feature | CCDC breaks (Landsat, fit to as-of year) | 0.88 | |
| `nc_D_years_continuous_forest` | feature | Landsat: years of continuous closed forest before as-of year | 0.85 | |
| `nc_D_lt_n_vertices` | feature | LandTrendr vertices (fit to as-of year) | 0.71 | |
| `nc_D_ndvi_1985_89` | feature | Landsat NDVI 1985–89 | 0.7 | |
| `nc_D_h_total_loss_px` | feature | Hansen loss pixels within 150 m, to as-of year | 0.79 | |
| `nc_nfw` | feature | Natural Forests of the World 2020 probability (B0/250) | 0.96 | |
| `nc_H_mapbiomas10m_plantation` | feature | MapBiomas 10 m map = plantation in as-of year (MAPBIOMAS-DERIVED) | 0.89 | **yes** |
| `rh_n_cycles` | feature | Cut-and-regrow cycles (NBR fall ≥ 0.20 within 3 yrs that regrows half within 10 yrs), 1985 to as-of year | 0.8 | |
| `rh_s_std` | feature | SD of yearly NBR, 1985 to as-of year (teammates chose this after first results) | 0.96 | |
| `rh_s_frac_below_forest` | feature | Share of years with NBR below the closed-canopy level | 0.8 | |
| `rh_abrupt_start` | feature | ≥ 3 years at open-land NBR, then closed canopy within 5 years (planting on open land) | 0.63 | |
| `nc_hansen_tc2000` | feature | Hansen tree cover 2000 at the point (our export) | 0.75 | |
| `lh_natural_run_years` | feature | Years of continuous natural woody cover (MapBiomas classes 3, 4, 5, 6, 49) ending at the as-of year | 0.91 | **yes** |
| `lh_n_changes` | feature | Number of MapBiomas class changes, 1985 to as-of year (max 2023) | 0.79 | **yes** |
| `lh_ever_open` | feature | Ever farmland, grassland, or other open land in MapBiomas history before the as-of year | 0.71 | **yes** |
| `nc_A_SAG` | benchmark | SAG Oct–Jan composite, ONE season (Cropin-style) | 0.85 | |
| `nc_chm_mean_r15` | benchmark | Meta canopy height mean, 15 m radius (~2019-20) | 0.52 | |
| `nc_chm_stdDev_r15` | benchmark | Meta canopy height SD, 15 m radius | 0.54 | |
| `rh_forest_since_1985` | veto (tested, did not help) | No cut and NBR never below closed canopy (veto idea; made forest errors worse in our test) | 0.68 | |

All features have values for every point (one QC point lacks radar).

## 4. Known issues

1. **MapBiomas-derived features** (`nc_H_mapbiomas10m_plantation`, `lh_`) are circular on MapBiomas points. Report models with and without them.
2. **History features look much better on MapBiomas points than at Cropin's alerts** (e.g., `rh_s_std` 0.96 vs 0.75). Judge models on the QC numbers.
3. **The QC held-back half is used.** Further model choices should use only the development half; Cropin's blind holdout is the next clean test.
4. **Rebuilt harvest features differ slightly from the teammates' originals:** our Landsat Collection 2 surface reflectance has no cross-sensor adjustment, and the series runs to each point's as-of year rather than 2020.
5. **Multipart QC polygons:** our QC features use the first part of 25 multipart polygons.
6. **Meera's measurements** use our 3-year as-of windows, not her 2019–2022, so values differ slightly from hers. Her SAG Oct–Jan is represented by `nc_B_SAG_SON` (ρ 0.90).
