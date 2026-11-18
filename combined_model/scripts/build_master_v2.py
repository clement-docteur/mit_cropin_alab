"""Master feature table v2 (13 Nov 2026): v1 + Meera's six measurements + 3-year Oct–Jan NDVI, duplicates resolved.
Duplicate rule (decided before looking at QC results; uses MapBiomas training points only):
  1. Group columns that measure the same idea.
  2. Rank inside a group: covers both MapBiomas and QC points > 3-year window > one season > single-year (2022) table;
     ties broken by single-feature AUC on MapBiomas training points.
  3. Keep the top one; drop any other member with |Spearman rho| >= 0.85 to a kept member, or that measures the same thing
     from the same source. Keep the rest (they carry different information).
  4. MapBiomas-only columns (no QC values yet) are kept only if no column covering both sets measures that idea: role 'pending_QC'.
Columns Cropin's baseline needs are kept with role 'benchmark' even when dropped as features."""
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, '/home/claude/master'); import meera_features as MF
from sklearn.metrics import roc_auc_score
OUT = '/home/claude/master/out/'; U = '/mnt/user-data/uploads/cropin_texture/'
M = pd.read_csv(OUT + 'master_features_v1.csv').merge(MF.build(), on=['id', 'set'], how='left')

# B6 winter, if the GEE export (m1_b6_winter_*.csv) has arrived
if os.path.exists(U + 'm1_b6_winter_mb.csv') and os.path.exists(U + 'm1_b6_winter_qc.csv'):
    B = pd.concat([pd.read_csv(U + 'm1_b6_winter_mb.csv'), pd.read_csv(U + 'm1_b6_winter_qc.csv')]).dropna(subset=['B6'])
    B['yr'] = pd.to_datetime(B.month).dt.year; B['B6'] = B.B6 / 1e4
    y = B.groupby(['id', 'set', 'yr']).B6.agg(['median', 'size']).reset_index()
    b6 = y[y['size'] >= 2].groupby(['id', 'set'])['median'].median().rename('b6').reset_index()
    M = M.drop(columns='mm_B6_win').merge(b6.rename(columns={'b6': 'mm_B6_win'}), on=['id', 'set'], how='left')

# 3-year Oct–Jan NDVI (replaces the one-season nc_A_sg_NDVI)
S = pd.concat([pd.read_csv(U + 'w2_s2_monthly_mb.csv'), pd.read_csv(U + 'w2_s2_monthly_qc.csv')]).dropna(subset=['B8'])
S['NDVI'] = (S.B8 - S.B4) / (S.B8 + S.B4); d = pd.to_datetime(S.month); S['mo'] = d.dt.month; S['yr'] = d.dt.year
S = S[S.mo.isin([10, 11, 12, 1])]; S['ss'] = np.where(S.mo >= 10, S.yr, S.yr - 1)
y = S.groupby(['id', 'set', 'ss']).NDVI.agg(['median', 'size']).reset_index()
M = M.merge(y[y['size'] >= 2].groupby(['id', 'set'])['median'].median().rename('nc_sg3_NDVI').reset_index(), on=['id', 'set'], how='left')

# ---------------- decisions (rank order inside each group, then role) ----------------
# (column, role, reason). Roles: feature, veto (rule-type feature), pending_QC, benchmark, dropped
D = [
 # SAG level
 ('nc_B_SAG_SON', 'feature', 'SAG level: best MapBiomas AUC (0.88) among both-set, 3-year versions'),
 ('mm_SAG_OctJan', 'dropped', 'rho 0.90 with nc_B_SAG_SON (Meera\'s SAG Oct–Jan base)'),
 ('nc_B_SAG_DJ', 'dropped', 'rho 0.92 with SAG Oct–Jan and 0.86 with SAG mean (same index, Dec–Jan window)'),
 ('nc_B_SAG_mean', 'dropped', 'rho 0.87 with nc_B_SAG_SON'),
 ('nc_A_SAG', 'benchmark', 'One-season SAG Oct–Jan: Cropin baseline input; rho 0.82–0.90 with kept SAG'),
 ('r2_sag_oct_nov_mean', 'dropped', 'rho 0.87 with nc_B_SAG_SON; MapBiomas only'),
 ('r2_sag_year_mean', 'dropped', 'Same index (SAG), single year, MapBiomas only; rho 0.86 with SAG mean'),
 # SAG timing / season
 ('nc_B_SAG_phase_sin', 'feature', 'SAG seasonal timing; rho < 0.3 with other SAG columns'),
 ('mm_SAG_win', 'feature', 'Meera: SAG in winter; rho < 0.15 with other SAG columns'),
 ('r2_sag_swing_mean', 'pending_QC', 'SAG seasonal swing: distinct (rho 0.26 with phase), no QC values yet'),
 # NDMI
 ('nc_wi3_NDMI', 'feature', 'Winter NDMI, 3 years (identical to Meera\'s winter NDMI, rho 1.00)'),
 ('mm_NDMI_win', 'dropped', 'Identical to nc_wi3_NDMI'),
 ('nc_su3_NDMI', 'feature', 'Summer NDMI, 3 years; rho 0.81 with winter NDMI'),
 ('nc_A_su_NDMI', 'dropped', 'One-season version of nc_su3_NDMI (rho 0.90)'),
 ('r2_ndmi_aug_oct_mean', 'dropped', 'Winter/spring NDMI, single year, MapBiomas only; covered by nc_wi3_NDMI'),
 # NDVI
 ('nc_sg3_NDVI', 'feature', 'Oct–Jan NDVI, 3 years (Lean 4 idea, now 3-year)'),
 ('nc_A_sg_NDVI', 'dropped', 'One-season version of nc_sg3_NDVI'),
 ('mm_NDVI_win', 'feature', 'Meera: winter NDVI; rho 0.60 with Oct–Jan NDVI'),
 ('r2_ndvi_aug_oct_mean', 'dropped', 'Single year, MapBiomas only, AUC 0.54; covered by winter NDVI'),
 ('r2_ndvi_p10_jul_nov_mean', 'dropped', 'Single year, MapBiomas only, AUC 0.51'),
 ('r2_ndvi_year_mean', 'dropped', 'Single year, MapBiomas only, AUC 0.54'),
 # red edge / NIR / SWIR
 ('mm_B6_win', 'feature', 'Meera: winter red edge B6 (filled after GEE batch M1)'),
 ('mm_B8A_win', 'feature', 'Meera: winter near-infrared B8A'),
 ('nc_B_NDRE_phase_sin', 'feature', 'Red-edge seasonal timing; rho 0.05 with B8A'),
 ('nc_B_SWIR_phase_sin', 'feature', 'SWIR timing (sine); keep with cosine as a pair'),
 ('nc_B_SWIR_phase_cos', 'feature', 'SWIR timing (cosine)'),
 # height
 ('nc_E_eth_h', 'feature', 'Canopy height (ETH 2020)'),
 ('r2_height_mean', 'dropped', 'rho 1.00 with nc_E_eth_h: run 2 height is the ETH map, not Meta'),
 ('nc_chm_mean_r15', 'benchmark', 'Meta height: Cropin baseline input; MapBiomas AUC 0.52'),
 ('nc_E_eth_sd50', 'feature', 'Height variability (ETH, 50 m)'),
 ('r2_height_stdDev', 'dropped', 'ETH height SD in 15 m: rho 0.75 with nc_E_eth_sd50, same source, MapBiomas only'),
 ('nc_chm_stdDev_r15', 'benchmark', 'Meta height SD: Cropin baseline input; MapBiomas AUC 0.54'),
 # radar
 ('nc_F_VHminusVV_median', 'feature', 'Radar cross-ratio, 3 years'),
 ('nc_F_VH_median', 'feature', 'Radar VH, 3 years; rho -0.18 with cross-ratio'),
 ('r2_vhvv_mean', 'dropped', 'Cross-ratio, 2021–22, MapBiomas only; rho 0.78 with nc version'),
 ('r2_vh_mean', 'dropped', 'rho 0.98 with nc_F_VH_median'),
 ('r2_vv_mean', 'dropped', 'rho 0.95 with nc_F_VH_median'),
 # Landsat history (greedy by MapBiomas AUC, rho >= 0.85 dropped)
 ('hc_s_std', 'feature', 'NBR variability 1985–2020 (chosen after first results)'),
 ('hc_s_min', 'dropped', 'rho -0.93 with hc_s_std'),
 ('hc_s_max_drop3', 'dropped', 'rho 0.86 with hc_s_std'),
 ('nc_D_max_nbr_drop', 'feature', 'Largest NBR drop, 25 years to as-of; rho 0.62 with hc_s_std'),
 ('nc_D_cc_n_breaks', 'feature', 'CCDC breaks; rho <= 0.56 with others'),
 ('hc_s_frac_below_forest', 'feature', 'Share of years below closed canopy; rho 0.83 with hc_s_std'),
 ('hc_max_drop', 'feature', 'Largest cut (segmented); rho 0.68 with hc_s_std'),
 ('hc_years_since_last_cut', 'dropped', 'rho -0.99 with hc_max_drop'),
 ('hc_n_cuts', 'dropped', 'rho 0.98 with hc_max_drop'),
 ('hc_n_cycles', 'dropped', 'rho 0.96 with hc_max_drop'),
 ('nc_D_years_continuous_forest', 'feature', 'Years of continuous forest; rho <= 0.62 with others'),
 ('hc_forest_since_1985', 'veto', 'rho -0.85 with hc_s_frac_below_forest, kept as a forest-protection rule (not a model input)'),
 ('hc_abrupt_start', 'feature', 'Planting on open land; rho <= 0.62 with others'),
 ('nc_D_lt_n_vertices', 'feature', 'LandTrendr vertices; rho <= 0.36 with others'),
 ('nc_D_ndvi_1985_89', 'feature', 'NDVI 1985–89; rho <= 0.64 with others'),
 ('hc_regrow_speed', 'dropped', 'Teammate finding: poor predictor (regrowth equally fast in both classes)'),
 ('hc_has_cut', 'dropped', 'Thresholded copy of cut count'), ('hc_has_cycle', 'dropped', 'Thresholded copy of cycle count'),
 ('hc_two_plus_cycles', 'dropped', 'Thresholded copy of cycle count'), ('hc_n_years', 'dropped', 'Data-availability count, not a feature (AUC 0.51)'),
 # Hansen
 ('hc_tc2000', 'feature', 'Hansen tree cover 2000'),
 ('nc_D_h_total_loss_px', 'feature', 'Hansen loss within 150 m, to as-of year'),
 ('hc_hansen_loss', 'dropped', 'Same source and idea as nc_D_h_total_loss_px (pixel only, to 2020; lower AUC)'),
 ('hc_hansen_gain', 'feature', 'Hansen gain 2000–2012; rho <= 0.47 with others'),
 # public maps
 ('nc_nfw', 'feature', 'Natural Forests of the World 2020'),
 ('nc_H_mapbiomas10m_plantation', 'feature', 'MAPBIOMAS-DERIVED: 10 m map says plantation (allowed, report with and without)'),
 ('r2_ever_farm_mean', 'pending_QC', 'MAPBIOMAS-DERIVED farmland history; no QC values yet'),
 ('r2_farm_years_mean', 'dropped', 'rho 0.99 with r2_ever_farm_mean'),
]
dec = pd.DataFrame(D, columns=['column', 'role', 'reason'])
featlike = [c for c in M.columns if c.startswith(('nc_', 'r2_', 'hc_', 'mm_')) and c not in ('nc_wc_dist_nontree_m', 'r2_group_INFO_ONLY')]
sd_cols = [c for c in featlike if c.startswith('r2_') and c.endswith('_stdDev') and c not in set(dec.column)]
dec = pd.concat([dec, pd.DataFrame({'column': sd_cols, 'role': 'dropped', 'reason': 'Within-buffer SD from the run 2 table: not in run 2\'s own model, MapBiomas only'})])
missing = sorted(set(featlike) - set(dec.column)); assert not missing, missing
T = M[M.train_eligible]
def auc(c):
    x = T[[c, 'label']].dropna()
    if len(x) < 30: return np.nan
    a = roc_auc_score(x.label, x[c]); return round(max(a, 1 - a), 2)
dec['auc_mb_train'] = dec.column.map(auc)
dec['coverage'] = dec.column.map(lambda c: 'MapBiomas only' if c.startswith('r2_') else ('both (border points missing)' if c.startswith('hc_') else 'both'))
dec['mapbiomas_derived'] = dec.column.isin(['nc_H_mapbiomas10m_plantation', 'r2_ever_farm_mean', 'r2_farm_years_mean'])
keepcols = [c for c in M.columns if c not in set(dec[dec.role == 'dropped'].column)]
M2 = M[keepcols]
M2.to_csv(OUT + 'master_features_v2.csv', index=False); dec.to_csv(OUT + 'feature_decisions_v2.csv', index=False)
print(dec.role.value_counts().to_dict(), M2.shape)
print(dec[dec.role.isin(['feature', 'veto', 'pending_QC', 'benchmark'])][['column', 'role', 'auc_mb_train', 'coverage']].to_string(index=False))
