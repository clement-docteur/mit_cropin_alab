"""Combined model – master feature table, v1 (option A: merge existing tables).
Universe: our points (MapBiomas dev + lockbox, Cropin QC dev + held-back). Label 1 = plantation.
Teammate features joined by exact ID (MapBiomas TARGETID = V-number; QC GeoJSON index = Q-number).
Missing teammate features are left empty (decision 13 Nov 2026)."""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0, '/home/claude/master'); import harvest_features as H
REPO = '/tmp/claude-0/-home-claude/999d0128-2648-5e0d-9375-1863dd874899/scratchpad/repo/'
U = '/mnt/user-data/uploads/cropin_texture/'
OUT = '/home/claude/master/out/'; import os; os.makedirs(OUT, exist_ok=True)

# ---------------- universe + our features ----------------
OURS = {  # column -> (new name, short definition)
 'B_SAG_phase_sin': 'SAG seasonal timing (sine of harmonic phase), monthly S2, 3 yrs to as-of year',
 'B_SAG_SON': 'SAG median Sep–Nov, 3 yrs', 'A_su_NDMI': 'NDMI summer composite Dec–Mar, ONE season',
 'A_sg_NDVI': 'NDVI composite Oct–Jan, ONE season', 'B_NDRE_phase_sin': 'NDRE seasonal timing (sine), 3 yrs',
 'B_SWIR_phase_sin': 'SWIR1 seasonal timing (sine), 3 yrs', 'B_SWIR_phase_cos': 'SWIR1 seasonal timing (cosine), 3 yrs',
 'B_SAG_mean': 'SAG 3-yr harmonic mean level', 'B_SAG_DJ': 'SAG median Dec–Jan, 3 yrs (Cropin window)',
 'A_SAG': 'SAG Oct–Jan composite, ONE season (Cropin-style)', 'chm_mean_r15': 'Meta canopy height mean, 15 m radius (~2019-20)',
 'chm_stdDev_r15': 'Meta canopy height SD, 15 m radius', 'E_eth_h': 'ETH canopy height 2020 at point',
 'E_eth_sd50': 'ETH canopy height SD within 50 m', 'nfw': 'Natural Forests of the World 2020 probability (B0/250)',
 'D_years_continuous_forest': 'Landsat: years of continuous closed forest before as-of year', 'D_max_nbr_drop': 'Landsat: largest NBR drop, 25 yrs',
 'D_cc_n_breaks': 'CCDC breaks (Landsat, fit to as-of year)', 'D_h_total_loss_px': 'Hansen loss pixels within 150 m, to as-of year',
 'D_ndvi_1985_89': 'Landsat NDVI 1985–89', 'D_lt_n_vertices': 'LandTrendr vertices (fit to as-of year)',
 'F_VH_median': 'Sentinel-1 VH median dB, 3 yrs', 'F_VHminusVV_median': 'Sentinel-1 VH−VV median dB, 3 yrs',
 'H_mapbiomas10m_plantation': 'MapBiomas 10 m map = plantation in as-of year (MAPBIOMAS-DERIVED)',
 'wc_dist_nontree_m': 'Distance to non-tree land (ESA WorldCover 2021), for the edge category',
}
META = ['id', 'set', 'label', 'agree', 'border', 'state', 'block', 'cutoff', 'A_su_NDVI']
parts = []
for f, split in (('/home/claude/w3/w3_mb_dev_features.csv', 'dev'), ('/home/claude/p7/sealed/w3_mb_dev_features.csv', 'lockbox'),
                 ('/home/claude/w3/w3_qc_dev_features.csv', 'dev'), ('/home/claude/p7/sealed/w3_qc_dev_features.csv', 'heldback')):
    d = pd.read_csv(f); d['split'] = split
    parts.append(d[[c for c in META + list(OURS) if c in d.columns] + ['split']])
M = pd.concat(parts, ignore_index=True)
xy = pd.concat([pd.read_csv('/home/claude/p0/mb_dev_table.csv'), pd.read_csv('/home/claude/p0/LOCKBOX_mb_table_DO_NOT_OPEN.csv')])[['id', 'lon', 'lat']].assign(set='mb')
q = pd.read_csv('/home/claude/p0/pts/qc_points_all.csv') if os.path.exists('/home/claude/p0/pts/qc_points_all.csv') else None
qxy = pd.concat([pd.read_csv('/home/claude/p0/qc_dev_table.csv'), pd.read_csv('/home/claude/p0/HELDBACK_qc_table_DO_NOT_OPEN.csv')])[['id', 'lon', 'lat', 'area_ha']].assign(set='qc')
M = M.merge(pd.concat([xy, qxy]), on=['id', 'set'], how='left')
cd = pd.read_csv('/home/claude/p0/qc_clearing_dates.csv') if os.path.exists('/home/claude/p0/qc_clearing_dates.csv') else None
if cd is not None and 'clear_month' in cd:
    M = M.merge(cd[['id', 'clear_month']].assign(set='qc'), on=['id', 'set'], how='left')
M = M.rename(columns={'A_su_NDVI': 'tree_ndvi_summer', 'cutoff': 'asof_year'}).rename(columns={c: 'nc_' + c for c in OURS})

# 3-year summer and winter NDMI (median of yearly medians; >= 2 clear months per season)
S = pd.concat([pd.read_csv(U + 'w2_s2_monthly_mb.csv'), pd.read_csv(U + 'w2_s2_monthly_qc.csv')]).dropna(subset=['B8'])
S['NDMI'] = (S.B8 - S.B11) / (S.B8 + S.B11); dt = pd.to_datetime(S.month); S['mo'] = dt.dt.month; S['yr'] = dt.dt.year
S['ss'] = np.where(S.mo == 12, S.yr, S.yr - 1)
def three_yr(sub, key):
    y = sub.groupby(['id', 'set', key]).NDMI.agg(['median', 'size']).reset_index()
    return y[y['size'] >= 2].groupby(['id', 'set'])['median'].median()
W = pd.DataFrame({'nc_su3_NDMI': three_yr(S[S.mo.isin([12, 1, 2, 3])], 'ss'), 'nc_wi3_NDMI': three_yr(S[S.mo.between(4, 9)], 'yr')}).reset_index()
M = M.merge(W, on=['id', 'set'], how='left')

# ---------------- flags ----------------
M['tree_filter_pass'] = np.where(M.set == 'mb', M.tree_ndvi_summer > 0.6, True)
M['train_eligible'] = (M.set == 'mb') & (M.split == 'dev') & (M.agree >= 2) & M.tree_filter_pass

M['team_pid'] = np.where(M.set == 'mb', 'V' + M.id.astype(str), 'Q' + M.id.astype(int).map('{:03d}'.format))

# ---------------- run 2 table (MapBiomas points only, 2022 imagery) ----------------
hp = pd.read_csv(REPO + 'public_data/harvest_cycles/points.csv')
r2 = pd.read_csv(REPO + 'cropin_model_features_2022_updated.csv')
R2COLS = [c for c in r2.columns if c.endswith(('_mean', '_stdDev'))]
r2 = r2.merge(hp[['pid', 'pt_id']], on='pt_id', how='left')[['pid', 'plantation', 'group'] + R2COLS]
r2 = r2.rename(columns={c: 'r2_' + c for c in R2COLS}).rename(columns={'pid': 'team_pid', 'plantation': 'r2_label_check', 'group': 'r2_group_INFO_ONLY'})
M = M.merge(r2, on='team_pid', how='left')

# ---------------- harvest cycles (MapBiomas + QC, Landsat to 2020) ----------------
_, hc, CAN, OPN = H.run(REPO + 'public_data/harvest_cycles')
HCKEEP = ['n_years', 'n_cuts', 'n_cycles', 'max_drop', 'regrow_speed', 'years_since_last_cut', 'forest_since_1985', 'abrupt_start',
          's_std', 's_min', 's_max_drop3', 's_frac_below_forest', 'has_cut', 'has_cycle', 'two_plus_cycles', 'hansen_loss', 'hansen_gain', 'tc2000']
hc = hc[HCKEEP].rename(columns=lambda c: 'hc_' + c).reset_index().rename(columns={'pid': 'team_pid'})
M = M.merge(hc, on='team_pid', how='left')

# ---------------- checks ----------------
chk = {}
m = M.r2_label_check.notna(); chk['run2 rows matched'] = int(m.sum()); chk['run2 label agreement'] = float((M.loc[m, 'r2_label_check'] == M.loc[m, 'label']).mean())
qc_geo = hp[hp.dataset == 'qc'].set_index('pid')
mq = M[M.set == 'qc'].join(qc_geo[['lon', 'lat']], on='team_pid', rsuffix='_team')
chk['QC id->Q-number max distance (m)'] = float((np.hypot(mq.lon - mq.lon_team, mq.lat - mq.lat_team) * 111000).max())
M = M.drop(columns=['r2_label_check'])
cov = M.groupby(['set', 'split']).agg(n=('id', 'size'), plantation=('label', 'sum'), run2=('r2_height_mean', lambda s: s.notna().sum()),
                                       harvest=('hc_n_cycles', lambda s: s.notna().sum()), su3=('nc_su3_NDMI', lambda s: s.notna().sum()))
# AlphaEarth (separate file, 64 dims, as-of year)
A = pd.read_csv(U + 'p7_alphaearth.csv'); A = A[['id', 'set'] + [f'A{i:02d}' for i in range(64)]].rename(columns=lambda c: 'ae_' + c if c.startswith('A') else c)
A = M[['id', 'set', 'split']].merge(A, on=['id', 'set'], how='left')
M.insert(0, 'uid', M.set + '_' + M.id.astype(str))
A.insert(0, 'uid', A.set + '_' + A.id.astype(str))
M.to_csv(OUT + 'master_features_v1.csv', index=False); A.to_csv(OUT + 'master_alphaearth_v1.csv', index=False)
cov.to_csv(OUT + 'coverage_v1.csv')
json.dump({'checks': chk, 'harvest_levels': {'closed_canopy_nbr': CAN, 'open_land_nbr': OPN}, 'shape': list(M.shape)}, open(OUT + 'build_checks_v1.json', 'w'), indent=1)
print(json.dumps(chk, indent=1)); print(cov); print(M.shape)
