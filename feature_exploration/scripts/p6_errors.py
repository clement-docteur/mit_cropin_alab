"""Phase 6 – error breakdown for the final candidates (Lean 4, P4) vs the baseline, on dev data only.
Part 1: where forest error and missed plantations concentrate (MapBiomas dev, blocked OOF, 5% threshold).
Part 2: hard-case proxies (HC1, HC2, HC9, HC10).
Part 3: QC dev errors, listed point by point.
Part 4: where the MapBiomas 10 m map fails at Cropin's alerts (M1–M4, M6; QC dev; Task 1.5)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p6_common as c
mb, qc = c.load(); y = mb.label.values; trm = (mb.agree >= 2).values
p0 = pd.read_csv('/home/claude/p0/mb_dev_table.csv')[['id', 'lon', 'lat']].merge(
    pd.read_csv('/mnt/user-data/uploads/Cropin - A Lab/Feature Exploration/results/phase6/mb_dev_forest_history.csv')[['id', 'forest_all_1985_2012', 'first_forest_year', 'NM_MUN']], on='id')
# (Phase 0's forest_all flag was all 0 because CLASS_1992 is stored as text in the dbf; recomputed here.)
mb = mb.merge(p0, on='id', how='left'); assert len(mb) == len(y)
mb['NM_MUN'] = mb.NM_MUN.astype(str).map(lambda t: t.encode('latin-1').decode('utf-8', errors='replace'))   # dbf text is UTF-8 read as latin-1
qd = pd.read_csv('/home/claude/p0/qc_dev_table.csv')[['id', 'lon', 'lat', 'area_ha']]
qc = qc.merge(qd, on='id', how='left')
CAND = {'Baseline': c.BASE, 'Lean 4': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_SAG_SON', 'A_sg_NDVI'],
        'P4': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_NDRE_phase_sin', 'B_SAG_SON', 'A_sg_NDVI', 'B_SWIR_phase_sin', 'B_SAG_mean', 'B_SWIR_phase_cos', 'B_SAG_DJ', 'A_SAG']}
def wil(k, n):
    if n == 0: return ''
    p = k / n; z = 1.96; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{max(0, m - h):.2f}–{min(1, m + h):.2f}'
S, T, Q = {}, {}, {}
for k, cols in CAND.items():
    S[k] = c.oof(mb, cols, trm); T[k] = c.thr(y, S[k], 0.05)
    Q[k] = c.lr().fit(mb[trm][cols], y[trm]).predict_proba(qc[cols])[:, 1]
# ---------- Part 1 + 2: groups on MapBiomas dev ----------
mb['edge_dist'] = pd.cut(mb.wc_dist_nontree_m, [-1, 10, 30, 100, 1e9], labels=['≤10 m', '10–30 m', '30–100 m', '>100 m'])
mb['tree100'] = pd.cut(mb.wc_tree_frac_100m, [-0.01, 0.5, 0.8, 1.01], labels=['<50%', '50–80%', '>80%'])
mb['slope_g'] = pd.cut(mb.slope, [-1, 5, 15, 90], labels=['<5°', '5–15°', '>15°'])
mb['ter'] = pd.qcut(mb.su_n_clear, 3, labels=['few', 'mid', 'many'])
G = {'All': mb.index == mb.index,
     'Border flag = 1 (HC9)': mb.border == 1, 'Border flag = 0': mb.border == 0,
     **{f'Distance to non-tree {g} (HC9)': mb.edge_dist == g for g in mb.edge_dist.cat.categories},
     **{f'Tree cover within 100 m {g}': mb.tree100 == g for g in mb.tree100.cat.categories},
     'Interpreters agree 3/3': mb.agree == 3, 'Interpreters agree 2/3': mb.agree == 2, 'Interpreters agree 1/3 (not trained on)': mb.agree == 1,
     **{f'State {s}': mb.state == s for s in ('PR', 'SC', 'RS')},
     'Atlantic Forest': mb.biome.str.startswith('MATA'), 'Pampa': mb.biome == 'PAMPA',
     **{f'Slope {g}': mb.slope_g == g for g in mb.slope_g.cat.categories},
     **{f'Clear images: {g}': mb.ter == g for g in ('few', 'mid', 'many')},
     'HC2 proxy: forest in every year since 1985 (older)': (mb.forest_all_1985_2012 == 1) | (mb.label == 1),
     'HC2 proxy: forest only since 1986–2012 (younger / secondary)': (mb.forest_all_1985_2012 == 0) | (mb.label == 1),
     'HC2 proxy: forest only since 2001–2013 (youngest)': (mb.first_forest_year > 2000) | (mb.label == 1),
     'Disturbed natural forest (D events ≥1)': (mb.D_n_events > 0) | (mb.label == 1),
     'HC1 proxy: elevation ≥ 800 m (Araucaria zone, PR/SC)': (mb.elev >= 800) & mb.state.isin(['PR', 'SC']),
     'HC10 proxy: Curitiba metro (bracatinga area)': mb.lat.between(-25.8, -25.0) & mb.lon.between(-49.7, -48.9)}
rows = []
for g, m in G.items():
    m = np.asarray(m); nf, npl = int(((y == 0) & m).sum()), int(((y == 1) & m).sum())
    r = dict(group=g, n_forest=nf, n_plant=npl)
    for k in CAND:
        kf = int((S[k][m & (y == 0)] > T[k]).sum()); r[f'{k} FE'] = kf / nf if nf else np.nan; r[f'{k} FE CI'] = wil(kf, nf)
        r[f'{k} recall'] = (S[k][m & (y == 1)] > T[k]).mean() if npl else np.nan
    rows.append(r)
GR = pd.DataFrame(rows)
# spatial clusters of forest errors (Lean 4 and P4)
mb['err_L4'] = (y == 0) & (S['Lean 4'] > T['Lean 4']); mb['err_P4'] = (y == 0) & (S['P4'] > T['P4'])
CL = mb[mb.label == 0].groupby('block').agg(n_forest=('id', 'size'), err_L4=('err_L4', 'sum'), err_P4=('err_P4', 'sum'),
                                             lat=('lat', 'mean'), lon=('lon', 'mean'), state=('state', lambda s: s.mode()[0]),
                                             municipalities=('NM_MUN', lambda s: ', '.join(s.value_counts().index[:3]))).reset_index()
CL = CL[(CL.err_L4 >= 3) | (CL.err_P4 >= 3)].sort_values('err_L4', ascending=False)
# ---------- Part 3: QC dev point list ----------
yq = qc.label.values
QL = qc[['id', 'label', 'lon', 'lat', 'area_ha', 'clear_month', 'wc_dist_nontree_m', 'wc_tree_frac_100m', 'H_mapbiomas10m_plantation', 'H_gft_planted', 'nfw']].copy()
for k in CAND: QL[f'score_{k}'] = Q[k]; QL[f'flag_{k}'] = (Q[k] > T[k]).astype(int)
QL['error_type'] = np.select([(yq == 0) & ((QL['flag_Lean 4'] == 1) | (QL['flag_P4'] == 1)), (yq == 1) & (QL['flag_Lean 4'] == 0) & (QL['flag_P4'] == 0)],
                             ['FOREST called plantation (dangerous)', 'plantation missed'], '')
QE = QL[QL.error_type != ''].sort_values(['error_type', 'score_Lean 4'], ascending=[True, False])
qrows = []
for g, m in {'QC all': np.ones(len(qc), bool), 'QC ≤10 m from non-tree': qc.wc_dist_nontree_m <= 10, 'QC >10 m from non-tree': qc.wc_dist_nontree_m > 10,
             'QC tree cover within 100 m <50%': qc.wc_tree_frac_100m < 0.5, 'QC tree cover within 100 m ≥50%': qc.wc_tree_frac_100m >= 0.5}.items():
    m = np.asarray(m); nf, npl = int(((yq == 0) & m).sum()), int(((yq == 1) & m).sum()); r = dict(group=g, n_forest=nf, n_plant=npl)
    for k in CAND:
        kf = int((Q[k][m & (yq == 0)] > T[k]).sum()); r[f'{k} FE'] = kf / nf if nf else np.nan; r[f'{k} FE CI'] = wil(kf, nf)
        r[f'{k} recall'] = (Q[k][m & (yq == 1)] > T[k]).mean() if npl else np.nan
    qrows.append(r)
QG = pd.DataFrame(qrows)
# ---------- Part 4: where the MapBiomas 10 m map fails on QC dev (Task 1.5) ----------
m10 = pd.read_csv('/home/claude/p0/gee/p0_mapbiomas10m.csv'); m10 = m10[m10.set == 'qc'].set_index('id')
pl = qc[qc.label == 1].copy(); pl['mb_hit'] = pl.H_mapbiomas10m_plantation == 1
def rate(d): return f"{d.mb_hit.mean():.2f} ({int(d.mb_hit.sum())}/{len(d)})"
M = []
pl['patch_small'] = pl.wc_patch_px < 1024
M.append(('M1 small stands: plantation in tree patch < 10 ha (WorldCover)', rate(pl[pl.patch_small]), 'vs ≥ 10 ha', rate(pl[~pl.patch_small])))
pl['recent'] = pl.D_years_since_last_event <= 5
M.append(('M2 harvested in the last 5 years before cutoff', rate(pl[pl.recent]), 'vs not', rate(pl[~pl.recent])))
# M3: does the 10 m map contain any plantation class near QC points? (block-level presence among all QC/MB points)
allm = pd.read_csv('/home/claude/p0/gee/p0_mapbiomas10m.csv')
# dev points only (MapBiomas dev + QC dev); sealed points are not used even for map-class presence
pts = pd.concat([mb[['id', 'lon', 'lat']].assign(set='mb'), qc[['id', 'lon', 'lat']].assign(set='qc')])
allm = allm.merge(pts, on=['id', 'set']); allm['blk'] = np.floor(allm.lon / 0.5).astype(int).astype(str) + '_' + np.floor(allm.lat / 0.5).astype(int).astype(str)
has9 = allm.groupby('blk').classification_2022.apply(lambda s: (s == 9).any())
pl['blk'] = np.floor(pl.lon / 0.5).astype(int).astype(str) + '_' + np.floor(pl.lat / 0.5).astype(int).astype(str)
pl['blk_has9'] = pl.blk.map(has9).fillna(False)
M.append(('M3 0.5° block where the 10 m map shows no plantation at any sampled point (2022)', rate(pl[~pl.blk_has9]), 'vs block with plantation', rate(pl[pl.blk_has9])))
pl['nfw_low'] = pl.nfw < 130
M.append(('M4 NFW disagrees with MapBiomas-forest (NFW < 130 = not natural)', rate(pl[pl.nfw_low]), 'vs NFW ≥ 130', rate(pl[~pl.nfw_low])))
yrs = [f'classification_{y}' for y in range(2017, 2026)]
share = (m10.loc[pl.id, yrs] == 9).mean().round(2)
M.append(('M6 share of QC plantations mapped as plantation, by map year', '; '.join(f'{k[-4:]} {v}' for k, v in share.items()), '', ''))
MT = pd.DataFrame(M, columns=['hypothesis', 'MapBiomas catches plantation (group)', 'comparison', 'MapBiomas catches plantation (comparison)'])
for nm, d in [('p6_groups_mb', GR), ('p6_clusters_mb', CL), ('p6_qc_groups', QG), ('p6_qc_error_points', QE), ('p6_mapbiomas_failures', MT)]: d.to_csv(f'{nm}.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 100); pd.set_option('display.max_colwidth', 70)
cols = ['group', 'n_forest', 'n_plant'] + [f'{k} {s}' for k in CAND for s in ('FE', 'FE CI', 'recall')]
print(GR[cols].round(3).to_string()); print(CL.to_string()); print(QG[cols].round(3).to_string())
print(QE.drop(columns=['lon', 'lat']).round(3).to_string()); print(MT.to_string())
