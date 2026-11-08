"""Wave 2 – Family B (seasonal: H21–H23, H27) and Family F (radar: H25–H26) features.
All inputs already restricted (in GEE) to the 3 calendar years ending at each point's cutoff."""
import pandas as pd, numpy as np
mb = pd.read_csv('w1/w1_mb_dev_features.csv'); qc = pd.read_csv('w1/w1_qc_dev_features.csv')
keys = pd.concat([mb[['id', 'set', 'cutoff']], qc[['id', 'set', 'cutoff']]])
S = pd.concat([pd.read_csv('gee/w2_s2_monthly_mb.csv'), pd.read_csv('gee/w2_s2_monthly_qc.csv')])
S = S.drop(columns=['system:index', '.geo', 'label'], errors='ignore').dropna(subset=['B8'])
b = lambda x: S[x] / 1e4
S['NDVI'] = (b('B8') - b('B4')) / (b('B8') + b('B4')); S['NDMI'] = (b('B8') - b('B11')) / (b('B8') + b('B11'))
S['NDRE'] = (b('B8A') - b('B5')) / (b('B8A') + b('B5')); S['SWIR'] = b('B11')
xb, xg, xr = 0.4966, 0.560, 0.6645
v1 = np.stack([np.full(len(S), xb - xg), b('B2') - b('B3')]); v2 = np.stack([np.full(len(S), xr - xg), b('B4') - b('B3')])
S['SAG'] = np.degrees(np.arccos(np.clip((v1 * v2).sum(0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)), -1, 1)))
S['t'] = pd.to_datetime(S.month); S['m'] = S.t.dt.month
S['tyr'] = S.t.dt.year + (S.m - 0.5) / 12
R = pd.read_csv('gee/w2_chirps.csv')[['id', 'set', 'month', 'rain_mm', 'clim_mm']].sort_values(['id', 'set', 'month'])
g2 = R.groupby(['id', 'set'])
R['dry2'] = (g2.rain_mm.transform(lambda s: s.rolling(2).sum()) / g2.clim_mm.transform(lambda s: s.rolling(2).sum()))
S = S.merge(R[['id', 'set', 'month', 'dry2']], on=['id', 'set', 'month'], how='left')
IDX = ['NDVI', 'NDMI', 'NDRE', 'SWIR', 'SAG']
SEAS = {'DJ': [12, 1], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
rows = []
for (i, st), g in S.groupby(['id', 'set']):
    f = {'id': i, 'set': st, 'B_n_months': len(g)}
    X = np.c_[np.ones(len(g)), np.cos(2 * np.pi * g.tyr), np.sin(2 * np.pi * g.tyr)]
    for k in IDX:
        y = g[k].values
        if len(g) >= 12:
            c, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ c
            f[f'B_{k}_mean'] = c[0]; f[f'B_{k}_amp'] = np.hypot(c[1], c[2])          # H21
            ph = np.arctan2(c[2], c[1]); f[f'B_{k}_phase_cos'] = np.cos(ph); f[f'B_{k}_phase_sin'] = np.sin(ph)   # H22 (circular)
            f[f'B_{k}_resid_sd'] = res.std()
            if k in ('NDVI', 'NDMI'):                                                 # H23: standardised anomaly in drought months
                d = g.dry2.values < 0.6
                f[f'B_{k}_drought_anom'] = (res[d] / res.std()).mean() if d.sum() >= 2 and res.std() > 0 else np.nan
                f['B_n_drought_months'] = int(d.sum())
        sm = g[g.m.isin(SEAS['DJ'])][k].median(); wm = g[g.m.isin(SEAS['JJA'])][k].median()
        f[f'B_{k}_summer_minus_winter'] = sm - wm
    for s, ms in SEAS.items(): f[f'B_SAG_{s}'] = g[g.m.isin(ms)].SAG.median()
    rows.append(f)
B = pd.DataFrame(rows)
L = pd.read_csv('gee/w2_lst.csv').dropna(subset=['LST_summer', 'LST_winter'], how='all')
L = L.groupby(['id', 'set'])[['LST_summer', 'LST_winter']].first().reset_index()
L['B_LST_summer'] = L.LST_summer; L['B_LST_winter'] = L.LST_winter; L['B_LST_range'] = L.LST_summer - L.LST_winter
s1 = pd.read_csv('gee/w2_s1.csv')
F1 = pd.DataFrame({'id': s1.id, 'set': s1.set,
    'F_VV_median': s1.VV_median, 'F_VH_median': s1.VH_median, 'F_VHminusVV_median': s1.VHminusVV_median,
    'F_VV_range': s1.VV_p90 - s1.VV_p10, 'F_VH_range': s1.VH_p90 - s1.VH_p10,
    'F_VV_sd': s1.VV_stdDev, 'F_VH_sd': s1.VH_stdDev, 'F_VHminusVV_sd': s1.VHminusVV_stdDev,
    'S1_n_images': s1.VV_count, 'S1_angle': s1.angle_median})
P = pd.read_csv('gee/w2_palsar.csv')[['id', 'set', 'year', 'HH', 'HV']]
for p in ('HH', 'HV'): P[p + '_dB'] = 10 * np.log10(P[p].astype(float).clip(lower=1) ** 2) - 83.0   # JAXA gamma-0 conversion
P = P.merge(keys, on=['id', 'set']); P = P[P.year <= P.cutoff]
last = P.sort_values('year').groupby(['id', 'set']).tail(1)
F2 = pd.DataFrame({'id': last.id, 'set': last.set, 'F_PALSAR_HH': last.HH_dB, 'F_PALSAR_HV': last.HV_dB,
                   'F_PALSAR_HVminusHH': last.HV_dB - last.HH_dB, 'PALSAR_year': last.year})
P5 = P[P.year > P.cutoff - 3].groupby(['id', 'set']).HV_dB.std().rename('F_PALSAR_HV_sd3yr').reset_index()   # 3-year as-of window (reviewer fix)
def build(d):
    return d.merge(B, on=['id', 'set'], how='left').merge(L[['id', 'set', 'B_LST_summer', 'B_LST_winter', 'B_LST_range']], on=['id', 'set'], how='left') \
            .merge(F1, on=['id', 'set'], how='left').merge(F2, on=['id', 'set'], how='left').merge(P5, on=['id', 'set'], how='left')
MB, QC = build(mb), build(qc)
MB.to_csv('w2_mb_dev_features.csv', index=False); QC.to_csv('w2_qc_dev_features.csv', index=False)
fam = {p: [c for c in MB.columns if c.startswith(p + '_')] for p in 'BF'}
print({k: len(v) for k, v in fam.items()}, MB.shape, QC.shape)
for nm, d in [('MB', MB), ('QC', QC)]:
    na = d[fam['B'] + fam['F']].isna().mean(); print(nm, 'missing >1%:', na[na > 0.01].round(3).to_dict())
print('months per point (median MB, QC):', MB.B_n_months.median(), QC.B_n_months.median(), '| drought months median:', MB.B_n_drought_months.median(), QC.B_n_drought_months.median())
print('PALSAR year used:', MB.PALSAR_year.value_counts().to_dict(), QC.PALSAR_year.value_counts().to_dict())
