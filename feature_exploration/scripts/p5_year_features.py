"""Phase 5 – rebuild the candidate models' features from the monthly Sentinel-2 medians, for the full
3-year window and for each single calendar year (MapBiomas 2020, 2021, 2022; QC 2023, 2024, 2025).
'M_' = monthly-based versions of the Wave 1 composite features (summer NDMI, Oct–Jan NDVI, Oct–Jan SAG)."""
import pandas as pd, numpy as np
S = pd.concat([pd.read_csv('gee/w2_s2_monthly_mb.csv'), pd.read_csv('gee/w2_s2_monthly_qc.csv')])
S = S.drop(columns=['system:index', '.geo'], errors='ignore').dropna(subset=['B8'])
b = lambda x: S[x] / 1e4
S['NDVI'] = (b('B8') - b('B4')) / (b('B8') + b('B4')); S['NDMI'] = (b('B8') - b('B11')) / (b('B8') + b('B11'))
S['NDRE'] = (b('B8A') - b('B5')) / (b('B8A') + b('B5')); S['SWIR'] = b('B11')
xb, xg, xr = 0.4966, 0.560, 0.6645
v1 = np.stack([np.full(len(S), xb - xg), b('B2') - b('B3')]); v2 = np.stack([np.full(len(S), xr - xg), b('B4') - b('B3')])
S['SAG'] = np.degrees(np.arccos(np.clip((v1 * v2).sum(0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)), -1, 1)))
S['t'] = pd.to_datetime(S.month); S['y'] = S.t.dt.year; S['m'] = S.t.dt.month; S['tyr'] = S.y + (S.m - 0.5) / 12
def feats(g, min_months):
    f = {'n_months': len(g)}
    if len(g) >= min_months:
        X = np.c_[np.ones(len(g)), np.cos(2 * np.pi * g.tyr), np.sin(2 * np.pi * g.tyr)]
        for k in ('SAG', 'NDRE', 'SWIR'):
            c, *_ = np.linalg.lstsq(X, g[k].values, rcond=None); ph = np.arctan2(c[2], c[1])
            f[f'B_{k}_phase_sin'] = np.sin(ph); f[f'B_{k}_phase_cos'] = np.cos(ph)
    f['B_SAG_SON'] = g[g.m.isin([9, 10, 11])].SAG.median(); f['B_SAG_DJ'] = g[g.m.isin([12, 1])].SAG.median()
    f['B_SAG_mean'] = g.SAG.mean()
    f['M_su_NDMI'] = g[g.m.isin([12, 1, 2, 3])].NDMI.median(); f['M_sg_NDVI'] = g[g.m.isin([10, 11, 12, 1])].NDVI.median()
    f['M_SAG_octjan'] = g[g.m.isin([10, 11, 12, 1])].SAG.median()
    return f
rows = []
for (i, st), g in S.groupby(['id', 'set']):
    rows.append(dict(id=i, set=st, window='3yr', **feats(g, 12)))
    for yy, gy in g.groupby('y'): rows.append(dict(id=i, set=st, window=str(yy), **feats(gy, 8)))
F = pd.DataFrame(rows); F.to_csv('p5_year_features.csv', index=False)
print(F.groupby(['set', 'window']).size().to_dict())
print('single-year rows lacking a phase fit (<8 clear months):', F[F.window != '3yr'].B_SAG_phase_sin.isna().mean().round(3))
