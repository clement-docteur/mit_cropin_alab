"""Meera's six measurements (Findings deck 12 Nov, slide 'Winter imagery carries the signal SAG misses'):
SAG Oct–Jan + winter (Apr–Sep) B6, B8A, SAG, NDMI, NDVI. Rebuilt from our Sentinel-2 monthly medians
(w2_s2_monthly_*), with OUR windows: 3 calendar years ending at each point's as-of year (MapBiomas 2020–2022;
QC: 3 years before clearing). Meera used 2019–2022 for both. Season value = median of yearly medians (>= 2 clear months).
B6 is not in our monthly export, so mm_B6_win is left empty until a GEE export adds it."""
import numpy as np, pandas as pd
U = '/mnt/user-data/uploads/cropin_texture/'
def build():
    S = pd.concat([pd.read_csv(U + 'w2_s2_monthly_mb.csv'), pd.read_csv(U + 'w2_s2_monthly_qc.csv')]).dropna(subset=['B8'])
    b = lambda x: S[x] / 1e4
    S['NDVI'] = (b('B8') - b('B4')) / (b('B8') + b('B4')); S['NDMI'] = (b('B8') - b('B11')) / (b('B8') + b('B11')); S['B8A_r'] = b('B8A')
    xb, xg, xr = 0.4966, 0.560, 0.6645
    v1 = np.stack([np.full(len(S), xb - xg), b('B2') - b('B3')]); v2 = np.stack([np.full(len(S), xr - xg), b('B4') - b('B3')])
    S['SAG'] = np.degrees(np.arccos(np.clip((v1 * v2).sum(0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)), -1, 1)))
    d = pd.to_datetime(S.month); S['mo'] = d.dt.month; S['yr'] = d.dt.year
    S['sg_season'] = np.where(S.mo >= 10, S.yr, S.yr - 1)
    def seas(sub, key, var):
        y = sub.groupby(['id', 'set', key])[var].agg(['median', 'size']).reset_index()
        return y[y['size'] >= 2].groupby(['id', 'set'])['median'].median()
    W = S[S.mo.between(4, 9)]; SG = S[S.mo.isin([10, 11, 12, 1])]
    out = pd.DataFrame({'mm_SAG_OctJan': seas(SG, 'sg_season', 'SAG'), 'mm_B8A_win': seas(W, 'yr', 'B8A_r'),
                        'mm_SAG_win': seas(W, 'yr', 'SAG'), 'mm_NDMI_win': seas(W, 'yr', 'NDMI'), 'mm_NDVI_win': seas(W, 'yr', 'NDVI')}).reset_index()
    out['mm_B6_win'] = np.nan
    return out
