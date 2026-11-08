"""Phase 0 – build analysis tables (MapBiomas dev, QC dev). Lockbox / held-back rows are
written to separate files and NOT loaded by any analysis script."""
import pandas as pd, numpy as np
P = pd.read_csv('pts/mb_points_all.csv'); Q = pd.read_csv('pts/qc_points_all.csv')
s2 = pd.read_csv('gee/p0_mb_s2.csv').drop(columns=['system:index', '.geo'], errors='ignore')
nfw = pd.read_csv('gee/p0_nfw.csv')[['id', 'set', 'nfw']]
chm = pd.read_csv('gee/p0_chm.csv')
chm = chm.pivot_table(index=['id', 'set'], columns='radius_m', values=['mean', 'stdDev', 'p90']).reset_index()
chm.columns = [c[0] if c[1] == '' else f'chm_{c[0]}_r{c[1]}' for c in chm.columns]
m10 = pd.read_csv('gee/p0_mapbiomas10m.csv').drop(columns=['system:index', '.geo', 'label'], errors='ignore')

def sag(d, p):
    xb, xg, xr = 0.4966, 0.560, 0.6645
    yb, yg, yr = d[p+'B2']/1e4, d[p+'B3']/1e4, d[p+'B4']/1e4
    v1 = np.stack([np.full(len(d), xb-xg), yb-yg]); v2 = np.stack([np.full(len(d), xr-xg), yr-yg])
    cos = (v1*v2).sum(0) / (np.linalg.norm(v1, axis=0)*np.linalg.norm(v2, axis=0))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

# ---- MapBiomas ----
mb = P.merge(s2.drop(columns=['label']), on='id').merge(nfw[nfw.set=='mb'].drop(columns='set'), on='id') \
      .merge(chm[chm.set=='mb'].drop(columns='set'), on='id') \
      .merge(m10[m10.set=='mb'].drop(columns='set')[['id', 'classification_2022']], on='id')
mb['SAG'] = sag(mb, 'sg_'); mb['mb10_plantation'] = (mb.classification_2022 == 9).astype(int)
assert len(mb) == len(P)
mb[mb.split == 'lockbox'].to_csv('LOCKBOX_mb_table_DO_NOT_OPEN.csv', index=False)
mb = mb[mb.split == 'dev']; mb.to_csv('mb_dev_table.csv', index=False)
# ---- QC ----
qs = pd.read_csv('gee/p0_qc_s2_seasons.csv').drop(columns=['system:index', '.geo', 'label'], errors='ignore')
cd = pd.read_csv('qc_clearing_dates.csv')[['id', 'clear_month', 'clearing_detected', 'season_start']]
qc = Q.merge(cd, on='id').dropna(subset=['season_start'])
qc['season_start'] = qc.season_start.astype(int)
qc = qc.merge(qs, on=['id', 'season_start']).merge(nfw[nfw.set=='qc'].drop(columns='set'), on='id') \
       .merge(chm[chm.set=='qc'].drop(columns='set'), on='id').merge(m10[m10.set=='qc'].drop(columns='set'), on='id')
qc['SAG'] = sag(qc, 'sg_')
yr = (qc.season_start + 1).clip(upper=2025)   # MapBiomas class in the year the pre-clearing season ends (latest map: 2025)
qc['mb10_plantation'] = [int(r[f'classification_{y}'] == 9) for (_, r), y in zip(qc.iterrows(), yr)]
qc[qc.split == 'heldback'].to_csv('HELDBACK_qc_table_DO_NOT_OPEN.csv', index=False)
qc = qc[qc.split == 'dev']; qc.to_csv('qc_dev_table.csv', index=False)
print('MB dev', mb.shape, mb.label.value_counts().to_dict(), '| QC dev', qc.shape, qc.label.value_counts().to_dict())
print('MB dev su_NDVI<=0.6:', (mb.su_NDVI <= 0.6).sum(), '| QC dev su_NDVI<=0.6:', (qc.su_NDVI <= 0.6).sum())
print(mb.groupby('label').SAG.median().round(1).to_dict(), qc.groupby('label').SAG.median().round(1).to_dict())
