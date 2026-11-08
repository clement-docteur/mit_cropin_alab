import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
D='/home/claude/w2/'; G='/mnt/user-data/uploads/cropin_texture/'
mb=pd.read_csv(D+'w2_mb_dev_features.csv'); qc=pd.read_csv(D+'w2_qc_dev_features.csv')
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True)
A=pd.concat([mb.assign(S='MB'),qc.assign(S='QC')])
pd.set_option('display.width',250)
print('== design vars by set x label (median)')
print(A.groupby(['S','label'])[['B_n_months','B_n_drought_months','S1_n_images','S1_angle','PALSAR_year','su_n_clear']].median())
print(A.groupby(['S','label'])[['B_n_months','S1_n_images']].agg(['mean','min']))
print('PALSAR year', A.groupby('S').PALSAR_year.value_counts().to_dict())
for v in ['B_n_months','B_n_drought_months','S1_n_images','S1_angle','su_n_clear','PALSAR_year']:
    print(f'{v:20s} AUC(label) MB={roc_auc_score(mb.label,mb[v].fillna(-1)):.3f} QC={roc_auc_score(qc.label,qc[v].fillna(-1)):.3f}')
# rainfall per window
R=pd.read_csv(G+'w2_chirps.csv'); R=R[R.id.isin(mb.id)&(R.set=='mb') | R.id.isin(qc.id)&(R.set=='qc')]
w=R.groupby('set')[['rain_mm','clim_mm']].sum(); print('rain/clim ratio by set', (w.rain_mm/w.clim_mm).round(3).to_dict())
R['yr']=R.month.str[:4]; y=R.groupby(['set','yr'])[['rain_mm','clim_mm']].sum(); print((y.rain_mm/y.clim_mm).round(2).to_dict())
# raw monthly valid months by calendar month / year per set
for nm,f,ids in [('MB','w2_s2_monthly_mb.csv',mb.id),('QC','w2_s2_monthly_qc.csv',qc.id)]:
    S=pd.read_csv(G+f); S=S[S.id.isin(ids)]; S['ok']=S.B8.notna(); S['mo']=S.month.str[5:7].astype(int)
    print(nm,'valid share by month', S.groupby('mo').ok.mean().round(2).to_dict())
    print(nm,'mean n_clear', S.n_clear.mean().round(2), 'rows per id', S.groupby('id').size().median())
# S1 per-year? S1 file
s1=pd.read_csv(G+'w2_s1.csv'); print(s1.groupby('set').VV_count.describe())
