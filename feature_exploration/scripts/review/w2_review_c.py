import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
D='/home/claude/w2/'; G='/mnt/user-data/uploads/cropin_texture/'
mb=pd.read_csv(D+'w2_mb_dev_features.csv'); qc=pd.read_csv(D+'w2_qc_dev_features.csv')
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True)
mb['lon']=mb.block.str.split('_').str[0].astype(float)*.5+.25; mb['lat']=mb.block.str.split('_').str[1].astype(float)*.5+.25
qc['lon']=qc.block.str.split('_').str[0].astype(float)*.25+.125; qc['lat']=qc.block.str.split('_').str[1].astype(float)*.25+.125
print('lat range MB',mb.lat.min(),mb.lat.max(),' QC',qc.lat.min(),qc.lat.max())
print('MB lat by state', mb.groupby('state').lat.agg(['min','max']).to_dict())
P=['B_SAG_phase_sin','B_SAG_phase_cos','B_NDRE_phase_sin','B_NDRE_phase_cos','B_SWIR_phase_sin','B_SWIR_phase_cos','B_NDVI_phase_sin','B_NDVI_phase_cos']
FB=['B_SAG_SON','B_SAG_phase_sin','B_NDRE_phase_sin','B_SWIR_phase_sin','B_SAG_DJ','B_SWIR_phase_cos','B_SAG_mean']
print('\n== spearman with lat/lon/elev (MB all, MB within label 0, MB within label 1, QC)')
rows=[]
for f in P+['B_SAG_SON','B_SAG_DJ','B_SAG_mean','A_SAG','F_VV_median','F_VHminusVV_median']:
    r=dict(f=f)
    for nm,d in [('MB',mb),('MB0',mb[mb.label==0]),('MB1',mb[mb.label==1]),('QC',qc)]:
        r[nm+'_lat']=spearmanr(d[f],d.lat,nan_policy='omit')[0]
    r['MB0_elev']=spearmanr(mb[mb.label==0][f],mb[mb.label==0].elev,nan_policy='omit')[0]
    r['MB_AUC']=roc_auc_score(mb.label,mb[f].fillna(mb[f].median()))
    r['QC_AUC']=roc_auc_score(qc.label,qc[f].fillna(qc[f].median()))
    for s in ['PR','SC','RS']:
        d=mb[mb.state==s]; r['AUC_'+s]=roc_auc_score(d.label,d[f].fillna(d[f].median()))
    rows.append(r)
pd.set_option('display.width',250); print(pd.DataFrame(rows).round(3).to_string())
print('label vs lat AUC MB', roc_auc_score(mb.label,mb.lat), 'QC',roc_auc_score(qc.label,qc.lat))
print('class counts by state', pd.crosstab(mb.state,mb.label).to_dict())
# lat-only and lat+lon model vs phase-only: does phase add beyond location?
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
def oof(cols,d=mb):
    s=np.full(len(d),np.nan); tr=(d.agree>=2).values; y=d.label.values
    for a,b in GroupKFold(5).split(d,y,d.block):
        a=a[tr[a]]; s[b]=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000)).fit(d.iloc[a][cols],y[a]).predict_proba(d.iloc[b][cols])[:,1]
    return s
PH=['B_SAG_phase_sin','B_NDRE_phase_sin','B_SWIR_phase_sin','B_SWIR_phase_cos']
for nm,c in [('lat+lon+elev',['lat','lon','elev']),('phase4',PH),('phase4+lat+lon+elev',PH+['lat','lon','elev']),('B',FB),('B+lat+lon+elev',FB+['lat','lon','elev']),('B minus phase',['B_SAG_SON','B_SAG_DJ','B_SAG_mean'])]:
    s=oof(c); print(f'{nm:22s} OOF AUC {roc_auc_score(mb.label,s):.3f}  within-state:',{st:round(roc_auc_score(mb.label[mb.state==st],s[mb.state==st]),3) for st in ['PR','SC','RS']})
# Family B within-state OOF, also within-state retrain
s=oof(FB); 
for st in ['PR','SC','RS']:
    d=mb[mb.state==st].reset_index(drop=True); 
    try: ss=oof(FB,d); print(st,'retrained-within-state OOF AUC',round(roc_auc_score(d.label,ss),3),'n',len(d),'n1',d.label.sum(),'blocks',d.block.nunique())
    except Exception as e: print(st,e)
