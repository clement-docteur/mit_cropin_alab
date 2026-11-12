import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
D='/home/claude/w2/'; G='/mnt/user-data/uploads/cropin_texture/'
mb=pd.read_csv(D+'w2_mb_dev_features.csv'); qc=pd.read_csv(D+'w2_qc_dev_features.csv'); mb=mb[mb.A_su_NDVI>0.6]
out=[]
for nm,f,F in [('MB','w2_s2_monthly_mb.csv',mb),('QC','w2_s2_monthly_qc.csv',qc)]:
    S=pd.read_csv(G+f); S=S[S.id.isin(F.id)].dropna(subset=['B8'])
    b=lambda x:S[x].values/1e4
    v1=np.c_[np.full(len(S),.4966-.56),b('B2')-b('B3')]; v2=np.c_[np.full(len(S),.6645-.56),b('B4')-b('B3')]
    S['SAG']=np.degrees(np.arccos(np.clip((v1*v2).sum(1)/np.linalg.norm(v1,axis=1)/np.linalg.norm(v2,axis=1),-1,1)))
    S['NDRE']=(b('B8A')-b('B5'))/(b('B8A')+b('B5')); S['SWIR']=b('B11'); S['NDVI']=(b('B8')-b('B4'))/(b('B8')+b('B4'))
    S['t']=S.month.str[:4].astype(int)+(S.month.str[5:7].astype(int)-.5)/12
    for i,g in S.groupby('id'):
        r={'S':nm,'id':i}
        t=g.t.values; tc=t-t.mean()
        for k in ['SAG','NDRE','SWIR','NDVI']:
            yv=g[k].values
            X=np.c_[np.ones(len(g)),np.cos(2*np.pi*t),np.sin(2*np.pi*t)]
            X2=np.c_[X,tc]
            c=np.linalg.lstsq(X,yv,rcond=None)[0]; c2=np.linalg.lstsq(X2,yv,rcond=None)[0]
            r[k+'_psin']=np.sin(np.arctan2(c[2],c[1])); r[k+'_psin_detr']=np.sin(np.arctan2(c2[2],c2[1])); r[k+'_slope']=c2[3]
            r[k+'_pcos_detr']=np.cos(np.arctan2(c2[2],c2[1]))
        out.append(r)
O=pd.DataFrame(out)
O=O.merge(pd.concat([mb[['id','label']].assign(S='MB'),qc[['id','label']].assign(S='QC')]),on=['S','id'])
for k in ['SAG','NDRE','SWIR','NDVI']:
    s=[]
    for nm in ['MB','QC']:
        d=O[O.S==nm]
        s.append(f"{nm}: psin {roc_auc_score(d.label,d[k+'_psin']):.3f} psin_detrended {roc_auc_score(d.label,d[k+'_psin_detr']):.3f} pcos_detr {roc_auc_score(d.label,d[k+'_pcos_detr']):.3f} slope {roc_auc_score(d.label,d[k+'_slope']):.3f} corr(psin,slope) {np.corrcoef(d[k+'_psin'],d[k+'_slope'])[0,1]:.2f}")
    print(k,' | '.join(s))
print(O.groupby(['S','label'])[['SAG_slope','NDVI_slope']].median())
