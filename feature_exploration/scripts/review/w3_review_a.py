import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
D='/home/claude/w3/'
mb=pd.read_csv(D+'w3_mb_dev_features.csv'); qc=pd.read_csv(D+'w3_qc_dev_features.csv')
print('mb raw',len(mb)); mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True); print('mb filt',len(mb), mb.label.value_counts().to_dict(), 'agree>=2',(mb.agree>=2).sum())
print('qc',len(qc),qc.label.value_counts().to_dict(), 'qc NDVI>0.6:',(qc.A_su_NDVI>0.6).sum())
def mk(): return Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(C=1.0,max_iter=5000))])
BASE=['A_SAG','chm_mean_r15','chm_stdDev_r15']
sets={'BASE':BASE,'BASE+E':BASE+['E_h_per_year_since_dist','E_eth_model_sd'],'BASE+H':BASE+['H_nfw','H_mapbiomas10m_plantation','H_gft_planted']}
y=mb.label.values; tm=(mb.agree>=2).values; yq=qc.label.values
def thrq(s0,fe): return np.quantile(s0,1-fe,method='higher')
out=[]
for nm,cols in sets.items():
    s=np.full(len(mb),np.nan)
    for tr,te in GroupKFold(5).split(mb,y,mb.block):
        tr=tr[tm[tr]]; s[te]=mk().fit(mb.iloc[tr][cols],y[tr]).predict_proba(mb.iloc[te][cols])[:,1]
    r=dict(model=nm,MB_AUC=roc_auc_score(y,s),MB_AUC_agree23=roc_auc_score(y[tm],s[tm]))
    for fe in (.01,.02,.05):
        t=thrq(s[y==0],fe); r[f'rec{int(fe*100)}']=(s[y==1]>t).mean(); r[f'FEchk{int(fe*100)}']=(s[y==0]>t).mean()
    t5=thrq(s[y==0],.05)
    q=mk().fit(mb[tm][cols],y[tm]).predict_proba(qc[cols])[:,1]
    r['QC_AUC']=roc_auc_score(yq,q); r['QC_FE']=(q[yq==0]>t5).mean(); r['QC_rec']=(q[yq==1]>t5).mean()
    r['QC_nFE']=int((q[yq==0]>t5).sum()); r['t5']=t5
    out.append(r)
R=pd.DataFrame(out); pd.set_option('display.width',250); print(R.round(4).to_string())
R.to_csv('a_out.csv',index=False)
