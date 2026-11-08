import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import norm
D='/home/claude/w2/'
mb=pd.read_csv(D+'w2_mb_dev_features.csv'); qc=pd.read_csv(D+'w2_qc_dev_features.csv')
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True)
def model(): return Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('l',LogisticRegression(C=1.0,max_iter=5000))])
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
BASE=['A_SAG','chm_mean_r15','chm_stdDev_r15']
FB=['B_SAG_SON','B_SAG_phase_sin','B_NDRE_phase_sin','B_SWIR_phase_sin','B_SAG_DJ','B_SWIR_phase_cos','B_SAG_mean']
FF=['F_VHminusVV_median','F_VV_median']
sets={'BASE':BASE,'B alone':FB,'BASE+B':BASE+FB,'BASE+F':BASE+FF}
y=mb.label.values; tr=(mb.agree>=2).values; yq=qc.label.values
out=[]
for nm,cols in sets.items():
    s=np.full(len(mb),np.nan)
    for a,b in GroupKFold(5).split(mb,y,mb.block):
        a=a[tr[a]]; s[b]=model().fit(mb.iloc[a][cols],y[a]).predict_proba(mb.iloc[b][cols])[:,1]
    r={'model':nm,'MB_AUC':roc_auc_score(y,s)}
    f0=s[y==0]
    for fe in (.01,.02,.05):
        t=np.quantile(f0,1-fe,method='higher'); r[f'rec@{int(fe*100)}']=(s[y==1]>t).mean()
    t5=np.quantile(f0,.95,method='higher'); r['MB_FE_at_t5']=(f0>t5).mean()
    m=model().fit(mb[tr][cols],y[tr]); q=m.predict_proba(qc[cols])[:,1]
    r['QC_AUC']=roc_auc_score(yq,q)
    k=int((q[yq==0]>t5).sum()); n=int((yq==0).sum()); kp=int((q[yq==1]>t5).sum())
    r['QC_FE']=k/n; r['QC_FE_k/n']=f'{k}/{n}'; lo,hi=wilson(k,n); r['QC_FE_wilson']=f'{lo:.3f}-{hi:.3f}'
    r['QC_rec']=kp/(yq==1).sum(); r['QC_rec_k/n']=f'{kp}/{(yq==1).sum()}'
    r['n_mb']=len(mb); r['n_tr']=tr.sum()
    out.append(r)
R=pd.DataFrame(out); pd.set_option('display.width',250); print(R.round(4).to_string())
R.to_csv('A_recomputed.csv',index=False)
# compare
M=pd.read_csv(D+'w2_models.csv').set_index('model')
mp={'BASE':'Baseline (decision rule): SAG + height + height SD','B alone':'Family B (main) alone','BASE+B':'Baseline + Family B (main)','BASE+F':'Baseline + Family F (main)'}
cm={'MB_AUC':'MB_AUC','rec@1':'MB_rec@1%','rec@2':'MB_rec@2%','rec@5':'MB_rec@5%','QC_AUC':'QC_AUC','QC_FE':'QC_FE_at_MB_t','QC_rec':'QC_rec_at_MB_t'}
for r in out:
    for a,b in cm.items():
        rep=M.loc[mp[r['model']],b]; d=r[a]-rep
        print(f"{r['model']:8s} {a:7s} mine={r[a]:.4f} rep={rep:.4f} diff={d:+.4f} {'<<' if abs(d)>0.01 else ''}")
