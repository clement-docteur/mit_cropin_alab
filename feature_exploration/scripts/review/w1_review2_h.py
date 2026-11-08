import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import beta
mb=pd.read_csv('/home/claude/w1/w1_mb_dev_features.csv'); qc=pd.read_csv('/home/claude/w1/w1_qc_dev_features.csv')
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True); print(len(mb),len(qc))
mk=lambda: Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(max_iter=5000))])
cols=['A_SAG','chm_mean_r15','chm_stdDev_r15','D_max_nbr_drop','D_cc_n_breaks','D_years_continuous_forest','D_h_total_loss_px','D_ndvi_1985_89','D_lt_n_vertices']
y=mb.label.values; tm=(mb.agree>=2).values; yq=qc.label.values
s=np.full(len(mb),np.nan)
for tr,te in GroupKFold(5).split(mb,y,mb.block):
    tr=tr[tm[tr]]; s[te]=mk().fit(mb.iloc[tr][cols],y[tr]).predict_proba(mb.iloc[te][cols])[:,1]
t=lambda fe: np.quantile(s[y==0],1-fe,method='higher')
r={'MB_AUC':roc_auc_score(y,s)}
for fe in (.01,.02,.05): r[f'MB_rec@{int(fe*100)}%']=(s[y==1]>t(fe)).mean()
q=mk().fit(mb[tm][cols],y[tm]).predict_proba(qc[cols])[:,1]
r['QC_AUC']=roc_auc_score(yq,q); k=(q[yq==0]>t(.05)).sum(); r['QC_FE_at_MB_t']=k/58; r['QC_rec_at_MB_t']=(q[yq==1]>t(.05)).mean()
R=pd.read_csv('/home/claude/w1/w1_models.csv'); row=R[R.model=='Baseline + Family D (main)'].iloc[0]
print('reported features:',row.features)
for kk,v in r.items(): print(f'{kk:16s} mine {v:.4f} reported {row[kk]:.4f} diff {v-row[kk]:+.4f}')
print('QC FE count',k,'/58, CP95 %.3f-%.3f'%(beta.ppf(.025,k,59-k),beta.ppf(.975,k+1,58-k)))
# (c) span check
for f in ['D_years_continuous_forest','D_years_since_last_event','D_mean_interval','D_lt_years_since_greatest','D_cc_years_since_break','D_cc_n_breaks','D_n_years_obs']:
    print(f,'MB min/max',mb[f].min(),mb[f].max(),'| QC min/max',qc[f].min(),qc[f].max(),'| ceiling share MB %.3f QC %.3f'%((mb[f]==mb[f].max()).mean(),(qc[f]==qc[f].max()).mean()))
