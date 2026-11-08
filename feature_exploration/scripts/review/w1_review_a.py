import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
mb=pd.read_csv('/home/claude/w1/w1_mb_dev_features.csv'); qc=pd.read_csv('/home/claude/w1/w1_qc_dev_features.csv')
print('MB raw',mb.shape, 'QC',qc.shape)
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True)
print('MB filtered',len(mb), mb.label.value_counts().to_dict(),'agree>=2:',(mb.agree>=2).sum(),'QC',qc.label.value_counts().to_dict(),'blocks',mb.block.nunique())
def mk(): return Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(C=1.0,max_iter=5000))])
def thr(y,s,fe): return np.quantile(s[y==0],1-fe,method='higher')
def rec(y,s,t): return (s[y==1]>t).mean()
def fe_(y,s,t): return (s[y==0]>t).mean()
B=['A_SAG','chm_mean_r15','chm_stdDev_r15']
A=['A_SAG','A_su_NDMI','A_sg_NDVI']
D=['D_cc_n_breaks','D_years_continuous_forest','D_h_total_loss_share','D_ndvi_1985_89','D_max_nbr_drop','D_lt_greatest_mag']
sets={'i':B,'ii':list(dict.fromkeys(B+A)),'iii':B+D,'iv':list(dict.fromkeys(B+A+['C_B8_idm_w3']+D))}
y=mb.label.values; tm=(mb.agree>=2).values; yq=qc.label.values
rows=[]
for k,cols in sets.items():
    s=np.full(len(mb),np.nan)
    for tr,te in GroupKFold(5).split(mb,y,mb.block):
        tr=tr[tm[tr]]; s[te]=mk().fit(mb.iloc[tr][cols],y[tr]).predict_proba(mb.iloc[te][cols])[:,1]
    r={'set':k,'nf':len(cols),'MB_AUC':roc_auc_score(y,s)}
    for fe in (.01,.02,.05): r[f'rec{int(fe*100)}']=rec(y,s,thr(y,s,fe))
    t5=thr(y,s,.05); r['MB_FE_at_t5']=fe_(y,s,t5)
    q=mk().fit(mb[tm][cols],y[tm]).predict_proba(qc[cols])[:,1]
    r['QC_AUC']=roc_auc_score(yq,q); r['QC_FE']=fe_(yq,q,t5); r['QC_rec']=rec(yq,q,t5)
    r['QC_nFE']=f"{(q[yq==0]>t5).sum()}/{(yq==0).sum()}"
    # agree>=2 only eval
    m2=tm; r['MB_AUC_agree2+']=roc_auc_score(y[m2],s[m2])
    rows.append(r)
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(4).to_string())
rep=pd.read_csv('/home/claude/w1/w1_models.csv').iloc[[0,3,11,14]]
print(rep[['MB_AUC','MB_rec@1%','MB_rec@2%','MB_rec@5%','QC_AUC','QC_FE_at_MB_t','QC_rec_at_MB_t']].round(4).to_string())
