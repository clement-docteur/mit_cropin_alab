import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as A
D='/home/claude/w3/'
mb=pd.read_csv(D+'w3_mb_dev_features.csv'); qc=pd.read_csv(D+'w3_qc_dev_features.csv')
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True); tr=mb[mb.agree>=2]
def auc(d,f):
    d=d[d[f].notna()]; a=A(d.label,d[f]); return round(a,3)
for nm,d in [('MB tr',tr),('QC',qc)]:
    print(nm,'n',len(d))
    yrs=d.D_years_since_last_event
    print(' years_since dist:',yrs.describe()[['min','50%','max']].to_dict(),' share at max',round((yrs==yrs.max()).mean(),3),'max=',yrs.max(), 'share<=1',round((yrs<=1).mean(),3))
    print(' value counts top', yrs.value_counts().head(5).to_dict())
    for f in ['E_h_per_year_since_dist','D_years_since_last_event','E_eth_h']: print('  AUC',f,auc(d,f))
    print('  spearman hpy vs years',round(spearmanr(d.E_h_per_year_since_dist,d.D_years_since_last_event)[0],3),' vs 1/years',round(spearmanr(d.E_h_per_year_since_dist,1/d.D_years_since_last_event.clip(lower=1))[0],3),' pearson log',round(np.corrcoef(np.log(d.E_h_per_year_since_dist+.01),np.log(d.D_years_since_last_event.clip(lower=1)))[0,1],3))
    cap=yrs==yrs.max(); print('  capped share by label', d.groupby('label').apply(lambda g:(g.D_years_since_last_event==yrs.max()).mean()).round(3).to_dict())
    nc=d[~cap]; print('  among non-capped n',len(nc),nc.label.value_counts().to_dict(),'AUC hpy',auc(nc,'E_h_per_year_since_dist'),'AUC years',auc(nc,'D_years_since_last_event'),'AUC eth_h',auc(nc,'E_eth_h'))
    if cap.sum()>20 and d[cap].label.nunique()==2: print('  among capped n',cap.sum(),d[cap].label.value_counts().to_dict(),'AUC hpy',auc(d[cap],'E_h_per_year_since_dist'),'AUC eth_h',auc(d[cap],'E_eth_h'))
    # partial: does hpy add over years+eth_h? logistic in-sample comparison
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
y=mb.label.values; tm=(mb.agree>=2).values
def oof(cols):
    s=np.zeros(len(mb))
    for a,b in GroupKFold(5).split(mb,y,mb.block):
        a=a[tm[a]]; s[b]=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000)).fit(mb.iloc[a][cols],y[a]).predict_proba(mb.iloc[b][cols])[:,1]
    return s
def rep(cols):
    s=oof(cols); t=np.quantile(s[y==0],.95,method='higher'); return round(A(y,s),4), round((s[y==1]>t).mean(),3)
mb['log_years']=np.log(mb.D_years_since_last_event.clip(lower=1)); mb['log_hpy']=np.log(mb.E_h_per_year_since_dist+0.5)
B=['A_SAG','chm_mean_r15','chm_stdDev_r15']
for c in [B, B+['E_h_per_year_since_dist'], B+['D_years_since_last_event'], B+['log_years'], B+['D_years_since_last_event','E_eth_h'], B+['log_years','E_eth_h'], B+['log_years','E_eth_h','E_h_per_year_since_dist'], B+['E_eth_model_sd'], B+['E_h_per_year_since_dist','E_eth_model_sd'], B+['log_years','E_eth_model_sd']]:
    print('OOF AUC, rec@5%FE', rep(c), c[3:])
print('D_years_since in w3_screen / earlier waves?'); S=pd.read_csv(D+'w3_screen.csv'); print(S[S.feature.isin(['E_h_per_year_since_dist','E_eth_model_sd','E_eth_h','E_eth_sd50'])][['feature','AUC','AUC_lo','QC_AUC_raw','state_AUCs','d_rec5','PASS']].to_string())
