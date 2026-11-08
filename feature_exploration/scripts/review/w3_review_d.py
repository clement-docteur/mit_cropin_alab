import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as A
D='/home/claude/w3/'; G='/mnt/user-data/uploads/cropin_texture/'
mb=pd.read_csv(D+'w3_mb_dev_features.csv'); mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True); qc=pd.read_csv(D+'w3_qc_dev_features.csv')
y=mb.label.values; tm=(mb.agree>=2).values
def oof(cols):
    s=np.zeros(len(mb))
    for a,b in GroupKFold(5).split(mb,y,mb.block):
        a=a[tm[a]]; s[b]=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000)).fit(mb.iloc[a][cols],y[a]).predict_proba(mb.iloc[b][cols])[:,1]
    return s
B=['A_SAG','chm_mean_r15','chm_stdDev_r15']
dist=(mb.D_years_since_last_event<25).values
for nm,c in [('BASE',B),('BASE+E',B+['E_h_per_year_since_dist','E_eth_model_sd'])]:
    s=oof(c); t=np.quantile(s[y==0],.95,method='higher')
    f0=(y==0)
    print(nm,'FE disturbed forest',round((s[f0&dist]>t).mean(),3),'n',(f0&dist).sum(),'| FE undisturbed forest',round((s[f0&~dist]>t).mean(),3),'n',(f0&~dist).sum(), '| rec disturbed plant',round((s[(y==1)&dist]>t).mean(),3),'rec undist plant',round((s[(y==1)&~dist]>t).mean(),3))
# edge season check
e=pd.read_csv(G+'w3_edge.csv'); x=pd.concat([mb.assign(set='mb')[['id','set','season_start']],qc[['id','set','season_start']]]).merge(e[['id','set','season_start']],on=['id','set'],suffixes=('','_e'))
print('edge season mismatch',(x.season_start!=x.season_start_e).sum(),'of',len(x))
g=pd.read_csv(G+'w3_gedi_v2.csv'); print('gedi raw cols with year?',[c for c in g.columns if 'year' in c.lower() or 'date' in c.lower()])
# E_eth_model_sd within-state for QC: state missing; use block coords lat proxy
for nm,d in [('MB',mb[tm]),('QC',qc)]:
    lat=d.block.str.split('_').str[0].astype(float); lon=d.block.str.split('_').str[1].astype(float)
    from scipy.stats import spearmanr
    print(nm,'model_sd vs block-lat idx',round(spearmanr(d.E_eth_model_sd,lat)[0],2),'vs lon idx',round(spearmanr(d.E_eth_model_sd,lon)[0],2),'label vs lat',round(spearmanr(d.label,lat)[0],2))
