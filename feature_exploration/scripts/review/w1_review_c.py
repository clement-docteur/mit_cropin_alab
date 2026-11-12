import numpy as np, pandas as pd, json, re
from sklearn.metrics import roc_auc_score
G='/mnt/user-data/uploads/cropin_texture/'
ann=pd.read_csv(G+'w1_landsat_annual.csv'); cc=pd.read_csv(G+'w1_ccdc.csv'); lt=pd.read_csv(G+'w1_landtrendr.csv'); han=pd.read_csv(G+'w1_hansen.csv')
mb=pd.read_csv('/home/claude/w1/w1_mb_dev_features.csv'); qc=pd.read_csv('/home/claude/w1/w1_qc_dev_features.csv')
mbf=mb[mb.A_su_NDVI>0.6]
# full check of 3 features over all rows
A={k:g for k,g in ann.groupby(['id','set'])}; C=cc.set_index(['id','set'])
bad=0
for d,st in [(mb,'mb'),(qc,'qc')]:
    for _,r in d.iterrows():
        g=A[(r.id,st)]; g=g[g.year<=r.cutoff].dropna(subset=['NDVI','NBR']).sort_values('year')
        y,nd,nb=g.year.values,g.NDVI.values,g.NBR.values
        low=y[nd<0.6]; ycf=r.cutoff-low.max() if len(low) else r.cutoff-1984
        ev=[]
        for i in range(2,len(y)):
            if np.median(nb[i-2:i])-nb[i]>=0.2 and (not ev or y[i]-ev[-1]>2): ev.append(y[i])
        tb=json.loads(C.loc[(r.id,st),'tBreak'].replace('E','e')); nbk=sum(1 for b in tb if 0<b<r.cutoff+1)
        if (ycf,len(ev),nbk)!=(r.D_years_continuous_forest,r.D_n_events,r.D_cc_n_breaks): bad+=1
print('ALL rows mismatches on 3 features:',bad,'of',len(mb)+len(qc))
# Hansen years present
ks=set(int(k) for h in han.histogram.dropna() for k in re.findall(r'(\d+)=',h)); print('Hansen lossyear codes',min(ks),max(ks))
# missing raw records
for nm,df in [('lt',lt),('cc',cc),('han',han)]:
    s=set(zip(df.id,df.set)); print(nm,'missing mb',sum((i,'mb') not in s for i in mb.id),'missing qc',sum((i,'qc') not in s for i in qc.id))
print('hansen NaN histogram rows',han.histogram.isna().sum())
# span asymmetry
for f in ['D_years_continuous_forest','D_ndvi_1985_89','D_cc_n_breaks','D_max_nbr_drop','D_lt_greatest_mag','D_h_total_loss_share']:
    print(f, 'MB max',mbf[f].max(),'QC max',qc[f].max(),
      '| MB AUC %.3f QC AUC %.3f'%(roc_auc_score(mbf.label,mbf[f].fillna(mbf[f].median())),roc_auc_score(qc.label,qc[f].fillna(qc[f].median()))),
      '| MB miss by lab',mbf.groupby('label')[f].apply(lambda s:s.isna().mean()).round(3).to_dict(),'QC miss',qc[f].isna().mean().round(3))
print('share at YCF ceiling: MB',(mbf.D_years_continuous_forest==38).groupby(mbf.label).mean().round(3).to_dict(),'QC',(qc.D_years_continuous_forest==41).groupby(qc.label).mean().round(3).to_dict())
print('MB forest with YCF>=10 :',(mbf[mbf.label==0].D_years_continuous_forest>=10).mean().round(3),'plantation:',(mbf[mbf.label==1].D_years_continuous_forest>=10).mean().round(3))
# LandTrendr vertex at/after cutoff
def lastv(s):
    a=np.array(json.loads(s)); return a[0][a[3]==1]
lt['v']=lt['first'].map(lastv)
lm=lt[lt.set=='mb'].set_index('id').v
print('MB pts with an LT vertex in 2023-2025 (fit uses post-cutoff data, segment crossing cutoff dropped):',np.mean([any((v>2022)&(v<2025)) for v in lm.loc[mbf.id]]).round(3))
cm=cc[cc.set=='mb'].set_index('id').tBreak.map(lambda s:json.loads(s.replace('E','e')))
print('MB pts with CCDC break in 2023-2025:',np.mean([any((np.array(b)>=2023)) for b in cm.loc[mbf.id]]).round(3))
# QC rows with early cutoff: are post-clearing years in LT/CCDC fit?
print(qc[qc.cutoff<2025][['id','label','clear_month','season_start','cutoff','D_cc_n_breaks','D_years_continuous_forest']])
# geography of QC vs MB
m0=pd.read_csv('/home/claude/p0/mb_dev_table.csv',usecols=['id','lon','lat']); q0=pd.read_csv('/home/claude/p0/qc_dev_table.csv',usecols=['id','lon','lat'])
from scipy.spatial import cKDTree
t=cKDTree(np.c_[m0.lon,m0.lat]); dd,_=t.query(np.c_[q0.lon,q0.lat]); km=dd*111
print('QC->nearest MB dev point km: min %.2f median %.1f; share <1km %.3f'%(km.min(),np.median(km),(km<1).mean()))
print('MB blocks overlapping QC 0.5deg cells:', len(set(mb.block)&set((np.floor(q0.lon/0.5).astype(int).astype(str)+'_'+np.floor(q0.lat/0.5).astype(int).astype(str)))), 'MB block example',mb.block.iloc[0])
