import numpy as np, pandas as pd
D='/home/claude/w2/'; G='/mnt/user-data/uploads/cropin_texture/'
mb=pd.read_csv(D+'w2_mb_dev_features.csv'); qc=pd.read_csv(D+'w2_qc_dev_features.csv')
Sm=pd.read_csv(G+'w2_s2_monthly_mb.csv'); Sq=pd.read_csv(G+'w2_s2_monthly_qc.csv')
for nm,S,F in [('MB',Sm,mb),('QC',Sq,qc)]:
    S=S.copy(); S['yr']=S.month.str[:4].astype(int)
    print(nm,'rows',len(S),'ids',S.id.nunique(),'feature ids',F.id.nunique(),'feature ids missing in raw',len(set(F.id)-set(S.id)), 'raw ids not in dev',len(set(S.id)-set(F.id)))
    # cutoff consistency raw vs feature file
    cc=S.groupby('id').cutoff.nunique().max(); m=S.groupby('id').cutoff.first().rename('rc').to_frame().join(F.set_index('id').cutoff)
    print(' cutoff per id unique max',cc,' raw vs feat cutoff mismatches',(m.rc!=m.cutoff).sum(), 'NaN feat', m.cutoff.isna().sum())
    S=S.merge(F[['id','cutoff']].rename(columns={'cutoff':'fc'}),on='id')
    bad=S[(S.yr>S.fc)|(S.yr<S.fc-2)]; print(' months outside [fc-2,fc]:',len(bad))
    print(' rows with B8 NaN',S.B8.isna().sum(),' n_clear dist',S.n_clear.describe()[['min','50%','max']].to_dict())
    if nm=='QC':
        q=S.merge(F[['id','clear_month']],on='id'); after=q[(q.month>=q.clear_month)&q.B8.notna()]
        print(' QC valid months at/after clearing month:',len(after), after.id.nunique())
        # months within 3 months before clearing
        cm=pd.to_datetime(q.clear_month); mm=pd.to_datetime(q.month)
        print(' QC valid months within 6 mo before clearing:',((cm-mm).dt.days.between(0,183)&q.B8.notna()).sum())
        x=F.dropna(subset=['clear_month']); print(' clear_month NaN',F.clear_month.isna().sum(),(x.clear_month.str[:4].astype(int)-x.cutoff).value_counts().to_dict())
# recompute features for samples
def feats(g):
    g=g.dropna(subset=['B8']); b=lambda x:g[x].values/1e4
    xb,xg,xr=.4966,.560,.6645
    v1=np.c_[np.full(len(g),xb-xg),b('B2')-b('B3')]; v2=np.c_[np.full(len(g),xr-xg),b('B4')-b('B3')]
    cos=(v1*v2).sum(1)/np.linalg.norm(v1,axis=1)/np.linalg.norm(v2,axis=1); sag=np.degrees(np.arccos(np.clip(cos,-1,1)))
    ndvi=(b('B8')-b('B4'))/(b('B8')+b('B4'))
    yr=g.month.str[:4].astype(int).values; mo=g.month.str[5:7].astype(int).values; t=yr+(mo-.5)/12
    X=np.c_[np.ones(len(g)),np.cos(2*np.pi*t),np.sin(2*np.pi*t)]
    o={}
    if len(g)>=12:
        c=np.linalg.lstsq(X,sag,rcond=None)[0]; o['B_SAG_phase_sin']=np.sin(np.arctan2(c[2],c[1]))
        c=np.linalg.lstsq(X,ndvi,rcond=None)[0]; o['B_NDVI_amp']=np.hypot(c[1],c[2])
    else: o['B_SAG_phase_sin']=o['B_NDVI_amp']=np.nan
    o['B_SAG_SON']=np.median(sag[np.isin(mo,[9,10,11])]) if np.isin(mo,[9,10,11]).any() else np.nan
    o['B_SAG_DJ']=np.median(sag[np.isin(mo,[12,1])]) if np.isin(mo,[12,1]).any() else np.nan
    o['n']=len(g); return o
rng=np.random.default_rng(1); res=[]
for nm,S,F,k in [('MB',Sm,mb,30),('QC',Sq,qc,20)]:
    for i in rng.choice(F.id.values,k,replace=False):
        o=feats(S[S.id==i]); r=F[F.id==i].iloc[0]
        for f in ['B_SAG_SON','B_SAG_DJ','B_SAG_phase_sin','B_NDVI_amp']:
            res.append(dict(set=nm,id=i,f=f,mine=o[f],rep=r[f]))
        res.append(dict(set=nm,id=i,f='B_n_months',mine=o['n'],rep=r['B_n_months']))
R=pd.DataFrame(res); R['ad']=(R.mine-R.rep).abs()
print(R.groupby(['set','f']).ad.agg(['max','count']), R[(R.ad>1e-6)|(R.mine.isna()!=R.rep.isna())].head(20))
