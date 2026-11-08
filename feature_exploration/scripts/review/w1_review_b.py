import numpy as np, pandas as pd, json
G='/mnt/user-data/uploads/cropin_texture/'
ann=pd.read_csv(G+'w1_landsat_annual.csv'); cc=pd.read_csv(G+'w1_ccdc.csv')
mb=pd.read_csv('/home/claude/w1/w1_mb_dev_features.csv'); qc=pd.read_csv('/home/claude/w1/w1_qc_dev_features.csv')
print('annual years',ann.year.min(),ann.year.max(),'sets',ann.set.value_counts().to_dict())
print('dup (id,set,year) in annual:',ann.duplicated(['id','set','year']).sum())
print('ccdc dup keys',cc.duplicated(['id','set']).sum(),'tEnd max', cc.tEnd.str.strip('[]').str.split(',').map(lambda v: max(map(float,v))).max())
rng=np.random.default_rng(7)
def mine(id_,st,cut):
    g=ann[(ann.id==id_)&(ann.set==st)&(ann.year<=cut)].dropna(subset=['NDVI','NBR']).sort_values('year')
    y=g.year.values; nd=g.NDVI.values; nb=g.NBR.values
    # continuous forest: years since last year with NDVI<0.6, else cutoff-1984
    low=y[nd<0.6]; ycf=cut-low.max() if len(low) else cut-1984
    ev=[]
    for i in range(2,len(y)):
        if np.median(nb[i-2:i])-nb[i]>=0.2 and (not ev or y[i]-ev[-1]>2): ev.append(y[i])
    r=cc[(cc.id==id_)&(cc.set==st)].iloc[0]
    tb=json.loads(r.tBreak.replace('E','e'))
    nbk=sum(1 for b in tb if 0<b<cut+1)
    return ycf,len(ev),nbk,(y.max() if len(y) else None)
out=[]
for d,st,n in [(mb,'mb',30),(qc,'qc',20)]:
    for _,r in d.sample(n,random_state=11).iterrows():
        a=mine(r.id,st,r.cutoff)
        out.append(dict(set=st,id=r.id,cut=r.cutoff,ycf=a[0],ycf_f=r.D_years_continuous_forest,nev=a[1],nev_f=r.D_n_events,nbk=a[2],nbk_f=r.D_cc_n_breaks,maxyr=a[3]))
O=pd.DataFrame(out)
O['ok']=(O.ycf==O.ycf_f)&(O.nev==O.nev_f)&(O.nbk==O.nbk_f)
print(O.to_string()); print('mismatches',(~O.ok).sum(),'any maxyr>cut',(O.maxyr>O.cut).sum())
