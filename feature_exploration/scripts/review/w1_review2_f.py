import numpy as np, pandas as pd, json, re
G='/mnt/user-data/uploads/cropin_texture/'
ltc=pd.read_csv(G+'w1c_landtrendr_cut.csv'); ccc=pd.read_csv(G+'w1c_ccdc_cut.csv')
lto=pd.read_csv(G+'w1_landtrendr.csv'); cco=pd.read_csv(G+'w1_ccdc.csv')
mb=pd.read_csv('/home/claude/w1/w1_mb_dev_features.csv'); qc=pd.read_csv('/home/claude/w1/w1_qc_dev_features.csv')
print('cut files: sets',ltc.set.value_counts().to_dict(), ccc.set.value_counts().to_dict(),'cutoffs',ltc.groupby('set').cutoff.value_counts().to_dict())
print('dup keys',ltc.duplicated(['id','set']).sum(),ccc.duplicated(['id','set']).sum())
print('QC table cutoff counts',qc.cutoff.value_counts().to_dict())
q10=qc[qc.cutoff<2025]
print('MB ids missing from cut LT/CC:',(~mb.id.isin(ltc[ltc.set=='mb'].id)).sum(),(~mb.id.isin(ccc[ccc.set=='mb'].id)).sum())
print('QC<2025 missing from cut:',(~q10.id.isin(ltc[ltc.set=='qc'].id)).sum(),(~q10.id.isin(ccc[ccc.set=='qc'].id)).sum())
# cutoff consistency between cut file and tables
for nm,cf in [('lt',ltc),('cc',ccc)]:
    m=pd.concat([mb[['id','cutoff']].assign(set='mb'),qc[['id','cutoff']].assign(set='qc')]).merge(cf[['id','set','cutoff']],on=['id','set'],suffixes=('','_file'))
    print(nm,'rows matched',len(m),'cutoff mismatches',(m.cutoff!=m.cutoff_file).sum())
    print(m[m.cutoff!=m.cutoff_file].head(10).to_string())
# max year / tEnd in cut files
ltc['ymax']=ltc['first'].map(lambda s: max(json.loads(s)[0]))
ccc['tend']=ccc.tEnd.map(lambda s: max(json.loads(s.replace('E','e'))))
ccc['tb']=ccc.tBreak.map(lambda s: max(json.loads(s.replace('E','e'))))
print('LT ymax>cutoff:',(ltc.ymax>ltc.cutoff).sum(),' CC tEnd>=cutoff+1:',(ccc.tend>=ccc.cutoff+1).sum(),' CC tBreak>=cutoff+1:',(ccc.tb>=ccc.cutoff+1).sum())
print('LT ymax<cutoff (short series):',(ltc.ymax<ltc.cutoff).sum())
# QC>=2025 rows should use original fits: original LT ends 2025
WIN,WC=25,20
def ltf(arr,cut):
    a=np.array(arr); yr,fit,vtx=a[0],a[2],a[3]
    v=yr[(vtx==1)&(yr<=cut)&(yr>=cut-WIN+1)]; fv=np.array([fit[yr==t][0] for t in v])
    seg=[(v[i],v[i+1],fv[i+1]-fv[i]) for i in range(len(v)-1)]; d=[s for s in seg if s[2]>=100]
    return len(v), len(d), (max(s[2] for s in d) if d else 0.0)
def ccf(r,cut):
    P=lambda k: json.loads(str(r[k]).replace('E','e'))
    ts,tb,ndc=P('tStart'),P('tBreak'),P('NDVI_coefs')
    nb=sum(1 for b in tb if cut+1-WC<=b<cut+1)
    k=None
    for i in range(len(ts)):
        if ts[i]<=cut+0.5: k=i
    amp=np.nan if k is None else float(np.hypot(ndc[k][2],ndc[k][3]))
    return nb,amp
LC=ltc.set_index(['id','set']); CC=ccc.set_index(['id','set']); LO=lto.set_index(['id','set']); CO=cco.set_index(['id','set'])
rows=[]
samp=[(r,'mb') for _,r in mb.sample(30,random_state=3).iterrows()]+[(r,'qc') for _,r in q10.iterrows()]+[(r,'qc') for _,r in qc[qc.cutoff==2025].sample(10,random_state=3).iterrows()]
for r,st in samp:
    key=(r.id,st); cut=r.cutoff
    usecut=key in LC.index
    lc=ltf(json.loads((LC if usecut else LO).loc[key,'first']),cut); cc_=ccf((CC if usecut else CO).loc[key],cut)
    lo=ltf(json.loads(LO.loc[key,'first']),cut); co=ccf(CO.loc[key],cut)
    tab=(r.D_lt_n_vertices,r.D_lt_n_dist,r.D_lt_greatest_mag,r.D_cc_n_breaks,r.D_cc_ndvi_amp)
    mine=(*lc,*cc_); orig=(*lo,*co)
    eq=lambda a,b: all((np.isnan(x) and np.isnan(y)) or abs(x-y)<1e-6 for x,y in zip(a,b))
    rows.append(dict(set=st,id=r.id,cut=cut,src='cut' if usecut else 'orig',match_mine=eq(tab,mine),match_origfit=eq(tab,orig)))
O=pd.DataFrame(rows); print(O.groupby(['set','src']).agg(n=('id','size'),match=('match_mine','sum'),same_as_origfit=('match_origfit','sum')))
print(O[~O.match_mine])
# how different are cut vs orig on MB overall?
m=mb.id.values[:]
d=[ (ltf(json.loads(LC.loc[(i,'mb'),'first']),2022)[0]!=ltf(json.loads(LO.loc[(i,'mb'),'first']),2022)[0]) for i in m]
print('MB share where n_vertices differs cut vs orig fit:',np.mean(d).round(3))
