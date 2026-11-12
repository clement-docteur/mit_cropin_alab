import numpy as np, pandas as pd
G='/mnt/user-data/uploads/cropin_texture/'; D='/home/claude/w3/'
mb=pd.read_csv(D+'w3_mb_dev_features.csv'); qc=pd.read_csv(D+'w3_qc_dev_features.csv')
F=pd.concat([mb.assign(set='mb'),qc.assign(set='qc')]) if 'set' not in mb else pd.concat([mb,qc])
print(F.set.value_counts().to_dict())
raw={f:pd.read_csv(G+f+'.csv') for f in ['w3_eth','w3_edge','w3_gft','w3_hansen_tc','w3_gedi_v2','w3_context','w3_shape']}
for k,v in raw.items(): print(k,'dup(id,set)=',v.duplicated(['id','set']).sum(),'sets',v.set.value_counts().to_dict(), 'dev ids missing', len(F.merge(v[['id','set']],how='left',indicator=True).query('_merge=="left_only"')))
# label consistency raw vs features
e=raw['w3_eth']; m=F[['id','set','label','cutoff']].merge(e[['id','set','label','cutoff']],on=['id','set'],suffixes=('','_r'))
print('label mismatch',(m.label!=m.label_r).sum(),'cutoff mismatch',(m.cutoff!=m.cutoff_r).sum())
rng=np.random.default_rng(7); S=pd.concat([F[F.set=='mb'].sample(20,random_state=1),F[F.set=='qc'].sample(10,random_state=1)])
x=S.merge(e,on=['id','set'],suffixes=('','_e')).merge(raw['w3_edge'],on=['id','set'],suffixes=('','_g')).merge(raw['w3_gft'].rename(columns={'first':'gft'}),on=['id','set'],suffixes=('','_f')).merge(raw['w3_hansen_tc'].rename(columns={'first':'tc'}),on=['id','set'],suffixes=('','_h'))
chk=pd.DataFrame({
 'eth_sd50':np.isclose(x.E_eth_sd50,x.eth_h_sd50,equal_nan=True),
 'hpy':np.isclose(x.E_h_per_year_since_dist, x.eth_h/np.maximum(x.D_years_since_last_event,1),equal_nan=True),
 'edge':np.isclose(x.G_edge_ndvi_contrast, x.ndvi_pt-x.ndvi_ring,equal_nan=True),
 'gft':np.isclose(x.H_gft_planted,(x.gft==20).astype(float)),
 'tc':np.isclose(x.H_hansen_tc2000,x.tc,equal_nan=True)})
print('30-id check all-match per feature:',chk.all().to_dict(), 'n',len(x))
# full-population check
y=F.merge(e,on=['id','set'],suffixes=('','_e')).merge(raw['w3_gft'].rename(columns={'first':'gft'}),on=['id','set'],suffixes=('','_f'))
print('full: hpy match', np.isclose(y.E_h_per_year_since_dist, y.eth_h/np.maximum(y.D_years_since_last_event,1),equal_nan=True).mean())
print('gft codes', y.groupby(['set','label']).gft.value_counts().unstack(fill_value=0))
print('D_years_since NaN',F.D_years_since_last_event.isna().sum(),'eth NaN',F.E_eth_h.isna().sum())
