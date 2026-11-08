import numpy as np, pandas as pd, json
G='/mnt/user-data/uploads/cropin_texture/'
ltc=pd.read_csv(G+'w1c_landtrendr_cut.csv').set_index(['id','set']); lto=pd.read_csv(G+'w1_landtrendr.csv').set_index(['id','set'])
qc=pd.read_csv('/home/claude/w1/w1_qc_dev_features.csv'); mb=pd.read_csv('/home/claude/w1/w1_mb_dev_features.csv')
cd=pd.read_csv('/home/claude/p0/qc_clearing_dates.csv')
extra=[i for (i,s) in ltc.index if s=='qc' and i not in set(qc.id)]
print('cut-file QC ids not in QC dev:',extra)
print(cd[cd.id.isin(extra)].to_string())
print('in lto (original batch)?',[ (i,'qc') in lto.index for i in extra])
mbx=[i for (i,s) in ltc.index if s=='mb' and i not in set(mb.id)]; print('cut MB ids not in MB dev:',len(mbx))
p=pd.read_csv('/home/claude/p0/mb_dev_table.csv',usecols=['id']); print('of which in mb_dev_table:',np.isin(mbx,p.id).sum())
print('orig LT MB ids not in dev:',sum(1 for (i,s) in lto.index if s=='mb' and i not in set(mb.id)))
# compare source series and vertex counts
dv=[];ds=[];nvo=[];nvc=[]
for i in mb.id:
    a=np.array(json.loads(lto.loc[(i,'mb'),'first'])); b=np.array(json.loads(ltc.loc[(i,'mb'),'first']))
    n=b.shape[1]; ds.append(np.nanmax(np.abs(a[1][:n]-b[1][:n])))
    nvo.append(int(a[3][a[0]<=2022].sum())); nvc.append(int(b[3].sum()))
ds=np.array(ds); print('max |source diff| over overlapping years: median',np.median(ds),'share >0',(ds>0).mean())
print('n vertices (all years<=2022) orig mean %.2f cut mean %.2f; share differ %.2f'%(np.mean(nvo),np.mean(nvc),np.mean(np.array(nvo)!=np.array(nvc))))
print('cut LT rows count of vertex flags at last year (endpoint):', np.mean([np.array(json.loads(ltc.loc[(i,'mb'),'first']))[3][-1]==1 for i in mb.id]))
print('orig LT vertex at 2022:', np.mean([ (lambda a: a[3][a[0]==2022][0]==1)(np.array(json.loads(lto.loc[(i,'mb'),'first']))) for i in mb.id]))
