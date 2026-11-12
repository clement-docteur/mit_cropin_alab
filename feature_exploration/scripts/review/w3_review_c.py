import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as A
D='/home/claude/w3/'
mb=pd.read_csv(D+'w3_mb_dev_features.csv'); qc=pd.read_csv(D+'w3_qc_dev_features.csv')
mb=mb[mb.A_su_NDVI>0.6].reset_index(drop=True); tr=mb[mb.agree>=2]
# time mismatch in hpy
for nm,d in [('MB',tr),('QC',qc)]:
    ev=d.cutoff-d.D_years_since_last_event
    post=(d.D_years_since_last_event<25)&(ev>2020)
    print(nm,'cutoffs',d.cutoff.value_counts().sort_index().to_dict(),'points w/ last event after 2020 (ETH yr):',post.sum(), d[post].label.value_counts().to_dict())
# C1: external maps
def tab(d,f): return pd.crosstab(d.label,d[f].round(2) if d[f].nunique()>3 else d[f])
for f in ['H_mapbiomas10m_plantation','H_gft_planted','H_nfw','nfw']:
    for nm,d in [('MB',tr),('QC',qc)]:
        x=d[d[f].notna()]
        s=f'{nm} {f}: n={len(x)} AUC={A(x.label,x[f]):.3f}'
        if x[f].nunique()<=3:
            p=x[f]>0.5; s+=f' | recall={p[x.label==1].mean():.3f} FE={p[x.label==0].mean():.3f}'
        print(s)
print('mb10 values',tr.H_mapbiomas10m_plantation.value_counts().to_dict())
# agree subsets for mb10 
for a in (1,2,3):
    x=mb[(mb.agree==a)&mb.H_mapbiomas10m_plantation.notna()]
    if x.label.nunique()==2: print('agree',a,'n',len(x),'mb10 AUC',round(A(x.label,x.H_mapbiomas10m_plantation),3),'nfw AUC',round(A(x.label,x.H_nfw),3))
# NFW threshold FE
for nm,d in [('MB',tr),('QC',qc)]:
    for t in (0.3,0.5,0.7): print(nm,'NFW<',t,'-> called plantation: rec',round((d[d.label==1].H_nfw<t).mean(),3),'FE',round((d[d.label==0].H_nfw<t).mean(),3))
# C2 GEDI
for nm,d in [('MB',tr),('QC',qc)]:
    for r in ('r100','r250'):
        n=d[f'gedi_n_{r}'].fillna(0)
        print(nm,r,'any shots',(n>0).sum(),'/',len(d),'by label',d[n>0].label.value_counts().to_dict(),'non-NaN rh98',d[f'E_gedi_rh98_{r}'].notna().sum())
S=pd.read_csv(D+'w3_screen.csv'); print(S[S.feature.str.contains('gedi')][['feature','n','AUC','AUC_lo','QC_AUC_raw','context_dependent','PASS']].to_string())
# C3 model_sd
print('--- E_eth_model_sd')
for nm,d in [('MB',tr),('QC',qc)]:
    print(nm,'by label mean', d.groupby('label').E_eth_model_sd.mean().round(2).to_dict())
    cs={}
    for c in ['su_n_clear','border','E_eth_h','chm_mean_r15','A_SAG','G_dist_nontree','G_patch_px','G_tree_frac_50m','G_elev','G_clim_temp','G_slope','E_eth_sd50','D_years_since_last_event']:
        if c in d and d[c].notna().sum()>20: cs[c]=round(spearmanr(d.E_eth_model_sd,d[c],nan_policy='omit')[0],2)
    print(' spearman',cs)
    print(' state AUCs',{s:round(A(g.label,g.E_eth_model_sd),3) for s,g in d.groupby('state') if g.label.nunique()==2},'state n',d.state.value_counts().to_dict())
    # block-level: lat proxy
    blk=d.groupby('block').E_eth_model_sd.mean(); print(' block mean sd range',round(blk.min(),1),round(blk.max(),1))
    print(' block example',d.block.head(3).tolist())
# within-block (demeaned) AUC
for nm,d in [('MB',tr),('QC',qc)]:
    dm=d.E_eth_model_sd-d.groupby('block').E_eth_model_sd.transform('mean')
    print(nm,'block-demeaned model_sd AUC',round(A(d.label,dm),3),'raw',round(A(d.label,d.E_eth_model_sd),3))
    # model_sd vs eth_h: is it just height-scaled?
    from sklearn.linear_model import LinearRegression
    X=d[['E_eth_h']].values; r=d.E_eth_model_sd-LinearRegression().fit(X,d.E_eth_model_sd).predict(X)
    print(nm,'model_sd residual on eth_h AUC',round(A(d.label,r),3), 'eth_h AUC',round(A(d.label,d.E_eth_h),3))
    if 'border' in d: print(nm,'border AUCs',{b:round(A(g.label,g.E_eth_model_sd),3) for b,g in d.groupby('border') if g.label.nunique()==2})
