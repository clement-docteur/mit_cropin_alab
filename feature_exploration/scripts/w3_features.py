"""Wave 3 – Family E (structure: H12–H14, H17), G (terrain/context/geometry: H29–H32, H35–H38),
H (external layers). Adds to the Wave 2 tables. GEDI is optional (w3_gedi_v2.csv)."""
import pandas as pd, numpy as np, os
mb = pd.read_csv('w2/w2_mb_dev_features.csv'); qc = pd.read_csv('w2/w2_qc_dev_features.csv')
def rd(f): return pd.read_csv(f'gee/{f}.csv').drop(columns=['system:index', '.geo', 'label', 'cutoff', 'season_start'], errors='ignore')
eth = rd('w3_eth'); ctx = rd('w3_context'); edge = rd('w3_edge')
shape = rd('w3_shape').rename(columns={'first': 'patch_edge_frac'}); gft = rd('w3_gft').rename(columns={'first': 'gft2020'})
htc = rd('w3_hansen_tc').rename(columns={'first': 'treecover2000'})
# MapBiomas 10 m map class in the cutoff year (as-of fix after Wave 3 review: Phase 0 used season_start + 1,
# which was after the cutoff for 4 QC dev points). MapBiomas points: 2022; QC: min(cutoff, 2025).
_m = pd.read_csv('gee/p0_mapbiomas10m.csv').drop(columns=['system:index', '.geo', 'label'], errors='ignore')
_k = pd.concat([mb[['id', 'set', 'cutoff']], qc[['id', 'set', 'cutoff']]]).merge(_m, on=['id', 'set'])
m10 = pd.DataFrame({'id': _k.id, 'set': _k.set,
                    'mb10_plantation': [int(r[f"classification_{min(int(r.cutoff), 2025)}"] == 9) for _, r in _k.iterrows()]})
def build(d):
    x = d.merge(eth, on=['id', 'set'], how='left').merge(ctx, on=['id', 'set'], how='left').merge(edge, on=['id', 'set'], how='left') \
         .merge(shape, on=['id', 'set'], how='left').merge(gft, on=['id', 'set'], how='left').merge(htc, on=['id', 'set'], how='left') \
         .merge(m10, on=['id', 'set'], how='left')
    o = pd.DataFrame(index=x.index)
    # E: structure
    o['E_eth_h'] = x.eth_h; o['E_eth_h_mean50'] = x.eth_h_mean50
    o['E_eth_sd15'] = x.eth_h_sd15; o['E_eth_sd50'] = x.eth_h_sd50
    o['E_eth_cv50'] = x.eth_h_sd50 / x.eth_h_mean50.replace(0, np.nan)
    o['E_eth_model_sd'] = x.eth_h_modelsd
    o['E_h_per_year_since_dist'] = x.eth_h / x.D_years_since_last_event.clip(lower=1)        # H13
    o['E_meta_cv15'] = x.chm_stdDev_r15 / x.chm_mean_r15.replace(0, np.nan)
    if os.path.exists('gee/w3_gedi_v2.csv'):
        g = rd('w3_gedi_v2').replace(-9999, np.nan); x = x.merge(g, on=['id', 'set'], how='left')
        for r in ('r100', 'r250'):
            o[f'E_gedi_rh98_{r}'] = x[f'rh98_mean_{r}']; o[f'E_gedi_rh98_sd_{r}'] = x[f'rh98_stdDev_{r}']
            o[f'E_gedi_fhd_{r}'] = x[f'fhd_normal_mean_{r}']; o[f'E_gedi_cover_{r}'] = x[f'cover_mean_{r}']; o[f'E_gedi_pai_{r}'] = x[f'pai_mean_{r}']
            o[f'gedi_n_{r}'] = x[f'rh98_count_{r}']
    # G: terrain, context, geometry
    o['G_slope'] = x.slope; o['G_elev'] = x.elev; o['G_dist_water'] = x.dist_water_m; o['G_hand'] = x.merit_hnd
    o['G_dist_protected'] = x.dist_protected_m; o['G_clim_temp'] = x.clim_temp_x10 / 10; o['G_clim_rain'] = x.clim_rain_mm
    for k in ('tree', 'shrub', 'grass', 'crop', 'built', 'water'):
        for r in ('250m', '500m'): o[f'G_wc_{k}_{r}'] = x[f'wc_{k}_{r}']
    o['G_tree_frac_50m'] = x.wc_tree_frac_50m; o['G_tree_frac_100m'] = x.wc_tree_frac_100m
    o['G_dist_nontree'] = x.wc_dist_nontree_m; o['G_patch_px'] = x.wc_patch_px
    o['G_patch_edge_frac'] = x.patch_edge_frac.where(x.patch_edge_frac >= 0)                # NaN = patch > 1,024 px or not tree
    o['G_edge_ndvi_contrast'] = x.ndvi_pt - x.ndvi_ring; o['G_edge_b11_contrast'] = x.b11_pt - x.b11_ring
    # H: external layers (tags: NFW unknown, GFT unknown (uses NFW), MapBiomas map yes, Hansen no)
    o['H_gft_planted'] = (x.gft2020 == 20).astype(float); o['H_gft_natural'] = x.gft2020.isin([1, 10]).astype(float)
    o['H_nfw'] = x.nfw; o['H_mapbiomas10m_plantation'] = x.mb10_plantation; o['H_hansen_tc2000'] = x.treecover2000
    return pd.concat([x[[c for c in d.columns]], o], axis=1)
MB, QC = build(mb), build(qc)
MB.to_csv('w3_mb_dev_features.csv', index=False); QC.to_csv('w3_qc_dev_features.csv', index=False)
fam = {p: [c for c in MB.columns if c.startswith(p + '_')] for p in 'EGH'}
print({k: len(v) for k, v in fam.items()}, MB.shape, QC.shape)
for nm, d in [('MB', MB), ('QC', QC)]:
    na = d[sum(fam.values(), [])].isna().mean(); print(nm, 'missing >1%:', na[na > .01].round(2).to_dict())
print('GFT classes MB:', MB.groupby('label').H_gft_planted.mean().round(2).to_dict(), 'QC:', QC.groupby('label').H_gft_planted.mean().round(2).to_dict())
