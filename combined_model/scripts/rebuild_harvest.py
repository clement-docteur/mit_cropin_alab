"""Harvest-cycle features REBUILT on our own data (13 Nov 2026).
Idea and algorithm: teammates' harvest_cycles notebook (commit 05bc61e), functions copied verbatim (segmentation, cuts, abrupt start).
Data: OUR Landsat Collection 2 surface-reflectance yearly NBR medians (w1_landsat_annual.csv, 1985–2025) at OUR points,
cut at each point's as-of year (MapBiomas 2022; QC: year before clearing). Border points included.
Levels (closed canopy, open land) are set from Hansen tree cover at our points, never from labels (same rule as the notebook,
using our Hansen values: tree cover 2000 and loss within 150 m up to the as-of year; gain is not in our export, so it is not used).
Known difference: no cross-sensor adjustment between Landsat 5/7 and 8/9 (the notebook shifted TOA values onto the Landsat 7 scale)."""
import json, numpy as np, pandas as pd
NB = '/tmp/claude-0/-home-claude/999d0128-2648-5e0d-9375-1863dd874899/scratchpad/repo/harvest_cycles/harvest_cycles_findings.ipynb'
U = '/mnt/user-data/uploads/cropin_texture/'
cells = json.load(open(NB))['cells']; src = lambda i: ''.join(cells[i]['source'])
YEARS = np.arange(1985, 2026)
g = dict(np=np, pd=pd, YEARS=YEARS)
exec(compile(src(8), 'nb_cell_8', 'exec'), g)                       # despike, segment, find_cuts, abrupt_start
c10 = src(10); exec(compile(c10[:c10.index('def all_features')], 'nb_cell_10', 'exec'), g)   # pixel_features

def build(master):
    L = pd.read_csv(U + 'w1_landsat_annual.csv'); L['year'] = L.year.astype(int)
    S = L.pivot_table(index=['id', 'set'], columns='year', values='NBR')
    S = S.reindex(columns=YEARS)
    asof = master.set_index(['id', 'set']).asof_year
    S = S.loc[S.index.intersection(asof.index)]
    for (i, s), y in asof.items():
        if (i, s) in S.index: S.loc[(i, s), [c for c in YEARS if c > y]] = np.nan
    # levels from Hansen at our points (no labels)
    h = master.set_index(['id', 'set'])[['hansen_tc2000', 'D_h_total_loss_px']]
    closed = h[(h.hansen_tc2000 >= 90) & (h.D_h_total_loss_px == 0)].index
    open_ = h[h.hansen_tc2000 <= 10].index
    vc = S.loc[S.index.intersection(closed), [y for y in YEARS if 2000 <= y <= 2020]].to_numpy().ravel()
    vo = S.loc[S.index.intersection(open_), [1999, 2000, 2001]].to_numpy().ravel()
    canopy = np.nanpercentile(vc, 5)
    open_level = min(np.nanpercentile(vo, 25), canopy - 0.20 / 2) if np.isfinite(vo).sum() >= 30 else canopy - 0.20 / 2
    rows = []
    for idx, row in S.iterrows():
        f = g['pixel_features'](row, canopy, open_level)
        f = {k: v for k, v in f.items() if k not in ('events', 'seg')}
        f['id'], f['set'] = idx; rows.append(f)
    F = pd.DataFrame(rows)
    F['years_since_last_cut'] = (asof.reindex(list(zip(F.id, F.set))).to_numpy() - F.last_cut_year).fillna(40)
    keep = {'n_cycles': 'rh_n_cycles', 's_std': 'rh_s_std', 's_frac_below_forest': 'rh_s_frac_below_forest',
            'forest_since_1985': 'rh_forest_since_1985', 'abrupt_start': 'rh_abrupt_start', 'max_drop': 'rh_max_drop',
            'n_years': 'rh_n_years'}
    meta = {'closed_canopy_nbr': float(canopy), 'open_land_nbr': float(open_level), 'n_closed_px': int(len(closed)),
            'n_open_px': int(len(open_)), 'n_open_values': int(np.isfinite(vo).sum())}
    return F[['id', 'set'] + list(keep)].rename(columns=keep), meta
