#!/usr/bin/env python3
"""Step 3: list Landsat C1 L1TP scenes per path/row directly from the bucket (the index CSV has gaps: it misses
1997 and Landsat 8 in 2018), attach cloud cover (index, else MTL) and select up to 6 scenes per path/row-year
(3 lowest-cloud in Apr-Sep, 3 in Oct-Mar; Landsat 7 SLC-off penalised).

Usage: python list_scenes.py <landsat_index_south.csv from filter_index.py> <data_dir with points.csv>"""
import sys, re, requests, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
INDEX, D = sys.argv[1], sys.argv[2]
API = "https://storage.googleapis.com/storage/v1/b/gcp-public-data-landsat/o"
pts = pd.read_csv(f'{D}/points.csv', dtype={'pr1': str, 'pr2': str})
prs = sorted(set(pts.pr1) | set(pts.loc[pts.d2_km < 105, 'pr2']))
print('path/rows:', len(prs))
SENS = ['LT04', 'LT05', 'LE07', 'LC08']
sess = requests.Session()

def list_prefix(prefix):
    out, tok = [], None
    while True:
        params = {'prefix': prefix, 'delimiter': '/', 'maxResults': 1000}
        if tok: params['pageToken'] = tok
        for k in range(5):
            try:
                r = sess.get(API, params=params, timeout=60); r.raise_for_status(); break
            except Exception:
                if k == 4: raise
        d = r.json(); out += d.get('prefixes', []); tok = d.get('nextPageToken')
        if not tok: return out

jobs = [f"{s}/01/{p[:3]}/{p[3:]}/" for p in prs for s in SENS]
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(list_prefix, jobs))
rows = []
pat = re.compile(r'(L[TEC]0\d)_(L1\w\w)_(\d{3})(\d{3})_(\d{8})_(\d{8})_01_(T1|T2|RT)/$')
for lst in res:
    for p in lst:
        m = pat.search(p)
        if m:
            sens, lvl, path, row, acq, proc, tier = m.groups()
            rows.append(dict(product_id=p.rstrip('/').split('/')[-1], prefix=p, sensor=sens, level=lvl,
                             pr=path + row, acq=acq, proc=proc, tier=tier))
sc = pd.DataFrame(rows)
print('listed', len(sc), sc.level.value_counts().to_dict(), sc.tier.value_counts().to_dict())
sc = sc[sc.level == 'L1TP']
# one product per acquisition: prefer T1 > T2 > RT, then latest processing
sc['tier_rank'] = sc.tier.map({'T1': 0, 'T2': 1, 'RT': 2})
sc = sc.sort_values(['sensor', 'pr', 'acq', 'tier_rank', 'proc'], ascending=[True, True, True, True, False])
sc = sc.drop_duplicates(['sensor', 'pr', 'acq'])
sc['date'] = pd.to_datetime(sc.acq); sc['year'] = sc.date.dt.year; sc['month'] = sc.date.dt.month
sc = sc[(sc.year >= 1985) & (sc.year <= 2020)]
ix = pd.read_csv(INDEX, usecols=['PRODUCT_ID', 'CLOUD_COVER'])
sc = sc.merge(ix.rename(columns={'PRODUCT_ID': 'product_id', 'CLOUD_COVER': 'cc'}), on='product_id', how='left')
# alternative match ignoring processing date/tier
key = lambda s: s.str[:25]
ix['k'] = key(ix.PRODUCT_ID); sc['k'] = key(sc.product_id)
alt = ix.drop_duplicates('k').set_index('k').CLOUD_COVER
sc['cc'] = sc.cc.fillna(sc.k.map(alt))
miss = sc[sc.cc.isna()]
print('scenes L1TP 1985-2020:', len(sc), ' missing cloud cover:', len(miss), miss.year.value_counts().head(8).to_dict())

def mtl_cc(prefix):
    pid = prefix.rstrip('/').split('/')[-1]
    url = f"https://storage.googleapis.com/gcp-public-data-landsat/{prefix}{pid}_MTL.txt"
    for k in range(4):
        try:
            t = sess.get(url, timeout=60).text
            m = re.search(r'CLOUD_COVER = ([\-\d\.]+)', t)
            return float(m.group(1)) if m else np.nan
        except Exception:
            pass
    return np.nan
with ThreadPoolExecutor(32) as ex:
    ccs = list(ex.map(mtl_cc, miss.prefix))
sc.loc[miss.index, 'cc'] = ccs
print('still missing cc:', sc.cc.isna().sum())
sc = sc[sc.cc.notna() & (sc.cc >= 0) & (sc.cc < 80)]
# rank: cloud cover, penalise Landsat 7 SLC-off
slc_off = (sc.sensor == 'LE07') & (sc.date >= '2003-05-31')
sc['rank_score'] = sc.cc + np.where(slc_off, 15, 0)
sc['season'] = np.where(sc.month.between(4, 9), 'dry', 'wet')
K_PER_SEASON = 3
sel = []
for (p, y), grp in sc.groupby(['pr', 'year']):
    picks = []
    for s in ['dry', 'wet']:
        picks += list(grp[grp.season == s].nsmallest(K_PER_SEASON, 'rank_score').index)
    if len(picks) < 2 * K_PER_SEASON:
        rest = grp.drop(picks).nsmallest(2 * K_PER_SEASON - len(picks), 'rank_score')
        picks += list(rest.index)
    sel += picks
sel = sc.loc[sel].copy()
print('selected scenes:', len(sel), ' per PR-year median:', sel.groupby(['pr', 'year']).size().median())
print('PR-years with <3 scenes:', (sel.groupby(['pr', 'year']).size() < 3).sum(), ' missing PR-years:',
      len(prs) * 36 - sel.groupby(['pr', 'year']).ngroups)
print(pd.crosstab(sel.year, sel.sensor).T.to_string())
sc.to_csv(f'{D}/scenes_all_l1tp.csv', index=False)
sel.to_csv(f'{D}/scenes_selected.csv', index=False)
