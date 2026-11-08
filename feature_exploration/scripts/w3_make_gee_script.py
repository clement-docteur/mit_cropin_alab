"""Generates w3_batch3_gee.js (Wave 3: E structure, G terrain/context/geometry, H external layers).
Each point carries its cutoff (as-of year) and Sentinel-2 summer season (same as Phase 0)."""
import pandas as pd, json
R = '../results/phase0'
mb = pd.read_csv(f'{R}/mb_points_all.csv'); qc = pd.read_csv(f'{R}/qc_points_all.csv')
cd = pd.read_csv(f'{R}/qc_clearing_dates.csv')[['id', 'clear_month', 'season_start']]
qc = qc.merge(cd, on='id'); y = pd.to_datetime(qc.clear_month, errors='coerce').dt.year
qc['cutoff'] = y.sub(1).fillna(2025).astype(int)
qc['season_start'] = qc.season_start.fillna(qc.cutoff - 1).astype(int)   # 1 point with no pre-clearing season: placeholder, dropped in analysis
pts = [[round(a, 6), round(b, 6), int(l), int(i), 'mb', 2022, 2021] for a, b, l, i in mb[['lon', 'lat', 'label', 'id']].values] + \
      [[round(a, 6), round(b, 6), int(l), int(i), 'qc', int(c), int(s)] for a, b, l, i, c, s in qc[['lon', 'lat', 'label', 'id', 'cutoff', 'season_start']].values]
cut = {s: sorted({p[5] for p in pts if p[4] == s}) for s in ('mb', 'qc')}
sea = sorted({p[6] for p in pts})
js = open('w3_batch3_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))).replace('__N__', str(len(pts))) \
    .replace('__CUT_MB__', json.dumps(cut['mb'])).replace('__CUT_QC__', json.dumps(cut['qc'])).replace('__SEASONS__', json.dumps(sea))
open('w3_batch3_gee.js', 'w').write(js); print('points', len(pts), 'cutoffs', cut, 'seasons', sea, 'written', len(js) // 1024, 'KB')
