"""Generates p7_alphaearth_gee.js (all 2,668 points with their as-of cutoff year). Labels are NOT embedded."""
import pandas as pd, json
R = '../results/phase0'
mb = pd.read_csv(f'{R}/mb_points_all.csv'); qc = pd.read_csv(f'{R}/qc_points_all.csv')
cd = pd.read_csv(f'{R}/qc_clearing_dates.csv')[['id', 'clear_month']]
qc = qc.merge(cd, on='id'); y = pd.to_datetime(qc.clear_month, errors='coerce').dt.year
qc['cutoff'] = y.sub(1).fillna(2025).astype(int)
pts = [[round(a, 6), round(b, 6), -1, int(i), 'mb', 2022] for a, b, i in mb[['lon', 'lat', 'id']].values] + \
      [[round(a, 6), round(b, 6), -1, int(i), 'qc', int(c)] for a, b, i, c in qc[['lon', 'lat', 'id', 'cutoff']].values]
js = open('p7_alphaearth_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))) \
    .replace('__N__', str(len(pts))).replace('__CUTOFFS__', json.dumps(sorted({p[5] for p in pts})))
open('p7_alphaearth_gee.js', 'w').write(js); print('points', len(pts), 'written', len(js) // 1024, 'KB')
