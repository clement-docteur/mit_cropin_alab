"""Generates m1_b6_winter_gee.js. Same points and as-of cutoffs as Wave 2; labels are not embedded."""
import pandas as pd, json
R = '/home/claude/p0'
mb = pd.read_csv(f'{R}/pts/mb_points_all.csv'); qc = pd.read_csv(f'{R}/pts/qc_points_all.csv')
cd = pd.read_csv(f'{R}/qc_clearing_dates.csv')[['id', 'clear_month']]
qc = qc.merge(cd, on='id'); y = pd.to_datetime(qc.clear_month, errors='coerce').dt.year
qc['cutoff'] = y.sub(1).fillna(2025).astype(int)
pts = [[round(a, 6), round(b, 6), int(i), 'mb', 2022] for a, b, i in mb[['lon', 'lat', 'id']].values] + \
      [[round(a, 6), round(b, 6), int(i), 'qc', int(c)] for a, b, i, c in qc[['lon', 'lat', 'id', 'cutoff']].values]
js = open('m1_b6_winter_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))).replace('__N__', str(len(pts))) \
    .replace('__CUT_MB__', json.dumps(sorted({p[4] for p in pts if p[3] == 'mb'}))).replace('__CUT_QC__', json.dumps(sorted({p[4] for p in pts if p[3] == 'qc'})))
open('m1_b6_winter_gee.js', 'w').write(js); print('points', len(pts), 'KB', len(js) // 1024)
