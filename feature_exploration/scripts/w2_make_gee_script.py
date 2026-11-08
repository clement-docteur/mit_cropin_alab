"""Generates w2_batch2_gee.js (Wave 2: Family B seasonal, Family F radar). Every point carries its
as-of cutoff year; all time windows are the 3 calendar years ending at the cutoff
(MapBiomas 2020–2022; most QC points 2023–2025)."""
import pandas as pd, json
R = '../results/phase0'
mb = pd.read_csv(f'{R}/mb_points_all.csv'); qc = pd.read_csv(f'{R}/qc_points_all.csv')
cd = pd.read_csv(f'{R}/qc_clearing_dates.csv')[['id', 'clear_month']]
qc = qc.merge(cd, on='id'); y = pd.to_datetime(qc.clear_month, errors='coerce').dt.year
qc['cutoff'] = y.sub(1).fillna(2025).astype(int)
pts = [[round(a, 6), round(b, 6), int(l), int(i), 'mb', 2022] for a, b, l, i in mb[['lon', 'lat', 'label', 'id']].values] + \
      [[round(a, 6), round(b, 6), int(l), int(i), 'qc', int(c)] for a, b, l, i, c in qc[['lon', 'lat', 'label', 'id', 'cutoff']].values]
cut = sorted({p[5] for p in pts})
print('points', len(pts), 'cutoff groups', {c: sum(p[5] == c for p in pts) for c in cut})
js = open('w2_batch2_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))) \
    .replace('__N__', str(len(pts))).replace('__CUT_MB__', json.dumps(sorted({p[5] for p in pts if p[4] == 'mb'}))).replace('__CUT_QC__', json.dumps(sorted({p[5] for p in pts if p[4] == 'qc'})))
open('w2_batch2_gee.js', 'w').write(js); print('written', len(js) // 1024, 'KB')
