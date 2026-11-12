"""Generates w1c_refit_gee.js: LandTrendr and CCDC refit with each series ending at the point's
as-of cutoff (reviewer item 7). MapBiomas cutoff = 2022; QC cutoff = year before the estimated
clearing year. QC points with cutoff 2025 are skipped (the original fit already ends in 2025)."""
import pandas as pd, json
R = '../results/phase0'
mb = pd.read_csv(f'{R}/mb_points_all.csv'); qc = pd.read_csv(f'{R}/qc_points_all.csv')
cd = pd.read_csv(f'{R}/qc_clearing_dates.csv')[['id', 'clear_month']]
qc = qc.merge(cd, on='id')
y = pd.to_datetime(qc.clear_month, errors='coerce').dt.year
qc['cutoff'] = y.sub(1).fillna(2025).astype(int)
qc = qc[qc.cutoff < 2025]
pts = [[round(a, 6), round(b, 6), int(l), int(i), 'mb', 2022] for a, b, l, i in mb[['lon', 'lat', 'label', 'id']].values] + \
      [[round(a, 6), round(b, 6), int(l), int(i), 'qc', int(c)] for a, b, l, i, c in qc[['lon', 'lat', 'label', 'id', 'cutoff']].values]
groups = sorted({(p[4], p[5]) for p in pts})
print('groups:', {f'{s}-{c}': sum(1 for p in pts if p[4] == s and p[5] == c) for s, c in groups})
js = open('w1c_refit_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))) \
    .replace('__N__', str(len(pts))).replace('__CUTOFFS__', json.dumps(sorted({p[5] for p in pts})))
open('w1c_refit_gee.js', 'w').write(js); print('written', len(js) // 1024, 'KB')
