"""Master v34, step 4: write the round-2 GEE script (all features that depend on the as-of year).
Points carry their as-of year (cutoff) and season; labels are NOT embedded (-1).
Written to the git-ignored 'Master Data/v34_work/gee/' because it holds coordinates."""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = f'{HERE}/..'
W = f'{ROOT}/Master Data/v34_work'
q = pd.read_csv(f'{W}/qcnov_points_asof.csv')
pts = [[round(a, 6), round(b, 6), -1, int(i), 'qc', int(c), int(s)]
       for a, b, i, c, s in q[['lon', 'lat', 'id', 'asof_year', 'season_start']].values]
cut = sorted({p[5] for p in pts})
print('points by as-of year:', {c: sum(p[5] == c for p in pts) for c in cut})
js = open(f'{HERE}/gee/r2_qcnov_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))) \
    .replace('__N__', str(len(pts))).replace('__CUTOFFS__', json.dumps(cut))
assert '__PTS__' not in js and '__CUTOFFS__' not in js
open(f'{W}/gee/v34_r2_gee.js', 'w').write(js)
print('written', f'{W}/gee/v34_r2_gee.js', len(js) // 1024, 'KB; tasks:', js.count('Export.table.toDrive'))
