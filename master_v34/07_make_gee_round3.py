"""Master v34, step 7: write the round-3 GEE script (Hansen loss + B6, which round 2 did not deliver).
Written to the git-ignored 'Master Data/v34_work/gee/' because it holds coordinates."""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); W = f'{HERE}/../Master Data/v34_work'
q = pd.read_csv(f'{W}/qcnov_points_asof.csv')
pts = [[round(a, 6), round(b, 6), -1, int(i), 'qc', int(c), int(s)]
       for a, b, i, c, s in q[['lon', 'lat', 'id', 'asof_year', 'season_start']].values]
js = open(f'{HERE}/gee/r3_qcnov_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))) \
    .replace('__N__', str(len(pts))).replace('__CUTOFFS__', json.dumps(sorted({p[5] for p in pts})))
open(f'{W}/gee/v34_r3_gee.js', 'w').write(js)
print('written', f'{W}/gee/v34_r3_gee.js', len(js) // 1024, 'KB; tasks:', js.count('Export.table.toDrive'))
