"""Master v34, step 2: write the round-1 GEE script with the November QC points embedded.
All 150 deduplicated points are exported, repeats included: their features should match v3,
which checks that the rebuilt pipeline reproduces v3. The script holds coordinates, so it is
written to the git-ignored 'Master Data/v34_work/gee/'."""
import json, os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = f'{ROOT}/Master Data/v34_work/gee'
os.makedirs(OUT, exist_ok=True)
q = pd.read_csv(f'{ROOT}/Master Data/v34_work/qcnov_points_all.csv')
raw = json.dumps([[round(a, 6), round(b, 6), int(c), int(d)] for a, b, c, d in q[['lon', 'lat', 'label', 'id']].values],
                 separators=(',', ':'))
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gee', 'r1_qcnov_template.js')).read()
js = tpl.replace('__QC_RAW__', raw).replace('__N_QC__', str(len(q)))
assert '__' not in js.replace('__proto__', ''), 'unfilled placeholder'
open(f'{OUT}/v34_r1_gee.js', 'w').write(js)
print('written', f'{OUT}/v34_r1_gee.js', len(js) // 1024, 'KB,', len(q), 'points')
