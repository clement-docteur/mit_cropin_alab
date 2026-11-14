"""Generates m2_mapbiomas_history_gee.js from the template (points embedded, no labels)."""
import pandas as pd, json
R = '/home/claude/p0'   # adjust: folder with pts/mb_points_all.csv and pts/qc_points_all.csv
mb = pd.read_csv(f'{R}/pts/mb_points_all.csv'); qc = pd.read_csv(f'{R}/pts/qc_points_all.csv')
pts = [[round(a, 6), round(b, 6), int(i), 'mb'] for a, b, i in mb[['lon', 'lat', 'id']].values] + \
      [[round(a, 6), round(b, 6), int(i), 'qc'] for a, b, i in qc[['lon', 'lat', 'id']].values]
js = open('m2_mapbiomas_history_template.js').read().replace('__PTS__', json.dumps(pts, separators=(',', ':'))).replace('__N__', str(len(pts)))
open('m2_mapbiomas_history_gee.js', 'w').write(js); print(len(pts))
