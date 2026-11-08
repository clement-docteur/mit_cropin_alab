"""Generates w1_batch1_gee.js with the Phase 0 points embedded."""
import pandas as pd, json
R = '../results/phase0'
mb = pd.read_csv(f'{R}/mb_points_all.csv'); qc = pd.read_csv(f'{R}/qc_points_all.csv')
mb_raw = json.dumps([[round(a,6), round(b,6), int(c), int(d)] for a,b,c,d in mb[['lon','lat','label','id']].values], separators=(',',':'))
qc_raw = json.dumps([[round(a,6), round(b,6), int(c), int(d)] for a,b,c,d in qc[['lon','lat','label','id']].values], separators=(',',':'))
js = open('w1_batch1_template.js').read().replace('__MB_RAW__', mb_raw).replace('__QC_RAW__', qc_raw) \
       .replace('__N_MB__', str(len(mb))).replace('__N_QC__', str(len(qc)))
open('w1_batch1_gee.js','w').write(js); print('written', len(js)//1024, 'KB')
