"""Phase 0 – split deduplicated QC points into dev / held-back halves by 0.25° blocks.
Picks, among 2,000 random block assignments (seed 2026), the one whose forest and
plantation counts are closest to 50/50."""
import numpy as np, pandas as pd
OUT = '../results/phase0'
qc = pd.read_csv(f'{OUT}/qc_points_prepped.csv')
qc['block'] = np.floor(qc.lon/0.25).astype(int).astype(str)+'_'+np.floor(qc.lat/0.25).astype(int).astype(str)
blocks = qc.block.unique(); rng = np.random.default_rng(2026)
nf, npl = (qc.label==0).sum(), (qc.label==1).sum()
best = None
for _ in range(2000):
    dev = set(rng.choice(blocks, size=len(blocks)//2, replace=False))
    m = qc.block.isin(dev)
    score = abs((qc[m].label==0).sum()-nf/2) + abs((qc[m].label==1).sum()-npl/2)
    if best is None or score < best[0]: best = (score, dev)
qc['split'] = np.where(qc.block.isin(best[1]), 'dev', 'heldback')
print(qc.groupby(['split','label']).size().unstack().rename(columns={0:'forest',1:'plantation'}))
qc.to_csv(f'{OUT}/qc_points_all.csv', index=False)
qc[qc.split=='heldback'].to_csv(f'{OUT}/HELDBACK_qc_points_DO_NOT_OPEN.csv', index=False)
qc[qc.split=='dev'].to_csv(f'{OUT}/qc_points_dev.csv', index=False)
