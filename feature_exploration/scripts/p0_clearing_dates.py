"""Phase 0 – estimate each QC point's clearing month from monthly Sentinel-2 NDVI
and pick the last full season (summer Dec–Mar) that ends before it (rule 8).
Clearing = first month where NDVI falls >= 0.20 below the point's 2022–2025 median
AND below 0.65, confirmed by the next clear observation (within 3 months)."""
import pandas as pd, numpy as np
m = pd.read_csv('gee/p0_qc_ndvi_monthly.csv')[['id','label','month','NDVI','NBR','n_clear']].dropna()
m['t'] = pd.PeriodIndex(m.month, freq='M')
out = []
for pid, g in m.sort_values('t').groupby('id'):
    base = g[g.t <= pd.Period('2025-12','M')].NDVI.median()
    low = (g.NDVI < base - 0.20) & (g.NDVI < 0.65)
    clear_t = None
    idx = list(g.index)
    for k, i in enumerate(idx):
        if not low[i] or g.t[i] < pd.Period('2022-03','M'): continue
        nxt = [j for j in idx[k+1:] if (g.t[j] - g.t[i]).n <= 3]
        if not nxt:   # last observation: accept only if it's the final month(s)
            if g.t[i] >= pd.Period('2026-09','M'): clear_t = g.t[i]; break
            continue
        if low[nxt[0]]: clear_t = g.t[i]; break
    out.append(dict(id=pid, label=g.label.iloc[0], base_ndvi=base, clear_month=str(clear_t) if clear_t else None))
c = pd.DataFrame(out)
def pick(cm):
    if not isinstance(cm, str): return 2025          # none detected: Cropin said clearings were ~3–6 months before mid-Oct 2026
    p = pd.Period(cm, 'M')
    ok = [y for y in [2022, 2023, 2024, 2025] if pd.Period(f'{y+1}-03', 'M') < p]
    return max(ok) if ok else None
c['season_start'] = c.clear_month.map(pick)
c['clearing_detected'] = c.clear_month.notna()
c.to_csv('qc_clearing_dates.csv', index=False)
print('Detected:', c.clearing_detected.sum(), 'of', len(c))
print(c.groupby('label').clearing_detected.mean().round(2))
print(c.clear_month.dropna().str[:7].value_counts().sort_index().to_string())
print(c.season_start.value_counts(dropna=False))
