"""Master v34, step 3: clearing month, as-of year and season for the November QC points.
Rule copied from feature_exploration/scripts/p0_clearing_dates.py (unchanged):
  clearing = first month from Mar 2022 where NDVI falls >= 0.20 below the point's median (to Dec 2025)
  AND below 0.65, confirmed by the next clear observation within 3 months.
  asof_year = clearing year - 1 (2025 if no clearing found)   [w1_features.py, p7_make_gee_script.py]
  season_start = last summer (Dec-Mar) that ends before the clearing (2025 if none found)
Check: the repeats of v3 polygons should get the same as-of year as in v3."""
import os
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
W = f'{ROOT}/Master Data/v34_work'
m = pd.read_csv(f'{W}/gee/v34_qc_ndvi_monthly.csv')[['id', 'label', 'month', 'NDVI', 'NBR', 'n_clear']].dropna()
m['t'] = pd.PeriodIndex(m.month, freq='M')
out = []
for pid, g in m.sort_values('t').groupby('id'):
    base = g[g.t <= pd.Period('2025-12', 'M')].NDVI.median()
    low = (g.NDVI < base - 0.20) & (g.NDVI < 0.65)
    clear_t = None; idx = list(g.index)
    for k, i in enumerate(idx):
        if not low[i] or g.t[i] < pd.Period('2022-03', 'M'): continue
        nxt = [j for j in idx[k + 1:] if (g.t[j] - g.t[i]).n <= 3]
        if not nxt:
            if g.t[i] >= pd.Period('2026-09', 'M'): clear_t = g.t[i]; break
            continue
        if low[nxt[0]]: clear_t = g.t[i]; break
    out.append(dict(id=pid, base_ndvi=base, clear_month=str(clear_t) if clear_t else None))
c = pd.DataFrame(out)


def pick(cm):
    if not isinstance(cm, str): return 2025
    p = pd.Period(cm, 'M')
    ok = [y for y in [2022, 2023, 2024, 2025] if pd.Period(f'{y + 1}-03', 'M') < p]
    return max(ok) if ok else None


c['season_start'] = c.clear_month.map(pick)
c['clearing_detected'] = c.clear_month.notna()
cy = pd.to_datetime(c.clear_month, errors='coerce').dt.year
c['asof_year'] = np.where(cy.notna(), cy - 1, 2025).astype(int)

q = pd.read_csv(f'{W}/qcnov_points_all.csv').merge(c, on='id', how='left')
assert len(q) == 150 and q.base_ndvi.notna().all()
print('Clearing detected:', int(q.clearing_detected.sum()), 'of', len(q))
print(q.groupby(['split', 'label']).clearing_detected.mean().round(2).unstack().rename(columns={0: 'forest', 1: 'plantation'}).to_string())
print('as-of year:', q.asof_year.value_counts().sort_index().to_dict())
print('season_start:', q.season_start.value_counts(dropna=False).sort_index().to_dict())
print('clear month by quarter:', pd.PeriodIndex(q.clear_month.dropna(), freq='M').asfreq('Q').value_counts().sort_index().astype(int).to_dict())

# ---- check against v3 for the repeats ----
v3 = pd.read_csv(f'{ROOT}/Master Data/master_features_v3.csv')
v3q = v3[v3.set == 'qc'][['id', 'lon', 'lat', 'asof_year', 'clear_month']]
rep = q[q.split == 'nov_repeat'].copy()
k = [int(np.argmin(np.hypot((v3q.lon.values - r.lon) * 111320 * np.cos(np.deg2rad(r.lat)), (v3q.lat.values - r.lat) * 110540)))
     for r in rep.itertuples()]
rep['v3_asof'] = v3q.asof_year.values[k]; rep['v3_clear'] = v3q.clear_month.values[k]
print(f'Repeats: same as-of year as v3 {int((rep.asof_year == rep.v3_asof).sum())} of {len(rep)}; '
      f'clearing found now but not in v3 {int((rep.clear_month.notna() & rep.v3_clear.isna()).sum())}; '
      f'in v3 but not now {int((rep.clear_month.isna() & rep.v3_clear.notna()).sum())}')
q.to_csv(f'{W}/qcnov_points_asof.csv', index=False)
print('written qcnov_points_asof.csv')
