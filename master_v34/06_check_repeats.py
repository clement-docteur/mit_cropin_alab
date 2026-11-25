"""Master v34, step 6: check the rebuilt pipeline on the 44 repeats of v3 polygons (same place, so features
should match v3), and recover the two harvest-cycle levels v3 used but did not save.
  closed-canopy NBR: rh_s_frac_below_forest = share of (despiked) years with NBR < canopy  -> pick the canopy value
                     that reproduces v3 on the most repeats (midpoint of the best interval)
  open-land NBR:     only used by rh_abrupt_start                                           -> same, on that column
Prints aggregates only. Writes 'Master Data/v34_work/rh_levels.json' and 'repeat_check.csv' (git-ignored)."""
import json, os, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = f'{HERE}/..'; W = f'{ROOT}/Master Data/v34_work'
spec = importlib.util.spec_from_file_location('bf', f'{HERE}/05_build_features.py'); bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

pts = pd.read_csv(f'{W}/qcnov_points_asof.csv')
v3 = pd.read_csv(f'{ROOT}/Master Data/master_features_v3.csv'); v3q = v3[v3.set == 'qc'].reset_index(drop=True)
rep = pts[pts.split == 'nov_repeat'].copy()
k = [int(np.argmin(np.hypot((v3q.lon.values - r.lon) * 111320 * np.cos(np.deg2rad(r.lat)), (v3q.lat.values - r.lat) * 110540)))
     for r in rep.itertuples()]
rep['v3_row'] = k
same_asof = rep.asof_year.values == v3q.asof_year.values[k]
print(f'Repeats: {len(rep)}; same as-of year as v3: {int(same_asof.sum())} (only these are compared)')
rep = rep[same_asof]; V = v3q.iloc[rep.v3_row.values].reset_index(drop=True)

# ---- recover the harvest levels ----
S = bf.nbr_series(pts).loc[rep.id.values]
desp = [bf._g['despike'](r[np.isfinite(r)]) for r in S.to_numpy(float)]
grid = np.round(np.arange(0.30, 0.80, 0.0005), 4)
hit = np.array([np.sum([abs(np.mean(d < c) - t) < 1e-9 for d, t in zip(desp, V.rh_s_frac_below_forest)]) for c in grid])
best = grid[hit == hit.max()]; canopy = float(np.round((best.min() + best.max()) / 2, 4))
print(f'closed-canopy NBR: {canopy} (matches rh_s_frac_below_forest on {hit.max()} of {len(rep)}; '
      f'interval {best.min()}-{best.max()})')
t_y = [(np.asarray(bf.YEARS[np.isfinite(r)], float), r[np.isfinite(r)]) for r in S.to_numpy(float)]
og = np.round(np.arange(-0.30, canopy - 0.10 + 1e-9, 0.0025), 4)
hit2 = np.array([np.sum([bf._g['abrupt_start'](t, y, canopy, o)[0] == a for (t, y), a in zip(t_y, V.rh_abrupt_start)]) for o in og])
best2 = og[hit2 == hit2.max()]
open_level = float(canopy - 0.10) if (canopy - 0.10) in best2 or np.isclose(best2.max(), canopy - 0.10) else float(np.median(best2))
print(f'open-land NBR: {open_level} (matches rh_abrupt_start on {hit2.max()} of {len(rep)}; '
      f'v3 rule caps it at canopy - 0.10 = {canopy - 0.10:.4f}; best range {best2.min()}-{best2.max()})')
json.dump({'closed_canopy_nbr': canopy, 'open_land_nbr': open_level}, open(f'{W}/rh_levels.json', 'w'))

# ---- compare every slim feature ----
F = bf.build(canopy, open_level).set_index('id').loc[rep.id.values].reset_index()
slim = pd.read_csv(f'{ROOT}/Master Data/master_features_v3_slim.csv', nrows=0).columns
feats = [c for c in slim if c not in ('uid', 'set', 'split', 'label_forest', 'train_eligible', 'border', 'block')]
out = []
for c in feats:
    a, b = F[c].astype(float).values, V[c].astype(float).values
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        out.append(dict(column=c, n=0)); continue
    d = np.abs(a[ok] - b[ok]); scale = np.nanstd(v3q[c].astype(float))
    out.append(dict(column=c, n=int(ok.sum()), exact_share=round(float(np.mean(d < 1e-6 * max(1, scale))), 2),
                    close_share=round(float(np.mean(d <= 0.05 * scale)), 2),
                    median_absdiff_in_sd=round(float(np.median(d) / scale), 3) if scale > 0 else 0,
                    spearman=round(float(pd.Series(a[ok]).corr(pd.Series(b[ok]), 'spearman')), 3) if ok.sum() > 3 else np.nan))
R = pd.DataFrame(out); R.to_csv(f'{W}/repeat_check.csv', index=False)
pd.set_option('display.width', 200); print(R.to_string(index=False))
