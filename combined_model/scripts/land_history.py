"""Land-history features rebuilt on our data (midterm idea: missed plantations often grew on former farmland;
Cropin's cue: a chain of land-cover changes vs stable forest). Source: MapBiomas Collection 9, 30 m, 1985–2023
(batch M2), read at our points, cut at each point's as-of year (MapBiomas points 2022; QC min(as-of, 2023)).
MAPBIOMAS-DERIVED: circular on MapBiomas points; report models with and without."""
import numpy as np, pandas as pd
NAT = {3, 4, 5, 6, 49}                                     # natural woody: forest, savanna, mangrove, floodable forest, wooded restinga
FARM = {15, 18, 19, 20, 21, 35, 36, 39, 40, 41, 46, 47, 48, 62}   # pasture, agriculture, mosaic of uses
OPEN = FARM | {12, 11, 23, 24, 25, 29, 30, 50}             # farm + grassland, wetland, beach, urban, other non-vegetated, rock, mining, herbaceous restinga
def build(master):
    H = pd.read_csv('/mnt/user-data/uploads/cropin_texture/m2_mapbiomas_history.csv')
    asof = master.set_index(['id', 'set']).asof_year
    rows = []
    for r in H.itertuples():
        key = (r.id, r.set)
        if key not in asof.index: continue
        last = int(min(asof[key], 2023))
        seq = [getattr(r, f'classification_{y}') for y in range(1985, last + 1)]
        seq = [int(v) if pd.notna(v) else -1 for v in seq]
        run = 0
        for v in reversed(seq):
            if v in NAT: run += 1
            else: break
        rows.append({'id': r.id, 'set': r.set, 'lh_natural_run_years': run, 'lh_forest_all_years': int(all(v in NAT for v in seq)),
                     'lh_ever_farm': int(any(v in FARM for v in seq)), 'lh_farm_years': sum(v in FARM for v in seq),
                     'lh_ever_open': int(any(v in OPEN for v in seq)), 'lh_n_changes': sum(a != b for a, b in zip(seq[:-1], seq[1:]))})
    return pd.DataFrame(rows)
