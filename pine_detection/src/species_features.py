"""Shared helpers for the pine-vs-eucalyptus notebooks (pine_detection/notebooks/P0x).

Statistics come from src/qc_analysis.py (auc_ci, mwu_test, holm, spatial_groups, cv_evaluate), which is
imported rather than copied. This file only adds what is specific to species:
- loading the master feature table together with the SDPT species labels;
- rotation-length and thinning features from the yearly Landsat NBR series;
- a species palette, and a savefig that writes to pine_detection/figures/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PD = Path(__file__).resolve().parents[1]          # pine_detection/
ROOT = PD.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PD / "src"))

import qc_analysis as qa  # noqa: E402  (shared helpers)
import sdpt_overlay as so  # noqa: E402

FIG = PD / "figures"
PAL_SPECIES = {"pine": "#2F5D50", "eucalyptus": "#C08A2B", "grey": qa.PAL["grey"], "ink": qa.PAL["ink"]}
SDPT_MAP = {"Pinus sp.": "pine", "Eucalyptus sp.": "eucalyptus", "Acacia sp.": "acacia", "Unknown": "unknown"}


def savefig(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")


# ------------------------------------------------------------------ labels
def load_master():
    """Master feature table v3 joined with AlphaEarth (`ae_*`), MapBiomas points only."""
    m = pd.read_csv(ROOT / "Master Data/master_features_v3.csv")
    ae = pd.read_csv(ROOT / "Master Data/master_alphaearth_v3.csv").drop(columns=["id", "set", "split"])
    m = m.merge(ae, on="uid", how="left")
    return m[m["set"] == "mb"].copy()   # QC rows are never sent to the tile server


def add_sdpt(d, cache=ROOT / "data/processed/sdpt_tiles", near_m=100.0):
    """Adds sdpt_in / sdpt_near / sdpt_near_m and `species` (pine, eucalyptus, acacia, unknown, none).
    Fetches any missing SDPT v2.0 tile around these PUBLIC MapBiomas points (cached locally)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for t in sorted({so.tile_of(lo, la) for lo, la in zip(d.lon, d.lat)}):
        so.fetch(t, cache)
    out = so.overlay(d, cache, near_m)
    out["species"] = out["sdpt_in"].map(SDPT_MAP).fillna("none")
    return out


# ------------------------------------------------------------------ rotation features (H S1/S2)
def load_nbr(pids):
    """Yearly median Landsat NBR 1985-2020 (public_data/harvest_cycles), one row per team_pid."""
    s = pd.read_csv(ROOT / "public_data/harvest_cycles/series_nbr_median.csv.gz", index_col=0)
    s.columns = s.columns.astype(int)
    return s.reindex(pids)


def rotation_features(nbr, cut=0.20, thin=0.08, start=2000, end=2020, min_gap=3):
    """Simple, fixed-rule harvest events from a yearly NBR series.

    fall(t) = NBR(t) - max(NBR(t-1), NBR(t-2))   (drop from the recent level)
    - clear-cut in year t: fall <= -cut, and at least `min_gap` years after the previous clear-cut
    - partial drop (thinning candidate): -cut < fall <= -thin, not within 1 year of a clear-cut
    All counts use start..end (2000-2020 by default) so every point has the same window.
    Returns one row per point:
      rot_n_cuts        number of clear-cuts
      rot_med_interval  median years between consecutive clear-cuts (NaN with < 2 cuts)
      rot_years_since   years from the last clear-cut to `end` (end - start + 1 if none)
      rot_n_partial     number of partial drops
      rot_partial_per_cut  partial drops per clear-cut + 1 (thinning intensity)
    """
    years = [y for y in nbr.columns if start <= y <= end]
    rows = []
    for _, s in nbr.iterrows():
        cuts, partial = [], []
        for y in years:
            prev = s.reindex([y - 1, y - 2]).max()
            if pd.isna(s.get(y)) or pd.isna(prev):
                continue
            f = s[y] - prev
            if f <= -cut and (not cuts or y - cuts[-1] >= min_gap):
                cuts.append(y)
            elif -cut < f <= -thin:
                partial.append(y)
        partial = [y for y in partial if all(abs(y - c) > 1 for c in cuts)]
        rows.append(dict(
            rot_n_cuts=len(cuts),
            rot_med_interval=float(np.median(np.diff(cuts))) if len(cuts) >= 2 else np.nan,
            rot_years_since=(end - cuts[-1]) if cuts else (end - start + 1),
            rot_n_partial=len(partial),
            rot_partial_per_cut=len(partial) / (len(cuts) + 1),
        ))
    return pd.DataFrame(rows, index=nbr.index)
