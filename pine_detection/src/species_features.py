"""Shared helpers for pine_detection/notebooks/pine_vs_eucalyptus.ipynb.

Statistics come from src/qc_analysis.py (auc_ci, mwu_test, holm, spatial_groups, cv_evaluate), which is
imported rather than copied. This file only adds what is specific to species:
- loading the master feature table together with the SDPT species labels;
- rotation-length and thinning features from the yearly Landsat series, plantation age from MapBiomas,
  and a wrapper around the team's harvest-cycle detector;
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


# Readable names for the master-table features (see combined_model/docs/v3_columns.csv) and the new ones.
NAMES = {
    "nc_B_SAG_SON": "SAG Sep–Nov", "nc_B_SAG_phase_sin": "SAG seasonal timing", "mm_SAG_win": "SAG winter",
    "nc_wi3_NDMI": "NDMI winter", "nc_su3_NDMI": "NDMI summer", "nc_sg3_NDVI": "NDVI seasonal range",
    "mm_NDVI_win": "NDVI winter", "mm_B6_win": "B6 red-edge winter", "mm_B8A_win": "B8A narrow NIR winter",
    "nc_B_NDRE_phase_sin": "NDRE seasonal timing", "nc_B_SWIR_phase_sin": "SWIR seasonal timing (sin)",
    "nc_B_SWIR_phase_cos": "SWIR seasonal timing (cos)", "nc_E_eth_h": "canopy height (ETH)",
    "nc_E_eth_sd50": "canopy height variability", "nc_F_VHminusVV_median": "radar VH − VV",
    "nc_F_VH_median": "radar VH", "nc_D_max_nbr_drop": "largest NBR drop", "nc_D_cc_n_breaks": "CCDC breaks",
    "nc_D_years_continuous_forest": "years of continuous forest", "nc_D_lt_n_vertices": "LandTrendr vertices",
    "nc_D_ndvi_1985_89": "NDVI 1985–89", "nc_D_h_total_loss_px": "Hansen loss within 150 m",
    "nc_nfw": "Natural Forests of the World", "nc_H_mapbiomas10m_plantation": "MapBiomas 10 m plantation",
    "rh_n_cycles": "harvest cycles (team)", "rh_s_std": "NBR variability (team)",
    "rh_s_frac_below_forest": "share of years below canopy", "rh_abrupt_start": "abrupt start on open land",
    "nc_hansen_tc2000": "Hansen tree cover 2000", "lh_natural_run_years": "years of natural cover (MapBiomas)",
    "lh_n_changes": "land-cover changes (MapBiomas)", "lh_ever_open": "ever open land (MapBiomas)",
    "nc_A_SAG": "SAG Oct–Jan (Cropin)", "nc_chm_mean_r15": "canopy height (Meta 1 m)",
    "nc_chm_stdDev_r15": "canopy height variability (Meta 1 m)",
    "rot_n_cuts": "clear-cuts 2000–20 (new)", "rot_med_interval": "years between clear-cuts (new)",
    "rot_years_since": "years since last clear-cut (new)", "rot_n_partial": "partial drops (new)",
    "rot_partial_per_cut": "partial drops per clear-cut (new)", "lat": "latitude", "lon": "longitude",
}


def master_feature_columns():
    """The master table's model features and benchmarks (combined_model/docs/v3_columns.csv)."""
    cols = pd.read_csv(ROOT / "combined_model/docs/v3_columns.csv")
    return cols.loc[cols.role.isin(["feature", "benchmark"]), "column"].tolist()


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
HC = ROOT / "public_data/harvest_cycles"


def load_series(pids, name="nbr_median"):
    """A yearly Landsat series 1985-2020 from public_data/harvest_cycles, one row per team_pid.
    name: nbr_median, nbr_dry_median, nbr_wet_median, nbr_max or ndvi_median."""
    s = pd.read_csv(HC / f"series_{name}.csv.gz", index_col=0)
    s.columns = s.columns.astype(int)
    return s.reindex(pids)


def load_nbr(pids):
    """Yearly median Landsat NBR 1985-2020, one row per team_pid."""
    return load_series(pids, "nbr_median")


def plantation_years(pids, start=1985, end=2020):
    """From MapBiomas's yearly class history (class 9 = forest plantation): first year as plantation and
    number of plantation years in start..end. NaN first year if never plantation."""
    h = pd.read_csv(HC / "val_label_history.csv", index_col=0).reindex(pids)
    h.columns = [int(c.split("_")[1]) for c in h.columns]
    is9 = h[list(range(start, end + 1))].round() == 9
    return pd.DataFrame({"first_plant_year": is9.idxmax(axis=1).where(is9.any(axis=1)),
                         "n_plant_years": is9.sum(axis=1)}, index=h.index)


def team_harvest_features():
    """The team's harvest-cycle detector (combined_model/scripts/harvest_features.py: segmented NBR fit,
    1985-2020), run unchanged on all public points. One row per team_pid; takes about 20 s."""
    sys.path.insert(0, str(ROOT / "combined_model/scripts"))
    import harvest_features as hf
    _, feat, _, _ = hf.run(str(HC))
    return feat


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
