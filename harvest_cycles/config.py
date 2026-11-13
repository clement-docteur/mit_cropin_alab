"""
Settings of the harvest-cycle pipeline: every threshold and choice in one place.

- PreprocessSettings: raw Landsat scenes -> cloud-masked dated observations -> one value per point and year
  (used by preprocess_landsat.py).
- FeatureSettings: yearly NBR series and dated observations -> harvest-cycle features
  (used by build_harvest_cycles_features.py and by harvest_cycles_findings.ipynb).

The defaults reproduce the analysis in harvest_cycles_findings.ipynb exactly. To change a default, edit it here.
To change a value for one run only, pass `--set name=value` to either script (repeatable). Every output comes
with a `.settings.json` file that records the values used, so any table can be traced back to its settings.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TypeVar


@dataclass(frozen=True)
class PreprocessSettings:
    """Raw Landsat -> dated observations -> yearly series."""

    # Period. Only imagery up to the EUDR cutoff (31 December 2020) is used.
    first_year: int = 1985
    last_year: int = 2020

    # Source: Landsat Collection 1 Level-1 (top-of-atmosphere reflectance) on Google's public bucket.
    # Collection 1 is deprecated; see README "Before production".
    landsat_bucket: str = "gcp-public-data-landsat"
    sensors: tuple = ("LT04", "LT05", "LE07", "LC08")
    hansen_url: str = ("https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/"
                       "Hansen_GFC-2024-v1.12_{layer}_{tile}.tif")
    index_year_padding: int = 1           # the scene index is read for first_year - 1 .. last_year + 1
    region_margin_degrees: float = 1.0    # the index is read for the points' bounding box plus this margin

    # Path/rows: each point uses its nearest path/row centre, plus the second nearest when it is this close.
    max_daytime_row: int = 122
    min_scenes_per_pathrow: int = 50      # path/rows with fewer L1TP scenes in the index are ignored
    second_pathrow_max_km: float = 105.0

    # Scene selection per path/row and year: the least cloudy scenes in each season.
    max_cloud_cover: float = 80.0         # scene cloud cover, %
    scenes_per_season: int = 3
    dry_months: tuple = (4, 5, 6, 7, 8, 9)   # Apr-Sep; the other months are the wet season. Used both to pick
    # scenes (so changing it means a new download) and for the dry/wet series; keep it equal to
    # FeatureSettings.seasonal_dry_months.
    slc_off_date: str = "2003-05-31"      # Landsat 7 scan-line corrector failure
    slc_off_penalty: float = 15.0         # added to a Landsat 7 SLC-off scene's cloud cover when ranking

    # Scenes that cannot be read after the retries are listed in failed_scenes.csv and left out. If they are more
    # than this share of the selected scenes (usually a network problem), the run stops before step 5 instead.
    max_failed_scene_share: float = 0.01

    # Valid observations (top-of-atmosphere reflectance and indices)
    max_reflectance: float = 1.2
    min_nir: float = 0.05                 # below: water, deep shadow or residual cloud
    min_nbr: float = -0.6
    min_ndvi: float = -0.3

    # Sensor harmonisation onto the Landsat 7 ETM+ scale: index offsets (sensor minus ETM+) removed from Landsat 8
    # OLI and Landsat 4/5 TM. Defaults: medians of same-point pairs <= 8 days apart at the 2,264 analysis points
    # (8,066 OLI pairs, 17,601 TM pairs). Set estimate_offsets=true to measure them on the new points instead
    # (needs at least offset_min_pairs pairs; small batches give noisy estimates).
    nbr_oli_minus_etm: float = -0.0012650421967451142
    nbr_tm_minus_etm: float = -0.0073901963057453335
    ndvi_oli_minus_etm: float = 0.0330674985708751
    ndvi_tm_minus_etm: float = -0.0304940269010735
    estimate_offsets: bool = False
    offset_pair_max_days: int = 8
    offset_years_oli: tuple = (2013, 2020)
    offset_years_tm: tuple = (1999, 2011)
    offset_min_pairs: int = 30

    # Yearly series written: (index, aggregation). Aggregations: median, dry_median, wet_median, max.
    series: tuple = (("nbr", "median"), ("nbr", "dry_median"), ("nbr", "wet_median"), ("nbr", "max"),
                     ("ndvi", "median"))


@dataclass(frozen=True)
class FeatureSettings:
    """Yearly NBR series and dated observations -> harvest-cycle features."""

    # Period and coverage
    first_year: int = 1985
    last_year: int = 2020                 # EUDR cutoff; also the reference year of years_since_last_cut
    min_years: int = 8                    # fewer years with data: the features are left empty

    # One-year spikes (LandTrendr-style): a value far from both neighbours, which agree with each other
    despike_threshold: float = 0.9
    despike_min_spike: float = 0.05

    # Segmentation: straight lines through the yearly values, number of segments chosen by BIC
    max_segments: int = 18                # LandTrendr's usual 6 leaves room for only about two cuts
    bic_penalty: float = 1.0

    # Cuts and cycles
    drop: float = 0.20                    # minimum NBR fall between two segment ends
    max_cut_years: int = 3                # ... within at most this many years
    cycle_regrow_fraction: float = 0.5    # a cycle grows back this share of the fall ...
    max_cycle_years: int = 10             # ... within this many years after the low point
    regrow_time_fraction: float = 0.8     # regrowth time = years to grow back this share of the fall
    fast_regrowth_years: float = 4.0      # fast_regrowth: a cycle regrowing within this many years
    fall_window_years: int = 3            # s_max_drop3: largest fall from the highest of the previous this-many
                                          # values (years with data); the column name keeps the default 3
    min_regrow_years: float = 1.0         # regrow_speed = 1 / max(regrowth time, this)
    years_since_last_cut_if_none: float = 40.0   # value of years_since_last_cut when no cut was found

    # NBR levels. Defaults: calibrated from Hansen tree cover at the 2,264 analysis points, never from labels (the
    # notebook recomputes them and checks they still equal these values). build_harvest_cycles_features.py
    # --calibrate-levels HANSEN.csv recalibrates them on a batch; features made with other levels are not comparable
    # with models trained on these.
    canopy_level: float = 0.5835294280425446     # closed canopy
    open_level: float = 0.4048285585388521       # open land

    # Abrupt start: >= min_open_years at open-land level among the open_lookback_values values ending at the last
    # open year, then closed canopy within fast_rise_years
    min_open_years: int = 3
    open_lookback_values: int = 8
    fast_rise_years: int = 5

    # Level calibration from Hansen Global Forest Change. These act only through --calibrate-levels (and the
    # notebook's check of the levels above); on their own they change nothing.
    closed_min_tree_cover: int = 90       # stable closed canopy: tree cover 2000 >= this, no gain, no loss up to
    closed_percentile: float = 5.0        # last_year in the pixel or its 3x3 neighbours; level = this percentile
    closed_first_year: int = 2000         # of their yearly NBR from closed_first_year to last_year
    open_max_tree_cover: int = 10         # open land: tree cover 2000 <= this, no gain; level = open_percentile
    open_percentile: float = 25.0         # of their NBR in open_years, capped at canopy_level - drop / 2
    open_years: tuple = (1999, 2000, 2001)
    calibration_min_pixels: int = 30      # fewer closed-canopy or open-land pixels in the batch: calibration refused

    # Seasonal features: median wet-season minus median dry-season value, per year, then the median over the
    # years of each window with both seasons observed (at least min_years of them).
    seasonal_dry_months: tuple = (4, 5, 6, 7, 8, 9)
    seasonal_indices: tuple = ("nbr", "ndvi")
    seasonal_windows: tuple = ((1985, 2020, 5), (2013, 2020, 3))   # (first year, last year, min_years)


S = TypeVar("S", PreprocessSettings, FeatureSettings)
INDICES = ("nbr", "ndvi")
AGGREGATIONS = ("median", "dry_median", "wet_median", "max")
SENSORS = ("LT04", "LT05", "LE07", "LC08")


def _coerce(value: Any, like: Any) -> Any:
    """Convert a parsed command-line value to the type of the default it replaces; raise ValueError if it does not fit."""
    if isinstance(like, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"expected true or false, got {value!r}")
    if isinstance(like, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"expected a JSON list such as [4, 5, 6], got {value!r}")
        return tuple(tuple(v) if isinstance(v, list) else v for v in value)
    if isinstance(like, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expected a number, got {value!r}")
        try:
            number = float(value)
        except OverflowError:
            raise ValueError(f"number too large: {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"expected a finite number, got {value!r}")
        if isinstance(like, int):
            if number != int(number):
                raise ValueError(f"expected an integer, got {value!r}")
            return int(value)
        return number
    if not isinstance(value, str):
        raise ValueError(f"expected text, got {value!r}")
    return value


def with_overrides(settings: S, overrides: Iterable[str] | None) -> S:
    """Apply `name=value` overrides (values in JSON, e.g. drop=0.15 or seasonal_dry_months=[5,6,7,8]) and check the
    result. Raises ValueError with a message naming the setting."""
    names = {f.name: getattr(settings, f.name) for f in fields(settings)}
    changes = {}
    for item in overrides or []:
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep or name not in names:
            raise ValueError(f"unknown setting in --set {item!r}; valid names: {', '.join(sorted(names))}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        try:
            changes[name] = _coerce(value, names[name])
        except ValueError as e:
            raise ValueError(f"--set {name}: {e}") from None
    new = replace(settings, **changes)
    try:
        validate(new)
    except TypeError:                  # a list of the wrong shape, e.g. [1985, 2020, 5] for [[1985, 2020, 5]]
        raise ValueError(f"invalid settings: a list value has the wrong shape ({', '.join(changes)})") from None
    return new


def _is_iso_date(text: str) -> bool:
    try:
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def validate(s: PreprocessSettings | FeatureSettings) -> None:
    """Basic range and shape checks, so that a bad value fails at once with its name, not deep in a run."""
    problems = []

    def need(ok: bool, name: str, what: str) -> None:
        if not ok:
            problems.append(f"{name} = {getattr(s, name)!r}: {what}")

    for f in fields(s):
        v = getattr(s, f.name)
        if isinstance(v, float) and not math.isfinite(v):
            problems.append(f"{f.name} must be a finite number")

    def months(v) -> bool:
        return all(isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= 12 for m in v)

    need(1984 <= s.first_year <= s.last_year, "first_year", "must be at least 1984 and not after last_year")
    if isinstance(s, PreprocessSettings):
        need(set(s.sensors) <= set(SENSORS) and len(s.sensors) > 0, "sensors", f"use some of {list(SENSORS)}")
        need(months(s.dry_months), "dry_months", "months must be integers 1-12")
        need(s.scenes_per_season >= 1, "scenes_per_season", "must be at least 1")
        need(0 <= s.max_failed_scene_share <= 1, "max_failed_scene_share", "must be between 0 and 1")
        need(_is_iso_date(s.slc_off_date), "slc_off_date", "must be a date written YYYY-MM-DD")
        for name in ["offset_years_oli", "offset_years_tm"]:
            v = getattr(s, name)
            need(len(v) == 2 and all(isinstance(y, int) for y in v) and v[0] <= v[1], name,
                 "must be [first year, last year]")
        need(s.offset_pair_max_days >= 0 and s.offset_min_pairs >= 1, "offset_min_pairs",
             "offset_pair_max_days must be >= 0 and offset_min_pairs >= 1")
        need(all(len(x) == 2 and x[0] in INDICES and x[1] in AGGREGATIONS for x in s.series), "series",
             f"pairs [index, aggregation] with index in {list(INDICES)} and aggregation in {list(AGGREGATIONS)}")
    else:
        need(s.min_years >= 3, "min_years", "must be at least 3")
        need(s.max_segments >= 1, "max_segments", "must be at least 1")
        need(s.drop > 0, "drop", "must be positive")
        for name in ["cycle_regrow_fraction", "regrow_time_fraction", "despike_threshold"]:
            need(0 < getattr(s, name) <= 1, name, "must be in (0, 1]")
        need(s.fall_window_years >= 1, "fall_window_years", "must be at least 1")
        need(s.min_regrow_years > 0, "min_regrow_years", "must be positive")
        need(s.open_level < s.canopy_level, "open_level", "must be below canopy_level")
        need(0 <= s.closed_percentile <= 100 and 0 <= s.open_percentile <= 100, "closed_percentile",
             "percentiles must be in 0-100")
        need(s.calibration_min_pixels >= 1, "calibration_min_pixels", "must be at least 1")
        need(months(s.seasonal_dry_months), "seasonal_dry_months", "months must be integers 1-12")
        need(set(s.seasonal_indices) <= set(INDICES) and len(set(s.seasonal_indices)) == len(s.seasonal_indices),
             "seasonal_indices", f"use some of {list(INDICES)}, each once")
        need(all(isinstance(w, tuple) and len(w) == 3 and all(isinstance(x, int) for x in w) and w[0] <= w[1]
                 and w[2] >= 1 for w in s.seasonal_windows)
             and len({w[:2] for w in s.seasonal_windows}) == len(s.seasonal_windows), "seasonal_windows",
             "a list of distinct [first year, last year, minimum years], e.g. [[1985, 2020, 5]]")
        need(s.min_open_years >= 1 and s.open_lookback_values >= s.min_open_years, "min_open_years",
             "must be at least 1 and at most open_lookback_values")
        need(s.fast_rise_years >= 0, "fast_rise_years", "must not be negative")
        need(len(s.open_years) > 0 and all(isinstance(y, int) and s.first_year <= y <= s.last_year for y in s.open_years),
             "open_years", "years between first_year and last_year")
        need(s.first_year <= s.closed_first_year <= s.last_year, "closed_first_year",
             "must lie between first_year and last_year")
    if problems:
        raise ValueError("invalid settings: " + "; ".join(problems))


def to_dict(settings: PreprocessSettings | FeatureSettings) -> dict:
    """Settings as a JSON-ready dictionary (tuples become lists)."""
    return json.loads(json.dumps(asdict(settings)))


def _sha256(path: Path) -> str:
    """SHA-256 of a file's content; for .gz files, of the decompressed content."""
    import gzip
    h = hashlib.sha256()
    with (gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")) as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    """Commit of the code that made an output, with "-dirty" if this folder's scripts had uncommitted changes."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=here, capture_output=True, text=True,
                                 check=True).stdout.strip()
        return commit + ("-dirty" if changed else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_record(out: Path, settings: PreprocessSettings | FeatureSettings, inputs: dict, extra: dict | None = None,
                     script: str | None = None, command: list[str] | None = None) -> Path:
    """Write the provenance of an output next to it: OUT.settings.json for a file, OUT/settings.json for a folder.
    Records the script and its command-line arguments, the settings, the input files with their SHA-256 (of the
    decompressed content for .gz files, so re-compressing identical data keeps the checksum), the software versions
    and the git commit."""
    import numpy as np
    import pandas as pd
    out = Path(out)
    record = dict(
        output=out.name,
        script=script or Path(sys.argv[0]).name,
        command_line=command,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        inputs={k: (dict(path=str(p), sha256=_sha256(Path(p))) if p else None) for k, p in inputs.items()},
        settings=to_dict(settings),
        **(extra or {}),
    )
    path = out / "settings.json" if out.is_dir() else out.with_name(out.name + ".settings.json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
