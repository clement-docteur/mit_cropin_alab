# MIT Analytics Lab × Cropin: telling plantations from natural forest in southern Brazil

Under the EU Deforestation Regulation (EUDR), Cropin's forest layer raises clearing alerts.
Many of them are routine **plantation harvests** (eucalyptus and pine), not deforestation. This
project builds an interpretable, 10 m plantation-probability layer for Paraná, Santa Catarina and
Rio Grande do Sul, to cut those false alarms **without hiding real deforestation**.

Team: Meera Mehta, Julia Hansen, Clement Docteur, Nathan Chu · Mentor: Raphael Raux · Sponsor: Cropin

> ⚠️ This repository is **public**. Cropin data (QC polygons and everything exported from them)
> and the large MapBiomas validation file are **never committed**. See `data/README.md`.

## Repository map
| Path | What |
|---|---|
| `data_exploration/` | Teammates' Earth Engine exploration scripts, QC map and first notebook |
| `gee/` | Earth Engine scripts behind the QC error analysis, plus `gee/cropin/` (Cropin's own forest layer and SAG code) |
| `data/` | Data folders (local only) and the data dictionary `data/README.md` |
| `src/` | Python helpers and the MapBiomas reference-subset builder |
| `notebooks/` | Reproducible analysis notebooks |
| `figures/` | Charts written by the notebooks |
| `docs/` | Written summaries |

<!-- Teammates: add your own sections below or above this line. -->

## QC error analysis and SAG validation (Meera)
**What it answers.** Why do Cropin's alerts include plantations, what separates them from real
deforestation, and does Cropin's SAG plantation signal work on labelled cases?

- **One-page summary for the team:** [`docs/eda_findings.md`](docs/eda_findings.md)
- **Notebook 01:** [`notebooks/01_qc_error_analysis.ipynb`](notebooks/01_qc_error_analysis.ipynb). The MapBiomas benchmark, six single-feature findings and a combined model. Statistics: Mann–Whitney / Fisher, bootstrap CIs, Holm correction, spatially blocked CV, nested threshold.
- **Notebook 02:** [`notebooks/02_cropin_layers_and_SAG.ipynb`](notebooks/02_cropin_layers_and_SAG.ipynb). Cropin's forest layer and MapBiomas 10 m benchmarks, the first SAG validation, Meta 1 m height, and SAG added to the model.
- **Earth Engine:** `gee/02_qc_error_analysis_export.js` and `gee/03_cropin_layers_and_SAG_qc_export.js` produce the CSVs. Cropin's scripts are explained in `gee/cropin/README.md`.

### Rerun
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the data files in data/ as described in data/README.md (ask Meera)
.venv/bin/python src/build_mapbiomas_ref_south.py          # optional: rebuild and check the reference subset
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_qc_error_analysis.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_cropin_layers_and_SAG.ipynb
```
Seeds are fixed, so the notebooks reproduce exactly. Each takes under a minute.
