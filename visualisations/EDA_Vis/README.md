# EDA Visualizations — Pre-Results Figures for the Paper

Publication-grade exploratory data analysis figures for the Graph-DANN
PM2.5 transfer-learning paper. These figures are intended to land in the
**Introduction**, **Data & Setup**, and **Discussion** sections — they
establish the data-scarcity story, the cross-city covariate shift, and the
spatial structure that motivates the GNN-TL contribution, all **before**
any model-result figures.

## Figures produced

| # | Script | Output | Paper section |
|---|---|---|---|
| 1 | `01_station_maps_geo.py` | `figures/01_station_maps.pdf` | §3 Data — Fig. 3 |
| 2 | `02_pm25_contour_maps.py` | `figures/02_pm25_contour_maps.pdf` | **§3 / §10 — contour figure** |
| 3 | `03_distribution_violins.py` | `figures/03_distribution_violins.pdf` | §3 / §5 — covariate shift |
| 4 | `04_seasonal_diurnal_heatmaps.py` | `figures/04_seasonal_diurnal_heatmaps.pdf` | §3 — temporal cycles |
| 5 | `05_wind_rose.py` | `figures/05_wind_rose.pdf` | §3 — meteorology |
| 6 | `06_correlation_matrix.py` | `figures/06_correlation_matrix.pdf` | App. A — features |
| 7 | `07_temporal_overview.py` | `figures/07_temporal_overview.pdf` | §3 — data span |
| 8 | `08_knn_graph_visualization.py` | `figures/08_knn_graph_visualization.pdf` | §5 — motivates GNN backbone |
| 9 | `09_missingness_pattern.py` | `figures/09_missingness_pattern.pdf` | App. A — data quality |

Each script writes a **vector PDF** (for the manuscript) and a **300-dpi PNG**
(for slides / preview) into `figures/`. All figures use a shared paper style
(`_common.set_paper_style`) so they look consistent.

## Requirements

```bash
pip install geopandas contextily windrose pyproj shapely scipy seaborn scikit-learn
```

The Python in this environment already has these installed (geopandas 1.1.3,
contextily 1.7.0). If you copy the scripts elsewhere, install the packages
above.

## Running

Run a single script:
```bash
python visualisations/EDA_Vis/02_pm25_contour_maps.py
```

Or run everything:
```bash
python visualisations/EDA_Vis/run_all.py
```

## Notes on the contour figure (`02_pm25_contour_maps.py`)

This is the **showpiece EDA visual** and the bridge into the paper's Act II
argument:

- IDW interpolation (power=2) of annual-mean PM2.5 over a 220×220 grid in
  EPSG:3857 (Web Mercator) per city.
- Filled `magma_r` contours at 12 levels, with white iso-lines and
  inline labels.
- Stations rendered as outlined white-fill circles with magma-coloured
  inner dots; city centroid as a gold star.
- CartoDB *PositronNoLabels* basemap under the contours at 55 % alpha so
  the spatial structure dominates.

The point of this figure in the paper is to make it **visually obvious**
that PM2.5 has spatial structure within each city. A station-independent
LSTM cannot exploit that structure — which is precisely what Act II of
the paper argues.

## Style conventions

- Vector PDFs for the manuscript; 300-dpi PNGs for previews.
- `pdf.fonttype = 42` so text is editable (Type 42, embedded TrueType).
- City colour palette is fixed in `_common.CITY_COLORS`
  (Delhi red, Kolkata blue, Guwahati green).
- Colour ramps:
  - PM2.5 scalar fields → `viridis` or `magma_r` (perceptually uniform).
  - Correlation matrices → `RdBu_r` (diverging).
  - Heatmaps of density → `rocket_r` / `YlGnBu`.

## Known environment gotcha

The host has a system-set `PROJ_LIB` pointing at PostgreSQL/PostGIS, which is
incompatible with the pyproj version used here. `_common.py` clears those
env vars at import time and pins them to pyproj's bundled database. If you
ever see *"DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a number >= 6 is
expected"*, that is the same problem — just re-import `_common` first.
