"""Fig. 10 — Q1-grade k-NN station graphs overlaid on city maps.

Per panel:
  - k-nearest-neighbour graph in haversine distance (k = 4)
  - edges drawn with alpha encoding distance (closer ⇒ more opaque)
  - nodes coloured by mean PM2.5 (shared viridis range, see Fig. 3)
  - CartoDB Positron basemap (light, low-clutter)
  - scale bar, north arrow, India inset, panel label
  - |V|, |E|, mean edge length and graph density stated in the title

Edge density and mean-edge-length both differ by ≈10× across cities,
which is exactly the cross-graph heterogeneity the Graph-DANN model must
handle.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point

from _common import (load_all, station_summary, save, set_paper_style,
                     CITY_COLORS, COL_DOUBLE, panel_label,
                     add_scale_bar, add_north_arrow, add_india_inset,
                     pick_scale_bar_length)

set_paper_style()

K = 4  # neighbours per node


def haversine_pairs(lat, lon):
    R = 6371.0
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :])
         * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


cities = load_all()

# Shared viridis range from global mean PM2.5
all_means = np.concatenate(
    [station_summary(df)["pm25_mean"].values for df in cities.values()]
)
vmin = float(np.floor(np.percentile(all_means, 5) / 10) * 10)
vmax = float(np.ceil(np.percentile(all_means, 95) / 10) * 10)

fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 3.6),
                         gridspec_kw={"wspace": 0.06})

panel_letters = ["a", "b", "c"]
sc = None
for ax, letter, (city, df) in zip(axes, panel_letters, cities.items()):
    stations = station_summary(df)
    n = len(stations)
    k_eff = min(K, n - 1)

    gdf = gpd.GeoDataFrame(
        stations,
        geometry=[Point(xy) for xy in zip(stations.lon, stations.lat)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    D = haversine_pairs(stations.lat.values, stations.lon.values)
    np.fill_diagonal(D, np.inf)

    edges = {}
    for i in range(n):
        nbrs = np.argsort(D[i])[:k_eff]
        for j in nbrs:
            a, b = sorted((int(i), int(j)))
            edges[(a, b)] = D[i, j]

    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad = max((xmax - xmin), (ymax - ymin)) * 0.18 + 2500
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       attribution_size=5)
    except Exception as e:
        print(f"  basemap fetch failed for {city}: {e}")

    if edges:
        dists = np.array(list(edges.values()))
        dmin, dmax = dists.min(), dists.max()
        for (i, j), d in edges.items():
            p1 = (gdf.geometry.x.iloc[i], gdf.geometry.y.iloc[i])
            p2 = (gdf.geometry.x.iloc[j], gdf.geometry.y.iloc[j])
            alpha = 0.85 - 0.55 * (d - dmin) / max(dmax - dmin, 1e-6)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=CITY_COLORS[city], lw=1.2, alpha=alpha, zorder=2)

    sc = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                    c=stations["pm25_mean"], s=90,
                    cmap="viridis", vmin=vmin, vmax=vmax,
                    edgecolor="black", linewidth=0.8, zorder=3)

    length = pick_scale_bar_length(ax.get_xlim()[1] - ax.get_xlim()[0])
    add_scale_bar(ax, length_m=length)
    add_north_arrow(ax)
    add_india_inset(ax, city)

    mean_d = float(np.mean(list(edges.values()))) if edges else 0.0
    density = (2 * len(edges)) / (n * (n - 1)) if n > 1 else 0.0
    ax.set_title(f"{city}  ·  |V|={n}, |E|={len(edges)}, "
                 f"d̄={mean_d:.1f} km, ρ={density:.2f}",
                 color=CITY_COLORS[city])
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#444"); s.set_linewidth(0.6)
    panel_label(ax, letter)

cbar = fig.colorbar(sc, ax=axes, orientation="horizontal",
                    fraction=0.045, pad=0.04, shrink=0.65, aspect=40)
cbar.set_label("Mean PM$_{2.5}$  ($\\mu$g m$^{-3}$)")
cbar.outline.set_visible(False)

save(fig, "08_knn_graph_visualization")
plt.close(fig)
print("done.")
