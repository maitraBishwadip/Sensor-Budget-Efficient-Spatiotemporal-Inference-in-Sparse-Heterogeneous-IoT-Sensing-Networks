"""Fig. 3 — Q1-grade station maps for Delhi, Kolkata, Guwahati.

Three-panel city map with:
  - station markers sized by record count, coloured by mean PM2.5
  - CartoDB Positron basemap (light, attribution-respected)
  - scale bar (metric), north arrow, India inset
  - panel labels (a)/(b)/(c), N = |V| annotation per panel
  - shared viridis colorbar so cross-panel comparison is valid
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

cities = load_all()
fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 3.4),
                         gridspec_kw={"wspace": 0.06})

# Cross-panel comparison → shared colour range from global percentiles.
all_means = np.concatenate(
    [station_summary(df)["pm25_mean"].values for df in cities.values()]
)
vmin = float(np.floor(np.percentile(all_means, 5) / 10) * 10)
vmax = float(np.ceil(np.percentile(all_means, 95) / 10) * 10)

panel_letters = ["a", "b", "c"]
sc = None

for ax, letter, (city, df) in zip(axes, panel_letters, cities.items()):
    stations = station_summary(df)
    gdf = gpd.GeoDataFrame(
        stations,
        geometry=[Point(xy) for xy in zip(stations.lon, stations.lat)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    sizes = 28 + 220 * (stations["n"] / stations["n"].max())

    # Pad the extent before placing scatter so basemap covers it.
    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad = max((xmax - xmin), (ymax - ymin)) * 0.22 + 2500
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       attribution_size=5)
    except Exception as e:
        print(f"  basemap fetch failed for {city}: {e}")

    sc = ax.scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=stations["pm25_mean"], s=sizes,
        cmap="viridis", vmin=vmin, vmax=vmax,
        edgecolor="black", linewidth=0.6, alpha=0.92, zorder=3,
    )

    # Scale bar, north arrow, India inset
    length = pick_scale_bar_length(ax.get_xlim()[1] - ax.get_xlim()[0])
    add_scale_bar(ax, length_m=length)
    add_north_arrow(ax)
    add_india_inset(ax, city)

    ax.set_title(f"{city}  ·  |V| = {len(stations)}",
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

# Size legend (record count) at figure level so it can't collide with the
# India inset placed inside panel (c).
size_legend_vals = [0.25, 0.6, 1.0]
ref_n = max(station_summary(df)["n"].max() for df in cities.values())
legend_handles = [
    plt.scatter([], [], s=28 + 220 * v, c="lightgrey",
                edgecolor="black", linewidth=0.6,
                label=f"{int(ref_n * v):,}")
    for v in size_legend_vals
]
fig.legend(handles=legend_handles, title="Records / station",
           loc="upper right", bbox_to_anchor=(0.995, 0.97),
           labelspacing=1.0, borderpad=0.5, fontsize=7,
           title_fontsize=7, frameon=True,
           facecolor="white", edgecolor="#888")

save(fig, "01_station_maps")
plt.close(fig)
print("done.")
