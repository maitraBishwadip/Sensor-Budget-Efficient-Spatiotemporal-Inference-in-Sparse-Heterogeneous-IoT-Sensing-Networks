"""Fig. 5 — Q1-grade IDW contour maps of PM2.5 (winter-mean).

Why winter mean rather than annual mean:
  spatial structure is maximally expressed during the winter inversion season
  in northern India; annual mean averages out the signal that motivates the
  paper's Act-II argument.

Per panel:
  - 220×220 IDW grid in EPSG:3857, masked to a convex-hull buffer so the
    contours never visualize pure extrapolation
  - CartoDB PositronNoLabels basemap underneath at 60% alpha
  - magma_r filled contours + white iso-lines with inline labels
  - station markers, city centroid (gold star)
  - scale bar, north arrow, India inset, panel label
  - per-panel value range stated in the title

The colourbar is shared so cross-city differences are legible.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point
from shapely.ops import unary_union

from _common import (load_all, station_summary, save, set_paper_style,
                     CITY_COLORS, CITY_CENTROIDS, COL_DOUBLE,
                     panel_label, add_scale_bar, add_north_arrow,
                     add_india_inset, pick_scale_bar_length)

set_paper_style()


def idw(xy_known, z_known, xy_query, power=2, eps=1e-9):
    d = np.linalg.norm(xy_query[:, None, :] - xy_known[None, :, :], axis=-1)
    w = 1.0 / (d ** power + eps)
    return (w * z_known[None, :]).sum(axis=1) / w.sum(axis=1)


def winter_station_means(df):
    """Per-station winter (Dec–Feb) mean PM2.5."""
    winter = df[df["month"].isin([12, 1, 2])]
    if len(winter) == 0:
        return station_summary(df)
    agg = (winter.groupby("Station Name")
           .agg(lat=("Latitude", "mean"),
                lon=("Longitude", "mean"),
                pm25_mean=("PM2.5", "mean"),
                n=("PM2.5", "size"))
           .reset_index())
    return agg


cities = load_all()
fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 3.6),
                         gridspec_kw={"wspace": 0.06})

# Global contour levels chosen on the union of winter means
all_z = np.concatenate(
    [winter_station_means(df)["pm25_mean"].values for df in cities.values()]
)
vmin = float(np.floor(np.percentile(all_z, 5) / 10) * 10)
vmax = float(np.ceil(np.percentile(all_z, 95) / 10) * 10)
levels = np.linspace(vmin, vmax, 11)

panel_letters = ["a", "b", "c"]
cf = None

for ax, letter, (city, df) in zip(axes, panel_letters, cities.items()):
    stations = winter_station_means(df)
    gdf = gpd.GeoDataFrame(
        stations,
        geometry=[Point(xy) for xy in zip(stations.lon, stations.lat)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    xy = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    z = stations["pm25_mean"].values

    # Extent: convex-hull bbox + 30 % pad
    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad = max((xmax - xmin), (ymax - ymin)) * 0.30 + 3000
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    try:
        cx.add_basemap(ax,
                       source=cx.providers.CartoDB.PositronNoLabels,
                       attribution_size=5)
    except Exception as e:
        print(f"  basemap fetch failed for {city}: {e}")

    nx, ny = 220, 220
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx),
                         np.linspace(ymin, ymax, ny))
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    zz = idw(xy, z, grid_xy, power=2).reshape(ny, nx)

    # Mask to a buffered convex hull of stations (avoid extrapolation art).
    hull = unary_union([Point(p) for p in xy]).convex_hull
    buffer_m = max((xmax - xmin), (ymax - ymin)) * 0.06 + 3000
    hull_buf = hull.buffer(buffer_m)
    from shapely.vectorized import contains
    inside = contains(hull_buf, xx, yy)
    zz_masked = np.where(inside, zz, np.nan)

    cf = ax.contourf(xx, yy, zz_masked, levels=levels, cmap="magma_r",
                     alpha=0.65, extend="both")
    # Only label every-other contour to reduce clutter
    cs = ax.contour(xx, yy, zz_masked, levels=levels, colors="white",
                    linewidths=0.35, alpha=0.85)
    ax.clabel(cs, levels=levels[::2], inline=True,
              fontsize=6, fmt="%d")

    # Stations
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=58,
               facecolor="white", edgecolor="black",
               linewidth=1.0, zorder=4)
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=14,
               c=z, cmap="magma_r",
               vmin=vmin, vmax=vmax,
               edgecolor="black", linewidth=0.3, zorder=5)

    # Centroid star
    lat, lon = CITY_CENTROIDS[city]
    cgdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
    ax.scatter(cgdf.geometry.x, cgdf.geometry.y, marker="*",
               s=240, c="gold", edgecolor="black", linewidth=0.8, zorder=6)

    length = pick_scale_bar_length(ax.get_xlim()[1] - ax.get_xlim()[0])
    add_scale_bar(ax, length_m=length)
    add_north_arrow(ax)
    add_india_inset(ax, city)

    ax.set_title(f"{city}  ·  winter μ: {z.min():.0f}–{z.max():.0f}",
                 color=CITY_COLORS[city])
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#444"); s.set_linewidth(0.6)
    panel_label(ax, letter)

cbar = fig.colorbar(cf, ax=axes, orientation="horizontal",
                    fraction=0.045, pad=0.04, shrink=0.65, aspect=40)
cbar.set_label("Winter-mean PM$_{2.5}$  ($\\mu$g m$^{-3}$)  ·  IDW (p=2)")
cbar.outline.set_visible(False)

save(fig, "02_pm25_contour_maps")
plt.close(fig)
print("done.")
