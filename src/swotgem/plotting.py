"""Optional Cartopy plotting helpers for reconstructed SWOT swaths."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


def swath_extent(
    longitude: xr.DataArray,
    latitude: xr.DataArray,
    *,
    margin_fraction: float = 0.08,
    minimum_margin: float = 1.0,
) -> list[float]:
    """Return a Plate Carree extent around all finite swath coordinates."""

    longitude_values = ((longitude.values + 180.0) % 360.0) - 180.0
    latitude_values = latitude.values
    valid = np.isfinite(longitude_values) & np.isfinite(latitude_values)
    if not valid.any():
        raise ValueError("No finite longitude/latitude pairs are available.")

    lon_min, lon_max = np.nanmin(longitude_values[valid]), np.nanmax(
        longitude_values[valid]
    )
    lat_min, lat_max = np.nanmin(latitude_values[valid]), np.nanmax(
        latitude_values[valid]
    )
    lon_margin = max(minimum_margin, margin_fraction * (lon_max - lon_min))
    lat_margin = max(minimum_margin, margin_fraction * (lat_max - lat_min))
    return [
        lon_min - lon_margin,
        lon_max + lon_margin,
        max(-90.0, lat_min - lat_margin),
        min(90.0, lat_max + lat_margin),
    ]


def plot_swath_fields(
    fields: xr.Dataset,
    *,
    variables: Sequence[str] = ("CT", "SA", "sigma0"),
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    pressure: float | None = None,
    extent: Sequence[float] | None = None,
    figsize: tuple[float, float] = (18.0, 7.0),
):
    """Plot reconstructed CT, SA, and sigma0 fields; return ``(fig, axes)``."""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    metadata = {
        "CT": (cmocean.cm.thermal, "Conservative Temperature", "°C"),
        "SA": (cmocean.cm.haline, "Absolute Salinity", "g kg$^{-1}$"),
        "sigma0": (cmocean.cm.dense, r"$\sigma_0$", "kg m$^{-3}$"),
    }
    missing = [name for name in variables if name not in fields]
    if missing:
        raise KeyError(f"Missing field(s): {', '.join(missing)}")

    data_crs = ccrs.PlateCarree()
    longitude = fields[longitude_name]
    latitude = fields[latitude_name]
    extent = list(extent) if extent is not None else swath_extent(longitude, latitude)
    fig, axes = plt.subplots(
        1,
        len(variables),
        figsize=figsize,
        subplot_kw={"projection": data_crs},
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, variable_name in zip(axes, variables):
        cmap, title, units = metadata.get(variable_name, ("viridis", variable_name, ""))
        axis.set_extent(extent, crs=data_crs)
        axis.set_facecolor("0.85")
        mesh = axis.pcolormesh(
            longitude,
            latitude,
            fields[variable_name],
            transform=data_crs,
            shading="auto",
            cmap=cmap,
            zorder=2,
        )
        axis.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor="0.65",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        axis.coastlines(resolution="50m", linewidth=0.8, zorder=6)
        gridlines = axis.gridlines(
            crs=data_crs,
            draw_labels=True,
            linewidth=0.6,
            color="black",
            alpha=0.5,
            linestyle="--",
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xformatter = LONGITUDE_FORMATTER
        gridlines.yformatter = LATITUDE_FORMATTER
        gridlines.xlocator = mticker.MaxNLocator(5)
        gridlines.ylocator = mticker.MaxNLocator(5)
        colorbar = fig.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.08)
        colorbar.set_label(units)
        pressure_text = "" if pressure is None else f"\n{pressure:.0f} dbar"
        axis.set_title(f"{title}{pressure_text}")
    return fig, axes


def plot_gradient_fields(
    gradients: xr.Dataset,
    longitude: xr.DataArray,
    latitude: xr.DataArray,
    *,
    variable_name: str,
    distance: float = 100_000.0,
    distance_label: str = "100 km",
    percentile: float = 98.0,
    extent: Sequence[float] | None = None,
    figsize: tuple[float, float] = (18.0, 7.0),
):
    """Plot eastward and northward gradients scaled to a chosen distance."""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean
    import matplotlib.pyplot as plt

    names = [f"d{variable_name}_dx", f"d{variable_name}_dy"]
    missing = [name for name in names if name not in gradients]
    if missing:
        raise KeyError(f"Missing gradient field(s): {', '.join(missing)}")
    scaled = [gradients[name] * distance for name in names]
    values = np.concatenate([field.values.ravel() for field in scaled])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite gradient values are available to plot.")
    limit = float(np.nanpercentile(np.abs(values), percentile))

    data_crs = ccrs.PlateCarree()
    extent = list(extent) if extent is not None else swath_extent(longitude, latitude)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        subplot_kw={"projection": data_crs},
        constrained_layout=True,
    )
    for axis, field, direction in zip(axes, scaled, ("x", "y")):
        axis.set_extent(extent, crs=data_crs)
        mesh = axis.pcolormesh(
            longitude,
            latitude,
            field,
            transform=data_crs,
            shading="auto",
            cmap=cmocean.cm.balance,
            vmin=-limit,
            vmax=limit,
        )
        axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.65", zorder=5)
        axis.coastlines(resolution="50m", linewidth=0.8, zorder=6)
        axis.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        fig.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.08).set_label(
            f"{variable_name} change per {distance_label}"
        )
        axis.set_title(rf"$\partial {variable_name}/\partial {direction}$")
    return fig, axes

