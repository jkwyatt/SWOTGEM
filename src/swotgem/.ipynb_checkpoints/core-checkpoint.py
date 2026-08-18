"""Core SWOT/SatGEM data transformations.

Functions in this module do not read files, plot, or depend on notebook globals.
That makes them straightforward to test and reuse in scripts and notebooks.
"""

from __future__ import annotations

from collections.abc import Sequence

import gsw
import numpy as np
import xarray as xr


def wrap_longitude(longitude: xr.DataArray) -> xr.DataArray:
    """Wrap longitude values to the half-open interval [-180, 180)."""

    wrapped = (longitude + 180.0) % 360.0 - 180.0
    wrapped.attrs = longitude.attrs.copy()
    return wrapped


def crop_swot_dataset(
    dataset: xr.Dataset,
    lat_bounds: Sequence[float] = (-70.0, -60.0),
    lon_bounds: Sequence[float] = (-180.0, 180.0),
    *,
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    line_dim: str = "num_lines",
    pixel_dim: str = "num_pixels",
) -> xr.Dataset | None:
    """Crop a two-dimensional SWOT swath by latitude and longitude.

    Longitude is first wrapped to ``[-180, 180)``. ``None`` is returned when
    the requested box has no line or pixel samples, which is convenient while
    looping over all passes in a cycle.
    """

    if latitude_name not in dataset or longitude_name not in dataset:
        raise KeyError(
            f"Dataset must contain {latitude_name!r} and {longitude_name!r}."
        )

    lat_min, lat_max = map(float, lat_bounds)
    lon_min, lon_max = map(float, lon_bounds)
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("Bounds must be supplied in increasing order.")

    longitude = wrap_longitude(dataset[longitude_name])
    dataset = dataset.assign_coords({longitude_name: longitude})
    mask = (
        (dataset[latitude_name] >= lat_min)
        & (dataset[latitude_name] <= lat_max)
        & (dataset[longitude_name] >= lon_min)
        & (dataset[longitude_name] <= lon_max)
    )
    cropped = dataset.where(mask, drop=True)

    if cropped.sizes.get(line_dim, 0) == 0:
        return None
    if pixel_dim in dataset.dims and cropped.sizes.get(pixel_dim, 0) == 0:
        return None
    return cropped


def swot_ssh(
    dataset: xr.Dataset,
    *,
    mdt_name: str = "mdt",
    anomaly_name: str = "ssha_filtered",
    clip: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Calculate absolute SWOT SSH as mean dynamic topography plus anomaly."""

    missing = [name for name in (mdt_name, anomaly_name) if name not in dataset]
    if missing:
        raise KeyError(f"Missing required SWOT variable(s): {', '.join(missing)}")

    ssh = (dataset[mdt_name] + dataset[anomaly_name]).rename("ssh")
    if clip is not None:
        ssh = ssh.clip(min=float(clip[0]), max=float(clip[1]))
    ssh.attrs.update(
        long_name="sea surface height used to query SatGEM",
        units=dataset[mdt_name].attrs.get("units", "m"),
    )
    return ssh


def select_gem_pressure(
    gem: xr.Dataset,
    *,
    pressure: float | None = None,
    pressure_index: int | None = None,
    pressure_name: str = "pressure",
    method: str = "nearest",
) -> xr.Dataset:
    """Select one pressure level from a SatGEM dataset.

    Specify exactly one of ``pressure`` (coordinate value) or
    ``pressure_index`` (integer position).
    """

    if (pressure is None) == (pressure_index is None):
        raise ValueError("Specify exactly one of pressure or pressure_index.")
    if pressure_name not in gem.dims and pressure_name not in gem.coords:
        raise KeyError(f"GEM dataset has no {pressure_name!r} coordinate.")

    if pressure_index is not None:
        return gem.isel({pressure_name: int(pressure_index)})
    return gem.sel({pressure_name: float(pressure)}, method=method)


def reconstruct_swath(
    gem: xr.Dataset,
    swot: xr.Dataset,
    *,
    pressure: float | None = None,
    pressure_index: int | None = None,
    pressure_name: str = "pressure",
    gem_temperature_name: str = "temp",
    gem_salinity_name: str = "sal",
    gem_ssh_name: str = "ssh",
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    mdt_name: str = "mdt",
    anomaly_name: str = "ssha_filtered",
    clip_ssh_to_gem: bool = True,
    interpolation_method: str = "linear",
) -> xr.Dataset:
    """Interpolate SatGEM temperature and salinity onto one SWOT swath.

    The returned dataset contains ``CT``, ``SA``, ``sigma0``, and the original
    (unclipped) ``ssh``. SSH values outside the lookup range are clipped only
    for interpolation and missing SWOT SSH remains masked in every product.
    """

    needed_gem = {gem_temperature_name, gem_salinity_name, gem_ssh_name}
    missing_gem = sorted(needed_gem.difference(gem.variables))
    if missing_gem:
        raise KeyError(f"Missing required GEM variable(s): {', '.join(missing_gem)}")
    if longitude_name not in swot or latitude_name not in swot:
        raise KeyError(
            f"SWOT dataset must contain {longitude_name!r} and {latitude_name!r}."
        )

    if pressure_name in gem.dims:

        # Keep every pressure level when neither selector is supplied
        if pressure is None and pressure_index is None:
    
            gem_at_pressure = gem
    
        # Select one level when pressure or pressure_index is supplied
        else:
    
            gem_at_pressure = select_gem_pressure(
                gem,
                pressure=pressure,
                pressure_index=pressure_index,
                pressure_name=pressure_name,
            )
    
    else:
    
        if pressure is not None or pressure_index is not None:
            raise ValueError(
                "GEM is already pressure-selected; "
                "omit pressure arguments."
            )
    
        gem_at_pressure = gem

    ssh = swot_ssh(swot, mdt_name=mdt_name, anomaly_name=anomaly_name)
    ssh_for_lookup = ssh
    if clip_ssh_to_gem:
        ssh_min = float(gem_at_pressure[gem_ssh_name].min(skipna=True).compute())
        ssh_max = float(gem_at_pressure[gem_ssh_name].max(skipna=True).compute())
        ssh_for_lookup = ssh.clip(min=ssh_min, max=ssh_max)

    longitude = wrap_longitude(swot[longitude_name])
    latitude = swot[latitude_name]
    lookup = gem_at_pressure[[gem_temperature_name, gem_salinity_name]].interp(
        {longitude_name: longitude, gem_ssh_name: ssh_for_lookup},
        method=interpolation_method,
    )
    lookup = lookup.drop_vars(
    gem_ssh_name,
    errors="ignore",
    )
    lookup = lookup.where(np.isfinite(ssh))

    conservative_temperature = lookup[gem_temperature_name].rename("CT")
    absolute_salinity = lookup[gem_salinity_name].rename("SA")
    sigma0 = xr.apply_ufunc(
        gsw.sigma0,
        absolute_salinity,
        conservative_temperature,
        dask="parallelized",
        output_dtypes=[float],
    ).rename("sigma0")

    conservative_temperature.attrs.update(
        long_name="Conservative Temperature",
        standard_name="sea_water_conservative_temperature",
        units="degree_Celsius",
    )
    absolute_salinity.attrs.update(
        long_name="Absolute Salinity",
        standard_name="sea_water_absolute_salinity",
        units="g kg-1",
    )
    sigma0.attrs.update(
        long_name="potential density anomaly referenced to 0 dbar",
        units="kg m-3",
    )

    result = xr.Dataset(
        {
            "CT": conservative_temperature,
            "SA": absolute_salinity,
            "sigma0": sigma0,
            "ssh": ssh,
        }
    )
    return result.assign_coords(
        {longitude_name: longitude, latitude_name: latitude}
    )

