"""Metric-aware horizontal gradients on curvilinear SWOT swaths."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


def swath_xy_gradients(
    variable: xr.DataArray,
    longitude: xr.DataArray,
    latitude: xr.DataArray,
    *,
    line_dim: str = "num_lines",
    pixel_dim: str = "num_pixels",
) -> xr.Dataset:
    """Calculate eastward and northward gradients on a SWOT swath.

    A local azimuthal-equidistant projection converts longitude and latitude
    to metres. The native line/pixel derivatives are then transformed through
    the two-dimensional grid Jacobian, so both along-track and cross-track
    changes in x and y are retained.
    """

    if not isinstance(variable, xr.DataArray):
        raise TypeError("variable must be an xarray DataArray")
    if not isinstance(longitude, xr.DataArray) or not isinstance(latitude, xr.DataArray):
        raise TypeError("longitude and latitude must be xarray DataArrays")
    for name, array in {
        "variable": variable,
        "longitude": longitude,
        "latitude": latitude,
    }.items():
        missing = [dim for dim in (line_dim, pixel_dim) if dim not in array.dims]
        if missing:
            raise ValueError(f"{name} is missing dimension(s): {', '.join(missing)}")

    index_coords = {
        line_dim: np.arange(variable.sizes[line_dim], dtype=float),
        pixel_dim: np.arange(variable.sizes[pixel_dim], dtype=float),
    }
    variable = variable.transpose(line_dim, pixel_dim).assign_coords(index_coords)
    longitude = longitude.transpose(line_dim, pixel_dim).assign_coords(index_coords)
    latitude = latitude.transpose(line_dim, pixel_dim).assign_coords(index_coords)
    variable = variable.assign_coords(longitude=longitude, latitude=latitude)

    longitude_radians = np.deg2rad(longitude.values)
    reference_lon = float(
        np.rad2deg(
            np.arctan2(
                np.nanmean(np.sin(longitude_radians)),
                np.nanmean(np.cos(longitude_radians)),
            )
        )
    )
    reference_lat = float(latitude.mean(skipna=True))
    if not np.isfinite(reference_lon) or not np.isfinite(reference_lat):
        raise ValueError("Longitude and latitude contain no finite reference point.")

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={reference_lat} +lon_0={reference_lon} "
        "+datum=WGS84 +units=m"
    )
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), local_crs, always_xy=True
    )
    x_values, y_values = transformer.transform(longitude.values, latitude.values)

    x = xr.DataArray(
        x_values,
        dims=(line_dim, pixel_dim),
        coords=index_coords,
        name="x",
        attrs={"long_name": "local eastward distance", "units": "m"},
    )
    y = xr.DataArray(
        y_values,
        dims=(line_dim, pixel_dim),
        coords=index_coords,
        name="y",
        attrs={"long_name": "local northward distance", "units": "m"},
    )

    df_dline = variable.differentiate(line_dim)
    df_dpixel = variable.differentiate(pixel_dim)
    dx_dline = x.differentiate(line_dim)
    dx_dpixel = x.differentiate(pixel_dim)
    dy_dline = y.differentiate(line_dim)
    dy_dpixel = y.differentiate(pixel_dim)

    determinant = dx_dline * dy_dpixel - dx_dpixel * dy_dline
    determinant_scale = float(np.nanmedian(np.abs(determinant.values)))
    if not np.isfinite(determinant_scale) or determinant_scale == 0:
        raise ValueError("Could not calculate a valid grid Jacobian.")
    valid = np.isfinite(determinant) & (
        np.abs(determinant) > 1e-6 * determinant_scale
    )

    dvariable_dx = (
        (df_dline * dy_dpixel - df_dpixel * dy_dline) / determinant
    ).where(valid)
    dvariable_dy = (
        (dx_dline * df_dpixel - dx_dpixel * df_dline) / determinant
    ).where(valid)
    gradient_magnitude = np.hypot(dvariable_dx, dvariable_dy)

    variable_name = variable.name or "variable"
    variable_units = variable.attrs.get("units", "")
    gradient_units = f"{variable_units} m-1" if variable_units else "m-1"
    dvariable_dx = dvariable_dx.rename(f"d{variable_name}_dx")
    dvariable_dy = dvariable_dy.rename(f"d{variable_name}_dy")
    gradient_magnitude = gradient_magnitude.rename(
        f"{variable_name}_gradient_magnitude"
    )
    dvariable_dx.attrs.update(
        long_name=f"eastward gradient of {variable_name}", units=gradient_units
    )
    dvariable_dy.attrs.update(
        long_name=f"northward gradient of {variable_name}", units=gradient_units
    )
    gradient_magnitude.attrs.update(
        long_name=f"horizontal gradient magnitude of {variable_name}",
        units=gradient_units,
    )

    result = xr.Dataset(
        {
            "x": x,
            "y": y,
            dvariable_dx.name: dvariable_dx,
            dvariable_dy.name: dvariable_dy,
            gradient_magnitude.name: gradient_magnitude,
        }
    )
    result.attrs.update(
        reference_longitude=reference_lon,
        reference_latitude=reference_lat,
        projection="local azimuthal equidistant (WGS84)",
    )
    return result

