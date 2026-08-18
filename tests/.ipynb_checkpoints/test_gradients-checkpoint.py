import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from swotgem import swath_xy_gradients


def test_gradient_of_plane_in_local_xy():
    x_values, y_values = np.meshgrid(
        np.linspace(-20_000.0, 20_000.0, 6),
        np.linspace(-30_000.0, 30_000.0, 7),
    )
    local = CRS.from_proj4(
        "+proj=aeqd +lat_0=-55 +lon_0=20 +datum=WGS84 +units=m"
    )
    inverse = Transformer.from_crs(local, CRS.from_epsg(4326), always_xy=True)
    longitude, latitude = inverse.transform(x_values, y_values)
    dims = ("num_lines", "num_pixels")
    coords = {"num_lines": np.arange(7), "num_pixels": np.arange(6)}
    variable = xr.DataArray(
        2.0 * x_values + 3.0 * y_values,
        dims=dims,
        coords=coords,
        name="plane",
        attrs={"units": "test"},
    )
    longitude = xr.DataArray(longitude, dims=dims, coords=coords)
    latitude = xr.DataArray(latitude, dims=dims, coords=coords)

    gradients = swath_xy_gradients(variable, longitude, latitude)
    np.testing.assert_allclose(gradients["dplane_dx"], 2.0, rtol=2e-3)
    np.testing.assert_allclose(gradients["dplane_dy"], 3.0, rtol=2e-3)
    np.testing.assert_allclose(
        gradients["plane_gradient_magnitude"], np.hypot(2.0, 3.0), rtol=2e-3
    )

