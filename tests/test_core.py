import numpy as np
import xarray as xr

from swotgem import crop_swot_dataset, reconstruct_swath, swot_ssh, wrap_longitude


def sample_swot():
    line = np.arange(3)
    pixel = np.arange(4)
    longitude, latitude = np.meshgrid(np.linspace(10.0, 13.0, 4), np.linspace(-60.0, -58.0, 3))
    return xr.Dataset(
        {
            "mdt": (("num_lines", "num_pixels"), np.full((3, 4), -0.5)),
            "ssha_filtered": (("num_lines", "num_pixels"), np.full((3, 4), 0.1)),
        },
        coords={
            "num_lines": line,
            "num_pixels": pixel,
            "longitude": (("num_lines", "num_pixels"), longitude),
            "latitude": (("num_lines", "num_pixels"), latitude),
        },
    )


def test_wrap_and_crop():
    longitude = xr.DataArray([0.0, 181.0, 359.0])
    np.testing.assert_allclose(wrap_longitude(longitude), [0.0, -179.0, -1.0])
    cropped = crop_swot_dataset(sample_swot(), lat_bounds=(-59.1, -57.0))
    assert cropped is not None
    assert cropped.sizes["num_lines"] == 2


def test_swot_ssh():
    ssh = swot_ssh(sample_swot())
    np.testing.assert_allclose(ssh, -0.4)
    assert ssh.name == "ssh"


def test_reconstruct_swath():
    gem = xr.Dataset(
        {
            "temp": (
                ("pressure", "longitude", "ssh"),
                np.array([[[1.0, 2.0], [1.0, 2.0]]]),
            ),
            "sal": (
                ("pressure", "longitude", "ssh"),
                np.array([[[34.0, 35.0], [34.0, 35.0]]]),
            ),
        },
        coords={"pressure": [500.0], "longitude": [10.0, 13.0], "ssh": [-0.5, -0.3]},
    )
    result = reconstruct_swath(gem, sample_swot(), pressure_index=0)
    assert set(("CT", "SA", "sigma0", "ssh")).issubset(result.data_vars)
    assert result["CT"].dims == ("num_lines", "num_pixels")
    np.testing.assert_allclose(result["CT"], 1.5)
    np.testing.assert_allclose(result["SA"], 34.5)
    assert np.isfinite(result["sigma0"]).all()
