from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from swotgem import crop_swot_dataset, reconstruct_swath, swath_xy_gradients
from swotgem.plotting import plot_gradient_fields, plot_swath_fields

GEM_FILE = Path("GEM_smoothed_2000dbar_for_SWOT.nc")
SWOT_FILE = Path("SWOT_L3_LR_SSH_Expert_048_091_example.nc")
PRESSURE_INDEX = 50
LATITUDE_BOUNDS = (-70.0, -45.0)


with xr.open_dataset(GEM_FILE) as gem, xr.open_dataset(SWOT_FILE) as raw_swath:
    swath = crop_swot_dataset(raw_swath, lat_bounds=LATITUDE_BOUNDS)
    if swath is None:
        raise ValueError("This pass does not intersect the requested latitude bounds.")

    fields = reconstruct_swath(gem, swath, pressure_index=PRESSURE_INDEX).load()
    pressure = float(gem.pressure.isel(pressure=PRESSURE_INDEX))

fig, axes = plot_swath_fields(fields, pressure=pressure)
fig.suptitle("SWOTGEM individual swath")

gradients = swath_xy_gradients(
    fields["sigma0"], fields["longitude"], fields["latitude"]
)
fig_gradient, axes_gradient = plot_gradient_fields(
    gradients,
    fields["longitude"],
    fields["latitude"],
    variable_name="sigma0",
)
fig_gradient.suptitle(r"SWOTGEM $\sigma_0$ gradients")
plt.show()

