"""Tools for reconstructing hydrography from SWOT SSH and SatGEM."""

from .batch import iter_reconstructed_swaths
from .core import (
    crop_swot_dataset,
    reconstruct_swath,
    select_gem_pressure,
    swot_ssh,
    wrap_longitude,
)
from .download import DEFAULT_BASE_URL, download_cycle, list_cycle_files
from .gradients import swath_xy_gradients

__all__ = [
    "DEFAULT_BASE_URL",
    "crop_swot_dataset",
    "download_cycle",
    "iter_reconstructed_swaths",
    "list_cycle_files",
    "reconstruct_swath",
    "select_gem_pressure",
    "swath_ssh",
    "swath_xy_gradients",
    "swot_ssh",
    "wrap_longitude",
]

__version__ = "0.1.0"

# Backwards-friendly alias: the notebook calls this quantity swot_ssh.
swath_ssh = swot_ssh
