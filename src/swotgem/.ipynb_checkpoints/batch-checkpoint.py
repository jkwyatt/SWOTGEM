"""File-loop helpers kept separate from the numerical core."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import xarray as xr

from .core import crop_swot_dataset, reconstruct_swath


def iter_reconstructed_swaths(
    swot_files: Iterable[str | Path],
    gem: xr.Dataset,
    *,
    lat_bounds: Sequence[float] = (-70.0, -45.0),
    lon_bounds: Sequence[float] = (-180.0, 180.0),
    errors: str = "warn",
    **reconstruct_kwargs,
) -> Iterator[tuple[Path, xr.Dataset]]:
    """Yield a loaded reconstruction for every intersecting SWOT file.

    Parameters accepted by :func:`swotgem.reconstruct_swath`, including
    ``pressure`` or ``pressure_index``, can be supplied as keyword arguments.
    ``errors='warn'`` skips unreadable passes with a warning; ``errors='raise'``
    stops at the first failure.
    """

    if errors not in {"warn", "raise"}:
        raise ValueError("errors must be 'warn' or 'raise'.")

    for item in swot_files:
        path = Path(item)
        try:
            with xr.open_dataset(path) as raw_swath:
                swath = crop_swot_dataset(
                    raw_swath,
                    lat_bounds=lat_bounds,
                    lon_bounds=lon_bounds,
                )
                if swath is None:
                    continue
                fields = reconstruct_swath(
                    gem,
                    swath,
                    **reconstruct_kwargs,
                ).load()
            yield path, fields
        except Exception as error:
            if errors == "raise":
                raise
            warnings.warn(f"Skipping {path.name}: {error}", stacklevel=2)

