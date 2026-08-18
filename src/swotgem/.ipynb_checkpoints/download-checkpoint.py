"""Authenticated download helpers for SWOT Expert cycle files."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_BASE_URL = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/catalog/"
    "dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/"
    "v3_0/Expert/forward"
)


def _authenticated_session(
    username: str | None = None,
    password: str | None = None,
) -> requests.Session:
    username = username or os.getenv("SWOT_USERNAME")
    password = password or os.getenv("SWOT_PASSWORD")
    if not username or not password:
        raise ValueError(
            "Supply username/password or set SWOT_USERNAME and SWOT_PASSWORD."
        )
    session = requests.Session()
    session.auth = (username, password)
    return session


def list_cycle_files(
    cycle: int,
    *,
    session: requests.Session,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
) -> tuple[list[str], str]:
    """Return NetCDF filenames and the file-server base URL for one cycle."""

    cycle_string = f"{int(cycle):03d}"
    catalog_url = f"{base_url.rstrip('/')}/cycle_{cycle_string}/catalog.html"
    response = session.get(catalog_url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    catalog_path = catalog_url.split("/thredds/catalog/", maxsplit=1)[1]
    catalog_path = catalog_path.removesuffix("/catalog.html")
    fileserver_base = (
        "https://tds-odatis.aviso.altimetry.fr/thredds/fileServer/"
        f"{catalog_path}/"
    )

    matched: set[str] = set()
    for link in soup.find_all("a"):
        filename = link.get_text(strip=True)
        if not filename.endswith(".nc"):
            continue
        parts = filename.split("_")
        if len(parts) >= 8 and parts[5] == cycle_string:
            matched.add(filename)
    return sorted(matched), fileserver_base


def download_cycle(
    cycle: int,
    output_directory: str | Path,
    *,
    username: str | None = None,
    password: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    overwrite: bool = False,
    timeout: tuple[float, float] = (60.0, 600.0),
    chunk_size: int = 1024 * 1024,
) -> list[Path]:
    """Download every NetCDF pass in a SWOT cycle.

    Existing non-empty files are reused unless ``overwrite=True``. A partial
    ``.part`` file is used during transfer and removed after any failure.
    Credentials should be supplied at runtime or through environment variables;
    never place them in a notebook or source file.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    session = _authenticated_session(username=username, password=password)
    filenames, fileserver_base = list_cycle_files(
        cycle, session=session, base_url=base_url, timeout=timeout[0]
    )

    downloaded: list[Path] = []
    for filename in filenames:
        destination = output_directory / filename
        if destination.exists() and destination.stat().st_size > 0 and not overwrite:
            downloaded.append(destination)
            continue

        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with session.get(
                urljoin(fileserver_base, filename), stream=True, timeout=timeout
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            stream.write(chunk)
            partial.replace(destination)
            downloaded.append(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return downloaded

