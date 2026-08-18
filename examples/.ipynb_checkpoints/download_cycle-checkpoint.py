"""Download a SWOT cycle without storing credentials in source code."""

from getpass import getpass
from pathlib import Path

from swotgem import download_cycle

cycle = 48
username = input("AVISO username: ")
password = getpass("AVISO password: ")
output_directory = Path("EXPERT_DATA") / f"cycle_{cycle:03d}"

files = download_cycle(
    cycle,
    output_directory,
    username=username,
    password=password,
)
print(f"Ready: {len(files)} NetCDF files in {output_directory}")

