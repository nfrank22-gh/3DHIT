"""HDF5 key-naming for the files `FieldWriter` / `save_series` write.

Used by both the writer (`callbacks`) and the reader (`report`), so the
on-disk layout has exactly one definition instead of matching string
literals typed independently on each side.

Layout:

    grid/Nx, grid/Ny, grid/Nz, grid/Lx, grid/Ly, grid/Lz, grid/nu  # optional
    step_00000000/t, step_00000000/<field>, ...   # one group per snapshot
    series/<name>/t, series/<name>/<field>, ...   # one group per Diagnostic
"""

from __future__ import annotations

__all__ = [
    "STEP_PREFIX",
    "step_key",
    "step_num",
    "is_step_key",
    "grid_key",
    "series_key",
    "series_field",
]

STEP_PREFIX = "step_"
STEP_DIGITS = 8


def step_key(n):
    """Group name for snapshot step `n`, e.g. ``step_key(50) == "step_00000050"``."""
    return f"{STEP_PREFIX}{n:0{STEP_DIGITS}d}"


def is_step_key(key):
    """Whether `key` is a snapshot group name."""
    return key.startswith(STEP_PREFIX)


def step_num(key):
    """Step number encoded in a snapshot group name."""
    return int(key.rsplit("_", 1)[-1])


def grid_key(name):
    """Key for grid-metadata field `name` ("Nx", "Lx", "nu", ...)."""
    return f"grid/{name}"


def series_key(name):
    """Group name for the dense series recorded under `name`."""
    return f"series/{name}"


def series_field(name, fld):
    """Key for field `fld` of the dense series `name`."""
    return f"{series_key(name)}/{fld}"
