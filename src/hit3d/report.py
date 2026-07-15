"""Reading/parsing for `FieldWriter` snapshot files (HDF5).

Deliberately matplotlib-free (unlike `plotting`), so it is exercised by the
ordinary test suite; the plotting functions call these and only build
figures from the result — they do no HDF5 parsing of their own.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import h5py
import numpy as np

from .grid import make_grid
from .schema import grid_key, is_step_key, series_key, step_key, step_num

__all__ = ["load_run", "read_series", "step_keys", "read_grid", "read_snapshot"]


def step_keys(f):
    """Sorted ``step_########`` group names in an open h5py file."""
    return sorted(k for k in f.keys() if is_step_key(k))


def read_grid(f, dtype):
    """Reconstruct the `Grid` (precision `dtype`) and read ν from a
    `FieldWriter` file's self-describing `grid` group. Falls back to a 2π
    cube with unknown ν for files predating the metadata group, inferring
    the sizes from the rfft layout of the first snapshot (assumes even Nz).
    """
    if "grid" in f:
        g = make_grid(
            int(f[grid_key("Nx")][()]),
            int(f[grid_key("Ny")][()]),
            int(f[grid_key("Nz")][()]),
            Lx=float(f[grid_key("Lx")][()]),
            Ly=float(f[grid_key("Ly")][()]),
            Lz=float(f[grid_key("Lz")][()]),
            dtype=dtype,
        )
        nu = float(f[grid_key("nu")][()]) if grid_key("nu") in f else None
        return g, nu
    keys = step_keys(f)
    if not keys:
        raise ValueError("no snapshots found in file")
    shape = f[f"{keys[0]}/u_hat"].shape  # (3, Nx, Ny, Nz//2+1)
    nz = 2 * (shape[3] - 1)
    warnings.warn(
        f"no grid metadata in file; assuming a 2π cube "
        f"({shape[1]}x{shape[2]}x{nz}) and unknown nu",
        stacklevel=2,
    )
    return make_grid(shape[1], shape[2], nz, dtype=dtype), None


def _state_dtype(f):
    """Real precision of the stored state, from the first snapshot."""
    c = f[f"{step_keys(f)[0]}/u_hat"].dtype
    return np.float64 if c == np.complex128 else np.float32


def read_snapshot(f, step):
    """Spectral state of snapshot `step` from an open file, as NumPy."""
    return np.asarray(f[f"{step_key(step)}/u_hat"])


def load_run(path):
    """Open the `FieldWriter` file at `path` and return a namespace with
    its self-describing ``grid``, ``nu`` (None if the file has no grid/nu
    entry), state ``dtype``, and the sorted list of snapshot ``steps``.
    Cheap: does not load any field arrays. Errors if the file has no
    snapshots."""
    with h5py.File(path, "r") as f:
        keys = step_keys(f)
        if not keys:
            raise ValueError(f"no snapshots found in {path}")
        dtype = _state_dtype(f)
        grid, nu = read_grid(f, dtype)
        return SimpleNamespace(
            grid=grid, nu=nu, dtype=dtype, steps=[step_num(k) for k in keys]
        )


def read_series(path_or_file, name):
    """Read the dense ``series/<name>`` group written by `save_series` —
    columnar: `t` plus either `values` (scalar diagnostics) or one vector
    per budget field (e.g. E, eps, P) — as a dict of NumPy arrays, or None
    if the file has no such group."""
    if isinstance(path_or_file, h5py.File):
        return _read_series(path_or_file, name)
    with h5py.File(path_or_file, "r") as f:
        return _read_series(f, name)


def _read_series(f, name):
    grp = series_key(name)
    if grp not in f:
        return None
    return {k: np.asarray(f[grp][k]) for k in f[grp].keys()}
