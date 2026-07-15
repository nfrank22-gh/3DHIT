"""Filesystem-safe run labels (characters ``[A-Za-z0-9._-]``).

`label(x)` gives a short slug describing `x` and its parameters; compose
grid, params, and scheme into run directory names with `run_label`, e.g.
``results/N64_NavierStokes_nu0.001_NoForcing_RK4``. Equal parameters give
equal labels, so reruns land in the same folder. Numbers are formatted with
%g so float32/float64 print identically ("0.001").

Pretty printing is the classes' own ``__repr__``; this is the slug layer.
"""

from __future__ import annotations

from .grid import Grid

__all__ = ["label", "run_label"]


def label(x):
    """Short filesystem-safe identifier for a `Grid`, forcing, or
    `NSParams` (the latter two carry their own ``label`` property)."""
    if isinstance(x, Grid):
        if x.Nx == x.Ny == x.Nz:
            return f"N{x.Nx}"
        return f"N{x.Nx}x{x.Ny}x{x.Nz}"
    return x.label


def run_label(grid, params, scheme="RK4"):
    """Directory name for a run: ``<grid>_<params>_<scheme>``."""
    return f"{label(grid)}_{label(params)}_{scheme}"
