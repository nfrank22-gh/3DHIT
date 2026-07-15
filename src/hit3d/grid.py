"""Geometry + precomputed spectral operators for a triply periodic box.

The `Grid` is a frozen pytree dataclass, shareable and passed (or closed
over) by every function that needs spectral operators. It is never
differentiated — treat it as a constant.

State layout (the load-bearing invariant of the package): the spectral
velocity is one complex array of shape ``(3, Nx, Ny, Nz//2 + 1)`` —
velocity components first, rfft-halved axis last (`jnp.fft.rfftn` over
axes ``(1, 2, 3)``). Hermitian double-counting therefore applies along
the kz axis.

Precision: everything is parametric in `dtype`; float32 is the default.
Float64 requires ``jax.config.update("jax_enable_x64", True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "Grid",
    "make_grid",
    "spectral_shape",
    "ddx",
    "ddy",
    "ddz",
    "laplacian",
    "dealias",
    "project",
    "to_physical",
    "to_spectral",
]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["kx", "ky", "kz", "k2", "inv_k2", "dealias", "weights"],
    meta_fields=["Nx", "Ny", "Nz", "Lx", "Ly", "Lz"],
)
@dataclass(frozen=True)
class Grid:
    """Parameters and spectral operators for an Nx×Ny×Nz periodic box.

    - ``kx, ky, kz``: wavenumber arrays shaped for broadcasting against one
      component of the spectral state — ``(Nx, 1, 1)``, ``(1, Ny, 1)``,
      ``(1, 1, Nz//2+1)`` (rfft layout: kz holds only non-negative modes)
    - ``k2``: |k|² on the full spectral grid ``(Nx, Ny, Nz//2+1)``
    - ``inv_k2``: 1/|k|² with the k = 0 mode set to zero
    - ``dealias``: per-direction (Orszag) 2/3-rule bool mask, product of
      three 1D masks |n_i| ≤ N_i//3; also False at k = 0, enforcing the
      zero-mean invariant on anything it is applied to
    - ``weights``: Hermitian shell-sum weights ``(1, 1, Nz//2+1)`` — 2 for
      kz > 0 modes (each represents an unstored conjugate pair), 1 for the
      kz = 0 plane and, for even Nz, the Nyquist plane
    """

    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    kx: jax.Array
    ky: jax.Array
    kz: jax.Array
    k2: jax.Array
    inv_k2: jax.Array
    dealias: jax.Array
    weights: jax.Array

    @property
    def dtype(self):
        """Real dtype of the grid (and hence the run's precision)."""
        return self.k2.dtype

    def __repr__(self):
        return (
            f"Grid[{self.dtype}]({self.Nx}x{self.Ny}x{self.Nz}, "
            f"L=({self.Lx:g}, {self.Ly:g}, {self.Lz:g}))"
        )


def make_grid(Nx, Ny=None, Nz=None, *, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi,
              dtype=jnp.float32):
    """Build a `Grid`, precomputing wavenumbers, masks, and weights.

    ``make_grid(N)`` is the cubic convenience form. Arrays are built in
    NumPy and converted once, so construction never traces.
    """
    Ny = Nx if Ny is None else Ny
    Nz = Nx if Nz is None else Nz
    dtype = np.dtype(dtype)

    # Integer frequencies (full fft layout in x/y, rfft layout in z).
    nx = np.fft.fftfreq(Nx, 1 / Nx)
    ny = np.fft.fftfreq(Ny, 1 / Ny)
    nz = np.arange(Nz // 2 + 1, dtype=float)

    kx = (2 * np.pi / Lx * nx).astype(dtype).reshape(-1, 1, 1)
    ky = (2 * np.pi / Ly * ny).astype(dtype).reshape(1, -1, 1)
    kz = (2 * np.pi / Lz * nz).astype(dtype).reshape(1, 1, -1)

    k2 = kx**2 + ky**2 + kz**2
    with np.errstate(divide="ignore"):
        inv_k2 = np.where(k2 == 0, dtype.type(0), 1 / k2).astype(dtype)

    # Per-direction 2/3-rule box mask; k = 0 is masked too (zero-mean
    # invariant). Nyquist modes fall outside automatically.
    mask = (
        (np.abs(nx).reshape(-1, 1, 1) <= Nx // 3)
        & (np.abs(ny).reshape(1, -1, 1) <= Ny // 3)
        & (np.abs(nz).reshape(1, 1, -1) <= Nz // 3)
    )
    mask[0, 0, 0] = False

    w = np.full(Nz // 2 + 1, 2, dtype=dtype)
    w[0] = 1
    if Nz % 2 == 0:
        w[-1] = 1

    return Grid(
        Nx=Nx, Ny=Ny, Nz=Nz, Lx=float(Lx), Ly=float(Ly), Lz=float(Lz),
        kx=jnp.asarray(kx), ky=jnp.asarray(ky), kz=jnp.asarray(kz),
        k2=jnp.asarray(k2), inv_k2=jnp.asarray(inv_k2),
        dealias=jnp.asarray(mask), weights=jnp.asarray(w.reshape(1, 1, -1)),
    )


def spectral_shape(grid):
    """Shape of the spectral state: ``(3, Nx, Ny, Nz//2+1)``."""
    return (3, grid.Nx, grid.Ny, grid.Nz // 2 + 1)


# ---------------------------------------------------------------------------
# Spectral operators — trivial broadcasts against the stored wavenumbers.
# All are pure: they take and return arrays.
# ---------------------------------------------------------------------------


def ddx(u_hat, grid):
    """Spectral x-derivative ``i·kx·û`` (any spectral array)."""
    return 1j * grid.kx * u_hat


def ddy(u_hat, grid):
    """Spectral y-derivative."""
    return 1j * grid.ky * u_hat


def ddz(u_hat, grid):
    """Spectral z-derivative."""
    return 1j * grid.kz * u_hat


def laplacian(u_hat, grid):
    """Spectral Laplacian ``-|k|²·û``."""
    return -grid.k2 * u_hat


def dealias(u_hat, grid):
    """Zero all modes outside the 2/3-rule mask (including k = 0)."""
    return u_hat * grid.dealias


def project(u_hat, grid):
    """Divergence-free projection P(k) = I − k kᵀ/|k|² of the 3-component
    spectral field. At k = 0 the projection is the identity (`inv_k2` is
    zero there); the zero mode is handled by the dealias mask instead."""
    div = (
        grid.kx * u_hat[0] + grid.ky * u_hat[1] + grid.kz * u_hat[2]
    ) * grid.inv_k2
    return u_hat - jnp.stack((grid.kx * div, grid.ky * div, grid.kz * div))


def to_physical(u_hat, grid):
    """Inverse transform to the physical field ``(3, Nx, Ny, Nz)`` (real)."""
    return jnp.fft.irfftn(u_hat, s=(grid.Nx, grid.Ny, grid.Nz), axes=(1, 2, 3))


def to_spectral(u, grid):
    """Forward transform of a physical field to ``(3, Nx, Ny, Nz//2+1)``."""
    return jnp.fft.rfftn(u, axes=(1, 2, 3))
