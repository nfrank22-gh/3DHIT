"""State allocation and initial conditions.

The solver state is a single complex array of shape ``(3, Nx, Ny, Nz//2+1)``
(rfft layout, velocity components first). All initial conditions are
dealiased and zero-mean (hard invariants of the solver — see the dealias
mask), and randomness goes through explicit `jax.random` keys, so seeds
reproduce on every backend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .grid import dealias, project, spectral_shape, to_spectral

__all__ = ["spectral_state", "random_field", "taylor_green", "default_spectrum"]


def _complex_dtype(grid):
    return jnp.promote_types(grid.dtype, jnp.complex64)


def spectral_state(grid):
    """Zeroed spectral velocity state ``(3, Nx, Ny, Nz//2+1)``."""
    return jnp.zeros(spectral_shape(grid), dtype=_complex_dtype(grid))


def default_spectrum(k):
    """Default target spectrum for `random_field`: k⁴·exp(−2k²)."""
    return k**4 * jnp.exp(-2 * k**2)


def random_field(key, grid, spectrum=default_spectrum):
    """Solenoidal random field whose shell-summed energy spectrum matches
    ``spectrum(k)`` (shells of unit spacing k₀ = 2π/Lx; cubic box assumed
    for the binning).

    Route: uniform white noise in physical space → rfft → dealias →
    project → per-shell rescale. Hermitian symmetry is automatic (we
    transform a real field), and the shell rescaling is diagonal in k so it
    preserves the divergence-free projection.
    """
    u = jax.random.uniform(key, (3, grid.Nx, grid.Ny, grid.Nz),
                           dtype=grid.dtype) - 0.5
    u_hat = project(dealias(to_spectral(u, grid), grid), grid)

    # Shell-by-shell rescale of the white-noise field to the target
    # spectrum, via one bincount over shell indices.
    k0 = 2 * np.pi / grid.Lx
    smax = int(np.ceil(np.sqrt(3) * max(grid.Nx, grid.Ny, grid.Nz) / 2)) + 1
    shell = jnp.floor(jnp.sqrt(grid.k2) / k0 + 0.5).astype(jnp.int32)
    ntot2 = jnp.asarray(grid.Nx * grid.Ny * grid.Nz, grid.dtype) ** 2
    e_mode = grid.weights * jnp.sum(jnp.abs(u_hat) ** 2, axis=0) / (2 * ntot2)
    Es = jnp.bincount(shell.ravel(), weights=e_mode.ravel(), length=smax + 1)

    ks = jnp.arange(smax + 1, dtype=grid.dtype) * k0
    target = spectrum(ks)
    # Empty shells stay untouched; occupied shells with zero target are
    # zeroed; otherwise scale energy to the target.
    Es_safe = jnp.where(Es > 0, Es, 1)
    c = jnp.where(
        Es > 0,
        jnp.where(target > 0, jnp.sqrt(target / Es_safe), 0.0),
        1.0,
    ).astype(grid.dtype)
    return u_hat * c[shell]


def taylor_green(grid):
    """Taylor–Green vortex (useful deterministic validation case):

        u = (sin θx cos θy cos θz, −cos θx sin θy cos θz, 0)

    with θᵢ = 2π xᵢ/Lᵢ, so the field is periodic on any box.
    """
    t = grid.dtype
    x = (jnp.arange(grid.Nx, dtype=t) * (2 * np.pi / grid.Nx)).reshape(-1, 1, 1)
    y = (jnp.arange(grid.Ny, dtype=t) * (2 * np.pi / grid.Ny)).reshape(1, -1, 1)
    z = (jnp.arange(grid.Nz, dtype=t) * (2 * np.pi / grid.Nz)).reshape(1, 1, -1)
    u = jnp.stack((
        jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
        -jnp.cos(x) * jnp.sin(y) * jnp.cos(z),
        jnp.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=t),
    ))
    return dealias(to_spectral(u, grid), grid)
