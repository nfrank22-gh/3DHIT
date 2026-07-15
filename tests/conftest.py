"""Test configuration: enable float64 (tests validate physics in double
precision on CPU; the package default stays float32) and share the ABC/
Beltrami field used across test modules."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hit3d import dealias, make_grid, to_spectral  # noqa: E402


def beltrami(grid):
    """ABC flow (A = B = C = 1): a Beltrami field (ω = u, so u × ω ≡ 0).
    The exact Navier–Stokes solution from it is pure viscous decay
    û(t) = û₀·e^(−νt) — all active modes have |k|² = 1."""
    t = grid.dtype
    x = (jnp.arange(grid.Nx, dtype=t) * (2 * np.pi / grid.Nx)).reshape(-1, 1, 1)
    y = (jnp.arange(grid.Ny, dtype=t) * (2 * np.pi / grid.Ny)).reshape(1, -1, 1)
    z = (jnp.arange(grid.Nz, dtype=t) * (2 * np.pi / grid.Nz)).reshape(1, 1, -1)
    shape = (grid.Nx, grid.Ny, grid.Nz)
    u = jnp.stack((
        jnp.broadcast_to(jnp.sin(z) + jnp.cos(y), shape),
        jnp.broadcast_to(jnp.sin(x) + jnp.cos(z), shape),
        jnp.broadcast_to(jnp.sin(y) + jnp.cos(x), shape),
    ))
    return to_spectral(u, grid)


@pytest.fixture
def grid64():
    return make_grid(32, dtype=jnp.float64)


def fill3(f, grid):
    """4D physical field with the same 3D pattern in every component."""
    return jnp.broadcast_to(f, (3, grid.Nx, grid.Ny, grid.Nz))


def dealiased_random_spectral(key, grid):
    """rfft of a random real field (Hermitian-consistent by construction)."""
    import jax.random as jr

    u = jr.normal(key, (3, grid.Nx, grid.Ny, grid.Nz), dtype=grid.dtype)
    return dealias(to_spectral(u, grid), grid)
