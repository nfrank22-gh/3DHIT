import jax.numpy as jnp
import jax.random as jr
import numpy as np

from conftest import fill3
from hit3d import (
    ddx,
    ddy,
    ddz,
    laplacian,
    make_grid,
    project,
    to_physical,
    to_spectral,
)

N = 32


def test_shapes(grid64):
    g = grid64
    assert g.kx.shape == (N, 1, 1)
    assert g.ky.shape == (1, N, 1)
    assert g.kz.shape == (1, 1, N // 2 + 1)
    assert g.k2.shape == (N, N, N // 2 + 1)
    assert g.inv_k2[0, 0, 0] == 0
    assert g.weights.shape == (1, 1, N // 2 + 1)
    assert g.weights[0, 0, 0] == 1
    assert g.weights[0, 0, -1] == 1  # even Nz: Nyquist plane single-counted
    assert g.weights[0, 0, 1] == 2


def test_dealias_mask(grid64):
    # per-direction 2/3 box, False at k = 0 (z is the rfft-halved axis)
    g = grid64
    cut = N // 3
    m = np.asarray(g.dealias)
    assert not m[0, 0, 0]
    assert m[cut, 0, 0]  # nx = cut retained
    assert not m[cut + 1, 0, 0]  # nx = cut + 1 masked
    assert m[0, N - cut, 0]  # ny = -cut retained
    assert not m[0, N - cut - 1, 0]  # ny = -(cut + 1) masked
    assert m[0, 0, cut]  # nz = cut retained
    assert not m[0, 0, cut + 1]
    assert m[1, 0, 0]  # low modes retained


def test_transform_roundtrip(grid64):
    u = jr.normal(jr.key(0), (3, N, N, N), dtype=jnp.float64)
    assert jnp.allclose(to_physical(to_spectral(u, grid64), grid64), u)


def test_derivatives_exact_on_trig(grid64):
    g = grid64
    x = (jnp.arange(N) * (2 * np.pi / N)).reshape(-1, 1, 1)
    y = (jnp.arange(N) * (2 * np.pi / N)).reshape(1, -1, 1)
    z = (jnp.arange(N) * (2 * np.pi / N)).reshape(1, 1, -1)
    f = jnp.sin(x) * jnp.cos(2 * y) * jnp.sin(3 * z)
    u_hat = to_spectral(fill3(f, g), g)

    def phys(a):
        return to_physical(a, g)

    assert jnp.allclose(
        phys(ddx(u_hat, g)),
        fill3(jnp.cos(x) * jnp.cos(2 * y) * jnp.sin(3 * z), g),
        atol=1e-10,
    )
    assert jnp.allclose(
        phys(ddy(u_hat, g)),
        fill3(-2 * jnp.sin(x) * jnp.sin(2 * y) * jnp.sin(3 * z), g),
        atol=1e-10,
    )
    assert jnp.allclose(
        phys(ddz(u_hat, g)),
        fill3(3 * jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(3 * z), g),
        atol=1e-10,
    )
    assert jnp.allclose(
        phys(laplacian(u_hat, g)), fill3(-(1 + 4 + 9) * f, g), atol=1e-9
    )


def test_projection_divergence_free_and_idempotent(grid64):
    g = grid64
    key1, key2 = jr.split(jr.key(1))
    u_hat = jr.normal(key1, (3, N, N, N // 2 + 1)) + 1j * jr.normal(
        key2, (3, N, N, N // 2 + 1)
    )
    u_hat = project(u_hat, g)
    div = g.kx * u_hat[0] + g.ky * u_hat[1] + g.kz * u_hat[2]
    assert jnp.max(jnp.abs(div)) < 1e-10
    assert jnp.allclose(project(u_hat, g), u_hat)


def test_anisotropic_grid_shapes():
    g = make_grid(16, 8, 4, dtype=jnp.float64)
    assert g.k2.shape == (16, 8, 4 // 2 + 1)
    u = jr.normal(jr.key(2), (3, 16, 8, 4), dtype=jnp.float64)
    assert jnp.allclose(to_physical(to_spectral(u, g), g), u)
