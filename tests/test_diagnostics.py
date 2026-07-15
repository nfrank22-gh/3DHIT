import jax.numpy as jnp
import jax.random as jr
import numpy as np

from conftest import beltrami
from hit3d import (
    default_spectrum,
    dissipation,
    energy,
    energy_spectrum,
    enstrophy,
    random_field,
    taylor_microscale,
    taylor_reynolds,
    taylor_reynolds_field,
    to_spectral,
)

N = 32
NU = 0.05


def test_parseval(grid64):
    # spectral energy (with Hermitian double-counting along kz) equals the
    # physical-space mean ½⟨|u|²⟩
    u = jr.normal(jr.key(0), (3, N, N, N), dtype=jnp.float64)
    u_hat = to_spectral(u, grid64)
    assert jnp.isclose(energy(u_hat, grid64), 0.5 * jnp.sum(u**2) / N**3)


def test_spectrum_sums_to_energy(grid64):
    u_hat = random_field(jr.key(1), grid64)
    _, Es = energy_spectrum(u_hat, grid64)
    assert np.isclose(Es.sum(), float(energy(u_hat, grid64)))


def test_random_field_matches_target_spectrum(grid64):
    # the per-shell rescale makes occupied shells match the target exactly
    u_hat = random_field(jr.key(2), grid64)
    ks, Es = energy_spectrum(u_hat, grid64)
    target = np.asarray(default_spectrum(jnp.asarray(ks)))
    resolved = slice(1, N // 3)  # shells 2..cut, safely inside the mask
    assert np.allclose(Es[resolved], target[resolved], rtol=1e-8)


def test_beltrami_scales(grid64):
    # Beltrami field: ω = u, so enstrophy = energy exactly, and the
    # derived scales follow analytically
    g = grid64
    u_hat = beltrami(g)
    assert jnp.isclose(enstrophy(u_hat, g), energy(u_hat, g))
    assert jnp.isclose(dissipation(u_hat, g, NU), 2 * NU * energy(u_hat, g))
    assert jnp.isclose(taylor_microscale(u_hat, g), np.sqrt(5))

    # the field-based Re_λ delegates to the scalar (E, ε, ν) formula —
    # exactly one implementation, shared with postprocessing
    E, eps = energy(u_hat, g), dissipation(u_hat, g, NU)
    assert jnp.isclose(
        taylor_reynolds_field(u_hat, g, NU), taylor_reynolds(E, eps, NU)
    )


def test_autocorrelation_and_integral_scale(grid64):
    from hit3d import integral_lengthscale, longitudinal_autocorrelation

    u_hat = random_field(jr.key(3), grid64)
    r, f = longitudinal_autocorrelation(u_hat, grid64)
    assert f[0] == 1.0
    assert len(r) == N // 2 + 1
    ell = integral_lengthscale(r, f)
    assert 0 < ell < grid64.Lx


def test_velocity_moments():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=200_000)
    mom = np.array([0.0, 1.0, 0.0, 3.0])
    got = __import__("hit3d").velocity_moments(samples)
    assert np.allclose(
        [got["mean"], got["variance"], got["skewness"], got["flatness"]],
        mom,
        atol=0.05,
    )
