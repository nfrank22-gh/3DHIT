import jax.numpy as jnp
import jax.random as jr

from conftest import beltrami, dealiased_random_spectral
from hit3d import NSParams, make_ns_rhs, project, rollout

NU = 0.05


def test_beltrami_rhs_is_pure_viscous_decay(grid64):
    # ABC flow: u × ω ≡ 0, so a single rhs evaluation must be exactly the
    # viscous term −ν·û (all active modes have |k|² = 1). Exercises the
    # full pipeline end-to-end: any error in the transforms, cross product,
    # dealiasing, or projection breaks it loudly.
    g = grid64
    u0 = beltrami(g)
    rhs = make_ns_rhs(g)
    du = rhs(u0, NSParams(nu=NU), 0.0)
    assert jnp.allclose(du, -NU * u0, atol=1e-10)


def test_beltrami_decay_matches_analytic(grid64):
    g = grid64
    u0 = beltrami(g)
    rhs = make_ns_rhs(g)
    dt, nsteps = 1e-2, 100
    u = rollout(rhs, u0, NSParams(nu=NU), dt, nsteps)
    exact = jnp.exp(-NU * dt * nsteps) * u0
    # norm-relative comparison (elementwise fails on ~1e-12 roundoff noise
    # in components whose exact value is 0)
    err = jnp.linalg.norm((u - exact).ravel()) / jnp.linalg.norm(exact.ravel())
    assert err < 1e-10


def test_rollout_preserves_solver_invariants(grid64):
    # divergence-free and dealiased are invariants of the evolution
    g = grid64
    u0 = project(dealiased_random_spectral(jr.key(3), g), g)
    rhs = make_ns_rhs(g)
    u = rollout(rhs, u0, NSParams(nu=0.02), 1e-3, 10)
    div = g.kx * u[0] + g.ky * u[1] + g.kz * u[2]
    assert jnp.max(jnp.abs(div)) < 1e-8
    assert jnp.allclose(u * g.dealias, u)


def test_inviscid_energy_conservation(grid64):
    # ν = 0: energy is conserved up to RK4's O(dt⁵)-per-step error
    from hit3d import energy

    g = grid64
    u0 = project(dealiased_random_spectral(jr.key(4), g), g)
    rhs = make_ns_rhs(g)
    u = rollout(rhs, u0, NSParams(nu=0.0), 1e-3, 20)
    assert jnp.isclose(energy(u, g), energy(u0, g), rtol=1e-8)
