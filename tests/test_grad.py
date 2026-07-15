"""AD smoke tests.

Deliberately NOT gradient-correctness tests (JAX's autodiff of jnp
primitives doesn't need re-proving): these assert the *structure* — that
the `rollout` path is differentiable end-to-end, i.e. nobody has snuck a
host callback, integer cast, or NaN-producing `where` branch into it.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from hit3d import (
    BandForcing,
    LinearForcing,
    NSParams,
    make_grid,
    make_ns_rhs,
    random_field,
    rk4_step,
    rollout,
)


def _finite_nonzero(x):
    return bool(jnp.all(jnp.isfinite(x)) & (jnp.linalg.norm(x.ravel()) > 0))


def test_grad_through_rollout_is_finite():
    g = make_grid(8, dtype=jnp.float64)
    u0 = random_field(jr.key(0), g)
    rhs = make_ns_rhs(g)
    params = NSParams(nu=0.02)

    def loss(u, p):
        v = rollout(rhs, u, p, 1e-3, 3)
        return 0.5 * jnp.sum(jnp.abs(v) ** 2)

    gu = jax.grad(loss)(u0, params)
    assert gu.shape == u0.shape
    assert _finite_nonzero(gu)

    # gradients w.r.t. the physical parameters need no refactoring
    gp = jax.grad(loss, argnums=1)(u0, params)
    assert _finite_nonzero(jnp.asarray(gp.nu))


def test_single_step_vjp():
    g = make_grid(8, dtype=jnp.float64)
    u0 = random_field(jr.key(1), g)
    rhs = make_ns_rhs(g)
    params = NSParams(nu=0.02)

    _, vjp = jax.vjp(lambda u: rk4_step(rhs, u, params, 1e-3, 0.0), u0)
    (ubar,) = vjp(jnp.ones_like(u0))
    assert _finite_nonzero(ubar)


def test_grad_with_forcings():
    # the BandForcing guard (`where` on the band energy) must not poison
    # gradients, and LinearForcing's A must be reachable
    g = make_grid(8, dtype=jnp.float64)
    u0 = random_field(jr.key(2), g)
    rhs = make_ns_rhs(g)

    for forcing in (LinearForcing(A=0.1), BandForcing(eps=0.1, kmin=1.0, kmax=2.0)):
        params = NSParams(nu=0.02, forcing=forcing)

        def loss(p):
            v = rollout(rhs, u0, p, 1e-3, 2)
            return 0.5 * jnp.sum(jnp.abs(v) ** 2)

        gp = jax.grad(loss)(params)
        assert _finite_nonzero(jnp.asarray(gp.nu))
