"""Incompressible Navier–Stokes in spectral space, rotational form:

    dû/dt = P(k) [ (u × ω)^ + f̂ ] − ν k² û

`make_ns_rhs(grid)` returns a pure function ``rhs(u_hat, params, t)``;
`NSParams` carries the differentiable physical scalars (ν and the forcing's
coefficients), so gradients w.r.t. them need no refactoring — just
``jax.grad`` with the params pytree as the argument.

Evaluation pipeline (XLA owns all buffers and FFT plans):
  1. ω̂ = ik × û; transform ω̂ and û to physical space
  2. compute u × ω pointwise in physical space
  3. forward-transform and add the forcing — BEFORE projection, so a
     non-solenoidal forcing's gradient part is absorbed into the pressure
     (the ik·f̂/|k|² term of the eliminated pressure equation)
  4. dealias (2/3 mask, which also zeroes k = 0) and project
     divergence-free
  5. add the viscous term −ν k² û (û stays exactly dealiased, so it needs
     no extra masking)

Call `make_ns_rhs` once per grid and reuse the returned function: jitted
callers (`rollout`, `evolve`) treat it as a static argument, so every new
closure triggers a fresh compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from .forcing import NoForcing
from .grid import dealias, project, to_physical, to_spectral

__all__ = ["NSParams", "make_ns_rhs", "linear_operator"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["nu", "forcing"],
    meta_fields=[],
)
@dataclass(frozen=True)
class NSParams:
    """Differentiable parameters of the Navier–Stokes RHS: the kinematic
    viscosity `nu` plus the forcing (whose scalar coefficients are pytree
    leaves of their own)."""

    nu: jax.Array
    forcing: object = field(default_factory=NoForcing)

    @property
    def label(self):
        return f"NavierStokes_nu{float(self.nu):g}_{self.forcing.label}"

    def __repr__(self):
        return f"NavierStokes(nu={float(self.nu):g}, forcing={self.forcing!r})"


def make_ns_rhs(grid):
    """Build the rotational-form Navier–Stokes RHS on `grid`.

    Returns ``rhs(u_hat, params: NSParams, t) -> du_hat``, pure and
    jit/grad-transparent.
    """

    def rhs(u_hat, params, t):
        kx, ky, kz = grid.kx, grid.ky, grid.kz

        # 1. ω̂ = ik × û, then both to physical space
        w_hat = 1j * jnp.stack((
            ky * u_hat[2] - kz * u_hat[1],
            kz * u_hat[0] - kx * u_hat[2],
            kx * u_hat[1] - ky * u_hat[0],
        ))
        u = to_physical(u_hat, grid)
        w = to_physical(w_hat, grid)

        # 2. nonlinear term u × ω, pointwise
        nl = jnp.stack((
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ))

        # 3. back to spectral space; forcing enters before dealias/projection
        du = to_spectral(nl, grid) + params.forcing.term(u_hat, grid, t)

        # 4. dealias (also zeroes k = 0) and project divergence-free
        du = project(dealias(du, grid), grid)

        # 5. viscous term
        return du - params.nu * grid.k2 * u_hat

    return rhs


def linear_operator(grid, params):
    """Diagonal viscous operator −ν k² for future integrating-factor / ETD
    schemes."""
    return -params.nu * grid.k2
