"""Pluggable forcing slot for the RHS.

A forcing is a small frozen pytree dataclass whose scalar coefficients are
pytree leaves — it travels inside `NSParams`, so `jax.grad` can reach the
coefficients with no extra plumbing. Extension contract (all a new forcing
needs):

  * ``term(u_hat, grid, t)`` — return the spectral forcing contribution
    (added to dû by the RHS). It need NOT be divergence-free and need NOT
    dealias itself: it is applied BEFORE the projection and the 2/3 mask,
    so any gradient part is absorbed into the pressure (the ik·f̂/|k|² term
    of the eliminated pressure equation) and any aliased / k = 0 content is
    removed by the mask. The RHS guarantees this ordering.
  * ``injection(u_hat, grid)`` — instantaneous injected power P = ⟨u·f⟩,
    exact per forcing type. Diagnostics use this for the energy budget
    dE/dt = P − ε; keeping it a per-forcing method matters for forcings
    whose power cannot be measured numerically from a single evaluation
    (e.g. stochastic forcing, whose mean injection carries an Itô
    correction).
  * ``label`` property — filesystem-safe slug for run-directory names.

Forcings live in the RHS parameters (part of the dynamics), never in
callbacks — callbacks are pure observers so the solver stays AD-compatible.

TODO later: StochasticForcing (needs a jax.random key-threading strategy;
note that stochastic forcing also complicates the AD/adjoint story).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["NoForcing", "LinearForcing", "BandForcing"]


def _fmt(x):
    return f"{float(x):g}"


def _ntot2(grid):
    return jnp.asarray(grid.Nx * grid.Ny * grid.Nz, grid.dtype) ** 2


def _weighted_2E(u_hat, grid, mask=None):
    """2·E (twice the kinetic energy) restricted to `mask`, with Hermitian
    double-counting along kz."""
    w = grid.weights if mask is None else grid.weights * mask
    return jnp.sum(w * jnp.sum(jnp.abs(u_hat) ** 2, axis=0)) / _ntot2(grid)


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
@dataclass(frozen=True)
class NoForcing:
    """No forcing (decaying turbulence)."""

    def term(self, u_hat, grid, t):
        return jnp.zeros((), dtype=u_hat.dtype)  # broadcasts away in dû + term

    def injection(self, u_hat, grid):
        return jnp.zeros((), dtype=grid.dtype)

    @property
    def label(self):
        return "NoForcing"

    def __repr__(self):
        return "NoForcing()"


@partial(jax.tree_util.register_dataclass, data_fields=["A"], meta_fields=[])
@dataclass(frozen=True)
class LinearForcing:
    """Lundgren's linear forcing f = A·u (Rosales & Meneveau, Phys. Fluids
    17, 095106 (2005), Sec. II), applied in its exact spectral equivalent
    f̂ = A·û (Eq. 8 — the Fourier transform is linear, so the physical- and
    spectral-space implementations coincide).

    Prescribing the constant `A` imposes an inverse turnover timescale; the
    flow converges to a statistically stationary state with ε = 3A·u_rms²
    and integral scale ℓ = u_rms³/ε ≈ 0.19·L, independent of the initial
    spectrum. For a target injection rate ε on a box of size L, Eq. 12
    gives A ≈ ε^(1/3)/L^(2/3).

    Divergence-free and dealiased by construction (proportional to û,
    which the solver keeps exactly dealiased).
    """

    A: jax.Array  # forcing coefficient (inverse timescale), f = A·u

    def term(self, u_hat, grid, t):
        return self.A * u_hat

    def injection(self, u_hat, grid):
        """Exact injected power P = ⟨u·(Au)⟩ = 2A·E."""
        return self.A * _weighted_2E(u_hat, grid)

    @property
    def label(self):
        return f"LinearForcing_A{_fmt(self.A)}"

    def __repr__(self):
        return f"LinearForcing(A={_fmt(self.A)})"


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["eps", "kmin", "kmax"],
    meta_fields=[],
)
@dataclass(frozen=True)
class BandForcing:
    """Constant-power low-wavenumber forcing: injects energy at exactly the
    rate `eps` into the shell kmin ≤ |k| ≤ kmax by amplifying the velocity
    in the band,

        f̂ = (ε / 2E_band) û    on the band,

    so that ⟨u·f⟩ = ε identically (E_band is the current kinetic energy in
    the band, Hermitian-weighted). In a statistically stationary state the
    dissipation therefore equals ε.

    Divergence-free by construction (proportional to û). If the band is
    empty of energy (2E_band ≤ machine eps), the forcing is zero rather
    than dividing by ~0.
    """

    eps: jax.Array  # exact injection rate ⟨u·f⟩
    kmin: jax.Array
    kmax: jax.Array

    def _mask(self, grid):
        return (self.kmin**2 <= grid.k2) & (grid.k2 <= self.kmax**2)

    def term(self, u_hat, grid, t):
        mask = self._mask(grid)
        twoE = _weighted_2E(u_hat, grid, mask)
        tiny = jnp.finfo(grid.dtype).eps
        c = jnp.where(twoE > tiny, self.eps / jnp.where(twoE > tiny, twoE, 1), 0)
        return c * mask * u_hat

    def injection(self, u_hat, grid):
        """Exact injected power: `eps` by construction, or zero when the
        band is empty and `term` skips (same guard)."""
        twoE = _weighted_2E(u_hat, grid, self._mask(grid))
        tiny = jnp.finfo(grid.dtype).eps
        return jnp.where(twoE > tiny, self.eps, 0.0).astype(grid.dtype)

    @property
    def label(self):
        return (
            f"BandForcing_eps{_fmt(self.eps)}"
            f"_k{_fmt(self.kmin)}-{_fmt(self.kmax)}"
        )

    def __repr__(self):
        return (
            f"BandForcing(eps={_fmt(self.eps)}, "
            f"k=[{_fmt(self.kmin)}, {_fmt(self.kmax)}])"
        )
