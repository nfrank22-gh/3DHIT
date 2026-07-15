"""Analysis functions on ``(u_hat, grid)`` — usable both offline
(postprocessing) and inside `Diagnostic` callbacks during a run.

Spectral reductions are Parseval-type quadratic forms and account for the
rfft layout via the grid's Hermitian `weights`: kz > 0 modes represent two
conjugate modes each and are double-counted in every reduction. Quantities
are volume means (⟨·⟩ over the box), so with the unnormalized rfft
convention each sum carries a 1/(Nx·Ny·Nz)² factor.

The vorticity-based quantities assume `u_hat` is divergence-free (the
solver invariant), so that |ω̂|² = |k × û|² = k²|û|².

Shell-summed spectra use shells of unit spacing k₀ = 2π/Lx (cubic box
assumed for the binning), implemented as one `bincount` over shell indices.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .grid import to_physical

__all__ = [
    "energy",
    "enstrophy",
    "dissipation",
    "energy_spectrum",
    "component_spectra",
    "compensated_spectrum",
    "dissipation_constant",
    "kolmogorov_scale",
    "taylor_microscale",
    "taylor_reynolds",
    "taylor_reynolds_field",
    "energy_budget",
    "energy_budget_cb",
    "velocity_samples",
    "velocity_moments",
    "longitudinal_autocorrelation",
    "integral_lengthscale",
]


def _norm2(grid):
    return jnp.asarray(grid.Nx * grid.Ny * grid.Nz, grid.dtype) ** 2


def _weighted_sum(a2, grid, extra=None):
    """Hermitian-weighted Parseval sum ``Σ w·(extra·)a2 / Ntot²`` of a
    non-negative spectral density `a2` of shape (Nx, Ny, Nz//2+1)."""
    w = grid.weights if extra is None else grid.weights * extra
    return jnp.sum(w * a2) / _norm2(grid)


def energy(u_hat, grid):
    """Total kinetic energy ½⟨|u|²⟩ from the spectral state."""
    return _weighted_sum(jnp.sum(jnp.abs(u_hat) ** 2, axis=0), grid) / 2


def enstrophy(u_hat, grid):
    """Total enstrophy ½⟨|ω|²⟩ (assumes `u_hat` divergence-free)."""
    return _weighted_sum(
        jnp.sum(jnp.abs(u_hat) ** 2, axis=0), grid, extra=grid.k2
    ) / 2


def dissipation(u_hat, grid, nu):
    """Energy dissipation rate ε = ν⟨|ω|²⟩ = 2ν · enstrophy."""
    return 2 * nu * enstrophy(u_hat, grid)


# --- shell-summed spectra ---------------------------------------------------


def _shells(grid):
    """Shell indices (per spectral mode) and shell count for unit-k₀ bins:
    shell s covers |k| ∈ [(s−½)k₀, (s+½)k₀)."""
    k0 = 2 * np.pi / grid.Lx
    smax = int(np.ceil(np.sqrt(3) * max(grid.Nx, grid.Ny, grid.Nz) / 2)) + 1
    shell = jnp.floor(jnp.sqrt(grid.k2) / k0 + 0.5).astype(jnp.int32)
    return shell, smax, k0


def _shell_sum(a2, grid):
    """Bin the Hermitian-weighted spectral density `a2` into shells;
    returns a length-smax vector for shells 1..smax (the s = 0 bin — the
    k = 0 mode only — is dropped; solver states have no k = 0 content)."""
    shell, smax, _ = _shells(grid)
    e = (grid.weights * a2) / (2 * _norm2(grid))
    return jnp.bincount(shell.ravel(), weights=e.ravel(), length=smax + 1)[1:]


def energy_spectrum(u_hat, grid):
    """Shell-summed energy spectrum ``(k, E)``: `E[s]` is the energy in
    shell s+1, so ``sum(E) == energy(u_hat, grid)`` for dealiased states.
    Returned as host NumPy vectors ready for plotting."""
    shell, smax, k0 = _shells(grid)
    ks = np.arange(1, smax + 1) * k0
    Es = _shell_sum(jnp.sum(jnp.abs(u_hat) ** 2, axis=0), grid)
    return ks, np.asarray(Es)


def component_spectra(u_hat, grid):
    """Per-component shell spectra ``(k, E11, E22, E33)``, same shells as
    `energy_spectrum` (whose E is their sum). Isotropic turbulence has
    E11 ≈ E22 ≈ E33 throughout the resolved range; a systematic split flags
    anisotropy from the initial condition, forcing, or a solver bug."""
    shell, smax, k0 = _shells(grid)
    ks = np.arange(1, smax + 1) * k0
    spectra = [
        np.asarray(_shell_sum(jnp.abs(u_hat[i]) ** 2, grid)) for i in range(3)
    ]
    return ks, *spectra


def compensated_spectrum(ks, Es, eps, nu):
    """Kolmogorov-normalized spectrum for inertial-range validation:
    ``(k·η, E(k)·ε^(-2/3)·k^(5/3))`` with η = (ν³/ε)^(1/4). A resolved
    inertial range shows a plateau at C_K ≈ 1.5–1.6."""
    eta = (nu**3 / eps) ** 0.25
    return ks * eta, Es * eps ** (-2 / 3) * ks ** (5 / 3)


def dissipation_constant(eps, ell, u_rms):
    """Dissipation constant C_ε = ε·ℓ/u_rms³ (Rosales & Meneveau §II). `ell`
    must come from an *independent* lengthscale estimate (e.g.
    `integral_lengthscale` from the real-space autocorrelation) — computing
    ℓ as u_rms³/ε and feeding it back in makes C_ε ≡ 1 by construction."""
    return eps * ell / u_rms**3


def kolmogorov_scale(u_hat, grid, nu):
    """Kolmogorov length scale η = (ν³/ε)^(1/4)."""
    return (nu**3 / dissipation(u_hat, grid, nu)) ** 0.25


def taylor_microscale(u_hat, grid):
    """Taylor microscale λ = √(5E/Ω) (from ε = 15ν u′²/λ² with u′² = 2E/3)."""
    return jnp.sqrt(5 * energy(u_hat, grid) / enstrophy(u_hat, grid))


def taylor_reynolds(E, eps, nu):
    """Taylor-microscale Reynolds number Re_λ = u′λ/ν, with u′ = √(2E/3)
    and λ = √(10νE/ε). The scalar form is the single implementation shared
    by `taylor_reynolds_field` and postprocessing that only has a recorded
    (E, ε, ν) series."""
    u_p = jnp.sqrt(2 * E / 3)
    lam = jnp.sqrt(10 * nu * E / eps)
    return u_p * lam / nu


def taylor_reynolds_field(u_hat, grid, nu):
    return taylor_reynolds(
        energy(u_hat, grid), dissipation(u_hat, grid, nu), nu
    )


# --- energy budget ----------------------------------------------------------


def energy_budget(u_hat, grid, nu, forcing):
    """The three raw terms of the kinetic-energy budget dE/dt = P − ε at
    one instant, as a dict: energy `E`, dissipation `eps`, and the
    forcing's exact injected power `P` (see the forcing contract). Derived
    quantities (dE/dt, residual, cumulative integrals) are left to
    postprocessing — see `plot_energy_balance`."""
    return {
        "E": energy(u_hat, grid),
        "eps": dissipation(u_hat, grid, nu),
        "P": forcing.injection(u_hat, grid),
    }


def energy_budget_cb(state):
    """Callback form of `energy_budget`, unpacking the `State` tuple:

        Diagnostic(energy_budget_cb, every=10, path=..., name="energy_budget")
    """
    return energy_budget(
        state.u_hat, state.grid, state.params.nu, state.params.forcing
    )


# --- physical-space reductions ----------------------------------------------


def velocity_samples(u_hat, grid, component=0):
    """Physical-space values of one velocity component as a flat host
    vector, ready for a histogram/PDF or moment calculation. Costs one
    inverse transform of the full state."""
    u = to_physical(u_hat, grid)
    return np.asarray(u[component]).ravel()


def velocity_moments(samples):
    """Central moments of a velocity-component sample, as a dict with keys
    mean, variance, skewness, flatness. Flatness (kurtosis) is 3.0 for a
    Gaussian; skewness is 0 for a symmetric distribution."""
    samples = np.asarray(samples)
    m = samples.mean()
    c = samples - m
    v = np.mean(c**2)
    return {
        "mean": m,
        "variance": v,
        "skewness": np.mean(c**3) / v**1.5,
        "flatness": np.mean(c**4) / v**2,
    }


def longitudinal_autocorrelation(u_hat, grid, component=0):
    """Longitudinal autocorrelation f(r) = ⟨u₁(x)u₁(x+r·ê₁)⟩/⟨u₁²⟩ of one
    velocity component along its own axis (Pope §6.2), via Wiener–Khinchin:
    the inverse transform of the component's spectral power |û|² gives the
    (unnormalized) circular autocorrelation directly. Returned over
    r ∈ [0, Lx/2] (the periodic wrap folds the correlation back on itself
    beyond the half-box)."""
    a2 = jnp.abs(u_hat[component]) ** 2
    ntot = grid.Nx * grid.Ny * grid.Nz
    corr = (
        jnp.fft.irfftn(a2 + 0j, s=(grid.Nx, grid.Ny, grid.Nz), axes=(0, 1, 2))
        / ntot
    )
    n = grid.Nx // 2 + 1
    f = np.asarray(corr[:n, 0, 0], dtype=float)
    r = np.arange(n) * (grid.Lx / grid.Nx)
    if f[0] <= 0:
        return r, np.zeros_like(f)
    return r, f / f[0]


def integral_lengthscale(r, f):
    """Longitudinal integral scale L₁₁ = ∫₀^∞ f(r) dr (trapezoidal),
    truncated at the first zero-crossing of `f` to avoid integrating the
    (weakly negative, periodic-wrap) tail past where the correlation has
    physically decayed."""
    f = np.asarray(f)
    r = np.asarray(r)
    below = np.nonzero(f < 0)[0]
    n = (below[0] + 1) if below.size else len(f)
    if n < 2:
        return 0.0
    return float(np.trapezoid(f[:n], r[:n]))
