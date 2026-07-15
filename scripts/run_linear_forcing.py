# Example driver: linearly forced HIT with plots and a stationary-state
# validation printout.
#
# Lundgren's linear forcing f = A·u, following Rosales & Meneveau,
# Phys. Fluids 17, 095106 (2005), Sec. II. Constant A imposes an inverse
# turnover timescale (τ = 1/(3A)); the flow converges to a stationary state
# with ε = 3A·u_rms² and integral scale ℓ = u_rms³/ε ≈ 0.19·L regardless of
# the initial spectrum (their Figs. 8 and 9).
#
# Scaled-down version of their case 2c (128³, ν = 4.491e-3): same A and
# initial spectrum shape, but 64³ with ν raised to keep the dissipation
# range resolved (at stationarity ε ≈ A³L², so η = (ν³/ε)^¼ needs
# kmax·η ≳ 1).
#
#   uv run scripts/run_linear_forcing.py

import os

import jax
import numpy as np

import hit3d as h3

N = 64
nu = 4.491e-3
A = 0.0667  # imposed inverse timescale, τ = 1/(3A) ≈ 5
dt = 5e-3
tau = 1 / (3 * A)
nsteps = round(25 * tau / dt)  # ~25 turnover times

g = h3.make_grid(N)  # float32, L = 2π

# Initial condition: Eq. (9) of the paper — solenoidal random field with
# E(k) = 16·√(2/π)·(u₀²/k₀⁵)·k⁴·exp(−2k²/k₀²), peaked at k₀ (their type-c
# initial condition, k₀ = 2). u₀ is set near the expected stationary u_rms
# (u_rms² = ε/(3A) with ε ≈ A³L²) to shorten the transient.
k0, u0 = 2.0, 0.5
u_hat = h3.random_field(
    jax.random.key(0), g,
    spectrum=lambda k: 16 * np.sqrt(2 / np.pi) * u0**2 / k0**5
    * k**4 * jax.numpy.exp(-2 * k**2 / k0**2),
)

rhs = h3.make_ns_rhs(g)
params = h3.NSParams(nu=nu, forcing=h3.LinearForcing(A=A))

rundir = os.path.join(
    os.path.dirname(__file__), "..", "results", h3.run_label(g, params)
)
snapfile = os.path.join(rundir, "linear_forcing.h5")

budget = h3.Diagnostic(
    h3.energy_budget_cb, every=10, path=snapfile, name="energy_budget"
)
# dense enough that ~10 full-field snapshots land in the last third of the
# run (needed for plot_validation's stationary-window averages); the scalar
# energy budget above is far denser but only carries (E, ε, P).
writer = h3.FieldWriter(snapfile, every=nsteps // 30)

u_hat = h3.evolve(
    rhs, u_hat, params, g, dt, nsteps, callbacks=(budget, writer),
    progress=True,
)

# spectra.png axes pinned to the paper's Fig. 6 for a direct visual
# comparison. Our 64³ run's Nyquist (k ≈ 32) won't reach the right edge of
# that range — expected, matches how the paper's lower-Re curve falls
# short of it too.
h3.plot_summary(snapfile, spectra_xlims=(1, 100), spectra_ylims=(1e-8, 1e-1))
h3.plot_energy_balance(snapfile)
h3.plot_slices(snapfile)
# -> compensated_spectrum.png, isotropy_spectra.png, velocity_pdf.png,
# each averaged over the last third of the run (same window as below)
h3.plot_validation(snapfile)

# Stationary-state check against the paper: averages over the final third.
# Expect A(t) = ε/(3u_rms²) ≈ imposed A (Fig. 8) and ℓ/L ≈ 0.19 (Fig. 9).
stat = budget.values[2 * len(budget.values) // 3:]
E_mean = np.mean([v["E"] for v in stat])
u2 = np.mean([2 * v["E"] / 3 for v in stat])  # u_rms² = 2E/3
eps_mean = np.mean([v["eps"] for v in stat])
ell_energy = u2**1.5 / eps_mean  # ℓ = u_rms³/ε (assumes C_ε = 1)

# Independent real-space cross-check: longitudinal autocorrelation of u₁
# on the final snapshot, integrated to L₁₁ (Pope §6.2). Unlike ell_energy,
# this doesn't presuppose C_ε = 1, so ε·ℓ_corr/u_rms³ is a genuine check
# of the dissipation constant rather than being 1 by construction.
rlag, facf = h3.longitudinal_autocorrelation(u_hat, g)
ell_corr = h3.integral_lengthscale(rlag, facf)
C_eps = h3.dissipation_constant(eps_mean, ell_corr, np.sqrt(u2))

eta = (nu**3 / eps_mean) ** 0.25
kmax = (N // 2) * (2 * np.pi / g.Lx)
Re_lam = float(h3.taylor_reynolds(E_mean, eps_mean, nu))
mom = h3.velocity_moments(h3.velocity_samples(u_hat, g))

print(f"stationary window (last third of {25 * tau:.1f} time units):")
print(f"  u_rms^2                = {u2:.4g}")
print(f"  eps                    = {eps_mean:.4g}")
print(f"  A(t) = eps/3u_rms^2    = {eps_mean / (3 * u2):.4g}   (imposed A = {A})")
print(f"  l/L (= u_rms^3/eps)    = {ell_energy / g.Lx:.3g}   (paper: ~0.19)")
print(f"  l/L (autocorrelation)  = {ell_corr / g.Lx:.3g}")
print(f"  C_eps                  = {C_eps:.3g}")
print(f"  k_max*eta              = {kmax * eta:.3g}   (want >~ 1)")
print(f"  Re_lambda              = {Re_lam:.4g}   (compare paper case 2c)")
print(f"  skewness, flatness (u1, final snapshot) = "
      f"{mom['skewness']:.3g}, {mom['flatness']:.3g}   (Gaussian: 0, 3)")
print("results written to", os.path.abspath(rundir))
