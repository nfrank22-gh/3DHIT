# Validation driver for LinearForcing datasets (Rosales & Meneveau, Phys.
# Fluids 17, 095106 (2005)) — a postprocessor, not a solver driver. It reads
# datasets already written by scripts/generate_dataset.py (one run per A
# value, via separate configs/*.toml files sharing N, ν, dt, and the initial
# condition), reproduces the single-run validation plots + stationary-state
# printout for each, and additionally overlays their stationary-averaged
# energy spectra into one comparison plot — the paper's Fig. 2 (spectra for
# different A, same ν).
#
#   uv run scripts/linear_forcing_validation.py
#
# Parameters are hardcoded below (mirrors scripts/run_linear_forcing.py)
# rather than read from a config, since this script sweeps one parameter
# (A) while holding everything else fixed. Only N and ν actually feed into
# the run-directory label.

import os
import sys

import h5py
import numpy as np

import hit3d as h3
from hit3d.report import read_grid, read_snapshot, step_keys, step_num

N = 64
nu = 4.491e-3

As = [0.0667, 0.1333, 0.2]  # paper's cases 1, 2, 3 (same ν, k0 = 2)

# Axis limits for every E(k)-vs-k plot, matched to the paper's Fig. 2 so
# runs overlay directly on it.
SPECTRUM_XLIMS = (1e0, 1e2)
SPECTRUM_YLIMS = (1e-8, 1e-1)

here = os.path.dirname(__file__)
output_path_file = os.path.join(here, "..", "configs", "output_path.txt")
if not os.path.isfile(output_path_file):
    sys.exit(f"missing {output_path_file} — create it with one line: the "
             "base results directory generate_dataset.py wrote these "
             "datasets into")
with open(output_path_file) as f:
    base_dir = f.read().strip()

g = h3.make_grid(N)


def stationary_spectrum(snapfile, window=1 / 3):
    """Stationary-window (last `window` fraction) average energy spectrum,
    reading snapshots directly from the FieldWriter file."""
    with h5py.File(snapfile, "r") as f:
        keys = step_keys(f)
        if not keys:
            sys.exit(f"no snapshots found in {snapfile}")
        gc, _ = read_grid(f, np.float32)
        ts = [float(f[f"{k}/t"][()]) for k in keys]
        cutoff = ts[-1] - window * (ts[-1] - ts[0])
        stat = [k for k, t in zip(keys, ts, strict=True) if t >= cutoff]
        ks = Es = None
        for i, k in enumerate(stat):
            kk, Ek = h3.energy_spectrum(read_snapshot(f, step_num(k)), gc)
            ks, Es = (kk, Ek) if i == 0 else (ks, Es + Ek)
        return ks, Es / len(stat)


comparison = []

for A in As:
    params = h3.NSParams(nu=nu, forcing=h3.LinearForcing(A=A))
    rundir = os.path.join(base_dir, h3.run_label(g, params))
    snapfile = os.path.join(rundir, "dataset.h5")
    if not os.path.isfile(snapfile):
        sys.exit(f"no dataset at {snapfile} — run generate_dataset.py for "
                 f"A = {A} first")

    print(f"== A = {A}  ({rundir}) ==")

    h3.plot_summary(snapfile, spectra_xlims=SPECTRUM_XLIMS,
                    spectra_ylims=SPECTRUM_YLIMS)
    h3.plot_slices(snapfile)
    h3.plot_energy_balance(snapfile)
    h3.plot_validation(snapfile, spectra_xlims=SPECTRUM_XLIMS,
                       spectra_ylims=SPECTRUM_YLIMS)

    series = h3.read_series(snapfile, "energy_budget")
    if series is None:
        sys.exit(f"no series/energy_budget group in {snapfile} — was this "
                 "dataset generated with generate_dataset.py?")
    idx = slice(2 * len(series["t"]) // 3, None)  # last third
    E_mean = float(np.mean(series["E"][idx]))
    u2 = float(np.mean(2 * series["E"][idx] / 3))  # u_rms² = 2E/3
    eps_mean = float(np.mean(series["eps"][idx]))

    run = h3.load_run(snapfile)
    with h5py.File(snapfile, "r") as f:
        u_final = read_snapshot(f, run.steps[-1])

    rlag, facf = h3.longitudinal_autocorrelation(u_final, run.grid)
    ell_corr = h3.integral_lengthscale(rlag, facf)
    ell_energy = u2**1.5 / eps_mean
    C_eps = h3.dissipation_constant(eps_mean, ell_corr, np.sqrt(u2))

    eta = (run.nu**3 / eps_mean) ** 0.25
    kmax = (run.grid.Nx // 2) * (2 * np.pi / run.grid.Lx)
    Re_lam = float(h3.taylor_reynolds(E_mean, eps_mean, run.nu))
    mom = h3.velocity_moments(h3.velocity_samples(u_final, run.grid))

    print(f"  u_rms^2                = {u2:.4g}")
    print(f"  eps                    = {eps_mean:.4g}")
    print(f"  A(t) = eps/3u_rms^2    = {eps_mean / (3 * u2):.4g}"
          f"   (imposed A = {A})")
    print(f"  l/L (= u_rms^3/eps)    = {ell_energy / run.grid.Lx:.3g}"
          "   (paper: ~0.19)")
    print(f"  l/L (autocorrelation)  = {ell_corr / run.grid.Lx:.3g}")
    print(f"  C_eps                  = {C_eps:.3g}")
    print(f"  k_max*eta              = {kmax * eta:.3g}   (want >~ 1)")
    print(f"  Re_lambda              = {Re_lam:.4g}")
    print(f"  skewness, flatness (u1, final snapshot) = "
          f"{mom['skewness']:.3g}, {mom['flatness']:.3g}   (Gaussian: 0, 3)")

    ks, Es = stationary_spectrum(snapfile)
    comparison.append((A, ks, Es))

# Fig. 2-style comparison: raw E(k) vs k, log-log, one curve per A.
# Categorical hues in fixed assignment order (validated set).
import matplotlib.pyplot as plt  # noqa: E402

colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]
fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("k")
ax.set_ylabel("E(k)")
ax.set_title(f"Stationary energy spectra vs A (ν = {nu:g})")
for i, (A, ks, Es) in enumerate(comparison):
    keep = Es > 0
    ax.plot(ks[keep], Es[keep], color=colors[i % len(colors)], lw=2,
            label=f"A = {A}")
ax.legend(loc="lower left", frameon=False)
ax.set_xlim(*SPECTRUM_XLIMS)
ax.set_ylim(*SPECTRUM_YLIMS)
comparison_path = os.path.join(base_dir, "spectra_comparison.png")
fig.savefig(comparison_path, dpi=150)
print("comparison spectrum written to", os.path.abspath(comparison_path))
