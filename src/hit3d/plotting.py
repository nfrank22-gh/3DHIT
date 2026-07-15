"""Plotting entry points (matplotlib, optional dependency).

All functions are self-contained postprocessors: they read everything they
need (grid geometry, ν, snapshots, dense series) from the HDF5 file written
by `FieldWriter` / `save_series`. matplotlib is imported lazily inside each
function, so the solver core never depends on it — install with
``uv sync --extra plots``.

Palette: one blue for single-series panels; a light→dark single-hue ramp
for time-ordered spectra; sequential viridis for magnitude fields and a
diverging map with neutral midpoint for signed components. The categorical
hues (fixed assignment, validated as a set for CVD separation and surface
contrast) serve the multi-series energy-balance panels.
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from .diagnostics import (
    compensated_spectrum,
    component_spectra,
    dissipation,
    energy,
    energy_spectrum,
    taylor_reynolds,
    velocity_moments,
    velocity_samples,
)
from .grid import to_physical
from .report import read_grid, read_series, read_snapshot, step_keys, step_num

__all__ = [
    "plot_summary",
    "plot_energy_balance",
    "plot_validation",
    "plot_slices",
]

LINE_BLUE = "#2563eb"
GUIDE_GRAY = "#6b7280"
CAT_RED = "#dc2626"
CAT_GREEN = "#059669"
CAT_VIOLET = "#7c3aed"
CAT_AMBER = "#d97706"
TIME_RAMP = ("#93c5fd", "#1e3a8a")  # light → dark, ordered by time


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _time_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("time", TIME_RAMP)


def _outdir(path, outdir):
    outdir = outdir or os.path.dirname(path) or "."
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _state_dtype(f):
    c = f[f"{step_keys(f)[0]}/u_hat"].dtype
    return np.float64 if c == np.complex128 else np.float32


# --- summary + spectra -------------------------------------------------------


def plot_summary(path, *, outdir=None, spectra_xlims=None, spectra_ylims=None):
    """Read a `FieldWriter` snapshot file and write two figures:

    - ``summary.png`` — kinetic energy, dissipation, and Re_λ over time
      (the ν-dependent panels are skipped if the file has no viscosity
      metadata). When the file also carries a dense
      ``series/energy_budget`` group, the time panels use it instead of
      the sparse snapshots;
    - ``spectra.png`` — log-log energy spectra of every snapshot, colored
      by time, with a k^(-5/3) reference slope. Axis limits auto-scale by
      default; pass ``spectra_xlims``/``spectra_ylims = (lo, hi)`` to fix
      them (e.g. to match a reference plot from a paper).

    Returns the paths of the files written.
    """
    plt = _plt()
    outdir = _outdir(path, outdir)
    written = []

    with h5py.File(path, "r") as f:
        keys = step_keys(f)
        if not keys:
            raise ValueError(f"no snapshots found in {path}")
        grid, nu = read_grid(f, _state_dtype(f))

        snap_ts, Es, eps_s, Res, spectra = [], [], [], [], []
        ks = None
        for key in keys:
            u_hat = read_snapshot(f, step_num(key))
            snap_ts.append(float(f[f"{key}/t"][()]))
            Es.append(float(energy(u_hat, grid)))
            ks, Ek = energy_spectrum(u_hat, grid)
            spectra.append(Ek)
            if nu is not None:
                eps_s.append(float(dissipation(u_hat, grid, nu)))
                Res.append(float(taylor_reynolds(Es[-1], eps_s[-1], nu)))

        # prefer the dense series saved by `save_series` for the time
        # panels (the snapshot-derived values stay as the fallback;
        # spectra are snapshot-only either way).
        series = read_series(f, "energy_budget")
        dense = series is not None
        ts = np.asarray(snap_ts)
        if dense:
            ts = series["t"].astype(float)
            Es = series["E"].astype(float)
            eps_s = series["eps"].astype(float)
            Res = (
                []
                if nu is None
                else [float(taylor_reynolds(E, e, nu))
                      for E, e in zip(Es, eps_s, strict=True)]
            )

    def draw(ax, ys):
        if dense:
            ax.plot(ts, ys, color=LINE_BLUE, lw=2)
        else:
            ax.plot(ts, ys, "o-", color=LINE_BLUE, lw=2, ms=4)

    # summary.png — one measure per panel, shared time axis
    npanel = 1 if nu is None else 3
    fig, axes = plt.subplots(
        npanel, 1, figsize=(6.4, 2.3 * npanel), sharex=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("E")
    axes[0].set_title("Kinetic energy")
    draw(axes[0], Es)
    if nu is None:
        axes[0].set_xlabel("t")
    else:
        axes[1].set_ylabel("ε")
        axes[1].set_title(f"Dissipation rate  (ν = {nu:g})")
        draw(axes[1], eps_s)
        axes[2].set_ylabel("Re_λ")
        axes[2].set_xlabel("t")
        axes[2].set_title("Taylor-microscale Reynolds number")
        draw(axes[2], Res)
    p = os.path.join(outdir, "summary.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # spectra.png — all snapshots, single-hue ramp by time
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title("Energy spectra")
    cmap = _time_cmap()
    t0, t1 = snap_ts[0], snap_ts[-1]
    for t, Ek in zip(snap_ts, spectra, strict=True):
        c = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        Ep = np.where(Ek > 0, Ek, np.nan)
        ax.plot(ks, Ep, color=cmap(c), lw=2)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig.colorbar(
        ScalarMappable(Normalize(t0, t1), cmap), ax=ax, label="t"
    )
    # keep the view on the developed spectra (the initial condition's steep
    # tail would otherwise stretch the axis over ~40 decades), unless the
    # caller pinned the range
    if spectra_ylims is None:
        emax = max(s.max() for s in spectra)
        ax.set_ylim(emax * 1e-12, emax * 5)
    else:
        ax.set_ylim(*spectra_ylims)
    if spectra_xlims is not None:
        ax.set_xlim(*spectra_xlims)
    # k^(-5/3) reference, anchored above the last (most developed) spectrum
    nk = len(ks)
    i0 = min(3, nk - 1)
    if spectra[-1][i0] > 0:
        C = 3 * spectra[-1][i0] * ks[i0] ** (5 / 3)
        kg = ks[max(1, i0 - 2): min(nk, 4 * (i0 + 1))]
        ax.plot(kg, C * kg ** (-5 / 3), "--", color=GUIDE_GRAY, lw=2)
        ax.annotate(
            "k$^{-5/3}$", (kg[-1], C * kg[-1] ** (-5 / 3)),
            color=GUIDE_GRAY, ha="left", va="bottom",
        )
    p = os.path.join(outdir, "spectra.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# --- energy balance ----------------------------------------------------------


def _ddt(t, y):
    """Central differences on a (possibly nonuniform) time grid, one-sided
    at the ends. Needs ≥ 2 samples."""
    d = np.empty_like(y)
    d[0] = (y[1] - y[0]) / (t[1] - t[0])
    d[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    d[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    return d


def _cumtrapz(t, y):
    """Cumulative trapezoidal integral of y(t), starting at 0."""
    out = np.zeros_like(y)
    out[1:] = np.cumsum((t[1:] - t[:-1]) * (y[1:] + y[:-1]) / 2)
    return out


def plot_energy_balance(path, *, outdir=None):
    """Write ``energy_balance.png`` from the dense ``series/energy_budget``
    group. Two panels sharing the time axis:

    - **budget terms** — dE/dt (central differences of the recorded E),
      −ε, the injected power P, and the residual dE/dt + ε − P (zero for a
      perfect balance; its magnitude measures time-integration + sampling
      error);
    - **cumulative** — E(t) against the reconstruction E(0) − ∫ε dt + ∫P dt
      (trapezoidal), the integrated form of the same budget.

    Errors if the file has no ``series/energy_budget`` group.
    """
    plt = _plt()
    outdir = _outdir(path, outdir)

    series = read_series(path, "energy_budget")
    if series is None:
        raise ValueError(
            f"no series/energy_budget group in {path} — record it with "
            "Diagnostic(energy_budget_cb, ..., path=<this file>, "
            "name='energy_budget') so evolve persists it automatically"
        )
    ts = series["t"].astype(float)
    Es = series["E"].astype(float)
    eps_s = series["eps"].astype(float)
    Ps = series["P"].astype(float)
    if len(ts) < 2:
        raise ValueError(f"energy_budget series in {path} has fewer than 2 samples")

    dEdt = _ddt(ts, Es)
    resid = dEdt + eps_s - Ps
    recon = Es[0] - _cumtrapz(ts, eps_s) + _cumtrapz(ts, Ps)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 5.6), sharex=True, layout="constrained"
    )
    ax1.set_ylabel("dE/dt")
    ax1.set_title("Energy budget  dE/dt = P − ε")
    ax1.axhline(0.0, color=GUIDE_GRAY, ls="--", lw=1)
    ax1.plot(ts, dEdt, color=LINE_BLUE, lw=2, label="dE/dt")
    ax1.plot(ts, -eps_s, color=CAT_RED, lw=2, label="−ε")
    ax1.plot(ts, Ps, color=CAT_GREEN, lw=2, label="P")
    ax1.plot(ts, resid, color=CAT_VIOLET, lw=2, label="residual")
    ax1.legend(loc="lower left", frameon=False)

    ax2.set_ylabel("E")
    ax2.set_xlabel("t")
    ax2.set_title("Cumulative budget")
    ax2.plot(ts, Es, color=LINE_BLUE, lw=2, label="E(t)")
    ax2.plot(ts, recon, "--", color=CAT_AMBER, lw=2,
             label="E(0) − ∫ε dt + ∫P dt")
    ax2.legend(loc="upper right", frameon=False)

    p = os.path.join(outdir, "energy_balance.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]


# --- validation against a reference (e.g. a paper figure) --------------------


def plot_validation(path, *, outdir=None, window=1 / 3,
                    spectra_xlims=None, spectra_ylims=None):
    """Write three figures, each averaged over snapshots whose time falls
    in the last `window` fraction of the run (a statistically-stationary
    window for a forced run; for a decaying run this is just its final
    segment):

    - ``compensated_spectrum.png`` — E(k)·ε^(-2/3)·k^(5/3) vs k·η, with a
      horizontal reference at the Kolmogorov constant C_K ≈ 1.5;
    - ``isotropy_spectra.png`` — the three component spectra E₁₁, E₂₂,
      E₃₃; they should coincide throughout the resolved range;
    - ``velocity_pdf.png`` — standardized PDF of one velocity component
      against a unit Gaussian, with skewness/flatness in the title.

    Errors if the file has no viscosity metadata (needed for the
    compensated spectrum).
    """
    plt = _plt()
    outdir = _outdir(path, outdir)
    written = []

    with h5py.File(path, "r") as f:
        keys = step_keys(f)
        if not keys:
            raise ValueError(f"no snapshots found in {path}")
        grid, nu = read_grid(f, _state_dtype(f))
        if nu is None:
            raise ValueError(
                f"plot_validation needs viscosity metadata (none found in {path})"
            )

        ts = [float(f[f"{k}/t"][()]) for k in keys]
        cutoff = ts[-1] - window * (ts[-1] - ts[0])
        stat_keys = [k for k, t in zip(keys, ts, strict=True) if t >= cutoff]

        ks = Es = E11 = E22 = E33 = None
        eps = 0.0
        samples = []
        for i, k in enumerate(stat_keys):
            u_hat = read_snapshot(f, step_num(k))
            kk, Ek = energy_spectrum(u_hat, grid)
            _, e11, e22, e33 = component_spectra(u_hat, grid)
            if i == 0:
                ks, Es, E11, E22, E33 = kk, Ek, e11, e22, e33
            else:
                Es = Es + Ek
                E11, E22, E33 = E11 + e11, E22 + e22, E33 + e33
            eps += float(dissipation(u_hat, grid, nu))
            samples.append(velocity_samples(u_hat, grid))
        n = len(stat_keys)
        Es, E11, E22, E33 = Es / n, E11 / n, E22 / n, E33 / n
        eps /= n
        samples = np.concatenate(samples)

    # compensated_spectrum.png — Kolmogorov normalization, stationary avg
    keta, Ecomp = compensated_spectrum(ks, Es, eps, nu)
    keep = Ecomp > 0
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k·η")
    ax.set_ylabel("E(k)·ε$^{-2/3}$·k$^{5/3}$")
    ax.set_title("Compensated spectrum (stationary window)")
    ax.plot(keta[keep], Ecomp[keep], "o-", color=LINE_BLUE, lw=2, ms=4)
    ax.axhline(1.5, color=GUIDE_GRAY, ls="--")
    ax.annotate("C$_K$ ≈ 1.5", (keta[keep][-1], 1.5), color=GUIDE_GRAY,
                ha="right", va="bottom")
    p = os.path.join(outdir, "compensated_spectrum.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # isotropy_spectra.png — component spectra, should coincide
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("E$_{ii}$(k)")
    ax.set_title("Component spectra (isotropy check)")
    for Ei, lbl, color in ((E11, "E$_{11}$", CAT_RED),
                           (E22, "E$_{22}$", CAT_GREEN),
                           (E33, "E$_{33}$", CAT_VIOLET)):
        keep = Ei > 0
        ax.plot(ks[keep], Ei[keep], color=color, lw=2, label=lbl)
    if spectra_xlims is not None:
        ax.set_xlim(*spectra_xlims)
    if spectra_ylims is not None:
        ax.set_ylim(*spectra_ylims)
    ax.legend(loc="lower left", frameon=False)
    p = os.path.join(outdir, "isotropy_spectra.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # velocity_pdf.png — standardized PDF vs unit Gaussian
    mom = velocity_moments(samples)
    z = (samples - mom["mean"]) / np.sqrt(mom["variance"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.set_yscale("log")
    ax.set_xlabel("(u₁ − ⟨u₁⟩)/u₁′")
    ax.set_ylabel("PDF")
    ax.set_title(
        f"Velocity PDF  (skewness = {mom['skewness']:g}, "
        f"flatness = {mom['flatness']:g})"
    )
    ax.hist(z, bins=60, density=True, color=LINE_BLUE, alpha=0.5,
            label="measured")
    xs = np.linspace(-6, 6, 200)
    ax.plot(xs, np.exp(-(xs**2) / 2) / np.sqrt(2 * np.pi), "--",
            color=GUIDE_GRAY, lw=2, label="Gaussian")
    ax.legend(loc="upper right", frameon=False)
    p = os.path.join(outdir, "velocity_pdf.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# --- velocity slices -----------------------------------------------------------

# plane -> (in-plane dims, cut dim, axis labels); dims index the physical
# field's spatial axes (0 = x, 1 = y, 2 = z)
_PLANES = {
    "xy": ((0, 1), 2, ("x", "y")),
    "xz": ((0, 2), 1, ("x", "z")),
    "yz": ((1, 2), 0, ("y", "z")),
}


def plot_slices(path, *, steps="auto", component="mag", plane="xy",
                index=None, outdir=None):
    """Write ``slices.png``: 2D cuts of the velocity field for a few
    snapshots side by side with a shared color scale (so decay is visible
    across panels).

    - ``steps``     — snapshot step numbers to plot, or "auto"
      (first/middle/last)
    - ``component`` — "mag" (velocity magnitude, default) or "u1"/"u2"/"u3"
    - ``plane``     — "xy", "xz", or "yz"
    - ``index``     — grid index of the cut along the remaining axis
      (default: middle)
    """
    if plane not in _PLANES:
        raise ValueError(f"plane must be xy, xz, or yz (got {plane})")
    if component not in ("mag", "u1", "u2", "u3"):
        raise ValueError(f"component must be mag, u1, u2, or u3 (got {component})")
    plt = _plt()
    outdir = _outdir(path, outdir)

    with h5py.File(path, "r") as f:
        keys = step_keys(f)
        if not keys:
            raise ValueError(f"no snapshots found in {path}")
        grid, _ = read_grid(f, _state_dtype(f))

        available = [step_num(k) for k in keys]
        if steps == "auto":
            picked = sorted({0, (len(keys) - 1) // 2, len(keys) - 1})
        else:
            picked = []
            for s in steps:
                if s not in available:
                    raise ValueError(f"step {s} not in file (has {available})")
                picked.append(available.index(s))

        dims, cutdim, (xlab, ylab) = _PLANES[plane]
        Ns = (grid.Nx, grid.Ny, grid.Nz)
        Ls = (grid.Lx, grid.Ly, grid.Lz)
        icut = Ns[cutdim] // 2 if index is None else index

        slices, ts = [], []
        for i in picked:
            u_hat = read_snapshot(f, available[i])
            u = np.asarray(to_physical(u_hat, grid))
            if component == "mag":
                field = np.sqrt(np.sum(u**2, axis=0))
            else:
                field = u[int(component[-1]) - 1]
            sl = np.take(field, icut, axis=cutdim)
            slices.append(np.asarray(sl, dtype=float))
            ts.append(float(f[f"{keys[i]}/t"][()]))

    # shared color scale across panels so the decay stays visible
    if component == "mag":
        vmin, vmax = 0.0, max(s.max() for s in slices)
        cmap = "viridis"
    else:
        m = max(np.abs(s).max() for s in slices)
        vmin, vmax = -m, m
        cmap = "RdBu_r"

    n = len(slices)
    fig, axes = plt.subplots(
        1, n, figsize=(3.0 * n + 1.0, 3.4), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    extent = (0, Ls[dims[0]], 0, Ls[dims[1]])
    im = None
    for j, (ax, sl, t) in enumerate(zip(axes, slices, ts, strict=True)):
        im = ax.imshow(sl.T, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"t = {t:g}")
        ax.set_xlabel(xlab)
        if j == 0:
            ax.set_ylabel(ylab)
        else:
            ax.set_yticklabels([])
    name = "|u|" if component == "mag" else component
    fig.colorbar(im, ax=axes, label=name)
    fig.suptitle(f"{name} on the {plane} mid-plane")
    p = os.path.join(outdir, "slices.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return [p]
