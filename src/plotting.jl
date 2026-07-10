# Plotting entry points — implemented by the HIT3DMakieExt package extension,
# which activates when any Makie backend is loaded (`using CairoMakie` /
# `using GLMakie`). The package itself carries no plotting dependencies.
#
# Both functions are self-contained postprocessors: they read everything they
# need (grid geometry, ν, snapshots) from the .jld2 file written by
# `FieldWriter`.

"""
    plot_summary(path; outdir = dirname(path)) -> Vector{String}

Read a `FieldWriter` snapshot file and write two figures to `outdir`:

- `summary.png` — kinetic energy, dissipation, and Re_λ over time (the
  ν-dependent panels are skipped if the file has no viscosity metadata).
  When the file also carries a dense `series/energy_budget` group (see
  `save_series`), the time panels use it instead of the sparse snapshots;
- `spectra.png` — log-log energy spectra of every snapshot, colored by time,
  with a k^(-5/3) reference slope.

Returns the paths of the files written. Requires a Makie backend:
`using CairoMakie` (headless/PNG) or `using GLMakie`.
"""
function plot_summary end

"""
    plot_slices(path; steps = :auto, component = :mag, plane = :xy,
                index = nothing, outdir = dirname(path)) -> Vector{String}

Write `slices.png`: 2D cuts of the velocity field for a few snapshots side by
side with a shared color scale (so decay is visible across panels).

- `steps`     — snapshot step numbers to plot, or `:auto` (first/middle/last)
- `component` — `:mag` (velocity magnitude, default) or `:u1`/`:u2`/`:u3`
- `plane`     — `:xy`, `:xz`, or `:yz`
- `index`     — grid index of the cut along the remaining axis (default: middle)

Returns the paths of the files written. Requires a Makie backend, like
[`plot_summary`](@ref).
"""
function plot_slices end

"""
    plot_energy_balance(path; outdir = dirname(path)) -> Vector{String}

Write `energy_balance.png` from the dense `series/energy_budget` group saved
by `save_series` (see `energy_budget`). Two panels sharing the time axis:

- **budget terms** — dE/dt (central differences of the recorded E), −ε, the
  injected power P, and the residual dE/dt + ε − P (zero for a perfect
  balance; its magnitude measures time-integration + sampling error);
- **cumulative** — E(t) against the reconstruction E(0) − ∫ε dt + ∫P dt
  (trapezoidal), the integrated form of the same budget.

Errors if the file has no `series/energy_budget` group. Returns the paths of
the files written. Requires a Makie backend, like [`plot_summary`](@ref).
"""
function plot_energy_balance end

const _NEEDS_MAKIE = """
requires a Makie backend to be loaded first, e.g.:
    using CairoMakie   # headless, writes PNGs
    using GLMakie      # interactive
"""

plot_summary(args...; kwargs...) = error("plot_summary ", _NEEDS_MAKIE)
plot_slices(args...; kwargs...) = error("plot_slices ", _NEEDS_MAKIE)
plot_energy_balance(args...; kwargs...) =
    error("plot_energy_balance ", _NEEDS_MAKIE)
