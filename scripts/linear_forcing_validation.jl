# Validation driver for LinearForcing datasets (Rosales & Meneveau, Phys.
# Fluids 17, 095106 (2005)) — a postprocessor, not a solver driver. It reads
# datasets already written by scripts/generate_dataset.jl (one run per A
# value, via separate configs/*.toml files sharing N, ν, dt, and the initial
# condition), reproduces the single-run validation plots + stationary-state
# printout for each, and additionally overlays their stationary-averaged
# energy spectra into one comparison plot — the paper's Fig. 2 (spectra for
# different A, same ν).
#
# Run from the repo root with the scripts environment:
#   julia --project=scripts scripts/linear_forcing_validation.jl
#
# Parameters are hardcoded below (mirrors scripts/run_linear_forcing.jl)
# rather than read from a config file, since this script sweeps one
# parameter (A) while holding everything else fixed. Only N and ν actually
# feed into the run-directory label (see labels.jl); dt/k0/u0/warmup_time/
# save_dt/backend are recorded here for documentation and must match what
# generate_dataset.jl was actually run with for these datasets to exist.

using HIT3D
using HIT3D: _stepkeys, _read_grid, _state_T, stepkey
using HIT3D.Integrators: jldopen
using CairoMakie
using Statistics: mean

N  = 64
ν  = 4.491f-3
dt = 5f-3               # documentation only — doesn't affect the label
k0 = 2f0                # documentation only
u0 = 0.5f0               # documentation only
warmup_time = 10f0       # documentation only
save_dt = 2.5f0          # documentation only

As = [0.0667f0, 0.1333f0, 0.2f0]   # paper's cases 1, 2, 3 (same ν, k0 = 2)

output_path_file = joinpath(@__DIR__, "..", "configs", "output_path.txt")
isfile(output_path_file) ||
    error("missing $output_path_file — create it with one line: the base " *
          "results directory generate_dataset.jl wrote these datasets into")
base_dir = strip(read(output_path_file, String))

g = Grid(N)
scheme = RK4(spectral_state(g))   # only used for its label ("RK4")

"""Stationary-window (last `window` fraction) average energy spectrum,
reading snapshots directly from the FieldWriter file at `snapfile`."""
function stationary_spectrum(snapfile; window = 1 // 3)
    jldopen(snapfile, "r") do file
        keys_ = _stepkeys(file)
        isempty(keys_) && error("no snapshots found in $snapfile")
        gc, _ = _read_grid(file, _state_T(file))
        ts = Float64[file[k * "/t"] for k in keys_]
        t0, t1 = first(ts), last(ts)
        cutoff = t1 - window * (t1 - t0)
        stat_keys = [k for (k, t) in zip(keys_, ts) if t >= cutoff]
        ks = Float64[]
        Es = Float64[]
        for (i, k) in enumerate(stat_keys)
            û = file[k * "/û"]
            kk, Ek = energy_spectrum(û, gc)
            if i == 1
                ks, Es = Float64.(kk), Float64.(Ek)
            else
                Es .+= Ek
            end
        end
        Es ./= length(stat_keys)
        return ks, Es
    end
end

comparison_spectra = Vector{Tuple{Float32, Vector{Float64}, Vector{Float64}}}()

for A in As
    forcing = LinearForcing(A)
    r = NavierStokes(g; ν, forcing)   # scratch-only; used for label(r)
    rundir = joinpath(base_dir, join((label(g), label(r), label(scheme)), "_"))
    snapfile = joinpath(rundir, "dataset.jld2")
    isfile(snapfile) ||
        error("no dataset at $snapfile — run generate_dataset.jl for A = $A first")

    println("== A = ", A, "  (", rundir, ") ==")

    plot_summary(snapfile)
    plot_slices(snapfile)
    plot_energy_balance(snapfile)
    plot_validation(snapfile)

    series = read_series(snapfile, "energy_budget")
    series === nothing &&
        error("no series/energy_budget group in $snapfile — was this " *
              "dataset generated with the amended generate_dataset.jl?")
    len = length(series.t)
    idx = (2 * len ÷ 3):len       # last third, same window as run_linear_forcing.jl
    Ē  = mean(series.E[idx])
    u2 = mean(2 .* series.E[idx] ./ 3)      # u_rms² = 2E/3
    ε̄  = mean(series.ε[idx])

    run = load_run(snapfile)
    last_step = last(run.steps)
    û_final = jldopen(file -> file[stepkey(last_step) * "/û"], snapfile, "r")

    rlag, facf = longitudinal_autocorrelation(û_final, run.grid)
    ℓ_corr = Float64(integral_lengthscale(rlag, facf))
    ℓ_energy = u2^1.5 / ε̄
    Cε = dissipation_constant(ε̄, ℓ_corr, sqrt(u2))

    η    = (Float64(run.ν)^3 / ε̄)^(1 / 4)
    kmax = (run.grid.Nx ÷ 2) * (2 * Float64(π) / Float64(run.grid.Lx))
    Reλ  = taylor_reynolds(Ē, ε̄, Float64(run.ν))
    mom  = velocity_moments(velocity_samples(û_final, run.grid))

    println("  u_rms²          = ", round(u2; sigdigits = 4))
    println("  ε               = ", round(ε̄; sigdigits = 4))
    println("  A(t) = ε/3u_rms² = ", round(ε̄ / 3u2; sigdigits = 4),
            "   (imposed A = ", A, ")")
    println("  ℓ/L (= u_rms³/ε) = ", round(ℓ_energy / Float64(run.grid.Lx); sigdigits = 3),
            "   (paper: ≈ 0.19)")
    println("  ℓ/L (real-space autocorrelation) = ",
            round(ℓ_corr / Float64(run.grid.Lx); sigdigits = 3))
    println("  C_ε = ε·ℓ_corr/u_rms³ = ", round(Cε; sigdigits = 3))
    println("  k_max·η         = ", round(kmax * η; sigdigits = 3),
            "   (want ≳ 1 for a resolved dissipation range)")
    println("  Re_λ            = ", round(Reλ; sigdigits = 4))
    println("  skewness, flatness (u₁, final snapshot) = ",
            round(mom.skewness; sigdigits = 3), ", ",
            round(mom.flatness; sigdigits = 3), "   (Gaussian: 0, 3)")

    ks, Es = stationary_spectrum(snapfile)
    push!(comparison_spectra, (A, ks, Es))
end

# Fig. 2-style comparison: raw E(k) vs k, log-log, one curve per A.
colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]
fig = Figure(size = (640, 460))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
         xlabel = "k", ylabel = "E(k)",
         title = "Stationary energy spectra vs A (ν = $ν)")
for (i, (A, ks, Es)) in enumerate(comparison_spectra)
    keep = Es .> 0
    lines!(ax, ks[keep], Es[keep]; color = colors[mod1(i, length(colors))],
          linewidth = 2, label = "A = $A")
end
axislegend(ax; position = :lb, framevisible = false)
comparison_path = joinpath(base_dir, "spectra_comparison.png")
save(comparison_path, fig)
println("comparison spectrum written to ", abspath(comparison_path))
