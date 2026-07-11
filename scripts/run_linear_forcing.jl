# Example driver: linearly forced HIT (Float32, CPU or GPU) with plots.
#
# Lundgren's linear forcing f = A·u, following Rosales & Meneveau,
# Phys. Fluids 17, 095106 (2005), Sec. II. Constant A imposes an inverse
# turnover timescale (τ = 1/(3A)); the flow converges to a stationary state
# with ε = 3A·u_rms² and integral scale ℓ = u_rms³/ε ≈ 0.19·L regardless of
# the initial spectrum (their Figs. 8 and 9).
#
# Scaled-down version of their case 2c (128³, ν = 4.491e-3): same A and
# initial spectrum shape, but 64³ with ν raised to keep the dissipation
# range resolved (at stationarity ε ≈ A³L², so η = (ν³/ε)^¼ needs kmax·η ≳ 1).
#
# Run from the repo root with the scripts environment:
#   julia --project=scripts scripts/run_linear_forcing.jl

using HIT3D
using CairoMakie   # loads Makie → activates HIT3D's plotting extension
using Statistics: mean

# Backend: :cpu, :cuda (NVIDIA), or :metal (Apple). Everything downstream is
# backend-agnostic — the choice only sets the array type the Grid (and hence
# state, scratch, and stage buffers) is built on. On an NVIDIA machine, add
# CUDA to the scripts environment first:
#   julia --project=scripts -e 'using Pkg; Pkg.add("CUDA")'
backend = :metal

if backend === :cuda
    using CUDA
    ArrayType = CuArray
elseif backend === :metal
    using Metal
    ArrayType = MtlArray
else
    ArrayType = Array
end

N  = 64
ν  = 4.491f-3
A  = 0.0667f0      # imposed inverse timescale, τ = 1/(3A) ≈ 2.5
dt = 5f-3
τ  = 1 / (3A)
nsteps = round(Int, 25τ / dt)     # ~25 turnover times

g = Grid(N; ArrayType)            # Float32 (Metal has no Float64), L = 2π
û = spectral_state(g)

# Initial condition: Eq. (9) of the paper — solenoidal random field with
# E(k) = 16·√(2/π)·(u₀²/k₀⁵)·k⁴·exp(−2k²/k₀²), peaked at k₀ (their type-c
# initial condition, k₀ = 2). u₀ is set near the expected stationary u_rms
# (u_rms² = ε/(3A) with ε ≈ A³L²) to shorten the transient.
k0 = 2f0
u0 = 0.5f0
random_field!(û, g; spectrum = k -> 16 * sqrt(2 / π) * u0^2 / k0^5 *
                                    k^4 * exp(-2k^2 / k0^2))

r      = NavierStokes(g; ν, forcing = LinearForcing(A))
scheme = RK4(û)

rundir = joinpath(@__DIR__, "..", "results",
                  join((label(g), label(r), label(scheme)), "_"))
snapfile = joinpath(rundir, "linear_forcing.jld2")

budget = Diagnostic(energy_budget; every = 10,
                    valuetype = @NamedTuple{E::Float64, ε::Float64,
                                            P::Float64},
                    path = snapfile, name = "energy_budget")
# dense enough that ~10 full-field snapshots land in the last third of the
# run (needed for plot_validation's stationary-window averages — spectra,
# isotropy, velocity PDF); the scalar energy budget above is far denser
# already but only carries (E, ε, P), not full fields.
io = FieldWriter(snapfile; every = nsteps ÷ 30)

evolve!(û, r, scheme, dt, nsteps; callbacks = (budget, io), progress = true)

# spectra.png axes pinned to the paper's Fig. 6 (the "cases 2a,2b,2c" curve,
# our case 2c analogue) for a direct visual comparison: E(k) vs k, log-log,
# k ∈ [1, 100], E(k) ∈ [1e-8, 1e-1]. Our 64³ run's Nyquist (k ≈ 32) won't
# reach the right edge of that range — expected, matches how the paper's own
# lower-Re "cases 1" curve falls short of it too.
plot_summary(snapfile; spectra_xlims = (1, 100), spectra_ylims = (1e-8, 1e-1))
plot_energy_balance(snapfile)     # -> energy_balance.png
plot_slices(snapfile)             # -> slices.png (|u|, xy mid-plane)
# -> compensated_spectrum.png, isotropy_spectra.png, velocity_pdf.png,
# each averaged over the last third of the run (same window as the
# stationary-state check below)
plot_validation(snapfile)

# Stationary-state check against the paper: averages over the final third of
# the run. Expect A(t) = ε/(3u_rms²) ≈ imposed A (their Fig. 8) and
# ℓ/L ≈ 0.19 (their Fig. 9).
stat = budget.values[(2 * end ÷ 3):end]
Ē  = mean(v.E for v in stat)
u2 = mean(2v.E / 3 for v in stat)         # u_rms² = 2E/3
ε̄  = mean(v.ε for v in stat)
ℓ_energy = u2^1.5 / ε̄                     # ℓ = u_rms³/ε (assumes C_ε = 1)

# Independent real-space cross-check: longitudinal autocorrelation of u₁ on
# the final snapshot, integrated to L₁₁ (Pope §6.2). Unlike ℓ_energy above,
# this doesn't presuppose C_ε = 1, so ε·ℓ_corr/u_rms³ is a genuine check of
# the dissipation constant rather than being 1 by construction.
rlag, facf = longitudinal_autocorrelation(û, g)
ℓ_corr = Float64(integral_lengthscale(rlag, facf))
Cε = dissipation_constant(ε̄, ℓ_corr, sqrt(u2))

η    = (Float64(ν)^3 / ε̄)^(1 / 4)
kmax = (N ÷ 2) * (2 * Float64(π) / Float64(g.Lx))
Reλ  = taylor_reynolds(Ē, ε̄, Float64(ν))
mom  = velocity_moments(velocity_samples(û, g))

println("stationary window (last third of ", round(25τ; digits = 1),
        " time units):")
println("  u_rms²          = ", round(u2; sigdigits = 4))
println("  ε               = ", round(ε̄; sigdigits = 4))
println("  A(t) = ε/3u_rms² = ", round(ε̄ / 3u2; sigdigits = 4),
        "   (imposed A = ", A, ")")
println("  ℓ/L (= u_rms³/ε) = ", round(ℓ_energy / Float64(g.Lx); sigdigits = 3),
        "   (paper: ≈ 0.19)")
println("  ℓ/L (real-space autocorrelation, independent of the ε-based",
        " estimate above) = ", round(ℓ_corr / Float64(g.Lx); sigdigits = 3))
println("  C_ε = ε·ℓ_corr/u_rms³ = ", round(Cε; sigdigits = 3),
        "   (compare against the paper's reported value)")
println("  k_max·η         = ", round(kmax * η; sigdigits = 3),
        "   (want ≳ 1 for a resolved dissipation range)")
println("  Re_λ            = ", round(Reλ; sigdigits = 4),
        "   (compare against the paper's case 2c)")
println("  skewness, flatness (u₁, final snapshot) = ",
        round(mom.skewness; sigdigits = 3), ", ",
        round(mom.flatness; sigdigits = 3),
        "   (Gaussian: 0, 3)")
println("results written to ", abspath(rundir))
