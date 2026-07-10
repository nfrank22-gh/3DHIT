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
io = FieldWriter(snapfile; every = nsteps ÷ 5)

evolve!(û, r, scheme, dt, nsteps; callbacks = (budget, io), progress = true)

plot_summary(snapfile)            # -> summary.png, spectra.png
plot_energy_balance(snapfile)     # -> energy_balance.png
plot_slices(snapfile)             # -> slices.png (|u|, xy mid-plane)

# Stationary-state check against the paper: averages over the final third of
# the run. Expect A(t) = ε/(3u_rms²) ≈ imposed A (their Fig. 8) and
# ℓ/L ≈ 0.19 (their Fig. 9).
stat = budget.values[(2 * end ÷ 3):end]
u2 = mean(2v.E / 3 for v in stat)         # u_rms² = 2E/3
ε̄  = mean(v.ε for v in stat)
ℓ  = u2^1.5 / ε̄                            # ℓ = u_rms³/ε
println("stationary window (last third of ", round(25τ; digits = 1),
        " time units):")
println("  u_rms²          = ", round(u2; sigdigits = 4))
println("  ε               = ", round(ε̄; sigdigits = 4))
println("  A(t) = ε/3u_rms² = ", round(ε̄ / 3u2; sigdigits = 4),
        "   (imposed A = ", A, ")")
println("  ℓ/L             = ", round(ℓ / Float64(g.Lx); sigdigits = 3),
        "   (paper: ≈ 0.19)")
println("results written to ", abspath(rundir))
