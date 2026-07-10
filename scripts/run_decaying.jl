# Example driver: decaying HIT (Float32, CPU) with plots.
#
# Run from the repo root with the scripts environment (which has CairoMakie;
# see scripts/Project.toml):
#   julia --project=scripts scripts/run_decaying.jl
#
# For GPU, construct the grid with the backend array type, e.g.
#   using CUDA;  g = Grid(N; ArrayType = CuArray)
#   using Metal; g = Grid(N; ArrayType = MtlArray)   # verify plan_rfft support!
# Everything downstream is backend-agnostic.

using HIT3D
using CairoMakie   # loads Makie → activates HIT3D's plotting extension

N  = 64
ν  = 1f-3
dt = 1f-3

g = Grid(N)                       # Float32, Array by default
û = spectral_state(g)
random_field!(û, g)

r      = NavierStokes(g; ν, forcing = NoForcing())
scheme = RK4(û)

# Everything for this run lands in results/<run label>/, named from the
# grid / RHS / scheme parameters — reruns with equal parameters overwrite.
rundir = joinpath(@__DIR__, "..", "results",
                  join((label(g), label(r), label(scheme)), "_"))
snapfile = joinpath(rundir, "decaying.jld2")

budget = Diagnostic(energy_budget; every = 10,
                    valuetype = @NamedTuple{E::Float64, ε::Float64,
                                            P::Float64},
                    path = snapfile, name = "energy_budget")
io = FieldWriter(snapfile; every = 500)

# `budget`'s series (`path`/`name` above) is persisted into the snapshot
# file automatically once evolve! finishes, so postprocessing needs nothing
# but the one self-describing .jld2:
evolve!(û, r, scheme, dt, 5_000; callbacks = (budget, io))

plot_summary(snapfile)            # -> summary.png, spectra.png
plot_energy_balance(snapfile)     # -> energy_balance.png
plot_slices(snapfile)             # -> slices.png (|u|, xy mid-plane)
println("results written to ", abspath(rundir))
