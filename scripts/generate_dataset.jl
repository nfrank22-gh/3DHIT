# Dataset generation driver: runs one HIT simulation and saves full-field
# snapshots at a fixed simulation-time cadence after an initial warmup, plus
# the dense energy-budget series needed by scripts/linear_forcing_validation.jl.
#
# Parameters come from a TOML config (see configs/ — gitignored, untracked
# scratch; there is no checked-in example). The base output directory comes
# from the sibling file configs/output_path.txt (one line: an absolute or
# repo-relative path), also gitignored since it's machine-local.
#
# Config schema:
#
#   [grid]
#   N = 64
#   backend = "cpu"            # "cpu" | "cuda" | "metal"
#
#   [physics]
#   nu = 4.491e-3
#   dt = 5e-3
#   total_time = 125.0         # nsteps = round(total_time / dt)
#
#   [forcing]
#   type = "LinearForcing"     # "NoForcing" | "BandForcing" | "LinearForcing"
#   A = 0.0667                 # LinearForcing
#   # eps = 0.1; kmin = 1.0; kmax = 2.0   # BandForcing instead
#
#   [initial_condition]
#   k0 = 2.0
#   u0 = 0.5
#   # seed = 42                # optional, CPU-only
#
#   [dataset]
#   warmup_time = 10.0         # no snapshots before this simulation time
#   save_dt = 2.5              # snapshot cadence after warmup
#
# Run from the repo root with the scripts environment:
#   julia --project=scripts scripts/generate_dataset.jl configs/my_dataset.toml

using HIT3D
using TOML
using Random: MersenneTwister

length(ARGS) == 1 ||
    error("usage: julia --project=scripts scripts/generate_dataset.jl <config.toml>")
config_path = ARGS[1]
cfg = TOML.parsefile(config_path)

output_path_file = joinpath(@__DIR__, "..", "configs", "output_path.txt")
isfile(output_path_file) ||
    error("missing $output_path_file — create it with one line: the base " *
          "results directory for generated datasets")
base_dir = strip(read(output_path_file, String))

# --- [grid] ------------------------------------------------------------

N = cfg["grid"]["N"]
backend = get(cfg["grid"], "backend", "cpu")

if backend == "cpu"
    ArrayType = Array
elseif backend == "cuda"
    using CUDA
    ArrayType = CuArray
elseif backend == "metal"
    using Metal
    ArrayType = MtlArray
else
    error("unknown [grid].backend = $(repr(backend)) (expected cpu/cuda/metal)")
end

g = Grid(N; ArrayType)

# --- [initial_condition] ------------------------------------------------

ic = cfg["initial_condition"]
k0 = Float32(ic["k0"])
u0 = Float32(ic["u0"])
seed = get(ic, "seed", nothing)
rng = seed === nothing ? nothing : MersenneTwister(seed)

û = spectral_state(g)
# Rosales & Meneveau (2005) Eq. (9): E(k) = 16√(2/π)(u0²/k0⁵)k⁴exp(−2k²/k0²).
random_field!(û, g; rng,
             spectrum = k -> 16 * sqrt(2 / π) * u0^2 / k0^5 *
                             k^4 * exp(-2k^2 / k0^2))

# --- [forcing] -----------------------------------------------------------

fc = cfg["forcing"]
forcing_type = fc["type"]
forcing = if forcing_type == "NoForcing"
    NoForcing()
elseif forcing_type == "LinearForcing"
    LinearForcing(Float32(fc["A"]))
elseif forcing_type == "BandForcing"
    BandForcing(g; ε = Float32(fc["eps"]), kmin = Float32(fc["kmin"]),
               kmax = Float32(fc["kmax"]))
else
    error("unknown [forcing].type = $(repr(forcing_type))")
end

# --- [physics] / [dataset] ------------------------------------------------

ν = Float32(cfg["physics"]["nu"])
dt = Float32(cfg["physics"]["dt"])
total_time = Float32(cfg["physics"]["total_time"])
nsteps = round(Int, total_time / dt)

warmup_time = Float32(cfg["dataset"]["warmup_time"])
save_dt = Float32(cfg["dataset"]["save_dt"])

r = NavierStokes(g; ν, forcing)
scheme = RK4(û)

rundir = joinpath(base_dir, join((label(g), label(r), label(scheme)), "_"))
snapfile = joinpath(rundir, "dataset.jld2")

budget = Diagnostic(energy_budget; every = 10,
                    valuetype = @NamedTuple{E::Float64, ε::Float64,
                                            P::Float64},
                    path = snapfile, name = "energy_budget")
io = FieldWriter(snapfile; every_time = save_dt, warmup_time = warmup_time)

evolve!(û, r, scheme, dt, nsteps; callbacks = (budget, io), progress = true)

mkpath(rundir)
cp(config_path, joinpath(rundir, basename(config_path)); force = true)

println("dataset written to ", abspath(rundir))
