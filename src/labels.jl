# Human-readable `show` and filesystem-safe `label` for the user-facing
# structs (Python-__repr__-like functionality).
#
# Two layers, deliberately separate:
#   * `Base.show`  — pretty, for the REPL/logs (also stops the default show
#     from dumping the scratch buffers of RHS/Grid structs);
#   * `label(x)`   — short slug restricted to [A-Za-z0-9._-], composable into
#     run directory names, e.g.
#         joinpath("results", join((label(g), label(r), label(scheme)), "_"))
#         # -> results/N64_NavierStokes_nu0.001_NoForcing_RK4
#
# Numbers are formatted with %g so Float32/Float64 print identically
# ("0.001", never "0.001f0").

using Printf: @sprintf

_fmt(x::Real) = @sprintf("%g", x)

"""
    label(x) -> String

Short filesystem-safe identifier (characters `[A-Za-z0-9._-]`) describing `x`
and its parameters. Compose labels of the grid, RHS, and scheme into run
directory names; equal parameters give equal labels, so reruns land in the
same folder.
"""
function label end

label(g::Grid) =
    g.Nx == g.Ny == g.Nz ? "N$(g.Nx)" : "N$(g.Nx)x$(g.Ny)x$(g.Nz)"

label(::NoForcing) = "NoForcing"
label(f::BandForcing) =
    "BandForcing_eps$(_fmt(f.ε))_k$(_fmt(f.kmin))-$(_fmt(f.kmax))"
label(f::LinearForcing) = "LinearForcing_A$(_fmt(f.A))"

label(r::NavierStokes) = "NavierStokes_nu$(_fmt(r.ν))_$(label(r.forcing))"

label(::RK4) = "RK4"

# --- pretty printing -------------------------------------------------------

Base.show(io::IO, g::Grid) =
    print(io, "Grid{", eltype(g.k2), "}(", g.Nx, "×", g.Ny, "×", g.Nz,
          ", L=(", _fmt(g.Lx), ", ", _fmt(g.Ly), ", ", _fmt(g.Lz), "))")

Base.show(io::IO, ::NoForcing) = print(io, "NoForcing()")
Base.show(io::IO, f::BandForcing) =
    print(io, "BandForcing(ε=", _fmt(f.ε),
          ", k∈[", _fmt(f.kmin), ", ", _fmt(f.kmax), "])")
Base.show(io::IO, f::LinearForcing) =
    print(io, "LinearForcing(A=", _fmt(f.A), ")")

Base.show(io::IO, r::NavierStokes) =
    print(io, "NavierStokes(ν=", _fmt(r.ν), ", forcing=", r.forcing, ")")

Base.show(io::IO, s::RK4) =
    print(io, "RK4(", join(size(s.k1), "×"), " ", eltype(s.k1), ")")
