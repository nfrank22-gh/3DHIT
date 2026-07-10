"""
    RHS

Right-hand sides. Each governing equation is one concrete subtype of
`AbstractRHS` implementing the single interface function

    rhs!(dû, û, r::AbstractRHS, t)

RHS structs own their scratch buffers (physical-space work arrays etc.), so a
`Grid` can be shared between several RHS objects. Forcing is a pluggable slot
(`AbstractForcing`) inside each equation struct — see `forcing.jl`.
"""
module RHS

using ..Grids: Grid, project!, dealias!, hermitian_weights
using LinearAlgebra: mul!

export AbstractRHS, rhs!, linear_operator
export AbstractForcing, NoForcing, BandForcing, LinearForcing, apply_forcing!,
       injection

abstract type AbstractRHS end

"""
    rhs!(dû, û, r::AbstractRHS, t)

Evaluate the right-hand side at state `û` and time `t`, writing into `dû`.
This is the only function integrators call.
"""
function rhs! end

"""
    linear_operator(r::AbstractRHS) -> L or nothing

Optional accessor returning the diagonal stiff linear operator in spectral
space (e.g. `-ν k²` for Navier–Stokes), so future integrators can treat it
exactly (integrating factor / ETD). Return `nothing` if not applicable.
"""
linear_operator(::AbstractRHS) = nothing

include("forcing.jl")
include("navierstokes.jl")

export NavierStokes

end # module
