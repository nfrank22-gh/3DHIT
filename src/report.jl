# Pure JLD2 reading/parsing for `FieldWriter` snapshot files. Deliberately
# Makie-free (unlike plotting.jl/HIT3DMakieExt.jl) so it lives in the
# package proper and is exercised by the ordinary test suite; the Makie
# extension calls these functions and only builds `Figure`s from the
# result — it does no JLD2 parsing of its own.

using .Schema: is_stepkey, stepnum, gridkey, serieskey
using JLD2: jldopen

"""Sorted `"step_########"` group names in an open `file`."""
_stepkeys(file) = sort(filter(is_stepkey, keys(file)))

"""
    _read_grid(file, T) -> (grid, ν)

Reconstruct the `Grid` (CPU, precision `T`) and read ν from a `FieldWriter`
file's self-describing `grid` group. Falls back to a 2π cube with unknown ν
for files predating the metadata group, inferring `Nx` from the rfft layout
of the first snapshot (assumes even `Nx`).
"""
function _read_grid(file, T)
    if haskey(file, "grid")
        g = Grid(file[gridkey(:Nx)], file[gridkey(:Ny)], file[gridkey(:Nz)];
                 Lx = file[gridkey(:Lx)], Ly = file[gridkey(:Ly)],
                 Lz = file[gridkey(:Lz)], T = T)
        ν = haskey(file, gridkey(:nu)) ? file[gridkey(:nu)] : nothing
        return g, ν
    end
    keys_ = _stepkeys(file)
    isempty(keys_) && error("no snapshots found in file")
    û = file[first(keys_) * "/û"]
    Nx = 2 * (size(û, 1) - 1)   # rfft layout; assumes even Nx
    @warn "no grid metadata in file; assuming a 2π cube (N = $Nx) and unknown ν"
    return Grid(Nx, size(û, 2), size(û, 3); T = T), nothing
end

"""Precision of the stored state, inferred from the first snapshot's `û`."""
_state_T(file) = real(eltype(file[first(_stepkeys(file)) * "/û"]))

"""
    load_run(path) -> (; grid, ν, T, steps)

Open the `FieldWriter` file at `path` and return its self-describing grid,
viscosity (`nothing` if the file has no `grid/nu` entry), state precision,
and the sorted vector of snapshot step numbers. Cheap: does not load any
field arrays. Errors if the file has no snapshots.
"""
function load_run(path)
    jldopen(path, "r") do file
        keys_ = _stepkeys(file)
        isempty(keys_) && error("no snapshots found in $path")
        T = _state_T(file)
        g, ν = _read_grid(file, T)
        return (; grid = g, ν, T, steps = stepnum.(keys_))
    end
end

"""
    read_series(path_or_file, name) -> NamedTuple or nothing

Read the dense `series/<name>` group written by `save_series` — columnar:
`t` plus either `values` (scalar diagnostics) or one vector per
named-tuple field (e.g. `E`, `ε`, `P` for `energy_budget`) — as a
`NamedTuple`, or `nothing` if the file has no such group.
"""
function read_series(path::AbstractString, name)
    jldopen(path, "r") do file
        read_series(file, name)
    end
end

function read_series(file, name)
    grp = serieskey(name)
    haskey(file, grp) || return nothing
    fields = [k for k in keys(file[grp]) if k != "t"]
    pairs = Pair{Symbol, Any}[:t => file[grp * "/t"]]
    for fld in fields
        push!(pairs, Symbol(fld) => file[grp * "/" * fld])
    end
    return NamedTuple(pairs)
end
