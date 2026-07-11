# Callback system.
#
# Design rules (AD compatibility):
#   * User callback functions are PURE: `f(state) -> value`, where `state`
#     is the named tuple `(; û, t, step, rhs, grid)`. They never mutate.
#   * All accumulation/IO is owned by the wrapper structs below and executed
#     inside `@ignore_derivatives`, so callbacks contribute no gradient paths.
#   * Anything that is part of the dynamics (forcing!) belongs in the RHS,
#     never in a callback.
#
# Scheduling is the harness's job: wrappers carry `every` (steps) or
# `every_time` (simulation-time interval) and `evolve!` decides when to fire.
# All wrappers fire at step 0 (the initial state) as well.

using JLD2: jldopen
using ..Schema: stepkey, gridkey, serieskey, seriesfield

"""
    Callback(f; every = 1, every_time = nothing)

Generic wrapper: call the pure function `f(state)` on the given schedule and
discard the result (side-effect-free hook point, e.g. progress printing —
which is IO, handled by the harness). If `every_time` is given it takes
precedence over `every` and fires whenever the simulation time crosses a
multiple of `every_time`.
"""
struct Callback{F, T}
    f::F
    every::Int
    every_time::T
end

Callback(f; every = 1, every_time = nothing) = Callback(f, every, every_time)

"""
    Diagnostic(f; every = 1, timetype = Float64, valuetype = Float64,
               path = nothing, name = nothing)

Scalar/small-vector timeseries recorder: appends `f(state)` and the current
time to internally owned vectors. Cheap; meant to run often (energy,
dissipation, max divergence, ...). After the run, `d.times` / `d.values`
are the plot data. For non-scalar diagnostics pass e.g.
`valuetype = Vector{Float64}`.

If `path` and `name` are given, `evolve!` persists the recorded series to
`path` under `series/<name>` (via `save_series`) automatically once the run
finishes — no separate manual `save_series` call needed. This suits the
common single-`evolve!`-call driver; for a run split across multiple
`evolve!` calls, leave `path`/`name` unset (they only fire once, at the end
of whichever `evolve!` call happens to finish first) and call `save_series`
by hand after the last one.
"""
struct Diagnostic{F, T, V}
    f::F
    every::Int
    times::Vector{T}
    values::Vector{V}
    path::Union{Nothing, String}
    name::Union{Nothing, String}
end

function Diagnostic(f; every = 1, timetype = Float64, valuetype = Float64,
                     path = nothing, name = nothing)
    path !== nothing && name === nothing &&
        error("Diagnostic(...; path = ...) also needs `name` " *
              "(the series/<name> group to persist it under)")
    return Diagnostic(f, every, timetype[], valuetype[], path, name)
end

"""
    FieldWriter(path; every = 1, fields = (:û,), overwrite = true,
                every_time = nothing, warmup_time = nothing)

Full-field snapshot writer: copies the requested state fields to the CPU and
appends them (plus the time) to a JLD2 file under `"step_<n>/<field>"`.
Expensive; run rarely.

The file is self-describing: on the first write a `grid` group is stored
(`Nx, Ny, Nz, Lx, Ly, Lz`, and the RHS's `ν` when it has one), so
postprocessing tools like `plot_summary` can reconstruct the `Grid` from the
file alone.

With `overwrite = true` (default) an existing file at `path` is deleted at
construction time — rerunning a driver replaces its snapshots instead of
erroring on duplicate group names mid-run. Pass `overwrite = false` to
append across multiple `evolve!` calls (steps must not repeat).

By default snapshots are scheduled by step count (`every`), like any other
callback. Pass `every_time` to schedule by simulation time instead: the
first snapshot lands at `t = warmup_time` (default `0`) and recurs every
`every_time` thereafter — no snapshots are taken before `warmup_time`. This
suits dataset-generation drivers that want to discard an initial transient
and then sample at a fixed physical-time cadence regardless of `dt`.
"""
struct FieldWriter{T, ET}
    path::String
    every::Int
    fields::T
    every_time::ET
    warmup_time::ET
end

function FieldWriter(path; every = 1, fields = (:û,), overwrite = true,
                     every_time = nothing, warmup_time = nothing)
    every_time !== nothing && warmup_time === nothing &&
        (warmup_time = zero(every_time))
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    overwrite && rm(path; force = true)
    return FieldWriter(path, every, fields, every_time, warmup_time)
end

"""
    save_series(path, name, d::Diagnostic)

Persist a `Diagnostic`'s recorded time series into the JLD2 file at `path`
(appending — typically the run's `FieldWriter` snapshot file, so the run
directory keeps a single self-describing artifact). Called once after
`evolve!`.

Layout is columnar under `series/<name>/`: the times as `t`, and the values
either as one `values` vector (scalar diagnostics) or as one vector per
field for named-tuple values (e.g. `E`, `ε`, `P` for `energy_budget`).

Errors if `series/<name>` already exists in the file; rerunning a driver
with the default `FieldWriter(overwrite = true)` starts from a fresh file,
so this only bites when saving the same name twice into one file.
"""
function save_series(path, name, d::Diagnostic)
    grp = serieskey(name)
    jldopen(path, "a+") do file
        haskey(file, grp) &&
            error("$grp already exists in $path (saving the same series " *
                  "twice? reruns should start from a fresh file)")
        file[seriesfield(name, "t")] = d.times
        V = eltype(d.values)
        if V <: NamedTuple
            for fld in fieldnames(V)
                file[seriesfield(name, fld)] =
                    [getfield(v, fld) for v in d.values]
            end
        else
            file[seriesfield(name, "values")] = d.values
        end
    end
    return path
end

# --- harness internals -----------------------------------------------------

"""Return true if callback `cb` is due at (step, t), where `dt` is the step
just taken (needed for time-based schedules)."""
function is_due(cb::Callback, step, t, dt)
    cb.every_time === nothing && return step % cb.every == 0
    return step == 0 ||
           floor(t / cb.every_time) > floor((t - dt) / cb.every_time)
end
is_due(cb::Diagnostic, step, t, dt) = step % cb.every == 0

function is_due(cb::FieldWriter, step, t, dt)
    cb.every_time === nothing && return step % cb.every == 0
    t < cb.warmup_time && return false
    return floor((t - cb.warmup_time) / cb.every_time) >
           floor((t - dt - cb.warmup_time) / cb.every_time)
end

"""Fire `cb` on `state`, performing any accumulation/IO inside
`@ignore_derivatives`."""
function fire!(cb::Callback, state)
    @ignore_derivatives cb.f(state)
    return nothing
end

function fire!(cb::Diagnostic, state)
    @ignore_derivatives begin
        push!(cb.times, state.t)
        push!(cb.values, cb.f(state))
    end
    return nothing
end

"""Run once, after `evolve!`'s last step (no-op unless overridden). Used to
persist accumulated state that only makes sense to write once a run is
done, e.g. a `Diagnostic`'s recorded series."""
finalize!(cb) = nothing

function finalize!(cb::Diagnostic)
    cb.path === nothing && return nothing
    @ignore_derivatives save_series(cb.path, cb.name, cb)
    return nothing
end

function fire!(cb::FieldWriter, state)
    @ignore_derivatives begin
        grp = stepkey(state.step)
        jldopen(cb.path, "a+") do file
            if !haskey(file, "grid")
                g = state.grid
                file[gridkey(:Nx)] = g.Nx
                file[gridkey(:Ny)] = g.Ny
                file[gridkey(:Nz)] = g.Nz
                file[gridkey(:Lx)] = g.Lx
                file[gridkey(:Ly)] = g.Ly
                file[gridkey(:Lz)] = g.Lz
                hasproperty(state.rhs, :ν) &&
                    (file[gridkey(:nu)] = state.rhs.ν)
            end
            file[grp * "/t"] = state.t
            for fld in cb.fields
                file[grp * "/" * string(fld)] = Array(getproperty(state, fld))
            end
        end
    end
    return nothing
end
