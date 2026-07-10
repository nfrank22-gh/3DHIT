"""
    Integrators

Hand-rolled explicit time steppers. Kept deliberately simple: fixed-step
explicit RK over a plain 4D array state, fully in-place with preallocated
stage buffers — a loop Enzyme (or hand-written adjoints) can see through.

Interface:
    evolve!(û, r::AbstractRHS, scheme::AbstractScheme, dt, nsteps;
            t0 = 0, callbacks = ())
"""
module Integrators

using ..RHS: AbstractRHS, rhs!
using ChainRulesCore: @ignore_derivatives
using Printf: @printf, @sprintf

export AbstractScheme, RK4, step!, evolve!
export Callback, Diagnostic, FieldWriter, save_series

abstract type AbstractScheme end

include("rk4.jl")
include("callbacks.jl")

"""
    step!(û, r::AbstractRHS, scheme, dt, t)

Advance `û` by one step of size `dt` from time `t`, in place.
Each scheme implements this.
"""
function step! end

"""
    evolve!(û, r::AbstractRHS, scheme::AbstractScheme, dt, nsteps;
            t0 = 0, callbacks = (), progress = false)

Main driver: repeatedly `step!` and fire callbacks that are due. Callbacks
receive a read-only state snapshot `(; û, t, step, rhs, grid)` and must be
pure (all accumulation happens inside the harness, wrapped in
`@ignore_derivatives` so differentiated runs skip it).

`progress = true` prints a progress line (step, simulation time, wall time,
ETA) roughly 20 times over the run; an integer prints every that many steps.
Printing never touches the state, so it costs no reductions or GPU syncs.

TODO later: replace the fixed `dt` with a timestepper object
(`FixedDt(dt)` / `CFL(cfl)`), keeping this signature as the convenience form.
"""
function evolve!(û, r::AbstractRHS, scheme::AbstractScheme, dt, nsteps;
                 t0 = 0, callbacks = (), progress = false)
    every = _progress_every(progress, nsteps)
    wall0 = time()
    t = t0
    _fire_due!(callbacks, û, r, 0, t, dt)
    for n in 1:nsteps
        step!(û, r, scheme, dt, t)
        t = t0 + n * dt
        _fire_due!(callbacks, û, r, n, t, dt)
        every > 0 && (n % every == 0 || n == nsteps) &&
            @ignore_derivatives _print_progress(n, nsteps, t, wall0)
    end
    _finalize!(callbacks)
    return û
end

# Bool is more specific than Integer, so `progress = true/false` dispatches
# here and an explicit interval on the method below.
_progress_every(p::Bool, nsteps) = p ? max(1, nsteps ÷ 20) : 0
_progress_every(p::Integer, nsteps) = max(Int(p), 1)

function _print_progress(n, nsteps, t, wall0)
    elapsed = time() - wall0
    eta = elapsed / n * (nsteps - n)
    @printf("step %*d/%d (%3d%%)  t = %-9.4g  elapsed %s  eta %s\n",
            ndigits(nsteps), n, nsteps, round(Int, 100n / nsteps), t,
            _fmt_secs(elapsed), _fmt_secs(eta))
    flush(stdout)
    return nothing
end

_fmt_secs(s) = s < 60 ? @sprintf("%.1fs", s) :
               @sprintf("%dm%02ds", s ÷ 60, round(Int, s % 60))

function _fire_due!(callbacks, û, r, step, t, dt)
    isempty(callbacks) && return nothing
    state = (; û, t, step, rhs = r, grid = r.grid)
    for cb in callbacks
        is_due(cb, step, t, dt) && fire!(cb, state)
    end
    return nothing
end

function _finalize!(callbacks)
    isempty(callbacks) && return nothing
    for cb in callbacks
        finalize!(cb)
    end
    return nothing
end

end # module
