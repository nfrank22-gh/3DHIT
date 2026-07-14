# Single-step vector–Jacobian products via Enzyme — the AD entry point is
# implemented by the HIT3DEnzymeExt package extension (`using Enzyme`
# activates it). The workspace lives here because building shadow structs is
# plain allocation, no Enzyme required.

"""
    VJPWorkspace(r::NavierStokes, s::RK4)

Preallocated shadow structures for [`vjp_step!`](@ref): a shadow RHS and a
shadow scheme — same types as the primal ones with freshly zeroed scratch,
sharing the primal's read-only `Grid` (wavenumbers, masks, and FFT plans are
never differentiated) — plus a working copy of the state so the primal `û`
stays untouched. Roughly doubles the solver's memory footprint. Construct
once and reuse; `vjp_step!` re-zeroes the shadows on every call.
"""
struct VJPWorkspace{R, S, A}
    rhs_shadow::R
    scheme_shadow::S
    û_work::A
end

VJPWorkspace(r::NavierStokes, s::RK4) =
    VJPWorkspace(_shadow(r), _shadow(s), zero(s.k1))

# Shadows share every read-only field (grid, forcing, parameters) with the
# primal; under Enzyme's runtime activity, shadow === primal marks a field
# as non-differentiated. Only the mutated scratch gets distinct zeroed arrays.
_shadow(r::NavierStokes) =
    NavierStokes(zero(r.ν), r.forcing, r.grid, zero(r.u_phys), zero(r.ω_phys),
                 zero(r.nl_phys), zero(r.scratch_spec))
_shadow(s::RK4) = RK4(zero(s.k1), zero(s.k2), zero(s.k3), zero(s.k4),
                      zero(s.tmp))

function _zero_shadow!(r::NavierStokes)
    for a in (r.u_phys, r.ω_phys, r.nl_phys, r.scratch_spec)
        fill!(a, zero(eltype(a)))
    end
    return r
end

function _zero_shadow!(s::RK4)
    for a in (s.k1, s.k2, s.k3, s.k4, s.tmp)
        fill!(a, zero(eltype(a)))
    end
    return s
end

"""
    vjp_step!(ū, û, r::NavierStokes, s::RK4, dt, t, ws::VJPWorkspace) -> ū

Exact vector–Jacobian product of one discrete `step!`: with `F` the map
`û ↦ step!(û, r, s, dt, t)`, overwrite `ū` with `(∂F/∂û)ᵀ ū`.

Semantics:
- `û` is the **pre-step** state and is left unmodified (the forward pass runs
  on an internal copy in `ws`).
- On entry `ū` is the cotangent of the step's *output*; on exit it holds the
  cotangent of the step's *input*. Chain calls backwards through stored
  forward states to sweep an adjoint over a trajectory.
- The transpose is with respect to the plain real inner product on the stored
  rfft coefficients, `Re⟨a, b⟩ = Σ (Re aᵢ Re bᵢ + Im aᵢ Im bᵢ)` — Enzyme's
  convention, with no Hermitian shell weighting. Seed `ū` consistently (e.g.
  a cotangent obtained by differentiating a physical-space functional through
  the same convention) and the sweep is self-consistent.

Requires Enzyme: `using Enzyme` activates the implementation.
"""
vjp_step!(args...) =
    error("vjp_step! requires Enzyme to be loaded first: `using Enzyme`.")
