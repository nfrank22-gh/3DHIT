"""
    HIT3DEnzymeExt

Enzyme reverse-mode differentiation of one solver step. Two pieces:

1. `EnzymeRules` for applying the grid's FFT plans via `mul!`. Both
   transforms are ℝ-linear, so the reverse pass is a single adjoint-plan
   application (`AbstractFFTs.adjoint` handles the rfft Hermitian weighting);
   intercepting `mul!` here also hides the input-destroying c2r transform
   from Enzyme. The rules are typed to `AbstractFFTs.Plan` so they fire for
   every backend the grid can hold (FFTW on CPU, CUFFT on CUDA, …) — the
   reverse body is backend-generic array code. This is deliberate, contained
   type piracy (rules for AbstractFFTs-owned types) pending official Enzyme
   rules upstream — if those land, delete this block.
2. The `vjp_step!` implementation declared in `src/adjoint.jl`.
"""
module HIT3DEnzymeExt

using HIT3D
using HIT3D: VJPWorkspace, _zero_shadow!
using HIT3D.RHS: NavierStokes
using HIT3D.Integrators: RK4, step!
using Enzyme: Enzyme, EnzymeRules, Const, Duplicated, Annotation, Reverse,
              autodiff, set_runtime_activity
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: mul!

# Any AbstractFFTs plan the Grid can hold: raw r2c plans and the ScaledPlan
# wrapper `plan_irfft` returns around the c2r plan (`ScaledPlan <: Plan`),
# on any backend (FFTW.rFFTWPlan, CUDA's rCuFFTPlan, …).
const RealFFTPlan = AbstractFFTs.Plan

# Plans hold no differentiable state — tell activity analysis so it never
# tries to shadow them.
EnzymeRules.inactive_type(::Type{<:AbstractFFTs.Plan}) = true

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mul!)}, ::Type{RT}, y::Duplicated,
        p::Annotation{<:RealFFTPlan}, x::Duplicated) where {RT}
    mul!(y.val, p.val, x.val)
    primal = EnzymeRules.needs_primal(config) ? y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? y.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mul!)}, ::Type{RT}, tape, y::Duplicated,
        p::Annotation{<:RealFFTPlan}, x::Duplicated) where {RT}
    # x̄ += Pᵀ ȳ; the primal fully overwrote y, so its incoming cotangent is
    # consumed here and must not flow further back.
    x.dval .+= adjoint(p.val) * y.dval
    fill!(y.dval, zero(eltype(y.dval)))
    return (nothing, nothing, nothing)
end

# Enzyme reverse mode wants a scalar-or-nothing return; `step!` returns `û`.
_step_void!(û, r, s, dt, t) = (step!(û, r, s, dt, t); nothing)

function HIT3D.vjp_step!(ū, û, r::NavierStokes, s::RK4, dt, t,
                         ws::VJPWorkspace)
    copyto!(ws.û_work, û)
    _zero_shadow!(ws.rhs_shadow)
    _zero_shadow!(ws.scheme_shadow)
    # Runtime activity: shadow fields that alias their primal (grid, forcing)
    # are treated as constants rather than raising activity errors.
    autodiff(set_runtime_activity(Reverse), Const(_step_void!), Const,
             Duplicated(ws.û_work, ū),
             Duplicated(r, ws.rhs_shadow),
             Duplicated(s, ws.scheme_shadow),
             Const(dt), Const(t))
    return ū
end

end # module
