# Pluggable forcing slot for RHS structs.
#
# A forcing adds its contribution to `dû` in spectral space via
#
#     apply_forcing!(dû, û, u_phys, f::AbstractForcing, g::Grid, t)
#
# Extension contract (this is all a new forcing needs):
#   * one struct <: AbstractForcing, plus one `apply_forcing!` method that
#     ADDS its spectral contribution into `dû` in place;
#   * `u_phys` is the dealiased physical-space velocity the RHS has already
#     computed — read-only, provided so pointwise-in-u forcings don't pay for
#     a second inverse transform. Forcings that need their own FFTs own their
#     own scratch/plans;
#   * the forcing need NOT be divergence-free and need NOT dealias itself:
#     it is applied BEFORE the projection and the 2/3 mask, so any gradient
#     part is absorbed into the pressure (ik·f̂/|k|² term) and any aliased /
#     k = 0 content is removed by the mask. The RHS guarantees this ordering.
#
# Forcings live in the RHS (part of the dynamics), never in callbacks —
# callbacks are pure observers so the solver stays AD-compatible.

using Base.Broadcast: broadcasted, instantiate

abstract type AbstractForcing end

"""
    apply_forcing!(dû, û, u_phys, f::AbstractForcing, g::Grid, t)

Add the forcing contribution to `dû` in place (spectral space). Called by the
RHS before dealiasing and projection — see the extension contract in
`forcing.jl`.
"""
function apply_forcing! end

"""
    injection(f::AbstractForcing, û, g::Grid) -> P

Instantaneous injected power P = ⟨u·f⟩ at state `û`, exact per forcing type
(part of the forcing contract alongside `apply_forcing!`). Diagnostics use
this for the energy budget dE/dt = P − ε; keeping it a per-forcing method
matters for forcings whose power cannot be measured numerically from a
single evaluation (e.g. stochastic forcing, whose mean injection carries an
Itô correction).
"""
function injection end

"""No forcing (decaying turbulence)."""
struct NoForcing <: AbstractForcing end

apply_forcing!(dû, û, u_phys, ::NoForcing, g::Grid, t) = dû

injection(::NoForcing, û, g::Grid) = zero(real(eltype(û)))

"""
    BandForcing(g::Grid; ε, kmin, kmax)

Constant-power low-wavenumber forcing: injects energy at exactly the rate `ε`
into the shell `kmin ≤ |k| ≤ kmax` by amplifying the velocity in the band,

    f̂ = (ε / 2E_band) û    on the band,

so that ⟨u·f⟩ = ε identically (E_band is the current kinetic energy in the
band, computed with Hermitian double-counting of the kx > 0 modes). In a
statistically stationary state the dissipation therefore equals ε.

Divergence-free by construction (proportional to û). If the band is empty of
energy (`2E_band ≤ eps`), the forcing is skipped rather than dividing by ~0.
"""
struct BandForcing{T, AM, AW} <: AbstractForcing
    ε::T          # exact injection rate ⟨u·f⟩
    kmin::T
    kmax::T
    mask::AM      # Bool mask of the forced shell
    w::AW         # mask .* hermitian_weights, for the band-energy reduction
end

function BandForcing(g::Grid; ε, kmin, kmax)
    T = eltype(g.k2)
    mask = @. (T(kmin)^2 <= g.k2) & (g.k2 <= T(kmax)^2)
    w = hermitian_weights(g) .* mask
    return BandForcing(T(ε), T(kmin), T(kmax), mask, w)
end

function apply_forcing!(dû, û, u_phys, f::BandForcing, g::Grid, t)
    T = typeof(f.ε)
    Ntot2 = (T(g.Nx) * T(g.Ny) * T(g.Nz))^2
    # 2·E_band = Σ' w |û|² / Ntot²  (allocation-free reduction over a
    # lazy Broadcasted; works on GPU arrays too)
    twoE = sum(instantiate(broadcasted((wi, ui) -> wi * abs2(ui), f.w, û))) /
           Ntot2
    twoE > eps(T) || return dû
    c = f.ε / twoE
    @. dû += c * f.mask * û
    return dû
end

"""Exact injected power of [`BandForcing`](@ref): `ε` by construction, or
zero when the band is empty and `apply_forcing!` skips (same guard)."""
function injection(f::BandForcing, û, g::Grid)
    T = typeof(f.ε)
    Ntot2 = (T(g.Nx) * T(g.Ny) * T(g.Nz))^2
    twoE = sum(instantiate(broadcasted((wi, ui) -> wi * abs2(ui), f.w, û))) /
           Ntot2
    return twoE > eps(T) ? f.ε : zero(T)
end

"""
    LinearForcing(A)

Lundgren's linear forcing f = A·u (Rosales & Meneveau, Phys. Fluids 17,
095106 (2005), Sec. II): a force proportional to the velocity at every point,
applied here in its exact spectral equivalent f̂ = A·û (Eq. 8 of the paper —
the Fourier transform is linear, so the physical- and spectral-space
implementations coincide).

Prescribing the constant `A` imposes an inverse turnover timescale; the flow
converges to a statistically stationary state with ε = 3A·u_rms² and integral
scale ℓ = u_rms³/ε ≈ 0.19·L, independent of the initial spectrum. For a
target injection rate ε on a box of size L, Eq. 12 gives A ≈ ε^(1/3)/L^(2/3).

Divergence-free and dealiased by construction (proportional to û, which the
solver keeps exactly dealiased).
"""
struct LinearForcing{T} <: AbstractForcing
    A::T          # forcing coefficient (inverse timescale), f = A·u
end

function apply_forcing!(dû, û, u_phys, f::LinearForcing, g::Grid, t)
    @. dû += f.A * û
    return dû
end

"""Exact injected power of [`LinearForcing`](@ref): P = ⟨u·(Au)⟩ = 2A·E,
with E the current kinetic energy (Hermitian-weighted reduction)."""
function injection(f::LinearForcing, û, g::Grid)
    T = typeof(f.A)
    Ntot2 = (T(g.Nx) * T(g.Ny) * T(g.Nz))^2
    w = hermitian_weights(g)
    twoE = sum(instantiate(broadcasted((wi, ui) -> wi * abs2(ui), w, û))) /
           Ntot2
    return f.A * twoE
end

# TODO later: StochasticForcing (needs GPU-friendly RNG strategy; note that
# stochastic forcing also complicates the AD/adjoint story).
