"""
    Diagnostics

Analysis functions on `(û, grid)` — usable both offline (postprocessing) and
inside `Diagnostic` callbacks during a run. Plotting deliberately lives
outside the package (in `scripts/`, later possibly a Makie extension).

All functions account for the rfft layout via `hermitian_weights`: kx > 0
modes represent two conjugate modes each and are double-counted in every
reduction. Quantities are volume means (⟨·⟩ over the box), so with the
unnormalized rfft convention each sum carries a 1/(NxNyNz)² factor.

The vorticity-based quantities assume `û` is divergence-free (the solver
invariant), so that |ω̂|² = |k × û|² = k²|û|².
"""
module Diagnostics

using ..Grids: Grid, hermitian_weights
using ..RHS: AbstractForcing, injection
using Base.Broadcast: broadcasted, instantiate

export energy, enstrophy, dissipation, energy_spectrum, energy_budget
export kolmogorov_scale, taylor_microscale, taylor_reynolds

# Allocation-free weighted reduction over a lazy Broadcasted (GPU-friendly).
_norm2(g::Grid) = (T = eltype(g.k2); (T(g.Nx) * T(g.Ny) * T(g.Nz))^2)

"""Total kinetic energy ½⟨|u|²⟩ from the spectral state."""
function energy(û, g::Grid)
    w = hermitian_weights(g)
    return sum(instantiate(broadcasted((wi, ui) -> wi * abs2(ui), w, û))) /
           (2 * _norm2(g))
end

"""Total enstrophy ½⟨|ω|²⟩ (assumes `û` divergence-free)."""
function enstrophy(û, g::Grid)
    w = hermitian_weights(g)
    return sum(instantiate(broadcasted(
               (wi, k2i, ui) -> wi * k2i * abs2(ui), w, g.k2, û))) /
           (2 * _norm2(g))
end

"""Energy dissipation rate ε = ν⟨|ω|²⟩ = 2ν · enstrophy."""
dissipation(û, g::Grid, ν) = 2 * ν * enstrophy(û, g)

"""
    energy_spectrum(û, g::Grid) -> (k, E)

Shell-summed energy spectrum over shells of unit spacing `k₀ = 2π/Lx`
(cubic box assumed for the binning): `E[s]` is the energy in the shell
`|k| ∈ [(s−½)k₀, (s+½)k₀)`, so `sum(E) == energy(û, g)`. Returned as CPU
vectors ready for plotting.
"""
function energy_spectrum(û, g::Grid)
    T = eltype(g.k2)
    w = hermitian_weights(g)
    kmag = sqrt.(g.k2)
    k0 = T(2) * T(π) / g.Lx
    smax = ceil(Int, sqrt(3) * max(g.Nx, g.Ny, g.Nz) / 2) + 1
    ks = collect(T, (1:smax) .* k0)
    Es = zeros(T, smax)
    for s in 1:smax
        lo, hi = (s - T(0.5)) * k0, (s + T(0.5)) * k0
        shell = @. (lo <= kmag) & (kmag < hi)
        Es[s] = sum(instantiate(broadcasted(
                    (wi, mi, ui) -> wi * mi * abs2(ui), w, shell, û))) /
                (2 * _norm2(g))
    end
    return ks, Es
end

"""
    energy_budget(û, g::Grid, ν, forcing) -> (; E, ε, P)
    energy_budget(state)                  -> (; E, ε, P)

The three raw terms of the kinetic-energy budget dE/dt = P − ε at one
instant: energy `E`, dissipation `ε`, and the forcing's exact injected power
`P` (see `injection`). Derived quantities (dE/dt, residual, cumulative
integrals) are left to postprocessing — see `plot_energy_balance`.

The one-argument method unpacks the callback state named tuple, so a run
records the budget with

    Diagnostic(energy_budget; every = 10,
               valuetype = @NamedTuple{E::Float64, ε::Float64, P::Float64})
"""
energy_budget(û, g::Grid, ν, forcing::AbstractForcing) =
    (; E = energy(û, g), ε = dissipation(û, g, ν),
       P = injection(forcing, û, g))

energy_budget(state) =
    energy_budget(state.û, state.grid, state.rhs.ν, state.rhs.forcing)

"""Kolmogorov length scale η = (ν³/ε)^{1/4}."""
kolmogorov_scale(û, g::Grid, ν) = (ν^3 / dissipation(û, g, ν))^(1 // 4)

"""Taylor microscale λ = √(5E/Ω)  (from ε = 15ν u′²/λ² with u′² = 2E/3)."""
taylor_microscale(û, g::Grid) = sqrt(5 * energy(û, g) / enstrophy(û, g))

"""
    taylor_reynolds(E, ε, ν) -> Re_λ
    taylor_reynolds(û, g::Grid, ν) -> Re_λ

Taylor-microscale Reynolds number Re_λ = u′λ/ν, with u′ = √(2E/3) and
λ = √(10νE/ε) (from ε = 15ν u′²/λ², Ω = ε/2ν, λ = √(5E/Ω)). The scalar
form is the single implementation shared by the field-based method above
and any postprocessing that only has the recorded (E, ε, ν) — e.g. a dense
`energy_budget` series."""
function taylor_reynolds(E, ε, ν)
    u′ = sqrt(2 * E / 3)
    λ = sqrt(10 * ν * E / ε)
    return u′ * λ / ν
end

taylor_reynolds(û, g::Grid, ν) = taylor_reynolds(energy(û, g), dissipation(û, g, ν), ν)

end # module
