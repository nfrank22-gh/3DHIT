# Reductions over the spectral state. Everything here is a Parseval-type
# quadratic form, so it shares one primitive (`volume_integral`) instead of
# each function hand-rolling its own weighted sum.

_norm2(g::Grid) = (T = eltype(g.k2); (T(g.Nx) * T(g.Ny) * T(g.Nz))^2)

"""
    volume_integral(f, w, a, g::Grid) -> scalar

Weighted spectral reduction `Σ w·f(a) / Ntot²`, allocation-free over a lazy
Broadcasted (GPU-friendly). `w` and `a` must broadcast against each other —
`w` is typically `hermitian_weights(g)`, optionally combined with a shell
mask or `g.k2` via a nested `broadcasted(*, ...)` (kept lazy, so no extra
array is materialized), and `a` is the spectral field being reduced.
`f = abs2` recovers the Parseval sum `energy`/`enstrophy` use; other choices
give the rest of the quadratic-form diagnostics below.
"""
volume_integral(f, w, a, g::Grid) =
    sum(instantiate(broadcasted((wi, ai) -> wi * f(ai), w, a))) / _norm2(g)

"""Total kinetic energy ½⟨|u|²⟩ from the spectral state."""
energy(û, g::Grid) = volume_integral(abs2, hermitian_weights(g), û, g) / 2

"""Total enstrophy ½⟨|ω|²⟩ (assumes `û` divergence-free)."""
enstrophy(û, g::Grid) =
    volume_integral(abs2, broadcasted(*, hermitian_weights(g), g.k2), û, g) / 2

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
        Es[s] = volume_integral(abs2, broadcasted(*, w, shell), û, g) / 2
    end
    return ks, Es
end

"""
    component_spectra(û, g::Grid) -> (k, E11, E22, E33)

Per-component shell-summed spectra, same shells as `energy_spectrum` (whose
`E` is their sum: `E11 .+ E22 .+ E33 == E`). Isotropic turbulence has
`E11 ≈ E22 ≈ E33` throughout the resolved range; a systematic, persistent
split flags anisotropy from the initial condition, forcing, or a
solver/dealiasing bug.
"""
function component_spectra(û, g::Grid)
    T = eltype(g.k2)
    w = hermitian_weights(g)
    kmag = sqrt.(g.k2)
    k0 = T(2) * T(π) / g.Lx
    smax = ceil(Int, sqrt(3) * max(g.Nx, g.Ny, g.Nz) / 2) + 1
    ks = collect(T, (1:smax) .* k0)
    E11, E22, E33 = zeros(T, smax), zeros(T, smax), zeros(T, smax)
    for s in 1:smax
        lo, hi = (s - T(0.5)) * k0, (s + T(0.5)) * k0
        shell = @. (lo <= kmag) & (kmag < hi)
        wshell = broadcasted(*, w, shell)
        E11[s] = volume_integral(abs2, wshell, view(û, :, :, :, 1), g) / 2
        E22[s] = volume_integral(abs2, wshell, view(û, :, :, :, 2), g) / 2
        E33[s] = volume_integral(abs2, wshell, view(û, :, :, :, 3), g) / 2
    end
    return ks, E11, E22, E33
end

"""
    compensated_spectrum(ks, Es, ε, ν) -> (kη, Ecomp)

Kolmogorov-normalized spectrum for inertial-range validation: `kη = ks·η`
with the Kolmogorov length `η = (ν³/ε)^(1/4)`, and
`Ecomp = Es·ε^(-2/3)·ks^(5/3)`. A resolved inertial range shows a plateau at
the Kolmogorov constant C_K ≈ 1.5–1.6.
"""
function compensated_spectrum(ks, Es, ε, ν)
    η = (ν^3 / ε)^(1 // 4)
    return ks .* η, Es .* ε^(-2 / 3) .* ks .^ (5 / 3)
end

"""
    dissipation_constant(ε, ℓ, u_rms) -> C_ε

Dissipation constant `C_ε = ε·ℓ/u_rms³` (Rosales & Meneveau §II), expected to
approach a universal, Re-independent value at high Re. `ℓ` must come from an
*independent* lengthscale estimate (e.g. [`integral_lengthscale`](@ref) from
the real-space autocorrelation) — computing `ℓ` as `u_rms³/ε` and feeding it
back in here would make `C_ε ≡ 1` by construction rather than a check.
"""
dissipation_constant(ε, ℓ, u_rms) = ε * ℓ / u_rms^3

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
