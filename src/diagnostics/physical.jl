# Reductions that need physical space: velocity PDF/moments and the
# longitudinal autocorrelation. Kept separate from spectral.jl's Parseval-sum
# family since these are pointwise/real-space operations, not weighted
# spectral sums — a single `volume_integral`-style primitive can't cover both.

"""
    velocity_samples(û, g::Grid; component = 1) -> Vector

Physical-space values of one velocity component as a flattened CPU vector,
ready for a histogram/PDF plot or moment calculation (see
[`velocity_moments`](@ref)). Costs one inverse transform of the full state
(all three components; the single-component view is free after that).
"""
function velocity_samples(û, g::Grid; component = 1)
    T = real(eltype(û))
    u = similar(û, T, (g.Nx, g.Ny, g.Nz, 3))
    mul!(u, g.iplan, û)
    return Vector(vec(Array(view(u, :, :, :, component))))
end

"""
    velocity_moments(samples) -> (; mean, variance, skewness, flatness)

Central moments of a physical-space velocity-component sample (e.g. from
[`velocity_samples`](@ref)). `flatness` (kurtosis) is 3.0 for a Gaussian;
`skewness` is 0 for a symmetric distribution.
"""
function velocity_moments(samples)
    m = mean(samples)
    c = samples .- m
    v = mean(abs2, c)
    sk = mean(x -> x^3, c) / v^1.5
    fl = mean(x -> x^4, c) / v^2
    return (; mean = m, variance = v, skewness = sk, flatness = fl)
end

"""
    longitudinal_autocorrelation(û, g::Grid; component = 1) -> (r, f)

Longitudinal autocorrelation `f(r) = ⟨u₁(x)u₁(x+r·ê₁)⟩ / ⟨u₁²⟩` of one
velocity component along its own axis (Pope §6.2), via Wiener–Khinchin:
`irfft` of the component's spectral power `|û₁|²` gives the (unnormalized)
circular autocorrelation directly — no extra inverse transform of the full
field needed. Returned over `r ∈ [0, Lx/2]` (the periodic wrap folds the
correlation back on itself beyond the half-box).
"""
function longitudinal_autocorrelation(û, g::Grid; component = 1)
    T = eltype(g.k2)
    a = view(û, :, :, :, component)
    corr = irfft(abs2.(a), g.Nx, 1:3) ./ T(g.Nx * g.Ny * g.Nz)
    n = g.Nx ÷ 2 + 1
    f = Array(view(corr, 1:n, 1, 1))
    r = collect(T, 0:n-1) .* (g.Lx / g.Nx)
    f0 = f[1]
    f0 > 0 || return r, zero(f)
    f ./= f0
    return r, f
end

"""
    integral_lengthscale(r, f) -> L

Longitudinal integral scale `L₁₁ = ∫₀^∞ f(r) dr` (trapezoidal), truncated at
the first zero-crossing of `f` to avoid integrating the (weakly negative,
periodic-wrap) tail past where the correlation has physically decayed.
"""
function integral_lengthscale(r, f)
    i = findfirst(<(0), f)
    n = i === nothing ? length(f) : i
    n < 2 && return zero(eltype(f))
    return sum((f[k] + f[k+1]) / 2 * (r[k+1] - r[k]) for k in 1:(n - 1))
end
