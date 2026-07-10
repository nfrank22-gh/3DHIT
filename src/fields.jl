"""
State allocation and initial conditions.

The solver state is a single complex 4D array `û` of size
`(Nx÷2+1, Ny, Nz, 3)` — rfft layout, velocity components in the last
dimension. These helpers allocate it on the grid's backend and fill it with
standard HIT initial conditions. All initial conditions are dealiased and
zero-mean (hard invariants of the solver — see the dealias mask).
"""

using .Grids: Grid, project!, dealias!, hermitian_weights
using LinearAlgebra: mul!
using Random: rand!

export spectral_state, random_field!, taylor_green!

"""
    spectral_state(g::Grid)

Allocate a zeroed spectral velocity state `(Nx÷2+1, Ny, Nz, 3)` of
`Complex{T}` on the same backend as the grid's wavenumber arrays.
"""
function spectral_state(g::Grid)
    T = eltype(g.k2)
    û = similar(g.k2, Complex{T}, (g.Nx ÷ 2 + 1, g.Ny, g.Nz, 3))
    fill!(û, zero(Complex{T}))
    return û
end

"""
    random_field!(û, g::Grid; spectrum, rng)

Fill `û` with a solenoidal random field whose shell-summed energy spectrum
matches `spectrum(k)` (shells of unit spacing `2π/Lx`; cubic box assumed for
the binning).

Route: backend-native `rand!` in physical space → rfft → dealias → project →
per-shell rescale. Hermitian symmetry is automatic (we transform a real
field), and the shell rescaling is diagonal in k so it preserves the
divergence-free projection.

`rng` is respected on the CPU; on GPU backends the device's global RNG is
used unless a device-compatible rng is passed.
"""
function random_field!(û, g::Grid; spectrum = k -> k^4 * exp(-2k^2),
                       rng = nothing)
    T = eltype(g.k2)
    u = similar(g.k2, T, (g.Nx, g.Ny, g.Nz, 3))
    rng === nothing ? rand!(u) : rand!(rng, u)
    u .-= T(0.5)

    mul!(û, g.plan, u)
    dealias!(û, g)          # truncation + zero mean
    project!(û, g)

    # Shell-by-shell rescale of the white-noise field to the target spectrum.
    # One-time init cost: O(kmax) full-array reductions.
    w = hermitian_weights(g)
    kmag = sqrt.(g.k2)
    k0 = T(2) * T(π) / g.Lx
    Ntot2 = (T(g.Nx) * T(g.Ny) * T(g.Nz))^2
    smax = ceil(Int, sqrt(3) * max(g.Nx, g.Ny, g.Nz) / 2) + 1
    for s in 1:smax
        lo, hi = (s - T(0.5)) * k0, (s + T(0.5)) * k0
        shell = @. (lo <= kmag) & (kmag < hi)
        Es = sum(Broadcast.instantiate(Broadcast.broadcasted(
                 (wi, mi, ui) -> wi * mi * abs2(ui), w, shell, û))) /
             (2 * Ntot2)
        Es > 0 || continue
        target = T(spectrum(s * k0))
        c = target > 0 ? sqrt(target / Es) : zero(T)
        û .*= ifelse.(shell, c, one(T))
    end
    return û
end

"""
    taylor_green!(û, g::Grid)

Taylor–Green vortex initial condition (useful deterministic validation case):

    u = (sin θx cos θy cos θz, −cos θx sin θy cos θz, 0)

with θᵢ = 2π xᵢ/Lᵢ, so the field is periodic on any box.
"""
function taylor_green!(û, g::Grid)
    T = eltype(g.k2)
    x = similar(g.k2, T, (g.Nx, 1, 1))
    y = similar(g.k2, T, (1, g.Ny, 1))
    z = similar(g.k2, T, (1, 1, g.Nz))
    copyto!(x, reshape(collect(T, (0:g.Nx - 1) .* (2π / g.Nx)), :, 1, 1))
    copyto!(y, reshape(collect(T, (0:g.Ny - 1) .* (2π / g.Ny)), 1, :, 1))
    copyto!(z, reshape(collect(T, (0:g.Nz - 1) .* (2π / g.Nz)), 1, 1, :))

    u = similar(g.k2, T, (g.Nx, g.Ny, g.Nz, 3))
    u1 = view(u, :, :, :, 1)
    u2 = view(u, :, :, :, 2)
    @. u1 =  sin(x) * cos(y) * cos(z)
    @. u2 = -cos(x) * sin(y) * cos(z)
    fill!(view(u, :, :, :, 3), zero(T))

    mul!(û, g.plan, u)
    dealias!(û, g)
    return û
end
