"""
    Grids

Geometry + precomputed spectral operators for a triply periodic box.

The `Grid` is immutable-in-spirit and shareable between multiple RHS objects.
It owns no scratch buffers (those live in RHS structs). All wavenumber arrays
are allocated on the target backend at construction, so downstream code is
backend-agnostic broadcasts.
"""
module Grids

using AbstractFFTs
using FFTW
using Adapt

export Grid, ∂x!, ∂y!, ∂z!, laplacian!, project!, dealias!, hermitian_weights

"""
    Grid{T}

Parameters and spectral operators for an `Nx × Ny × Nz` periodic box of size
`Lx × Ly × Lz`.

Fields
- `Nx, Ny, Nz`   : grid points per direction
- `Lx, Ly, Lz`   : domain lengths
- `kx, ky, kz`   : wavenumber arrays, shaped for broadcasting against the
                   spectral state — `(Nx÷2+1, 1, 1)`, `(1, Ny, 1)`, `(1, 1, Nz)`
                   (rfft layout: kx holds only non-negative wavenumbers)
- `k2`           : |k|² on the full spectral grid `(Nx÷2+1, Ny, Nz)`
- `invk2`        : 1/|k|² with the k = 0 mode set to zero
- `dealias`      : Bool mask — per-direction (Orszag) 2/3 rule, i.e. the
                   product of three 1D masks `|n_i| ≤ N_i÷3`; also false at
                   k = 0, enforcing the zero-mean invariant on anything the
                   mask is applied to
- `plan, iplan`  : batched rfft / irfft plans over dims 1:3 of the 4D state

Type parameters are left loose so the same struct works with Array, CuArray,
MtlArray, etc.
"""
struct Grid{T, AKX, AKY, AKZ, AK2, AM, PF, PI}
    Nx::Int
    Ny::Int
    Nz::Int
    Lx::T
    Ly::T
    Lz::T
    kx::AKX
    ky::AKY
    kz::AKZ
    k2::AK2
    invk2::AK2
    dealias::AM
    plan::PF
    iplan::PI
end

"""
    Grid(Nx, Ny, Nz; Lx=2π, Ly=2π, Lz=2π, T=Float32, ArrayType=Array)
    Grid(N; kwargs...)   # cubic convenience constructor

Build the grid, precompute wavenumber arrays / masks on `ArrayType`, and
create batched rfft plans for the `(Nx÷2+1, Ny, Nz, 3)` state layout.
"""
function Grid(Nx::Int, Ny::Int, Nz::Int;
              Lx = 2π, Ly = 2π, Lz = 2π,
              T::Type = Float32, ArrayType = Array)
    Lx, Ly, Lz = T(Lx), T(Ly), T(Lz)

    # Integer frequencies (rfft layout in x, full fft layout in y/z).
    nx = collect(T, 0:(Nx ÷ 2))
    ny = collect(T, fftfreq(Ny, Ny))
    nz = collect(T, fftfreq(Nz, Nz))

    kx = adapt(ArrayType, reshape((T(2) * T(π) / Lx) .* nx, :, 1, 1))
    ky = adapt(ArrayType, reshape((T(2) * T(π) / Ly) .* ny, 1, :, 1))
    kz = adapt(ArrayType, reshape((T(2) * T(π) / Lz) .* nz, 1, 1, :))

    k2 = @. kx^2 + ky^2 + kz^2
    invk2 = @. ifelse(iszero(k2), zero(T), inv(k2))

    # Per-direction 2/3-rule box mask; k = 0 is masked too (zero-mean
    # invariant — see design log). Nyquist modes fall outside automatically.
    mx = abs.(nx) .<= Nx ÷ 3
    my = abs.(ny) .<= Ny ÷ 3
    mz = abs.(nz) .<= Nz ÷ 3
    mask = reshape(mx, :, 1, 1) .& reshape(my, 1, :, 1) .& reshape(mz, 1, 1, :)
    mask[1, 1, 1] = false
    dealias = adapt(ArrayType, mask)

    # Batched plans over dims 1:3 of the 4D state. FFTW's MEASURE overwrites
    # the planning buffers, which is fine — they're throwaway. GPU backends
    # take no flags.
    tmp_r = adapt(ArrayType, zeros(T, Nx, Ny, Nz, 3))
    tmp_c = adapt(ArrayType, zeros(Complex{T}, Nx ÷ 2 + 1, Ny, Nz, 3))
    if tmp_r isa Array
        plan  = plan_rfft(tmp_r, 1:3; flags = FFTW.MEASURE)
        iplan = plan_irfft(tmp_c, Nx, 1:3; flags = FFTW.MEASURE)
    else
        plan  = plan_rfft(tmp_r, 1:3)
        iplan = plan_irfft(tmp_c, Nx, 1:3)
    end

    return Grid(Nx, Ny, Nz, Lx, Ly, Lz, kx, ky, kz, k2, invk2, dealias,
                plan, iplan)
end

Grid(N::Int; kwargs...) = Grid(N, N, N; kwargs...)

# ---------------------------------------------------------------------------
# Spectral operators — all trivial broadcasts against the stored wavenumbers.
# In-place versions write into a caller-provided output array.
# ---------------------------------------------------------------------------

"""In-place spectral x-derivative: `out .= im .* g.kx .* û`."""
∂x!(out, û, g::Grid) = (@. out = im * g.kx * û; out)

"""In-place spectral y-derivative."""
∂y!(out, û, g::Grid) = (@. out = im * g.ky * û; out)

"""In-place spectral z-derivative."""
∂z!(out, û, g::Grid) = (@. out = im * g.kz * û; out)

"""In-place spectral Laplacian: `out .= -g.k2 .* û`."""
laplacian!(out, û, g::Grid) = (@. out = -g.k2 * û; out)

"""
    project!(û, g::Grid[, div])

Apply the divergence-free projection P(k) = I − k kᵀ/|k|² to the 3-component
spectral field `û` in place. `div` is an optional complex scratch array of
size `(Nx÷2+1, Ny, Nz)` holding (k·û)/|k|²; without it one is allocated, so
hot paths should pass their own. At k = 0 the projection is the identity
(`invk2` is zero there); the zero mode is handled by the dealias mask instead.
"""
function project!(û, g::Grid, div)
    û1 = view(û, :, :, :, 1)
    û2 = view(û, :, :, :, 2)
    û3 = view(û, :, :, :, 3)
    @. div = (g.kx * û1 + g.ky * û2 + g.kz * û3) * g.invk2
    @. û1 -= g.kx * div
    @. û2 -= g.ky * div
    @. û3 -= g.kz * div
    return û
end

project!(û, g::Grid) = project!(û, g, similar(û, ntuple(i -> size(û, i), 3)))

"""Zero all modes outside the 2/3-rule mask (including k = 0), in place."""
dealias!(û, g::Grid) = (û .*= g.dealias; û)

"""
    hermitian_weights(g::Grid) -> (Nx÷2+1, 1, 1) array

Shell-sum weights for the rfft layout: 2 for kx > 0 modes (each represents a
conjugate pair not stored explicitly), 1 for the kx = 0 plane and — for even
Nx — the Nyquist plane. Broadcast against `abs2.(û)` in any reduction that
must count the full spectral grid (energy, band energy, spectra).
"""
function hermitian_weights(g::Grid)
    T = eltype(g.k2)
    w_cpu = fill(T(2), g.Nx ÷ 2 + 1)
    w_cpu[1] = one(T)
    iseven(g.Nx) && (w_cpu[end] = one(T))
    w = similar(g.k2, T, (g.Nx ÷ 2 + 1, 1, 1))
    copyto!(w, reshape(w_cpu, :, 1, 1))
    return w
end

end # module
