"""
    NavierStokes{T}

Incompressible Navier–Stokes in spectral space, rotational form:

    dû/dt = P(k) [ (u × ω)^ + f̂ ] − ν k² û

Evaluation pipeline (all in-place, dominated by the FFTs):
  1. ω̂ = ik × û in the spectral scratch; transform ω̂, then a copy of û, to
     physical space (FFTW's multidim c2r transforms destroy their input even
     out-of-place, so û itself is never transformed — this ordering is
     load-bearing)
  2. compute u × ω pointwise in physical space (own buffer `nl_phys`)
  3. forward-transform into `dû` and add the forcing — BEFORE projection, so
     a non-solenoidal forcing's gradient part is absorbed into the pressure
     (the ik·f̂/|k|² term of the eliminated pressure equation)
  4. dealias (2/3 mask, which also zeroes k = 0) and project divergence-free
  5. add the viscous term −ν k² û (û stays exactly dealiased, so it needs no
     extra masking)

Owns its scratch buffers so multiple RHS objects can share one `Grid`.
"""
struct NavierStokes{T, F <: AbstractForcing, G <: Grid, AP, AS} <: AbstractRHS
    ν::T              # kinematic viscosity
    forcing::F
    grid::G
    # --- scratch buffers (preallocated on the grid's backend) ---
    u_phys::AP        # physical-space velocity    (Nx, Ny, Nz, 3)
    ω_phys::AP        # physical-space vorticity   (Nx, Ny, Nz, 3)
    nl_phys::AP       # physical-space u × ω       (Nx, Ny, Nz, 3)
    scratch_spec::AS  # spectral work array        (Nx÷2+1, Ny, Nz, 3)
end

"""
    NavierStokes(g::Grid; ν, forcing = NoForcing())

Construct the RHS and allocate its scratch buffers on the grid's backend.
"""
function NavierStokes(g::Grid; ν, forcing = NoForcing())
    T = eltype(g.k2)
    u_phys  = similar(g.k2, T, (g.Nx, g.Ny, g.Nz, 3))
    ω_phys  = similar(u_phys)
    nl_phys = similar(u_phys)
    scratch_spec = similar(g.k2, Complex{T}, (g.Nx ÷ 2 + 1, g.Ny, g.Nz, 3))
    return NavierStokes(T(ν), forcing, g, u_phys, ω_phys, nl_phys,
                        scratch_spec)
end

function rhs!(dû, û, r::NavierStokes, t)
    g = r.grid
    ŝ = r.scratch_spec

    û1 = view(û, :, :, :, 1); û2 = view(û, :, :, :, 2); û3 = view(û, :, :, :, 3)
    ŝ1 = view(ŝ, :, :, :, 1); ŝ2 = view(ŝ, :, :, :, 2); ŝ3 = view(ŝ, :, :, :, 3)

    # 1. ω̂ = ik × û  (into the spectral scratch)
    @. ŝ1 = im * (g.ky * û3 - g.kz * û2)
    @. ŝ2 = im * (g.kz * û1 - g.kx * û3)
    @. ŝ3 = im * (g.kx * û2 - g.ky * û1)

    # 2. to physical space. c2r destroys its input: ω̂ first (expendable),
    #    then a copy of û — never û itself.
    mul!(r.ω_phys, g.iplan, ŝ)
    copyto!(ŝ, û)
    mul!(r.u_phys, g.iplan, ŝ)

    # 3. nonlinear term u × ω, pointwise
    u1 = view(r.u_phys, :, :, :, 1); ω1 = view(r.ω_phys, :, :, :, 1)
    u2 = view(r.u_phys, :, :, :, 2); ω2 = view(r.ω_phys, :, :, :, 2)
    u3 = view(r.u_phys, :, :, :, 3); ω3 = view(r.ω_phys, :, :, :, 3)
    n1 = view(r.nl_phys, :, :, :, 1)
    n2 = view(r.nl_phys, :, :, :, 2)
    n3 = view(r.nl_phys, :, :, :, 3)
    @. n1 = u2 * ω3 - u3 * ω2
    @. n2 = u3 * ω1 - u1 * ω3
    @. n3 = u1 * ω2 - u2 * ω1

    # 4. back to spectral space; forcing enters before dealias/projection
    mul!(dû, g.plan, r.nl_phys)
    apply_forcing!(dû, û, r.u_phys, r.forcing, g, t)

    # 5. dealias (also zeroes k = 0) and project; ŝ is free again after the
    #    transforms, so its first component serves as the projection scratch
    dealias!(dû, g)
    project!(dû, g, ŝ1)

    # 6. viscous term
    @. dû -= r.ν * g.k2 * û
    return dû
end

"""Diagonal viscous operator `-ν k²` for integrating-factor schemes.
Allocates a fresh array on each call."""
linear_operator(r::NavierStokes) = -r.ν .* r.grid.k2
