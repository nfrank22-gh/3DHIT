# GPU functionality check: forward solver on any backend, Enzyme VJP where
# supported. One script for every machine —
#
# setup (once, from the repo root):
#   julia --project=scripts/gpu -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
# run:
#   julia --project=scripts/gpu scripts/gpu/test_gpu.jl          # metal on macOS, cuda elsewhere
#   julia --project=scripts/gpu scripts/gpu/test_gpu.jl cuda     # explicit backend: cpu|metal|cuda
#
# What runs where:
#   forward pass (Float32)      — all backends
#   Enzyme FFT-plan rules       — cpu/cuda; attempted on metal (expected to fail:
#                                 Enzyme has no Metal support — a failure there is
#                                 informational, not a solver bug)
#   Enzyme single-step VJP      — cpu/cuda only (needs Float64 for the FD identity;
#                                 Metal has no Float64)

using Test, LinearAlgebra, Random
using Adapt: adapt

backend = lowercase(get(ARGS, 1, Sys.isapple() ? "metal" : "cuda"))
if backend == "metal"
    using Metal
    ArrayType = MtlArray
    Metal.functional() || error("Metal not functional on this machine")
elseif backend == "cuda"
    using CUDA
    ArrayType = CuArray
    CUDA.functional() || error("CUDA not functional on this machine")
elseif backend == "cpu"
    ArrayType = Array
else
    error("unknown backend '$backend' (expected cpu, metal, or cuda)")
end
has_f64 = backend != "metal"   # Metal has no Float64
@info "Testing HIT3D on backend=$backend (ArrayType=$ArrayType)"

using HIT3D
using HIT3D.Grids: dealias!, project!
using HIT3D.Integrators: step!
using Enzyme: autodiff, set_runtime_activity, Reverse, Const, Duplicated

Random.seed!(1234)
dotr(a, b) = real(dot(a, b))

"ABC-flow Beltrami field (ω = u ⇒ exact solution is pure viscous decay)."
function beltrami(g)
    T = eltype(g.k2)
    x = similar(g.k2, T, (g.Nx, 1, 1))
    y = similar(g.k2, T, (1, g.Ny, 1))
    z = similar(g.k2, T, (1, 1, g.Nz))
    copyto!(x, reshape(collect(T, (0:g.Nx-1) .* (2π / g.Nx)), :, 1, 1))
    copyto!(y, reshape(collect(T, (0:g.Ny-1) .* (2π / g.Ny)), 1, :, 1))
    copyto!(z, reshape(collect(T, (0:g.Nz-1) .* (2π / g.Nz)), 1, 1, :))
    u = similar(g.k2, T, (g.Nx, g.Ny, g.Nz, 3))
    u1 = view(u, :, :, :, 1); u2 = view(u, :, :, :, 2); u3 = view(u, :, :, :, 3)
    @. u1 = sin(z) + cos(y)
    @. u2 = sin(x) + cos(z)
    @. u3 = sin(y) + cos(x)
    û = spectral_state(g)
    mul!(û, g.plan, u)
    return û
end

# max |k·û| relative to the field magnitude (stored rfft coefficients are
# unnormalized, so an absolute threshold would be resolution-dependent)
reldiv(û, g) = maximum(abs, @. g.kx * $view(û, :, :, :, 1) +
                            g.ky * $view(û, :, :, :, 2) +
                            g.kz * $view(û, :, :, :, 3)) / maximum(abs, û)

# Adjoint identity ⟨ȳ, P v⟩ = ⟨Pᵀȳ, v⟩ for both plans, exercising the
# widened Enzyme mul! rules on this backend's plan types.
function plan_rule_tests(g, T, rtol)
    N = g.Nx
    todev(a) = adapt(ArrayType, a)
    randspec() = g.plan * todev(randn(T, N, N, N, 3))
    f!(y, p, x) = (mul!(y, p, x); nothing)

    # forward r2c plan
    ȳ = randspec()
    x̄ = todev(zeros(T, N, N, N, 3))
    autodiff(set_runtime_activity(Reverse), Const(f!), Const,
             Duplicated(similar(ȳ), copy(ȳ)), Const(g.plan),
             Duplicated(todev(randn(T, N, N, N, 3)), x̄))
    v = todev(randn(T, N, N, N, 3))
    @test dotr(ȳ, g.plan * v) ≈ dot(x̄, v) rtol = rtol

    # inverse plan (ScaledPlan around c2r; c2r destroys input, hence copies)
    w = todev(randn(T, N, N, N, 3))
    x̄c = todev(zeros(Complex{T}, N ÷ 2 + 1, N, N, 3))
    autodiff(set_runtime_activity(Reverse), Const(f!), Const,
             Duplicated(similar(w), copy(w)), Const(g.iplan),
             Duplicated(randspec(), x̄c))
    vc = randspec()
    @test dot(w, g.iplan * copy(vc)) ≈ dotr(x̄c, vc) rtol = rtol
    return nothing
end

@testset "HIT3D on $backend" begin
    @testset "forward: Beltrami viscous decay (Float32, N=32)" begin
        T = Float32
        g = Grid(32; T, ArrayType)
        ν = T(0.05)
        û0 = beltrami(g)
        û = copy(û0)
        r = NavierStokes(g; ν)
        dt, nsteps = T(1e-2), 50
        evolve!(û, r, RK4(û), dt, nsteps)
        @test Array(û) ≈ exp(-ν * dt * nsteps) .* Array(û0) rtol = 1e-3
        @test reldiv(û, g) < 1e-4
        @test isfinite(energy(û, g))
    end

    @testset "forward: GPU trajectory matches CPU (Float32, N=32)" begin
        T = Float32
        g_cpu = Grid(32; T)
        û_cpu = spectral_state(g_cpu)
        random_field!(û_cpu, g_cpu)
        g_dev = Grid(32; T, ArrayType)
        û_dev = adapt(ArrayType, copy(û_cpu))
        ν, dt, nsteps = T(0.02), T(5e-3), 20
        evolve!(û_cpu, NavierStokes(g_cpu; ν), RK4(û_cpu), dt, nsteps)
        evolve!(û_dev, NavierStokes(g_dev; ν), RK4(û_dev), dt, nsteps)
        @test Array(û_dev) ≈ û_cpu rtol = 5e-3
    end

    if backend == "metal"
        # Informational only: the mul! rule body is plain array code, but
        # Enzyme must still compile the surrounding function for MtlArrays,
        # which it does not officially support.
        try
            g = Grid(8; T = Float32, ArrayType)
            plan_rule_tests(g, Float32, 1e-4)
            @info "Enzyme FFT-plan rules unexpectedly work on Metal"
        catch err
            @warn "Enzyme FFT-plan rules do not run on Metal (expected)" typeof(err)
        end
        @info "Skipping VJP tests: Metal has no Float64 / Enzyme support. " *
              "Run this script on the CUDA machine for the AD checks."
    else
        @testset "Enzyme FFT-plan rules (adjoint identity, Float64, N=8)" begin
            g = Grid(8; T = Float64, ArrayType)
            plan_rule_tests(g, Float64, 1e-12)
        end

        @testset "Enzyme single-step VJP vs FD-JVP (Float64, N=8)" begin
            N = 8
            T = Float64
            g = Grid(N; T, ArrayType)
            randspec() = g.plan * adapt(ArrayType, randn(T, N, N, N, 3))
            û0 = randspec()
            dealias!(û0, g)
            project!(û0, g)
            r = NavierStokes(g; ν = T(0.02))
            s = RK4(û0)
            ws = VJPWorkspace(r, s)
            dt = T(1e-3)
            t = zero(T)
            stepnm(u) = (v = copy(u); step!(v, r, s, dt, t); v)

            for _ in 1:3
                v = randspec()
                w = randspec()
                ε = T(1e-5) * norm(û0) / norm(v)
                Jv = (stepnm(û0 .+ ε .* v) .- stepnm(û0 .- ε .* v)) ./ (2ε)
                ū = copy(w)
                vjp_step!(ū, û0, r, s, dt, t, ws)
                @test dotr(w, Jv) ≈ dotr(ū, v) rtol = 1e-6
            end

            # purity: primal state untouched, repeat calls deterministic
            û_before = copy(û0)
            w = randspec()
            ū1 = copy(w); vjp_step!(ū1, û0, r, s, dt, t, ws)
            ū2 = copy(w); vjp_step!(ū2, û0, r, s, dt, t, ws)
            @test ū1 == ū2
            @test û0 == û_before
        end
    end
end
