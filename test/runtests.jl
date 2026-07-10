using HIT3D
using HIT3D.Grids: ∂x!, ∂y!, ∂z!, laplacian!, project!, dealias!
using HIT3D.RHS: rhs!
using JLD2: jldopen
using Test

# 4D physical field with the same 3D pattern in every velocity component.
fill3(f) = repeat(f, 1, 1, 1, 3)

@testset "HIT3D" begin
    @testset "Grid" begin
        N = 32
        g = Grid(N; T = Float64)

        # shapes and structural properties
        @test size(g.kx) == (N ÷ 2 + 1, 1, 1)
        @test size(g.ky) == (1, N, 1)
        @test size(g.kz) == (1, 1, N)
        @test size(g.k2) == (N ÷ 2 + 1, N, N)
        @test g.invk2[1, 1, 1] == 0

        # dealias mask: per-direction 2/3 box, false at k = 0
        cut = N ÷ 3
        @test !g.dealias[1, 1, 1]
        @test g.dealias[cut + 1, 1, 1]        # nx = cut retained
        @test !g.dealias[cut + 2, 1, 1]       # nx = cut + 1 masked
        @test g.dealias[1, N - cut + 1, 1]    # ny = -cut retained
        @test !g.dealias[1, N - cut, 1]       # ny = -(cut + 1) masked
        @test g.dealias[2, 1, 1]              # low modes retained

        # plan roundtrip (note: c2r destroys its input, hence the copy)
        u = randn(N, N, N, 3)
        û = g.plan * u
        @test g.iplan * copy(û) ≈ u

        # spectral derivatives / Laplacian exact on a trig field
        x = reshape((0:N-1) .* (2π / N), :, 1, 1)
        y = reshape((0:N-1) .* (2π / N), 1, :, 1)
        z = reshape((0:N-1) .* (2π / N), 1, 1, :)
        f = @. sin(x) * cos(2y) * sin(3z)
        û = g.plan * fill3(f)
        out = similar(û)

        ∂x!(out, û, g)
        @test g.iplan * out ≈ fill3(@. cos(x) * cos(2y) * sin(3z))
        ∂y!(out, û, g)
        @test g.iplan * out ≈ fill3(@. -2 * sin(x) * sin(2y) * sin(3z))
        ∂z!(out, û, g)
        @test g.iplan * out ≈ fill3(@. 3 * sin(x) * cos(2y) * cos(3z))
        laplacian!(out, û, g)
        @test g.iplan * out ≈ fill3(@. -(1 + 4 + 9) * f)

        # projection: output divergence-free, and idempotent
        û_r = randn(ComplexF64, N ÷ 2 + 1, N, N, 3)
        project!(û_r, g)
        v1 = view(û_r, :, :, :, 1)
        v2 = view(û_r, :, :, :, 2)
        v3 = view(û_r, :, :, :, 3)
        div̂ = @. g.kx * v1 + g.ky * v2 + g.kz * v3
        @test maximum(abs, div̂) < 1e-10
        û_r2 = copy(û_r)
        project!(û_r2, g)
        @test û_r2 ≈ û_r
    end

    @testset "NavierStokes RHS (Beltrami decay)" begin
        # ABC flow (A = B = C = 1) is a Beltrami field: ω = u, so u × ω ≡ 0
        # and the exact Navier–Stokes solution is pure viscous decay
        # û(t) = û₀ e^{-νt} (all active modes have |k|² = 1). This exercises
        # the full rhs! pipeline end-to-end: any error in the transforms,
        # cross product, dealiasing, or projection breaks it loudly.
        N = 32
        ν = 0.05
        g = Grid(N; T = Float64)

        x = reshape((0:N-1) .* (2π / N), :, 1, 1)
        y = reshape((0:N-1) .* (2π / N), 1, :, 1)
        z = reshape((0:N-1) .* (2π / N), 1, 1, :)
        u = zeros(N, N, N, 3)
        @. u[:, :, :, 1] = sin(z) + cos(y)
        @. u[:, :, :, 2] = sin(x) + cos(z)
        @. u[:, :, :, 3] = sin(y) + cos(x)
        û0 = g.plan * u

        r = NavierStokes(g; ν)

        # single evaluation: dû must be exactly the viscous term −ν û₀
        dû = similar(û0)
        rhs!(dû, û0, r, 0.0)
        @test dû ≈ -ν .* û0

        # full time integration against the analytic decay
        û = copy(û0)
        scheme = RK4(û)
        dt, nsteps = 1e-2, 100
        evolve!(û, r, scheme, dt, nsteps)
        @test û ≈ exp(-ν * dt * nsteps) .* û0 rtol = 1e-10
    end

    @testset "Diagnostics" begin
        N = 32
        g = Grid(N; T = Float64)

        # Parseval: spectral energy (with Hermitian double-counting) equals
        # the physical-space mean ½⟨|u|²⟩
        u = randn(N, N, N, 3)
        û = g.plan * u
        @test energy(û, g) ≈ 0.5 * sum(abs2, u) / N^3

        # spectrum shells sum back to the total energy
        û = spectral_state(g)
        random_field!(û, g)
        k, Ek = energy_spectrum(û, g)
        @test sum(Ek) ≈ energy(û, g)

        # Beltrami field: ω = u, so enstrophy = energy exactly, and the
        # derived scales follow analytically
        x = reshape((0:N-1) .* (2π / N), :, 1, 1)
        y = reshape((0:N-1) .* (2π / N), 1, :, 1)
        z = reshape((0:N-1) .* (2π / N), 1, 1, :)
        ub = zeros(N, N, N, 3)
        @. ub[:, :, :, 1] = sin(z) + cos(y)
        @. ub[:, :, :, 2] = sin(x) + cos(z)
        @. ub[:, :, :, 3] = sin(y) + cos(x)
        ûb = g.plan * ub
        ν = 0.05
        @test enstrophy(ûb, g) ≈ energy(ûb, g)
        @test dissipation(ûb, g, ν) ≈ 2ν * energy(ûb, g)
        @test taylor_microscale(ûb, g) ≈ sqrt(5)

        # taylor_reynolds: the field-based method delegates to the scalar
        # (E, ε, ν) formula — same one used by postprocessing on a dense
        # energy_budget series, so there is exactly one implementation
        E, ε = energy(ûb, g), dissipation(ûb, g, ν)
        @test taylor_reynolds(ûb, g, ν) ≈ taylor_reynolds(E, ε, ν)
    end

    @testset "Integrators / callbacks" begin
        N = 32
        ν = 0.05
        g = Grid(N; T = Float64)
        x = reshape((0:N-1) .* (2π / N), :, 1, 1)
        y = reshape((0:N-1) .* (2π / N), 1, :, 1)
        z = reshape((0:N-1) .* (2π / N), 1, 1, :)
        u = zeros(N, N, N, 3)
        @. u[:, :, :, 1] = sin(z) + cos(y)
        @. u[:, :, :, 2] = sin(x) + cos(z)
        @. u[:, :, :, 3] = sin(y) + cos(x)
        û = g.plan * u

        r = NavierStokes(g; ν)
        scheme = RK4(û)
        dt, nsteps = 1e-2, 100

        E = Diagnostic(s -> energy(s.û, s.grid); every = 10)
        ncalls = Ref(0)
        probe = Callback(s -> ncalls[] += 1; every_time = 0.25)
        path = joinpath(mktempdir(), "snap.jld2")
        io = FieldWriter(path; every = 50)

        evolve!(û, r, scheme, dt, nsteps; callbacks = (E, probe, io))

        # Diagnostic: fires at step 0 and every 10th step; energy follows
        # the analytic Beltrami decay e^{-2νt}
        @test length(E.times) == nsteps ÷ 10 + 1
        @test E.times[2] ≈ 10dt
        @test E.values ≈ E.values[1] .* exp.(-2ν .* E.times) rtol = 1e-8

        # Callback with a time-based schedule: initial + crossings of 0.25
        @test 4 <= ncalls[] <= 6

        # FieldWriter: snapshots at steps 0, 50, 100; final one matches û;
        # self-describing grid/ν metadata group
        jldopen(path, "r") do file
            @test sort(filter(startswith("step_"), keys(file))) ==
                  ["step_00000000", "step_00000050", "step_00000100"]
            @test file["step_00000100/û"] ≈ û
            @test file["step_00000100/t"] ≈ dt * nsteps
            @test file["grid/Nx"] == N
            @test file["grid/Lx"] ≈ 2π
            @test file["grid/nu"] == ν
        end

        # overwrite semantics: default construction deletes an existing file,
        # overwrite = false preserves it
        @test isfile(path)
        FieldWriter(path; every = 50, overwrite = false)
        @test isfile(path)
        FieldWriter(path; every = 50)
        @test !isfile(path)
    end

    @testset "Schema" begin
        @test HIT3D.stepkey(50) == "step_00000050"
        @test HIT3D.stepnum("step_00000050") == 50
        @test HIT3D.is_stepkey("step_00000050")
        @test !HIT3D.is_stepkey("grid")
        @test HIT3D.gridkey(:Nx) == "grid/Nx"
        @test HIT3D.serieskey("energy_budget") == "series/energy_budget"
        @test HIT3D.seriesfield("energy_budget", :E) ==
              "series/energy_budget/E"
    end

    @testset "report (load_run / read_series)" begin
        N = 32
        ν = 0.05
        g = Grid(N; T = Float64)
        û = spectral_state(g)
        random_field!(û, g)

        r = NavierStokes(g; ν)
        scheme = RK4(û)
        path = joinpath(mktempdir(), "snap.jld2")
        io = FieldWriter(path; every = 5)
        evolve!(û, r, scheme, 1e-3, 10; callbacks = (io,))

        # metadata round-trips through the self-describing grid group
        run = load_run(path)
        @test run.grid.Nx == run.grid.Ny == run.grid.Nz == N
        @test run.ν == ν
        @test run.T == Float64
        @test run.steps == [0, 5, 10]

        # no dense series recorded -> read_series is nothing
        @test read_series(path, "energy_budget") === nothing

        # legacy file with no "grid" group: load_run falls back to a 2π
        # cube with unknown ν, with a warning (schema seam from the
        # architecture review, now covered instead of only exercised
        # manually via the Makie extension)
        legacy = joinpath(mktempdir(), "legacy.jld2")
        jldopen(legacy, "w") do file
            file["step_00000000/t"] = 0.0
            file["step_00000000/û"] = Array(û)
        end
        run2 = @test_logs (:warn, r"no grid metadata") load_run(legacy)
        @test run2.grid.Nx == N
        @test run2.ν === nothing

        # a file with no snapshots at all is a clear error, not a raw
        # JLD2 KeyError
        empty_path = joinpath(mktempdir(), "empty.jld2")
        jldopen(_ -> nothing, empty_path, "w")
        @test_throws ErrorException load_run(empty_path)
    end

    @testset "energy budget" begin
        N = 32
        ν = 0.05
        g = Grid(N; T = Float64)
        û = spectral_state(g)
        random_field!(û, g)

        # injection: exact per forcing type
        @test injection(NoForcing(), û, g) == 0
        f = BandForcing(g; ε = 0.3, kmin = 1.0, kmax = 3.0)
        @test injection(f, û, g) == 0.3
        @test injection(f, zero(û), g) == 0        # empty band → skip guard

        # apply_forcing! and injection agree: ⟨u·f⟩ measured from a forcing
        # evaluation equals the reported P (f̂ ∝ û here, so the Hermitian-
        # weighted spectral inner product is exact)
        dû = zero(û)
        HIT3D.RHS.apply_forcing!(dû, û, nothing, f, g, 0.0)
        w = HIT3D.Grids.hermitian_weights(g)
        P_meas = sum(w .* real.(conj.(û) .* dû)) / N^6
        @test P_meas ≈ injection(f, û, g)

        # LinearForcing: P = ⟨u·(Au)⟩ = 2A·E exactly, and the measured
        # ⟨u·f⟩ from an evaluation agrees
        lf = LinearForcing(0.1333)
        @test injection(lf, û, g) ≈ 2 * lf.A * energy(û, g)
        dû_lf = zero(û)
        HIT3D.RHS.apply_forcing!(dû_lf, û, nothing, lf, g, 0.0)
        P_lf = sum(w .* real.(conj.(û) .* dû_lf)) / N^6
        @test P_lf ≈ injection(lf, û, g)

        # energy_budget: raw terms match the individual diagnostics
        b = energy_budget(û, g, ν, f)
        @test b.E ≈ energy(û, g)
        @test b.ε ≈ dissipation(û, g, ν)
        @test b.P == 0.3

        # state-tuple forwarding method
        r = NavierStokes(g; ν)
        state = (; û, t = 0.0, step = 0, rhs = r, grid = g)
        bs = energy_budget(state)
        @test bs.E ≈ b.E && bs.ε ≈ b.ε && bs.P == 0

        # short viscous decay: the finite-difference residual
        # dE/dt + ε − P is small at the recorded samples
        d = Diagnostic(energy_budget;
                       valuetype = @NamedTuple{E::Float64, ε::Float64,
                                               P::Float64})
        dt, nsteps = 1e-3, 50
        evolve!(û, r, RK4(û), dt, nsteps; callbacks = (d,))
        E_ = [v.E for v in d.values]
        ε_ = [v.ε for v in d.values]
        dEdt = (E_[3:end] .- E_[1:end-2]) ./ (2dt)   # central differences
        resid = dEdt .+ ε_[2:end-1]
        @test maximum(abs, resid) < 1e-4 * maximum(ε_)

        # save_series: columnar round-trip into a jld2, named-tuple fields
        path = joinpath(mktempdir(), "series.jld2")
        save_series(path, "energy_budget", d)
        jldopen(path, "r") do file
            @test file["series/energy_budget/t"] == d.times
            @test file["series/energy_budget/E"] == E_
            @test file["series/energy_budget/ε"] == ε_
            @test file["series/energy_budget/P"] == zeros(nsteps + 1)
        end
        @test_throws ErrorException save_series(path, "energy_budget", d)

        # scalar diagnostics persist as a single values vector
        ds = Diagnostic(s -> energy(s.û, s.grid))
        push!(ds.times, 0.0); push!(ds.values, 1.5)
        save_series(path, "energy", ds)
        jldopen(path, "r") do file
            @test file["series/energy/values"] == [1.5]
        end

        # Diagnostic(...; path, name): evolve! persists the series
        # automatically, no manual save_series call needed
        autopath = joinpath(mktempdir(), "auto.jld2")
        io = FieldWriter(autopath; every = 1000)  # just for the grid group
        d = Diagnostic(energy_budget; every = 1,
                       valuetype = @NamedTuple{E::Float64, ε::Float64,
                                               P::Float64},
                       path = autopath, name = "energy_budget")
        evolve!(û, r, RK4(û), dt, 5; callbacks = (io, d))
        series = read_series(autopath, "energy_budget")
        @test series !== nothing
        @test series.t == d.times
        @test series.E == [v.E for v in d.values]

        # path without name is a clear construction-time error
        @test_throws ErrorException Diagnostic(energy_budget; path = autopath)
    end

    @testset "progress printing" begin
        g = Grid(16; T = Float64)
        û = spectral_state(g)
        random_field!(û, g)
        r = NavierStokes(g; ν = 0.05)

        capture(f) = begin
            path, io = mktemp()
            redirect_stdout(f, io)
            close(io)
            read(path, String)
        end

        # explicit interval: fires at 5, 10 — and always at the last step
        out = capture(() -> evolve!(û, r, RK4(û), 1e-3, 10; progress = 5))
        @test count(==('\n'), out) == 2
        @test occursin("step 10/10 (100%)", out)
        @test occursin("t = ", out) && occursin("eta ", out)

        # default is silent
        out = capture(() -> evolve!(û, r, RK4(û), 1e-3, 5))
        @test isempty(out)

        # progress = true picks its own interval, ends at 100%
        out = capture(() -> evolve!(û, r, RK4(û), 1e-3, 5; progress = true))
        @test !isempty(out) && occursin("(100%)", out)
    end

    @testset "labels / show" begin
        g = Grid(16; T = Float64)
        @test label(g) == "N16"
        @test label(Grid(16, 8, 4; T = Float64)) == "N16x8x4"

        r = NavierStokes(g; ν = 1e-3)
        @test label(r) == "NavierStokes_nu0.001_NoForcing"

        f = BandForcing(g; ε = 0.1, kmin = 1.0, kmax = 2.5)
        @test label(f) == "BandForcing_eps0.1_k1-2.5"
        rf = NavierStokes(g; ν = 1e-3, forcing = f)
        @test label(rf) == "NavierStokes_nu0.001_BandForcing_eps0.1_k1-2.5"

        lf = LinearForcing(0.1333)
        @test label(lf) == "LinearForcing_A0.1333"
        @test label(NavierStokes(g; ν = 1e-3, forcing = lf)) ==
              "NavierStokes_nu0.001_LinearForcing_A0.1333"
        @test repr(lf) == "LinearForcing(A=0.1333)"

        scheme = RK4(spectral_state(g))
        @test label(scheme) == "RK4"

        # %g formatting is precision-independent (no "0.001f0" in paths)
        g32 = Grid(16)   # Float32 default
        @test label(NavierStokes(g32; ν = 1f-3)) ==
              "NavierStokes_nu0.001_NoForcing"

        # show: pretty, and never dumps the scratch buffers
        @test repr(g) == "Grid{Float64}(16×16×16, L=(6.28319, 6.28319, 6.28319))"
        @test repr(f) == "BandForcing(ε=0.1, k∈[1, 2.5])"
        @test repr(rf) ==
              "NavierStokes(ν=0.001, forcing=BandForcing(ε=0.1, k∈[1, 2.5]))"
        @test occursin("ComplexF64", repr(scheme))
        @test length(repr(rf)) < 200

        # plotting stubs give a helpful error without a Makie backend
        @test_throws ErrorException plot_summary("nonexistent.jld2")
    end
end
