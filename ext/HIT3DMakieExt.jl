"""
    HIT3DMakieExt

Implements `HIT3D.plot_summary`, `HIT3D.plot_energy_balance`, and
`HIT3D.plot_slices` when a Makie backend
is loaded. Written against the backend-agnostic Makie frontend only, so it
works with CairoMakie (headless PNG writing) and GLMakie alike.

Everything is reconstructed from the `.jld2` file written by `FieldWriter`
(grid metadata group + snapshots) and, for the dense time series, the
`series/` groups written by `save_series`; no live solver objects are needed.
"""
module HIT3DMakieExt

using HIT3D
using HIT3D: _fmt, _stepkeys, stepnum, _read_grid, _state_T, read_series
using HIT3D.Grids: Grid
using HIT3D.Integrators: jldopen   # JLD2 via the parent package
using Makie

# Palette: one blue for single-series panels; a light→dark single-hue ramp
# for time-ordered spectra; sequential viridis for magnitude fields and a
# diverging map with neutral midpoint for signed components. The categorical
# hues (fixed assignment, CVD-validated as a set) serve the multi-series
# energy-balance panels.
const LINE_BLUE = Makie.Colors.colorant"#2563eb"
const TIME_RAMP = Makie.cgrad([Makie.Colors.colorant"#93c5fd",
                               Makie.Colors.colorant"#1e3a8a"])
const GUIDE_GRAY = Makie.Colors.colorant"#6b7280"
const CAT_RED = Makie.Colors.colorant"#dc2626"
const CAT_GREEN = Makie.Colors.colorant"#059669"
const CAT_VIOLET = Makie.Colors.colorant"#7c3aed"
const CAT_AMBER = Makie.Colors.colorant"#d97706"

# --- summary + spectra -----------------------------------------------------
# File access (_stepkeys, stepnum, _read_grid, _state_T, read_series) is
# implemented in HIT3D's core `report.jl` — Makie-free and covered by the
# ordinary test suite — and just consumed here.

function HIT3D.plot_summary(path::AbstractString; outdir = dirname(path),
                             spectra_xlims = nothing, spectra_ylims = nothing)
    isempty(outdir) && (outdir = ".")
    mkpath(outdir)
    written = String[]

    jldopen(path, "r") do file
        keys_ = _stepkeys(file)
        isempty(keys_) && error("no snapshots found in $path")
        g, ν = _read_grid(file, _state_T(file))

        snap_ts = Float64[]
        Es = Float64[]
        εs = Float64[]
        Res = Float64[]
        ks = Float64[]
        spectra = Vector{Float64}[]
        for key in keys_
            û = file[key * "/û"]
            push!(snap_ts, file[key * "/t"])
            push!(Es, energy(û, g))
            k, Ek = energy_spectrum(û, g)
            ks = Float64.(k)
            push!(spectra, Float64.(Ek))
            if ν !== nothing
                push!(εs, dissipation(û, g, ν))
                push!(Res, taylor_reynolds(û, g, ν))
            end
        end

        # prefer the dense series saved by `save_series` for the time panels
        # (the snapshot-derived values above stay as the fallback; spectra
        # are snapshot-only either way).
        series = read_series(file, "energy_budget")
        dense = series !== nothing
        ts = snap_ts
        if dense
            ts = Float64.(series.t)
            Es = Float64.(series.E)
            εs = Float64.(series.ε)
            Res = ν === nothing ? Float64[] :
                  [taylor_reynolds(E, ε, ν) for (E, ε) in zip(Es, εs)]
        end
        series!(ax, xs, ys) = dense ?
            lines!(ax, xs, ys; color = LINE_BLUE, linewidth = 2) :
            scatterlines!(ax, xs, ys; color = LINE_BLUE, linewidth = 2,
                          markersize = 8)

        # summary.png — one measure per panel, shared time axis
        npanel = ν === nothing ? 1 : 3
        fig = Figure(size = (640, 230 * npanel))
        axE = Axis(fig[1, 1]; ylabel = "E", title = "Kinetic energy")
        series!(axE, ts, Es)
        if ν === nothing
            axE.xlabel = "t"
        else
            axε = Axis(fig[2, 1]; ylabel = "ε",
                       title = "Dissipation rate  (ν = $(_fmt(ν)))")
            series!(axε, ts, εs)
            axR = Axis(fig[3, 1]; ylabel = "Re_λ", xlabel = "t",
                       title = "Taylor-microscale Reynolds number")
            series!(axR, ts, Res)
            linkxaxes!(axE, axε, axR)
            hidexdecorations!(axE; grid = false)
            hidexdecorations!(axε; grid = false)
        end
        p = joinpath(outdir, "summary.png")
        save(p, fig)
        push!(written, p)

        # spectra.png — all snapshots, single-hue ramp by time
        fig2 = Figure(size = (640, 460))
        ax = Axis(fig2[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "k", ylabel = "E(k)", title = "Energy spectra")
        t0, t1 = first(snap_ts), last(snap_ts)
        for (t, Ek) in zip(snap_ts, spectra)
            c = t1 > t0 ? (t - t0) / (t1 - t0) : 0.0
            Ep = [e > 0 ? e : NaN for e in Ek]
            lines!(ax, ks, Ep; color = get(TIME_RAMP, c), linewidth = 2)
        end
        Colorbar(fig2[1, 2]; colormap = TIME_RAMP, colorrange = (t0, t1),
                 label = "t")
        # keep the view on the developed spectra (the initial condition's
        # steep tail would otherwise stretch the axis over ~40 decades),
        # unless the caller pinned the range (e.g. to match a paper figure)
        if spectra_ylims === nothing
            Emax = maximum(maximum, spectra)
            ylims!(ax, Emax * 1e-12, Emax * 5)
        else
            ylims!(ax, spectra_ylims...)
        end
        spectra_xlims === nothing || xlims!(ax, spectra_xlims...)
        # k^(-5/3) reference, anchored above the last (most developed)
        # snapshot's spectrum
        nk = length(ks)
        i0 = min(4, nk)
        if spectra[end][i0] > 0
            C = 3 * spectra[end][i0] * ks[i0]^(5 / 3)
            kg = ks[max(2, i0 - 2):min(nk, 4 * i0)]
            lines!(ax, kg, C .* kg .^ (-5 / 3); color = GUIDE_GRAY,
                   linestyle = :dash, linewidth = 2)
            text!(ax, kg[end], C * kg[end]^(-5 / 3); text = "k⁻⁵ᐟ³",
                  color = GUIDE_GRAY, align = (:left, :bottom))
        end
        p2 = joinpath(outdir, "spectra.png")
        save(p2, fig2)
        push!(written, p2)
    end
    return written
end

# --- energy balance --------------------------------------------------------

"""Central differences on a (possibly nonuniform) time grid, one-sided at
the ends. Needs ≥ 2 samples."""
function _ddt(t, y)
    n = length(t)
    d = similar(y)
    d[1] = (y[2] - y[1]) / (t[2] - t[1])
    d[n] = (y[n] - y[n-1]) / (t[n] - t[n-1])
    for i in 2:n-1
        d[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
    end
    return d
end

"""Cumulative trapezoidal integral of `y(t)`, starting at 0."""
function _cumtrapz(t, y)
    I = zero(y)
    for i in 2:length(t)
        I[i] = I[i-1] + (t[i] - t[i-1]) * (y[i] + y[i-1]) / 2
    end
    return I
end

function HIT3D.plot_energy_balance(path::AbstractString;
                                   outdir = dirname(path))
    isempty(outdir) && (outdir = ".")
    mkpath(outdir)

    series = read_series(path, "energy_budget")
    series === nothing &&
        error("no series/energy_budget group in $path — record it with " *
              "Diagnostic(energy_budget; ..., path = <this file>, " *
              "name = \"energy_budget\") so evolve! persists it automatically")
    ts, Es, εs, Ps = (Float64.(series.t), Float64.(series.E),
                      Float64.(series.ε), Float64.(series.P))
    length(ts) >= 2 ||
        error("energy_budget series in $path has fewer than 2 samples")

    dEdt = _ddt(ts, Es)
    resid = dEdt .+ εs .- Ps
    # cumulative form of the same budget: E(0) − ∫ε dt + ∫P dt
    recon = Es[1] .- _cumtrapz(ts, εs) .+ _cumtrapz(ts, Ps)

    fig = Figure(size = (640, 560))

    # budget terms — all rates on one panel; a perfect balance puts dE/dt
    # on top of P − ε and the residual on the zero guide
    ax1 = Axis(fig[1, 1]; ylabel = "dE/dt",
               title = "Energy budget  dE/dt = P − ε")
    hlines!(ax1, 0.0; color = GUIDE_GRAY, linestyle = :dash, linewidth = 1)
    lines!(ax1, ts, dEdt; color = LINE_BLUE, linewidth = 2,
           label = "dE/dt")
    lines!(ax1, ts, -εs; color = CAT_RED, linewidth = 2, label = "−ε")
    lines!(ax1, ts, Ps; color = CAT_GREEN, linewidth = 2, label = "P")
    lines!(ax1, ts, resid; color = CAT_VIOLET, linewidth = 2,
           label = "residual")
    axislegend(ax1; position = :lb, framevisible = false)

    ax2 = Axis(fig[2, 1]; ylabel = "E", xlabel = "t",
               title = "Cumulative budget")
    lines!(ax2, ts, Es; color = LINE_BLUE, linewidth = 2, label = "E(t)")
    lines!(ax2, ts, recon; color = CAT_AMBER, linewidth = 2,
           linestyle = :dash, label = "E(0) − ∫ε dt + ∫P dt")
    axislegend(ax2; position = :rt, framevisible = false)

    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false)

    p = joinpath(outdir, "energy_balance.png")
    save(p, fig)
    return [p]
end

# --- validation against a reference (e.g. a paper figure) ------------------

function HIT3D.plot_validation(path::AbstractString; outdir = dirname(path),
                               window = 1 // 3)
    isempty(outdir) && (outdir = ".")
    mkpath(outdir)
    written = String[]

    jldopen(path, "r") do file
        keys_ = _stepkeys(file)
        isempty(keys_) && error("no snapshots found in $path")
        g, ν = _read_grid(file, _state_T(file))
        ν === nothing &&
            error("plot_validation needs viscosity metadata (none found in $path)")

        ts = Float64[file[k * "/t"] for k in keys_]
        t0, t1 = first(ts), last(ts)
        cutoff = t1 - window * (t1 - t0)
        stat_keys = [k for (k, t) in zip(keys_, ts) if t >= cutoff]

        # accumulate stationary-window averages: shell spectra, component
        # spectra, dissipation, and pooled physical-space samples
        ks = Float64[]
        Es = Float64[]
        E11 = Float64[]; E22 = Float64[]; E33 = Float64[]
        ε = 0.0
        samples = Float64[]
        for (i, k) in enumerate(stat_keys)
            û = file[k * "/û"]
            kk, Ek = energy_spectrum(û, g)
            _, e11, e22, e33 = component_spectra(û, g)
            if i == 1
                ks, Es, E11, E22, E33 = Float64.(kk), Float64.(Ek),
                                        Float64.(e11), Float64.(e22), Float64.(e33)
            else
                Es .+= Ek; E11 .+= e11; E22 .+= e22; E33 .+= e33
            end
            ε += dissipation(û, g, ν)
            append!(samples, velocity_samples(û, g))
        end
        n = length(stat_keys)
        Es ./= n; E11 ./= n; E22 ./= n; E33 ./= n
        ε /= n

        # compensated_spectrum.png — Kolmogorov normalization, stationary avg
        kη, Ecomp = compensated_spectrum(ks, Es, ε, ν)
        keep = Ecomp .> 0
        fig1 = Figure(size = (640, 460))
        ax1 = Axis(fig1[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "k·η", ylabel = "E(k)·ε^(-2/3)·k^(5/3)",
                  title = "Compensated spectrum (stationary window)")
        scatterlines!(ax1, kη[keep], Ecomp[keep]; color = LINE_BLUE,
                     linewidth = 2, markersize = 6)
        hlines!(ax1, [1.5]; color = GUIDE_GRAY, linestyle = :dash)
        text!(ax1, kη[keep][end], 1.5; text = "C_K ≈ 1.5", color = GUIDE_GRAY,
             align = (:right, :bottom))
        p1 = joinpath(outdir, "compensated_spectrum.png")
        save(p1, fig1)
        push!(written, p1)

        # isotropy_spectra.png — component spectra, should coincide
        fig2 = Figure(size = (640, 460))
        ax2 = Axis(fig2[1, 1]; xscale = log10, yscale = log10,
                  xlabel = "k", ylabel = "Eᵢᵢ(k)",
                  title = "Component spectra (isotropy check)")
        for (Ei, lbl, color) in ((E11, "E₁₁", CAT_RED), (E22, "E₂₂", CAT_GREEN),
                                 (E33, "E₃₃", CAT_VIOLET))
            k2 = Ei .> 0
            lines!(ax2, ks[k2], Ei[k2]; color, label = lbl, linewidth = 2)
        end
        axislegend(ax2; position = :lb, framevisible = false)
        p2 = joinpath(outdir, "isotropy_spectra.png")
        save(p2, fig2)
        push!(written, p2)

        # velocity_pdf.png — standardized PDF vs unit Gaussian
        mom = velocity_moments(samples)
        z = (samples .- mom.mean) ./ sqrt(mom.variance)
        fig3 = Figure(size = (640, 460))
        ax3 = Axis(fig3[1, 1]; xlabel = "(u₁ − ⟨u₁⟩)/u₁′", ylabel = "PDF",
                  yscale = log10,
                  title = "Velocity PDF  (skewness = $(_fmt(mom.skewness)), " *
                          "flatness = $(_fmt(mom.flatness)))")
        hist!(ax3, z; normalization = :pdf, bins = 60,
             color = (LINE_BLUE, 0.5), strokewidth = 0, label = "measured")
        xs = range(-6, 6; length = 200)
        lines!(ax3, xs, @.(exp(-xs^2 / 2) / sqrt(2π)); color = GUIDE_GRAY,
              linestyle = :dash, linewidth = 2, label = "Gaussian")
        axislegend(ax3; position = :rt, framevisible = false)
        p3 = joinpath(outdir, "velocity_pdf.png")
        save(p3, fig3)
        push!(written, p3)
    end
    return written
end

# --- velocity slices -------------------------------------------------------

# plane -> (in-plane dims, cut dim, axis labels)
const _PLANES = Dict(:xy => ((1, 2), 3, ("x", "y")),
                     :xz => ((1, 3), 2, ("x", "z")),
                     :yz => ((2, 3), 1, ("y", "z")))

function HIT3D.plot_slices(path::AbstractString; steps = :auto,
                           component = :mag, plane = :xy, index = nothing,
                           outdir = dirname(path))
    haskey(_PLANES, plane) ||
        error("plane must be :xy, :xz, or :yz (got $plane)")
    component in (:mag, :u1, :u2, :u3) ||
        error("component must be :mag, :u1, :u2, or :u3 (got $component)")
    isempty(outdir) && (outdir = ".")
    mkpath(outdir)

    p = jldopen(path, "r") do file
        keys_ = _stepkeys(file)
        isempty(keys_) && error("no snapshots found in $path")
        g, _ = _read_grid(file, _state_T(file))

        available = stepnum.(keys_)
        picked = if steps === :auto
            unique([1, (length(keys_) + 1) ÷ 2, length(keys_)])
        else
            [something(findfirst(==(s), available),
                       error("step $s not in file (has $(available))"))
             for s in steps]
        end

        (dims, cutdim, (xlab, ylab)) = _PLANES[plane]
        Ns = (g.Nx, g.Ny, g.Nz)
        Ls = (g.Lx, g.Ly, g.Lz)
        icut = index === nothing ? Ns[cutdim] ÷ 2 + 1 : index
        coords(d) = (0:Ns[d]-1) .* (Ls[d] / Ns[d])

        slices = Matrix{Float64}[]
        ts = Float64[]
        for i in picked
            û = file[keys_[i] * "/û"]
            u = g.iplan * copy(û)          # c2r destroys input
            f = component === :mag ?
                dropdims(sqrt.(sum(abs2, u; dims = 4)); dims = 4) :
                u[:, :, :, parse(Int, string(component)[end:end])]
            sl = cutdim == 3 ? f[:, :, icut] :
                 cutdim == 2 ? f[:, icut, :] : f[icut, :, :]
            push!(slices, Float64.(sl))
            push!(ts, file[keys_[i] * "/t"])
        end

        # shared color scale across panels so the decay stays visible
        if component === :mag
            crange = (0.0, maximum(maximum, slices))
            cmap = :viridis
        else
            m = maximum(sl -> maximum(abs, sl), slices)
            crange = (-m, m)
            cmap = :RdBu
        end

        n = length(slices)
        fig = Figure(size = (300 * n + 100, 340))
        local hm
        for (j, (sl, t)) in enumerate(zip(slices, ts))
            ax = Axis(fig[1, j]; title = "t = $(_fmt(t))", xlabel = xlab,
                      ylabel = j == 1 ? ylab : "", aspect = DataAspect())
            hm = heatmap!(ax, coords(dims[1]), coords(dims[2]), sl;
                          colormap = cmap, colorrange = crange)
            j > 1 && hideydecorations!(ax; grid = false)
        end
        name = component === :mag ? "|u|" : string(component)
        Colorbar(fig[1, n + 1], hm; label = name)
        Label(fig[0, 1:n], "$name on the $(String(plane)) mid-plane";
              fontsize = 16)
        pth = joinpath(outdir, "slices.png")
        save(pth, fig)
        pth
    end
    return [p]
end

end # module
