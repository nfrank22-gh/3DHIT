# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HIT3D.jl — a modular pseudo-spectral solver for 3D homogeneous isotropic turbulence in Julia. Core solver, diagnostics, callbacks, and a Makie plotting extension are implemented; remaining TODOs are marked in code comments (stochastic forcing, IF/ETD schemes, low-storage RK).

## Commands

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'   # install deps
julia --project -e 'using Pkg; Pkg.test()'          # run tests
julia --project test/runtests.jl                    # run tests directly

# driver scripts use the scripts/ environment (adds CairoMakie; devs the package):
julia --project=scripts -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'  # once
julia --project=scripts scripts/run_decaying.jl     # example driver (decaying HIT)
```

There is no single-test filtering wired up; to iterate on one testset, run the relevant `@testset` block in a `julia --project` REPL after `using HIT3D, Test`.

## Architecture

One package, four submodules under `src/`, wired together in `src/HIT3D.jl`:

- **`Grids`** (`src/grid.jl`) — `Grid{T}`: geometry plus precomputed spectral operators (broadcast-shaped wavenumber arrays `kx/ky/kz`, `k²`, `1/k²` with k=0 zeroed, 2/3-rule dealias mask, batched rfft/irfft plans). Immutable-in-spirit and shareable; owns **no** scratch buffers. Derivatives/Laplacian/projection are small broadcast functions, not stored operators.
- **`RHS`** (`src/rhs/`) — `AbstractRHS`; each governing equation is one concrete struct implementing `rhs!(dû, û, r, t)`. `NavierStokes` uses rotational form (u × ω in physical space, projected divergence-free in spectral space; pressure eliminated). RHS structs own their scratch buffers. Forcing is a pluggable slot: `AbstractForcing` implementations add to `dû` via `apply_forcing!`. Deliberately equation-level modularity, **not** a generic sum-of-terms design (terms must share FFTs and scratch). `linear_operator(r)` optionally exposes the diagonal stiff part (−νk²) for future integrating-factor/ETD schemes.
- **`Integrators`** (`src/integrators/`) — hand-rolled fixed-step explicit RK (deliberately not OrdinaryDiffEq: AD-transparency, Metal control, light deps). Schemes own preallocated stage buffers, constructed from the state (`RK4(û)`). Driver: `evolve!(û, r, scheme, dt, nsteps; callbacks)`.
- **`Diagnostics`** (`src/diagnostics.jl`) — analysis functions on `(û, grid)` (energy, spectra, Re_λ…), used both offline and inside callbacks. All reductions double-count kx > 0 via `hermitian_weights`.
- **Plotting** (`ext/HIT3DMakieExt.jl`) — package extension, weakdep on `Makie` (frontend, backend-agnostic): loading CairoMakie/GLMakie activates `plot_summary(jld2)` / `plot_slices(jld2)`, whose stubs live in `src/plotting.jl`. The extension is entirely file-driven: `FieldWriter` files are self-describing (one-time `grid` group with `Nx…Lz` and `ν`). No Makie in the package deps or the test suite (rendering is verified manually by running the driver).
- **Run outputs** — drivers write everything into `results/<label(g)>_<label(r)>_<label(scheme)>/` (repo-rooted via `@__DIR__`), e.g. `results/N64_NavierStokes_nu0.001_NoForcing_RK4/`. `label(x)` (`src/labels.jl`) is the filesystem-safe slug layer; `Base.show` methods on the same structs are the pretty layer. `FieldWriter(overwrite = true)` is the default, so rerunning a driver replaces the folder contents instead of erroring on append.

### Core invariants (decided by design interview — don't casually reverse)

- **State layout**: a single complex 4D array `(Nx÷2+1, Ny, Nz, 3)` — rfft layout, velocity components last. Integrators treat it as one plain array; component access via `view(û, :,:,:, i)`. Shell sums must double-count kx > 0 modes (Hermitian symmetry).
- **GPU portability**: broadcast-only array programming over `AbstractArray`. The backend is chosen solely by constructing `Grid` (and state) with the target array type (`Array`/`CuArray`/`MtlArray`); no CUDA/Metal code or dependencies in this package. Known risk: `plan_rfft` maturity on Metal.
- **Precision**: everything parametric in `T`; Float32 is the default (Metal has no Float64).
- **In-place everywhere** (`rhs!`, `mul!` with plans, preallocated buffers). AD target is Enzyme / adjoint equations — not Zygote — so mutation is fine, but keep scratch in structs (not globals/closures) and the step loop a plain function of plain arrays.
- **Callbacks are pure observers**: user functions are `f(state) -> value` on a state named tuple `(; û, t, step, rhs, grid)`; scheduling and all accumulation/IO happen in the harness wrappers (`Callback`/`Diagnostic`/`FieldWriter`), wrapped in `@ignore_derivatives`. Anything that affects the dynamics (e.g. forcing) belongs in the RHS, never in a callback.

`scripts/run_decaying.jl` shows the intended end-to-end user API.
