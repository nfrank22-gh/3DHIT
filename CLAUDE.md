# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HIT3D — a pseudo-spectral solver for 3D homogeneous isotropic turbulence in Python/JAX. Ported from a Julia implementation (git history before the port, ending at `6753971`) specifically to make autodiff trivial: the entire hand-built Enzyme adjoint machinery of the Julia version collapses into `jax.vjp`/`jax.grad` over pure functions. Remaining TODOs are marked in code comments (stochastic forcing, IF/ETD schemes, low-storage RK, remat wiring for long rollouts).

## Commands

```bash
uv sync --extra plots        # install (CPU + matplotlib); plain `uv sync` for solver core only
uv sync --extra cuda         # on the CUDA machine (dev happens on Mac = CPU; no Metal — JAX has no usable Metal backend)
uv run pytest                # run tests
uv run pytest tests/test_grid.py -k dealias   # one test
uv run ruff check .          # lint

uv run scripts/run_decaying.py            # example driver (decaying HIT, ~3 min CPU)
uv run scripts/run_linear_forcing.py      # forced HIT + Rosales–Meneveau validation printout
uv run scripts/generate_dataset.py        # TOML-config dataset generation (needs configs/dataset.toml + configs/output_path.txt, both gitignored)
uv run scripts/linear_forcing_validation.py  # postprocessor over generated datasets
uv run scripts/check_gpu.py               # device report + jitted rollout + grad smoke
```

## Architecture

One package, `src/hit3d/`, all pure functions over plain arrays (no classes with mutable state anywhere):

- **`grid.py`** — `Grid`: frozen pytree dataclass (arrays are leaves, sizes/lengths static metadata) holding precomputed wavenumbers (broadcast-shaped `kx/ky/kz`), `k2`, `inv_k2` (k=0 zeroed), 2/3-rule dealias mask (False at k=0 → zero-mean invariant), and Hermitian `weights`. Built once by `make_grid` (NumPy, never traced); never differentiated. Spectral ops (`ddx…`, `laplacian`, `project`, `dealias`, `to_physical`/`to_spectral`) are small functions — no FFT plans, XLA plans internally.
- **`navierstokes.py`** — `make_ns_rhs(grid)` returns `rhs(u_hat, params, t)`; rotational form (u × ω in physical space, forcing added *before* dealias+projection so non-solenoidal parts are absorbed into pressure). `NSParams` (frozen pytree) carries ν + the forcing — the differentiable parameters. Equation-level modularity, **not** sum-of-terms.
- **`forcing.py`** — forcings are frozen pytree dataclasses with `term(u_hat, grid, t)`, `injection(u_hat, grid)` (exact per-type injected power for the energy budget), and a `label` property. Their scalar coefficients are pytree leaves → reachable by `jax.grad` through `NSParams`.
- **`integrators.py`** — the two-entry-point split that is the core design decision:
  - `rollout(rhs, u_hat, params, dt, nsteps, t0)` — jitted `lax.scan` over `rk4_step`; **the only AD path**; structurally incapable of running callbacks.
  - `evolve(rhs, u_hat, params, grid, dt, nsteps, callbacks=, progress=)` — host loop over jitted `rollout` segments split at callback firings; observation/dataset generation only; never differentiated.
  - `rhs` and `nsteps` are static jit args: call `make_ns_rhs` once per grid or every closure recompiles.
- **`callbacks.py`** — `Callback`/`Diagnostic`/`FieldWriter` + `State` named tuple `(u_hat, t, step, params, grid)`. User functions are pure observers `f(state) -> value`; scheduling (`every` / `every_time` + `warmup_time`) and all accumulation/IO live in the wrappers. Anything affecting dynamics belongs in the RHS params, never here.
- **`diagnostics.py`** — analysis on `(u_hat, grid)` (energy, spectra, Re_λ, budget, autocorrelation…). All spectral reductions double-count kz > 0 via `grid.weights`; shell spectra are one `bincount` over shell indices. `energy_budget_cb` is the callback-form wrapper.
- **`schema.py` / `report.py` / `plotting.py`** — one definition of the HDF5 layout (`grid/…`, `step_00000000/…`, `series/<name>/…`, state stored as `u_hat`); `report.load_run`/`read_series` are the matplotlib-free readers covered by tests; `plotting.plot_summary/plot_slices/plot_energy_balance/plot_validation` are file-driven postprocessors that lazy-import matplotlib (optional `[plots]` extra).
- **`labels.py`** — `label(x)` / `run_label(grid, params)`: filesystem-safe slugs (`%g` numbers), e.g. `results/N64_NavierStokes_nu0.001_NoForcing_RK4`. Drivers write everything into `results/<run_label>/`; `FieldWriter(overwrite=True)` is the default so reruns replace the folder contents.

### Core invariants (decided by design interview — don't casually reverse)

- **State layout**: one complex array `(3, Nx, Ny, Nz//2+1)` — components FIRST, rfft-halved axis LAST (`rfftn` over axes 1–3; C-order native). Shell sums must double-count kz > 0 modes (Hermitian symmetry). This is transposed from the Julia layout — don't copy Julia-era conventions from old commits.
- **AD wall**: `rollout` is the differentiable unit and carries no callbacks; `evolve` has callbacks and is never differentiated. Keep the wall — no flags mixing the two.
- **Params vs. structure**: anything you might want a gradient of (ν, forcing coefficients) flows through the `NSParams` pytree; anything structural (grid, forcing *type*, equation form) is fixed at closure/construction time.
- **GPU portability**: pure `jnp` array programming; backend is solely which jaxlib is installed (CPU on Mac, `[cuda]` extra on the Linux box). No device code in the package.
- **Precision**: float32 default; float64 available on CPU via `jax.config.update("jax_enable_x64", True)` (the test suite does this in `conftest.py` — package default stays float32).
- **Callbacks are pure observers** on `State`; IO/accumulation only in the harness wrappers.
- **Tests don't re-prove JAX's autodiff**: `test_grad.py` is structure-smoke only (finite, nonzero grads through `rollout`). Physics is validated by property tests (Beltrami exact decay, Parseval, ν=0 energy conservation, budget residual), not oracle files.

`scripts/run_decaying.py` shows the intended end-to-end user API.
