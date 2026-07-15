# HIT3D

Pseudo-spectral solver for 3D homogeneous isotropic turbulence in JAX.

Rotational-form incompressible Navier–Stokes on a triply periodic box:
2/3-rule dealiasing, divergence-free spectral projection (pressure
eliminated), RK4 time stepping via `jax.lax.scan`, pluggable forcing
(decaying / band / Lundgren linear), HDF5 run outputs, and matplotlib
postprocessing. Everything is pure functions over one complex state array
`(3, Nx, Ny, Nz//2+1)`, so `jax.vjp` / `jax.grad` through a step or a
rollout works out of the box.

```bash
uv sync                      # core (CPU)
uv sync --extra plots        # + matplotlib postprocessing
uv sync --extra cuda         # on a CUDA machine
uv run pytest                # tests
uv run scripts/run_decaying.py
```

See `CLAUDE.md` for architecture notes and `scripts/` for end-to-end
drivers.
