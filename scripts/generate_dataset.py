# Dataset generation driver: runs one HIT simulation and saves full-field
# snapshots at a fixed simulation-time cadence after an initial warmup, plus
# the dense energy-budget series needed by linear_forcing_validation.py.
#
# Parameters come from a TOML config (configs/ — gitignored, untracked
# scratch; no checked-in example). The base output directory comes from the
# sibling file configs/output_path.txt (one line: an absolute or
# repo-relative path), also gitignored since it's machine-local.
#
# Config schema:
#
#   [grid]
#   N = 64
#   backend = "cpu"            # "cpu" | "cuda"
#
#   [physics]
#   nu = 4.491e-3
#   dt = 5e-3
#   total_time = 125.0         # nsteps = round(total_time / dt)
#
#   [forcing]
#   type = "LinearForcing"     # "NoForcing" | "BandForcing" | "LinearForcing"
#   A = 0.0667                 # LinearForcing
#   # eps = 0.1; kmin = 1.0; kmax = 2.0   # BandForcing instead
#
#   [initial_condition]
#   k0 = 2.0
#   u0 = 0.5
#   seed = 42
#
#   [dataset]
#   warmup_time = 10.0         # no snapshots before this simulation time
#   save_dt = 2.5              # snapshot cadence after warmup
#
# Run from the repo root:
#   uv run scripts/generate_dataset.py
#
# Always reads configs/dataset.toml — no CLI args. To generate a different
# dataset, edit that file (or swap in a different one under that name).

import os
import shutil
import sys
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

import hit3d as h3

here = os.path.dirname(__file__)
config_path = os.path.join(here, "..", "configs", "dataset.toml")
if not os.path.isfile(config_path):
    sys.exit(f"missing {config_path} — create it (see the schema documented "
             "at the top of this file)")
with open(config_path, "rb") as f:
    cfg = tomllib.load(f)

output_path_file = os.path.join(here, "..", "configs", "output_path.txt")
if not os.path.isfile(output_path_file):
    sys.exit(f"missing {output_path_file} — create it with one line: the "
             "base results directory for generated datasets")
with open(output_path_file) as f:
    base_dir = f.read().strip()

# --- [grid] -----------------------------------------------------------------

N = cfg["grid"]["N"]
backend = cfg["grid"].get("backend", "cpu")
try:
    device = jax.devices({"cuda": "gpu"}.get(backend, backend))[0]
except RuntimeError:
    sys.exit(f"no {backend!r} devices available to JAX "
             f"(have: {jax.devices()}) — on the CUDA machine install with "
             "`uv sync --extra cuda`")
jax.config.update("jax_default_device", device)

g = h3.make_grid(N)

# --- [initial_condition] ------------------------------------------------------

ic = cfg["initial_condition"]
k0, u0 = float(ic["k0"]), float(ic["u0"])
seed = ic.get("seed", 0)

# Rosales & Meneveau (2005) Eq. (9): E(k) = 16√(2/π)(u0²/k0⁵)k⁴exp(−2k²/k0²)
u_hat = h3.random_field(
    jax.random.key(seed), g,
    spectrum=lambda k: 16 * np.sqrt(2 / np.pi) * u0**2 / k0**5
    * k**4 * jnp.exp(-2 * k**2 / k0**2),
)

# --- [forcing] ----------------------------------------------------------------

fc = cfg["forcing"]
match fc["type"]:
    case "NoForcing":
        forcing = h3.NoForcing()
    case "LinearForcing":
        forcing = h3.LinearForcing(A=float(fc["A"]))
    case "BandForcing":
        forcing = h3.BandForcing(eps=float(fc["eps"]),
                                 kmin=float(fc["kmin"]),
                                 kmax=float(fc["kmax"]))
    case other:
        sys.exit(f"unknown [forcing].type = {other!r}")

# --- [physics] / [dataset] ----------------------------------------------------

nu = float(cfg["physics"]["nu"])
dt = float(cfg["physics"]["dt"])
nsteps = round(float(cfg["physics"]["total_time"]) / dt)

warmup_time = float(cfg["dataset"]["warmup_time"])
save_dt = float(cfg["dataset"]["save_dt"])

rhs = h3.make_ns_rhs(g)
params = h3.NSParams(nu=nu, forcing=forcing)

rundir = os.path.join(base_dir, h3.run_label(g, params))
snapfile = os.path.join(rundir, "dataset.h5")

budget = h3.Diagnostic(
    h3.energy_budget_cb, every=10, path=snapfile, name="energy_budget"
)
writer = h3.FieldWriter(snapfile, every_time=save_dt, warmup_time=warmup_time)

h3.evolve(rhs, u_hat, params, g, dt, nsteps, callbacks=(budget, writer),
          progress=True)

os.makedirs(rundir, exist_ok=True)
shutil.copy(config_path, os.path.join(rundir, os.path.basename(config_path)))
print("dataset written to", os.path.abspath(rundir))
