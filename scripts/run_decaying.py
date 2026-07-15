# Example driver: decaying HIT (float32, whatever backend jaxlib provides)
# with plots. Shows the intended end-to-end user API.
#
#   uv run scripts/run_decaying.py
#
# Plots need the extra:  uv sync --extra plots

import os

import jax

import hit3d as h3

N = 64
nu = 1e-3
dt = 1e-3
nsteps = 5_000

g = h3.make_grid(N)  # float32, 2π box by default
u_hat = h3.random_field(jax.random.key(0), g)

rhs = h3.make_ns_rhs(g)
params = h3.NSParams(nu=nu)  # NoForcing by default

# Everything for this run lands in results/<run label>/, named from the
# grid / params / scheme — reruns with equal parameters overwrite.
rundir = os.path.join(
    os.path.dirname(__file__), "..", "results", h3.run_label(g, params)
)
snapfile = os.path.join(rundir, "decaying.h5")

# `budget`'s series (path/name below) is persisted into the snapshot file
# automatically once evolve finishes, so postprocessing needs nothing but
# the one self-describing .h5:
budget = h3.Diagnostic(
    h3.energy_budget_cb, every=10, path=snapfile, name="energy_budget"
)
writer = h3.FieldWriter(snapfile, every=500)

u_hat = h3.evolve(
    rhs, u_hat, params, g, dt, nsteps, callbacks=(budget, writer),
    progress=True,
)

h3.plot_summary(snapfile)  # -> summary.png, spectra.png
h3.plot_energy_balance(snapfile)  # -> energy_balance.png
h3.plot_slices(snapfile)  # -> slices.png (|u|, xy mid-plane)
print("results written to", os.path.abspath(rundir))
