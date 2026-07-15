"""HIT3D — pseudo-spectral solver for 3D homogeneous isotropic turbulence
in JAX.

Design invariants (see CLAUDE.md):
  - State: one complex array ``(3, Nx, Ny, Nz//2+1)`` (rfft layout,
    components first); Hermitian double-counting along kz in reductions.
  - Rotational-form nonlinear term, divergence-free projection, pressure
    eliminated.
  - Pure functions everywhere; `rollout` is the only AD path and carries no
    callbacks; `evolve` is the observation driver and is never
    differentiated.
  - Backend (CPU/CUDA) is whatever jaxlib is installed; float32 default.
"""

from .callbacks import Callback, Diagnostic, FieldWriter, State, save_series
from .diagnostics import (
    compensated_spectrum,
    component_spectra,
    dissipation,
    dissipation_constant,
    energy,
    energy_budget,
    energy_budget_cb,
    energy_spectrum,
    enstrophy,
    integral_lengthscale,
    kolmogorov_scale,
    longitudinal_autocorrelation,
    taylor_microscale,
    taylor_reynolds,
    taylor_reynolds_field,
    velocity_moments,
    velocity_samples,
)
from .fields import default_spectrum, random_field, spectral_state, taylor_green
from .forcing import BandForcing, LinearForcing, NoForcing
from .grid import (
    Grid,
    ddx,
    ddy,
    ddz,
    dealias,
    laplacian,
    make_grid,
    project,
    spectral_shape,
    to_physical,
    to_spectral,
)
from .integrators import evolve, rk4_step, rollout
from .labels import label, run_label
from .navierstokes import NSParams, linear_operator, make_ns_rhs
from .plotting import (
    plot_energy_balance,
    plot_slices,
    plot_summary,
    plot_validation,
)
from .report import load_run, read_series

__all__ = [
    "Grid", "make_grid", "spectral_shape", "ddx", "ddy", "ddz", "laplacian",
    "dealias", "project", "to_physical", "to_spectral",
    "spectral_state", "random_field", "taylor_green", "default_spectrum",
    "NoForcing", "LinearForcing", "BandForcing",
    "NSParams", "make_ns_rhs", "linear_operator",
    "rk4_step", "rollout", "evolve",
    "State", "Callback", "Diagnostic", "FieldWriter", "save_series",
    "energy", "enstrophy", "dissipation", "energy_spectrum",
    "component_spectra", "compensated_spectrum", "dissipation_constant",
    "kolmogorov_scale", "taylor_microscale", "taylor_reynolds",
    "taylor_reynolds_field", "energy_budget", "energy_budget_cb",
    "velocity_samples", "velocity_moments", "longitudinal_autocorrelation",
    "integral_lengthscale",
    "label", "run_label",
    "load_run", "read_series",
    "plot_summary", "plot_slices", "plot_energy_balance", "plot_validation",
]
