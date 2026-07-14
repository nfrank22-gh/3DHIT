"""
    HIT3D

Modular pseudo-spectral solver for 3D homogeneous isotropic turbulence.

Design notes (see interview log / README):
  - State: spectral velocity `û`, one complex 4D array of size (Nx÷2+1, Ny, Nz, 3)
    (rfft layout, components in the last dimension).
  - Formulation: rotational form nonlinear term (u × ω), divergence-free
    projection in spectral space; pressure eliminated.
  - GPU: broadcast-only array programming over AbstractArray. Backend is chosen
    by constructing the `Grid` (and state) with the desired array type
    (Array / CuArray / MtlArray). No CUDA/Metal code in this package.
  - Precision: everything parametric in `T`; Float32 by default.
  - Style: fully in-place (`rhs!`, preallocated buffers); AD target is
    Enzyme / adjoint equations. Callbacks are pure observers.
"""
module HIT3D

include("schema.jl")
include("grid.jl")
include("fields.jl")
include("rhs/rhs.jl")
include("integrators/integrators.jl")
include("diagnostics/diagnostics.jl")

using .Schema
using .Grids
using .RHS
using .Integrators
using .Diagnostics

include("labels.jl")
include("report.jl")
include("plotting.jl")
include("adjoint.jl")

# Re-export the main user-facing names
export Grid
export AbstractRHS, NavierStokes, AbstractForcing, NoForcing, BandForcing,
       LinearForcing
export injection
export AbstractScheme, RK4, step!, evolve!, Callback, Diagnostic, FieldWriter
export save_series
export energy, enstrophy, dissipation, energy_spectrum, energy_budget
export component_spectra, compensated_spectrum, dissipation_constant
export kolmogorov_scale, taylor_microscale, taylor_reynolds
export velocity_samples, velocity_moments
export longitudinal_autocorrelation, integral_lengthscale
export label, plot_summary, plot_slices, plot_energy_balance, plot_validation
export VJPWorkspace, vjp_step!
export load_run, read_series

end # module
