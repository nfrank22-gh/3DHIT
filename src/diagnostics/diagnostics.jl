"""
    Diagnostics

Analysis functions on `(û, grid)` — usable both offline (postprocessing) and
inside `Diagnostic` callbacks during a run. Plotting deliberately lives
outside the package (in the Makie extension), and this module is itself split
so the *calculation* layer stays independent of it:

- `spectral.jl`  — reductions over the spectral state (energy, spectra,
  isotropy/compensated spectra, ...), built on the shared `volume_integral`
  primitive;
- `physical.jl`  — reductions that need physical space (velocity PDF/moments,
  the longitudinal autocorrelation and its integral lengthscale).

All spectral functions account for the rfft layout via `hermitian_weights`:
kx > 0 modes represent two conjugate modes each and are double-counted in
every reduction. Quantities are volume means (⟨·⟩ over the box), so with the
unnormalized rfft convention each sum carries a 1/(NxNyNz)² factor.

The vorticity-based quantities assume `û` is divergence-free (the solver
invariant), so that |ω̂|² = |k × û|² = k²|û|².
"""
module Diagnostics

using ..Grids: Grid, hermitian_weights
using ..RHS: AbstractForcing, injection
using Base.Broadcast: broadcasted, instantiate
using LinearAlgebra: mul!
using Statistics: mean
using AbstractFFTs: irfft

export energy, enstrophy, dissipation, energy_spectrum, energy_budget,
       component_spectra, compensated_spectrum, dissipation_constant
export kolmogorov_scale, taylor_microscale, taylor_reynolds
export velocity_samples, velocity_moments
export longitudinal_autocorrelation, integral_lengthscale

include("spectral.jl")
include("physical.jl")

end # module
