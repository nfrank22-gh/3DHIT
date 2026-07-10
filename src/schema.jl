"""
    Schema

JLD2 key-naming for the files `FieldWriter`/`save_series` write. Used by
both the writer (`Integrators.callbacks`) and the reader (`report.jl`), so
the on-disk layout has exactly one definition instead of matching string
literals typed independently on each side.

Layout:

    grid/Nx, grid/Ny, grid/Nz, grid/Lx, grid/Ly, grid/Lz, grid/nu   # optional
    step_00000000/t, step_00000000/<field>, ...                    # one per snapshot
    series/<name>/t, series/<name>/<field>, ...                    # one per Diagnostic
"""
module Schema

export stepkey, stepnum, is_stepkey, gridkey, serieskey, seriesfield

const STEP_PREFIX = "step_"
const STEP_DIGITS = 8

"""Group name for snapshot step `n`, e.g. `stepkey(50) == "step_00000050"`."""
stepkey(n::Integer) = STEP_PREFIX * lpad(string(n), STEP_DIGITS, '0')

"""Whether `key` is a snapshot group name."""
is_stepkey(key::AbstractString) = startswith(key, STEP_PREFIX)

"""Step number encoded in a snapshot group name."""
stepnum(key::AbstractString) = parse(Int, split(key, "_")[end])

"""Key for grid-metadata field `name` (`:Nx`, `:Lx`, `:nu`, ...)."""
gridkey(name::Symbol) = "grid/" * String(name)

"""Group name for the dense series recorded under `name`."""
serieskey(name) = "series/" * String(name)

"""Key for field `fld` of the dense series `name`."""
seriesfield(name, fld) = serieskey(name) * "/" * String(fld)

end # module
