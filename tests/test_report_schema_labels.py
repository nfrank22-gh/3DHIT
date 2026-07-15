import h5py
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from hit3d import (
    BandForcing,
    FieldWriter,
    LinearForcing,
    NoForcing,
    NSParams,
    evolve,
    label,
    load_run,
    make_grid,
    make_ns_rhs,
    random_field,
    read_series,
    run_label,
)
from hit3d.schema import (
    grid_key,
    is_step_key,
    series_field,
    series_key,
    step_key,
    step_num,
)

N = 32
NU = 0.05


def test_schema():
    assert step_key(50) == "step_00000050"
    assert step_num("step_00000050") == 50
    assert is_step_key("step_00000050")
    assert not is_step_key("grid")
    assert grid_key("Nx") == "grid/Nx"
    assert series_key("energy_budget") == "series/energy_budget"
    assert series_field("energy_budget", "E") == "series/energy_budget/E"


def test_load_run_roundtrip(grid64, tmp_path):
    g = grid64
    u_hat = random_field(jr.key(0), g)
    rhs = make_ns_rhs(g)
    path = str(tmp_path / "snap.h5")
    evolve(rhs, u_hat, NSParams(nu=NU), g, 1e-3, 10,
           callbacks=(FieldWriter(path, every=5),))

    run = load_run(path)
    assert run.grid.Nx == run.grid.Ny == run.grid.Nz == N
    assert run.nu == NU
    assert run.dtype == np.float64
    assert run.steps == [0, 5, 10]

    # no dense series recorded -> read_series is None
    assert read_series(path, "energy_budget") is None


def test_load_run_legacy_file_fallback(grid64, tmp_path):
    # file with no "grid" group: load_run falls back to a 2π cube with
    # unknown ν, with a warning, inferring sizes from the rfft layout
    u_hat = random_field(jr.key(1), grid64)
    legacy = str(tmp_path / "legacy.h5")
    with h5py.File(legacy, "w") as f:
        f["step_00000000/t"] = 0.0
        f["step_00000000/u_hat"] = np.asarray(u_hat)
    with pytest.warns(UserWarning, match="no grid metadata"):
        run = load_run(legacy)
    assert run.grid.Nx == N
    assert run.nu is None


def test_load_run_empty_file_errors(tmp_path):
    empty = str(tmp_path / "empty.h5")
    h5py.File(empty, "w").close()
    with pytest.raises(ValueError):
        load_run(empty)


def test_labels():
    g = make_grid(16, dtype=jnp.float64)
    assert label(g) == "N16"
    assert label(make_grid(16, 8, 4)) == "N16x8x4"

    p = NSParams(nu=1e-3)
    assert label(p) == "NavierStokes_nu0.001_NoForcing"

    f = BandForcing(eps=0.1, kmin=1.0, kmax=2.5)
    assert label(f) == "BandForcing_eps0.1_k1-2.5"
    assert label(NSParams(nu=1e-3, forcing=f)) == (
        "NavierStokes_nu0.001_BandForcing_eps0.1_k1-2.5"
    )

    lf = LinearForcing(A=0.1333)
    assert label(lf) == "LinearForcing_A0.1333"
    assert repr(lf) == "LinearForcing(A=0.1333)"

    assert run_label(g, p) == "N16_NavierStokes_nu0.001_NoForcing_RK4"

    # %g formatting is precision-independent (no float32 noise in paths)
    assert label(NSParams(nu=jnp.float32(1e-3))) == (
        "NavierStokes_nu0.001_NoForcing"
    )

    # repr: pretty, and never dumps the wavenumber arrays
    assert repr(g) == "Grid[float64](16x16x16, L=(6.28319, 6.28319, 6.28319))"
    assert repr(NoForcing()) == "NoForcing()"
    assert repr(BandForcing(eps=0.1, kmin=1.0, kmax=2.5)) == (
        "BandForcing(eps=0.1, k=[1, 2.5])"
    )
    rp = repr(NSParams(nu=1e-3, forcing=f))
    assert rp == "NavierStokes(nu=0.001, forcing=BandForcing(eps=0.1, k=[1, 2.5]))"
    assert len(rp) < 200
