import os

import h5py
import numpy as np

from conftest import beltrami
from hit3d import (
    Callback,
    Diagnostic,
    FieldWriter,
    NSParams,
    energy,
    evolve,
    make_ns_rhs,
)

N = 32
NU = 0.05


def run_beltrami(grid, callbacks, dt=1e-2, nsteps=100, progress=False):
    rhs = make_ns_rhs(grid)
    u0 = beltrami(grid)
    return evolve(
        rhs, u0, NSParams(nu=NU), grid, dt, nsteps,
        callbacks=callbacks, progress=progress,
    )


def test_diagnostic_and_callback_scheduling(grid64, tmp_path):
    dt, nsteps = 1e-2, 100
    E = Diagnostic(lambda s: energy(s.u_hat, s.grid), every=10)
    ncalls = []
    probe = Callback(lambda s: ncalls.append(s.step), every_time=0.25)
    path = str(tmp_path / "snap.h5")
    writer = FieldWriter(path, every=50)

    u = run_beltrami(grid64, (E, probe, writer), dt, nsteps)

    # Diagnostic: fires at step 0 and every 10th step; energy follows the
    # analytic Beltrami decay e^(−2νt)
    assert len(E.times) == nsteps // 10 + 1
    assert np.isclose(E.times[1], 10 * dt)
    expected = E.values[0] * np.exp(-2 * NU * np.asarray(E.times))
    assert np.allclose(np.asarray(E.values), expected, rtol=1e-8)

    # Callback with a time-based schedule: initial + crossings of 0.25
    assert 4 <= len(ncalls) <= 6

    # FieldWriter: snapshots at steps 0, 50, 100; final one matches the
    # evolved state; self-describing grid/ν metadata group
    with h5py.File(path, "r") as f:
        steps = sorted(k for k in f if k.startswith("step_"))
        assert steps == ["step_00000000", "step_00000050", "step_00000100"]
        assert np.allclose(f["step_00000100/u_hat"][()], np.asarray(u))
        assert np.isclose(f["step_00000100/t"][()], dt * nsteps)
        assert f["grid/Nx"][()] == N
        assert np.isclose(f["grid/Lx"][()], 2 * np.pi)
        assert f["grid/nu"][()] == NU


def test_fieldwriter_overwrite_semantics(grid64, tmp_path):
    path = str(tmp_path / "snap.h5")
    run_beltrami(grid64, (FieldWriter(path, every=50),), nsteps=10)
    assert os.path.exists(path)
    FieldWriter(path, every=50, overwrite=False)
    assert os.path.exists(path)
    FieldWriter(path, every=50)  # default overwrite=True deletes
    assert not os.path.exists(path)


def test_fieldwriter_time_scheduling(grid64, tmp_path):
    # dataset-generation schedule: no snapshots before warmup_time, then a
    # fixed simulation-time cadence regardless of dt
    path = str(tmp_path / "snap.h5")
    writer = FieldWriter(path, every_time=0.5, warmup_time=1.0)
    run_beltrami(grid64, (writer,), dt=0.1, nsteps=30)
    with h5py.File(path, "r") as f:
        steps = sorted(k for k in f if k.startswith("step_"))
    # t = 1.0, 1.5, 2.0, 2.5, 3.0 → steps 10, 15, 20, 25, 30
    assert steps == [f"step_{s:08d}" for s in (10, 15, 20, 25, 30)]


def test_progress_printing(grid64, capsys):
    # explicit interval: fires at 5 and 10 — always including the last step
    run_beltrami(grid64, (), dt=1e-3, nsteps=10, progress=5)
    out = capsys.readouterr().out
    assert out.count("\n") == 2
    assert "step 10/10 (100%)" in out
    assert "t = " in out and "eta " in out

    # default is silent
    run_beltrami(grid64, (), dt=1e-3, nsteps=5)
    assert capsys.readouterr().out == ""

    # progress=True picks its own interval, ends at 100%
    run_beltrami(grid64, (), dt=1e-3, nsteps=5, progress=True)
    out = capsys.readouterr().out
    assert out and "(100%)" in out


def test_diagnostic_path_without_name_errors():
    import pytest

    with pytest.raises(ValueError):
        Diagnostic(lambda s: 0.0, path="somewhere.h5")
