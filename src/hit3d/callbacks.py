"""Callback system for `evolve` (dataset generation / observation only).

Design rules (AD compatibility):
  * Callbacks exist only on the `evolve` path. The differentiable path is
    `rollout`, which has no callback machinery at all — nothing here can
    interfere with a gradient computation.
  * User callback functions are PURE: ``f(state) -> value``, where `state`
    is the named tuple ``(u_hat, t, step, params, grid)``. They never
    mutate. All accumulation/IO is owned by the wrapper classes below.
  * Anything that is part of the dynamics (forcing!) belongs in the RHS
    params, never in a callback.

Scheduling is the harness's job: wrappers carry `every` (steps) or
`every_time` (simulation-time interval) and `evolve` decides when to fire.
All wrappers fire at step 0 (the initial state) as well.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import h5py
import jax
import numpy as np

from .schema import grid_key, series_field, series_key, step_key

__all__ = ["State", "Callback", "Diagnostic", "FieldWriter", "save_series"]


class State(NamedTuple):
    """Read-only snapshot handed to callback functions."""

    u_hat: object
    t: float
    step: int
    params: object
    grid: object


class Callback:
    """Generic wrapper: call the pure function `f(state)` on the given
    schedule and discard the result (side-effect-free hook point). If
    `every_time` is given it takes precedence over `every` and fires
    whenever the simulation time crosses a multiple of `every_time`."""

    def __init__(self, f, *, every=1, every_time=None):
        self.f = f
        self.every = every
        self.every_time = every_time

    def is_due(self, step, t, dt):
        if self.every_time is None:
            return step % self.every == 0
        return step == 0 or (
            math.floor(t / self.every_time)
            > math.floor((t - dt) / self.every_time)
        )

    def fire(self, state):
        self.f(state)

    def finalize(self):
        pass


class Diagnostic:
    """Scalar/small-record timeseries recorder: appends `f(state)` and the
    current time to internally owned lists. Cheap; meant to run often
    (energy, dissipation, ...). After the run, `d.times` / `d.values` are
    the plot data. `f` may return a scalar or a dict of scalars (e.g.
    `energy_budget_cb`); values are pulled to the host on every fire.

    If `path` and `name` are given, `evolve` persists the recorded series
    to `path` under ``series/<name>`` (via `save_series`) automatically
    once the run finishes. This suits the common single-`evolve`-call
    driver; for a run split across multiple `evolve` calls, leave them
    unset and call `save_series` by hand after the last one.
    """

    def __init__(self, f, *, every=1, path=None, name=None):
        if path is not None and name is None:
            raise ValueError(
                "Diagnostic(..., path=...) also needs `name` "
                "(the series/<name> group to persist it under)"
            )
        self.f = f
        self.every = every
        self.path = path
        self.name = name
        self.times = []
        self.values = []

    def is_due(self, step, t, dt):
        return step % self.every == 0

    def fire(self, state):
        self.times.append(float(state.t))
        self.values.append(jax.device_get(self.f(state)))

    def finalize(self):
        if self.path is not None:
            save_series(self.path, self.name, self)


class FieldWriter:
    """Full-field snapshot writer: copies the requested state fields to the
    host and appends them (plus the time) to an HDF5 file under
    ``step_<n>/<field>``. Expensive; run rarely.

    The file is self-describing: on the first write a `grid` group is
    stored (Nx, Ny, Nz, Lx, Ly, Lz, and the params' `nu` when present), so
    postprocessing tools like `plot_summary` can reconstruct the `Grid`
    from the file alone.

    With ``overwrite=True`` (default) an existing file at `path` is deleted
    at construction time — rerunning a driver replaces its snapshots
    instead of erroring on duplicate group names mid-run. Pass
    ``overwrite=False`` to append across multiple `evolve` calls (steps
    must not repeat).

    By default snapshots are scheduled by step count (`every`). Pass
    `every_time` to schedule by simulation time instead: the first snapshot
    lands at ``t = warmup_time`` (default 0) and recurs every `every_time`
    thereafter — no snapshots are taken before `warmup_time`. This suits
    dataset-generation drivers that want to discard an initial transient
    and then sample at a fixed physical-time cadence regardless of `dt`.
    """

    def __init__(self, path, *, every=1, fields=("u_hat",), overwrite=True,
                 every_time=None, warmup_time=None):
        if every_time is not None and warmup_time is None:
            warmup_time = 0.0
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if overwrite and os.path.exists(path):
            os.remove(path)
        self.path = path
        self.every = every
        self.fields = tuple(fields)
        self.every_time = every_time
        self.warmup_time = warmup_time

    def is_due(self, step, t, dt):
        if self.every_time is None:
            return step % self.every == 0
        if t < self.warmup_time:
            return False
        # Snap cadence indices to the nearest integer within tol, guarding
        # against float32 noise: `t` at step n is t0 + n*dt (accumulated by
        # `evolve`), but the previous instant is reconstructed here as
        # t - dt — the two don't always agree to the last bit, which
        # without snapping can make a boundary crossing detected twice.
        tol = (dt / self.every_time) / 4

        def index(x):
            r = round(x)
            return r if abs(x - r) < tol else x

        n_now = math.floor(index((t - self.warmup_time) / self.every_time))
        n_prev = math.floor(index((t - dt - self.warmup_time) / self.every_time))
        return n_now > n_prev

    def fire(self, state):
        with h5py.File(self.path, "a") as f:
            if "grid" not in f:
                g = state.grid
                for name, val in (("Nx", g.Nx), ("Ny", g.Ny), ("Nz", g.Nz),
                                  ("Lx", g.Lx), ("Ly", g.Ly), ("Lz", g.Lz)):
                    f[grid_key(name)] = val
                nu = getattr(state.params, "nu", None)
                if nu is not None:
                    f[grid_key("nu")] = float(nu)
            grp = step_key(state.step)
            f[f"{grp}/t"] = float(state.t)
            for fld in self.fields:
                f[f"{grp}/{fld}"] = np.asarray(
                    jax.device_get(getattr(state, fld))
                )

    def finalize(self):
        pass


def save_series(path, name, diagnostic):
    """Persist a `Diagnostic`'s recorded time series into the HDF5 file at
    `path` (appending — typically the run's `FieldWriter` snapshot file, so
    the run directory keeps a single self-describing artifact).

    Layout is columnar under ``series/<name>/``: the times as `t`, and the
    values either as one `values` vector (scalar diagnostics) or as one
    vector per key for dict values (e.g. E, eps, P for the energy budget).

    Errors if ``series/<name>`` already exists in the file; rerunning a
    driver with the default ``FieldWriter(overwrite=True)`` starts from a
    fresh file, so this only bites when saving the same name twice into
    one file.
    """
    with h5py.File(path, "a") as f:
        if series_key(name) in f:
            raise ValueError(
                f"{series_key(name)} already exists in {path} (saving the "
                "same series twice? reruns should start from a fresh file)"
            )
        f[series_field(name, "t")] = np.asarray(diagnostic.times)
        values = diagnostic.values
        if values and isinstance(values[0], dict):
            for fld in values[0]:
                f[series_field(name, fld)] = np.asarray(
                    [v[fld] for v in values]
                )
        else:
            f[series_field(name, "values")] = np.asarray(values)
    return path
