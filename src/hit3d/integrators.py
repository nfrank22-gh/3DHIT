"""Time integration.

Two entry points with a hard wall between them:

- `rollout` — a pure, jitted `lax.scan` over `rk4_step`. **The only AD
  path.** No callbacks, no IO, no host round-trips: differentiate it
  directly (``jax.vjp(lambda u: rollout(rhs, u, params, dt, n), u_hat)``)
  or wrap a scalar loss around it and use `jax.grad`. For long rollouts,
  wrap the step in `jax.checkpoint` before scanning (see TODO below).
- `evolve` — the observation/dataset-generation driver: a host-side loop
  over jitted `rollout` segments, firing callbacks at segment boundaries.
  Never differentiated.

Hand-rolled fixed-step RK4 (deliberately not diffrax: transparency, light
deps). TODO later: low-storage schemes; `jax.checkpoint`/remat wiring for
thousand-step rollouts; integrating-factor/ETD using `linear_operator`.
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp

from .callbacks import State

__all__ = ["rk4_step", "rollout", "evolve"]


def rk4_step(rhs, u_hat, params, dt, t):
    """One classic 4th-order Runge–Kutta step of size `dt` from time `t`."""
    k1 = rhs(u_hat, params, t)
    k2 = rhs(u_hat + 0.5 * dt * k1, params, t + 0.5 * dt)
    k3 = rhs(u_hat + 0.5 * dt * k2, params, t + 0.5 * dt)
    k4 = rhs(u_hat + dt * k3, params, t + dt)
    return u_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@partial(jax.jit, static_argnames=("rhs", "nsteps"))
def rollout(rhs, u_hat, params, dt, nsteps, t0=0.0):
    """Advance `u_hat` by `nsteps` RK4 steps of size `dt` from time `t0`.

    Pure and jitted; `rhs` and `nsteps` are static (a new `rhs` closure or
    step count triggers a recompile — build the rhs once per grid).
    """

    def body(u, i):
        return rk4_step(rhs, u, params, dt, t0 + i * dt), None

    u_hat, _ = jax.lax.scan(body, u_hat, jnp.arange(nsteps))
    return u_hat


def evolve(rhs, u_hat, params, grid, dt, nsteps, *, t0=0.0, callbacks=(),
           progress=False):
    """Observation driver: advance `nsteps` steps, firing callbacks that
    are due. Callbacks receive a read-only `State` named tuple
    ``(u_hat, t, step, params, grid)`` and must be pure — scheduling and
    all accumulation/IO happen in the wrapper classes (`Callback` /
    `Diagnostic` / `FieldWriter`).

    Internally the run is split into jitted `rollout` segments between
    callback firings, so the device never syncs except when a callback is
    due. Not differentiable by design — use `rollout` for AD.

    ``progress=True`` prints a progress line (step, simulation time, wall
    time, ETA) roughly 20 times over the run; an integer prints every that
    many steps.
    """
    if isinstance(progress, bool):
        print_every = max(1, nsteps // 20) if progress else 0
    else:
        print_every = max(int(progress), 1)

    def due_any(n):
        t = t0 + n * dt
        return any(cb.is_due(n, t, dt) for cb in callbacks)

    stops = sorted(
        {n for n in range(1, nsteps + 1) if due_any(n)}
        | ({n for n in range(print_every, nsteps + 1, print_every)}
           if print_every else set())
        | {nsteps}
    ) if nsteps > 0 else []

    def fire(n):
        if not callbacks:
            return
        t = t0 + n * dt
        state = State(u_hat=u_hat, t=t, step=n, params=params, grid=grid)
        for cb in callbacks:
            if cb.is_due(n, t, dt):
                cb.fire(state)

    wall0 = time.time()
    fire(0)
    prev = 0
    for n in stops:
        u_hat = rollout(rhs, u_hat, params, dt, n - prev, t0 + prev * dt)
        fire(n)
        if print_every and (n % print_every == 0 or n == nsteps):
            _print_progress(n, nsteps, t0 + n * dt, wall0)
        prev = n

    for cb in callbacks:
        cb.finalize()
    return u_hat


def _print_progress(n, nsteps, t, wall0):
    elapsed = time.time() - wall0
    eta = elapsed / n * (nsteps - n)
    width = len(str(nsteps))
    print(
        f"step {n:{width}d}/{nsteps} ({round(100 * n / nsteps):3d}%)  "
        f"t = {t:<9.4g}  elapsed {_fmt_secs(elapsed)}  eta {_fmt_secs(eta)}",
        flush=True,
    )


def _fmt_secs(s):
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{round(s % 60):02d}s"
