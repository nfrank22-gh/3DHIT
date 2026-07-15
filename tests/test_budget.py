import h5py
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from hit3d import (
    BandForcing,
    Diagnostic,
    FieldWriter,
    LinearForcing,
    NoForcing,
    NSParams,
    State,
    dissipation,
    energy,
    energy_budget,
    energy_budget_cb,
    evolve,
    make_ns_rhs,
    random_field,
    read_series,
    save_series,
)

N = 32
NU = 0.05


def measured_power(u_hat, term, grid):
    """⟨u·f⟩ measured from a forcing evaluation (Hermitian-weighted
    spectral inner product; exact here since f̂ ∝ û)."""
    ntot2 = (grid.Nx * grid.Ny * grid.Nz) ** 2
    return float(
        jnp.sum(grid.weights * jnp.sum(jnp.real(jnp.conj(u_hat) * term), axis=0))
        / ntot2
    )


def test_injection_exact_per_forcing(grid64):
    g = grid64
    u_hat = random_field(jr.key(0), g)

    assert float(NoForcing().injection(u_hat, g)) == 0.0

    f = BandForcing(eps=0.3, kmin=1.0, kmax=3.0)
    assert float(f.injection(u_hat, g)) == 0.3
    assert float(f.injection(jnp.zeros_like(u_hat), g)) == 0.0  # empty band

    # apply and injection agree
    assert np.isclose(
        measured_power(u_hat, f.term(u_hat, g, 0.0), g),
        float(f.injection(u_hat, g)),
    )

    lf = LinearForcing(A=0.1333)
    assert jnp.isclose(lf.injection(u_hat, g), 2 * 0.1333 * energy(u_hat, g))
    assert np.isclose(
        measured_power(u_hat, lf.term(u_hat, g, 0.0), g),
        float(lf.injection(u_hat, g)),
    )


def test_energy_budget_terms(grid64):
    g = grid64
    u_hat = random_field(jr.key(1), g)
    f = BandForcing(eps=0.3, kmin=1.0, kmax=3.0)

    b = energy_budget(u_hat, g, NU, f)
    assert jnp.isclose(b["E"], energy(u_hat, g))
    assert jnp.isclose(b["eps"], dissipation(u_hat, g, NU))
    assert float(b["P"]) == 0.3

    # state-tuple forwarding
    state = State(u_hat=u_hat, t=0.0, step=0,
                  params=NSParams(nu=NU), grid=g)
    bs = energy_budget_cb(state)
    assert jnp.isclose(bs["E"], b["E"])
    assert jnp.isclose(bs["eps"], b["eps"])
    assert float(bs["P"]) == 0.0


def test_budget_residual_small_over_decay(grid64):
    # short viscous decay: the finite-difference residual dE/dt + ε − P is
    # small at the recorded samples
    g = grid64
    u_hat = random_field(jr.key(2), g)
    rhs = make_ns_rhs(g)
    d = Diagnostic(energy_budget_cb, every=1)
    dt, nsteps = 1e-3, 50
    evolve(rhs, u_hat, NSParams(nu=NU), g, dt, nsteps, callbacks=(d,))
    E = np.array([v["E"] for v in d.values])
    eps = np.array([v["eps"] for v in d.values])
    dEdt = (E[2:] - E[:-2]) / (2 * dt)
    resid = dEdt + eps[1:-1]
    assert np.max(np.abs(resid)) < 1e-4 * eps.max()


def test_save_series_roundtrip(grid64, tmp_path):
    g = grid64
    u_hat = random_field(jr.key(3), g)
    rhs = make_ns_rhs(g)
    d = Diagnostic(energy_budget_cb, every=1)
    dt, nsteps = 1e-3, 5
    evolve(rhs, u_hat, NSParams(nu=NU), g, dt, nsteps, callbacks=(d,))

    path = str(tmp_path / "series.h5")
    save_series(path, "energy_budget", d)
    with h5py.File(path, "r") as f:
        assert np.allclose(f["series/energy_budget/t"][()], d.times)
        assert np.allclose(
            f["series/energy_budget/E"][()], [v["E"] for v in d.values]
        )
        assert np.allclose(f["series/energy_budget/P"][()], 0.0)

    import pytest

    with pytest.raises(ValueError):
        save_series(path, "energy_budget", d)  # duplicate name

    # scalar diagnostics persist as a single values vector
    ds = Diagnostic(lambda s: 0.0)
    ds.times.append(0.0)
    ds.values.append(1.5)
    save_series(path, "energy", ds)
    with h5py.File(path, "r") as f:
        assert np.allclose(f["series/energy/values"][()], [1.5])


def test_diagnostic_autopersists_via_evolve(grid64, tmp_path):
    g = grid64
    u_hat = random_field(jr.key(4), g)
    rhs = make_ns_rhs(g)
    path = str(tmp_path / "auto.h5")
    writer = FieldWriter(path, every=1000)  # just for the grid group
    d = Diagnostic(energy_budget_cb, every=1, path=path, name="energy_budget")
    evolve(rhs, u_hat, NSParams(nu=NU), g, 1e-3, 5, callbacks=(writer, d))
    series = read_series(path, "energy_budget")
    assert series is not None
    assert np.allclose(series["t"], d.times)
    assert np.allclose(series["E"], [v["E"] for v in d.values])
