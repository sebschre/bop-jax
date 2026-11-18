"""External magnetic field (Zeeman term), magnetic moments as field
derivatives, first-order barrier shifts and the NEB driver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import neb, zeeman
from bopjax.models import canonical_d
from bopjax.zeeman import G_SPIN, MU_B

# --- bcc ferromagnet, canonical d band ---------------------------------------------

BCC_STONER = 0.6  # unsaturated (2.67 muB); at 0.9 the majority band is full
A_BCC = 2.9


@pytest.fixture(scope="module")
def bcc():
    """Self-consistent two-atom bcc ferromagnet in a field, collinear or with
    non-collinear axes."""
    pos = A_BCC * jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    cell = A_BCC * jnp.eye(3)
    base = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[BCC_STONER]]))
    start = bj.stoner_shifts(np.full((2, 1), 2.0), BCC_STONER)
    solvers = {}
    for spin in ("collinear", "noncollinear"):
        cfg = bj.BOPConfig(nmom=6, rcut=3.4, dcut=0.4, bandwidth="averaged", spin=spin)
        topo = bj.build_topology(pos, [0, 0], cfg.rcut, cfg.nmom, cell)
        energy = bj.make_energy_fn(topo, cfg)
        solvers[spin] = (energy, bj.make_scf(energy))

    def solve(spin, axes=None, shifts0=start, **fields):
        energy, scf = solvers[spin]
        params = dict(base, **{k: jnp.asarray(v, dtype=float) for k, v in fields.items()})
        if spin == "noncollinear":
            params["spin_axes"] = jnp.array([[0.0, 0.0, 1.0]] * 2 if axes is None else axes)
        res = scf(pos, cell, params, shifts0=shifts0, tol=1e-12, max_iter=40)
        assert res.converged
        return res, params, energy, pos, cell

    return solve


def test_zero_field_changes_nothing(bcc):
    for spin in ("collinear", "noncollinear"):
        plain = bcc(spin)[0]
        zero = bcc(spin, magnetic_field=[0.0, 0.0, 0.0])[0]
        assert zero.energy == pytest.approx(plain.energy, abs=1e-12)
        np.testing.assert_allclose(zero.shifts, plain.shifts, atol=1e-10)


def test_collinear_moment_and_first_order_energy(bcc):
    res, params, energy, pos, cell = bcc("collinear")
    m = bj.make_magnetic_moment(energy)(res.shifts, pos, cell, params)
    # g/2 m along z, nothing else
    np.testing.assert_allclose(m["spin"][:, 2], 0.5 * G_SPIN * res.aux["moment"], atol=1e-10)
    np.testing.assert_allclose(m["spin"][:, :2], 0.0, atol=1e-14)
    np.testing.assert_allclose(m["orbital"], 0.0, atol=1e-14)
    total = float(jnp.sum(m["total"][:, 2]))
    # it is the field derivative of the self-consistent energy
    h = 5.0
    plus = bcc("collinear", shifts0=res.shifts, magnetic_field=[0, 0, h])[0]
    minus = bcc("collinear", shifts0=res.shifts, magnetic_field=[0, 0, -h])[0]
    assert -(plus.energy - minus.energy) / (2 * h) / MU_B == pytest.approx(total, abs=1e-6)
    # the field lowers the energy and raises the moment along it (paraprocess);
    # the shifts stay the internal potential: the field moves the levels itself
    assert plus.energy < res.energy < minus.energy
    assert float(plus.aux["moment"][0]) > float(res.aux["moment"][0])
    np.testing.assert_allclose(plus.aux["energy_zeeman"],
                               -MU_B * h * 0.5 * G_SPIN * plus.aux["moment"], rtol=1e-12)


def test_noncollinear_spin_field_matches_collinear(bcc):
    # axes along z and a spin-only source along z: the same functional
    for h in (0.01, -0.02):
        col = bcc("collinear", spin_field=[[0, 0, h]] * 2)[0]
        nc = bcc("noncollinear", spin_field=[[0, 0, h]] * 2)[0]
        assert nc.energy == pytest.approx(col.energy, abs=1e-11)
        np.testing.assert_allclose(nc.aux["moment"], col.aux["moment"], atol=1e-9)
    # with the orbital term the full field adds a (Van Vleck) orbital moment of
    # first order in B, which lowers the energy further at second order
    b = 200.0
    col = bcc("collinear", magnetic_field=[0, 0, b])[0]
    nc = bcc("noncollinear", magnetic_field=[0, 0, b])[0]
    assert nc.energy < col.energy - 1e-6
    res, params, energy, pos, cell = bcc("noncollinear", magnetic_field=[0, 0, b])
    orbital = bj.make_magnetic_moment(energy)(res.shifts, pos, cell, params)["orbital"]
    assert float(orbital[0, 2]) > 0.0
    np.testing.assert_allclose(orbital[:, :2], 0.0, atol=1e-12)


def test_field_rotates_the_isotropic_ferromagnet_in_one_piece(bcc):
    # without spin-orbit coupling a common rotation of axes and field changes
    # nothing, and a field across the axes exerts the torque of a rigid moment
    b = 50.0
    along = bcc("noncollinear", magnetic_field=[0, 0, b])[0]
    axes = [[1.0, 0.0, 0.0]] * 2
    turned = bcc("noncollinear", axes=axes, magnetic_field=[b, 0, 0])[0]
    assert turned.energy == pytest.approx(along.energy, abs=1e-11)

    res, params, energy, pos, cell = bcc("noncollinear", magnetic_field=[1.0, 0, 0])
    g = jax.grad(lambda ax: energy(res.shifts, pos, cell, dict(params, spin_axes=ax))[0])(
        params["spin_axes"])
    rigid = -MU_B * 0.5 * G_SPIN * np.asarray(res.aux["moment"])
    np.testing.assert_allclose(g[:, 0], rigid, rtol=1e-3)
    # the moment along the axis is the whole spin moment: no transverse spin
    m = bj.make_magnetic_moment(energy)(res.shifts, pos, cell, params)["spin"]
    np.testing.assert_allclose(m[:, 2], 0.5 * G_SPIN * res.aux["moment"], rtol=1e-6)
    np.testing.assert_allclose(m[:, 1], 0.0, atol=1e-12)


def test_field_needs_magnetism():
    pos = jnp.zeros((1, 3))
    cell = jnp.diag(jnp.array([12.0, 12.0, 2.2]))
    cfg = bj.BOPConfig(nmom=4, rcut=3.3)
    topo = bj.build_topology(pos, [0], cfg.rcut, cfg.nmom, cell)
    energy = bj.make_energy_fn(topo, cfg)
    params = dict(canonical_d(), magnetic_field=jnp.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="magnetic calculation"):
        energy(jnp.zeros(energy.shift_shape), pos, cell, params)
    with pytest.raises(ValueError, match="magnetic calculation"):
        bj.make_magnetic_moment(energy)
    cfg = bj.BOPConfig(nmom=4, rcut=3.3, spin="collinear")
    energy = bj.make_energy_fn(bj.build_topology(pos, [0], 3.3, 4, cell), cfg)
    params = dict(canonical_d(), stoner=jnp.array([[0.6]]), orbital_field=jnp.zeros((1, 3)))
    with pytest.raises(ValueError, match="noncollinear"):
        energy(jnp.zeros(energy.shift_shape), pos, cell, params)


# --- chain with spin-orbit coupling ----------------------------------------------------

CHAIN_STONER = 0.6


@pytest.fixture(scope="module")
def chain():
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged", spin="noncollinear",
                       spin_orbit=True)
    pos, cell = jnp.zeros((1, 3)), jnp.diag(jnp.array([12.0, 12.0, 2.2]))
    topo = bj.build_topology(pos, [0], cfg.rcut, cfg.nmom, cell)
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)
    moment = jax.jit(bj.make_magnetic_moment(energy))
    orbital = jax.jit(bj.make_orbital_moment(energy))
    start = bj.stoner_shifts(np.full((1, 1), 2.0), CHAIN_STONER)

    def solve(axis, xi, field=None, shifts0=start):
        params = dict(canonical_d(n_electrons=7.0, hubbard=3.0),
                      stoner=jnp.array([[CHAIN_STONER]]),
                      spin_axes=jnp.array([axis], dtype=float), spin_orbit=jnp.array([[xi]]))
        if field is not None:
            params["magnetic_field"] = jnp.array(field, dtype=float)
        res = scf(pos, cell, params, shifts0=shifts0, tol=1e-12, max_iter=40)
        assert res.converged
        return (res, moment(res.shifts, pos, cell, params),
                orbital(res.shifts, pos, cell, params))

    return solve


def test_moment_is_the_field_derivative_with_spin_orbit_coupling(chain):
    axis, xi, h = [1.0, 0.0, 1.0], 0.05, 2.0
    res, m, orbital = chain(axis, xi)
    np.testing.assert_allclose(m["orbital"], orbital, atol=1e-12)
    # spin-orbit coupling induces spin density across the tilted axis
    e = np.array(axis) / np.linalg.norm(axis)
    spin = np.asarray(m["spin"][0])
    along = 0.5 * G_SPIN * float(res.aux["moment"][0])
    assert float(spin @ e) == pytest.approx(along, abs=1e-6)
    transverse = spin - (spin @ e) * e
    assert 1e-4 < np.linalg.norm(transverse) < 0.05 * along
    for comp in range(3):
        step = np.eye(3)[comp] * h
        plus = chain(axis, xi, field=step, shifts0=res.shifts)[0].energy
        minus = chain(axis, xi, field=-step, shifts0=res.shifts)[0].energy
        fd = -(plus - minus) / (2 * h) / MU_B
        assert fd == pytest.approx(float(m["total"][0, comp]), abs=2e-6)
    # without spin-orbit coupling there is neither transverse spin nor <L>
    res0, m0, _ = chain(axis, 0.0)
    np.testing.assert_allclose(m0["orbital"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m0["spin"][0], 0.5 * G_SPIN * float(res0.aux["moment"][0]) * e,
                               atol=1e-9)


# --- first-order barrier and Peierls stress ---------------------------------------------

def test_first_order_barrier_on_a_synthetic_path():
    p = 9
    s = np.linspace(0.0, 1.0, p)
    energies = 0.04 * np.sin(np.pi * s) ** 2
    moments = np.zeros((p, 4, 3))
    moments[..., 2] = 2.2
    moments[:, 0, 2] += 0.1 * np.sin(np.pi * s)  # one core atom changes its moment
    out = zeeman.first_order_barrier(energies, moments)
    assert out["saddle"] == 4
    assert out["barrier"] == pytest.approx(0.04)
    np.testing.assert_allclose(out["delta_moment"], [0.0, 0.0, 0.1], atol=1e-14)
    np.testing.assert_allclose(out["gradient"], [0.0, 0.0, -0.1 * MU_B], rtol=1e-14)
    de = zeeman.first_order_energies(moments, [0.0, 0.0, 10.0])
    np.testing.assert_allclose(de, -MU_B * 10.0 * 0.1 * np.sin(np.pi * s), atol=1e-15)


def test_peierls_stress_of_a_sinusoidal_profile():
    # E = E_P sin^2(pi x / d): max dE/dx = pi E_P / d
    d, ep, b, length = 2.33, 0.035, 2.47, 2.47
    x = np.linspace(0.0, d, 11)
    e = ep * np.sin(np.pi * x / d) ** 2
    slopes = ep * np.pi / d * np.sin(2 * np.pi * x / d)
    tau, where = zeeman.peierls_stress(x, e, b, length, slopes=slopes)
    assert tau == pytest.approx(np.pi * ep / d / (b * length), rel=2e-3)
    assert where == pytest.approx(d / 4, abs=0.02)
    tau_fd, _ = zeeman.peierls_stress(x, e, b, length)
    assert tau_fd == pytest.approx(tau, rel=0.05)  # second-order finite differences


# --- NEB ------------------------------------------------------------------------------

def test_neb_finds_the_saddle_of_a_curved_valley():
    # minima at (+-1, 0), saddle at (0, 0.3) with energy 1
    def force_fn(_, r):
        x, y = r[0, 0], r[0, 1]
        q = y - 0.3 * (1 - x * x)
        e = (x * x - 1) ** 2 + 2 * q * q
        dx = 4 * x * (x * x - 1) + 4 * q * 0.6 * x
        dy = 4 * q
        return e, -np.array([[dx, dy, 0.0]])

    p = 9
    images = np.zeros((p, 1, 3))
    images[:, 0, 0] = np.linspace(-1.0, 1.0, p)
    res = neb.neb(force_fn, images, spring=1.0, fmax=1e-4, steps=3000, dt=0.05)
    assert res.converged
    top = res.climbing
    np.testing.assert_allclose(res.images[top, 0, :2], [0.0, 0.3], atol=1e-3)
    assert res.energies[top] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(res.path) > 0)
    # slopes are dE/ds: positive on the way up, zero at the saddle
    assert np.all(res.slopes[1:top] > 0) and abs(res.slopes[top]) < 1e-3
