"""bopfox-rs against bopjax: energy, forces, virial, dE/dshifts and the
quantities along the way, for the models both codes support.

Run from the bop-jax root after ``maturin develop`` (see bopfox-rs/README.md):
    uv run pytest bopfox-rs/tests
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import hamiltonian, pairwise
from bopjax.models import (CARBON_CONFIG, IRON_CONFIG, IRON_EGOROV_CONFIG, canonical_d,
                           carbon, iron, iron_egorov)

bopfox_rs = pytest.importorskip("bopfox_rs")

RNG = np.random.default_rng(7)


def lattice(base, a, n, noise):
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    pos = a * (grid[:, None, :] + np.asarray(base)[None]).reshape(-1, 3)
    return pos + RNG.normal(scale=noise, size=pos.shape), a * n * np.eye(3)


FCC = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
BCC = [[0, 0, 0], [0.5, 0.5, 0.5]]
DIAMOND = FCC + [[x + 0.25 for x in p] for p in FCC]


def compare(cfg, params, radial, pos, cell, shifts, tol=1e-9):
    n = len(pos)
    topo = bj.build_topology(pos, [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=radial)
    shifts = np.asarray(shifts).reshape(energy.shift_shape)
    p, c, s = jnp.array(pos), jnp.array(cell), jnp.array(shifts)

    e_fn = lambda s_, p_, c_: energy(s_, p_, c_, params)[0]  # noqa: E731
    e_ref, aux = energy(s, p, c, params)
    g_shift, g_pos = jax.grad(e_fn, argnums=(0, 1))(s, p, c)

    def strained(eps):
        d = jnp.eye(3) + 0.5 * (eps + eps.T)
        return e_fn(s, p @ d, c @ d)

    virial = jax.grad(strained)(jnp.zeros((3, 3)))

    rs = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, radial))
    out = rs.compute(pos, [0] * n, cell, shifts)

    scale = max(1.0, abs(float(e_ref)))
    assert out["energy"] == pytest.approx(float(e_ref), abs=tol * scale)
    assert out["fermi"] == pytest.approx(float(aux["fermi"]), abs=1e-9)
    np.testing.assert_allclose(out["forces"], -np.asarray(g_pos), atol=tol * 10)
    np.testing.assert_allclose(out["virial"], np.asarray(virial), atol=tol * 10 * n)
    np.testing.assert_allclose(out["shift_gradient"].reshape(shifts.shape), np.asarray(g_shift),
                               atol=tol * 10)
    np.testing.assert_allclose(out["energy_atom"], np.asarray(aux["energy_atom"]), atol=tol * 10)
    return out, aux


def test_canonical_d_band():
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged")
    pos, cell = lattice(FCC, 3.5, 2, 0.05)
    compare(cfg, canonical_d(hubbard=8.0), hamiltonian.powerlaw, pos, cell,
            RNG.normal(scale=0.01, size=(32, 1)))


@pytest.mark.parametrize("basis", [("d",), ("s", "d")])
def test_magnetic_iron(basis):
    cfg = bj.BOPConfig(spin="collinear", basis=basis, **IRON_CONFIG)
    pos, cell = lattice(BCC, 2.83, 2, 0.04)
    n, nsh = len(pos), len(basis)
    split = np.zeros((n, 2, nsh))
    split[:, 0, -1], split[:, 1, -1] = -1.2, 1.2  # exchange splitting of the d shell
    shifts = split + RNG.normal(scale=0.01, size=split.shape)
    out, aux = compare(cfg, iron(basis=basis), hamiltonian.v0_bond, pos, cell, shifts)
    np.testing.assert_allclose(out["moment"], np.asarray(aux["moment"]), atol=1e-9)


def test_carbon_sp():
    cfg = bj.BOPConfig(**CARBON_CONFIG)
    pos, cell = lattice(DIAMOND, 3.57, 1, 0.04)
    compare(cfg, carbon(), hamiltonian.v0_bond, pos, cell,
            RNG.normal(scale=0.02, size=(8, 2)))


def test_egorov_iron_without_yukawa():
    conf = dict(IRON_EGOROV_CONFIG, yukawa=False)
    cfg = bj.BOPConfig(spin="collinear", **conf)
    pos, cell = lattice(BCC, 2.84, 2, 0.03)
    shifts = np.zeros((len(pos), 2, 1))
    shifts[:, 0], shifts[:, 1] = -1.0, 1.0
    compare(cfg, iron_egorov(), pairwise.exp_sum, pos, cell, shifts)


def compare_scf(cfg, params, radial, pos, cell, shifts0, tol=1e-7):
    n = len(pos)
    topo = bj.build_topology(pos, [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=radial)
    shifts0 = np.asarray(shifts0, dtype=float).reshape(energy.shift_shape)
    p, c = jnp.array(pos), jnp.array(cell)
    ref = bj.make_scf(energy)(p, c, params, shifts0=jnp.array(shifts0), tol=1e-10)
    assert ref.converged
    f_ref = bj.make_forces(energy)(ref.shifts, p, c, params)

    rs = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, radial))
    out = rs.scf(pos, [0] * n, cell, shifts0, tol=1e-10)
    assert out["converged"]
    assert out["energy"] == pytest.approx(ref.energy, abs=1e-9 * n)
    np.testing.assert_allclose(out["shifts"].reshape(ref.shifts.shape), np.asarray(ref.shifts),
                               atol=tol)
    np.testing.assert_allclose(out["forces"], np.asarray(f_ref), atol=tol)
    return out


def test_scf_canonical_d_band():
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged")
    pos, cell = lattice(FCC, 3.5, 2, 0.05)
    out = compare_scf(cfg, canonical_d(hubbard=8.0), hamiltonian.powerlaw, pos, cell,
                      np.zeros((32, 1)))
    assert len(out["history"]) <= 20  # gradient evaluations


@pytest.mark.parametrize("basis", [("d",), ("s", "d")])
def test_scf_magnetic_iron(basis):
    cfg = bj.BOPConfig(spin="collinear", basis=basis, **IRON_CONFIG)
    pos, cell = lattice(BCC, 2.83, 2, 0.04)
    shifts0 = np.zeros((len(pos), 2, len(basis)))
    shifts0[:, 0, -1], shifts0[:, 1, -1] = -1.3, 1.3
    out = compare_scf(cfg, iron(basis=basis), hamiltonian.v0_bond, pos, cell, shifts0)
    assert np.all(out["moment"] > 1.5)  # ferromagnetic everywhere


def test_scf_carbon_sp():
    """Metallic simple-cubic carbon.  In distorted diamond the density of
    states at the Fermi level nearly vanishes, the stationary point is set by
    the non-variational part of the BOP energy, and there are several of them
    far apart (bopjax's Newton solver stalls there)."""
    cfg = bj.BOPConfig(**CARBON_CONFIG)
    pos, cell = lattice([[0, 0, 0]], 1.9218, 2, 0.02)
    compare_scf(cfg, carbon(), hamiltonian.v0_bond, pos, cell, np.zeros((8, 2)))


def bcc_cluster(a=2.83):
    """15-atom bcc cluster: a centre, its 8 nearest and 6 second neighbours."""
    pts = [[0, 0, 0]] + [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    pts += [[s * (i == 0), s * (i == 1), s * (i == 2)] for i in range(3) for s in (-1, 1)]
    rng = np.random.default_rng(7)  # fixed: bopjax converges for these positions
    return a * np.array(pts, dtype=float) + rng.normal(scale=0.03, size=(15, 3))


@pytest.mark.parametrize("structure", ["bulk", "cluster"])
def test_potential_condition(structure):
    cfg = bj.BOPConfig(spin="collinear", **IRON_CONFIG)
    params = iron()
    if structure == "bulk":
        pos, cell = lattice(BCC, 2.83, 2, 0.05)
    else:
        pos, cell = bcc_cluster(), None
    n = len(pos)
    shifts0 = np.zeros((n, 2, 1))
    shifts0[:, 0], shifts0[:, 1] = -1.3, 1.3
    topo = bj.build_topology(pos, [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=hamiltonian.v0_bond)
    # bopjax's plain Anderson mixing needs a small step on the cluster, whose
    # surface moments are close to saturation (3000 iterations there; bopfox-rs: ~120)
    mixing = 0.1 if structure == "bulk" else 0.05
    pot = bj.make_potential(energy, condition="potential", tol=1e-11, max_iter=3000, mixing=mixing)
    c = None if cell is None else jnp.array(cell)
    (e_ref, aux), g = jax.value_and_grad(lambda r: pot(r, c, params, jnp.array(shifts0)),
                                         has_aux=True)(jnp.array(pos))
    assert float(aux["scf_residual"]) < 1e-10

    rs = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, hamiltonian.v0_bond))
    out = rs.scf(pos, [0] * n, cell, shifts0, condition="potential", tol=1e-11)
    assert out["converged"]
    assert out["energy"] == pytest.approx(float(e_ref), abs=1e-8)
    np.testing.assert_allclose(out["shifts"], np.asarray(aux["shifts"]), atol=1e-8)
    np.testing.assert_allclose(out["forces"], -np.asarray(g), atol=1e-7)
    # not stationary, so the partial forces differ from the total ones
    partial = rs.compute(pos, [0] * n, cell, out["shifts"])["forces"]
    assert np.abs(partial - out["forces"]).max() > 1e-5


@pytest.mark.parametrize("xi", [0.0, 0.06])
def test_noncollinear(xi):
    """Spinor Hamiltonian with canted local axes, with and without spin-orbit
    coupling: energy, forces, dE/dshifts and dE/daxes at given shifts, and the
    self-consistent state."""
    cfg = bj.BOPConfig(spin="noncollinear", spin_orbit=xi > 0, **dict(IRON_CONFIG, nmom=7))
    a = 2.83  # 2x1x1 bcc supercell, 4 atoms
    pos = a * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0], [1.5, 0.5, 0.5]])
    pos = pos + RNG.normal(scale=0.04, size=pos.shape)
    cell = np.diag([2 * a, a, a])
    n = len(pos)
    axes = np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 0.9], [-0.2, 0.4, 0.8], [0.1, -0.3, 1.0]])
    params = dict(iron(), spin_axes=jnp.array(axes))
    if xi > 0:
        params["spin_orbit"] = jnp.array([[xi]])
    shifts = np.zeros((n, 2, 1))
    shifts[:, 0], shifts[:, 1] = -1.2, 1.2
    shifts += RNG.normal(scale=0.02, size=shifts.shape)

    topo = bj.build_topology(pos, [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=hamiltonian.v0_bond)
    p, c, s = jnp.array(pos), jnp.array(cell), jnp.array(shifts)

    def e_fn(s_, p_, a_):
        return energy(s_, p_, c, dict(params, spin_axes=a_))[0]

    e_ref = e_fn(s, p, jnp.array(axes))
    g_s, g_p, g_a = jax.grad(e_fn, argnums=(0, 1, 2))(s, p, jnp.array(axes))

    rs = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, hamiltonian.v0_bond))
    out = rs.compute(pos, [0] * n, cell, shifts, axes=axes)
    assert out["energy"] == pytest.approx(float(e_ref), abs=1e-9)
    np.testing.assert_allclose(out["forces"], -np.asarray(g_p), atol=1e-8)
    np.testing.assert_allclose(out["shift_gradient"].reshape(shifts.shape), np.asarray(g_s),
                               atol=1e-8)
    np.testing.assert_allclose(out["axes_gradient"], np.asarray(g_a), atol=1e-8)

    ref = bj.make_scf(energy)(p, c, params, shifts0=s, tol=1e-10)
    assert ref.converged
    out = rs.scf(pos, [0] * n, cell, shifts, tol=1e-10, axes=axes)
    assert out["converged"]
    assert out["energy"] == pytest.approx(ref.energy, abs=1e-9)
    np.testing.assert_allclose(out["shifts"], np.asarray(ref.shifts), atol=1e-7)


@pytest.mark.parametrize("spin,xi", [("collinear", 0.0), ("noncollinear", 0.0),
                                     ("noncollinear", 0.06)])
def test_magnetic_field(spin, xi):
    """Zeeman term of an external field: energy, forces, dE/dshifts,
    dE/daxes and the magnetic moments of bopjax.make_magnetic_moment, then
    the self-consistent state in the field."""
    a = 2.83
    pos = a * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0], [1.5, 0.5, 0.5]])
    pos = pos + np.random.default_rng(11).normal(scale=0.04, size=pos.shape)
    cell = np.diag([2 * a, a, a])
    n = len(pos)
    field = np.array([3.0, -2.0, 10.0])  # Tesla
    cfg = bj.BOPConfig(spin=spin, spin_orbit=xi > 0, **dict(IRON_CONFIG, nmom=7))
    params = dict(iron(), magnetic_field=jnp.array(field))
    axes = None
    if spin == "noncollinear":
        axes = np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 0.9], [-0.2, 0.4, 0.8], [0.1, -0.3, 1.0]])
        params["spin_axes"] = jnp.array(axes)
    if xi > 0:
        params["spin_orbit"] = jnp.array([[xi]])
    shifts = np.zeros((n, 2, 1))
    shifts[:, 0], shifts[:, 1] = -1.2, 1.2
    shifts += np.random.default_rng(12).normal(scale=0.02, size=shifts.shape)

    topo = bj.build_topology(pos, [0] * n, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=hamiltonian.v0_bond)
    p, c, s = jnp.array(pos), jnp.array(cell), jnp.array(shifts)
    e_ref = energy(s, p, c, params)[0]
    g_s, g_p = jax.grad(lambda s_, p_: energy(s_, p_, c, params)[0], argnums=(0, 1))(s, p)
    mom = bj.make_magnetic_moment(energy)(s, p, c, params)

    rs = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, params, hamiltonian.v0_bond))
    out = rs.compute(pos, [0] * n, cell, shifts, axes=axes, magnetic_field=field)
    assert out["energy"] == pytest.approx(float(e_ref), abs=1e-9)
    np.testing.assert_allclose(out["forces"], -np.asarray(g_p), atol=1e-8)
    np.testing.assert_allclose(out["shift_gradient"].reshape(shifts.shape), np.asarray(g_s),
                               atol=1e-8)
    for key in ("spin", "orbital", "total"):
        name = {"spin": "spin_moment", "orbital": "orbital_moment", "total": "magnetic_moment"}[key]
        np.testing.assert_allclose(out[name], np.asarray(mom[key]), atol=1e-8)
    if axes is not None:
        g_a = jax.grad(lambda a_: energy(s, p, c, dict(params, spin_axes=a_))[0])(jnp.array(axes))
        np.testing.assert_allclose(out["axes_gradient"], np.asarray(g_a), atol=1e-8)

    ref = bj.make_scf(energy)(p, c, params, shifts0=s, tol=1e-10)
    assert ref.converged
    out = rs.scf(pos, [0] * n, cell, shifts, tol=1e-10, axes=axes, magnetic_field=field)
    assert out["converged"]
    assert out["energy"] == pytest.approx(ref.energy, abs=1e-9)
    np.testing.assert_allclose(out["shifts"], np.asarray(ref.shifts), atol=1e-7)
