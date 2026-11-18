"""Newton self-consistency beyond the single-species d-band case."""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.models import canonical_d

A = 3.5
CELL = jnp.array(A * np.eye(3))
POS = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
POS[0] += [0.15, -0.08, 0.05]
POS[2] += [-0.05, 0.10, 0.02]


def fd_force(scf, pos, params, shifts, i, d, h=1e-3):
    def e(delta):
        return scf(pos.at[i, d].add(delta), CELL, params, shifts0=shifts, tol=1e-13).energy

    return -(-e(2 * h) + 8 * e(h) - 8 * e(-h) + e(-2 * h)) / (12 * h)


def test_two_species_charge_transfer():
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5)
    topo = bj.build_topology(POS, [0, 1, 0, 1], cfg.rcut, cfg.nmom, np.array(CELL))
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)
    params = canonical_d(n_electrons=[4.0, 5.6], onsite=[0.3, -0.2], hubbard=[6.0, 8.0])
    pos = jnp.array(POS)
    res = scf(pos, CELL, params, tol=1e-11)

    assert res.converged and len(res.history) <= 5
    assert float(jnp.sum(res.aux["charge"])) == pytest.approx(0.0, abs=1e-9)
    f = bj.make_forces(energy)(res.shifts, pos, CELL, params)
    assert float(f[1, 0]) == pytest.approx(fd_force(scf, pos, params, res.shifts, 1, 0), abs=1e-8)


def test_two_shells_per_atom():
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5, basis=("s", "d"))
    topo = bj.build_topology(POS, [0] * 4, cfg.rcut, cfg.nmom, np.array(CELL))
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)

    def channel(prefactor, power):
        return jnp.array([[[prefactor, 2.081, power]]])

    params = {
        "onsite": jnp.array([[0.4, 0.0]]),
        "n0": jnp.array([[1.0, 4.8]]),
        "hubbard": jnp.array([6.0]),
        "bond": {
            "sss": channel(-1.0, 2.0),
            "sds": channel(-0.9, 3.5),
            "dds": channel(-1.458, 5.0),
            "ddp": channel(0.972, 5.0),
            "ddd": channel(-0.243, 5.0),
        },
    }
    pos = jnp.array(POS)
    res = scf(pos, CELL, params, tol=1e-11)

    assert res.converged and len(res.history) <= 8
    assert res.shifts.shape == (4, 2)
    f = bj.make_forces(energy)(res.shifts, pos, CELL, params)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
    assert float(f[0, 1]) == pytest.approx(fd_force(scf, pos, params, res.shifts, 0, 1), abs=1e-8)


def test_krylov_solver_matches_dense():
    """Hessian-free Newton gives the same self-consistent state."""
    n = 2
    base = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    pos = A * (grid[:, None, :] + base[None, :, :]).reshape(-1, 3)
    pos = pos + np.random.default_rng(1).normal(scale=0.05, size=pos.shape)
    cell = jnp.array(A * n * np.eye(3))
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5)
    topo = bj.build_topology(pos, [0] * len(pos), cfg.rcut, cfg.nmom, np.array(cell))
    energy = bj.make_energy_fn(topo, cfg)
    params = canonical_d(hubbard=8.0)
    pos = jnp.array(pos)

    dense = bj.make_scf(energy, method="dense")(pos, cell, params, tol=1e-10)
    krylov = bj.make_scf(energy, method="krylov")(pos, cell, params, tol=1e-10)
    assert dense.converged and krylov.converged
    assert len(krylov.history) <= 6
    assert krylov.energy == pytest.approx(dense.energy, abs=1e-10)
    np.testing.assert_allclose(krylov.shifts, dense.shifts, atol=1e-8)

    e, aux = bj.make_potential(energy, method="krylov")(pos, cell, params)
    assert float(e) == pytest.approx(dense.energy, abs=1e-9)
    assert float(aux["scf_residual"]) < 1e-10


def test_clusters_grow_by_hop_shells():
    """Perfect fcc with nearest neighbours: 1, 13, 55, 147 atoms within 0..3 hops."""
    pos = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    topo = bj.build_topology(pos, [0] * 4, 3.01, 6, A * np.eye(3))
    assert topo.cluster_sizes == (1, 13, 55, 147)
    # every listed neighbour of the central atom is in its first shell
    assert set(topo.cluster_nbr[0, 0]) == set(range(1, 13))


def test_potential_with_frozen_levels():
    # self-consistency on atoms 0 and 1 only; the levels of 2 and 3 stay as given
    import jax

    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5, spin="collinear")
    topo = bj.build_topology(POS, [0] * 4, cfg.rcut, cfg.nmom, np.array(CELL))
    energy = bj.make_energy_fn(topo, cfg)
    params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[0.6]]))
    pos = jnp.array(POS)
    start = bj.stoner_shifts(np.full((4, 1), 2.0), 0.6)
    full = bj.make_scf(energy)(pos, CELL, params, shifts0=start, tol=1e-12)
    assert full.converged
    # frozen levels near, but not at, self-consistency (as for a fixed outer shell)
    # (not a common shift of all levels: that only moves the Fermi level)
    frozen = full.shifts.at[2, :, 0].add(jnp.array([0.03, -0.03])).at[3, 0, 0].add(0.05)
    free = np.array([True, True, False, False])
    potential = bj.make_potential(energy, tol=1e-12, free=free)
    e, aux = potential(pos, CELL, params, frozen)
    np.testing.assert_allclose(aux["shifts"][2:], frozen[2:], atol=0)
    g = jax.grad(lambda s: energy(s, pos, CELL, params)[0])(aux["shifts"])
    assert float(jnp.max(jnp.abs(g[:2]))) < 1e-12
    assert float(jnp.max(jnp.abs(g[2:]))) > 1e-3  # not stationary in the frozen levels
    # forces are still the total derivative: the energy is stationary in the free levels
    force = -jax.grad(lambda x: potential(x, CELL, params, frozen)[0])(pos)
    h = 1e-3
    for i, d in ((0, 0), (3, 2)):
        def e_at(delta):
            return float(potential(pos.at[i, d].add(delta), CELL, params, frozen)[0])

        fd = -(-e_at(2 * h) + 8 * e_at(h) - 8 * e_at(-h) + e_at(-2 * h)) / (12 * h)
        assert float(force[i, d]) == pytest.approx(fd, abs=1e-8)


# --- potential self-consistency (Anderson mixing, implicit gradients) ---------------

def _magnetic_cell(positions, cell, stoner=0.6):
    cfg = bj.BOPConfig(nmom=6, rcut=3.3, dcut=0.5, spin="collinear", bandwidth="averaged")
    topo = bj.build_topology(positions, [0] * len(positions), cfg.rcut, cfg.nmom,
                             np.asarray(cell))
    params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[stoner]]))
    return bj.make_energy_fn(topo, cfg), params


# distorted fcc cell: magnetic only from I = 1.4, and then close to saturation
# (3.0 muB for 7 d electrons), where dN/d(shifts) nearly vanishes
DISTORTED_STONER = 1.4


def test_potential_condition_matches_stationary_in_a_perfect_crystal():
    a = 2.9
    pos = a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    cell = jnp.array(a * np.eye(3))
    energy, params = _magnetic_cell(pos, cell)
    start = bj.stoner_shifts(np.full((2, 1), 2.0), 0.6)
    stat = bj.make_potential(energy, tol=1e-12)(jnp.array(pos), cell, params, start)
    mixed = bj.make_potential(energy, tol=1e-12, condition="potential")(
        jnp.array(pos), cell, params, start)
    assert float(mixed[1]["scf_residual"]) < 1e-12
    assert float(mixed[0]) == pytest.approx(float(stat[0]), abs=1e-10)
    np.testing.assert_allclose(mixed[1]["moment"], stat[1]["moment"], atol=1e-9)
    # the stationary shifts already satisfy the potential condition
    target = energy.potential_target(stat[1]["occupations"], jnp.array(pos), cell, params)
    np.testing.assert_allclose(target - stat[1]["shifts"], 0.0, atol=1e-9)


def test_potential_condition_gradients_carry_the_response():
    import jax

    pos = jnp.array(POS)
    energy, params = _magnetic_cell(POS, CELL, DISTORTED_STONER)
    start = bj.stoner_shifts(np.full((4, 1), 2.0), DISTORTED_STONER)
    potential = bj.make_potential(energy, tol=1e-12, condition="potential")
    e, aux = potential(pos, CELL, params, start)
    assert float(aux["scf_residual"]) < 1e-12
    assert float(jnp.min(aux["moment"])) > 2.9
    # in a distorted cell the energy is not stationary in the shifts...
    g = jax.grad(lambda s: energy(s, pos, CELL, params)[0])(aux["shifts"])
    assert float(jnp.max(jnp.abs(g))) > 1e-6
    # ...but jax.grad of the potential is the total derivative
    force = -jax.grad(lambda x: potential(x, CELL, params, start)[0])(pos)
    h = 1e-3
    for i, d in ((0, 0), (2, 1)):
        def e_at(delta):
            return float(potential(pos.at[i, d].add(delta), CELL, params, aux["shifts"])[0])

        fd = -(-e_at(2 * h) + 8 * e_at(h) - 8 * e_at(-h) + e_at(-2 * h)) / (12 * h)
        assert float(force[i, d]) == pytest.approx(fd, abs=1e-7)
    np.testing.assert_allclose(jnp.sum(force, axis=0), 0.0, atol=1e-9)
    # and with respect to a parameter that enters the potentials only
    def e_of(stoner):
        return potential(pos, CELL, dict(params, stoner=stoner), start)[0]

    d_stoner = float(jax.grad(e_of)(params["stoner"])[0, 0])
    fd = (float(e_of(params["stoner"] + 1e-4)) - float(e_of(params["stoner"] - 1e-4))) / 2e-4
    assert d_stoner == pytest.approx(fd, abs=1e-6)


def test_potential_condition_with_frozen_levels():
    import jax

    pos = jnp.array(POS)
    energy, params = _magnetic_cell(POS, CELL, DISTORTED_STONER)
    start = bj.stoner_shifts(np.full((4, 1), 2.0), DISTORTED_STONER)
    full = bj.make_potential(energy, tol=1e-12, condition="potential")(pos, CELL, params, start)
    frozen = full[1]["shifts"].at[3, 0, 0].add(0.05)
    free = np.array([True, True, True, False])
    potential = bj.make_potential(energy, tol=1e-12, condition="potential", free=free)
    e, aux = potential(pos, CELL, params, frozen)
    np.testing.assert_allclose(aux["shifts"][3], frozen[3], atol=0)
    target = energy.potential_target(aux["occupations"], pos, CELL, params)
    np.testing.assert_allclose((target - aux["shifts"])[:3], 0.0, atol=1e-11)
    force = -jax.grad(lambda x: potential(x, CELL, params, frozen)[0])(pos)
    h = 1e-3

    def e_at(delta):
        return float(potential(pos.at[1, 2].add(delta), CELL, params, frozen)[0])

    fd = -(-e_at(2 * h) + 8 * e_at(h) - 8 * e_at(-h) + e_at(-2 * h)) / (12 * h)
    assert float(force[1, 2]) == pytest.approx(fd, abs=1e-7)
    with pytest.raises(ValueError, match="condition"):
        bj.make_potential(energy, condition="mixing")
