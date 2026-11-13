"""Bandwidth estimators, terminators, pairwise terms, magnetism and stress.

Reference numbers were produced with BOPfox (v0.15) on the distorted fcc cell
below: canonical d-band integrals, 9 moments, rcut = 3.3, dcut = 0.5, Broyden
mixing with scftol = 1e-10.  The non-collinear reference needed a BOPfox
binary compiled with MAGNETISM = NCM.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import pairwise as pw
from bopjax.hamiltonian import v0_bond
from bopjax.models import canonical_d, tungsten_v0

A = 3.5
CELL = jnp.array(A * np.eye(3))
POS_NP = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
POS_NP[0] += [0.15, -0.08, 0.05]
POS_NP[2] += [-0.05, 0.10, 0.02]
POS = jnp.array(POS_NP)


def setup(species=(0, 0, 0, 0), rcut2=None, **kwargs):
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, rcut2=rcut2, **kwargs)
    topo = bj.build_topology(POS_NP, species, cfg.rcut, cfg.nmom, np.array(CELL), rcut2=rcut2)
    return bj.make_energy_fn(topo, cfg)


def voigt(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def iron(stoner, hubbard=3.0):
    return dict(canonical_d(n_electrons=7.0, hubbard=hubbard), stoner=jnp.array([[stoner]]))


def iron_tungsten(hubbard):
    """Two species with different band filling, onsite level and bond integrals."""

    def channel(aa, bb, ab):
        return jnp.array([[[aa, 2.081, 5.0], [ab, 2.081, 5.0]],
                          [[ab, 2.081, 5.0], [bb, 2.081, 5.0]]])

    return {
        "onsite": jnp.array([[0.0], [0.3]]),
        "n0": jnp.array([[7.0], [4.8]]),
        "hubbard": jnp.array(hubbard),
        "bond": {"dds": channel(-1.458, -1.458, -1.3), "ddp": channel(0.972, 0.972, 0.9),
                 "ddd": channel(-0.243, -0.243, -0.2)},
        "taper": jnp.broadcast_to(jnp.array([0.0, 5.0]), (2, 2, 2)),
    }


# --- pair repulsion and embedding -------------------------------------------------

def test_dimer_with_repulsion_and_embedding():
    """bopfox/BOPtest/tests/BOP/energy/dimer.2.5 (v0 model)."""
    pos = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cfg = bj.BOPConfig(nmom=6, rcut=4.0, dcut=0.01, rcut2=4.0, dcut2=0.01, repulsion=pw.V0)
    topo = bj.build_topology(pos, [0, 0], cfg.rcut, cfg.nmom, None, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=v0_bond)
    e, aux = energy(jnp.zeros((2, 1)), jnp.array(pos), None, tungsten_v0())
    assert float(aux["energy_rep"][0]) == pytest.approx(1.3819420293910154, abs=1e-12)
    assert float(aux["energy_bond"][0]) == pytest.approx(-5.574744039342524, abs=1e-8)
    assert float(e) / 2 == pytest.approx(-4.192802009951508, abs=1e-8)


# --- bandwidth estimators and terminators ----------------------------------------

BANDWIDTH_REFERENCE = {  # U_binding/atom and charges after self-consistency
    "azero": (-5.911551908540781, [-0.0088069697204398, 0.0126731850693913,
                                   -0.0049430137611424, 0.0010767984122824]),
    "lowest": (-5.960581212747969, [0.0001545669605045, 5.2637544913e-05,
                                    6.7937547508e-06, -0.0002139982601737]),
    "highest": (-5.967038146792762, [-3.45317954897e-05, 0.000162902822586,
                                     -5.56465454373e-05, -7.27244816394e-05]),
    "avgan_maxbn": (-5.9679847027556985, [0.0003390897735347, -0.0002882990365336,
                                          9.4385926162e-05, -0.0001451766630813]),
    "averaged": (-5.971588638945967, [0.0002690585933038, -0.0001782658493701,
                                      6.62025785241e-05, -0.0001569953224516]),
    "findeminemax": (-5.971773993132445, [-0.0002493818126084, 0.0004029374142469,
                                          -0.0001354813745307, -1.80742271336e-05]),
    "gershgorin": (-5.966629735067617, [-0.0001652406930903, 0.0003377694347506,
                                        -0.0001154485799999, -5.70801617572e-05]),
}


@pytest.mark.parametrize("bandwidth", list(BANDWIDTH_REFERENCE))
def test_bandwidth_estimators_self_consistent(bandwidth):
    energy = setup(bandwidth=bandwidth)
    res = bj.make_scf(energy)(POS, CELL, canonical_d(hubbard=8.0), tol=1e-11)
    u_ref, q_ref = BANDWIDTH_REFERENCE[bandwidth]
    assert res.converged and len(res.history) <= 6
    assert res.energy / 4 == pytest.approx(u_ref, abs=1e-11)
    np.testing.assert_allclose(res.aux["charge"], q_ref, atol=1e-9)


def test_global_band_shifts_and_stress():
    """With fixed band edges a uniform level shift is no longer a null mode."""
    energy = setup(bandwidth="global", global_ab=(0.1, 1.6))
    params = canonical_d(hubbard=8.0)
    res = bj.make_scf(energy)(POS, CELL, params, tol=1e-11)
    assert res.energy / 4 == pytest.approx(-6.001159530859818, abs=1e-11)
    np.testing.assert_allclose(
        res.shifts[:, 0],
        [0.11215575432420709, 0.11350396123784932, 0.1121548230357236, 0.11176297728589112],
        atol=1e-9)
    stress = bj.make_stress(bj.make_potential(energy))(POS, CELL, params)
    np.testing.assert_allclose(
        voigt(stress),
        [0.90619683, 0.90847802, 0.90898885, -0.00187475, 0.01031925, -0.02824306],
        atol=2e-8)


TERMINATOR_REFERENCE = [  # U_binding/atom at zero shifts (scfsteps = 1)
    (dict(terminator="constantabn", nexp=30), -5.974590311348095),
    (dict(terminator="oscillation", nexp=30), -6.030754179602523),
    (dict(terminator="weighted", nexp=30), -5.970349170164698),
    (dict(terminator="weighted", nexp=40), -5.9692193774684705),
    (dict(terminator="constantabn", nexp=40, kernel="jackson"), -5.975203365387686),
    (dict(kernel="jackson"), -6.001315422100687),
]


@pytest.mark.parametrize("options,u_ref", TERMINATOR_REFERENCE)
def test_terminators_and_kernel(options, u_ref):
    energy = setup(bandwidth="averaged", **options)
    e, _ = energy(jnp.zeros((4, 1)), POS, CELL, canonical_d(hubbard=8.0))
    assert float(e) / 4 == pytest.approx(u_ref, abs=1e-11)


def test_beer_pettifor_aoki():
    params = canonical_d(hubbard=8.0)
    zero = jnp.zeros((4, 1))
    # the variant that mimics BOPfox, whose root search stops at 1e-4
    e, _ = setup(bandwidth="beerpettiforaoki_bopfox")(zero, POS, CELL, params)
    assert float(e) / 4 == pytest.approx(-5.965609330914566, abs=1e-7)
    # the consistent variant does not depend on the origin of the energy scale
    energy = setup(bandwidth="beerpettiforaoki")
    e0, _ = energy(zero, POS, CELL, params)
    e1, _ = energy(zero + 0.37, POS, CELL, params)
    assert float(e0) == pytest.approx(float(e1), abs=1e-10)
    assert bj.make_scf(energy)(POS, CELL, params, tol=1e-11).converged


# --- charge transfer between two species ------------------------------------------

def test_onsite_coulomb_two_species():
    energy = setup(species=(0, 1, 0, 1), bandwidth="averaged")
    params = iron_tungsten([3.0, 5.0])
    res = bj.make_scf(energy)(POS, CELL, params, tol=1e-11)
    assert res.energy / 4 == pytest.approx(-5.086926078846509, abs=1e-11)
    np.testing.assert_allclose(
        res.aux["charge"], [-0.04240019, 0.04156652, -0.03846017, 0.03929385], atol=1e-8)
    stress = bj.make_stress(bj.make_potential(energy))(POS, CELL, params, res.shifts)
    np.testing.assert_allclose(
        voigt(stress),
        [0.81657554, 0.73903461, 0.82165599, -0.00037108, 0.00276540, -0.00854273],
        atol=2e-8)


def test_intersite_coulomb_two_species():
    energy = setup(species=(0, 1, 0, 1), rcut2=5.0, bandwidth="averaged", coulomb="frauenheim")
    params = iron_tungsten([8.0, 10.0])
    scf = bj.make_scf(energy)
    res = scf(POS, CELL, params, tol=1e-11)
    assert res.converged
    assert res.energy / 4 == pytest.approx(-5.095995780783055, abs=1e-11)
    assert float(jnp.mean(res.aux["energy_coulomb"])) == pytest.approx(
        -0.0848227152736586, abs=1e-10)
    np.testing.assert_allclose(
        res.aux["charge"], [-0.15452859, 0.14704056, -0.13454386, 0.14203189], atol=1e-8)

    # BOPfox leaves the Coulomb term out of its stress, so check against a
    # finite difference of the self-consistent energy instead.
    stress = bj.make_stress(bj.make_potential(energy))(POS, CELL, params, res.shifts)

    def e_strained(eps):
        deform = jnp.eye(3).at[0, 0].add(eps)
        return scf(POS @ deform, CELL @ deform, params, shifts0=res.shifts, tol=1e-12).energy

    h = 1e-4
    fd = (-e_strained(2 * h) + 8 * e_strained(h) - 8 * e_strained(-h) + e_strained(-2 * h)) / (12 * h)
    assert float(stress[0, 0]) == pytest.approx(fd / A**3, abs=1e-7)
    np.testing.assert_allclose(stress, stress.T, atol=1e-12)


# --- magnetism --------------------------------------------------------------------

def test_collinear_antiferromagnet():
    energy = setup(bandwidth="averaged", spin="collinear")
    params = iron(1.2)
    start = bj.stoner_shifts(np.array([2.0, 2.0, -2.0, -2.0])[:, None], 1.2)
    res = bj.make_scf(energy)(POS, CELL, params, shifts0=start, tol=1e-11)
    assert res.converged
    assert res.energy / 4 == pytest.approx(-5.1532921051325555, abs=1e-11)
    np.testing.assert_allclose(res.aux["moment"], [2.412, 2.475, -2.428, -2.451], atol=1e-3)
    np.testing.assert_allclose(
        np.array(res.shifts[:, :, 0]).ravel(),
        [-1.44023912, 1.3932056, -1.38424642, 1.43237472,
         1.40353631, -1.41689487, 1.41753744, -1.40527365],
        atol=1e-7)
    stress = bj.make_stress(bj.make_potential(energy))(POS, CELL, params, res.shifts)
    np.testing.assert_allclose(
        voigt(stress),
        [0.44575894, 0.55553283, 0.56814935, -0.00148834, 0.00301142, -0.00638631],
        atol=2e-8)


def test_collinear_ferromagnet():
    energy = setup(bandwidth="averaged", spin="collinear")
    start = bj.stoner_shifts(np.full((4, 1), 2.0), 1.2)
    res = bj.make_scf(energy)(POS, CELL, iron(1.2), shifts0=start, tol=1e-11)
    assert res.converged
    assert res.energy / 4 == pytest.approx(-5.618528785963795, abs=1e-11)
    np.testing.assert_allclose(res.aux["moment"], [2.991, 3.009, 2.996, 3.002], atol=1e-3)


AXES = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [2.0, 1.0, 0.0], [-2.0, 0.5, 0.0]])


@pytest.fixture(scope="module")
def noncollinear():
    energy = setup(bandwidth="averaged", spin="noncollinear")
    params = dict(iron(0.95), spin_axes=jnp.array(AXES))
    scf = bj.make_scf(energy)
    start = bj.stoner_shifts(np.full((4, 1), 2.0), 0.95)
    return energy, scf, params, scf(POS, CELL, params, shifts0=start, tol=1e-11, max_iter=40)


def test_noncollinear_matches_bopfox(noncollinear):
    energy, _, params, res = noncollinear
    assert res.converged
    assert res.energy / 4 == pytest.approx(-4.883417474987876, abs=1e-11)
    np.testing.assert_allclose(
        np.array(res.shifts[:, :, 0]).ravel(),
        [-1.04717233, 1.17404423, -1.07399621, 1.21030509,
         -1.05860875, 1.20821795, -1.30674108, 0.89395109],
        atol=1e-7)
    # BOPfox prints the moment vectors; their lengths are the local moments
    np.testing.assert_allclose(
        res.aux["moment"],
        [2.3088529619, np.hypot(1.6947673880, 1.6947673880),
         np.hypot(2.1074100886, 1.0537050443), np.hypot(1.9680434749, 0.4920108687)],
        atol=1e-8)
    f = bj.make_forces(energy)(res.shifts, POS, CELL, params)
    f_bopfox = [[3.4205417548, -1.3149911809, 0.8632717058],
                [0.8572289705, -0.2242111615, -0.5054191258],
                [-2.6584089256, 1.6237534073, -0.2394072515],
                [-1.6193616970, -0.0845509884, -0.1184453834]]
    np.testing.assert_allclose(f, f_bopfox, atol=1e-6)


def test_noncollinear_symmetries(noncollinear):
    energy, _, params, res = noncollinear
    # rotating all spins together changes nothing
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    e_rot, _ = energy(res.shifts, POS, CELL, dict(params, spin_axes=jnp.array(AXES @ q.T)))
    assert float(e_rot) == pytest.approx(res.energy, abs=1e-11)
    # so the torques sum to zero, and the energy does not depend on |axis|
    grad = jax.grad(lambda ax: energy(res.shifts, POS, CELL, dict(params, spin_axes=ax))[0])(
        jnp.array(AXES))
    np.testing.assert_allclose(jnp.sum(grad * AXES, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(jnp.cross(AXES, grad), axis=0), 0.0, atol=1e-12)


def test_noncollinear_torque_is_total_derivative(noncollinear):
    energy, scf, params, res = noncollinear
    grad = jax.grad(lambda ax: energy(res.shifts, POS, CELL, dict(params, spin_axes=ax))[0])(
        jnp.array(AXES))

    def e_scf(delta):
        axes = jnp.array(AXES).at[1, 1].add(delta)
        return scf(POS, CELL, dict(params, spin_axes=axes), shifts0=res.shifts, tol=1e-12).energy

    h = 1e-3
    fd = (-e_scf(2 * h) + 8 * e_scf(h) - 8 * e_scf(-h) + e_scf(-2 * h)) / (12 * h)
    assert float(grad[1, 1]) == pytest.approx(fd, abs=1e-8)


def test_noncollinear_reduces_to_collinear():
    energy = setup(bandwidth="averaged", spin="noncollinear")
    params = dict(iron(1.2), spin_axes=jnp.array([[1.0, 1.0, 0.0]] * 4))
    start = bj.stoner_shifts(np.array([2.0, 2.0, -2.0, -2.0])[:, None], 1.2)
    res = bj.make_scf(energy)(POS, CELL, params, shifts0=start, tol=1e-11)
    assert res.energy / 4 == pytest.approx(-5.1532921051325555, abs=1e-11)


# --- compiled potential and molecular dynamics ------------------------------------

def test_potential_gradient_is_the_force():
    energy = setup(bandwidth="averaged")
    params = canonical_d(hubbard=8.0)
    potential = jax.jit(bj.make_potential(energy))
    e, aux = potential(POS, CELL, params)
    res = bj.make_scf(energy)(POS, CELL, params, tol=1e-11)
    assert float(e) == pytest.approx(res.energy, abs=1e-10)
    assert float(aux["scf_residual"]) < 1e-10
    f_potential = -jax.grad(lambda p: potential(p, CELL, params)[0])(POS)
    f_fixed = bj.make_forces(energy)(res.shifts, POS, CELL, params)
    np.testing.assert_allclose(f_potential, f_fixed, atol=1e-8)


def test_needs_rebuild():
    topo = bj.build_topology(POS_NP, [0] * 4, 3.3, 6, np.array(CELL), skin=0.4)
    assert not bj.needs_rebuild(POS_NP, topo)
    moved = POS_NP.copy()
    moved[1, 0] += 0.25
    assert bj.needs_rebuild(moved, topo)


def test_jaxmd_conserves_energy():
    pytest.importorskip("jax_md")
    from jax_md import quantity, simulate

    from bopjax import md

    a = 3.165
    cell = a * np.eye(3)
    pos = a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    pos = pos + np.random.default_rng(0).normal(scale=0.03, size=(2, 3))
    cfg = bj.BOPConfig(nmom=6, rcut=3.8, dcut=0.5, rcut2=3.8, dcut2=0.5, repulsion=pw.V0)
    topo = bj.build_topology(pos, [0, 0], cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2, skin=0.3)
    energy = md.jaxmd_energy_fn(bj.make_energy_fn(topo, cfg, radial=v0_bond), cell, tungsten_v0())
    _, shift = md.jaxmd_space()

    init, apply = simulate.nve(energy, shift, 1.0 / md.TIME_UNIT_FS)
    state = init(jax.random.PRNGKey(0), jnp.array(pos), mass=183.84, kT=300 * md.KB_EV)
    apply, energy = jax.jit(apply), jax.jit(energy)

    def total(s):
        return float(energy(s.position) + quantity.kinetic_energy(momentum=s.momentum, mass=s.mass))

    e0 = total(state)
    for _ in range(20):
        state = apply(state)
    assert total(state) == pytest.approx(e0, abs=1e-6)
    assert not bj.needs_rebuild(state.position, topo)
