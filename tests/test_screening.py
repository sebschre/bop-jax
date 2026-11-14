"""Screening of the d-d bond integrals by a valence s orbital on the
surrounding atoms.

No BOPfox reference numbers yet.  The angular factors and weights of the
screening terms are checked against explicit Slater-Koster matrices of a
trimer, the rest against limits, symmetries and finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import hamiltonian as hm
from bopjax.geometry import bond_vectors, reverse_slots
from bopjax.models import canonical_d

A = 3.5
CELL_NP = A * np.eye(3)
CELL = jnp.array(CELL_NP)
POS_NP = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
POS_NP[0] += [0.15, -0.08, 0.05]
POS_NP[2] += [-0.05, 0.10, 0.02]
POS = jnp.array(POS_NP)
RCUT, DCUT = 3.3, 0.5


def screening(nsp=1, scale=1.0):
    """Made-up screening functions of a realistic size (power laws)."""

    def channel(prefactor, power):
        return jnp.broadcast_to(jnp.array([prefactor, 2.5, power]), (nsp, nsp, 3))

    return {
        "sds": channel(-1.2 * scale, 3.5),
        "sds_overlap": channel(0.18 * scale, 3.0),
        "dds_overlap": channel(-0.06 * scale, 4.0),
        "ddp_overlap": channel(0.04 * scale, 4.0),
        "ddd_overlap": channel(-0.01 * scale, 4.0),
    }


def screened_model(hubbard=8.0, **kwargs):
    return dict(canonical_d(hubbard=hubbard), screening=screening(**kwargs))


def setup(nmom=6, screening=True):
    cfg = bj.BOPConfig(nmom=nmom, rcut=RCUT, dcut=DCUT, screening=screening)
    topo = bj.build_topology(POS_NP, [0] * 4, cfg.rcut, cfg.nmom, CELL_NP)
    return bj.make_energy_fn(topo, cfg), topo


def terms(pos, species, params, cell=None, rcut=RCUT, dcut=DCUT):
    topo = bj.build_topology(pos, species, rcut, 2, cell)
    rij = bond_vectors(jnp.array(pos), None if cell is None else jnp.array(cell), topo)
    out = hm.screening_terms(rij, topo.nbr_mask, topo.species, topo.nbr_origin,
                             reverse_slots(topo), params, rcut, dcut)
    return topo, rij, out


def projectors(u):
    """Projectors on the sigma, pi and delta d orbitals about the axis u."""
    bu = hm.D_TENSORS @ u
    q = bu @ u
    sig = np.sqrt(1.5) * q
    p_sig = np.outer(sig, sig)
    p_pi = 2.0 * (bu @ bu.T - np.outer(q, q))
    return [p_sig, p_pi, np.eye(5) - p_sig - p_pi]


def test_terms_match_slater_koster_matrices_of_a_trimer():
    """c1, mu2 and mu3 are orbital averages of products of the overlap and
    Hamiltonian matrices of d orbitals on i, j and an s orbital on k."""
    rng = np.random.default_rng(1)
    pos = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [1.1, 2.0, 0.0]])
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    pos = pos @ rot.T + rng.normal(size=3)
    par = screening()
    topo, _, (c1b, mu2, mu3, o_dd) = terms(pos, [0, 0, 0], par, rcut=6.0, dcut=0.0)

    def radial(key, r):
        return float(hm.powerlaw(par[key][0, 0], r))

    def sd(key, d_atom, s_atom):
        """<d on d_atom | s on s_atom> as a 5-vector."""
        rvec = pos[d_atom] - pos[s_atom]
        r = np.linalg.norm(rvec)
        u = jnp.array(rvec / r)
        return np.asarray(hm._block_upper("s", "d", u, {"sds": radial(key, r)}))[0]

    i, j, k = 0, 1, 2
    slot = int(np.nonzero(topo.nbr_origin[i] == j)[0][0])
    rvec = pos[j] - pos[i]
    r = np.linalg.norm(rvec)
    proj = projectors(rvec / r)
    o_tau = [radial(c + "_overlap", r) for c in ("dds", "ddp", "ddd")]
    o_block = sum(o * p for o, p in zip(o_tau, proj))  # dd overlap, i with j
    oi, oj = sd("sds_overlap", i, k), sd("sds_overlap", j, k)
    hi, hj = sd("sds", i, k), sd("sds", j, k)

    for t, (p, deg) in enumerate(zip(proj, (1, 2, 2))):
        def avg(m):
            return np.trace(p @ m) / deg

        # -(O H + H O) / 2 through k: the first-order change of the bond integral
        assert c1b[i, slot, t] == pytest.approx(
            0.5 * avg(np.outer(oi, hj) + np.outer(hi, oj)), abs=1e-13)
        # (O^2)_ii and (O^2)_jj
        ring = 0.5 * avg(np.outer(oi, oi) + np.outer(oj, oj))
        assert mu2[i, slot, t] == pytest.approx(o_tau[t] ** 2 + ring, abs=1e-13)
        # (O^3)_ii: the two ways around the triangle
        assert mu3[i, slot, t] == pytest.approx(
            avg(o_block @ np.outer(oj, oi) + np.outer(oi, oj) @ o_block), abs=1e-13)
        assert o_dd[i, slot, t] == pytest.approx(o_tau[t], abs=1e-14)


def test_dimer_is_scaled_by_inverse_overlap():
    """Two atoms: beta~ = beta / (1 - O^2), the off-diagonal of O^-1 H."""
    pos = np.array([[0.0, 0.0, 0.0], [0.3, 2.4, -0.5]])
    par = screening()
    topo, rij, _ = terms(pos, [0, 0], par, rcut=6.0, dcut=0.0)
    scale, shift = hm.screening_factors(rij, topo.nbr_mask, topo.species,
                                        topo.nbr_origin, reverse_slots(topo), par, 6.0, 0.0)
    r = np.linalg.norm(pos[1])
    for t, c in enumerate(("dds", "ddp", "ddd")):
        o = float(hm.powerlaw(par[c + "_overlap"][0, 0], r))
        assert scale[0, 0, t] == pytest.approx(1.0 / (1.0 - o * o), abs=1e-14)
    np.testing.assert_allclose(shift, 0.0, atol=1e-15)


def test_no_overlap_gives_the_unscreened_energy():
    plain, _ = setup(screening=False)
    screened, _ = setup()
    shifts = jnp.zeros((4, 1))
    e0, _ = plain(shifts, POS, CELL, canonical_d(hubbard=8.0))
    e1, _ = screened(shifts, POS, CELL, screened_model(scale=0.0))
    e2, _ = screened(shifts, POS, CELL, screened_model())
    assert float(e1) == pytest.approx(float(e0), abs=1e-13)
    assert abs(float(e2 - e0)) > 1e-2  # and screening does change it


def test_screened_blocks_are_hermitian():
    _, topo = setup()
    par = screened_model()
    rij = bond_vectors(POS, CELL, topo)
    rev = reverse_slots(topo)
    screen = hm.screening_factors(rij, topo.nbr_mask, topo.species, topo.nbr_origin,
                                  rev, par["screening"], RCUT, DCUT)
    species_i = np.broadcast_to(topo.species[:, None], topo.nbr_origin.shape)
    blocks = hm.bond_blocks(rij, topo.nbr_mask, species_i, topo.species[topo.nbr_origin],
                            par["bond"], ("d",), RCUT, DCUT, dd_screen=screen)
    back = blocks[topo.nbr_origin, rev]
    np.testing.assert_allclose(blocks, np.swapaxes(back, -1, -2), atol=1e-14)
    assert np.abs(screen[0] - 1.0).max() > 1e-2


def test_two_species_blocks_are_hermitian():
    """sd functions that depend on which species carries the s orbital."""
    par = screening(nsp=2)
    par["sds"] = par["sds"] * jnp.array([[1.0, 0.7], [1.4, 1.2]])[:, :, None]
    par["sds_overlap"] = par["sds_overlap"] * jnp.array([[1.0, 1.3], [0.6, 0.9]])[:, :, None]
    topo, rij, _ = terms(POS_NP, [0, 1, 1, 0], par, cell=CELL_NP)
    rev = reverse_slots(topo)
    scale, shift = hm.screening_factors(rij, topo.nbr_mask, topo.species,
                                        topo.nbr_origin, rev, par, RCUT, DCUT)
    np.testing.assert_allclose(scale, scale[topo.nbr_origin, rev], atol=1e-14)
    np.testing.assert_allclose(shift, shift[topo.nbr_origin, rev], atol=1e-14)


def test_forces_and_stress_match_finite_differences():
    energy, _ = setup()
    par = screened_model()
    potential = jax.jit(bj.make_potential(energy, tol=1e-12))
    (e, aux), grad = jax.jit(jax.value_and_grad(potential, has_aux=True))(POS, CELL, par)
    force = -np.asarray(grad)
    assert float(aux["scf_residual"]) < 1e-11

    h = 1e-4
    for atom, axis in [(0, 0), (2, 1), (3, 2)]:
        def shifted(d):
            return float(potential(POS.at[atom, axis].add(d), CELL, par)[0])

        fd = -(8 * (shifted(h) - shifted(-h)) - (shifted(2 * h) - shifted(-2 * h))) / (12 * h)
        assert force[atom, axis] == pytest.approx(fd, abs=1e-8)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-10)

    stress = bj.make_stress(potential)(POS, CELL, par)
    np.testing.assert_allclose(stress, stress.T, atol=1e-10)
    eps = np.zeros((3, 3))
    eps[0, 1] = eps[1, 0] = 0.5
    eps[2, 2] = 1.0

    def strained(d):
        f = jnp.eye(3) + d * jnp.array(eps)
        return float(potential(POS @ f.T, CELL @ f.T, par)[0])

    fd = (8 * (strained(h) - strained(-h)) - (strained(2 * h) - strained(-2 * h))) / (12 * h)
    volume = float(jnp.linalg.det(CELL))
    assert np.sum(np.asarray(stress) * eps) == pytest.approx(fd / volume, abs=1e-8)


def test_gradient_is_finite_for_collinear_atoms():
    """A screening atom on the bond axis (sin = 0) must not produce NaN."""
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, 5.0], [2.0, 0.0, 1.0]])
    cfg = bj.BOPConfig(nmom=4, rcut=5.5, dcut=0.5, screening=True)
    topo = bj.build_topology(pos, [0] * 4, cfg.rcut, cfg.nmom)
    energy = bj.make_energy_fn(topo, cfg)
    grad = jax.grad(lambda p: energy(jnp.zeros((4, 1)), p, None, screened_model())[0])(
        jnp.array(pos))
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-10)
