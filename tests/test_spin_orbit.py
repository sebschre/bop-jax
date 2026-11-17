"""Onsite spin-orbit coupling xi L.S in the non-collinear energy.

BOPfox has no spin-orbit coupling, so everything here is checked internally:
the angular momentum matrices against their algebra and against the rotation
of the Slater-Koster blocks, the moments against powers of an explicitly built
spinor matrix, and the energy against the symmetries that spin-orbit coupling
keeps (common rotation of lattice and spins) and breaks (rotation of the spins
alone).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import bopjax as bj
from bopjax import bop
from bopjax.geometry import bond_vectors
from bopjax.hamiltonian import angular_momentum, bond_blocks, shell_slices
from bopjax.models import canonical_d


def channel(prefactor):
    return jnp.array([[[prefactor, 2.081, 5.0]]])


PD_BOND = {"pps": channel(1.9), "ppp": channel(-0.6), "pds": channel(-1.1),
           "pdp": channel(0.7), "dds": channel(-1.458), "ddp": channel(0.972),
           "ddd": channel(-0.243)}


# --- angular momentum -------------------------------------------------------------

@pytest.mark.parametrize("shell, l", [("s", 0), ("p", 1), ("d", 2)])
def test_angular_momentum_algebra(shell, l):
    lx, ly, lz = L = angular_momentum(shell)
    assert L.shape == (3, 2 * l + 1, 2 * l + 1)
    np.testing.assert_allclose(L, np.conj(np.swapaxes(L, 1, 2)), atol=1e-15)
    for a, b, c in ((lx, ly, lz), (ly, lz, lx), (lz, lx, ly)):
        np.testing.assert_allclose(a @ b - b @ a, 1j * c, atol=1e-14)
    np.testing.assert_allclose(lx @ lx + ly @ ly + lz @ lz, l * (l + 1) * np.eye(2 * l + 1),
                               atol=1e-14)
    # real orbitals: imaginary matrices, no orbital moment
    np.testing.assert_allclose(L.real, 0.0, atol=1e-15)


def test_angular_momentum_rotates_the_slater_koster_blocks():
    # exp(-i phi n.L) is the rotation of the orbitals, so it must carry the
    # hopping block of a bond u into that of the rotated bond R u.  This ties
    # the matrices to the orbitals and their order in hamiltonian.py.
    basis = ("p", "d")
    rotvec = np.array([0.4, -0.7, 0.5])
    rot = Rotation.from_rotvec(rotvec).as_matrix()
    turn = np.zeros((8, 8))
    for sh, (a, b) in zip(basis, shell_slices(basis)[0]):
        d = expm(-1j * np.einsum("k,kab->ab", rotvec, angular_momentum(sh)))
        np.testing.assert_allclose(d.imag, 0.0, atol=1e-14)
        turn[a:b, a:b] = d.real

    def block(u):
        idx = np.zeros((1, 1), dtype=int)
        return bond_blocks(jnp.array(u).reshape(1, 1, 3), np.ones((1, 1)), idx, idx,
                           PD_BOND, basis, 5.0, 0.0)[0, 0]

    u = np.array([1.3, 0.8, -1.7])
    np.testing.assert_allclose(block(rot @ u), turn @ block(u) @ turn.T, atol=1e-13)


def test_spin_orbit_levels():
    # xi L.S = xi/2 [j(j+1) - l(l+1) - 3/4]: l/2 for j = l + 1/2 (2l + 2 states)
    # and -(l+1)/2 for j = l - 1/2 (2l states)
    ls = bop.spin_orbit_blocks(("s", "p", "d"))
    for k, l in enumerate((0, 1, 2)):
        ev = np.linalg.eigvalsh(ls[k].reshape(18, 18))
        expected = [-(l + 1) / 2] * (2 * l) + [0.0] * (18 - 2 * (2 * l + 1)) + [l / 2] * (2 * l + 2)
        np.testing.assert_allclose(ev, sorted(expected), atol=1e-13)


# --- moments ----------------------------------------------------------------------

TRIMER = np.array([[0.0, 0.0, 0.0], [2.3, 0.2, 0.1], [0.9, 2.1, -0.4]])
TRIMER_AXES = np.array([[0.3, 0.2, 1.0], [1.0, -0.5, 0.2], [-0.4, 1.0, 0.6]])


def test_moments_are_traces_of_the_spinor_matrix():
    basis, nmom, n = ("p", "d"), 7, 3
    slices, norb = shell_slices(basis)
    xi = np.array([0.4, 0.15])
    cfg = bj.BOPConfig(nmom=nmom, rcut=4.0, basis=basis, spin="noncollinear",
                       spin_orbit=True, bandwidth="averaged")
    topo = bj.build_topology(TRIMER, [0] * n, cfg.rcut, nmom, None)
    params = {"onsite": jnp.array([[0.8, -0.3]]), "n0": jnp.array([[1.0, 6.0]]),
              "hubbard": jnp.array([3.0]), "stoner": jnp.array([[0.3, 0.9]]),
              "bond": PD_BOND, "spin_axes": jnp.array(TRIMER_AXES),
              "spin_orbit": jnp.array([xi])}
    shifts = np.random.default_rng(0).normal(scale=0.7, size=(n, 2, 2))
    levels = np.array(params["onsite"])[0] + shifts  # (atom, spin, shell)

    # the spinor matrix, index (atom, orbital, spin)
    idx = np.zeros_like(topo.nbr_origin)
    hblk = bond_blocks(bond_vectors(jnp.array(TRIMER), None, topo), topo.nbr_mask, idx, idx,
                       PD_BOND, basis, cfg.rcut, cfg.dcut)
    e = TRIMER_AXES / np.linalg.norm(TRIMER_AXES, axis=1, keepdims=True)
    e_sigma = np.einsum("nx,xtu->ntu", e, bop.PAULI)
    proj = np.stack([0.5 * (np.eye(2) + e_sigma), 0.5 * (np.eye(2) - e_sigma)], axis=1)
    ham = np.zeros((n, norb, 2, n, norb, 2), dtype=complex)
    for i in range(n):
        for k in np.flatnonzero(topo.nbr_mask[i]):
            ham[i, :, :, topo.nbr_origin[i, k]] += np.einsum("ab,tu->atbu", hblk[i, k], np.eye(2))
        for k, (sh, (a, b)) in enumerate(zip(basis, slices)):
            exchange = levels[i, 0, k] * proj[i, 0] + levels[i, 1, k] * proj[i, 1]
            ls = 0.5 * np.einsum("xab,xtu->atbu", angular_momentum(sh), bop.PAULI)
            ham[i, a:b, :, i, a:b, :] += np.einsum("ab,tu->atbu", np.eye(b - a), exchange) + xi[k] * ls
    ham = ham.reshape(2 * n * norb, 2 * n * norb)
    np.testing.assert_allclose(ham, ham.conj().T, atol=1e-15)

    # <i a, s along e_i| H^k |i a, s along e_i>
    reference = np.zeros((n, 2, norb, nmom + 1))
    power = np.eye(len(ham), dtype=complex)
    for k in range(nmom + 1):
        diag = np.einsum("iatiau->iatu", power.reshape(n, norb, 2, n, norb, 2))
        reference[..., k] = np.einsum("isut,iatu->isa", proj, diag).real
        power = power @ ham

    orb_levels = jnp.array(np.concatenate(
        [np.repeat(levels[:, :, k:k + 1], b - a, axis=2) for k, (a, b) in enumerate(slices)],
        axis=2))
    so = jnp.einsum("k,katbu->atbu", jnp.array(xi), bop.spin_orbit_blocks(basis))
    mu = bop.moments_noncollinear(hblk, orb_levels, params["spin_axes"], topo, nmom,
                                  spin_orbit=jnp.broadcast_to(so, (n,) + so.shape))
    np.testing.assert_allclose(mu, reference, rtol=1e-12, atol=1e-11)
    # spin-orbit coupling enters at the second moment, and couples the orbitals
    plain = bop.moments_noncollinear(hblk, orb_levels, params["spin_axes"], topo, nmom)
    np.testing.assert_allclose(mu[..., :2], plain[..., :2], atol=1e-13)
    assert float(jnp.min(jnp.abs(mu[..., 2] - plain[..., 2]))) > 1e-3

    # the energy function averages them over the orbitals of each shell
    _, aux = bj.make_energy_fn(topo, cfg)(jnp.array(shifts), jnp.array(TRIMER), None, params)
    for k, (a, b) in enumerate(slices):
        np.testing.assert_allclose(aux["moments"][:, :, k], reference[:, :, a:b].mean(axis=2),
                                   rtol=1e-12, atol=1e-11)


def test_zero_coupling_and_settings():
    params = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[0.6]]),
                  spin_axes=jnp.array(TRIMER_AXES), spin_orbit=jnp.zeros((1, 1)))
    shifts = jnp.array(np.random.default_rng(1).normal(scale=0.5, size=(3, 2, 1)))
    out = []
    for flag in (False, True):
        cfg = bj.BOPConfig(nmom=6, rcut=4.0, spin="noncollinear", spin_orbit=flag)
        topo = bj.build_topology(TRIMER, [0] * 3, cfg.rcut, cfg.nmom, None)
        out.append(float(bj.make_energy_fn(topo, cfg)(shifts, jnp.array(TRIMER), None, params)[0]))
    assert out[1] == pytest.approx(out[0], abs=1e-12)
    with pytest.raises(ValueError, match="noncollinear"):
        bj.make_energy_fn(topo, bj.BOPConfig(nmom=6, rcut=4.0, spin="collinear",
                                             spin_orbit=True))


# --- magnetic anisotropy of a linear chain ------------------------------------------

CHAIN_STONER = 0.6


@pytest.fixture(scope="module")
def chain():
    """Self-consistent energy per atom of a ferromagnetic chain along z as a
    function of the common spin direction and of xi."""
    cfg = bj.BOPConfig(nmom=9, rcut=3.3, dcut=0.5, bandwidth="averaged", spin="noncollinear",
                       spin_orbit=True)
    pos, cell = jnp.zeros((1, 3)), jnp.diag(jnp.array([12.0, 12.0, 2.2]))
    topo = bj.build_topology(pos, [0], cfg.rcut, cfg.nmom, cell)
    scf = bj.make_scf(bj.make_energy_fn(topo, cfg))
    base = dict(canonical_d(n_electrons=7.0, hubbard=3.0), stoner=jnp.array([[CHAIN_STONER]]))
    start = bj.stoner_shifts(np.full((1, 1), 2.0), CHAIN_STONER)

    def solve(axis, xi):
        params = dict(base, spin_axes=jnp.array([axis], dtype=float),
                      spin_orbit=jnp.array([[xi]]))
        res = scf(pos, cell, params, shifts0=start, tol=1e-12, max_iter=40)
        assert res.converged
        return res

    return solve


ALONG, ACROSS = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]


def test_chain_without_coupling_is_isotropic(chain):
    along = chain(ALONG, 0.0)
    assert 1.0 < float(along.aux["moment"][0]) < 2.0  # not saturated
    assert chain(ACROSS, 0.0).energy == pytest.approx(along.energy, abs=1e-11)
    assert chain([0.3, -0.5, 0.8], 0.0).energy == pytest.approx(along.energy, abs=1e-11)


def test_chain_anisotropy_is_uniaxial_and_of_second_order(chain):
    barrier = {}
    for xi in (0.0125, 0.025, 0.05):
        along = chain(ALONG, xi).energy
        barrier[xi] = chain(ACROSS, xi).energy - along
        # E = K sin^2(theta): half the barrier at 45 degrees
        assert chain([1.0, 0.0, 1.0], xi).energy - along == pytest.approx(
            0.5 * barrier[xi], rel=5e-3)
        # and no dependence on the direction in the plane normal to the chain
        assert chain([0.3, 1.0, 0.0], xi).energy - along == pytest.approx(barrier[xi], abs=1e-11)
    # easy axis along the chain, 0.29 meV per atom at xi = 0.05 eV
    assert barrier[0.05] == pytest.approx(2.894e-4, rel=2e-3)
    # K / xi^2 = 0.114, 0.115, 0.116 eV^-1: second order with small corrections
    assert barrier[0.05] / barrier[0.025] == pytest.approx(4.0, rel=0.02)
    assert barrier[0.025] / barrier[0.0125] == pytest.approx(4.0, rel=0.02)


# --- torques ----------------------------------------------------------------------

TRIMER_STONER = 0.6


@pytest.fixture(scope="module")
def trimer():
    cfg = bj.BOPConfig(nmom=7, rcut=4.0, bandwidth="averaged", spin="noncollinear",
                       spin_orbit=True)
    topo = bj.build_topology(TRIMER, [0] * 3, cfg.rcut, cfg.nmom, None)
    energy = bj.make_energy_fn(topo, cfg)
    scf = bj.make_scf(energy)
    pos, axes = jnp.array(TRIMER), jnp.array(TRIMER_AXES)
    base = dict(canonical_d(n_electrons=7.0, hubbard=3.0),
                stoner=jnp.array([[TRIMER_STONER]]), spin_axes=axes)
    start = bj.stoner_shifts(np.full((3, 1), 2.0), TRIMER_STONER)
    gradients = jax.jit(jax.grad(
        lambda ax, p, shifts, params: energy(shifts, p, None, dict(params, spin_axes=ax))[0],
        argnums=(0, 1)))

    def solve(xi, axes=axes, shifts0=start):
        params = dict(base, spin_axes=axes, spin_orbit=jnp.array([[xi]]))
        res = scf(pos, None, params, shifts0=shifts0, tol=1e-12, max_iter=40)
        assert res.converged
        return res, gradients(axes, pos, res.shifts, params)

    return solve


def test_torques_no_longer_sum_to_zero(trimer):
    _, (g_axes, _) = trimer(0.0)
    np.testing.assert_allclose(jnp.sum(jnp.cross(TRIMER_AXES, g_axes), axis=0), 0.0, atol=1e-12)
    _, (g_axes, _) = trimer(0.1)
    total = jnp.sum(jnp.cross(TRIMER_AXES, g_axes), axis=0)
    assert float(jnp.linalg.norm(total)) > 3e-3
    # the energy still does not depend on the length of an axis
    np.testing.assert_allclose(jnp.sum(g_axes * TRIMER_AXES, axis=1), 0.0, atol=1e-12)


def test_torque_is_total_derivative(trimer):
    res, (g_axes, _) = trimer(0.1)
    h = 1e-3
    for atom, comp in ((0, 0), (1, 2), (2, 1)):
        def e_scf(delta):
            axes = jnp.array(TRIMER_AXES).at[atom, comp].add(delta)
            return trimer(0.1, axes, res.shifts)[0].energy

        fd = (-e_scf(2 * h) + 8 * e_scf(h) - 8 * e_scf(-h) + e_scf(-2 * h)) / (12 * h)
        assert float(g_axes[atom, comp]) == pytest.approx(fd, abs=1e-8)


def test_spin_torque_is_balanced_by_the_mechanical_torque(trimer):
    # Turning positions and axes together changes nothing:
    #   sum_i r_i x dE/dr_i + sum_i e_i x dE/de_i = 0,
    # so the torque -sum e x dE/de on the spins is minus the torque sum r x F
    # on the lattice.  Without spin-orbit coupling both vanish separately.
    for xi in (0.0, 0.1):
        _, (g_axes, g_pos) = trimer(xi)
        spin_part = jnp.sum(jnp.cross(TRIMER_AXES, g_axes), axis=0)
        lattice_part = jnp.sum(jnp.cross(TRIMER, g_pos), axis=0)
        np.testing.assert_allclose(spin_part + lattice_part, 0.0, atol=1e-11)
        np.testing.assert_allclose(jnp.sum(g_pos, axis=0), 0.0, atol=1e-12)
        if xi > 0.0:
            assert float(jnp.linalg.norm(lattice_part)) > 3e-3
        else:
            np.testing.assert_allclose(lattice_part, 0.0, atol=1e-11)
