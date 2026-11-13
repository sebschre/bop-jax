"""Internal consistency: Slater-Koster blocks, moments, derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.geometry import bond_vectors
from bopjax.hamiltonian import _block_upper, bond_blocks, shell_slices


def spd_params(nsp=2, seed=0):
    rng = np.random.default_rng(seed)
    channels = ["sss", "sps", "sds", "pps", "ppp", "pds", "pdp", "dds", "ddp", "ddd"]
    bond = {}
    for c in channels:
        pre = rng.uniform(-1.0, 1.0, (nsp, nsp))
        if c[0] == c[1]:
            pre = 0.5 * (pre + pre.T)  # same-shell channels are symmetric in species
        par = np.stack([pre, np.full((nsp, nsp), 2.0), np.full((nsp, nsp), 3.0)], -1)
        bond[c] = jnp.array(par)
    return {
        "onsite": jnp.array([[-1.0, 0.5, 0.2], [-0.6, 0.9, -0.1]])[:nsp],
        "n0": jnp.array([[1.0, 1.5, 3.0], [1.2, 2.0, 4.0]])[:nsp],
        "hubbard": jnp.array([4.0, 6.0])[:nsp],
        "bond": bond,
    }


CLUSTER = np.array(
    [[0.0, 0.0, 0.0], [2.1, 0.2, -0.1], [0.3, 2.2, 0.4], [1.1, 1.0, 2.0], [-1.5, 0.8, 1.2]]
)
SPECIES = [0, 1, 0, 1, 1]
SPD = bj.BOPConfig(nmom=4, rcut=4.0, dcut=0.5, basis=("s", "p", "d"))


def test_slater_koster_table_entries():
    l, m, n = u = jnp.array([0.36, 0.48, 0.8])
    v = {"pds": 0.7, "pdp": -0.3, "dds": -1.1, "ddp": 0.6, "ddd": -0.2, "sds": 0.4}
    pd = _block_upper("p", "d", u, v)
    dd = _block_upper("d", "d", u, v)
    sd = _block_upper("s", "d", u, v)
    s3 = np.sqrt(3.0)
    # d order: xy, yz, zx, x2-y2, 3z2-r2
    assert pd[0, 0] == pytest.approx(s3 * l * l * m * 0.7 + m * (1 - 2 * l * l) * -0.3)
    assert pd[2, 4] == pytest.approx(
        n * (n * n - 0.5 * (l * l + m * m)) * 0.7 + s3 * n * (l * l + m * m) * -0.3
    )
    assert dd[0, 0] == pytest.approx(
        3 * l * l * m * m * -1.1
        + (l * l + m * m - 4 * l * l * m * m) * 0.6
        + (n * n + l * l * m * m) * -0.2
    )
    assert dd[0, 1] == pytest.approx(
        3 * l * m * m * n * -1.1 + l * n * (1 - 4 * m * m) * 0.6 + l * n * (m * m - 1) * -0.2
    )
    assert sd[0, 4] == pytest.approx((n * n - 0.5 * (l * l + m * m)) * 0.4)


def dense_hamiltonian(pos, topo, cfg, params):
    rij = bond_vectors(jnp.array(pos), None, topo)
    sp = topo.species
    blk = bond_blocks(rij, topo.nbr_mask, np.broadcast_to(sp[:, None], topo.nbr_origin.shape),
                      sp[topo.nbr_origin], params["bond"], cfg.basis, cfg.rcut, cfg.dcut)
    slices, norb = shell_slices(cfg.basis)
    n = topo.n_atoms
    h = np.zeros((n * norb, n * norb))
    for i in range(n):
        for k in range(topo.nbr_mask.shape[1]):
            if topo.nbr_mask[i, k]:
                j = topo.nbr_origin[i, k]
                h[i * norb:(i + 1) * norb, j * norb:(j + 1) * norb] = blk[i, k]
        for (a, b), e in zip(slices, params["onsite"][sp[i]]):
            idx = np.arange(i * norb + a, i * norb + b)
            h[idx, idx] = e
    return h


def test_hamiltonian_is_symmetric_and_moments_match_matrix_powers():
    params = spd_params()
    topo = bj.build_topology(CLUSTER, SPECIES, SPD.rcut, SPD.nmom)
    h = dense_hamiltonian(CLUSTER, topo, SPD, params)
    np.testing.assert_allclose(h, h.T, atol=1e-13)

    energy = bj.make_energy_fn(topo, SPD)
    _, aux = energy(jnp.zeros((5, 3)), jnp.array(CLUSTER), None, params)
    slices, norb = shell_slices(SPD.basis)
    for n in range(SPD.nmom + 1):
        diag = np.diag(np.linalg.matrix_power(h, n)).reshape(5, norb)
        expected = np.stack([diag[:, a:b].mean(axis=1) for a, b in slices], axis=1)
        np.testing.assert_allclose(aux["moments"][:, :, n], expected, atol=1e-11)


def test_energy_is_rotation_invariant():
    params = spd_params()
    topo = bj.build_topology(CLUSTER, SPECIES, SPD.rcut, SPD.nmom)
    energy = bj.make_energy_fn(topo, SPD)
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    shifts = jnp.array(np.random.default_rng(2).normal(scale=0.05, size=(5, 3)))
    e1, _ = energy(shifts, jnp.array(CLUSTER), None, params)
    e2, _ = energy(shifts, jnp.array(CLUSTER @ q.T), None, params)
    assert float(e1) == pytest.approx(float(e2), abs=1e-11)


def test_gradient_and_hessian_in_shifts_match_finite_differences():
    params = spd_params()
    topo = bj.build_topology(CLUSTER, SPECIES, SPD.rcut, SPD.nmom)
    energy = bj.make_energy_fn(topo, SPD)
    pos = jnp.array(CLUSTER)

    def e(x):
        return energy(x.reshape(5, 3), pos, None, params)[0]

    x0 = jnp.array(np.random.default_rng(3).normal(scale=0.05, size=15))
    grad = jax.jit(jax.grad(e))
    hess = jax.jacfwd(grad)(x0)
    h = 1e-5
    eye = jnp.eye(15)
    fd_hess = np.stack([(grad(x0 + h * eye[k]) - grad(x0 - h * eye[k])) / (2 * h)
                        for k in range(15)])
    np.testing.assert_allclose(hess, fd_hess, atol=1e-7)
    np.testing.assert_allclose(hess, hess.T, atol=1e-10)
    # a uniform shift of all levels is a null direction
    np.testing.assert_allclose(grad(x0).sum(), 0.0, atol=1e-10)
    np.testing.assert_allclose(hess @ np.ones(15), 0.0, atol=1e-9)
