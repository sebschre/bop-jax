"""Per-species shells (BOPConfig.species) and the iron-carbon model."""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax.hamiltonian import v0_bond
from bopjax.models import (CARBON_CONFIG, IRON_CARBON_CONFIG, IRON_CONFIG, carbon, iron,
                           iron_carbon, tapered_exp)

# The Fe-C cross parameters are not fitted (see docs/iron-carbon.md); these
# tests check the machinery: per-species shells, shifts and spin, and that
# the elemental models are reproduced.

FCC = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])


def scf_energy(pos, cell, species, cfg, params, radial, fe_moment=None, shells=(0, 1)):
    """``shells``: where the iron s and d shells sit in the basis."""
    topo = bj.build_topology(pos, species, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=radial)
    start = None
    if fe_moment is not None:
        m = np.zeros(energy.shift_shape[::2])
        m[:, shells[1]] = fe_moment
        start = bj.stoner_shifts(m, float(np.max(params["stoner"])))
    res = bj.make_scf(energy)(jnp.asarray(pos), jnp.asarray(cell), params, shifts0=start,
                              tol=1e-11, max_iter=60)
    assert res.converged
    return res.energy, res.aux


def test_pure_iron_is_the_sd_iron_model():
    a = 2.855
    cell, pos = a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    e_ref, aux_ref = scf_energy(pos, cell, [0, 0],
                                bj.BOPConfig(spin="collinear", **dict(IRON_CONFIG,
                                                                      basis=("s", "d"))),
                                iron(basis=("s", "d")), v0_bond, 2.5)
    e, aux = scf_energy(pos, cell, [0, 0],
                        bj.BOPConfig(spin="collinear", **IRON_CARBON_CONFIG),
                        iron_carbon(), tapered_exp, 2.5, shells=(0, 2))
    assert e == pytest.approx(e_ref, abs=1e-9)
    assert np.allclose(aux["moment"], aux_ref["moment"], atol=1e-8)


def test_pure_carbon_is_the_carbon_model():
    a = 3.5651
    cell, pos = a * FCC, a * np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    e_ref, aux_ref = scf_energy(pos, cell, [0, 0], bj.BOPConfig(**CARBON_CONFIG), carbon(),
                                v0_bond)
    e, aux = scf_energy(pos, cell, [1, 1], bj.BOPConfig(**IRON_CARBON_CONFIG), iron_carbon(),
                        tapered_exp)
    assert e == pytest.approx(e_ref, abs=1e-9)
    assert np.allclose(aux["n_shell"][:, :2], aux_ref["n_shell"], atol=1e-8)
    assert np.allclose(aux["n_shell"][:, 2], 0.0)  # carbon has no d shell


def test_species_atomic_shifts_and_spin_polarization():
    """Carbon in the Fe-C model: one shift for its shells and spins, so the
    energy does not depend on how a shift is split between them."""
    a = 4.3
    cell, pos = a * FCC, np.array([[0, 0, 0], [0.5 * a, 0, 0]])
    cfg = bj.BOPConfig(spin="collinear", **IRON_CARBON_CONFIG)
    topo = bj.build_topology(pos, [0, 1], cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=tapered_exp)
    x = np.zeros(energy.shift_shape)
    x[0, 0, 2], x[0, 1, 2] = -1.0, 1.0  # iron d split
    x[1, :, :2] = 0.3
    y = x.copy()
    y[1, 0, 0], y[1, 0, 1], y[1, 1, 0], y[1, 1, 1] = 0.9, -0.3, 0.5, 0.1  # same mean
    p = iron_carbon()
    e1 = energy(jnp.asarray(x), jnp.asarray(pos), jnp.asarray(cell), p)[0]
    e2 = energy(jnp.asarray(y), jnp.asarray(pos), jnp.asarray(cell), p)[0]
    assert float(e1) == pytest.approx(float(e2), abs=1e-10)


def test_absent_shells_ignore_their_parameters():
    """Bond integrals, levels and n0 of shells a species lacks change nothing."""
    a = 4.3  # rock-salt FeC
    cell = a * FCC
    pos = np.array([[0, 0, 0], [0.5 * a, 0, 0]])
    cfg = bj.BOPConfig(**IRON_CARBON_CONFIG)
    p = iron_carbon()
    e, _ = scf_energy(pos, cell, [0, 1], cfg, p, tapered_exp)
    q = dict(p, bond=dict(p["bond"]), onsite=p["onsite"].at[0, 1].set(-3.0).at[1, 2].set(-2.0),
             n0=p["n0"].at[0, 1].set(1.0))
    junk = jnp.array([1.3, 1.0, 2.0, 3.0, 0.5])
    q["bond"]["sds"] = q["bond"]["sds"].at[0, 1].set(junk)  # Fe s - C d
    q["bond"]["pps"] = q["bond"]["pps"].at[0, 1].set(junk)  # Fe p - C p
    q["bond"]["dds"] = q["bond"]["dds"].at[1, 1].set(junk)  # C d - C d
    e2, _ = scf_energy(pos, cell, [0, 1], cfg, q, tapered_exp)
    assert e2 == pytest.approx(e, abs=1e-9)


def test_fe_c_forces_match_finite_differences():
    """Rock-salt FeC with carbon off its site, non-magnetic (keeps it quick)."""
    a = 4.0
    cell = a * FCC
    pos = np.array([[0, 0, 0], [0.5 * a + 0.12, 0.05, -0.08]])
    cfg = bj.BOPConfig(**IRON_CARBON_CONFIG)
    topo = bj.build_topology(pos, [0, 1], cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2, skin=0.1)
    pot = bj.make_potential(bj.make_energy_fn(topo, cfg, radial=tapered_exp), tol=1e-11)
    p = iron_carbon()
    import jax
    f = -jax.grad(lambda q: pot(q, jnp.asarray(cell), p)[0])(jnp.asarray(pos))
    h = 1e-4
    for i, k in [(1, 0), (1, 2)]:
        dp = np.zeros_like(pos)
        dp[i, k] = h
        num = -(float(pot(jnp.asarray(pos + dp), jnp.asarray(cell), p)[0])
                - float(pot(jnp.asarray(pos - dp), jnp.asarray(cell), p)[0])) / (2 * h)
        assert float(f[i, k]) == pytest.approx(num, abs=1e-5)
