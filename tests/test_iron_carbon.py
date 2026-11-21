"""Per-species shells (BOPConfig.species) and the iron-carbon model."""

import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import pairwise as pw
from bopjax.hamiltonian import v0_bond
from bopjax.models import (CARBON_CONFIG, IRON_CARBON_CONFIG, IRON_EGOROV_CONFIG, carbon,
                           iron_carbon, iron_egorov, tapered_exp_sum)

# Machinery of the iron-carbon model: per-species shells, shifts and spin,
# the elemental models reproduced, forces of the potential self-consistency.
# The fitted Fe-C properties need 17-atom cells and are checked by
# examples/fit_iron_carbon.py (docs/iron-carbon.md).

FCC = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])


def scf_energy(pos, cell, species, cfg, params, radial, fe_moment=None, shells=(0, 1),
               tol=1e-11):
    """``shells``: where the iron s and d shells sit in the basis."""
    topo = bj.build_topology(pos, species, cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=radial)
    start = None
    if fe_moment is not None:
        m = np.zeros(energy.shift_shape[::2])
        m[:, shells[1]] = fe_moment
        start = bj.stoner_shifts(m, float(np.max(params["stoner"])))
    res = bj.make_scf(energy)(jnp.asarray(pos), jnp.asarray(cell), params, shifts0=start,
                              tol=tol, max_iter=60)
    assert res.converged
    return res.energy, res.aux


def test_pure_iron_is_the_egorov_model():
    a = 2.8394
    cell, pos = a * np.eye(3), a * np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    e_ref, aux_ref = scf_energy(pos, cell, [0, 0],
                                bj.BOPConfig(spin="collinear", **IRON_EGOROV_CONFIG),
                                iron_egorov(), pw.exp_sum, 2.8, shells=(0, 0))
    e, aux = scf_energy(pos, cell, [0, 0], bj.BOPConfig(spin="collinear", **IRON_CARBON_CONFIG),
                        iron_carbon(), tapered_exp_sum, 2.8, shells=(0, 2))
    assert e == pytest.approx(e_ref, abs=1e-9)
    assert np.allclose(aux["moment"], aux_ref["moment"], atol=1e-8)


def test_pure_carbon_is_the_carbon_model():
    a = 3.5653
    cell, pos = a * FCC, a * np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    e_ref, aux_ref = scf_energy(pos, cell, [0, 0], bj.BOPConfig(**CARBON_CONFIG), carbon(),
                                v0_bond)
    e, aux = scf_energy(pos, cell, [1, 1], bj.BOPConfig(**IRON_CARBON_CONFIG), iron_carbon(),
                        tapered_exp_sum)
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
    energy = bj.make_energy_fn(topo, cfg, radial=tapered_exp_sum)
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
    # carbon's shell shifts are tied (only their mean enters), so the solver
    # drifts freely along their difference; 1e-8 is as far as it gets
    e, _ = scf_energy(pos, cell, [0, 1], cfg, p, tapered_exp_sum, tol=1e-8)
    q = dict(p, bond=dict(p["bond"]), onsite=p["onsite"].at[0, 1].set(-3.0).at[1, 2].set(-2.0),
             n0=p["n0"].at[0, 1].set(1.0))
    junk = jnp.array([1.3, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.5])
    q["bond"]["sds"] = q["bond"]["sds"].at[0, 1].set(junk)  # Fe s - C d
    q["bond"]["pps"] = q["bond"]["pps"].at[0, 1].set(junk)  # Fe p - C p
    q["bond"]["dds"] = q["bond"]["dds"].at[1, 1].set(junk)  # C d - C d
    e2, _ = scf_energy(pos, cell, [0, 1], cfg, q, tapered_exp_sum, tol=1e-8)
    assert e2 == pytest.approx(e, abs=1e-7)


def test_fe_c_forces_match_finite_differences():
    """Rock-salt FeC with carbon off its site, ferromagnetic, potential
    self-consistency (as the model is used): forces by implicit
    differentiation against finite differences of the energy."""
    import jax

    a = 4.0
    cell = a * FCC
    pos = np.array([[0, 0, 0], [0.5 * a + 0.12, 0.05, -0.08]])
    cfg = bj.BOPConfig(spin="collinear", **IRON_CARBON_CONFIG)
    topo = bj.build_topology(pos, [0, 1], cfg.rcut, cfg.nmom, cell, rcut2=cfg.rcut2, skin=0.1)
    energy = bj.make_energy_fn(topo, cfg, radial=tapered_exp_sum)
    pot = bj.make_potential(energy, tol=1e-12, condition="potential", max_iter=600)
    p = iron_carbon()
    m = np.zeros(energy.shift_shape[::2])
    m[0, 2] = 2.5
    start = bj.stoner_shifts(m, float(p["stoner"][0, 2]))
    e, aux = pot(jnp.asarray(pos), jnp.asarray(cell), p, start)
    assert float(aux["scf_residual"]) < 1e-10
    x0 = aux["shifts"]
    f = -jax.grad(lambda q: pot(q, jnp.asarray(cell), p, x0)[0])(jnp.asarray(pos))
    h = 1e-4
    for i, k in [(1, 0), (1, 2)]:
        dp = np.zeros_like(pos)
        dp[i, k] = h
        num = -(float(pot(jnp.asarray(pos + dp), jnp.asarray(cell), p, x0)[0])
                - float(pot(jnp.asarray(pos - dp), jnp.asarray(cell), p, x0)[0])) / (2 * h)
        assert float(f[i, k]) == pytest.approx(num, abs=1e-5)
