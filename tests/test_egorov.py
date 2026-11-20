"""The Fe part of the Fe-Co BOP of Egorov et al. (arXiv:2208.12973) against the
bcc Fe values of its Supplemental Material (V 11.45 A^3, 2.88 muB)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bopjax as bj
from bopjax import pairwise as pw
from bopjax.models import IRON_EGOROV_CONFIG, iron_egorov


def test_bcc_iron_of_the_egorov_model():
    a = 2.8395  # zero-pressure lattice constant of the model (V = 11.45 A^3)
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    cfg = bj.BOPConfig(**IRON_EGOROV_CONFIG, spin="collinear")
    topo = bj.build_topology(a * frac, [0, 0], cfg.rcut, cfg.nmom, a * np.eye(3), rcut2=cfg.rcut2)
    energy = bj.make_energy_fn(topo, cfg, radial=pw.exp_sum)
    params = iron_egorov()
    potential = bj.make_potential(energy, tol=1e-11)
    pos, cell = jnp.array(a * frac), jnp.array(a * np.eye(3))
    e, aux = jax.jit(potential)(pos, cell, params, bj.stoner_shifts(np.full((2, 1), 2.6), 0.751))
    assert float(aux["scf_residual"]) < 1e-10
    assert float(e) / 2 == pytest.approx(-8.292, abs=2e-3)  # DFT reference -8.31
    assert float(aux["moment"][0]) == pytest.approx(2.875, abs=2e-3)  # paper 2.88
    # no pressure at this lattice constant
    stress = bj.make_stress(potential)(pos, cell, params, aux["shifts"])
    assert float(jnp.trace(stress)) / 3 * 160.2 == pytest.approx(0.0, abs=0.3)  # GPa
    # the Yukawa term is environment dependent and switched on
    assert float(jnp.max(jnp.abs(aux["energy_rep"]))) > 0.0
