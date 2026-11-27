"""NEB (bopfox_rs.neb) and the first-order field effect on barriers
(bopfox_rs.zeeman).

    uv run pytest bopfox-rs/tests/test_neb.py
"""

import numpy as np
import pytest

bopfox_rs = pytest.importorskip("bopfox_rs")

import bopjax as bj  # noqa: E402
from bopjax import hamiltonian  # noqa: E402
from bopjax.models import IRON_CONFIG, iron  # noqa: E402
from bopfox_rs import neb, zeeman  # noqa: E402


def muller_brown(index, x):
    """Two-dimensional Mueller-Brown surface for one atom (z harmonic)."""
    A = [-200, -100, -170, 15]
    a, b, c = [-1, -1, -6.5, 0.7], [0, 0, 11, 0.6], [-10, -10, -6.5, 0.7]
    x0, y0 = [1, 0, -0.5, -1], [0, 0.5, 1.5, 1]
    px, py, pz = x[0]
    e, fx, fy = 0.0, 0.0, 0.0
    for k in range(4):
        dx, dy = px - x0[k], py - y0[k]
        t = A[k] * np.exp(a[k] * dx**2 + b[k] * dx * dy + c[k] * dy**2)
        e += t
        fx -= t * (2 * a[k] * dx + b[k] * dy)
        fy -= t * (b[k] * dx + 2 * c[k] * dy)
    scale = 0.01  # into eV-like units
    return scale * (e + 10 * pz**2), scale * np.array([[fx, fy, -20 * pz]])


def test_same_band_as_bopjax():
    from bopjax import neb as bneb
    start, end = np.array([[-0.558, 1.442, 0.0]]), np.array([[0.623, 0.028, 0.0]])
    band = np.array([start + t * (end - start) for t in np.linspace(0, 1, 9)])
    band[1:-1, 0, 2] = 0.05  # off the plane: the z force pulls back
    kw = dict(spring=1.0, fmax=1e-3, steps=3000, dt=0.01, dt_max=0.05, max_step=0.05)
    ours = neb.neb(muller_brown, band, **kw)
    ref = bneb.neb(muller_brown, band, **kw)
    assert ours.converged and ref.converged
    np.testing.assert_array_equal(ours.images, ref.images)
    np.testing.assert_array_equal(ours.energies, ref.energies)
    assert ours.climbing == ref.climbing
    # the known saddle of the Mueller-Brown surface between minima A and B
    saddle = ours.images[ours.climbing][0]
    assert saddle[:2] == pytest.approx([-0.822, 0.624], abs=2e-3)


def test_peierls_stress_matches_bopjax():
    from bopjax import zeeman as bz
    x = np.linspace(0, 2.5, 7)
    e = 0.03 * np.sin(np.pi * x / 2.5) ** 2
    assert zeeman.peierls_stress(x, e, 2.48, 10.0) == bz.peierls_stress(x, e, 2.48, 10.0)


def test_vacancy_hop_and_the_field_derivative_of_its_barrier():
    """Climbing-image NEB of a vacancy hop in 2x2x2 bcc iron on the
    self-consistent forces of the iron model (non-magnetic, potential
    condition), with only the hopping atom free; then the first-order shift
    of the barrier in a field from the moments of the magnetic model at the
    end point and the saddle, against the barrier recomputed in +-5 T.

    Only the hopping atom moves because this 15-atom cell is too small for
    the symmetric vacancy to be a minimum: with every atom free the band
    finds distorted structures 0.75 eV below the relaxed (stationary but
    unstable) end points.  The potential condition because along this hop
    the stationary condition is out of reach at some images (charge
    transfers near 0.5 e); bopjax's Newton solver does not converge them
    either."""
    a = 2.83
    params = iron()
    model = bopfox_rs.model_from_bopjax(bj.BOPConfig(spin="none", **IRON_CONFIG), params,
                                        hamiltonian.v0_bond)
    magnetic = bopfox_rs.model_from_bopjax(bj.BOPConfig(spin="collinear", **IRON_CONFIG), params,
                                           hamiltonian.v0_bond)
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], float)
    sites = a * np.concatenate([grid, grid + 0.5])
    cell = 2 * a * np.eye(3)
    vac, hop = 0, 8  # the atom on site 8 moves into the vacancy at site 0
    keep = [s for s in range(16) if s != vac]
    start = sites[keep]
    end = start.copy()
    end[keep.index(hop)] = sites[vac]
    n = len(start)
    species = [0] * n
    mobile = np.arange(n) == keep.index(hop)
    p_images = 5
    force_fn = neb.image_forces(model, species, cell, np.zeros((n, 1, 1)), p_images, tol=1e-9,
                                condition="potential")
    band = np.array([start + t * (end - start) for t in np.linspace(0, 1, p_images)])
    res = neb.neb(force_fn, band, mobile=mobile, spring=0.5, fmax=0.01, steps=400,
                  max_step=0.05)
    assert res.converged
    assert res.energies[0] == pytest.approx(res.energies[-1], abs=1e-8)  # equivalent sites
    barrier = res.energies[res.climbing] - res.energies[0]
    assert 0.1 < barrier < 5.0
    assert res.climbing == p_images // 2  # the middle image of the symmetric path
    # the saddle is the midpoint of the hop, where the force on the atom vanishes
    np.testing.assert_allclose(res.images[res.climbing][mobile][0], 0.5 * (sites[hop] + sites[vac]),
                               atol=0.02)

    pot = bopfox_rs.Potential(magnetic)
    s0 = np.zeros((n, 2, 1))
    s0[:, 0], s0[:, 1] = -1.3, 1.3
    images = [res.images[0], res.images[res.climbing]]

    def energies_moments(field):
        e, m = [], []
        for x in images:
            out = pot.scf(x, species, cell, s0, tol=1e-11, magnetic_field=field,
                          condition="potential")
            assert out["converged"]
            e.append(out["energy"])
            m.append(out["magnetic_moment"].sum(axis=0))
        return np.array(e), np.array(m)

    e, m = energies_moments(np.zeros(3))
    fo = zeeman.first_order_barrier(e, m)
    assert fo["saddle"] == 1
    b = np.array([0.0, 0.0, 5.0])
    e_plus, _ = energies_moments(b)
    e_minus, _ = energies_moments(-b)
    fd = ((e_plus[1] - e_plus[0]) - (e_minus[1] - e_minus[0])) / 2.0
    assert fd == pytest.approx(float(fo["gradient"] @ b), rel=1e-3, abs=1e-8)
    np.testing.assert_allclose(zeeman.first_order_energies(m, b), [0.0, float(fo["gradient"] @ b)])


@pytest.mark.parametrize("spin", ["collinear", "noncollinear"])
@pytest.mark.parametrize("condition", ["stationary", "potential"])
def test_moments_without_a_field(spin, condition):
    """The moments at zero field do not need a field to be passed: the
    explicit Zeeman term (h.e) m contributes m e to dE/dh at h = 0 as well."""
    cfg = bj.BOPConfig(spin=spin, **IRON_CONFIG)
    pot = bopfox_rs.Potential(bopfox_rs.model_from_bopjax(cfg, iron(), hamiltonian.v0_bond))
    a = 2.83
    pos = a * np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0], [1.5, 0.5, 0.5]])
    pos = pos + np.random.default_rng(5).normal(scale=0.05, size=pos.shape)
    cell = np.diag([2 * a, a, a])
    s0 = np.zeros((4, 2, 1))
    s0[:, 0], s0[:, 1] = -1.3, 1.3
    kw = dict(tol=1e-11, condition=condition)
    none = pot.scf(pos, [0] * 4, cell, s0, **kw)
    zero = pot.scf(pos, [0] * 4, cell, s0, magnetic_field=np.zeros(3), **kw)
    assert none["converged"] and zero["converged"]
    np.testing.assert_allclose(none["magnetic_moment"], zero["magnetic_moment"], atol=1e-9)
    assert none["magnetic_moment"][:, 2].sum() > 4.0  # ferromagnetic, about 2 muB per atom
    direct = pot.compute(pos, [0] * 4, cell, zero["shifts"])
    np.testing.assert_allclose(direct["magnetic_moment"],
                               pot.compute(pos, [0] * 4, cell, zero["shifts"],
                                           magnetic_field=np.zeros(3))["magnetic_moment"],
                               atol=1e-12)
