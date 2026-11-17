"""Trajectories for OVITO: extended XYZ output and, if the ovito module is
installed, the data pipeline (rendering needs a graphics driver and is not
tested)."""

from typing import NamedTuple

import numpy as np
import pytest

from bopjax import ovito as bov


class State(NamedTuple):  # the fields of spinlattice.SpinLatticeState that are used
    position: np.ndarray
    momentum: np.ndarray
    force: np.ndarray
    mass: np.ndarray
    axes: np.ndarray
    moment: np.ndarray
    energy: float


def trajectory(frames=3):
    rng = np.random.default_rng(0)
    traj = bov.Trajectory(2.8 * np.diag([1.0, 1.0, 2.0]), species=[0, 0, 1, 1],
                          names=("Fe", "Co"))
    for k in range(frames):
        axes = rng.normal(size=(4, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        state = State(rng.normal(size=(4, 3)), rng.normal(size=(4, 3)), rng.normal(size=(4, 3)),
                      np.full((4, 1), 55.85), axes, rng.uniform(1, 3, 4), -13.0 + k)
        traj.append(state, time=0.5 * k, orbital_moment=rng.normal(size=(4, 3)),
                    charge=rng.normal(size=4))
    return traj


def test_append_takes_what_a_state_carries():
    traj = trajectory()
    positions, per_atom, attributes = traj.frames[1]
    assert set(per_atom) == {"spin_moment", "force", "velocity", "orbital_moment", "charge"}
    assert attributes == {"energy": -12.0, "time": 0.5}
    assert positions.shape == (4, 3) and per_atom["charge"].shape == (4,)
    with pytest.raises(ValueError, match="same per-atom"):
        traj.append(positions=np.zeros((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        traj.append(positions=np.zeros((4, 3)), bad=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positions"):
        bov.Trajectory().append(time=1.0)


def test_extended_xyz_round_trip(tmp_path):
    traj = trajectory()
    path = traj.write(tmp_path / "run.xyz")
    lines = path.read_text().splitlines()
    assert lines[0] == "4"
    assert 'Lattice="2.8 0 0 0 2.8 0 0 0 5.6"' in lines[1]
    assert "spin_moment:R:3" in lines[1] and "charge:R:1" in lines[1]
    assert lines[2].startswith("Fe ") and lines[4].startswith("Co ")
    back = bov.Trajectory.read(path)
    assert len(back) == len(traj) and back.names == ("Fe", "Co")
    np.testing.assert_array_equal(back.species, traj.species)
    np.testing.assert_allclose(back.cell, traj.cell)
    for (p, a, g), (q, b, h) in zip(traj.frames, back.frames):
        np.testing.assert_allclose(q, p, atol=1e-8)
        assert set(a) == set(b) and g == pytest.approx(h)
        for name in a:
            np.testing.assert_allclose(b[name], a[name], atol=1e-8)


def test_pipeline():
    pytest.importorskip("ovito")
    traj = trajectory()
    pipeline = traj.pipeline(arrows={"spin_moment": bov.SPIN, "orbital_moment": bov.ORBITAL},
                             exaggerate=2.0, replicate=(1, 1, 2))
    assert pipeline.num_frames == 3
    data = pipeline.compute(2)
    assert data.particles.count == 8
    assert data.attributes["energy"] == pytest.approx(-11.0)
    spins = np.asarray(data.particles["spin_moment"])[:4]
    np.testing.assert_allclose(spins, traj.frames[2][1]["spin_moment"])
    assert data.particles["spin_moment"].vis.enabled
    # displacements from the first frame are exaggerated
    expected = traj.frames[0][0] + 2.0 * (traj.frames[2][0] - traj.frames[0][0])
    np.testing.assert_allclose(np.asarray(data.particles.positions)[:4], expected)
    with pytest.raises(ValueError, match="no per-atom vector"):
        traj.pipeline(arrows={"charge": bov.SPIN})
