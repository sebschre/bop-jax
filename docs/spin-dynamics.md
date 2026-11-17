# Spin dynamics

In a non-collinear calculation the spin axes `e_i` play the role positions
play in molecular dynamics: the self-consistent energy is a function of them,
and its gradient gives the torque `-e_i x dE/de_i`. `bopjax.spin` moves the
axes with the Landau-Lifshitz-Gilbert equation while the atoms stay fixed,

    S_i de_i/dt = -e_i x dE/de_i - alpha |S_i| e_i x de_i/dt,    S_i = hbar m_i / 2

where `m_i` is the self-consistent moment along the axis. The onsite levels,
and with them the lengths of the moments, are made self-consistent at every
step (adiabatic spin dynamics, Antropov et al., PRB 54, 1019 (1996)).

```python
from bopjax import md, spin

init, apply = spin.llg(energy_fn, cell, params, dt=0.2 / md.TIME_UNIT_FS,
                       alpha=0.0, kT=0.0)
state = init(positions, axes, shifts0=bj.stoner_shifts(moments, stoner))
state = jax.jit(apply)(state)
state.axes, state.moment, state.energy, spin.torque(state)
```

| Setting | Dynamics | Counterpart in MD |
|---|---|---|
| `alpha = 0` | precession, energy conserved | NVE |
| `alpha > 0` | Gilbert damping, energy falls until the torques vanish | damped relaxation |
| `alpha > 0, kT > 0` | random torques obeying fluctuation-dissipation (stochastic LLG), directions sample `exp(-E / kT)` | Langevin thermostat |

The integrator is Heun's predictor-corrector scheme with each update applied
as a rotation, so the axes keep unit length to machine precision. It is of
second order and takes two self-consistent gradient evaluations per step, both
warm-started from the onsite levels in the state. `free=mask` holds some axes
fixed. `shifts0` selects the magnetic solution: from zero the solver finds the
non-magnetic one. `spin.integrator` runs the same scheme on any function that
returns an energy gradient and moments, for example a Heisenberg model.

Precession is fast (0.15 rad/fs for a gradient of 0.1 eV on a moment of
2 Bohr magnetons), so steps are a fraction of a femtosecond without damping.

`examples/spin_dynamics.py` runs the 4-atom non-collinear cell of the
[validation](validation.md). Without damping the energy drifts by 1e-7 eV per 0.2 fs step
against torques of 0.1 eV, with two Newton steps per evaluation. With
`alpha = 1` it falls by 0.40 eV in 8 fs, equal to the integral of
`spin.damping_power` to 1e-4 eV. The cost is that of the non-collinear energy:
a step is two self-consistent solves on spinor moments (about 5 s for the 4
atoms on a 4-core cloud machine, most of it in the dense Hessian).

That damped run is stopped early. The model relaxes towards a ferromagnet
with a full spin-up band, and close to saturation `make_potential`'s Newton
solver fails and the state becomes NaN (see [Caveats](limitations.md#caveats)). Relaxing to vanishing
torques needs a model whose minimum is not saturated, or a safeguarded solver.

What is checked (`tests/test_spin.py`): precession and damping of a moment in
a field against the analytic solutions, for both signs of the moment; second
order convergence; conservation of energy and total spin for a Heisenberg
ring; the thermal average of the direction against the Langevin function; and
for the BOP energy, conservation of the energy, its loss to damping and
the equivalence of `(e, m)` and `(-e, -m)`. Thermal torques are tested on
the model energy only.

Limits: the atoms do not move here (see [spin-lattice dynamics](spin-lattice.md)); the
lengths of the moments follow adiabatically and have no fluctuations of their
own; atoms whose moment vanishes have no direction to integrate (`min_moment`
bounds |m| in the equation of motion); without spin-orbit coupling there is
no magnetic anisotropy and no transfer of angular momentum to the lattice.
With the [spin-orbit term](spin-orbit.md) the integrator has only been run
for the uniform precession of a chain (`examples/ovito_spin_orbit.py`, the
first animation in the [README](../README.md)).
