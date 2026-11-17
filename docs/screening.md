# Screening

With `BOPConfig(screening=True)` the d-d bond integrals depend on the
environment of the bond (Nguyen-Manh, Pettifor and Vitek, PRL 85, 4136
(2000); Mrovec et al., PRB 69, 094115 (2004)): a valence s orbital on every
atom k around the bond i-j screens it,

    beta~_tau = beta_tau (1 - S_tau),    tau = sigma, pi, delta
    S_tau = (c1 - mu2 + mu3) / (1 + O_tau^2 - 2 mu2)

where `c1` is the interference of the sd-sigma bond integral and overlap
through k, and `mu2`, `mu3` are moments of the overlap matrix
(`hamiltonian.screening_terms` has the expressions). The functions go in
`params["screening"]`:

```python
params["screening"] = {
    "sds": ...,            # sd-sigma bond integral, [s species, d species]
    "sds_overlap": ...,    # sd-sigma overlap
    "dds_overlap": ..., "ddp_overlap": ..., "ddd_overlap": ...,
}                          # each (nspecies, nspecies, npar)
energy = bj.make_energy_fn(topo, cfg, radial=..., screening_radial=...)
```

They share the cutoff of the bond integrals, so no further neighbour list is
needed and the cost is small next to the moments. Forces and stress, which
now contain three-body terms, come from the same automatic differentiation.

`examples/screening.py` prints the screened integrals of bcc, fcc and simple
cubic neighbours and the bcc-fcc energy difference with and without
screening. With its (made-up) functions the second neighbours of bcc keep
79 % of the pi and 57 % of the delta integral against 94 % and 74 % for the
first neighbours, although the unscreened integrals depend on distance alone.

**Not yet validated against BOPfox.** The angular factors and weights are
checked against explicit Slater-Koster matrices of a trimer, the dimer limit
`beta / (1 - O^2)`, hermiticity, and forces and stress against finite
differences (`tests/test_screening.py`). The prefactors of `mu3` and the
exact form BOPfox evaluates were not available when this was written and may
differ.
