# Performance

## CPU

Moments are evaluated on a local cluster around each atom (the atoms within
half the path length), so the cost is linear in the number of atoms. For more
than 16 unknowns the Newton step is solved with GMRES on Hessian-vector
products instead of a dense Hessian; the Hessian is well conditioned, so about
ten products per step are enough.

Distorted fcc supercells, canonical d-band, 9 moments, Apple M1 Pro. Times
exclude compilation. Reproduce with `benchmarks/scaling.py`.

| Atoms | bopjax, one energy + force | bopjax, self-consistent energy + force |
|---|---|---|
| 4 | 0.005 s | 0.06 s |
| 32 | 0.03 s | 1.2 s |
| 108 | 0.09 s | 3.6 s |
| 256 | 0.31 s | 14.9 s |

The self-consistent column starts from zero shifts and converges to 1e-12 in
three Newton steps.

Compilation takes 10 to 30 s per configuration, and building the clusters is
done in Python (1.5 s for 256 atoms).

## GPUs

- **Apple Silicon (Metal):** tried with `jax-metal` 0.1.1 on an M1 Pro. It
  runs and gives correct results, but it is about 200 times slower than the
  CPU (6.3 s against 0.03 s for one energy + force evaluation of 32 atoms) and
  needs the tensor contractions rewritten, because the backend cannot lower
  them. The plugin was last released in October 2024, requires JAX 0.4.34 and
  has no double precision. Not supported here.
- **Single precision** itself costs little for the 9-moment d-band case:
  about 1e-6 eV per atom in the energy and 4e-5 eV/A in the forces (set
  `jax_enable_x64` to False after importing bopjax and pass float32 arrays).
- **CUDA:** not tried. The original Jetson Nano is limited to CUDA 10.2,
  while current JAX needs CUDA 12.
