"""Distorted fcc cell shared by the examples."""

import numpy as np

A = 3.5
CELL = A * np.eye(3)
POSITIONS = A * np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
POSITIONS[0] += [0.15, -0.08, 0.05]
POSITIONS[2] += [-0.05, 0.10, 0.02]
