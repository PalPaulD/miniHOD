"""Shared fixtures for miniHOD tests."""

import numpy as np
import pytest

from miniHOD import HOD


def make_catalog(N=10_000, seed=0, box=500.0):
    """Generate a synthetic halo catalog for testing."""
    rng = np.random.default_rng(seed)
    masses = 10 ** rng.uniform(11, 15, N)
    pos    = rng.uniform(0, box, (N, 3))
    vel    = rng.uniform(-300, 300, (N, 3))
    rvir   = (masses / 1e14) ** (1/3) * 1.2   # rough R200m [Mpc/h]
    return masses, pos, vel, rvir
