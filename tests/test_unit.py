"""
Unit tests for miniHOD.

Tests individual components in isolation — occupation functions, RNG
determinism, boundary conditions, and basic population logic.

Run with:  pytest tests/test_unit.py -v
"""

import numpy as np
import pytest

from miniHOD import HOD, HODParams
from .conftest import make_catalog


PARAMS = dict(logMmin=12.5, sigma_logM=0.5, fmax=1.0,
              logMsat=13.5, logMcut=11.5, alpha=1.0)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_determinism():
    """Same seed must produce byte-identical output."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g1 = model.populate(**PARAMS, seed=7)
    g2 = model.populate(**PARAMS, seed=7)
    np.testing.assert_array_equal(g1["pos"], g2["pos"])
    np.testing.assert_array_equal(g1["vel"], g2["vel"])
    np.testing.assert_array_equal(g1["is_central"], g2["is_central"])


def test_different_seeds_differ():
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g1 = model.populate(**PARAMS, seed=1)
    g2 = model.populate(**PARAMS, seed=2)
    assert not np.array_equal(g1["pos"], g2["pos"])


# ── Output shapes and types ───────────────────────────────────────────────────

def test_output_shapes():
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**PARAMS)
    Ngal = len(g["pos"])
    assert g["pos"].shape        == (Ngal, 3)
    assert g["vel"].shape        == (Ngal, 3)
    assert g["is_central"].shape == (Ngal,)
    assert g["is_central"].dtype == bool
    assert g["halo_idx"].shape   == (Ngal,)
    assert g["halo_idx"].dtype   == np.int64


def test_empty_catalog():
    """N=0 halos must return empty arrays with correct shapes."""
    model = HOD(np.array([]), np.empty((0, 3)), np.empty((0, 3)),
                np.array([]), box_size=500.0)
    g = model.populate(**PARAMS)
    assert g["pos"].shape        == (0, 3)
    assert g["vel"].shape        == (0, 3)
    assert g["is_central"].shape == (0,)
    assert g["halo_idx"].shape   == (0,)


# ── Occupation function limits ────────────────────────────────────────────────

def test_ncen_limits():
    """
    At M = Mmin: <Ncen> = fmax/2.
    At M >> Mmin: <Ncen> → fmax.
    """
    box = 500.0
    logMmin, sigma, fmax = 12.5, 0.5, 0.9
    p1 = np.array([[250., 250., 250.]])
    v1 = np.zeros((1, 3))
    r1 = np.array([1.0])
    # logMcut=20 → exp(-Mcut/M) ≈ 0, no satellites
    nd_kw = dict(sigma_logM=sigma, fmax=fmax, logMsat=13.5, logMcut=20.0, alpha=1.0)

    # At M = Mmin: <Ncen> = fmax/2
    model = HOD(np.array([10**logMmin]), p1, v1, r1, box_size=box)
    nd = model.mean_number_density(logMmin, **nd_kw)
    assert abs(nd * box**3 - fmax / 2.0) < 1e-6

    # At M >> Mmin: <Ncen> → fmax
    model_high = HOD(np.array([10**16]), p1, v1, r1, box_size=box)
    nd_high = model_high.mean_number_density(logMmin, **nd_kw)
    assert abs(nd_high * box**3 - fmax) < 1e-6


# ── Periodic boundary conditions ──────────────────────────────────────────────

def test_periodic_wrap():
    """All galaxy positions must lie strictly within [0, box_size)."""
    m, p, v, r = make_catalog(N=50_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)
    for seed in range(10):
        g = model.populate(**PARAMS, seed=seed+1)
        assert np.all(g["pos"] >= 0.0), f"Negative position at seed={seed+1}"
        assert np.all(g["pos"] <  box), f"Position >= box at seed={seed+1}"


# ── Central–satellite coupling ────────────────────────────────────────────────

def test_fmax_zero_gives_no_galaxies():
    """fmax=0 → no centrals → no satellites."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**{**PARAMS, "fmax": 0.0})
    assert len(g["pos"]) == 0


def test_no_satellites_without_central():
    """Every satellite in the output must be preceded by its central."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**PARAMS, seed=42)
    ic = g["is_central"]
    # Walk the output: centrals start a new halo group, satellites follow.
    # A satellite at index 0 with no prior central is a bug.
    assert len(ic) > 0
    assert ic[0], "First galaxy must be a central"
    for i in range(1, len(ic)):
        if ic[i]:
            continue  # new central — ok
        # satellite: at least one central must have appeared before
        assert np.any(ic[:i]), f"Satellite at index {i} with no prior central"


# ── Poisson statistics ────────────────────────────────────────────────────────

def test_galaxy_count_mean():
    """Mean Ngal over many realisations must match mean_number_density * V."""
    m, p, v, r = make_catalog(N=100_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)

    n_expected = model.mean_number_density(**PARAMS) * box**3
    N_trials = 500
    counts = [len(model.populate(**PARAMS, seed=s)["pos"]) for s in range(N_trials)]
    n_mean = np.mean(counts)

    tol = 4.0 * np.std(counts) / np.sqrt(N_trials)
    assert abs(n_mean - n_expected) < tol, (
        f"Mean Ngal={n_mean:.1f} deviates from expected {n_expected:.1f} "
        f"by more than 4-sigma ({tol:.1f})"
    )


# ── fix_logMmin ───────────────────────────────────────────────────────────────

def test_fix_logMmin():
    """mean_number_density with solved logMmin must match n_target to <0.1%."""
    m, p, v, r = make_catalog(N=100_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)

    n_target = model.mean_number_density(13.0, 0.5, 1.0, 13.5, 11.5, 1.0) * 0.7
    lm = model.fix_logMmin(n_target, sigma_logM=0.5, fmax=1.0,
                            logMsat=13.5, logMcut=11.5, alpha=1.0)
    n_achieved = model.mean_number_density(lm, 0.5, 1.0, 13.5, 11.5, 1.0)
    assert abs(n_achieved - n_target) / n_target < 0.001


# ── Buffer overflow / reallocation ───────────────────────────────────────────

def test_buffer_overflow_realloc():
    """Starting with capacity=1 must still work (triggers reallocation)."""
    m, p, v, r = make_catalog(N=1_000)
    model = HOD(m, p, v, r, box_size=500.0, capacity=1)
    g = model.populate(**PARAMS, seed=42)
    assert len(g["pos"]) > 0
    assert g["pos"].shape[1] == 3


# ── Single halo (offset edge case) ──────────────────────────────────────────

def test_single_halo():
    """A single massive halo must produce at least a central."""
    model = HOD(np.array([1e15]), np.array([[250., 250., 250.]]),
                np.array([[100., -50., 30.]]), np.array([2.0]), box_size=500.0)
    g = model.populate(logMmin=10.0, sigma_logM=0.5, fmax=1.0,
                       logMsat=13.0, logMcut=10.0, alpha=1.0, seed=1)
    assert len(g["pos"]) >= 1
    assert g["is_central"][0] == True
    np.testing.assert_array_equal(g["pos"][0], [250., 250., 250.])
    np.testing.assert_array_equal(g["vel"][0], [100., -50., 30.])


# ── HODParams ────────────────────────────────────────────────────────────────

def test_populate_with_hodparams():
    """HODParams namedtuple must produce identical results to kwargs."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    params = HODParams(**PARAMS)
    g_kw = model.populate(**PARAMS, seed=42)
    g_nt = model.populate(params=params, seed=42)
    np.testing.assert_array_equal(g_kw["pos"], g_nt["pos"])
    np.testing.assert_array_equal(g_kw["vel"], g_nt["vel"])


# ── Parameter validation ─────────────────────────────────────────────────────

def test_seed_zero():
    """seed=0 must be remapped (xorshift64 fixed point) and still work."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**PARAMS, seed=0)
    assert len(g["pos"]) > 0


def test_multithreaded_determinism():
    """Same (seed, nthreads) must produce identical output across runs."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g1 = model.populate(**PARAMS, seed=7, nthreads=2)
    g2 = model.populate(**PARAMS, seed=7, nthreads=2)
    np.testing.assert_array_equal(g1["pos"], g2["pos"])
    np.testing.assert_array_equal(g1["vel"], g2["vel"])


def test_invalid_rvir():
    """halo_radii <= 0 must raise ValueError."""
    m, p, v, r = make_catalog()
    r[0] = 0.0
    with pytest.raises(ValueError, match="halo_radii"):
        HOD(m, p, v, r, box_size=500.0)


def test_validation_sigma_logM():
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    with pytest.raises(ValueError, match="sigma_logM"):
        model.populate(**{**PARAMS, "sigma_logM": 0.0})

def test_validation_fmax():
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    with pytest.raises(ValueError, match="fmax"):
        model.populate(**{**PARAMS, "fmax": 1.5})

def test_validation_alpha():
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    with pytest.raises(ValueError, match="alpha"):
        model.populate(**{**PARAMS, "alpha": -0.1})


# ── halo_idx output ─────────────────────────────────────────────────────────

def test_halo_idx_range():
    """All halo indices must be valid indices into the input halo arrays."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**PARAMS, seed=42)
    idx = g["halo_idx"]
    assert np.all(idx >= 0)
    assert np.all(idx < len(m))


def test_halo_idx_central_position():
    """Central galaxy positions must match their host halo positions (after wrap)."""
    m, p, v, r = make_catalog(N=50_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)
    g = model.populate(**PARAMS, seed=42)
    cen = g["is_central"]
    idx = g["halo_idx"][cen]
    # Central pos = wrapped halo pos
    halo_pos_wrapped = p[idx] % box
    np.testing.assert_allclose(g["pos"][cen], halo_pos_wrapped, atol=1e-10)


def test_halo_idx_central_velocity():
    """Central galaxy velocities must exactly match their host halo velocities."""
    m, p, v, r = make_catalog(N=50_000)
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**PARAMS, seed=42)
    cen = g["is_central"]
    idx = g["halo_idx"][cen]
    np.testing.assert_array_equal(g["vel"][cen], v[idx])


def test_halo_idx_determinism():
    """halo_idx must be deterministic (same seed → same output)."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g1 = model.populate(**PARAMS, seed=42)
    g2 = model.populate(**PARAMS, seed=42)
    np.testing.assert_array_equal(g1["halo_idx"], g2["halo_idx"])


def test_halo_idx_satellites_near_host():
    """Satellite positions must be within R200m of their host halo."""
    m, p, v, r = make_catalog(N=50_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)
    g = model.populate(**PARAMS, seed=42)
    sat = ~g["is_central"]
    idx = g["halo_idx"][sat]
    dr = g["pos"][sat] - (p[idx] % box)
    dr -= box * np.round(dr / box)  # periodic
    dist = np.linalg.norm(dr, axis=1)
    assert np.all(dist <= r[idx] * 1.01)  # 1% tolerance for float rounding


# ── solve_logMmin (C bisection) ──────────────────────────────────────────────

def test_solve_logMmin_accuracy():
    """C solver result must produce nbar matching target to <0.1%."""
    m, p, v, r = make_catalog(N=100_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)

    # Use several parameter sets
    rng = np.random.default_rng(99)
    for _ in range(5):
        sigma = rng.uniform(0.2, 0.6)
        fmax = rng.uniform(0.8, 1.0)
        logMsat = rng.uniform(13.5, 14.5)
        logMcut = rng.uniform(12.0, 13.0)
        alpha = rng.uniform(0.8, 1.4)
        n_target = model.mean_number_density(13.0, sigma, fmax,
                                              logMsat, logMcut, alpha) * 0.5
        lm = model.fix_logMmin(n_target, sigma, fmax, logMsat, logMcut, alpha)
        n_achieved = model.mean_number_density(lm, sigma, fmax,
                                                logMsat, logMcut, alpha)
        assert abs(n_achieved - n_target) / n_target < 0.001, (
            f"n_achieved={n_achieved:.6e}, n_target={n_target:.6e}"
        )


def test_solve_logMmin_matches_python():
    """C solver must agree with Python brentq + mean_number_density."""
    from scipy.optimize import brentq

    m, p, v, r = make_catalog(N=100_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)

    kw = dict(sigma_logM=0.5, fmax=0.9, logMsat=13.5, logMcut=11.5, alpha=1.0)
    n_target = model.mean_number_density(13.0, **kw) * 0.7

    # C solver
    lm_c = model.fix_logMmin(n_target, **kw)

    # Python fallback
    def residual(lm):
        return model.mean_number_density(lm, **kw) - n_target
    lm_py = brentq(residual, 10.0, 16.0, xtol=1e-4, maxiter=60)

    assert abs(lm_c - lm_py) < 2e-4, f"C={lm_c:.6f}, Python={lm_py:.6f}"


def test_n_target_populate_density():
    """populate(n_target=X) must produce Ngal close to X * V."""
    m, p, v, r = make_catalog(N=100_000)
    box = 500.0
    model = HOD(m, p, v, r, box_size=box)
    V = box ** 3

    rng = np.random.default_rng(77)
    for i in range(5):
        sigma = rng.uniform(0.3, 0.5)
        fmax = rng.uniform(0.85, 1.0)
        logMsat = rng.uniform(13.5, 14.0)
        logMcut = rng.uniform(11.5, 12.5)
        alpha = rng.uniform(0.9, 1.2)
        n_target = model.mean_number_density(13.0, sigma, fmax,
                                              logMsat, logMcut, alpha) * 0.5
        g = model.populate(n_target=n_target, sigma_logM=sigma, fmax=fmax,
                           logMsat=logMsat, logMcut=logMcut, alpha=alpha,
                           seed=100 + i)
        n_achieved = len(g["pos"]) / V
        assert abs(n_achieved - n_target) / n_target < 0.10, (
            f"n_achieved={n_achieved:.4e}, n_target={n_target:.4e}"
        )


def test_n_target_multithreaded_determinism():
    """populate(n_target=..., nthreads=2) must be deterministic."""
    m, p, v, r = make_catalog(N=50_000)
    model = HOD(m, p, v, r, box_size=500.0)
    kw = dict(n_target=1e-3, sigma_logM=0.5, fmax=1.0,
              logMsat=13.5, logMcut=11.5, alpha=1.0,
              seed=42, nthreads=2)
    g1 = model.populate(**kw)
    g2 = model.populate(**kw)
    np.testing.assert_array_equal(g1["pos"], g2["pos"])
    np.testing.assert_array_equal(g1["vel"], g2["vel"])
    np.testing.assert_array_equal(g1["is_central"], g2["is_central"])
