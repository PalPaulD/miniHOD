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


# ── Central–satellite decoupling ──────────────────────────────────────────────

def test_fmax_zero_gives_no_galaxies():
    """fmax=0 → <Ncen>=0 → <Nsat>=0 → no galaxies."""
    m, p, v, r = make_catalog()
    model = HOD(m, p, v, r, box_size=500.0)
    g = model.populate(**{**PARAMS, "fmax": 0.0})
    assert len(g["pos"]) == 0


def test_satellites_without_central():
    """With low fmax, some halos must produce satellites without a central."""
    # Single massive halo with low fmax → ~70% chance of no central per seed,
    # but high satellite expectation.  Over 200 seeds, the probability of
    # never seeing a satellite-only halo is negligible (~0.3^200).
    M = np.array([1e15])
    p = np.array([[250., 250., 250.]])
    v = np.zeros((1, 3))
    r = np.array([2.0])
    model = HOD(M, p, v, r, box_size=500.0)
    pop_kw = dict(logMmin=12.5, sigma_logM=0.5, fmax=0.3,
                  logMsat=13.0, logMcut=10.0, alpha=1.0)

    found = False
    for seed in range(200):
        g = model.populate(**pop_kw, seed=seed)
        has_cen = g["is_central"].any()
        has_sat = (~g["is_central"]).any()
        if has_sat and not has_cen:
            found = True
            break

    assert found, "Expected at least one realisation with satellites but no central"


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


# ── Redshift-dependent concentration (Duffy+08) ─────────────────────────────

def test_duffy08_redshift_z0_unchanged():
    """redshift=0 (default) must produce identical results to old behaviour."""
    m, p, v, r = make_catalog()
    model_default = HOD(m, p, v, r, box_size=500.0)
    model_z0      = HOD(m, p, v, r, box_size=500.0, redshift=0.0)
    g1 = model_default.populate(**PARAMS, seed=42)
    g2 = model_z0.populate(**PARAMS, seed=42)
    np.testing.assert_array_equal(g1["pos"], g2["pos"])
    np.testing.assert_array_equal(g1["vel"], g2["vel"])


def test_duffy08_redshift_lowers_concentration():
    """Higher redshift must yield lower concentrations (Duffy+08 C < 0)."""
    c_z0 = HOD._conc_duffy08_200m(np.array([1e13]), redshift=0.0)
    c_z1 = HOD._conc_duffy08_200m(np.array([1e13]), redshift=1.0)
    c_z2 = HOD._conc_duffy08_200m(np.array([1e13]), redshift=2.0)
    assert c_z0 > c_z1 > c_z2, f"Expected c(z=0)>{c_z0} > c(z=1)>{c_z1} > c(z=2)>{c_z2}"


def test_duffy08_redshift_scaling():
    """Verify the (1+z)^{-1.01} scaling factor quantitatively."""
    masses = np.logspace(11, 15, 50)
    c_z0 = HOD._conc_duffy08_200m(masses, redshift=0.0)
    c_z1 = HOD._conc_duffy08_200m(masses, redshift=1.0)
    ratio = c_z1 / c_z0
    expected = 2.0 ** (-1.01)
    np.testing.assert_allclose(ratio, expected, rtol=1e-10,
        err_msg="c(z=1)/c(z=0) should equal 2^{-1.01}")


def test_redshift_affects_satellite_positions():
    """Higher-z concentrations change satellite radial distribution."""
    m = np.array([1e14])
    p = np.array([[250., 250., 250.]])
    v = np.zeros((1, 3))
    r = np.array([2.0])

    pop_kw = dict(logMmin=10.0, sigma_logM=0.1, fmax=1.0,
                  logMsat=11.0, logMcut=8.0, alpha=1.0)

    def mean_sat_radius(redshift):
        model = HOD(m, p, v, r, box_size=500.0, redshift=redshift)
        radii = []
        for seed in range(200):
            g = model.populate(**pop_kw, seed=seed)
            sat = ~g["is_central"]
            if sat.any():
                dr = g["pos"][sat] - [250., 250., 250.]
                radii.append(np.linalg.norm(dr, axis=1))
        return np.concatenate(radii).mean()

    r_z0 = mean_sat_radius(0.0)
    r_z1 = mean_sat_radius(1.0)
    # Lower concentration → less centrally concentrated → larger mean radius
    assert r_z1 > r_z0, (
        f"Higher z should give larger mean satellite radius: r(z=0)={r_z0:.4f}, r(z=1)={r_z1:.4f}"
    )


def test_explicit_conc_ignores_redshift():
    """When halo_conc is provided, redshift parameter should have no effect."""
    m, p, v, r = make_catalog()
    conc = np.full_like(m, 8.0)
    model_z0 = HOD(m, p, v, r, box_size=500.0, halo_conc=conc, redshift=0.0)
    model_z1 = HOD(m, p, v, r, box_size=500.0, halo_conc=conc, redshift=1.0)
    g1 = model_z0.populate(**PARAMS, seed=42)
    g2 = model_z1.populate(**PARAMS, seed=42)
    np.testing.assert_array_equal(g1["pos"], g2["pos"])
    np.testing.assert_array_equal(g1["vel"], g2["vel"])


# ── Parameter validation ─────────────────────────────────────────────────────

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
