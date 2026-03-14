"""
Statistical and comparison tests for miniHOD.

These tests verify:
  1. Occupation moments match analytic expectations
  2. mean_number_density is self-consistent
  3. NFW radial CDF matches theory (KS test via scipy)
  4. Comparison against halotools (Zheng07) as external reference

Run with:  pytest tests/test_stats.py -v
Requires:  scipy
Optional:  halotools  (test_halotools_comparison is skipped if not installed)
"""

import numpy as np
import pytest
from scipy.special import erf
from scipy.stats import kstest

from miniHOD import HOD
from .conftest import make_catalog


# ── Helpers ───────────────────────────────────────────────────────────────────

def ncen_analytic(logM, logMmin, sigma_logM, fmax):
    return 0.5 * fmax * (1.0 + erf((logM - logMmin) / sigma_logM))


def nsat_analytic(M, logM, logMmin, sigma_logM, fmax, logMsat, logMcut, alpha):
    nc   = ncen_analytic(logM, logMmin, sigma_logM, fmax)
    Msat = 10**logMsat
    Mcut = 10**logMcut
    return nc * (M / Msat)**alpha * np.exp(-Mcut / M)


# fmax=0.9 (not 1.0) to exercise the f_cen < 1 path
PARAMS = dict(logMmin=12.5, sigma_logM=0.5, fmax=0.9,
              logMsat=13.5, logMcut=11.5, alpha=1.0)


# ── 1. Occupation moments vs. analytic ───────────────────────────────────────

def test_occupation_functions_match_analytic():
    """
    hod_mean_number_density must equal sum of analytic <Ncen>+<Nsat> / V.
    Tests that the C implementation matches the Python reference to 1e-8.
    """
    masses, pos, vel, rvir = make_catalog(N=100_000)
    box = 500.0
    model = HOD(masses, pos, vel, rvir, box_size=box)

    p = PARAMS
    logM  = np.log10(masses)
    nc    = ncen_analytic(logM, p["logMmin"], p["sigma_logM"], p["fmax"])
    ns    = nsat_analytic(masses, logM, p["logMmin"], p["sigma_logM"], p["fmax"],
                          p["logMsat"], p["logMcut"], p["alpha"])
    n_ref = (nc + ns).sum() / box**3

    n_c = model.mean_number_density(**p)
    np.testing.assert_allclose(n_c, n_ref, rtol=1e-8,
        err_msg="C mean_number_density disagrees with Python analytic reference")


# ── 2. Mean Ngal self-consistency ─────────────────────────────────────────────

def test_ngal_mean_matches_analytic():
    """
    <Ngal> over many realisations should equal sum_i(<Ncen>_i + <Nsat>_i).
    """
    masses, pos, vel, rvir = make_catalog(N=100_000)
    box = 500.0
    model = HOD(masses, pos, vel, rvir, box_size=box)

    n_expected = model.mean_number_density(**PARAMS) * box**3
    counts = np.array([
        len(model.populate(**PARAMS, seed=s)["pos"]) for s in range(500)
    ])
    mean, std = counts.mean(), counts.std()
    tol = 4.0 * std / np.sqrt(len(counts))

    assert abs(mean - n_expected) < tol, (
        f"<Ngal>={mean:.1f} vs expected={n_expected:.1f}, 4σ tol={tol:.1f}"
    )


def test_central_fraction():
    """
    Fraction of centrals must match sum(<Ncen>) / sum(<Ncen> + <Nsat>) analytically.
    """
    masses, pos, vel, rvir = make_catalog(N=100_000)
    box = 500.0
    model = HOD(masses, pos, vel, rvir, box_size=box)

    p = PARAMS
    logM = np.log10(masses)
    nc   = ncen_analytic(logM, p["logMmin"], p["sigma_logM"], p["fmax"])
    ns   = nsat_analytic(masses, logM, p["logMmin"], p["sigma_logM"], p["fmax"],
                         p["logMsat"], p["logMcut"], p["alpha"])
    f_cen_expected = nc.sum() / (nc + ns).sum()

    f_cens = np.array([
        model.populate(**p, seed=s)["is_central"].mean() for s in range(500)
    ])
    f_cen_mean = f_cens.mean()
    f_cen_std  = f_cens.std() / np.sqrt(len(f_cens))

    assert abs(f_cen_mean - f_cen_expected) < 4 * f_cen_std, (
        f"f_cen={f_cen_mean:.4f} vs expected={f_cen_expected:.4f}"
    )


# ── 3. NFW radial CDF (KS test) ───────────────────────────────────────────────

def nfw_cdf(x, c):
    """Analytic CDF of p(x) ∝ x/(1+cx)^2 on [0,1]."""
    def F(t):
        return np.log(1 + c*t) - c*t / (1 + c*t)
    return F(x) / F(1.0)


def test_nfw_radial_cdf():
    """
    Satellite r/Rvir distribution must pass a KS test against the NFW CDF.
    Uses a single massive halo so c(M) is fixed and the CDF is known.
    """
    box = 2000.0
    M0  = 1e14

    model = HOD(np.array([M0]), np.array([[box/2, box/2, box/2]]),
                np.zeros((1, 3)), np.array([1.0]), box_size=box)

    pop_kw = dict(logMmin=10.0, sigma_logM=0.1, fmax=1.0,
                  logMsat=np.log10(M0/100), logMcut=8.0, alpha=1.0)

    all_sats = []
    for seed in range(200):
        g = model.populate(**pop_kw, seed=seed)
        all_sats.append(g["pos"][~g["is_central"]].copy())
    all_sats = np.concatenate(all_sats)

    # r/Rvir (rvir=1, so r = |displacement from halo centre|)
    dr = all_sats - [box/2, box/2, box/2]
    dr -= box * np.round(dr / box)
    r_over_rvir = np.linalg.norm(dr, axis=1)

    assert len(r_over_rvir) >= 5000, "Too few satellites; check HOD parameters"

    c = 10.14 * (M0 / 2e12) ** (-0.081)  # Duffy+08, M200m, z=0
    stat, pval = kstest(r_over_rvir, lambda x: nfw_cdf(x, c))
    assert pval > 0.01, (
        f"NFW radial KS test failed: stat={stat:.4f}, p={pval:.4f}, c={c:.2f}, N={len(r_over_rvir)}"
    )


# ── 4. Comparison against halotools ───────────────────────────────────────────

def test_halotools_comparison():
    """
    Cross-check miniHOD mean occupation against halotools Zheng07 model.

    Setting fmax=1 and logMcut very negative recovers the Zheng07 form.
    """
    pytest.importorskip("halotools",
        reason="halotools not installed — skipping cross-code comparison")
    from halotools.empirical_models import Zheng07Cens, Zheng07Sats

    logMmin    = 12.0
    sigma_logM = 0.5
    logM1      = 13.5
    alpha      = 1.0

    cen_model = Zheng07Cens()
    sat_model = Zheng07Sats()
    cen_model.param_dict.update({"logMmin": logMmin, "sigma_logM": sigma_logM})
    sat_model.param_dict.update({"logM1": logM1, "alpha": alpha,
                                 "logM0": 0.0, "logMmin": logMmin,
                                 "sigma_logM": sigma_logM})

    logM_grid = np.linspace(11, 16, 200)
    M_grid    = 10**logM_grid

    nc_ht = cen_model.mean_occupation(prim_haloprop=M_grid)
    ns_ht = sat_model.mean_occupation(prim_haloprop=M_grid)

    nc_mh = ncen_analytic(logM_grid, logMmin, sigma_logM, fmax=1.0)
    ns_mh = nsat_analytic(M_grid, logM_grid, logMmin, sigma_logM, fmax=1.0,
                          logMsat=logM1, logMcut=-30.0, alpha=alpha)

    mask = nc_ht > 0.01
    np.testing.assert_allclose(nc_mh[mask], nc_ht[mask], rtol=1e-6,
        err_msg="<Ncen> disagrees with halotools Zheng07")

    # Compare conditional satellite mean: miniHOD <Nsat>/<Ncen> vs halotools <Nsat>
    # (equivalent when M0→0 and Mcut→-inf)
    cond_mh = ns_mh[mask] / nc_mh[mask]
    np.testing.assert_allclose(cond_mh, ns_ht[mask], rtol=1e-4,
        err_msg="miniHOD <Nsat>/<Ncen> disagrees with halotools <Nsat>")
