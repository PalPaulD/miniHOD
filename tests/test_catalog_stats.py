"""
Catalog-level statistical comparison tests for miniHOD.

Compares output galaxy catalogs against halotools Zheng07 as an independent
reference.  The two codes use different RNGs so catalogs are not identical —
we compare *statistics* and allow for Poisson and sample variance.

Tests:
  1. Projected 2PCF wp(rp) internal consistency
  2. Galaxy velocity distribution (mean, std per component)
  3. Central fraction
  4. Number density agreement vs halotools

Run with:  pytest tests/test_catalog_stats.py -v
Requires:  Corrfunc, halotools
"""

import numpy as np
import pytest
from scipy.special import erf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from miniHOD import HOD

Corrfunc = pytest.importorskip("Corrfunc",
    reason="Corrfunc not installed — skipping catalog stats tests")
halotools = pytest.importorskip("halotools",
    reason="halotools not installed — skipping catalog stats tests")

from halotools.empirical_models import PrebuiltHodModelFactory, NFWProfile
from halotools.sim_manager import UserSuppliedHaloCatalog


# ── Shared catalog ────────────────────────────────────────────────────────────

BOX    = 500.0
POP_KW = dict(logMmin=12.5, sigma_logM=0.5, fmax=1.0,
              logMsat=13.5, logMcut=11.5, alpha=1.0)

def _ncen(logM, logMmin, sigma_logM, fmax):
    return 0.5 * fmax * (1.0 + erf((logM - logMmin) / sigma_logM))


@pytest.fixture(scope="module")
def catalogs():
    rng = np.random.default_rng(1)
    N   = 100_000
    masses = 10 ** rng.uniform(11.5, 15.0, N)
    pos    = rng.uniform(0, BOX, (N, 3))
    vel    = rng.uniform(-500, 500, (N, 3))
    # Use halotools' Rvir so both codes have the same velocity scale
    _nfw = NFWProfile()
    rvir = _nfw.halo_mass_to_halo_radius(masses)  # Mpc/h, from halotools cosmology

    model = HOD(masses, pos, vel, rvir, box_size=BOX)

    # Build halotools catalog
    halo_ids = np.arange(N)
    halocat = UserSuppliedHaloCatalog(
        Lbox=BOX, particle_mass=1e9, redshift=0.0,
        halo_id=halo_ids, halo_upid=np.full(N, -1), halo_hostid=halo_ids,
        halo_x=pos[:, 0], halo_y=pos[:, 1], halo_z=pos[:, 2],
        halo_vx=vel[:, 0], halo_vy=vel[:, 1], halo_vz=vel[:, 2],
        halo_mvir=masses, halo_rvir=rvir,  # Mpc/h — same units as positions
        halo_nfw_conc=10.14 * (masses / 2e12)**(-0.081),
    )
    ht_model = PrebuiltHodModelFactory('zheng07', redshift=0.0)
    ht_model.param_dict.update({'logMmin': 12.5, 'sigma_logM': 0.5,
                                'logM1': 13.5, 'logM0': 0.0, 'alpha': 1.0})

    N_real = 40
    mini_cats = [model.populate(**POP_KW, seed=s) for s in range(N_real)]

    ht_cats = []
    for s in range(N_real):
        ht_model.populate_mock(halocat, seed=s)
        gt = ht_model.mock.galaxy_table
        ht_cats.append({
            "pos": np.column_stack([gt['x'], gt['y'], gt['z']]),
            "vel": np.column_stack([gt['vx'], gt['vy'], gt['vz']]),
            "is_central": np.array(gt['gal_type'] == 'centrals'),
        })

    return mini_cats, ht_cats, masses, pos, vel, rvir


# ── Helper: compute wp(rp) with Corrfunc ─────────────────────────────────────

def compute_wp(pos, box):
    from Corrfunc.theory.wp import wp as corrfunc_wp
    rp_bins = np.logspace(-1, 1.5, 16)   # 0.1 – 30 Mpc/h
    rp_cen  = np.sqrt(rp_bins[:-1] * rp_bins[1:])
    pimax   = 40.0
    result  = corrfunc_wp(boxsize=box, pimax=pimax, nthreads=2,
                          binfile=rp_bins,
                          X=pos[:, 0], Y=pos[:, 1], Z=pos[:, 2])
    return rp_cen, result["wp"]


# ── 1. wp(rp) internal consistency ───────────────────────────────────────────

def test_wp_internal_consistency(catalogs):
    """
    wp(rp) must be statistically consistent across different seeds of the same
    HOD model — check coefficient of variation is plausible.
    """
    mini_cats, *_ = catalogs

    wp_all = np.array([compute_wp(g["pos"], BOX)[1] for g in mini_cats])
    rp     = compute_wp(mini_cats[0]["pos"], BOX)[0]

    nonzero = wp_all.mean(axis=0) > 1.0
    if not nonzero.any():
        pytest.skip("All wp bins near zero — random halo catalog, no clustering signal")

    wp   = wp_all[:, nonzero]
    mean = wp.mean(axis=0)
    std  = wp.std(axis=0)

    cv = std / (np.abs(mean) + 1e-10)
    assert cv.max() < 0.5, (
        f"wp has unreasonably large scatter (CV={cv.max():.2f}) at rp={rp[nonzero][cv.argmax()]:.2f}"
    )
    assert (mean > 0).all(), "wp is negative in 1-halo regime"


# ── 2. Velocity distribution ─────────────────────────────────────────────────

def test_velocity_distribution(catalogs):
    """
    Galaxy velocity mean and dispersion per component must agree between
    miniHOD and halotools.  Both use the isotropic Jeans equation for NFW;
    Rvir is computed by halotools and shared with miniHOD so Vvir matches.
    """
    mini_cats, ht_cats, *_ = catalogs

    mu_mini  = np.array([g["vel"].mean(axis=0) for g in mini_cats])
    mu_ht    = np.array([g["vel"].mean(axis=0) for g in ht_cats])
    std_mini = np.array([g["vel"].std(axis=0)  for g in mini_cats])
    std_ht   = np.array([g["vel"].std(axis=0)  for g in ht_cats])

    N = len(mini_cats)
    for k, comp in enumerate(["vx", "vy", "vz"]):
        # Mean velocity should agree (both inherit from halo velocities)
        diff_mean = abs(mu_mini[:, k].mean() - mu_ht[:, k].mean())
        tol_mean  = 3 * np.sqrt(mu_mini[:, k].var()/N + mu_ht[:, k].var()/N)
        assert diff_mean < tol_mean + 1.0, (
            f"Mean {comp}: mini={mu_mini[:,k].mean():.2f}, ht={mu_ht[:,k].mean():.2f}"
        )
        # Velocity dispersion should agree within 5%
        frac_diff = abs(std_mini[:, k].mean() - std_ht[:, k].mean()) / std_ht[:, k].mean()
        assert frac_diff < 0.05, (
            f"Std {comp}: mini={std_mini[:,k].mean():.2f}, "
            f"ht={std_ht[:,k].mean():.2f}, diff={frac_diff:.1%}"
        )


# ── 3. Central fraction ─────────────────────────────────────────────────────

def test_central_fraction_vs_mass(catalogs):
    """
    Central fraction must match the analytic <Ncen>/<Ntot> prediction
    to within 3-sigma Poisson noise.
    """
    mini_cats, _, masses, pos, vel, rvir = catalogs

    logM = np.log10(masses)
    p = POP_KW
    Msat = 10**p['logMsat']
    Mcut = 10**p['logMcut']
    nc = _ncen(logM, p['logMmin'], p['sigma_logM'], p['fmax'])
    ns = nc * (masses / Msat)**p['alpha'] * np.exp(-Mcut / masses)

    f_cen_pred = nc.sum() / (nc + ns).sum()
    f_cen_emp  = np.array([g["is_central"].mean() for g in mini_cats])

    diff = abs(f_cen_emp.mean() - f_cen_pred)
    tol  = 3 * f_cen_emp.std() / np.sqrt(len(mini_cats))
    assert diff < tol + 0.005, (
        f"Global f_cen: empirical={f_cen_emp.mean():.4f}, analytic={f_cen_pred:.4f}"
    )


# ── 4. Number density agreement ─────────────────────────────────────────────

def test_number_density_agreement(catalogs):
    """
    Mean Ngal/V must agree between miniHOD and halotools to within 5%.

    Note: the satellite functional forms differ slightly
    (miniHOD: Ncen * (M/Msat)^alpha * exp(-Mcut/M),
     halotools: ((M-M0)/M1)^alpha) so exact agreement is not expected.
    """
    mini_cats, ht_cats, *_ = catalogs

    n_mini = np.array([len(g["pos"]) for g in mini_cats]) / BOX**3
    n_ht   = np.array([len(g["pos"]) for g in ht_cats])   / BOX**3

    frac_diff = abs(n_mini.mean() - n_ht.mean()) / n_ht.mean()
    assert frac_diff < 0.05, (
        f"n_gal: mini={n_mini.mean():.4e}, ht={n_ht.mean():.4e}, "
        f"frac diff={frac_diff:.3f}"
    )
