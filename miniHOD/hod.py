"""
miniHOD — HOD class with MCMC-optimized populate().

Dependencies: numpy, scipy (for brentq bisection).
"""

import numpy as np
from collections import namedtuple
from . import _core


HODParams = namedtuple("HODParams", [
    "logMmin", "sigma_logM", "fmax", "logMsat", "logMcut", "alpha"
])
HODParams.__new__.__defaults__ = (None, 0.5, 1.0, 13.5, 11.5, 1.0)


class HOD:
    """
    Minimal 6-parameter HOD model for MCMC.

    HOD functional forms:
        <Ncen>(M) = (fmax/2) * (1 + erf((log10(M) - logMmin) / sigma_logM))
        <Nsat>(M) = <Ncen>(M) * (M / Msat)^alpha * exp(-Mcut / M)

    All halo inputs should use a consistent mass definition (M200m recommended).

    Parameters
    ----------
    halo_masses : array_like, shape (N,)   [Msun/h]  — halo masses (M200m)
    halo_pos    : array_like, shape (N, 3) [Mpc/h]   — halo positions
    halo_vel    : array_like, shape (N, 3) [km/s]    — halo velocities
    halo_radii  : array_like, shape (N,)   [Mpc/h]   — halo radii (R200m)
    box_size    : float  [Mpc/h]  — required for periodic wrapping and fix_logMmin
    halo_conc   : array_like, shape (N,), optional — NFW concentrations (c200m).
                  If None, computed from Duffy+08 M200m relation (WMAP5).
                  Pass explicitly for Planck cosmology or measured concentrations.
    redshift    : float, optional — snapshot redshift for the Duffy+08 fallback
                  c-M relation.  Only used when halo_conc is None.  Default 0.0.
    capacity    : int, optional — initial output buffer size; defaults to 3 * N_halos.
                  Automatically doubled on overflow (amortized, rare after first calls).
    """

    @staticmethod
    def _conc_duffy08_200m(masses, redshift=0.0):
        """Default c-M: Duffy+08, M200m, full sample (WMAP5).
        c(M,z) = 10.14 * (M/2e12)^{-0.081} * (1+z)^{-1.01}
        Valid for z < 2, M in [1e11, 1e15] Msun/h.
        """
        return 10.14 * (masses / 2e12) ** (-0.081) * (1.0 + redshift) ** (-1.01)

    def __init__(self, halo_masses, halo_pos, halo_vel, halo_radii,
                 box_size=None, halo_conc=None, redshift=0.0, capacity=None):
        # Force contiguous float64 C-order once — never again during MCMC
        self._mass = np.ascontiguousarray(halo_masses, dtype=np.float64)
        self._pos  = np.ascontiguousarray(halo_pos,    dtype=np.float64)
        self._vel  = np.ascontiguousarray(halo_vel,    dtype=np.float64)

        self._rvir = np.ascontiguousarray(halo_radii, dtype=np.float64)

        if halo_conc is None:
            self._conc = np.ascontiguousarray(
                self._conc_duffy08_200m(self._mass, redshift), dtype=np.float64)
        else:
            self._conc = np.ascontiguousarray(halo_conc, dtype=np.float64)

        N = len(self._mass)
        if self._pos.shape != (N, 3):
            raise ValueError("halo_pos must be (N, 3)")
        if self._vel.shape != (N, 3):
            raise ValueError("halo_vel must be (N, 3)")
        if self._rvir.shape != (N,):
            raise ValueError("halo_radii must be (N,)")
        if N > 0 and np.any(self._rvir <= 0):
            raise ValueError("halo_radii must be > 0")
        if self._conc.shape != (N,):
            raise ValueError("halo_conc must be (N,)")
        if N > 0 and np.any(self._conc <= 0):
            raise ValueError("halo_conc must be > 0")

        self._N      = N
        self._box    = float(box_size)
        self._volume = self._box ** 3

        # Cache ctypes pointers for halo arrays (immutable after init)
        self._mass_ptr = _core.dptr(self._mass)
        self._pos_ptr  = _core.dptr(self._pos)
        self._vel_ptr  = _core.dptr(self._vel)
        self._rvir_ptr = _core.dptr(self._rvir)
        self._conc_ptr = _core.dptr(self._conc)

        cap = int(capacity) if capacity is not None else max(3 * N, 100_000)
        self._alloc(cap)

    # ── Buffer management ──────────────────────────────────────────────────

    def _alloc(self, capacity):
        self._out_pos  = np.empty((capacity, 3), dtype=np.float64)
        self._out_vel  = np.empty((capacity, 3), dtype=np.float64)
        self._out_cen  = np.empty(capacity,      dtype=np.uint8)
        self._out_hidx = np.empty(capacity,      dtype=np.int64)
        self._cap      = capacity

    # ── Core C call ────────────────────────────────────────────────────────

    def _call_c(self, logMmin, sigma_logM, fmax, logMsat, logMcut, alpha,
                seed, nthreads):
        """Call hod_populate; reallocate and retry on overflow (returns Ngal)."""
        if seed == 0:
            seed = 0xdeadbeef12345678  # xorshift64 fixed point; remap
        while True:
            Ngal = _core.lib.hod_populate(
                self._pos_ptr,
                self._vel_ptr,
                self._mass_ptr,
                self._rvir_ptr,
                self._conc_ptr,
                self._N,
                logMmin, sigma_logM, fmax,
                logMsat, logMcut, alpha,
                self._box,
                _core.dptr(self._out_pos),  # not cached — may reallocate
                _core.dptr(self._out_vel),
                _core.u8ptr(self._out_cen),
                _core.i64ptr(self._out_hidx),
                self._cap,
                seed,
                nthreads,
            )
            if Ngal >= 0:
                return int(Ngal)
            if Ngal == -2:
                raise MemoryError("hod_populate: malloc failed")
            # Ngal == -1: overflow — double buffer and retry
            self._alloc(self._cap * 2)

    # ── Validation ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate(logMmin, sigma_logM, fmax, logMsat, logMcut, alpha):
        if sigma_logM <= 0:
            raise ValueError(f"sigma_logM must be > 0, got {sigma_logM}")
        if not (0.0 <= fmax <= 1.0):
            raise ValueError(f"fmax must be in [0, 1], got {fmax}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

    # ── Public API ─────────────────────────────────────────────────────────

    def populate(self, logMmin=None, sigma_logM=0.5, fmax=1.0,
                 logMsat=13.5, logMcut=11.5, alpha=1.0,
                 n_target=None, seed=42, nthreads=1, params=None):
        """
        Populate halos with galaxies.

        Provide either `logMmin` (explicit), `n_target` (auto-solve),
        or a `params` (HODParams namedtuple).

        Parameters
        ----------
        logMmin     : float — log10(Mmin / [Msun/h])
        sigma_logM  : float — width of central occupation transition
        fmax        : float in [0, 1] — max central occupation at high mass
        logMsat     : float — log10(Msat / [Msun/h]), satellite mass scale
        logMcut     : float — log10(Mcut / [Msun/h]), satellite exponential cutoff
        alpha       : float — satellite power-law slope (>= 0)
        n_target    : float — target galaxy number density [(Mpc/h)^-3].
                      If given, logMmin is solved via bisection (logMmin ignored).
        seed        : int — RNG seed (same seed → identical catalog)
        nthreads    : int — OpenMP threads (0 = use all available cores)
        params      : HODParams — alternative to individual kwargs

        Returns
        -------
        dict with keys:
            'pos'        : ndarray (Ngal, 3) [Mpc/h]  — views into pre-alloc buffer
            'vel'        : ndarray (Ngal, 3) [km/s]   — views into pre-alloc buffer
            'is_central' : ndarray (Ngal,)  bool      — True = central
            'halo_idx'   : ndarray (Ngal,)  int64     — index into input halo arrays
        Arrays are *views*, not copies. Copy if you need persistence across calls.
        """
        if params is not None:
            logMmin, sigma_logM, fmax = params.logMmin, params.sigma_logM, params.fmax
            logMsat, logMcut, alpha   = params.logMsat, params.logMcut, params.alpha

        if n_target is not None:
            logMmin = self.fix_logMmin(n_target, sigma_logM, fmax,
                                       logMsat, logMcut, alpha,
                                       nthreads=nthreads)
        elif logMmin is None:
            raise ValueError("Provide either logMmin, n_target, or params.")

        self._validate(logMmin, sigma_logM, fmax, logMsat, logMcut, alpha)

        Ngal = self._call_c(logMmin, sigma_logM, fmax,
                            logMsat, logMcut, alpha, int(seed), int(nthreads))
        return {
            "pos":        self._out_pos[:Ngal],
            "vel":        self._out_vel[:Ngal],
            "is_central": self._out_cen[:Ngal].view(bool),
            "halo_idx":   self._out_hidx[:Ngal],
        }

    def fix_logMmin(self, n_target, sigma_logM, fmax,
                    logMsat, logMcut, alpha,
                    bracket=(10.0, 16.0), xtol=1e-4, nthreads=0):
        """
        Solve for logMmin that matches a target galaxy number density.

        Uses Brent's method in C with precomputed invariants (no random
        draws, no Python-C overhead per iteration).

        Parameters
        ----------
        n_target : float — desired n_gal in (Mpc/h)^-3
        bracket  : (lo, hi) search interval in logMmin
        nthreads : int — OpenMP threads (0 = all available)

        Returns
        -------
        float — solved logMmin
        """
        result = _core.lib.hod_solve_logMmin(
            self._mass_ptr, self._N,
            sigma_logM, fmax, logMsat, logMcut, alpha,
            n_target, self._volume,
            bracket[0], bracket[1],
            xtol, 60,
            int(nthreads),
        )
        if result != result:  # NaN check
            raise ValueError(
                f"Could not solve for logMmin: root not bracketed in "
                f"[{bracket[0]}, {bracket[1]}] for n_target={n_target:.4e}"
            )
        return result

    def mean_number_density(self, logMmin, sigma_logM, fmax,
                            logMsat, logMcut, alpha, nthreads=0):
        """
        Mean galaxy number density for given parameters (no random draws).
        Useful as a fast likelihood constraint in MCMC.
        """
        return _core.lib.hod_mean_number_density(
            self._mass_ptr, self._N,
            logMmin, sigma_logM, fmax,
            logMsat, logMcut, alpha,
            self._volume,
            int(nthreads),
        )
