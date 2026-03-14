/*
 * miniHOD — minimal 6-parameter HOD code
 *
 * HOD model:
 *   <Ncen>(M) = (fmax/2) * (1 + erf((log10(M) - logMmin) / sigma_logM))
 *   <Nsat>(M) = <Ncen>(M) * (M / Msat)^alpha * exp(-Mcut / M)
 *
 * Free parameters: logMmin, sigma_logM, fmax, logMsat, logMcut, alpha
 *
 * Units (caller's responsibility to be consistent):
 *   masses   — Msun/h  (M200m assumed)
 *   pos      — Mpc/h
 *   radii    — Mpc/h   (R200m)
 *   conc     — dimensionless (c200m = R200m / r_s)
 *   vel      — km/s
 *   box_size — Mpc/h
 *
 * Parallelism:
 *   Compiled with -fopenmp: hod_populate uses a two-pass parallel strategy;
 *   hod_mean_number_density uses an OpenMP reduction.
 *   Both functions accept a nthreads argument; pass 0 for OpenMP default.
 *   Without OpenMP the code is fully single-threaded and nthreads is ignored.
 */

#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double G_CONST = 4.3009e-9;  /* (km/s)^2 Mpc/Msun */

/* =========================================================================
 * Jeans velocity dispersion lookup table
 *
 * Isotropic Jeans equation for NFW potential (Łokas & Mamon 2001):
 *   sigma_r^2(r) / Vvir^2 = c*s*(1+s)^2 / g(c) * I(s)
 * where s = c * r/Rvir, g(c) = ln(1+c) - c/(1+c), and
 *   I(s) = int_s^inf [ln(1+y) - y/(1+y)] / [y^3*(1+y)^2] dy
 *
 * We tabulate I(s) on a log-spaced grid and interpolate at runtime.
 * =========================================================================*/

#define JEANS_NPTS  128
#define JEANS_LOG_SMIN (-6.0)  /* s_min = 1e-6 */
#define JEANS_LOG_SMAX  (2.5)  /* s_max ~ 316  */

static double jeans_log_I[JEANS_NPTS]; /* log10(I(s)) for log-log interpolation */
static double jeans_ds_inv;            /* 1 / (log10 spacing between points)   */
static int    jeans_ready = 0;

/* Integrand: g(y) / [y^3 (1+y)^2] computed as term1 - term2 to avoid
 * catastrophic cancellation in g(y) = ln(1+y) - y/(1+y) at small y. */
static double jeans_integrand(double y) {
    double y1 = 1.0 + y;
    return log(y1) / (y * y * y * y1 * y1) - 1.0 / (y * y * y1 * y1 * y1);
}

/* Numerical integration of I(s) via substitution y = s + t/(1-t)^2
 * mapping [s, inf) → [0, 1).  Trapezoidal rule, 512 panels.
 * Called only at startup to fill the lookup table. */
static double jeans_integrate(double s) {
    int npanel = 512;
    double sum = 0.0;
    double dt = 1.0 / npanel;
    for (int k = 0; k <= npanel; k++) {
        double t = k * dt;
        if (t >= 1.0) break;
        double omt = 1.0 - t;
        double y = s + t / (omt * omt);       /* map [0,1) → [s, inf) */
        double dydt = (1.0 + t) / (omt * omt * omt);  /* dy/dt */
        double f = jeans_integrand(y) * dydt;
        double w = (k == 0) ? 0.5 * dt : (k == npanel - 1) ? 0.5 * dt : dt;
        sum += f * w;
    }
    return sum;
}

/* Build the lookup table (called once, thread-safe via flag) */
static void jeans_init(void) {
    if (jeans_ready) return;
    double dlog = (JEANS_LOG_SMAX - JEANS_LOG_SMIN) / (JEANS_NPTS - 1);
    jeans_ds_inv = 1.0 / dlog;
    for (int k = 0; k < JEANS_NPTS; k++) {
        double log_s = JEANS_LOG_SMIN + k * dlog;
        jeans_log_I[k] = log10(jeans_integrate(pow(10.0, log_s)));
    }
    jeans_ready = 1;
}

/* Interpolate I(s) from table — linear in log-log space */
static inline double jeans_lookup(double s) {
    double ls = log10(s);
    if (ls <= JEANS_LOG_SMIN) ls = JEANS_LOG_SMIN + 1e-10;
    if (ls >= JEANS_LOG_SMAX) return 0.0;  /* I(s) → 0 for large s */
    double idx = (ls - JEANS_LOG_SMIN) * jeans_ds_inv;
    int k = (int)idx;
    if (k >= JEANS_NPTS - 1) k = JEANS_NPTS - 2;
    double frac = idx - k;
    double log_I = jeans_log_I[k] + frac * (jeans_log_I[k+1] - jeans_log_I[k]);
    return pow(10.0, log_I);
}

/* 1D velocity dispersion at scaled radius r_tilde = r/Rvir for halo with
 * mass M and virial radius Rvir.  Returns sigma in km/s. */
static inline double jeans_sigma1D(double r_tilde, double c, double M, double Rvir) {
    double Vvir2 = G_CONST * M / Rvir;          /* virial velocity^2 */
    double gc = log(1.0 + c) - c / (1.0 + c);   /* g(c)              */
    double s = c * r_tilde;
    double I_s = jeans_lookup(s);
    double sigma2 = Vvir2 * c * s * (1.0 + s) * (1.0 + s) / gc * I_s;
    /* sigma2 is sigma_r^2 (radial); for isotropic: sigma_1D = sigma_r */
    return (sigma2 > 0.0) ? sqrt(sigma2) : 0.0;
}

/* =========================================================================
 * RNG state — bundles xorshift64 + cached Box-Muller spare
 * =========================================================================*/

typedef struct {
    uint64_t s;
    double   spare;
    int      has_spare;
} rng_state;

static inline uint64_t xorshift64(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

/* Uniform draw in (0, 1) using 53-bit mantissa + midpoint offset */
static inline double rng_double(rng_state *r) {
    return ((xorshift64(&r->s) >> 11) + 0.5) * (1.0 / (double)(UINT64_C(1) << 53));
}

/* Gaussian draw via Box-Muller with spare caching */
static inline double rng_gaussian(rng_state *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    double u1 = rng_double(r);
    double u2 = rng_double(r);
    double mag = sqrt(-2.0 * log(u1));
    double angle = 2.0 * M_PI * u2;
    r->spare = mag * sin(angle);
    r->has_spare = 1;
    return mag * cos(angle);
}

/* =========================================================================
 * Poisson draw
 *   lambda < 30 : Knuth algorithm (exact, O(lambda) uniform draws)
 *   lambda >= 30: Normal approximation with continuity correction
 * =========================================================================*/

static inline int64_t rng_poisson(double lambda, rng_state *r) {
    if (lambda <= 0.0) return 0;
    if (lambda < 30.0) {
        double L = exp(-lambda);
        double p = 1.0;
        int64_t k = 0;
        do { p *= rng_double(r); k++; } while (p > L);
        return k - 1;
    }
    int64_t n = (int64_t)(lambda + sqrt(lambda) * rng_gaussian(r) + 0.5);
    return n < 0 ? 0 : n;
}

/* =========================================================================
 * Default concentration-mass relation: Duffy et al. 2008 (M200m, WMAP5, z=0)
 *   c(M) = 10.14 * (M / 2e12 Msun/h)^{-0.081}
 *
 * Used as fallback when caller does not supply per-halo concentrations.
 * For Planck cosmology the normalization is ~15-20% low; callers with
 * simulation catalogs should pass measured concentrations instead.
 * =========================================================================*/

static inline double concentration_duffy08_200m(double M) {
    return 10.14 * pow(M / 2.0e12, -0.081);
}

/* =========================================================================
 * NFW rejection sampler
 *
 * Samples x = r/Rvir from p(x) ∝ x / (1 + c*x)^2 on [0, 1].
 *
 * Envelope:  q(x) = 2x   =>   proposal via x = sqrt(U1)
 * Acceptance: U2 < 1 / (1 + c*x)^2
 * Expected draws: ~2 for c in [5, 15].
 * =========================================================================*/

static double nfw_sample_r(double c, rng_state *r) {
    for (;;) {
        double x     = sqrt(rng_double(r));
        double denom = 1.0 + c * x;
        if (rng_double(r) < 1.0 / (denom * denom)) return x;
    }
}

/* Periodic wrap — handles arbitrary displacements */
static inline double wrap(double x, double L) {
    x = fmod(x, L);
    if (x < 0.0) x += L;
    return x;
}

/* HOD mean central occupation */
static inline double _ncen(double logM, double logMmin,
                            double sigma_logM, double fmax) {
    return 0.5 * fmax * (1.0 + erf((logM - logMmin) / sigma_logM));
}

/* =========================================================================
 * Write one halo's galaxies into output arrays.
 * Shared by both OpenMP and single-threaded paths.
 * =========================================================================*/

static void write_halo(
    int64_t i, int64_t g, int64_t n_sat_i,
    const double *halo_pos, const double *halo_vel,
    const double *halo_mass, const double *halo_rvir,
    const double *halo_conc,
    double box_size,
    double *out_pos, double *out_vel, uint8_t *out_is_central,
    int64_t *out_halo_idx,
    rng_state *rng)
{
    /* Central: at halo position */
    out_pos[g*3+0] = wrap(halo_pos[i*3+0], box_size);
    out_pos[g*3+1] = wrap(halo_pos[i*3+1], box_size);
    out_pos[g*3+2] = wrap(halo_pos[i*3+2], box_size);
    out_vel[g*3+0] = halo_vel[i*3+0];
    out_vel[g*3+1] = halo_vel[i*3+1];
    out_vel[g*3+2] = halo_vel[i*3+2];
    out_is_central[g] = 1;
    out_halo_idx[g] = i;
    g++;

    if (n_sat_i == 0) return;

    double M     = halo_mass[i];
    double Rvir  = halo_rvir[i];
    double c     = halo_conc[i];

    /* Old constant-dispersion model (virial scaling):
     * double sig1D = sqrt(G_CONST * M / (3.0 * Rvir)); */

    for (int64_t s = 0; s < n_sat_i; s++, g++) {
        double r_tilde   = nfw_sample_r(c, rng);
        double r         = r_tilde * Rvir;
        double cos_theta = 2.0 * rng_double(rng) - 1.0;
        double sin2      = 1.0 - cos_theta * cos_theta;
        double sin_theta = (sin2 > 0.0) ? sqrt(sin2) : 0.0;
        double phi       = 2.0 * M_PI * rng_double(rng);

        double dx = r * sin_theta * cos(phi);
        double dy = r * sin_theta * sin(phi);
        double dz = r * cos_theta;

        out_pos[g*3+0] = wrap(halo_pos[i*3+0] + dx, box_size);
        out_pos[g*3+1] = wrap(halo_pos[i*3+1] + dy, box_size);
        out_pos[g*3+2] = wrap(halo_pos[i*3+2] + dz, box_size);

        double sig1D = jeans_sigma1D(r_tilde, c, M, Rvir);
        out_vel[g*3+0] = halo_vel[i*3+0] + sig1D * rng_gaussian(rng);
        out_vel[g*3+1] = halo_vel[i*3+1] + sig1D * rng_gaussian(rng);
        out_vel[g*3+2] = halo_vel[i*3+2] + sig1D * rng_gaussian(rng);

        out_is_central[g] = 0;
        out_halo_idx[g] = i;
    }
}

/* =========================================================================
 * Exported: mean number density (no RNG — used for fix_logMmin bisection)
 *
 *   n = (1/V) * sum_i [ <Ncen>(M_i) + <Nsat>(M_i) ]
 *
 * Parallelised with OpenMP reduction when available.
 * =========================================================================*/

double hod_mean_number_density(
    const double *masses, int64_t N,
    double logMmin, double sigma_logM, double fmax,
    double logMsat, double logMcut, double alpha,
    double box_volume,
    int nthreads)
{
    double Msat  = pow(10.0, logMsat);
    double Mcut  = pow(10.0, logMcut);
    double total = 0.0;

#ifdef _OPENMP
    int _nt = nthreads > 0 ? nthreads : omp_get_max_threads();
#   pragma omp parallel for reduction(+:total) schedule(static) num_threads(_nt)
#endif
    for (int64_t i = 0; i < N; i++) {
        double logM = log10(masses[i]);
        double nc   = _ncen(logM, logMmin, sigma_logM, fmax);
        double ns   = nc * pow(masses[i] / Msat, alpha) * exp(-Mcut / masses[i]);
        total += nc + ns;
    }
    return total / box_volume;
}

/* =========================================================================
 * Exported: populate halos with galaxies
 *
 * Writes galaxy data into caller-allocated output arrays.
 * Returns number of galaxies written, or -1 if out_capacity was exceeded.
 *
 * Parallelism strategy (two-pass):
 *   Pass 1 (parallel): draw n_cen[i], n_sat[i] for every halo using
 *           per-thread RNGs seeded from (seed ^ thread_id * PRIME).
 *   Prefix sum (serial): compute output offsets[i].
 *   Pass 2 (parallel): write galaxy positions/velocities using offsets.
 *
 * Determinism: results are identical for a given (seed, nthreads) pair.
 * Changing nthreads will change the output catalog (different RNG streams),
 * so fix nthreads across an MCMC run for reproducibility.
 *
 * Satellite velocity dispersion:
 *   Radially-dependent, from isotropic Jeans equation for NFW potential
 *   (Łokas & Mamon 2001).  Lookup table built once on first call.
 *   G = 4.3009e-9 (km/s)^2 Mpc/Msun  (h-independent).
 * =========================================================================*/

int64_t hod_populate(
    const double *halo_pos,       /* (N, 3) row-major C order */
    const double *halo_vel,       /* (N, 3) km/s              */
    const double *halo_mass,      /* (N,)   Msun/h            */
    const double *halo_rvir,      /* (N,)   Mpc/h  (R200m)    */
    const double *halo_conc,      /* (N,)   c200m              */
    int64_t       N_halos,
    double logMmin, double sigma_logM, double fmax,
    double logMsat, double logMcut,    double alpha,
    double box_size,           /* Mpc/h — for periodic wrapping   */
    double  *out_pos,          /* (capacity, 3) pre-allocated      */
    double  *out_vel,          /* (capacity, 3) pre-allocated      */
    uint8_t *out_is_central,   /* (capacity,)   1=central, 0=sat   */
    int64_t *out_halo_idx,     /* (capacity,)   host halo index    */
    int64_t  out_capacity,
    uint64_t seed,
    int      nthreads)
{
    if (N_halos <= 0) return 0;

    jeans_init();

    double Msat = pow(10.0, logMsat);
    double Mcut = pow(10.0, logMcut);

    /* ── Single allocation for intermediate arrays ────────────────────── */
    int64_t *buf = (int64_t *)malloc(3 * N_halos * sizeof(int64_t));
    if (!buf) return -2;
    int64_t *n_cen   = buf;
    int64_t *n_sat   = buf + N_halos;
    int64_t *offsets  = buf + 2 * N_halos;

    /* ── Pass 1: draw occupation numbers ──────────────────────────────── */
#ifdef _OPENMP
    int _nt1 = nthreads > 0 ? nthreads : omp_get_max_threads();
#   pragma omp parallel num_threads(_nt1)
    {
        int tid = omp_get_thread_num();
        rng_state rng_occ = {
            seed ^ ((uint64_t)(tid + 1) * UINT64_C(0x9e3779b97f4a7c15)),
            0.0, 0
        };
        for (int w = 0; w < 8; w++) xorshift64(&rng_occ.s);

#       pragma omp for schedule(static)
        for (int64_t i = 0; i < N_halos; i++) {
            double nc_m   = _ncen(log10(halo_mass[i]), logMmin, sigma_logM, fmax);
            int    has_c  = (rng_double(&rng_occ) < nc_m);
            n_cen[i]      = has_c;
            if (has_c) {
                double ns_m = pow(halo_mass[i] / Msat, alpha) * exp(-Mcut / halo_mass[i]);
                n_sat[i]    = rng_poisson(ns_m, &rng_occ);
            } else {
                n_sat[i] = 0;
            }
        }
    }
#else
    {
        rng_state rng_occ = { seed, 0.0, 0 };
        for (int64_t i = 0; i < N_halos; i++) {
            double nc_m = _ncen(log10(halo_mass[i]), logMmin, sigma_logM, fmax);
            int has_c   = (rng_double(&rng_occ) < nc_m);
            n_cen[i]    = has_c;
            if (has_c) {
                double ns_m = pow(halo_mass[i] / Msat, alpha) * exp(-Mcut / halo_mass[i]);
                n_sat[i]    = rng_poisson(ns_m, &rng_occ);
            } else {
                n_sat[i] = 0;
            }
        }
    }
#endif

    /* ── Prefix sum → output offsets ────────────────────────────────── */
    offsets[0] = 0;
    for (int64_t i = 1; i < N_halos; i++)
        offsets[i] = offsets[i-1] + n_cen[i-1] + n_sat[i-1];

    int64_t total_gal = offsets[N_halos-1] + n_cen[N_halos-1] + n_sat[N_halos-1];

    if (total_gal > out_capacity) {
        free(buf);
        return -1;  /* caller will reallocate and retry */
    }

    /* ── Pass 2: write galaxy data ───────────────────────────────────── */
    uint64_t seed_pos = seed ^ UINT64_C(0x9e3779b97f4a7c15);
    if (seed_pos == 0) seed_pos = UINT64_C(0x6c62272e07bb0142);

#ifdef _OPENMP
    int _nt2 = nthreads > 0 ? nthreads : omp_get_max_threads();
#   pragma omp parallel num_threads(_nt2)
    {
        int tid = omp_get_thread_num();
        rng_state rng_pos = {
            seed_pos ^ ((uint64_t)(tid + 1) * UINT64_C(0x6c62272e07bb0142)),
            0.0, 0
        };
        for (int w = 0; w < 8; w++) xorshift64(&rng_pos.s);

#       pragma omp for schedule(static)
        for (int64_t i = 0; i < N_halos; i++) {
            if (n_cen[i] == 0) continue;
            write_halo(i, offsets[i], n_sat[i],
                       halo_pos, halo_vel, halo_mass, halo_rvir, halo_conc,
                       box_size, out_pos, out_vel, out_is_central,
                       out_halo_idx, &rng_pos);
        }
    }
#else
    {
        rng_state rng_pos = { seed_pos, 0.0, 0 };
        for (int64_t i = 0; i < N_halos; i++) {
            if (n_cen[i] == 0) continue;
            write_halo(i, offsets[i], n_sat[i],
                       halo_pos, halo_vel, halo_mass, halo_rvir, halo_conc,
                       box_size, out_pos, out_vel, out_is_central,
                       out_halo_idx, &rng_pos);
        }
    }
#endif

    free(buf);
    return total_gal;
}
