"""
miniHOD example — populating a halo catalog with galaxies
"""

import numpy as np
from miniHOD import HOD

# ── 1. Create a mock halo catalog ─────────────────────────────────────────────
#
# In practice you'd load this from a simulation snapshot (e.g. Rockstar, AHF).
# Here we generate a simple random catalog for illustration.

BOX_SIZE = 500.0   # Mpc/h

rng = np.random.default_rng(0)
N_halos = 100_000

halo_masses = 10 ** rng.uniform(11.5, 15.0, N_halos)   # Msun/h
halo_pos    = rng.uniform(0, BOX_SIZE, (N_halos, 3))    # Mpc/h
halo_vel    = rng.uniform(-500, 500,   (N_halos, 3))    # km/s
halo_rvir   = (halo_masses / 1e14) ** (1/3) * 1.2      # R200m [Mpc/h] (rough scaling)

print(f"Halo catalog: {N_halos:,} halos in a {BOX_SIZE:.0f} Mpc/h box")
print(f"  Mass range: {halo_masses.min():.2e} – {halo_masses.max():.2e} Msun/h")
print()

# ── 2. Initialise the HOD model ───────────────────────────────────────────────
#
# This step pre-allocates output buffers and converts the halo arrays to
# contiguous float64. Do this once before your MCMC loop.

model = HOD(halo_masses, halo_pos, halo_vel, halo_rvir, box_size=BOX_SIZE)

# ── 3. Populate with explicit HOD parameters ──────────────────────────────────
#
# 6 free parameters:
#   logMmin    — log10(minimum halo mass for centrals) [Msun/h]
#   sigma_logM — width of the central occupation transition
#   fmax       — maximum central occupation (1 = all massive halos have a central)
#   logMsat    — log10(satellite mass scale) [Msun/h]
#   logMcut    — log10(satellite exponential cutoff mass) [Msun/h]
#   alpha      — satellite power-law slope

galaxies = model.populate(
    logMmin    = 12.5,
    sigma_logM = 0.5,
    fmax       = 1.0,
    logMsat    = 13.5,
    logMcut    = 11.5,
    alpha      = 1.0,
    seed       = 42,
)

pos        = galaxies["pos"]         # (Ngal, 3)  Mpc/h
vel        = galaxies["vel"]         # (Ngal, 3)  km/s
is_central = galaxies["is_central"]  # (Ngal,)    bool
halo_idx   = galaxies["halo_idx"]   # (Ngal,)    int64 — index into input halo arrays

n_cen = is_central.sum()
n_sat = (~is_central).sum()
n_gal = len(pos)

print("── Explicit parameters ──────────────────────────────────────────────────")
print(f"  Ngal = {n_gal:,}  ({n_cen:,} centrals, {n_sat:,} satellites)")
print(f"  f_cen = {n_cen / n_gal:.3f}")
print(f"  n_gal = {n_gal / BOX_SIZE**3:.4e} (Mpc/h)^-3")
print(f"  Unique host halos = {len(np.unique(halo_idx)):,}")
print(f"  Host mass of first galaxy = {halo_masses[halo_idx[0]]:.2e} Msun/h")
print()

# ── 4. Fix logMmin to match a target number density ───────────────────────────
#
# Useful when you want to use n_gal as a constraint in your MCMC likelihood.
# The bisection is fast (~100 ms) and involves no random draws.

n_target = 5e-4   # (Mpc/h)^-3

galaxies2 = model.populate(
    n_target   = n_target,
    sigma_logM = 0.5,
    fmax       = 1.0,
    logMsat    = 13.5,
    logMcut    = 11.5,
    alpha      = 1.0,
    seed       = 42,
)

n_achieved = len(galaxies2["pos"]) / BOX_SIZE**3

print("── Fixed number density ─────────────────────────────────────────────────")
print(f"  n_target   = {n_target:.4e} (Mpc/h)^-3")
print(f"  n_achieved = {n_achieved:.4e} (Mpc/h)^-3")
print()

# ── 5. Query mean number density without populating ───────────────────────────
#
# Even faster — pure arithmetic over halo masses, no RNG.
# Handy as a cheap prior check inside MCMC.

n_mean = model.mean_number_density(
    logMmin=12.5, sigma_logM=0.5, fmax=1.0,
    logMsat=13.5, logMcut=11.5,  alpha=1.0,
)
print("── Mean number density (no random draws) ────────────────────────────────")
print(f"  n_mean = {n_mean:.4e} (Mpc/h)^-3")
print()

# ── 6. Typical MCMC loop pattern ──────────────────────────────────────────────
#
# The model object is created once outside the loop. Each call to populate()
# reuses pre-allocated buffers — no heap allocation in the hot path.

import time

params_chain = [
    (12.3, 0.4, 0.95, 13.4, 11.3, 1.1),
    (12.5, 0.5, 1.00, 13.5, 11.5, 1.0),
    (12.7, 0.6, 0.90, 13.6, 11.7, 0.9),
]

print("── MCMC loop timing (3 steps) ───────────────────────────────────────────")
for i, (lMmin, sig, fmx, lMsat, lMcut, alp) in enumerate(params_chain):
    t0 = time.perf_counter()
    g = model.populate(logMmin=lMmin, sigma_logM=sig, fmax=fmx,
                       logMsat=lMsat, logMcut=lMcut, alpha=alp, seed=i)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  step {i}: Ngal={len(g['pos']):,}  ({dt:.1f} ms)")
