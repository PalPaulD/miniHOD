"""
ctypes ABI bindings for libhod.
All business logic lives in hod.py; this module is pure glue.
"""

import ctypes
from pathlib import Path

_DPTR = ctypes.POINTER(ctypes.c_double)
_U8PTR = ctypes.POINTER(ctypes.c_uint8)
_I64PTR = ctypes.POINTER(ctypes.c_int64)

# ── Library loading ────────────────────────────────────────────────────────────
_pkg = Path(__file__).parent          # miniHOD/ package dir (pip install)
_src = _pkg.parent / "src"            # src/ dir (development)

def _load_lib():
    for base in (_pkg, _src):
        for ext in ("dylib", "so"):
            p = base / f"libhod.{ext}"
            if p.exists():
                return ctypes.CDLL(str(p))
    raise OSError(
        "libhod shared library not found. "
        "Run `pip install .` or `make` in the repo root."
    )

lib = _load_lib()

# ── hod_populate ──────────────────────────────────────────────────────────────
lib.hod_populate.restype  = ctypes.c_int64
lib.hod_populate.argtypes = [
    _DPTR,              # halo_pos      (N,3)
    _DPTR,              # halo_vel      (N,3)
    _DPTR,              # halo_mass     (N,)
    _DPTR,              # halo_rvir     (N,)  R200m
    _DPTR,              # halo_conc     (N,)  c200m
    ctypes.c_int64,     # N_halos
    ctypes.c_double,    # logMmin
    ctypes.c_double,    # sigma_logM
    ctypes.c_double,    # fmax
    ctypes.c_double,    # logMsat
    ctypes.c_double,    # logMcut
    ctypes.c_double,    # alpha
    ctypes.c_double,    # box_size
    _DPTR,              # out_pos        (capacity,3)
    _DPTR,              # out_vel        (capacity,3)
    _U8PTR,             # out_is_central (capacity,)
    _I64PTR,            # out_halo_idx   (capacity,)
    ctypes.c_int64,     # out_capacity
    ctypes.c_uint64,    # seed
    ctypes.c_int,       # nthreads (0 = all available)
]

# ── hod_mean_number_density ───────────────────────────────────────────────────
lib.hod_mean_number_density.restype  = ctypes.c_double
lib.hod_mean_number_density.argtypes = [
    _DPTR,              # masses     (N,)
    ctypes.c_int64,     # N
    ctypes.c_double,    # logMmin
    ctypes.c_double,    # sigma_logM
    ctypes.c_double,    # fmax
    ctypes.c_double,    # logMsat
    ctypes.c_double,    # logMcut
    ctypes.c_double,    # alpha
    ctypes.c_double,    # box_volume
    ctypes.c_int,       # nthreads (0 = all available)
]

# ── hod_solve_logMmin ────────────────────────────────────────────────────
lib.hod_solve_logMmin.restype  = ctypes.c_double
lib.hod_solve_logMmin.argtypes = [
    _DPTR,              # masses      (N,)
    ctypes.c_int64,     # N
    ctypes.c_double,    # sigma_logM
    ctypes.c_double,    # fmax
    ctypes.c_double,    # logMsat
    ctypes.c_double,    # logMcut
    ctypes.c_double,    # alpha
    ctypes.c_double,    # target_nbar
    ctypes.c_double,    # box_volume
    ctypes.c_double,    # bracket_lo
    ctypes.c_double,    # bracket_hi
    ctypes.c_double,    # xtol
    ctypes.c_int,       # maxiter
    ctypes.c_int,       # nthreads (0 = all available)
]

# ── Pointer helpers ───────────────────────────────────────────────────────────
def dptr(arr):
    return arr.ctypes.data_as(_DPTR)

def u8ptr(arr):
    return arr.ctypes.data_as(_U8PTR)

def i64ptr(arr):
    return arr.ctypes.data_as(_I64PTR)
