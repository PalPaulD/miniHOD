"""
Build script for miniHOD.

Compiles src/hod.c into a shared library and places it inside the
miniHOD package directory so ctypes can find it after pip install.
"""
import os
import sys
import shutil
import platform
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


def _find_compiler():
    """Return (cc, omp_flag) — tries gcc with OpenMP, falls back to cc without."""
    candidates = []
    if platform.system() == "Darwin":
        # Homebrew GCC supports OpenMP; Apple clang does not
        for v in ("14", "13", "12", ""):
            name = f"gcc-{v}" if v else "gcc"
            candidates.append((name, "-fopenmp"))
    else:
        candidates.append(("gcc", "-fopenmp"))
    candidates.append(("cc", ""))  # fallback: no OpenMP

    for cc, omp in candidates:
        if shutil.which(cc) is not None:
            return cc, omp
    raise RuntimeError("No C compiler found. Install gcc.")


class build_py(_build_py):
    def run(self):
        self._compile_libhod()
        super().run()

    def _compile_libhod(self):
        src = os.path.join(os.path.dirname(__file__), "src", "hod.c")
        pkg_dir = os.path.join(self.build_lib, "miniHOD")
        os.makedirs(pkg_dir, exist_ok=True)

        if platform.system() == "Darwin":
            ext, shflag = "dylib", "-dynamiclib"
        else:
            ext, shflag = "so", "-shared"

        out = os.path.join(pkg_dir, f"libhod.{ext}")
        cc, omp = _find_compiler()

        cmd = [cc, "-O3", "-fPIC", "-std=c11", shflag,
               "-fno-math-errno", "-fno-trapping-math"]
        if omp:
            cmd.append(omp)
        cmd.extend(["-lm", "-o", out, src])
        if omp:
            cmd.append(omp)  # linker needs it too

        print(f"miniHOD: compiling {src} -> {out}")
        print(f"  {' '.join(cmd)}")
        subprocess.check_call(cmd)


setup(cmdclass={"build_py": build_py})
