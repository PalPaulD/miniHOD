"""
Build script for miniHOD.

Compiles src/hod.c into a shared library and places it inside the
miniHOD package directory so ctypes can find it after pip install.
"""
import os
import shutil
import platform
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_compiler():
    """Return (cc, omp_flag) — tries gcc with OpenMP, falls back to cc without."""
    candidates = []
    if platform.system() == "Darwin":
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


def _compile_libhod(output_dir):
    """Compile src/hod.c into output_dir/libhod.{so,dylib}."""
    src = os.path.join(_HERE, "src", "hod.c")
    os.makedirs(output_dir, exist_ok=True)

    if platform.system() == "Darwin":
        ext, shflag = "dylib", "-dynamiclib"
    else:
        ext, shflag = "so", "-shared"

    out = os.path.join(output_dir, f"libhod.{ext}")
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


class build_py(_build_py):
    def run(self):
        # Compile into the build directory (gets packaged into the wheel)
        _compile_libhod(os.path.join(self.build_lib, "miniHOD"))
        super().run()


class develop(_develop):
    def run(self):
        # Compile into the source tree (used directly by editable installs)
        _compile_libhod(os.path.join(_HERE, "miniHOD"))
        super().run()


class editable_wheel(_editable_wheel):
    def run(self):
        # Compile into the source tree (used directly by editable installs)
        _compile_libhod(os.path.join(_HERE, "miniHOD"))
        super().run()


setup(cmdclass={
    "build_py": build_py,
    "develop": develop,
    "editable_wheel": editable_wheel,
})
