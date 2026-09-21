"""Location of the optional TVM interpreter used by the sengine_cpu TVM backend.

TVM is kept in its own virtualenv because tilelang (used by the GPU engine)
bundles a TVM build whose ``tvm_ffi`` conflicts with a standalone Apache TVM.
The CPU engine only talks to that interpreter through a subprocess
(``build/tvm_compiler.py``); compiled kernels are plain ``.so`` files with no
TVM runtime dependency.

Resolution order:
  1. ``$SENGINE_CPU_TVM_PYTHON`` (path to a python binary with ``tvm`` importable)
  2. ``<repo>/sengine_cpu/.tvm-env/bin/python`` (what ``env_setup.sh`` creates)

If neither exists, the native C conv backend is used (see ``optimizer.py``).
"""

from __future__ import annotations

import os
import sys

ENV_VAR = "SENGINE_CPU_TVM_PYTHON"

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TVM_PYTHON = os.path.join(_PKG_DIR, ".tvm-env", "bin", "python")


def tvm_python() -> str:
    """Path of the TVM interpreter (may not exist)."""
    return os.environ.get(ENV_VAR) or DEFAULT_TVM_PYTHON


def tvm_available() -> bool:
    return os.path.isfile(tvm_python())


def require_tvm_python() -> str:
    """Return the interpreter path or raise a clear error."""
    path = tvm_python()
    if not os.path.isfile(path):
        raise RuntimeError(
            f"sengine_cpu TVM backend requested but no TVM interpreter at '{path}'. "
            f"Run sengine_cpu/env_setup.sh (creates {DEFAULT_TVM_PYTHON}) or set "
            f"{ENV_VAR}=/path/to/python with Apache TVM installed. "
            f"Unset SENGINE_CPU_BACKEND (or set it to 'native') to use the native C backend.")
    return path


def inject_tvm_site_packages() -> None:
    """Make ``import tvm`` work in-process when the TVM venv exists.

    Only needed when a kernel module is imported from the main venv (e.g. for
    inspection). The compile subprocess already runs under the TVM interpreter.
    """
    venv = os.path.dirname(os.path.dirname(tvm_python()))
    lib = os.path.join(venv, "lib")
    if not os.path.isdir(lib):
        return
    for name in sorted(os.listdir(lib)):
        site = os.path.join(lib, name, "site-packages")
        if name.startswith("python") and os.path.isdir(site) and site not in sys.path:
            sys.path.insert(0, site)
            return
