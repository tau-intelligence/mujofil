"""Backward-compatibility shim.

The package was renamed from ``mujofil-warp`` to ``mujofil`` in 0.2.0. This thin
module re-exports everything from :mod:`mujofil` so that any code written against
the old ``import mujofil_warp`` API keeps working. Prefer ``import mujofil``.
"""
from __future__ import annotations

import sys
import warnings

import mujofil as _mujofil

warnings.warn(
    "`mujofil_warp` was renamed to `mujofil` in 0.2.0; import `mujofil` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Make `mujofil_warp` and `mujofil` the SAME module object so attribute access,
# submodules (mujofil_warp.tools) and `from mujofil_warp import X` all resolve to
# the real package.
sys.modules[__name__] = _mujofil
