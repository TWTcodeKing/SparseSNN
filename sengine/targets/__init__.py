"""Target profiles (RTX 4090 / A100 / Jetson AGX Orin) and the active-profile registry.

    from sengine.targets import get_target, set_active_target, active_target
    profile = get_target('orin')      # explicit
    profile = get_target()            # env SENGINE_TARGET, else auto-detect

`sengine.build(..., target=...)` calls `set_active_target()`; every tuning
module then reads `active_target()`. Before any build the active profile is
auto-detected on first use.
"""

from __future__ import annotations

import os

from sengine.targets.base import TargetProfile
from sengine.targets.ada import ADA
from sengine.targets.a100 import A100
from sengine.targets.orin import ORIN

PROFILES: dict[str, TargetProfile] = {'ada': ADA, 'a100': A100, 'orin': ORIN}
TARGET_NAMES = ('auto',) + tuple(PROFILES)

_ACTIVE: TargetProfile | None = None


def detect_target() -> TargetProfile:
    """Pick a profile from the current GPU (name / arch / shared memory)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ADA
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        name = props.name
        if 'Orin' in name or (props.major, props.minor) == (8, 7):
            return ORIN
        smem = getattr(props, 'max_shared_memory_per_block_optin', 0) or 0
        if (props.major, props.minor) == (8, 0) or smem >= 160 * 1024 or 'A100' in name:
            return A100
        return ADA
    except Exception:
        return ADA


def get_target(name: str | None = None) -> TargetProfile:
    """Resolve a profile: explicit name > env SENGINE_TARGET > auto-detect."""
    if name is None or name == 'auto':
        name = os.environ.get('SENGINE_TARGET', 'auto')
    if name == 'auto':
        return detect_target()
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown sengine target '{name}'; choose from {list(PROFILES)}")


def set_active_target(profile: TargetProfile | str | None) -> TargetProfile:
    """Make `profile` (or a name / None for auto) the profile all tuning code reads."""
    global _ACTIVE
    _ACTIVE = profile if isinstance(profile, TargetProfile) else get_target(profile)
    return _ACTIVE


def active_target() -> TargetProfile:
    """The profile in effect (auto-detected on first use)."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = get_target(None)
    return _ACTIVE


__all__ = ['TargetProfile', 'PROFILES', 'TARGET_NAMES', 'ADA', 'A100', 'ORIN',
           'get_target', 'set_active_target', 'active_target', 'detect_target']
