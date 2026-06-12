"""Hand registry. Add new hands here; task code stays unchanged.

NOTE: imports isaaclab indirectly — only import after app launch.
"""

from __future__ import annotations

from collections.abc import Callable

from .hand_spec import HandSpec


def _allegro_factory(**kwargs) -> HandSpec:
    from .allegro import allegro_hand_spec

    return allegro_hand_spec(**kwargs)


HAND_REGISTRY: dict[str, Callable[..., HandSpec]] = {
    "allegro": _allegro_factory,
}


def get_hand_spec(name: str, **kwargs) -> HandSpec:
    """Build the HandSpec registered under ``name``."""
    if name not in HAND_REGISTRY:
        raise KeyError(f"Unknown hand {name!r}. Registered hands: {sorted(HAND_REGISTRY)}")
    return HAND_REGISTRY[name](**kwargs)


__all__ = ["HandSpec", "HAND_REGISTRY", "get_hand_spec"]
