"""Shared pytest setup.

The design goal for this suite is that ``pytest`` is green on a laptop with no GPU,
no Isaac Sim and no robot attached. Most tests achieve that by being pure-torch or by
reading source files as text; the rest declare what they need with a marker and get
skipped automatically here.

Two markers are available (declared in ``pyproject.toml``)::

    @pytest.mark.requires_isaac     # needs a working Isaac Lab / Isaac Sim install
    @pytest.mark.requires_hardware  # needs a physical LinkerHand on the CAN bus

``requires_hardware`` additionally refuses to run unless ``DEX_FORGE_HARDWARE=1`` is
exported. Moving a real hand is not something a test run should do by accident, so an
importable SDK is deliberately *not* enough to opt in.

Some older tests call ``pytest.skip()`` inline instead of using a marker. Both styles
work; prefer the markers for anything new.
"""

from __future__ import annotations

import importlib.util
import os

import pytest


def _isaac_available() -> bool:
    """Report whether Isaac Lab can be imported.

    Uses ``find_spec`` rather than a real import: importing ``isaaclab`` spins up a
    chunk of Omniverse and costs seconds, which is far too slow for collection time.
    """
    try:
        return importlib.util.find_spec("isaaclab") is not None
    except (ImportError, ValueError):
        # ValueError: a partially-initialised namespace package, seen when Isaac is
        # half-installed. Treat it as unavailable rather than exploding at collection.
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip marked tests whose dependency is missing, before anything executes."""
    skip_isaac = pytest.mark.skip(reason="Isaac Lab is not importable in this environment")
    skip_hardware = pytest.mark.skip(reason="hardware test; export DEX_FORGE_HARDWARE=1 to run")

    isaac_ok = _isaac_available()
    hardware_ok = os.environ.get("DEX_FORGE_HARDWARE") == "1"

    for item in items:
        if "requires_isaac" in item.keywords and not isaac_ok:
            item.add_marker(skip_isaac)
        if "requires_hardware" in item.keywords and not hardware_ok:
            item.add_marker(skip_hardware)
