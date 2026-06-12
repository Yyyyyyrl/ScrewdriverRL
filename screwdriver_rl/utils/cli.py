"""Dotted-path config overrides for dataclass/configclass-style objects.

We deliberately do not use Hydra: Isaac Lab scripts must parse CLI args and
boot the Omniverse app *before* importing anything heavy, which conflicts with
the ``@hydra.main`` decorator pattern. Instead every tunable lives on a config
object and can be overridden with repeated ``--cfg path.to.field=value`` args.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def _coerce(raw: str, current: Any) -> Any:
    """Coerce a raw CLI string to the type of the current config value."""
    if isinstance(current, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Cannot parse {raw!r} as bool")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (tuple, list)):
        parsed = json.loads(raw)
        return type(current)(parsed)
    if current is None:
        # No type information: try JSON, fall back to the raw string.
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


def apply_overrides(cfg: Any, overrides: list[str]) -> dict[str, Any]:
    """Apply ``path.to.field=value`` overrides onto a config object.

    Walks dotted attribute paths so both flat fields (``reward_turn_weight=500``)
    and nested ones (``dr.randomize_friction=false``) work. Raises on unknown
    attributes so typos fail loudly instead of silently training with defaults.

    Returns the applied {path: value} mapping for logging.
    """
    applied: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} is not of the form path.to.field=value")
        path, _, raw_value = override.partition("=")
        parts = path.strip().split(".")
        obj = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise AttributeError(f"Config has no attribute {part!r} (override {override!r})")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise AttributeError(f"Config has no attribute {leaf!r} (override {override!r})")
        value = _coerce(raw_value.strip(), getattr(obj, leaf))
        setattr(obj, leaf, value)
        applied[path.strip()] = value
    return applied


def set_dotted(cfg: Any, path: str, value: Any) -> bool:
    """Set an already-typed value at a dotted path. Returns False if missing."""
    parts = path.split(".")
    obj = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        return False
    setattr(obj, parts[-1], value)
    return True


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        # configclass instances and similar plain objects
        return {
            key: _to_jsonable(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def dump_config(output_dir: str | Path, **named_cfgs: Any) -> Path:
    """Dump fully-resolved config objects to <output_dir>/config.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: _to_jsonable(cfg) for name, cfg in named_cfgs.items()}
    path = output_dir / "config.json"
    path.write_text(json.dumps(payload, indent=2, default=repr))
    return path
