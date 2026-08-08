"""URDF spawner that makes nested collision shapes authorable before cloning.

Isaac Sim 5.1's URDF importer places each ``collisions`` subtree behind an
instance prim even when ``UrdfFileCfg.make_instanceable`` is false.  Isaac
Lab's nested schema helper intentionally skips instance prims, so a normal
``collision_props`` override never reaches those shapes.  This task-scoped
spawner de-instances collision subtrees on the source environment, reapplies
the requested collision properties, and only then lets Isaac Lab clone it.
"""

from __future__ import annotations

from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import clone


@clone
def spawn_urdf_with_authorable_collisions(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn one URDF source, author collider offsets, then clone it."""

    # Bypass the standard function's clone wrapper: this function owns cloning
    # so the source collision properties are authored before replicas exist.
    prim = from_files.spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    pending = [prim]
    while pending:
        child = pending.pop(0)
        path = child.GetPath().pathString
        if child.IsInstance():
            if "/collisions" in path:
                child.SetInstanceable(False)
                pending.extend(child.GetChildren())
            # Visual and other instances do not need physics overrides.
            continue
        pending.extend(child.GetChildren())

    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props, stage=prim.GetStage())
    return prim
