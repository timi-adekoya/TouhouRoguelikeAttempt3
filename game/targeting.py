from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import tcod.los

from components.skills import ActivationTargetingDef
from game.entity import Entity

if TYPE_CHECKING:
    from game.engine import Engine


@dataclass
class TargetPoint:
    """The unifying target abstraction: always a position, optionally the
    entity standing there. Entity- and location-targeting share one pipeline."""

    x: int
    y: int
    entity: Optional[Entity] = None

    @property
    def pos(self) -> Tuple[int, int]:
        return self.x, self.y


def _entity_at(engine: "Engine", x: int, y: int) -> Optional[Entity]:
    return next((e for e in engine.entities if e.x == x and e.y == y and e.stats is not None), None)


def _is_ally_faction(entity: Entity) -> bool:
    """On the "characters" side iff actually recruited — a boss carries a
    Character component (so it can be recruited on defeat) but must still
    read as a monster to everyone until that actually happens."""
    return entity.character is not None and entity.character.recruited


def is_hostile_to(observer: Entity, other: Entity) -> bool:
    """Two factions: recruited characters (player + party) vs. "monsters"
    (everything else with stats, including not-yet-recruited bosses). Each
    is hostile to the other; nothing is hostile to its own faction.
    Placeholder until a real faction component exists, but relative to the
    observer rather than hardcoded to the player's perspective — a monster
    casting at the player needs "enemy" to mean the player, not "anything
    without a Character"."""
    if other is observer or other.stats is None:
        return False
    return _is_ally_faction(observer) != _is_ally_faction(other)


def _passes_restriction(caster: Entity, entity: Optional[Entity], restriction: str) -> bool:
    if restriction == "none":
        return True
    if entity is None:
        return False
    if restriction == "self":
        return entity is caster
    if restriction == "all":
        return True
    if restriction == "enemy":
        return is_hostile_to(caster, entity)
    if restriction == "ally":
        return entity is caster or not is_hostile_to(caster, entity)
    return False


def has_line_of_sight(engine: "Engine", x1: int, y1: int, x2: int, y2: int) -> bool:
    """True if every tile between the endpoints (exclusive) is transparent."""
    for x, y in tcod.los.bresenham((x1, y1), (x2, y2)).tolist()[1:-1]:
        if not engine.game_map.tiles["transparent"][x, y]:
            return False
    return True


def _shape_tiles(
    engine: "Engine", caster: Entity, targeting: ActivationTargetingDef, chosen: Tuple[int, int]
) -> List[Tuple[int, int]]:
    shape = targeting.range_shape
    params = targeting.shape_params

    if shape == "self":
        return [(caster.x, caster.y)]

    if shape == "point":
        return [chosen]

    if shape == "circle":
        radius = params.get("radius", 1)
        cx, cy = chosen
        tiles = []
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                if not engine.game_map.in_bounds(x, y):
                    continue
                if (x - cx) ** 2 + (y - cy) ** 2 > radius**2:
                    continue
                # AoE respects walls: tiles hidden from the blast center are spared.
                if not has_line_of_sight(engine, cx, cy, x, y):
                    continue
                tiles.append((x, y))
        return tiles

    if shape == "line":
        length = params.get("length", 5)
        line = tcod.los.bresenham((caster.x, caster.y), chosen).tolist()[1:]
        tiles = []
        for x, y in line[:length]:
            if not engine.game_map.tiles["transparent"][x, y]:
                break
            tiles.append((x, y))
        return tiles

    raise ValueError(f"Unknown range_shape: {shape!r}")


def resolve_targets(
    engine: "Engine",
    caster: Entity,
    targeting: ActivationTargetingDef,
    chosen: Optional[Tuple[int, int]] = None,
) -> List[TargetPoint]:
    """Shape geometry x restriction filtering -> TargetPoints.

    `chosen` is the player-selected tile for shapes that need one. Returns []
    when the selection is invalid (out of range / no LOS / restriction
    unsatisfied for an entity-kind target).
    """
    if chosen is None:
        chosen = (caster.x, caster.y)

    max_range = targeting.shape_params.get("range")
    if max_range is not None:
        dx, dy = chosen[0] - caster.x, chosen[1] - caster.y
        if dx * dx + dy * dy > max_range * max_range:
            return []
        if not has_line_of_sight(engine, caster.x, caster.y, *chosen):
            return []

    points = []
    for x, y in _shape_tiles(engine, caster, targeting, chosen):
        entity = _entity_at(engine, x, y)
        if targeting.target_kind == "entity":
            if _passes_restriction(caster, entity, targeting.restriction):
                points.append(TargetPoint(x, y, entity))
        else:  # location/direction targeting keeps the tile either way
            points.append(TargetPoint(x, y, entity))
    return points


def remaining_picks(
    engine: "Engine",
    caster: Entity,
    targeting: ActivationTargetingDef,
    picked: List[TargetPoint],
) -> List[TargetPoint]:
    """For `selection_mode: "multiple"`: every other legal single target the
    caster could still add (not already picked), nearest first."""
    taken = {id(tp.entity) for tp in picked}
    candidates = []
    for entity in engine.entities:
        if entity.stats is None or id(entity) in taken:
            continue
        resolved = resolve_targets(engine, caster, targeting, (entity.x, entity.y))
        if resolved and resolved[0].entity is entity:
            candidates.append(resolved[0])
    candidates.sort(key=lambda tp: (tp.x - caster.x) ** 2 + (tp.y - caster.y) ** 2)
    return candidates


def preview_tiles(
    engine: "Engine",
    caster: Entity,
    targeting: ActivationTargetingDef,
    chosen: Tuple[int, int],
) -> Tuple[bool, List[Tuple[int, int]]]:
    """For the targeting UI: the tiles a cast at `chosen` would touch and
    whether that tile is a legal selection. Unlike `resolve_targets`, this
    returns the full shape footprint (not restriction-filtered) so the player
    sees the whole blast even over empty tiles."""
    max_range = targeting.shape_params.get("range")
    if max_range is not None:
        dx, dy = chosen[0] - caster.x, chosen[1] - caster.y
        if dx * dx + dy * dy > max_range * max_range:
            return False, [chosen]
        if not has_line_of_sight(engine, caster.x, caster.y, *chosen):
            return False, [chosen]
    return True, _shape_tiles(engine, caster, targeting, chosen)


class TargetSelector:
    """A movable map reticle for choosing where an active skill or item use
    lands. Drives the targeting input mode; owns only the first activation
    stage for now (multi-stage skills like teleport-other reuse this in a
    loop later). Decoupled from skills vs. items via `on_confirm`, since both
    share the same ActivationTargetingDef/effect pipeline but have distinct
    cast entry points (cooldowns/cost vs. charges/consumption)."""

    def __init__(
        self,
        engine: "Engine",
        caster: Entity,
        stage: ActivationTargetingDef,
        on_confirm: Callable[[Tuple[int, int]], bool],
    ):
        self.caster = caster
        self.stage = stage
        self.on_confirm = on_confirm
        self.x, self.y = self._initial_target(engine)

    def _initial_target(self, engine: "Engine") -> Tuple[int, int]:
        """Start on the player's focus-fire mark if it's a legal pick here,
        else the nearest visible legal target, else the caster."""
        focus = getattr(engine, "focus_target", None)
        if (
            focus is not None
            and focus is not self.caster
            and focus.stats is not None
            and engine.game_map.visible[focus.x, focus.y]
            and _passes_restriction(self.caster, focus, self.stage.restriction)
        ):
            return focus.x, focus.y

        valid = [
            e
            for e in engine.entities
            if e is not self.caster
            and e.stats is not None
            and engine.game_map.visible[e.x, e.y]
            and _passes_restriction(self.caster, e, self.stage.restriction)
        ]
        if valid:
            nearest = min(
                valid,
                key=lambda e: (e.x - self.caster.x) ** 2 + (e.y - self.caster.y) ** 2,
            )
            return nearest.x, nearest.y
        return self.caster.x, self.caster.y

    def move(self, dx: int, dy: int, engine: "Engine") -> None:
        nx, ny = self.x + dx, self.y + dy
        if engine.game_map.in_bounds(nx, ny):
            self.x, self.y = nx, ny

    def preview(self, engine: "Engine") -> Tuple[bool, List[Tuple[int, int]]]:
        return preview_tiles(engine, self.caster, self.stage, (self.x, self.y))

    def confirm(self, engine: "Engine") -> bool:
        return self.on_confirm((self.x, self.y))
