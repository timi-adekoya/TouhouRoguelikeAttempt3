from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import tcod.map

from game.entity import Entity
from game.game_map import GameMap


def _fov_mask(transparency: np.ndarray, pov: Tuple[int, int], radius: int) -> np.ndarray:
    """radius <= 0 means unlimited range."""
    return tcod.map.compute_fov(transparency, pov=pov, radius=radius, light_walls=True)


def darkness_mask(game_map: GameMap) -> np.ndarray:
    """Tiles blocked entirely by darkness sources, regardless of light."""
    transparency = game_map.tiles["transparent"]
    mask = np.zeros_like(transparency, dtype=bool)
    for darkness in game_map.darkness_sources:
        mask |= _fov_mask(transparency, (darkness.x, darkness.y), radius=darkness.radius)
    return mask


def compute_visibility(
    game_map: GameMap,
    player: Entity,
    player_light_radius: int,
    other_entities: Iterable[Entity] = (),
) -> np.ndarray:
    """Compute the set of currently-visible tiles.

    The player always emits its own light. Other light sources (static
    map-placed ones, or entity-carried ones) only count once the player has
    line of sight to them. Darkness sources then block vision entirely
    within their radius, unless the player can see through darkness.

    Note: this does not affect `game_map.explored` (see `darkness_mask` and
    how `Engine.update_fov` uses it) so darkened tiles still show up with
    the normal remembered/dim look rather than pure unexplored shroud.
    """
    transparency = game_map.tiles["transparent"]
    player_pos = (player.x, player.y)

    # Unlimited-range LOS from the player, used only to gate whether remote
    # light sources are visible at all.
    player_los = _fov_mask(transparency, player_pos, radius=0)

    sources: List[Tuple[int, int, int]] = [(player.x, player.y, player_light_radius)]
    sources.extend(
        (light.x, light.y, light.radius) for light in game_map.light_sources
    )
    sources.extend(
        (entity.x, entity.y, entity.light_radius)
        for entity in other_entities
        if entity.light_radius is not None
    )

    visible = np.zeros_like(transparency, dtype=bool)
    for x, y, radius in sources:
        if (x, y) != player_pos and not player_los[x, y]:
            continue
        if radius <= 0:
            continue
        visible |= _fov_mask(transparency, (x, y), radius=radius)

    game_map.explored |= visible

    if not player.sees_through_darkness:
        visible &= ~darkness_mask(game_map)

    return visible
