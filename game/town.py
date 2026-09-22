from __future__ import annotations

from typing import Tuple

from game import tile_types, ui
from game.game_map import GameMap

WIDTH = 80
HEIGHT = 60 - ui.HUD_HEIGHT

# Shrine building footprint (perimeter walls), with a door gap in the south wall.
_BUILDING_X0, _BUILDING_X1 = 14, 25
_BUILDING_Y0, _BUILDING_Y1 = 3, 8
_DOOR_X0, _DOOR_X1 = 19, 20

PLAYER_START: Tuple[int, int] = (20, 12)
MARISA_START: Tuple[int, int] = (19, 5)
RINNOSUKE_START: Tuple[int, int] = (22, 12)
DUNGEON_ENTRANCE: Tuple[int, int] = (40, 30)


def build_hakurei_shrine() -> GameMap:
    """Build the starting overworld map: shrine grounds + shrine building."""
    game_map = GameMap(WIDTH, HEIGHT, is_overworld=True)

    for x in range(WIDTH):
        for y in range(HEIGHT):
            on_border = x in (0, WIDTH - 1) or y in (0, HEIGHT - 1)
            game_map.tiles[x, y] = tile_types.wall if on_border else tile_types.floor

    for x in range(_BUILDING_X0, _BUILDING_X1 + 1):
        game_map.tiles[x, _BUILDING_Y0] = tile_types.wall
        game_map.tiles[x, _BUILDING_Y1] = tile_types.wall
    for y in range(_BUILDING_Y0, _BUILDING_Y1 + 1):
        game_map.tiles[_BUILDING_X0, y] = tile_types.wall
        game_map.tiles[_BUILDING_X1, y] = tile_types.wall

    for x in range(_DOOR_X0, _DOOR_X1 + 1):
        game_map.tiles[x, _BUILDING_Y1] = tile_types.floor

    game_map.tiles[DUNGEON_ENTRANCE] = tile_types.stairs_down
    game_map.stairs_down = DUNGEON_ENTRANCE

    return game_map
