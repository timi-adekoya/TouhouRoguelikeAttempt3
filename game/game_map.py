from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np
from tcod.console import Console

from game import tile_types, ui
from game.light_sources import DarknessSource, LightSource

DUNGEON_LIGHT_RADIUS = 8
OVERWORLD_LIGHT_RADIUS = 30

# The screen/console never resizes after startup (see main.py) — it's always
# exactly this many tiles of map viewport, regardless of how big any given
# floor's GameMap actually is. Bigger floors scroll under a camera instead of
# growing the console (see Engine.update_camera / GameMap.render's offset).
VIEWPORT_WIDTH = 80
VIEWPORT_HEIGHT = 60 - ui.HUD_HEIGHT


class GameMap:
    def __init__(self, width: int, height: int, is_overworld: bool = False):
        self.width = width
        self.height = height
        self.is_overworld = is_overworld
        self.tiles = np.full((width, height), fill_value=tile_types.wall, order="F")

        # Carve a single open room bordered by walls, as a starting point.
        self.tiles[1 : width - 1, 1 : height - 1] = tile_types.floor

        self.light_sources: List[LightSource] = []
        self.darkness_sources: List[DarknessSource] = []

        # Positions of closed doors, so a bump-to-open can find and open them.
        self.door_positions: Set[Tuple[int, int]] = set()
        self.stairs_down: Optional[Tuple[int, int]] = None
        self.stairs_up: Optional[Tuple[int, int]] = None
        # A gap to the next location/stage. Set instead of (never alongside)
        # stairs_down on a stage's final floor — sealed (None) on a location's
        # boss floor until the boss is defeated.
        self.gap: Optional[Tuple[int, int]] = None

        self.visible = np.full((width, height), fill_value=False, order="F")
        self.explored = np.full((width, height), fill_value=False, order="F")

    @property
    def ambient_light_radius(self) -> int:
        return OVERWORLD_LIGHT_RADIUS if self.is_overworld else DUNGEON_LIGHT_RADIUS

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and bool(self.tiles["walkable"][x, y])

    def render(self, console: Console, camera_x: int = 0, camera_y: int = 0) -> None:
        # max(0, ...) guards against a stale camera position from a bigger
        # map: a negative view_w/h here isn't just "nothing to draw" — used
        # as a slice stop it means "up to N from the end" in Python/numpy,
        # producing a wrongly-shaped array and a broadcast crash instead of
        # simply skipping the draw.
        view_w = max(0, min(VIEWPORT_WIDTH, self.width - camera_x))
        view_h = max(0, min(VIEWPORT_HEIGHT, self.height - camera_y))
        sl_x = slice(camera_x, camera_x + view_w)
        sl_y = slice(camera_y, camera_y + view_h)
        console.rgb[0:view_w, 0:view_h] = np.select(
            condlist=[self.visible[sl_x, sl_y], self.explored[sl_x, sl_y]],
            choicelist=[self.tiles["light"][sl_x, sl_y], self.tiles["dark"][sl_x, sl_y]],
            default=tile_types.SHROUD,
        )
