import numpy as np

graphic_dt = np.dtype(
    [
        ("ch", np.int32),
        ("fg", "3B"),
        ("bg", "3B"),
    ]
)

tile_dt = np.dtype(
    [
        ("walkable", bool),
        ("transparent", bool),
        ("dark", graphic_dt),  # Remembered but not currently visible.
        ("light", graphic_dt),  # Currently visible (lit).
    ]
)


def new_tile(*, walkable: int, transparent: int, dark, light) -> np.ndarray:
    return np.array((walkable, transparent, dark, light), dtype=tile_dt)


floor = new_tile(
    walkable=True,
    transparent=True,
    dark=(ord(" "), (255, 255, 255), (20, 20, 60)),
    light=(ord(" "), (255, 255, 255), (100, 100, 60)),
)
wall = new_tile(
    walkable=False,
    transparent=False,
    dark=(ord(" "), (255, 255, 255), (0, 0, 40)),
    light=(ord(" "), (255, 255, 255), (60, 60, 20)),
)
door_closed = new_tile(
    walkable=True,
    transparent=False,
    dark=(ord("+"), (110, 70, 40), (20, 20, 60)),
    light=(ord("+"), (200, 140, 70), (100, 100, 60)),
)
door_open = new_tile(
    walkable=True,
    transparent=True,
    dark=(ord("'"), (110, 70, 40), (20, 20, 60)),
    light=(ord("'"), (200, 140, 70), (100, 100, 60)),
)
stairs_down = new_tile(
    walkable=True,
    transparent=True,
    dark=(ord(">"), (180, 180, 180), (20, 20, 60)),
    light=(ord(">"), (255, 255, 255), (100, 100, 60)),
)
stairs_up = new_tile(
    walkable=True,
    transparent=True,
    dark=(ord("<"), (180, 180, 180), (20, 20, 60)),
    light=(ord("<"), (255, 255, 255), (100, 100, 60)),
)
#: A gap to the next location — appears at the end of a stage (tutorial
#: floors, or a location once its boss is defeated). Distinct from ordinary
#: stairs since using it opens the location-choice menu instead of a floor.
gap = new_tile(
    walkable=True,
    transparent=True,
    dark=(ord("%"), (120, 60, 160), (20, 20, 60)),
    light=(ord("%"), (200, 120, 255), (100, 100, 60)),
)

#: Fully unexplored tile, drawn as blank space.
SHROUD = np.array((ord(" "), (255, 255, 255), (0, 0, 0)), dtype=graphic_dt)
