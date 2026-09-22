from __future__ import annotations

import random
from typing import Iterator, List, Set, Tuple

import tcod.los

from game import tile_types
from game.game_map import GameMap


class Room:
    """Base class for a room shape. Subclasses set the bounding box
    (x1, y1, x2, y2, inclusive) and implement `carve`.

    Overlap checks use the bounding box for every shape, which is a bit
    conservative for circular/cross rooms but keeps room placement simple.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def intersects(self, other: "Room") -> bool:
        return (
            self.x1 <= other.x2
            and self.x2 >= other.x1
            and self.y1 <= other.y2
            and self.y2 >= other.y1
        )

    def carve(self, game_map: GameMap) -> None:
        raise NotImplementedError


class RectangularRoom(Room):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height

    @property
    def inner(self) -> Tuple[slice, slice]:
        """The room's floor area, excluding its walls."""
        return slice(self.x1 + 1, self.x2), slice(self.y1 + 1, self.y2)

    def carve(self, game_map: GameMap) -> None:
        game_map.tiles[self.inner] = tile_types.floor

    def wall_ring(self) -> Iterator[Tuple[int, int]]:
        for x in range(self.x1, self.x2 + 1):
            yield x, self.y1
            yield x, self.y2
        for y in range(self.y1 + 1, self.y2):
            yield self.x1, y
            yield self.x2, y


class CircularRoom(Room):
    def __init__(self, cx: int, cy: int, radius: int):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.x1 = cx - radius
        self.y1 = cy - radius
        self.x2 = cx + radius
        self.y2 = cy + radius

    def carve(self, game_map: GameMap) -> None:
        for x in range(self.x1 + 1, self.x2):
            for y in range(self.y1 + 1, self.y2):
                if (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius**2:
                    game_map.tiles[x, y] = tile_types.floor


class CrossRoom(Room):
    """A plus-shaped room: a horizontal arm and a vertical arm crossing
    through the center of the bounding box."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height

    def carve(self, game_map: GameMap) -> None:
        cx, cy = self.center
        arm_half_width = max(1, (self.x2 - self.x1) // 6)
        arm_half_height = max(1, (self.y2 - self.y1) // 6)
        game_map.tiles[self.x1 + 1 : self.x2, cy - arm_half_height : cy + arm_half_height + 1] = (
            tile_types.floor
        )
        game_map.tiles[cx - arm_half_width : cx + arm_half_width + 1, self.y1 + 1 : self.y2] = (
            tile_types.floor
        )


ROOM_SHAPE_WEIGHTS: List[Tuple[type, float]] = [
    (RectangularRoom, 0.6),
    (CircularRoom, 0.2),
    (CrossRoom, 0.2),
]


def _make_room(shape: type, x: int, y: int, size: int) -> Room:
    if shape is CircularRoom:
        return CircularRoom(cx=x + size // 2, cy=y + size // 2, radius=max(2, size // 2))
    return shape(x, y, size, size)


def _tunnel_between(start: Tuple[int, int], end: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
    """Yield an L-shaped tunnel between two points."""
    x1, y1 = start
    x2, y2 = end
    if random.random() < 0.5:
        corner_x, corner_y = x2, y1
    else:
        corner_x, corner_y = x1, y2

    for x, y in _line(x1, y1, corner_x, corner_y):
        yield x, y
    for x, y in _line(corner_x, corner_y, x2, y2):
        yield x, y


def _line(x1: int, y1: int, x2: int, y2: int) -> Iterator[Tuple[int, int]]:
    """A straight horizontal-then-vertical (or vice versa) line of points."""
    if x1 == x2:
        y_lo, y_hi = sorted((y1, y2))
        for y in range(y_lo, y_hi + 1):
            yield x1, y
    else:
        x_lo, x_hi = sorted((x1, x2))
        for x in range(x_lo, x_hi + 1):
            yield x, y1


def _place_doors(game_map: GameMap, rooms: List[Room]) -> None:
    """Turn any corridor tile that crosses a rectangular room's wall ring
    into a closed door. Circular/cross rooms are skipped since they have no
    fixed wall ring to test against.
    """
    for room in rooms:
        if not isinstance(room, RectangularRoom):
            continue
        for x, y in room.wall_ring():
            if game_map.tiles["walkable"][x, y]:
                game_map.tiles[x, y] = tile_types.door_closed
                game_map.door_positions.add((x, y))


def generate_dungeon(
    map_width: int,
    map_height: int,
    max_rooms: int,
    room_min_size: int,
    room_max_size: int,
) -> Tuple[GameMap, Tuple[int, int], List[Tuple[int, int]]]:
    """Generate a dungeon of varied-shape rooms connected by L-shaped
    corridors, with doors where corridors meet rectangular rooms, and a
    stairs-down in the last room placed.

    Returns the map, the (x, y) center of the first room (for spawning the
    player), and the centers of any "middle" rooms (neither first nor last)
    as candidate enemy spawn points.
    """
    game_map = GameMap(map_width, map_height, is_overworld=False)
    game_map.tiles[:, :] = tile_types.wall

    rooms: List[Room] = []

    for _ in range(max_rooms):
        shape = random.choices(
            [s for s, _ in ROOM_SHAPE_WEIGHTS], weights=[w for _, w in ROOM_SHAPE_WEIGHTS]
        )[0]
        size = random.randint(room_min_size, room_max_size)
        x = random.randint(1, map_width - size - 2)
        y = random.randint(1, map_height - size - 2)

        new_room = _make_room(shape, x, y, size)

        if any(new_room.intersects(other) for other in rooms):
            continue

        new_room.carve(game_map)

        if rooms:
            for tx, ty in _tunnel_between(rooms[-1].center, new_room.center):
                game_map.tiles[tx, ty] = tile_types.floor

        rooms.append(new_room)

    _place_doors(game_map, rooms)

    player_start = rooms[0].center if rooms else (map_width // 2, map_height // 2)
    if len(rooms) > 1:
        up_x, up_y = rooms[0].center
        game_map.tiles[up_x, up_y] = tile_types.stairs_up
        game_map.stairs_up = (up_x, up_y)

        down_x, down_y = rooms[-1].center
        game_map.tiles[down_x, down_y] = tile_types.stairs_down
        game_map.stairs_down = (down_x, down_y)

    enemy_spawns = [room.center for room in rooms[1:-1]]
    return game_map, player_start, enemy_spawns


# ---------------------------------------------------------------------------
# Cellular-automata caves — organic caverns instead of rooms-and-corridors.
# Used by locations flagged `generator: "cave"` (e.g. the Fantastic Blowhole).
# ---------------------------------------------------------------------------


def _ca_grid(width: int, height: int, wall_chance: float, iterations: int) -> List[List[int]]:
    """1 = wall, 0 = floor. Random-fill + repeated smoothing (a cell becomes
    a wall if >=5 of its 8 neighbors are walls) is the standard cave-CA
    recipe: it turns static into blobby, organic caverns."""
    grid = [
        [
            1 if x == 0 or y == 0 or x == width - 1 or y == height - 1 or random.random() < wall_chance else 0
            for y in range(height)
        ]
        for x in range(width)
    ]
    for _ in range(iterations):
        grid = _ca_smooth(grid, width, height)
    return grid


def _ca_smooth(grid: List[List[int]], width: int, height: int) -> List[List[int]]:
    new_grid = [[1] * height for _ in range(width)]
    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                continue
            wall_count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height or grid[nx][ny] == 1:
                        wall_count += 1
            new_grid[x][y] = 1 if wall_count >= 5 else 0
    return new_grid


def _largest_region(grid: List[List[int]], width: int, height: int) -> List[Tuple[int, int]]:
    """CA caves often fragment into several disconnected caverns — keep only
    the biggest connected floor region so the map is always fully walkable."""
    visited = [[False] * height for _ in range(width)]
    best: List[Tuple[int, int]] = []
    for x in range(width):
        for y in range(height):
            if grid[x][y] != 0 or visited[x][y]:
                continue
            region: List[Tuple[int, int]] = []
            stack = [(x, y)]
            visited[x][y] = True
            while stack:
                cx, cy = stack.pop()
                region.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny] and grid[nx][ny] == 0:
                        visited[nx][ny] = True
                        stack.append((nx, ny))
            if len(region) > len(best):
                best = region
    return best


def _carve_cave_blob(anchor: Tuple[int, int], blob_size: int, wall_chance: float, iterations: int) -> Set[Tuple[int, int]]:
    """One CA-cave 'room' generated in local coordinates and translated to
    world space around `anchor`. Retries locally (cheap — a small grid)
    if the roll produces too small a region to be usable."""
    region: List[Tuple[int, int]] = []
    for _ in range(20):
        grid = _ca_grid(blob_size, blob_size, wall_chance, iterations)
        region = _largest_region(grid, blob_size, blob_size)
        if len(region) >= max(20, blob_size * 2):
            break
    ox, oy = anchor[0] - blob_size // 2, anchor[1] - blob_size // 2
    return {(x + ox, y + oy) for x, y in region}


def generate_cave(
    map_width: int,
    map_height: int,
    wall_chance: float = 0.45,
    iterations: int = 5,
    num_spawns: int = 12,
    num_rooms: int = 5,
    blob_size: int = 16,
) -> Tuple[GameMap, Tuple[int, int], List[Tuple[int, int]]]:
    """A sequence of CA-cave 'rooms' (organic caverns, not boxy) connected
    by wide tunnels — replaces the old single-giant-cavern version, which
    read as too open/undifferentiated: no chokepoints meant the player's
    starting corner and the whole monster population both gravitated to
    the same spot every floor (that version also always placed the start
    at literally the top-left-most cell, deterministic every time). Room
    connection order — and so the start/end rooms — is shuffled fresh each
    floor. Retries the whole floor if room placement or connectivity
    doesn't work out (rare, and cheap to just try again)."""
    margin = blob_size
    anchors: List[Tuple[int, int]] = []
    attempts = 0
    while len(anchors) < num_rooms and attempts < 200:
        attempts += 1
        candidate = (
            random.randint(margin, map_width - margin - 1),
            random.randint(margin, map_height - margin - 1),
        )
        if all(abs(candidate[0] - a[0]) + abs(candidate[1] - a[1]) >= blob_size for a in anchors):
            anchors.append(candidate)
    if len(anchors) < 2:
        return generate_cave(map_width, map_height, wall_chance, iterations, num_spawns, num_rooms, blob_size)

    random.shuffle(anchors)  # connection order = which room is start/end, randomized each floor

    floor_cells: Set[Tuple[int, int]] = set()
    room_cells: List[Set[Tuple[int, int]]] = []
    for anchor in anchors:
        blob = {
            (x, y) for x, y in _carve_cave_blob(anchor, blob_size, wall_chance, iterations)
            if 1 <= x < map_width - 1 and 1 <= y < map_height - 1
        }
        room_cells.append(blob)
        floor_cells |= blob

    for a, b, anchor_a, anchor_b in zip(room_cells, room_cells[1:], anchors, anchors[1:]):
        if not a or not b:
            continue
        width = random.randint(3, 5)
        for x, y in _thick_line(anchor_a, anchor_b, width):
            if 1 <= x < map_width - 1 and 1 <= y < map_height - 1:
                floor_cells.add((x, y))

    game_map = GameMap(map_width, map_height, is_overworld=False)
    game_map.tiles[:, :] = tile_types.wall
    for x, y in floor_cells:
        game_map.tiles[x, y] = tile_types.floor

    if not room_cells[0] or not room_cells[-1]:
        return generate_cave(map_width, map_height, wall_chance, iterations, num_spawns, num_rooms, blob_size)
    start = random.choice(list(room_cells[0]))
    stairs_down = random.choice(list(room_cells[-1]))

    # Connectivity guard: tunnels are carved as straight thick lines between
    # room anchors, which almost always clips each room's own floor, but
    # isn't formally guaranteed — verify the whole floor is actually one
    # reachable region from `start` before committing to it.
    reachable = _largest_region(
        [[0 if (x, y) in floor_cells else 1 for y in range(map_height)] for x in range(map_width)],
        map_width, map_height,
    )
    if start not in reachable or len(reachable) < len(floor_cells):
        return generate_cave(map_width, map_height, wall_chance, iterations, num_spawns, num_rooms, blob_size)

    game_map.tiles[start] = tile_types.stairs_up
    game_map.stairs_up = start
    game_map.tiles[stairs_down] = tile_types.stairs_down
    game_map.stairs_down = stairs_down

    candidates = [p for p in floor_cells if p != start and p != stairs_down]
    random.shuffle(candidates)
    enemy_spawns = candidates[:num_spawns]
    return game_map, start, enemy_spawns


# ---------------------------------------------------------------------------
# Wide winding tunnels — distinct from both rooms-and-corridors and CA caves.
# Used by locations flagged `generator: "tunnels"` (e.g. the Deep Road to
# Hell): a chain of random waypoints connected by thick carved lines.
# ---------------------------------------------------------------------------


def _thick_line(a: Tuple[int, int], b: Tuple[int, int], width: int) -> Set[Tuple[int, int]]:
    radius = max(1, width // 2)
    cells: Set[Tuple[int, int]] = set()
    for x, y in tcod.los.bresenham(a, b).tolist():
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    cells.add((x + dx, y + dy))
    return cells


def generate_tunnels(
    map_width: int,
    map_height: int,
    min_segments: int = 6,
    max_segments: int = 9,
    min_segment_length: int = 6,
    max_segment_length: int = 12,
    min_width: int = 3,
    max_width: int = 5,
    num_spawns: int = 12,
) -> Tuple[GameMap, Tuple[int, int], List[Tuple[int, int]]]:
    """A 'snake' of wide, straight, axis-aligned tunnel segments turning at
    right angles — replaces the old free-floating diagonal-waypoint chain,
    which had nothing stopping waypoints landing close together or
    segments crossing each other, often producing a map that was really
    just a handful of short diagonal slashes. Each segment picks a random
    direction (never repeating the previous segment's axis, so it always
    genuinely turns) and length, clamped to stay in bounds. Start and end
    are wherever the snake happens to begin/finish — randomized fresh
    every floor, not fixed points. Retries the whole floor if a degenerate
    roll (e.g. repeated clamping at the map edge) produces too little
    floor to be usable."""
    margin = max_width + 1
    start = (random.randint(margin, map_width - margin - 1), random.randint(margin, map_height - margin - 1))
    pos = start
    floor_cells: Set[Tuple[int, int]] = set()

    last_axis = None
    segments = random.randint(min_segments, max_segments)
    for _ in range(segments):
        axis = random.choice([a for a in ("x", "y") if a != last_axis])
        last_axis = axis
        # Pick whichever direction actually has room to grow, rather than
        # rolling blind and clamping — a blind roll near an edge could
        # clamp to near-zero real progress, repeatedly, silently shrinking
        # the whole map (the exact "too short" complaint this rework is
        # for). Only truly boxed in (no room either way) skips a segment.
        if axis == "x":
            room_pos, room_neg = (map_width - margin - 1) - pos[0], pos[0] - margin
        else:
            room_pos, room_neg = (map_height - margin - 1) - pos[1], pos[1] - margin
        if room_pos <= 1 and room_neg <= 1:
            continue
        direction = 1 if room_neg <= 1 else (-1 if room_pos <= 1 else random.choice((1, -1)))
        available = room_pos if direction == 1 else room_neg
        length = min(random.randint(min_segment_length, max_segment_length), available)
        if length < 2:
            continue
        if axis == "x":
            end = (pos[0] + direction * length, pos[1])
        else:
            end = (pos[0], pos[1] + direction * length)
        width = random.randint(min_width, max_width)
        for x, y in _thick_line(pos, end, width):
            if 1 <= x < map_width - 1 and 1 <= y < map_height - 1:
                floor_cells.add((x, y))
        pos = end

    stairs_down = pos
    if stairs_down == start or len(floor_cells) < min_segments * min_segment_length * min_width:
        return generate_tunnels(
            map_width, map_height, min_segments, max_segments, min_segment_length,
            max_segment_length, min_width, max_width, num_spawns,
        )

    game_map = GameMap(map_width, map_height, is_overworld=False)
    game_map.tiles[:, :] = tile_types.wall
    for x, y in floor_cells:
        game_map.tiles[x, y] = tile_types.floor
    game_map.tiles[start] = tile_types.stairs_up
    game_map.stairs_up = start
    game_map.tiles[stairs_down] = tile_types.stairs_down
    game_map.stairs_down = stairs_down

    candidates = [p for p in floor_cells if p != start and p != stairs_down]
    random.shuffle(candidates)
    enemy_spawns = candidates[:num_spawns]
    return game_map, start, enemy_spawns


def generate_bridge_arena(
    map_width: int, map_height: int, bridge_height: int = 9
) -> Tuple[GameMap, Tuple[int, int], Tuple[int, int]]:
    """A boss arena shaped like a wide bridge: a long horizontal strip
    instead of the usual open room. Same signature/behavior as
    `generate_boss_arena` otherwise (stairs up at the entrance, no
    stairs-down — only the gap, once the boss falls)."""
    game_map = GameMap(map_width, map_height, is_overworld=False)
    margin = 3
    game_map.tiles[:, :] = tile_types.wall

    cy = map_height // 2
    y0 = max(margin, cy - bridge_height // 2)
    y1 = min(map_height - margin, cy + bridge_height // 2)
    game_map.tiles[margin : map_width - margin, y0:y1] = tile_types.floor

    start = (margin + 1, cy)
    boss_pos = (map_width - margin - 2, cy)

    game_map.tiles[start] = tile_types.stairs_up
    game_map.stairs_up = start

    return game_map, start, boss_pos


def generate_boss_arena(map_width: int, map_height: int) -> Tuple[GameMap, Tuple[int, int], Tuple[int, int]]:
    """A location's boss floor: one large open room, not the usual
    rooms-and-corridors dungeon — no fodder monsters, no loot, no
    stairs-down (the gap only appears once the boss falls). Stairs up sits
    at the entrance so the player can still retreat to the previous floor
    within the location."""
    game_map = GameMap(map_width, map_height, is_overworld=False)
    margin = 3
    game_map.tiles[:, :] = tile_types.wall
    game_map.tiles[margin : map_width - margin, margin : map_height - margin] = tile_types.floor

    start = (margin + 1, map_height // 2)
    boss_pos = (map_width - margin - 2, map_height // 2)

    game_map.tiles[start] = tile_types.stairs_up
    game_map.stairs_up = start

    return game_map, start, boss_pos
