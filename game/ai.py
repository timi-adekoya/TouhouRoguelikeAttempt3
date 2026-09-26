from __future__ import annotations

import random
import numpy as np
import tcod.los
import tcod.path
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from components.items import EquippableDef, UsableDef, WeaponDef, evaluate_formula
from components.skills import CATEGORY_ACTIVE, SkillInstance
from game import equipment, targeting
from game.actions import (
    Action,
    CastSkillAction,
    MeleeAttackAction,
    MovementAction,
    RangedAttackAction,
    SwapPlacesAction,
    UseItemAction,
    WaitAction,
)
from game.casting import can_pay
from game.entity import Entity
from game.targeting import TargetPoint

if TYPE_CHECKING:
    from game.engine import Engine

# ---------------------------------------------------------------------------
# Shared helpers: faction check, targeting, movement. Every tier of AI below
# (FSM, BT, Utility) is built out of these same primitives.
# ---------------------------------------------------------------------------


# A tile an ally is standing on is still walkable (pathfind_step will swap
# through it rather than detour), but weighted well above a plain floor
# tile's cost of 1 — otherwise A* treats "walk through my ally" and "walk
# around my ally" as equally good, so with 2+ followers all routing toward
# the same target/item they'd happily path straight into each other and
# swap back and forth forever instead of actually converging. This only
# needs to beat a short detour's cost, not the whole path.
ALLY_TILE_COST = 6


def chebyshev(a: Entity, b: Entity) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


# How far a hostile/ally can spot a target on its own, independent of what
# the player currently has in FOV — see `can_see`.
AGGRO_SIGHT_RADIUS = 8


def can_see(engine: "Engine", viewer: Entity, target: Entity) -> bool:
    """Whether `viewer` itself can spot `target` — its own range + line of
    sight, NOT `engine.game_map.visible` (only ever the player/currently
    -controlled character's FOV). Targeting used to gate on that single
    shared array, so a monster only aggroed onto whoever the ACTIVE
    character could see: switch which character you're controlling, or
    just be a party member off on your own, and monsters (and this
    character's own AI) stopped reacting to threats right next to them
    even in plain sight."""
    if chebyshev(viewer, target) > AGGRO_SIGHT_RADIUS:
        return False
    game_map = engine.game_map
    line = tcod.los.bresenham((viewer.x, viewer.y), (target.x, target.y)).tolist()
    return all(game_map.tiles["transparent"][x, y] for x, y in line[1:-1])


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def find_nearest_hostile(engine: "Engine", entity: Entity) -> Optional[Entity]:
    from game.effects import has_status  # deferred: avoids a module cycle with effects.py

    candidates = [
        e
        for e in engine.entities
        if targeting.is_hostile_to(entity, e)
        and can_see(engine, entity, e)
        and not has_status(engine, e, "stealthed")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: chebyshev(entity, e))


def _entity_blocking(engine: "Engine", mover: Entity, x: int, y: int) -> bool:
    return any(e is not mover and e.blocks_movement and e.x == x and e.y == y for e in engine.entities)


def find_adjacent_free_tile(engine: "Engine", around: Entity) -> Optional[Tuple[int, int]]:
    """A random open (walkable, unoccupied) tile next to `around` — used for
    "vanish and reappear beside the target" reposition skills, which just
    need any legal ambush spot rather than a player-chosen destination."""
    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x, y = around.x + dx, around.y + dy
            if engine.game_map.is_walkable(x, y) and not _entity_blocking(engine, around, x, y):
                candidates.append((x, y))
    if not candidates:
        return None
    return random.choice(candidates)


def step_toward(engine: "Engine", entity: Entity, tx: int, ty: int) -> Action:
    """Greedy move toward (tx, ty): try the direct diagonal, then each axis
    alone. No real pathfinding — good enough for "simple script" AI and as
    the shared movement primitive every tier uses."""
    dx, dy = _sign(tx - entity.x), _sign(ty - entity.y)
    for cdx, cdy in {(dx, dy), (dx, 0), (0, dy)}:
        if cdx == 0 and cdy == 0:
            continue
        nx, ny = entity.x + cdx, entity.y + cdy
        if engine.game_map.is_walkable(nx, ny) and not _entity_blocking(engine, entity, nx, ny):
            return MovementAction(entity, cdx, cdy)
    return WaitAction(entity)


def step_away(engine: "Engine", entity: Entity, tx: int, ty: int) -> Action:
    """Greedy move directly away from (tx, ty) — used for fleeing."""
    dx, dy = _sign(entity.x - tx), _sign(entity.y - ty)
    if dx == 0 and dy == 0:
        dx, dy = 1, 0  # already on top of the threat; pick an arbitrary direction
    return step_toward(engine, entity, entity.x + dx * 3, entity.y + dy * 3)


def find_frontier_tile(engine: "Engine", entity: Entity) -> Optional[Tuple[int, int]]:
    """Nearest (by real walking distance, not straight-line) walkable-but-
    unexplored tile reachable from `entity` — the destination auto-explore
    walks toward. Plain BFS over the map's static walkability (explored is
    just a player-knowledge overlay, not a separate walkability fact, so an
    unexplored tile can still be a legal pathfinding target) rather than
    A*-ing to every unexplored candidate, since we only need the nearest
    one, not a distance to each. Hostile-occupied tiles are treated as
    impassable — but only ones currently visible (see `compute_path`'s
    `only_visible_hostiles`), so an undiscovered monster in the fog can't
    silently seal off unexplored territory beyond it. None if nothing
    unexplored is reachable at all (exploration complete, or fully sealed
    off by something actually seen)."""
    from collections import deque

    game_map = engine.game_map
    hostile_tiles = {
        (e.x, e.y)
        for e in engine.entities
        if e is not entity
        and e.blocks_movement
        and targeting.is_hostile_to(entity, e)
        and game_map.visible[e.x, e.y]
    }
    start = (entity.x, entity.y)
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in seen or not game_map.is_walkable(nx, ny) or (nx, ny) in hostile_tiles:
                    continue
                if not game_map.explored[nx, ny]:
                    return nx, ny
                seen.add((nx, ny))
                queue.append((nx, ny))
    return None


def compute_path(
    engine: "Engine", entity: Entity, tx: int, ty: int, only_visible_hostiles: bool = False
) -> List[Tuple[int, int]]:
    """A* route from entity to (tx, ty), excluding the start tile. Hostile
    blocking entities are treated as walls; friendly ones are left walkable
    since `pathfind_step` will swap through them instead of detouring — with
    3+ party members, treating every ally as an obstacle caused long
    separating detours in tight corridors/rooms. Whatever's standing on the
    goal tile itself is also left alone, so a path can still be found up to
    it. Empty list means no route exists at all.

    `only_visible_hostiles`: only block on hostiles currently in
    `game_map.visible` (the player's FOV — the only FOV the engine tracks),
    rather than every hostile anywhere on the generated map regardless of
    whether they've ever been seen. Off by default (existing monster-AI
    behavior: a monster reasonably "knows" the whole map already, unlike the
    player). Auto-explore turns this on — without it, an undiscovered
    monster sitting in a fog-of-war corridor could silently seal off a
    route to unexplored territory the player hasn't even glimpsed yet,
    causing exploration to end early with map still left to see."""
    game_map = engine.game_map
    cost = np.array(game_map.tiles["walkable"], dtype=np.int8)
    for other in engine.entities:
        if other is entity or (other.x, other.y) == (tx, ty) or not other.blocks_movement:
            continue
        if targeting.is_hostile_to(entity, other):
            if only_visible_hostiles and not game_map.visible[other.x, other.y]:
                continue
            cost[other.x, other.y] = 0
        elif cost[other.x, other.y] > 0:
            cost[other.x, other.y] = ALLY_TILE_COST

    graph = tcod.path.SimpleGraph(cost=cost, cardinal=2, diagonal=3)
    pathfinder = tcod.path.Pathfinder(graph)
    pathfinder.add_root((entity.x, entity.y))
    return pathfinder.path_to((tx, ty))[1:].tolist()


def path_distance(engine: "Engine", entity: Entity, tx: int, ty: int) -> int:
    """Real walking distance (not straight-line chebyshev) — used so allies
    judge "how far is the player" by corridors actually walked, not raw
    tile distance, which is what let them fall behind rounding corners."""
    path = compute_path(engine, entity, tx, ty)
    return len(path) if path else 10_000


def pathfind_step(
    engine: "Engine", entity: Entity, tx: int, ty: int, stop_short: bool = True, only_visible_hostiles: bool = False
) -> Optional[Action]:
    """Real A* movement toward (tx, ty), routing around obstacles/other
    entities instead of the greedy step_toward's "try 3 directions" approach.
    `stop_short` leaves the mover one tile short of the goal (for chasing a
    living target you'll melee, or forming up near an ally) — set it False
    to actually walk onto the goal tile (e.g. to stand on a ground item).
    `only_visible_hostiles` — see `compute_path`."""
    path = compute_path(engine, entity, tx, ty, only_visible_hostiles=only_visible_hostiles)
    if stop_short:
        if len(path) <= 1:
            return None  # already adjacent, or no path at all
    elif not path:
        return None
    nx, ny = path[0]
    blocker = next(
        (e for e in engine.entities if e is not entity and e.blocks_movement and e.x == nx and e.y == ny),
        None,
    )
    if blocker is not None:
        return SwapPlacesAction(entity, blocker)
    return MovementAction(entity, nx - entity.x, ny - entity.y)


def move_toward(engine: "Engine", entity: Entity, tx: int, ty: int, stop_short: bool = True) -> Action:
    """Pathfind if possible, falling back to the greedy step for the (rare)
    case A* finds no route at all (e.g. fully boxed in)."""
    return pathfind_step(engine, entity, tx, ty, stop_short) or step_toward(engine, entity, tx, ty)


def find_ready_ranged_skill(
    entity: Entity, target: Entity, distance: int
) -> Optional[Tuple[SkillInstance, targeting.ActivationTargetingDef]]:
    """First active skill this entity could legally fire at `target` right
    now: off cooldown, affordable, and `target` within its first stage's
    range. Used by every AI tier to decide "cast" vs. "melee" vs. "chase"."""
    if entity.skill_book is None:
        return None
    for instance in entity.skill_book.by_category(CATEGORY_ACTIVE):
        skill = instance.resolved
        if not instance.ready or not can_pay(entity, skill.cost) or not skill.activation_targeting:
            continue
        stage = skill.activation_targeting[0]
        if stage.range_shape == "self":
            continue
        max_range = stage.shape_params.get("range", 1)
        if distance <= max_range:
            return instance, stage
    return None


def _item_ready(instance) -> bool:
    definition = instance.definition
    if definition.consumable:
        return instance.quantity > 0
    return instance.charges_remaining is None or instance.charges_remaining > 0


def find_usable_damage_item(
    entity: Entity, target: Entity, distance: int
) -> Optional[Tuple["ItemInstance", targeting.ActivationTargetingDef]]:
    """A ready offensive item (wand, scroll...) that reaches `target` right
    now — same idea as find_ready_ranged_skill, but for UsableDef items
    sitting unused in inventory (e.g. Wand of Sparks)."""
    if entity.inventory is None:
        return None
    for instance in entity.inventory.unequipped_items():
        definition = instance.definition
        if not isinstance(definition, UsableDef) or not definition.activation_targeting or not _item_ready(instance):
            continue
        stage = definition.activation_targeting[0]
        if stage.restriction != "enemy":
            continue
        if not any(e.type == "instant_damage_heal" and e.params.get("mode") == "damage" for e in definition.effects):
            continue
        max_range = stage.shape_params.get("range", 1)
        if distance <= max_range:
            return instance, stage
    return None


def choose_attack(engine: "Engine", entity: Entity, target: Entity) -> Optional[Action]:
    """Best available way to hurt `target` right now: a ready ranged skill,
    then a ready offensive item, then a ranged weapon shot, then melee if
    adjacent. None if nothing applies (out of range and nothing reaches)."""
    distance = chebyshev(entity, target)
    ranged = find_ready_ranged_skill(entity, target, distance)
    if ranged is not None:
        instance, stage = ranged
        targets = targeting.resolve_targets(engine, entity, stage, (target.x, target.y))
        if targets:
            if stage.selection_mode == "multiple" and stage.max_picks > 1:
                targets += targeting.remaining_picks(engine, entity, stage, targets)[: stage.max_picks - 1]
            return CastSkillAction(entity, instance, targets)
    item = find_usable_damage_item(entity, target, distance)
    if item is not None:
        instance, stage = item
        targets = targeting.resolve_targets(engine, entity, stage, (target.x, target.y))
        if targets:
            return UseItemAction(entity, instance, targets)
    weapon = equipment.equipped_weapon(entity)
    if weapon is not None and weapon.is_ranged and distance > 1 and equipment.has_ammo(entity, weapon):
        stage = equipment.ranged_attack_stage(weapon)
        if targeting.resolve_targets(engine, entity, stage, (target.x, target.y)):
            return RangedAttackAction(entity, target)
    if distance <= 1:
        return MeleeAttackAction(entity, target)
    return None


# ---------------------------------------------------------------------------
# Tier 1: dumb AI — a hand-scripted finite state machine for the weakest
# enemies (fairies). Explicit states, no lookahead, no memory beyond "state".
# ---------------------------------------------------------------------------


class FSMController:
    STATE_IDLE = "IDLE"
    STATE_CHASE = "CHASE"
    STATE_ATTACK = "ATTACK"

    def __init__(self):
        self.state = self.STATE_IDLE

    def decide(self, engine: "Engine", entity: Entity) -> Action:
        target = find_nearest_hostile(engine, entity)
        if target is None:
            self.state = self.STATE_IDLE
            return WaitAction(entity)

        attack = choose_attack(engine, entity, target)
        if attack is not None:
            self.state = self.STATE_ATTACK
            return attack

        self.state = self.STATE_CHASE
        return step_toward(engine, entity, target.x, target.y)


# ---------------------------------------------------------------------------
# Tier 2: moderate AI — a small generic behavior tree, used for mid-range
# enemies and most bosses. Demonstrates the fallback/composition a BT gives
# over the FSM above: a goblin will flee at low HP instead of always closing.
# ---------------------------------------------------------------------------


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()


class BTContext:
    def __init__(self, engine: "Engine", entity: Entity):
        self.engine = engine
        self.entity = entity
        self.target: Optional[Entity] = find_nearest_hostile(engine, entity)
        self.action: Optional[Action] = None


class Node:
    def tick(self, ctx: BTContext) -> Status:
        raise NotImplementedError


class Selector(Node):
    """Runs children in order; succeeds (and stops) at the first success."""

    def __init__(self, children: List[Node]):
        self.children = children

    def tick(self, ctx: BTContext) -> Status:
        for child in self.children:
            if child.tick(ctx) == Status.SUCCESS:
                return Status.SUCCESS
        return Status.FAILURE


class Sequence(Node):
    """Runs children in order; fails (and stops) at the first failure."""

    def __init__(self, children: List[Node]):
        self.children = children

    def tick(self, ctx: BTContext) -> Status:
        for child in self.children:
            if child.tick(ctx) == Status.FAILURE:
                return Status.FAILURE
        return Status.SUCCESS


class Condition(Node):
    def __init__(self, predicate: Callable[[BTContext], bool]):
        self.predicate = predicate

    def tick(self, ctx: BTContext) -> Status:
        return Status.SUCCESS if self.predicate(ctx) else Status.FAILURE


class ActionLeaf(Node):
    """Turns are atomic, so leaves always resolve to SUCCESS/FAILURE — no
    RUNNING/multi-tick actions yet."""

    def __init__(self, fn: Callable[[BTContext], Status]):
        self.fn = fn

    def tick(self, ctx: BTContext) -> Status:
        return self.fn(ctx)


def _low_hp(ctx: BTContext) -> bool:
    return ctx.entity.stats is not None and ctx.entity.stats.hp.current <= ctx.entity.stats.hp.max_value * 0.3


def _has_target(ctx: BTContext) -> bool:
    return ctx.target is not None


def _target_in_attack_range(ctx: BTContext) -> bool:
    return ctx.target is not None and choose_attack(ctx.engine, ctx.entity, ctx.target) is not None


def _do_flee(ctx: BTContext) -> Status:
    ctx.action = step_away(ctx.engine, ctx.entity, ctx.target.x, ctx.target.y)
    return Status.SUCCESS


def _do_attack(ctx: BTContext) -> Status:
    ctx.action = choose_attack(ctx.engine, ctx.entity, ctx.target)
    return Status.SUCCESS


def _do_chase(ctx: BTContext) -> Status:
    ctx.action = step_toward(ctx.engine, ctx.entity, ctx.target.x, ctx.target.y)
    return Status.SUCCESS


def _do_wait(ctx: BTContext) -> Status:
    ctx.action = WaitAction(ctx.entity)
    return Status.SUCCESS


def default_behavior_tree() -> Selector:
    """Flee at low HP, else attack if in range, else chase, else wait."""
    return Selector(
        [
            Sequence([Condition(_low_hp), Condition(_has_target), ActionLeaf(_do_flee)]),
            Sequence([Condition(_target_in_attack_range), ActionLeaf(_do_attack)]),
            Sequence([Condition(_has_target), ActionLeaf(_do_chase)]),
            ActionLeaf(_do_wait),
        ]
    )


class BehaviorTreeController:
    """Stateless tree shared across many entities; per-decision state lives
    entirely in the BTContext built fresh each call."""

    def __init__(self, root: Optional[Node] = None):
        self.root = root if root is not None else default_behavior_tree()

    def decide(self, engine: "Engine", entity: Entity) -> Action:
        ctx = BTContext(engine, entity)
        self.root.tick(ctx)
        return ctx.action if ctx.action is not None else WaitAction(entity)


class BossController:
    """Bosses reuse the moderate-tier's chase/attack/flee via `choose_attack`,
    but escalate to spellcard skills by HP phase: once HP drops to or below
    a phase's threshold, that phase's skill joins the boss's active kit for
    the rest of the fight (permanent, not a one-shot, and NOT replacing
    whatever phase skill(s) were already unlocked — a boss with a 70% and a
    40% phase keeps using both spellcards once she's past 40%, she doesn't
    forget the first one). Falls back to the normal attack/chase/flee
    behavior whenever no unlocked phase skill is ready, affordable, or in
    range."""

    def __init__(self, phases: List[Tuple[float, str]]):
        # Sorted ascending by threshold, so unlocking happens shallow-to-deep
        # and `decide` can try the deepest (presumably strongest) unlocked
        # skill first by walking the unlocked list in reverse.
        self.phases = sorted(phases, key=lambda p: p[0])
        self.unlocked_skill_ids: List[str] = []

    def decide(self, engine: "Engine", entity: Entity) -> Action:
        self._update_phase(engine, entity)
        target = find_nearest_hostile(engine, entity)
        if target is None:
            return WaitAction(entity)

        for skill_id in reversed(self.unlocked_skill_ids):
            phase_attack = self._try_phase_skill(engine, entity, target, skill_id)
            if phase_attack is not None:
                return phase_attack

        attack = choose_attack(engine, entity, target)
        if attack is not None:
            return attack

        # Bosses never blindly flee at low HP the way regular monsters do
        # (that's what derailed Parsee's fight when she ran out of MP —
        # she abandoned her spellcard pattern entirely) — always stand and
        # close distance instead. A genuine tactical retreat-to-heal is a
        # future addition, not this fallback.
        return step_toward(engine, entity, target.x, target.y)

    def _update_phase(self, engine: "Engine", entity: Entity) -> None:
        if entity.stats is None:
            return
        fraction = entity.stats.hp.current / entity.stats.hp.max_value
        for threshold, skill_id in self.phases:
            if fraction <= threshold and skill_id not in self.unlocked_skill_ids:
                self.unlocked_skill_ids.append(skill_id)
                engine.message_log.add_message(
                    f"{entity.name} shifts into a new attack pattern!",
                    stack=False,
                )

    def _try_phase_skill(
        self, engine: "Engine", entity: Entity, target: Entity, skill_id: str
    ) -> Optional[Action]:
        if entity.skill_book is None:
            return None
        instance = entity.skill_book.get(skill_id)
        if instance is None or not instance.ready:
            return None
        skill = instance.resolved
        if not can_pay(entity, skill.cost) or not skill.activation_targeting:
            return None
        stage = skill.activation_targeting[0]

        # A "reposition" skill (location-targeted teleport, e.g. an
        # ambush/vanish-and-reappear spellcard) doesn't aim at the target's
        # own tile — it aims at a free tile adjacent to them, so the AI
        # picks that itself rather than needing a dedicated targeting mode.
        is_reposition = stage.target_kind == "location" and any(
            getattr(effect, "movement_kind", None) == "teleport" for effect in skill.effects
        )
        if is_reposition:
            chosen = find_adjacent_free_tile(engine, target)
            if chosen is None:
                return None
        else:
            if chebyshev(entity, target) > stage.shape_params.get("range", 1):
                return None
            chosen = (target.x, target.y)

        targets = targeting.resolve_targets(engine, entity, stage, chosen)
        if not targets:
            return None
        return CastSkillAction(entity, instance, targets)


class _FakeCtx:
    """Lets BossController reuse the BT's `_low_hp` predicate without a full
    BTContext (which also computes a target via find_nearest_hostile)."""

    def __init__(self, entity: Entity):
        self.entity = entity


# ---------------------------------------------------------------------------
# Tier 3: Utility AI — player allies. Scores several candidate actions and
# takes the best one each turn; independent of the player but reactive to
# the battlefield. Squad coordination / commands / GOAP are future layers on
# top of this, not built yet.
# ---------------------------------------------------------------------------


def find_ally_needing_heal(engine: "Engine", entity: Entity, threshold: float = 0.6) -> Optional[Entity]:
    """Worst-off party member (including the player, excluding `entity`
    itself) below `threshold` HP fraction, or None if everyone's fine."""
    allies = [a for a in ([engine.player] + engine.party) if a is not entity and a.stats is not None]
    candidates = [a for a in allies if a.stats.hp.current < a.stats.hp.max_value * threshold]
    if not candidates:
        return None
    return min(candidates, key=lambda a: a.stats.hp.current / a.stats.hp.max_value)


def _heal_skills(entity: Entity):
    if entity.skill_book is None:
        return
    for instance in entity.skill_book.by_category(CATEGORY_ACTIVE):
        skill = instance.resolved
        if "healing" in skill.tags and skill.activation_targeting:
            yield instance, skill.activation_targeting[0]


def has_heal_for_others_skill(entity: Entity) -> bool:
    """Whether `entity` owns any healing skill that could ever target
    someone else — regardless of current cooldown/affordability. Used to
    decide whether it's worth approaching a hurt ally at all: a character
    who only has a self-only heal (First Aid) has nothing to offer them."""
    return any(stage.restriction != "self" for _, stage in _heal_skills(entity))


def find_ready_heal_for_others(entity: Entity) -> Optional[Tuple[SkillInstance, targeting.ActivationTargetingDef]]:
    for instance, stage in _heal_skills(entity):
        if stage.restriction == "self":
            continue
        skill = instance.resolved
        if not instance.ready or not can_pay(entity, skill.cost):
            continue
        return instance, stage
    return None


def find_ready_self_heal(entity: Entity) -> Optional[Tuple[SkillInstance, targeting.ActivationTargetingDef]]:
    """A ready healing skill this entity can cast on itself — "self"
    restriction obviously qualifies, and "ally" restriction legally includes
    the caster too (targeting._passes_restriction treats the caster as a
    valid ally target)."""
    for instance, stage in _heal_skills(entity):
        if stage.restriction not in ("self", "ally"):
            continue
        skill = instance.resolved
        if not instance.ready or not can_pay(entity, skill.cost):
            continue
        return instance, stage
    return None


def find_usable_heal_item(entity: Entity) -> Optional[Tuple["ItemInstance", Optional[targeting.ActivationTargetingDef]]]:
    """A ready self-healing item (potion...) — items never carry a "healing"
    tag in this data set, so this keys off the effect shape instead: a heal
    effect whose targeting (if any) allows self."""
    if entity.inventory is None:
        return None
    for instance in entity.inventory.unequipped_items():
        definition = instance.definition
        if not isinstance(definition, UsableDef) or not _item_ready(instance):
            continue
        stages = definition.activation_targeting
        stage = stages[0] if stages else None
        if stage is not None and stage.restriction not in ("self", "ally"):
            continue
        if not any(e.type == "instant_damage_heal" and e.params.get("mode") == "heal" for e in definition.effects):
            continue
        return instance, stage
    return None


def choose_heal_ally(engine: "Engine", entity: Entity) -> Tuple[Optional[Action], Optional[Entity]]:
    """(action, ally): `action` is a ready-to-perform heal cast if one's in
    range right now; `ally` is whoever's hurt worst, but ONLY set if `entity`
    actually owns a heal that can reach someone else — a First-Aid-only
    character has no business closing distance on a hurt teammate it can
    never do anything for."""
    ally = find_ally_needing_heal(engine, entity)
    if ally is None or not has_heal_for_others_skill(entity):
        return None, None

    ready = find_ready_heal_for_others(entity)
    if ready is None:
        return None, ally  # owns the kit, just not off cooldown/affordable yet
    instance, stage = ready

    distance = chebyshev(entity, ally)
    max_range = stage.shape_params.get("range", 0)
    if stage.range_shape != "self" and distance > max_range:
        return None, ally

    targets = targeting.resolve_targets(engine, entity, stage, (ally.x, ally.y))
    if not targets:
        return None, ally
    return CastSkillAction(entity, instance, targets), ally


def choose_self_heal(engine: "Engine", entity: Entity) -> Optional[Action]:
    """A skill or item this entity can use on itself right now — separate
    from choose_heal_ally since self-preservation shouldn't depend on owning
    a heal that reaches teammates."""
    ready = find_ready_self_heal(entity)
    if ready is not None:
        instance, stage = ready
        targets = targeting.resolve_targets(engine, entity, stage, (entity.x, entity.y))
        if targets:
            return CastSkillAction(entity, instance, targets)

    item = find_usable_heal_item(entity)
    if item is not None:
        instance, stage = item
        targets = (
            targeting.resolve_targets(engine, entity, stage, (entity.x, entity.y))
            if stage is not None
            else [TargetPoint(entity.x, entity.y, entity)]
        )
        if targets:
            return UseItemAction(entity, instance, targets)
    return None


def find_ready_buff_skill(entity: Entity) -> Optional[SkillInstance]:
    """A ready, affordable, self-targeted active skill that isn't a heal or
    a damage nuke — e.g. Bullseye's crit buff. Used to spend a turn buffing
    before/while fighting instead of the kit sitting unused."""
    if entity.skill_book is None:
        return None
    for instance in entity.skill_book.by_category(CATEGORY_ACTIVE):
        skill = instance.resolved
        if "healing" in skill.tags or not skill.activation_targeting:
            continue
        stage = skill.activation_targeting[0]
        if stage.restriction != "self":
            continue
        if not instance.ready or not can_pay(entity, skill.cost):
            continue
        if any(e.type == "instant_damage_heal" and e.params.get("mode") == "damage" for e in skill.effects):
            continue
        return instance
    return None


def use_self_skill(engine: "Engine", entity: Entity, instance: SkillInstance) -> Action:
    stage = instance.resolved.activation_targeting[0]
    targets = targeting.resolve_targets(engine, entity, stage, (entity.x, entity.y))
    return CastSkillAction(entity, instance, targets)


def _auto_equip_if_better(engine: "Engine", entity: Entity, instance) -> None:
    definition = instance.definition
    if not isinstance(definition, EquippableDef) or entity.stats is None:
        return
    if isinstance(definition, WeaponDef) and not equipment.has_ammo(entity, definition):
        return  # a bow with no arrows would leave the ally punching
    current = entity.inventory.equipped.get(definition.slot)
    if current is None:
        equipment.equip(engine, entity, instance)
        return
    current_def = current.definition
    if isinstance(definition, WeaponDef) and isinstance(current_def, WeaponDef):
        new_power = evaluate_formula(definition.damage_formula, entity.stats.attributes)
        cur_power = evaluate_formula(current_def.damage_formula, entity.stats.attributes)
        if new_power > cur_power:
            equipment.equip(engine, entity, instance)
    elif not isinstance(definition, WeaponDef) and not isinstance(current_def, WeaponDef):
        new_score = definition.defense + definition.magic_defense
        cur_score = current_def.defense + current_def.magic_defense
        if new_score > cur_score:
            equipment.equip(engine, entity, instance)


def try_auto_loot(engine: "Engine", entity: Entity) -> None:
    """Free (non-turn-costing) side effect, mirroring the player's own manual
    pickup key: grab whatever's underfoot and equip it if it's an upgrade.
    Without this, AI allies could never gear up mid-run on their own."""
    if entity.inventory is None:
        return
    ground = next(
        (e for e in engine.entities if e.ground_item is not None and e.x == entity.x and e.y == entity.y),
        None,
    )
    if ground is None:
        return
    instance = ground.ground_item
    if not equipment.pick_up(engine, entity, instance):
        return
    engine.remove_entity(ground)
    engine.message_log.add_message(
        f"{entity.name} picks up {instance.definition.name}.", stack=False
    )
    _auto_equip_if_better(engine, entity, instance)


def find_nearest_ground_item(engine: "Engine", entity: Entity, leash: int) -> Optional[Entity]:
    """Nearest loose item, restricted to within `leash` tiles of the player
    so allies don't wander off the map hunting for loot."""
    player = engine.player
    candidates = [
        e for e in engine.entities if e.ground_item is not None and chebyshev(player, e) <= leash
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: chebyshev(entity, e))


class UtilityAIController:
    FOLLOW_DISTANCE = 3
    LEASH_DISTANCE = 8  # won't chase a target/loot further than this from the player
    FLEE_HP_FRACTION = 0.1
    SELF_HEAL_HP_FRACTION = 0.6

    def _find_target(self, engine: "Engine", entity: Entity) -> Optional[Entity]:
        from game.effects import has_status  # deferred: avoids a module cycle with effects.py

        player = engine.player
        focus = engine.focus_target
        if (
            focus is not None
            and targeting.is_hostile_to(entity, focus)
            and engine.game_map.visible[focus.x, focus.y]
            and chebyshev(player, focus) <= self.LEASH_DISTANCE
            and not has_status(engine, focus, "stealthed")
        ):
            # Player-designated focus-fire target takes priority over
            # whichever hostile happens to be nearest.
            return focus

        candidates = [
            e
            for e in engine.entities
            if targeting.is_hostile_to(entity, e)
            and can_see(engine, entity, e)
            and chebyshev(player, e) <= self.LEASH_DISTANCE
            and not has_status(engine, e, "stealthed")
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: chebyshev(entity, e))

    def decide(self, engine: "Engine", entity: Entity) -> Action:
        try_auto_loot(engine, entity)

        heal_action, hurt_ally = choose_heal_ally(engine, entity)
        if heal_action is not None:
            return heal_action

        target = self._find_target(engine, entity)
        scored: List[Tuple[float, Action]] = []

        hp_frac = 1.0
        if entity.stats is not None:
            hp_frac = entity.stats.hp.current / entity.stats.hp.max_value

        if hp_frac <= self.SELF_HEAL_HP_FRACTION:
            self_heal = choose_self_heal(engine, entity)
            if self_heal is not None:
                scored.append((11.0, self_heal))

        if hurt_ally is not None:
            # Can see someone hurt and could actually help them, but can't
            # reach/cast the heal yet — close the distance instead of
            # ignoring them to go brawl.
            scored.append((9.0, move_toward(engine, entity, hurt_ally.x, hurt_ally.y)))

        if target is not None:
            if hp_frac <= self.FLEE_HP_FRACTION and find_ready_self_heal(entity) is None:
                scored.append((12.0, step_away(engine, entity, target.x, target.y)))

            attack = choose_attack(engine, entity, target)
            if attack is not None:
                scored.append((10.0 + (1.0 - hp_frac) * 2.0, attack))
            else:
                scored.append((6.0, move_toward(engine, entity, target.x, target.y)))

            buff = find_ready_buff_skill(entity)
            if buff is not None:
                scored.append((7.0, use_self_skill(engine, entity, buff)))
        else:
            # No visible threat and not already chasing anything — free to
            # go grab loot lying around, still on a leash to the player.
            loot = find_nearest_ground_item(engine, entity, self.LEASH_DISTANCE)
            if loot is not None:
                scored.append((5.0, move_toward(engine, entity, loot.x, loot.y, stop_short=False)))

            if path_distance(engine, entity, engine.player.x, engine.player.y) > self.FOLLOW_DISTANCE:
                scored.append((4.0, move_toward(engine, entity, engine.player.x, engine.player.y)))

        scored.append((1.0, WaitAction(entity)))
        scored.sort(key=lambda pair: -pair[0])
        return scored[0][1]
