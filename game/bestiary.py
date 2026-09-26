from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from components.character import Character
from components.inventory import Inventory, ItemInstance
from components.progression import Progression
from components.skills import SkillBook, SkillInstance
from components.stats import Attributes, StatBlock
from game import location_data
from game.ai import BehaviorTreeController, BossController, FSMController
from game.entity import Entity


@dataclass
class LootTable:
    """Rolled by whoever lands the killing blow: a gold range, plus a list
    of (item_id, chance) drops tied to that monster specifically (a goblin
    dropping its own sword, not a random grab-bag)."""

    gold_range: Tuple[int, int] = (0, 0)
    drops: List[Tuple[str, float]] = field(default_factory=list)


# AI controllers keyed by MonsterDef.ai. FSMController/BossController are
# cheap and stateless per-instance; BehaviorTreeController is stateless too
# (its per-decision state lives in a fresh BTContext each call), so a single
# shared instance is safe across every monster using this tier, regardless
# of species.
_BEHAVIOR_TREE = BehaviorTreeController()
AI_CONTROLLERS = {
    "fsm": lambda: FSMController(),
    "behavior_tree": lambda: _BEHAVIOR_TREE,
}


# Discrete weights by offset from the current floor, used to sample enemy
# level within a location's depth range — heavily favors the current floor,
# with a thin tail toward higher floors and a thinner one toward lower
# floors. Offsets that land outside the location's [min_depth, max_depth]
# range are dropped and the remainder renormalized (no reflecting the
# dropped weight elsewhere), so floors at the edge of the range naturally
# concentrate even harder on the levels that do exist.
LEVEL_OFFSET_WEIGHTS = {0: 10, -1: 3, 1: 3, -2: 1, 2: 1}


def sample_enemy_level(current_depth: int, min_depth: int, max_depth: int) -> int:
    """Picks an enemy level for a floor within a location's absolute depth
    range, weighted toward `current_depth` via LEVEL_OFFSET_WEIGHTS."""
    candidates = []
    weights = []
    for offset, weight in LEVEL_OFFSET_WEIGHTS.items():
        level = current_depth + offset
        if min_depth <= level <= max_depth:
            candidates.append(level)
            weights.append(weight)
    return random.choices(candidates, weights=weights, k=1)[0]


def spawn_monster(monster_def: "location_data.MonsterDef", x: int, y: int, level: int = 1) -> Entity:
    """Any regular (non-boss) monster, driven entirely by its MonsterDef —
    mirrors `spawn_boss`'s data-driven pattern instead of one hardcoded
    Python function per species."""
    attributes = Attributes()
    for attr_name, amount in monster_def.attributes.items():
        getattr(attributes, attr_name).modify(amount)
    stats = StatBlock.create(
        max_hp=monster_def.base_hp, max_mp=monster_def.base_mp, max_sp=monster_def.base_sp,
        attributes=attributes,
    )

    inventory = Inventory()
    if monster_def.weapon is not None:
        weapon = ItemInstance(monster_def.weapon)
        inventory.add(weapon)
        inventory.equipped["mainhand"] = weapon  # spawned pre-equipped; no engine/log needed

    progression = Progression(species_id=monster_def.species_id, innate_id=monster_def.innate_id)
    progression.species_level = level

    monster = Entity(
        x=x, y=y, char=monster_def.char, color=monster_def.color, name=monster_def.name,
        blocks_movement=True, stats=stats, skill_book=SkillBook(), inventory=inventory,
        progression=progression,
        loot=LootTable(gold_range=monster_def.loot_gold_range, drops=list(monster_def.loot_drops)),
    )
    controller_factory = AI_CONTROLLERS.get(monster_def.ai, AI_CONTROLLERS["fsm"])
    monster.ai = controller_factory()
    return monster


def spawn_random_monster(x: int, y: int, level: int = 1) -> Entity:
    monsters = location_data.all_monsters()
    monster_def = random.choices(monsters, weights=[m.spawn_weight for m in monsters])[0]
    return spawn_monster(monster_def, x, y, level=level)


def populate_dungeon_floor(spawn_points: List[Tuple[int, int]], level: int = 1) -> List[Entity]:
    return [spawn_random_monster(x, y, level=level) for x, y in spawn_points]


GENERIC_ROOM_LOOT = [
    "healing_potion", "iron_sword", "hunting_bow", "leather_cap", "leather_vest", "arrow_bundle", "wand_of_sparks",
]
ROOM_LOOT_CHANCE = 0.15


def scatter_room_loot(spawn_points: List[Tuple[int, int]]) -> List[Entity]:
    """A low, independent chance per candidate room to leave an item lying
    around — separate from monster spawns, so a room can have both, either,
    or neither."""
    ground_items = []
    for x, y in spawn_points:
        if random.random() < ROOM_LOOT_CHANCE:
            instance = ItemInstance(random.choice(GENERIC_ROOM_LOOT))
            ground_items.append(
                Entity(
                    x=x, y=y, char="!", color=(200, 200, 80), name=instance.definition.name,
                    ground_item=instance,
                )
            )
    return ground_items


def populate_location_floor(
    location: "location_data.LocationDef",
    spawn_points: List[Tuple[int, int]],
    current_depth: int,
    min_depth: int,
    max_depth: int,
) -> List[Entity]:
    """Same idea as populate_dungeon_floor, restricted to a location's own
    enemy roster (falls back to the full table if none of its species match
    a known monster). Each spawned monster independently rolls its level
    from the location's absolute depth range, weighted toward the current
    floor (see sample_enemy_level)."""
    monster_defs = [
        location_data.monster(species_id)
        for species_id in location.enemy_species
        if species_id in {m.monster_id for m in location_data.all_monsters()}
    ]
    if not monster_defs:
        monster_defs = location_data.all_monsters()
    return [
        spawn_monster(
            random.choice(monster_defs), x, y,
            level=sample_enemy_level(current_depth, min_depth, max_depth),
        )
        for x, y in spawn_points
    ]


def spawn_boss(boss_def: "location_data.BossDef", x: int, y: int) -> Entity:
    """A location's boss: a starting statblock (from here, scaled up to a
    real fight-worthy character by Engine._finalize_boss_progression —
    species growth, attribute investment, and an extra toughness multiplier
    all layered on afterward, same idea as any other level-scaled monster),
    a base skill kit plus every phase's spellcard granted up front (so
    they're ready the instant a phase triggers), and a BossController
    escalating through those phases by HP.

    `boss_def.attributes` is applied later (in the finalize step, via
    apply_stat_modifier) rather than here — doing it here would bypass the
    Vitality/Spirit -> HP/MP/SP linked bonus every other character gets from
    attribute investment."""
    stats = StatBlock.create(
        max_hp=boss_def.base_hp, max_mp=boss_def.base_mp, max_sp=boss_def.base_sp, attributes=Attributes()
    )

    book = SkillBook()
    for skill_id in boss_def.skills:
        book.add(SkillInstance(skill_id, source=f"innate:{boss_def.boss_id}"))
    for phase in boss_def.phases:
        if book.get(phase.skill_id) is None:
            book.add(SkillInstance(phase.skill_id, source=f"innate:{boss_def.boss_id}"))

    boss = Entity(
        x=x, y=y, char=boss_def.char, color=boss_def.color, name=boss_def.name,
        blocks_movement=True, stats=stats, skill_book=book, inventory=Inventory(),
        character=Character(name=boss_def.name, recruited=False),
        progression=Progression(species_id=boss_def.species_id, innate_id=boss_def.innate_id),
    )
    boss.ai = BossController([(p.hp_fraction, p.skill_id) for p in boss_def.phases])
    return boss
