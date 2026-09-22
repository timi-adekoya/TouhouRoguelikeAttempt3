from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class BossPhase:
    """At or below `hp_fraction` remaining HP, the boss starts favoring
    `skill_id` over its normal attacks (once triggered, permanent for the
    rest of the fight, not a one-shot)."""

    hp_fraction: float
    skill_id: str


@dataclass
class BossDef:
    """A location's boss: its own statblock (not the bestiary pattern's
    generic scaling), a base skill kit, and phase thresholds that escalate
    to spellcard-tier skills. Recruitable — permanently, after the first
    defeat — the same way any other Character-bearing NPC is.

    The base_* / attributes fields are the *boss fight* statblock only —
    deliberately decoupled from recruit_* (what they get as a party member).
    A boss can be a genuine threat in the fight without that becoming a
    broken permanent party member; recruiting resets them to a normal
    character baseline (species level matched to the party, no classes,
    no boss-only skills/spellcards)."""

    boss_id: str
    name: str
    description: str = ""
    species_id: str = ""
    innate_id: Optional[str] = None
    min_depth: int = 1
    base_hp: int = 50
    base_mp: int = 20
    base_sp: int = 20
    attributes: Dict[str, int] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    phases: List[BossPhase] = field(default_factory=list)
    recruit_hp: int = 28
    recruit_mp: int = 15
    recruit_sp: int = 18
    loot_gold_range: Tuple[int, int] = (0, 0)
    char: str = "B"
    color: Tuple[int, int, int] = (220, 60, 200)
    # Extra flat toughness multiplier applied to max HP on top of ordinary
    # species growth/attribute investment (see
    # Engine._finalize_boss_progression) — a boss still needs to be a
    # credible solo threat against a multi-person party, not just "a
    # same-level regular character."
    hp_multiplier: float = 2.0


@dataclass
class MonsterDef:
    """A regular (non-boss) monster: same idea as `BossDef` but for the
    fodder that populates dungeon/location floors — its own statblock,
    starting weapon (spawned pre-equipped, same as a player finding gear),
    AI tier by name (looked up in `bestiary.AI_CONTROLLERS`), and loot."""

    monster_id: str
    name: str
    char: str
    color: Tuple[int, int, int]
    species_id: str
    ai: str = "fsm"  # key into bestiary.AI_CONTROLLERS
    base_hp: int = 10
    base_mp: int = 0
    base_sp: int = 5
    attributes: Dict[str, int] = field(default_factory=dict)
    weapon: Optional[str] = None  # item_id, spawned pre-equipped in mainhand
    innate_id: Optional[str] = None  # e.g. an elemental fairy variant's shallow innate tree
    loot_gold_range: Tuple[int, int] = (0, 0)
    loot_drops: List[Tuple[str, float]] = field(default_factory=list)
    # Relative weight for the tutorial floors' uniform-random monster pool
    # (see bestiary.spawn_random_monster) — 1.0 is the default/plain rate.
    # Location floors are unaffected (they roll only from that location's
    # own enemy_species list).
    spawn_weight: float = 1.0


@dataclass
class LocationDef:
    """A stage of the dungeon past the tutorial: its own enemy roster, floor
    count range (scaled by how deep you are when you pick it), and boss.

    `generator` picks the non-boss floor layout ("dungeon" rooms-and-corridors,
    "cave" cellular-automata caverns, "tunnels" wide winding tunnels).
    `boss_arena` picks the boss floor's shape ("open" the default single big
    room, "bridge" a long wide strip).

    `boss_id` is a single fixed boss; `boss_ids`, if non-empty, means the
    location has several possible bosses and one is picked at random each
    time its boss floor is generated (takes priority over `boss_id`) — e.g.
    Kisume and Yamame both living in the Fantastic Blowhole."""

    location_id: str
    name: str
    description: str = ""
    min_depth: int = 1
    floor_range: Tuple[int, int] = (5, 8)
    enemy_species: List[str] = field(default_factory=list)
    boss_id: str = ""
    boss_ids: List[str] = field(default_factory=list)
    generator: str = "dungeon"
    boss_arena: str = "open"
    # Optional map-size override for this location's floors (both regular
    # and boss). None means "use the default viewport size" (every location
    # until Fantastic Blowhole) — the camera only needs to scroll for
    # locations that opt into a bigger map (see Engine.update_camera).
    map_width: Optional[int] = None
    map_height: Optional[int] = None


_locations: Dict[str, LocationDef] = {}
_bosses: Dict[str, BossDef] = {}
_monsters: Dict[str, MonsterDef] = {}
_loaded = False


def _read(name: str) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_all(force: bool = False) -> None:
    global _loaded
    if _loaded and not force:
        return

    _locations.clear()
    _bosses.clear()
    _monsters.clear()

    for raw in _read("locations.json"):
        _locations[raw["location_id"]] = LocationDef(
            location_id=raw["location_id"],
            name=raw["name"],
            description=raw.get("description", ""),
            min_depth=raw.get("min_depth", 1),
            floor_range=tuple(raw.get("floor_range", [5, 8])),
            enemy_species=raw.get("enemy_species", []),
            boss_id=raw.get("boss_id", ""),
            boss_ids=raw.get("boss_ids", []),
            generator=raw.get("generator", "dungeon"),
            boss_arena=raw.get("boss_arena", "open"),
            map_width=raw.get("map_width"),
            map_height=raw.get("map_height"),
        )

    for raw in _read("bosses.json"):
        _bosses[raw["boss_id"]] = BossDef(
            boss_id=raw["boss_id"],
            name=raw["name"],
            description=raw.get("description", ""),
            species_id=raw.get("species_id", ""),
            innate_id=raw.get("innate_id"),
            min_depth=raw.get("min_depth", 1),
            base_hp=raw.get("base_hp", 50),
            base_mp=raw.get("base_mp", 20),
            base_sp=raw.get("base_sp", 20),
            attributes=raw.get("attributes", {}),
            skills=raw.get("skills", []),
            phases=[
                BossPhase(hp_fraction=p["hp_fraction"], skill_id=p["skill"])
                for p in raw.get("phases", [])
            ],
            recruit_hp=raw.get("recruit_hp", 28),
            recruit_mp=raw.get("recruit_mp", 15),
            recruit_sp=raw.get("recruit_sp", 18),
            loot_gold_range=tuple(raw.get("loot_gold_range", [0, 0])),
            char=raw.get("char", "B"),
            color=tuple(raw.get("color", [220, 60, 200])),
            hp_multiplier=raw.get("hp_multiplier", 2.0),
        )

    for raw in _read("monsters.json"):
        _monsters[raw["monster_id"]] = MonsterDef(
            monster_id=raw["monster_id"],
            name=raw["name"],
            char=raw["char"],
            color=tuple(raw["color"]),
            species_id=raw.get("species_id", raw["monster_id"]),
            ai=raw.get("ai", "fsm"),
            base_hp=raw.get("base_hp", 10),
            base_mp=raw.get("base_mp", 0),
            base_sp=raw.get("base_sp", 5),
            attributes=raw.get("attributes", {}),
            weapon=raw.get("weapon"),
            innate_id=raw.get("innate_id"),
            loot_gold_range=tuple(raw.get("loot_gold_range", [0, 0])),
            loot_drops=[tuple(d) for d in raw.get("loot_drops", [])],
            spawn_weight=raw.get("spawn_weight", 1.0),
        )

    _loaded = True


def location(location_id: str) -> LocationDef:
    load_all()
    return _locations[location_id]


def boss(boss_id: str) -> BossDef:
    load_all()
    return _bosses[boss_id]


def monster(monster_id: str) -> MonsterDef:
    load_all()
    return _monsters[monster_id]


def all_locations() -> List[LocationDef]:
    load_all()
    return list(_locations.values())


def all_monsters() -> List[MonsterDef]:
    load_all()
    return list(_monsters.values())
