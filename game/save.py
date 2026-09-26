from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import msgpack
import numpy as np

import components.inventory as inventory_module
from components.character import Character
from components.inventory import Inventory, ItemInstance
from components.progression import ClassProgress, Progression
from components.skills import SkillBook, SkillInstance
from components.stats import Attributes, Stat, StatBlock
from game import location_data, tile_types
from game.ai import BehaviorTreeController, BossController, FSMController, UtilityAIController
from game.casting import ChargeState
from game.effects import TimedEffectInstance
from game.entity import Entity
from game.game_map import GameMap
from game.light_sources import DarknessSource, LightSource
from game.targeting import TargetPoint

if TYPE_CHECKING:
    from game.engine import Engine

SAVE_VERSION = 2
SAVE_DIR = Path(__file__).resolve().parent.parent / "saves"
SLOT_COUNT = 3
_LEGACY_QUICKSAVE = "quicksave"

_CORE_ATTRIBUTES = ("power", "technique", "speed", "vitality", "spirit", "presence")


class SaveError(Exception):
    pass


@dataclass
class SlotInfo:
    slot: int
    exists: bool
    character: str = ""
    level: int = 0
    where: str = ""
    saved_at: float = 0.0
    legacy: bool = False

    @property
    def label(self) -> str:
        if not self.exists:
            return f"Slot {self.slot}: Empty"
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.saved_at))
        if self.legacy:
            return f"Slot {self.slot}: Old save ({stamp})"
        return f"Slot {self.slot}: {self.character} Lv{self.level} - {self.where} ({stamp})"


def _slot_path(slot: int) -> Path:
    return SAVE_DIR / f"slot_{slot}.msgpack"


def slot_exists(slot: int) -> bool:
    return _slot_path(slot).exists()


def slot_info(slot: int) -> SlotInfo:
    if not slot_exists(slot):
        return SlotInfo(slot=slot, exists=False)
    try:
        meta = read_slot(slot).get("meta")
    except Exception:
        return SlotInfo(slot=slot, exists=True, character="(unreadable)")
    if meta is None:
        return SlotInfo(
            slot=slot, exists=True, legacy=True, saved_at=_slot_path(slot).stat().st_mtime
        )
    return SlotInfo(
        slot=slot,
        exists=True,
        character=meta.get("character", "?"),
        level=meta.get("level", 0),
        where=meta.get("where", "?"),
        saved_at=meta.get("saved_at", 0.0),
    )


def all_slots() -> List[SlotInfo]:
    return [slot_info(s) for s in range(1, SLOT_COUNT + 1)]


def most_recent_slot() -> Optional[int]:
    existing = [info for info in all_slots() if info.exists]
    if not existing:
        return None
    return max(existing, key=lambda info: info.saved_at).slot


def migrate_legacy_quicksave() -> None:
    """Pre-slot builds wrote a single quicksave.msgpack; adopt it as slot 1."""
    legacy = SAVE_DIR / f"{_LEGACY_QUICKSAVE}.msgpack"
    if legacy.exists() and not slot_exists(1):
        legacy.rename(_slot_path(1))


# ---------------------------------------------------------------------------
# Component (de)serialization
# ---------------------------------------------------------------------------


def _dump_stat(stat: Stat) -> dict:
    return {"current": stat.current, "max": stat.max_value, "min": stat.min_value}


def _load_stat(data: dict) -> Stat:
    return Stat(max_value=data["max"], current=data["current"], min_value=data["min"])


def _dump_attributes(attrs: Attributes) -> dict:
    return {name: _dump_stat(getattr(attrs, name)) for name in _CORE_ATTRIBUTES}


def _load_attributes(data: dict) -> Attributes:
    attrs = Attributes()
    for name in _CORE_ATTRIBUTES:
        setattr(attrs, name, _load_stat(data[name]))
    return attrs


def _dump_stats(stats: StatBlock) -> dict:
    return {
        "hp": _dump_stat(stats.hp),
        "mp": _dump_stat(stats.mp),
        "sp": _dump_stat(stats.sp),
        "faith": _dump_stat(stats.faith) if stats.faith is not None else None,
        "attributes": _dump_attributes(stats.attributes),
        "modifiers": dict(stats.modifiers),
    }


def _load_stats(data: dict) -> StatBlock:
    return StatBlock(
        hp=_load_stat(data["hp"]),
        mp=_load_stat(data["mp"]),
        sp=_load_stat(data["sp"]),
        faith=_load_stat(data["faith"]) if data["faith"] is not None else None,
        attributes=_load_attributes(data["attributes"]),
        modifiers=dict(data["modifiers"]),
    )


def _dump_character(character: Character) -> dict:
    return {"name": character.name, "recruited": character.recruited}


def _load_character(data: dict) -> Character:
    return Character(name=data["name"], recruited=data["recruited"])


def _dump_class_progress(progress: ClassProgress) -> dict:
    return {
        "class_id": progress.class_id,
        "level": progress.level,
        "exp": progress.exp,
        "claimed_nodes": list(progress.claimed_nodes),
    }


def _load_class_progress(data: dict) -> ClassProgress:
    progress = ClassProgress(data["class_id"])
    progress.level = data["level"]
    progress.exp = data["exp"]
    progress.claimed_nodes = set(data["claimed_nodes"])
    return progress


def _dump_progression(progression: Progression) -> dict:
    return {
        "species_id": progression.species_id,
        "innate_id": progression.innate_id,
        "species_level": progression.species_level,
        "species_exp": progression.species_exp,
        "species_claimed": list(progression.species_claimed),
        "innate_claimed": list(progression.innate_claimed),
        "classes": [_dump_class_progress(c) for c in progression.classes],
        "affinities": dict(progression.affinities),
        # Without this, a reloaded character's automatic per-species-level
        # HP/MP/SP growth bookkeeping comes back empty even though the
        # actual (already-inflated) stat values loaded correctly — the next
        # reapply_species_stat_growth call (e.g. giving up a run) then has
        # nothing to compare against and silently no-ops instead of
        # reverting the growth, permanently stranding it on the character.
        "species_stat_growth": [dict(p) for p in progression.species_stat_growth],
    }


def _load_progression(data: dict) -> Progression:
    progression = Progression(
        species_id=data["species_id"], innate_id=data["innate_id"], affinities=dict(data["affinities"])
    )
    progression.species_level = data["species_level"]
    progression.species_exp = data["species_exp"]
    progression.species_claimed = set(data["species_claimed"])
    progression.innate_claimed = set(data["innate_claimed"])
    progression.classes = [_load_class_progress(c) for c in data["classes"]]
    progression.species_stat_growth = [dict(p) for p in data.get("species_stat_growth", [])]
    return progression


def _dump_skill_book(book: SkillBook) -> dict:
    return {"skills": [instance.serialize() for instance in book.skills.values()]}


def _load_skill_book(data: dict) -> SkillBook:
    book = SkillBook()
    for raw in data["skills"]:
        instance = SkillInstance.deserialize(raw)
        book.skills[instance.def_id] = instance
    return book


def _dump_item(item: ItemInstance) -> dict:
    return {
        "uid": item.uid,
        "item_id": item.item_id,
        "charges": item.charges_remaining,
        "quantity": item.quantity,
        "enchantments": list(item.enchantments),
        "applied_deltas": [dict(p) for p in item.applied_deltas],
    }


def _load_item(data: dict) -> ItemInstance:
    item = ItemInstance(data["item_id"], charges=data["charges"], quantity=data["quantity"])
    item.uid = data["uid"]
    item.enchantments = list(data["enchantments"])
    item.applied_deltas = [dict(p) for p in data["applied_deltas"]]
    return item


def _dump_inventory(inv: Inventory) -> dict:
    return {
        "capacity": inv.capacity,
        "items": [_dump_item(i) for i in inv.items],
        "equipped": {slot: item.uid for slot, item in inv.equipped.items()},
    }


def _load_inventory(data: dict) -> Inventory:
    inv = Inventory(capacity=data["capacity"])
    inv.items = [_load_item(d) for d in data["items"]]
    by_uid = {i.uid: i for i in inv.items}
    inv.equipped = {slot: by_uid[uid] for slot, uid in data["equipped"].items()}
    return inv


def _ai_type_name(ai: Any) -> Optional[str]:
    if isinstance(ai, FSMController):
        return "fsm"
    if isinstance(ai, BehaviorTreeController):
        return "bt"
    if isinstance(ai, UtilityAIController):
        return "utility"
    return None


def _make_ai(type_name: Optional[str]) -> Any:
    if type_name == "fsm":
        return FSMController()
    if type_name == "bt":
        return BehaviorTreeController()
    if type_name == "utility":
        return UtilityAIController()
    return None


def _dump_loot(loot: Any) -> Optional[dict]:
    if loot is None:
        return None
    return {"gold_range": list(loot.gold_range), "drops": [list(d) for d in loot.drops]}


def _load_loot(data: Optional[dict]) -> Any:
    if data is None:
        return None
    from game.bestiary import LootTable

    return LootTable(
        gold_range=tuple(data["gold_range"]), drops=[tuple(d) for d in data["drops"]]
    )


def _dump_entity(entity: Entity) -> dict:
    return {
        "x": entity.x,
        "y": entity.y,
        "char": entity.char,
        "color": list(entity.color),
        "name": entity.name,
        "blocks_movement": entity.blocks_movement,
        "light_radius": entity.light_radius,
        "sees_through_darkness": entity.sees_through_darkness,
        "energy": entity.energy,
        "character": _dump_character(entity.character) if entity.character is not None else None,
        "stats": _dump_stats(entity.stats) if entity.stats is not None else None,
        "progression": _dump_progression(entity.progression) if entity.progression is not None else None,
        "skill_book": _dump_skill_book(entity.skill_book) if entity.skill_book is not None else None,
        "inventory": _dump_inventory(entity.inventory) if entity.inventory is not None else None,
        "ground_item": _dump_item(entity.ground_item) if entity.ground_item is not None else None,
        "ai_type": _ai_type_name(entity.ai),
        "is_shop": entity.is_shop,
        "loot": _dump_loot(entity.loot),
    }


def _load_entity(data: dict) -> Entity:
    entity = Entity(
        x=data["x"],
        y=data["y"],
        char=data["char"],
        color=tuple(data["color"]),
        name=data["name"],
        blocks_movement=data["blocks_movement"],
        light_radius=data["light_radius"],
        sees_through_darkness=data["sees_through_darkness"],
        character=_load_character(data["character"]) if data["character"] is not None else None,
        stats=_load_stats(data["stats"]) if data["stats"] is not None else None,
        progression=_load_progression(data["progression"]) if data["progression"] is not None else None,
        skill_book=_load_skill_book(data["skill_book"]) if data["skill_book"] is not None else None,
        inventory=_load_inventory(data["inventory"]) if data["inventory"] is not None else None,
        ground_item=_load_item(data["ground_item"]) if data["ground_item"] is not None else None,
        is_shop=data.get("is_shop", False),
        loot=_load_loot(data.get("loot")),
    )
    entity.energy = data["energy"]
    entity.ai = _make_ai(data["ai_type"])
    return entity


def _dump_game_map(game_map: GameMap) -> dict:
    return {
        "width": game_map.width,
        "height": game_map.height,
        "is_overworld": game_map.is_overworld,
        "tiles": game_map.tiles.tobytes(order="F"),
        "explored": game_map.explored.tobytes(order="F"),
        "light_sources": [
            {"x": s.x, "y": s.y, "radius": s.radius} for s in game_map.light_sources
        ],
        "darkness_sources": [
            {"x": s.x, "y": s.y, "radius": s.radius} for s in game_map.darkness_sources
        ],
        "door_positions": [list(p) for p in game_map.door_positions],
        "stairs_up": list(game_map.stairs_up) if game_map.stairs_up else None,
        "stairs_down": list(game_map.stairs_down) if game_map.stairs_down else None,
        "gap": list(game_map.gap) if game_map.gap else None,
    }


def _load_game_map(data: dict) -> GameMap:
    width, height = data["width"], data["height"]
    game_map = GameMap(width, height, is_overworld=data["is_overworld"])
    game_map.tiles = (
        np.frombuffer(data["tiles"], dtype=tile_types.tile_dt)
        .reshape((width, height), order="F")
        .copy()
    )
    game_map.explored = (
        np.frombuffer(data["explored"], dtype=bool).reshape((width, height), order="F").copy()
    )
    game_map.light_sources = [
        LightSource(x=s["x"], y=s["y"], radius=s["radius"]) for s in data["light_sources"]
    ]
    game_map.darkness_sources = [
        DarknessSource(x=s["x"], y=s["y"], radius=s["radius"]) for s in data["darkness_sources"]
    ]
    game_map.door_positions = {tuple(p) for p in data["door_positions"]}
    game_map.stairs_up = tuple(data["stairs_up"]) if data["stairs_up"] else None
    game_map.stairs_down = tuple(data["stairs_down"]) if data["stairs_down"] else None
    game_map.gap = tuple(data["gap"]) if data.get("gap") else None
    return game_map


# ---------------------------------------------------------------------------
# Whole-engine (de)serialization
# ---------------------------------------------------------------------------


def _dump_target_point(tp: TargetPoint, index_of) -> dict:
    return {"x": tp.x, "y": tp.y, "entity": index_of(tp.entity) if tp.entity is not None else None}


def _load_target_point(data: dict, entities: List[Entity]) -> TargetPoint:
    entity = entities[data["entity"]] if data["entity"] is not None else None
    return TargetPoint(x=data["x"], y=data["y"], entity=entity)


def _dump_charge(charge: ChargeState, index_of) -> dict:
    return {
        "skill_id": charge.instance.def_id,
        "targets": [_dump_target_point(tp, index_of) for tp in charge.targets],
        "extra_targets": [
            [stage, [_dump_target_point(tp, index_of) for tp in tps]]
            for stage, tps in charge.extra_targets.items()
        ],
        "turns_remaining": charge.turns_remaining,
        "lock_movement": charge.lock_movement,
        "lock_actions": charge.lock_actions,
    }


def _load_charge(data: dict, owner: Entity, entities: List[Entity]) -> Optional[ChargeState]:
    instance = owner.skill_book.get(data["skill_id"]) if owner.skill_book is not None else None
    if instance is None:
        return None
    return ChargeState(
        instance=instance,
        targets=[_load_target_point(tp, entities) for tp in data["targets"]],
        extra_targets={
            stage: [_load_target_point(tp, entities) for tp in tps] for stage, tps in data["extra_targets"]
        },
        turns_remaining=data["turns_remaining"],
        lock_movement=data["lock_movement"],
        lock_actions=data["lock_actions"],
    )


def _dump_timed_effect(effect: TimedEffectInstance, index_of) -> dict:
    return {
        "kind": effect.kind,
        "target": index_of(effect.target),
        "source": index_of(effect.source) if effect.source is not None else None,
        "remaining_turns": effect.remaining_turns,
        "payload": effect.payload,
        "stacks": effect.stacks,
    }


def _load_timed_effect(data: dict, entities: List[Entity]) -> TimedEffectInstance:
    return TimedEffectInstance(
        kind=data["kind"],
        target=entities[data["target"]],
        remaining_turns=data["remaining_turns"],
        payload=data["payload"],
        source=entities[data["source"]] if data["source"] is not None else None,
        stacks=data["stacks"],
    )


def _describe_location(engine: "Engine") -> str:
    if engine.floor_depth == 0:
        return "Town"
    if engine.current_location is None:
        return f"Tutorial F{engine.floor_depth}"
    try:
        name = location_data.location(engine.current_location).name
    except KeyError:
        name = engine.current_location
    return f"{name} F{engine.location_floor_index}"


def _build_meta(engine: "Engine") -> dict:
    progression = engine.player.progression
    return {
        "character": engine.player.name,
        "level": progression.species_level if progression is not None else 0,
        "where": _describe_location(engine),
        "saved_at": time.time(),
    }


def serialize_engine(engine: "Engine") -> Dict[str, Any]:
    """Full snapshot: floor history (indexed by depth), the player/party/
    downed references (by index into a dedup'd entity list, since the same
    Entity object appears in multiple places), and account-wide state.
    Everything already run-scoped vs. persistent is handled by the existing
    reset-on-town-return logic — a save just captures whatever the live
    state currently is, mid-run or in town."""
    engine._save_current_floor()

    entity_ids: Dict[int, int] = {}
    entity_objs: List[Entity] = []
    entities_data: List[dict] = []

    def index_of(entity: Entity) -> int:
        key = id(entity)
        if key not in entity_ids:
            entity_ids[key] = len(entities_data)
            entity_objs.append(entity)
            entities_data.append(_dump_entity(entity))
        return entity_ids[key]

    floors_data = []
    for game_map, entities, pos in engine.floor_history:
        floors_data.append(
            {
                "game_map": _dump_game_map(game_map),
                "entity_indices": [index_of(e) for e in entities],
                "player_pos": list(pos),
            }
        )

    player_index = index_of(engine.player)
    party_indices = [index_of(e) for e in engine.party]
    downed_indices = [index_of(e) for e in engine.downed]
    timed_effects = [_dump_timed_effect(e, index_of) for e in engine.timed_effects]
    pending_boss_index = index_of(engine.pending_boss) if engine.pending_boss is not None else None

    # Cross-entity links resolved after the fact, since dumping them inline
    # would recurse into index_of mid-append. The list can grow as we go.
    i = 0
    while i < len(entity_objs):
        entity = entity_objs[i]
        entities_data[i]["summoned_by"] = (
            index_of(entity.summoned_by) if entity.summoned_by is not None else None
        )
        entities_data[i]["charging"] = (
            _dump_charge(entity.charging, index_of) if entity.charging is not None else None
        )
        i += 1

    return {
        "version": SAVE_VERSION,
        "meta": _build_meta(engine),
        "timed_effects": timed_effects,
        "entities": entities_data,
        "floors": floors_data,
        "floor_depth": engine.floor_depth,
        "gold": engine.gold,
        "player_index": player_index,
        "party_indices": party_indices,
        "downed_indices": downed_indices,
        "unlocked_classes": list(engine.unlocked_classes),
        "current_location": engine.current_location,
        "location_floor_index": engine.location_floor_index,
        "location_floor_target": engine.location_floor_target,
        "location_entry_depth": engine.location_entry_depth,
        "pending_boss_index": pending_boss_index,
        "pending_boss_id": engine.pending_boss_id,
        "knowledge": {
            "locations": list(engine.knowledge["locations"]),
            "bosses": list(engine.knowledge["bosses"]),
            "recruited_bosses": list(engine.knowledge["recruited_bosses"]),
        },
    }


def apply_engine_state(engine: "Engine", data: Dict[str, Any]) -> None:
    """Mutate an existing Engine in place to match a loaded save — avoids
    needing every holder of the Engine reference (event handler, main loop)
    to swap to a new object."""
    data = _migrate(data)

    # Reused ItemInstance uids must not collide with newly-created items.
    max_uid = 0
    for entity_data in data["entities"]:
        for key in ("inventory", "ground_item"):
            payload = entity_data.get(key)
            if payload is None:
                continue
            items = payload["items"] if key == "inventory" else [payload]
            for item_data in items:
                max_uid = max(max_uid, item_data["uid"])
    inventory_module._uid_counter = itertools.count(max_uid + 1)

    entities = [_load_entity(e) for e in data["entities"]]
    for entity, entity_data in zip(entities, data["entities"]):
        summoner = entity_data.get("summoned_by")
        entity.summoned_by = entities[summoner] if summoner is not None else None
        charge = entity_data.get("charging")
        entity.charging = _load_charge(charge, entity, entities) if charge is not None else None

    floor_history = []
    for floor_data in data["floors"]:
        game_map = _load_game_map(floor_data["game_map"])
        floor_entities = [entities[i] for i in floor_data["entity_indices"]]
        pos = tuple(floor_data["player_pos"])
        floor_history.append((game_map, floor_entities, pos))

    floor_depth = data["floor_depth"]
    player = entities[data["player_index"]]
    party = [entities[i] for i in data["party_indices"]]
    downed = [entities[i] for i in data["downed_indices"]]

    engine.game_map, engine.entities, (player.x, player.y) = floor_history[floor_depth]
    engine.player = player
    engine.party = party
    engine.downed = downed
    engine.floor_history = floor_history
    engine.floor_depth = floor_depth
    engine.gold = data.get("gold", 0)
    engine.unlocked_classes = set(data["unlocked_classes"])
    engine.current_location = data.get("current_location")
    engine.location_floor_index = data.get("location_floor_index", 0)
    engine.location_floor_target = data.get("location_floor_target", 0)
    engine.location_entry_depth = data.get("location_entry_depth", 0)
    pending_boss_index = data.get("pending_boss_index")
    engine.pending_boss = entities[pending_boss_index] if pending_boss_index is not None else None
    engine.pending_boss_id = data.get("pending_boss_id")
    # BossController isn't serialized; rebuild it from the boss def. Unlocked
    # phases re-derive from current HP on its next decide().
    if engine.pending_boss is not None and engine.pending_boss.ai is None and engine.pending_boss_id:
        boss_def = location_data.boss(engine.pending_boss_id)
        engine.pending_boss.ai = BossController([(p.hp_fraction, p.skill_id) for p in boss_def.phases])
    knowledge = data.get("knowledge", {"locations": [], "bosses": [], "recruited_bosses": []})
    engine.knowledge = {
        "locations": set(knowledge.get("locations", [])),
        "bosses": set(knowledge.get("bosses", [])),
        "recruited_bosses": set(knowledge.get("recruited_bosses", [])),
    }
    engine.timed_effects = [_load_timed_effect(e, entities) for e in data["timed_effects"]]
    # Pending delayed triggers hold a live CastContext and aren't persisted;
    # a save simply drops any not-yet-fired follow-up hits.
    engine.delayed_triggers = []
    engine.focus_target = None
    engine.auto_exploring = False
    engine.active_menu = None
    engine.pause_menu_open = False
    engine.target_selector = None
    engine.character_screen_open = False
    engine.inspect_open = False
    engine.message_log_expanded = False
    engine.game_over = False
    engine.update_camera()
    engine.update_fov()


def _migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    version = data.get("version")
    if version == 1:
        # v1 never stored timed effects, charges or summoner links; their
        # absence is handled by .get() defaults at load time.
        data["timed_effects"] = []
        version = data["version"] = 2
    if version != SAVE_VERSION:
        raise SaveError(f"Unsupported save version: {version!r}")
    return data


def write_slot(data: Dict[str, Any], slot: int) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _slot_path(slot)
    # Write-then-rename so a crash mid-write (e.g. autosave on quit) can't
    # leave a truncated save behind.
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(msgpack.packb(data, use_bin_type=True))
    os.replace(tmp, path)


def read_slot(slot: int) -> Dict[str, Any]:
    path = _slot_path(slot)
    if not path.exists():
        raise SaveError(f"No save in slot {slot}")
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
