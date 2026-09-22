from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from components.inventory import ItemInstance
from components.items import EquippableDef, WeaponDef
from components.skills import EffectDef, SkillInstance
from components.stats import apply_stat_modifier, revert_stat_modifier
from game import item_data, message_log

if TYPE_CHECKING:
    from game.engine import Engine
    from game.entity import Entity


def _apply_permanent_effects(engine: "Engine", entity: "Entity", instance: ItemInstance, effects: List[EffectDef]) -> None:
    """Run non-combat permanent effects (stat mods) for a relic/enchant, and
    remember the revert payloads on the instance so they can be undone."""
    if entity.stats is None:
        return
    for effect in effects:
        if effect.type != "stat_modifier":
            continue  # other effect types aren't meaningful as passive holds
        instance.applied_deltas.append(
            apply_stat_modifier(entity.stats, effect.params["attribute"], effect.params["amount"])
        )


def _revert_permanent_effects(entity: "Entity", instance: ItemInstance) -> None:
    if entity.stats is None:
        return
    for payload in instance.applied_deltas:
        revert_stat_modifier(entity.stats, payload)
    instance.applied_deltas.clear()


def pick_up(engine: "Engine", entity: "Entity", instance: ItemInstance) -> bool:
    """Add an item to inventory; collectibles' relic effects apply immediately."""
    if entity.inventory is None or not entity.inventory.add(instance):
        return False
    definition = instance.definition
    if hasattr(definition, "passive_effects"):
        _apply_permanent_effects(engine, entity, instance, definition.passive_effects)
    return True


def drop(engine: "Engine", entity: "Entity", instance: ItemInstance) -> None:
    if entity.inventory is None:
        return
    for slot, equipped in list(entity.inventory.equipped.items()):
        if equipped is instance:
            unequip(engine, entity, slot)
    _revert_permanent_effects(entity, instance)
    entity.inventory.remove(instance)


def equip(engine: "Engine", entity: "Entity", instance: ItemInstance) -> bool:
    definition = instance.definition
    if not isinstance(definition, EquippableDef) or entity.inventory is None:
        return False

    slot = definition.slot
    if slot in entity.inventory.equipped:
        unequip(engine, entity, slot)

    entity.inventory.equipped[slot] = instance
    _apply_permanent_effects_for_enchants(engine, entity, instance)

    if entity.skill_book is not None:
        for skill_id in definition.granted_skills:
            # If the character already has this skill from somewhere else
            # (e.g. a Warrior's own class tree granting Slash, then
            # equipping Iron Sword — which also grants "slash" — on top),
            # leave the existing instance alone: SkillBook.add keys purely
            # by def_id, so blindly adding here would silently overwrite it
            # with a fresh equipment-sourced instance (losing its level/
            # upgrades), and unequipping the weapon would then delete a
            # skill the character should still have from its real source.
            if entity.skill_book.get(skill_id) is not None:
                continue
            entity.skill_book.add(SkillInstance(skill_id, source=f"equipment:{slot}"))

    engine.message_log.add_message(
        f"{entity.name} equips {definition.name}.", color=message_log.INFO_COLOR, stack=False
    )
    return True


def unequip(engine: "Engine", entity: "Entity", slot: str) -> bool:
    if entity.inventory is None:
        return False
    instance = entity.inventory.equipped.pop(slot, None)
    if instance is None:
        return False

    _revert_permanent_effects(entity, instance)

    if entity.skill_book is not None:
        for def_id in list(entity.skill_book.skills):
            skill = entity.skill_book.get(def_id)
            if skill.source == f"equipment:{slot}":
                entity.skill_book.remove(def_id)

    return True


def _apply_permanent_effects_for_enchants(engine: "Engine", entity: "Entity", instance: ItemInstance) -> None:
    effects: List[EffectDef] = []
    for enchant_id in instance.enchantments:
        effects += item_data.enchantment(enchant_id).effects
    if effects:
        _apply_permanent_effects(engine, entity, instance, effects)


def attach_enchantment(instance: ItemInstance, enchant_id: str) -> bool:
    definition = instance.definition
    if not isinstance(definition, EquippableDef):
        return False
    if len(instance.enchantments) >= definition.enchant_slots:
        return False
    instance.enchantments.append(enchant_id)
    return True


def total_defense(entity: "Entity", *, magic: bool = False) -> int:
    """Armor's flat defense plus any stat_modifier-driven bonus (skills,
    relics, enchants targeting "defense"/"magic_defense") stacked on top."""
    total = 0
    if entity.inventory is not None:
        for slot in ("head", "chest"):
            instance = entity.inventory.equipped.get(slot)
            if instance is None:
                continue
            definition = instance.definition
            total += definition.magic_defense if magic else definition.defense
    if entity.stats is not None:
        total += entity.stats.modifiers.get("magic_defense" if magic else "defense", 0.0)
    return int(total)


def equipped_weapon(entity: "Entity") -> Optional[WeaponDef]:
    if entity.inventory is None:
        return None
    instance = entity.inventory.equipped.get("mainhand")
    if instance is None:
        return None
    definition = instance.definition
    return definition if isinstance(definition, WeaponDef) else None
