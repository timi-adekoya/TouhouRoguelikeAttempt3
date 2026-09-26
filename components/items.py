from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from components.skills import ActivationTargetingDef, EffectDef, parse_effect, parse_targeting

CATEGORY_USABLE = "usable"
CATEGORY_EQUIPPABLE = "equippable"
CATEGORY_COLLECTIBLE = "collectible"

SLOT_HEAD = "head"
SLOT_CHEST = "chest"
SLOT_MAINHAND = "mainhand"
SLOT_OFFHAND = "offhand"
SLOT_ACCESSORY_1 = "accessory_1"
SLOT_ACCESSORY_2 = "accessory_2"
EQUIP_SLOTS = [SLOT_HEAD, SLOT_CHEST, SLOT_MAINHAND, SLOT_OFFHAND, SLOT_ACCESSORY_1, SLOT_ACCESSORY_2]


def evaluate_formula(formula: Dict[str, Any], attributes) -> float:
    """base + sum(coeff * attribute.current) over formula["scaling"]."""
    total = float(formula.get("base", 0))
    for attr_name, coeff in formula.get("scaling", {}).items():
        total += coeff * getattr(attributes, attr_name).current
    return total


@dataclass
class UsableDef:
    """Consumed (potions/scrolls) or charge/indefinitely reusable (wands).
    Runs through the same effect-executor pipeline as skills."""

    item_id: str
    name: str
    description: str = ""
    rarity: str = "common"
    price: int = 0  # 0 = not sold in shops
    tags: List[str] = field(default_factory=list)
    consumable: bool = True
    charges: Optional[int] = None  # None = indefinitely reusable
    activation_targeting: List[ActivationTargetingDef] = field(default_factory=list)
    effects: List[EffectDef] = field(default_factory=list)

    def get_effect(self, effect_id: str) -> Optional[EffectDef]:
        return next((e for e in self.effects if e.effect_id == effect_id), None)


@dataclass
class EquippableDef:
    """Base equipment: head/chest armor, accessories. Weapons subclass this."""

    item_id: str
    name: str
    slot: str
    description: str = ""
    rarity: str = "common"
    price: int = 0  # 0 = not sold in shops
    tags: List[str] = field(default_factory=list)
    defense: int = 0
    magic_defense: int = 0
    enchant_slots: int = 0
    granted_skills: List[str] = field(default_factory=list)


@dataclass
class WeaponDef(EquippableDef):
    """A mainhand/offhand weapon: damage/crit formulas scaling off attributes,
    optional MP cost (magic weapons), optional ammo requirement (ranged)."""

    damage_formula: Dict[str, Any] = field(default_factory=dict)
    crit_rate_formula: Dict[str, Any] = field(default_factory=dict)
    crit_damage_formula: Dict[str, Any] = field(default_factory=lambda: {"base": 1.5})
    mp_cost: int = 0
    ammo_type: Optional[str] = None
    # Hit chance: (accuracy - target's evasion) * hit_rate, floored at 1%.
    # accuracy is the weapon's own inherent base (1.0 = 100%, an inaccurate
    # weapon can set this lower directly, separate from hit_rate_formula).
    # hit_rate_formula scales off the attacker's attributes like crit does,
    # and can push the result above 1.0. ignore_evasion skips the roll
    # entirely (always hits) for weapons that should never miss.
    accuracy: float = 1.0
    hit_rate_formula: Dict[str, Any] = field(default_factory=dict)
    ignore_evasion: bool = False
    # Tiles a basic attack can reach via the fire command; 1 = melee only.
    range: int = 1

    @property
    def is_ranged(self) -> bool:
        return self.range > 1


@dataclass
class CollectibleDef:
    """Gold-like or relic items that mostly just sit in the inventory. Relic
    `passive_effects` apply while held (stat_modifier deltas are tracked per
    ItemInstance so they can be reverted if the item leaves the inventory)."""

    item_id: str
    name: str
    description: str = ""
    rarity: str = "common"
    price: int = 0  # 0 = not sold in shops
    tags: List[str] = field(default_factory=list)
    passive_effects: List[EffectDef] = field(default_factory=list)


@dataclass
class EnchantmentDef:
    """Attached to an equippable's enchant slot; effects apply on equip and
    revert on unequip, same as a relic."""

    enchant_id: str
    name: str
    description: str = ""
    rarity: str = "common"
    effects: List[EffectDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_usable(raw: Dict[str, Any]) -> UsableDef:
    return UsableDef(
        item_id=raw["item_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        rarity=raw.get("rarity", "common"),
        price=raw.get("price", 0),
        tags=raw.get("tags", []),
        consumable=raw.get("consumable", True),
        charges=raw.get("charges"),
        activation_targeting=[parse_targeting(t) for t in raw.get("activation_targeting", [])],
        effects=[parse_effect(e) for e in raw.get("effects", [])],
    )


def parse_equippable(raw: Dict[str, Any]) -> EquippableDef:
    common = dict(
        item_id=raw["item_id"],
        name=raw["name"],
        slot=raw["slot"],
        description=raw.get("description", ""),
        rarity=raw.get("rarity", "common"),
        price=raw.get("price", 0),
        tags=raw.get("tags", []),
        defense=raw.get("defense", 0),
        magic_defense=raw.get("magic_defense", 0),
        enchant_slots=raw.get("enchant_slots", 0),
        granted_skills=raw.get("granted_skills", []),
    )
    if raw["slot"] in (SLOT_MAINHAND, SLOT_OFFHAND) and "damage_formula" in raw:
        return WeaponDef(
            **common,
            damage_formula=raw.get("damage_formula", {}),
            crit_rate_formula=raw.get("crit_rate_formula", {}),
            crit_damage_formula=raw.get("crit_damage_formula", {"base": 1.5}),
            mp_cost=raw.get("mp_cost", 0),
            ammo_type=raw.get("ammo_type"),
            accuracy=raw.get("accuracy", 1.0),
            hit_rate_formula=raw.get("hit_rate_formula", {}),
            ignore_evasion=raw.get("ignore_evasion", False),
            range=raw.get("range", 1),
        )
    return EquippableDef(**common)


def parse_collectible(raw: Dict[str, Any]) -> CollectibleDef:
    return CollectibleDef(
        item_id=raw["item_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        rarity=raw.get("rarity", "common"),
        price=raw.get("price", 0),
        tags=raw.get("tags", []),
        passive_effects=[parse_effect(e) for e in raw.get("passive_effects", [])],
    )


def parse_enchantment(raw: Dict[str, Any]) -> EnchantmentDef:
    return EnchantmentDef(
        enchant_id=raw["enchant_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        rarity=raw.get("rarity", "common"),
        effects=[parse_effect(e) for e in raw.get("effects", [])],
    )
