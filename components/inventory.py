from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

from components.items import CollectibleDef, EquippableDef, UsableDef
from game import item_data

_uid_counter = itertools.count(1)


class ItemInstance:
    """An item in an inventory: which def, remaining charges (for
    charge/indefinitely-reusable usables), a stack quantity (for stackable
    ammo/collectibles/consumables), attached enchantments, and any deltas
    applied while it's been equipped/held so they can be reverted cleanly
    (relics, enchant effects)."""

    def __init__(self, item_id: str, charges: Optional[int] = None, quantity: int = 1):
        self.uid = next(_uid_counter)
        self.item_id = item_id
        self.charges_remaining = charges
        self.quantity = quantity
        self.enchantments: List[str] = []
        # Revert payloads (from components.stats.apply_stat_modifier) for
        # relic/enchant stat_modifier effects, so they can be cleanly undone.
        self.applied_deltas: List[Dict[str, Any]] = []

    @property
    def definition(self):
        return item_data.item(self.item_id)

    @property
    def is_stackable(self) -> bool:
        """Equipment never stacks (each piece can carry its own enchantments/
        charges). Collectibles (gold-likes, ammo, relics) and consumable
        usables (potions, scrolls) stack by quantity in one inventory slot;
        charge/indefinitely-reusable usables (wands) stay one-per-instance."""
        definition = self.definition
        if isinstance(definition, EquippableDef):
            return False
        if isinstance(definition, UsableDef):
            return definition.consumable
        return isinstance(definition, CollectibleDef)


class Inventory:
    """Held items, gold, and equipped slots. Equipped items stay referenced
    from `items` too — `equipped` just points at which slot holds which."""

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.items: List[ItemInstance] = []
        self.equipped: Dict[str, ItemInstance] = {}

    @property
    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def add(self, instance: ItemInstance) -> bool:
        if instance.is_stackable:
            existing = next(
                (i for i in self.unequipped_items() if i.item_id == instance.item_id), None
            )
            if existing is not None:
                existing.quantity += instance.quantity
                return True
        if self.is_full:
            return False
        self.items.append(instance)
        return True

    def remove(self, instance: ItemInstance) -> None:
        for slot, equipped in list(self.equipped.items()):
            if equipped is instance:
                del self.equipped[slot]
        if instance in self.items:
            self.items.remove(instance)

    def unequipped_items(self) -> List[ItemInstance]:
        equipped_uids = {i.uid for i in self.equipped.values()}
        return [i for i in self.items if i.uid not in equipped_uids]

    def find_tagged(self, tag: str) -> Optional[ItemInstance]:
        """First unequipped item instance carrying `tag` (e.g. an ammo type)."""
        return next((i for i in self.unequipped_items() if tag in i.definition.tags), None)

    def consume_one(self, instance: ItemInstance) -> None:
        """Use up one unit of a stacked item (ammo, a single potion...),
        removing the instance entirely once its stack is empty."""
        instance.quantity -= 1
        if instance.quantity <= 0:
            self.remove(instance)
