from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

from components.items import (
    CATEGORY_COLLECTIBLE,
    CATEGORY_EQUIPPABLE,
    CATEGORY_USABLE,
    CollectibleDef,
    EnchantmentDef,
    EquippableDef,
    UsableDef,
    parse_collectible,
    parse_enchantment,
    parse_equippable,
    parse_usable,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ItemDefUnion = Union[UsableDef, EquippableDef, CollectibleDef]

_items: Dict[str, ItemDefUnion] = {}
_enchantments: Dict[str, EnchantmentDef] = {}
_loaded = False


def _read(name: str):
    path = DATA_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_all(force: bool = False) -> None:
    global _loaded
    if _loaded and not force:
        return

    _items.clear()
    _enchantments.clear()

    for raw in _read("items.json"):
        category = raw["category"]
        if category == CATEGORY_USABLE:
            _items[raw["item_id"]] = parse_usable(raw)
        elif category == CATEGORY_EQUIPPABLE:
            _items[raw["item_id"]] = parse_equippable(raw)
        elif category == CATEGORY_COLLECTIBLE:
            _items[raw["item_id"]] = parse_collectible(raw)
        else:
            raise ValueError(f"Unknown item category: {category!r}")

    for raw in _read("enchantments.json"):
        _enchantments[raw["enchant_id"]] = parse_enchantment(raw)

    _loaded = True


def item(item_id: str) -> ItemDefUnion:
    load_all()
    return _items[item_id]


def enchantment(enchant_id: str) -> EnchantmentDef:
    load_all()
    return _enchantments[enchant_id]


def all_items() -> List[ItemDefUnion]:
    load_all()
    return list(_items.values())
