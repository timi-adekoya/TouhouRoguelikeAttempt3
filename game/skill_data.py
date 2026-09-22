from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.skills import SkillDef, UpgradeDef, parse_skill, parse_upgrade
from game.skill_tree import SkillTreeDef, parse_tree

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ClassDef:
    """A learnable class. No restrictions on who can learn it; availability
    is gated by the account-wide unlock (gold, back in town) and per-character
    slots from species level. `evolves_to` is for max-level evolution;
    combination classes carry their own unlock costs/requirements (later)."""

    class_id: str
    name: str
    description: str = ""
    unlock_cost_gold: int = 0
    tree: SkillTreeDef = field(default_factory=lambda: SkillTreeDef(""))
    evolves_to: Optional[str] = None
    max_level: int = 10
    # Class-tree skill points earned per class level, spent on `tree`'s nodes
    # via their own `cost`. Per-class rather than a global rate — how deep/
    # wide a class's tree is varies a lot (a 2-branch fork vs. a future
    # 4-branch one), so each class tunes its own curve.
    points_per_level: int = 1


@dataclass
class SpeciesDef:
    """A race/species: immutable per character, shared skill tree."""

    species_id: str
    name: str
    description: str = ""
    tree: SkillTreeDef = field(default_factory=lambda: SkillTreeDef(""))


@dataclass
class InnateDef:
    """A character's personal (innate) skill tree, leveled by species level.

    Most innate trees are still auto-claimed linear rivers (`manual=False`,
    every node cost 0 by convention, `points_per_level` irrelevant). A
    character whose kit should offer real non-exclusive branches (e.g.
    Reimu choosing between/eventually getting both Ofuda and Needles) sets
    `manual=True` and gives their nodes real `cost`s — same cost-gated
    mechanism class trees already use, just applied here too, claimed via
    the same "Innate skill tree" player menu instead of auto-flowing."""

    innate_id: str
    species_id: str
    description: str = ""
    tree: SkillTreeDef = field(default_factory=lambda: SkillTreeDef(""))
    manual: bool = False
    points_per_level: int = 1


_raw_skills: Dict[str, Dict[str, Any]] = {}
_skills: Dict[str, SkillDef] = {}
_upgrades: Dict[str, UpgradeDef] = {}
_classes: Dict[str, ClassDef] = {}
_species: Dict[str, SpeciesDef] = {}
_innate: Dict[str, InnateDef] = {}
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

    _raw_skills.clear(); _skills.clear(); _upgrades.clear()
    _classes.clear(); _species.clear(); _innate.clear()

    for raw in _read("skills.json"):
        _raw_skills[raw["skill_id"]] = raw
        _skills[raw["skill_id"]] = parse_skill(raw)

    for raw in _read("upgrades.json"):
        _upgrades[raw["upgrade_id"]] = parse_upgrade(raw)

    for raw in _read("classes.json"):
        _classes[raw["class_id"]] = ClassDef(
            class_id=raw["class_id"],
            name=raw["name"],
            description=raw.get("description", ""),
            unlock_cost_gold=raw.get("unlock_cost_gold", 0),
            tree=parse_tree(raw["class_id"], raw.get("tree", [])),
            evolves_to=raw.get("evolves_to"),
            max_level=raw.get("max_level", 10),
            points_per_level=raw.get("points_per_level", 1),
        )

    for raw in _read("species.json"):
        _species[raw["species_id"]] = SpeciesDef(
            species_id=raw["species_id"],
            name=raw["name"],
            description=raw.get("description", ""),
            tree=parse_tree(raw["species_id"], raw.get("tree", [])),
        )

    for raw in _read("innate.json"):
        _innate[raw["innate_id"]] = InnateDef(
            innate_id=raw["innate_id"],
            species_id=raw["species_id"],
            description=raw.get("description", ""),
            tree=parse_tree(raw["innate_id"], raw.get("tree", [])),
            manual=raw.get("manual", False),
            points_per_level=raw.get("points_per_level", 1),
        )

    _loaded = True


def raw_skill(skill_id: str) -> Dict[str, Any]:
    load_all()
    return _raw_skills[skill_id]


def skill(skill_id: str) -> SkillDef:
    load_all()
    return _skills[skill_id]


def upgrade(upgrade_id: str) -> UpgradeDef:
    load_all()
    return _upgrades[upgrade_id]


def class_def(class_id: str) -> ClassDef:
    load_all()
    return _classes[class_id]


def species_def(species_id: str) -> SpeciesDef:
    load_all()
    return _species[species_id]


def innate_def(innate_id: str) -> InnateDef:
    load_all()
    return _innate[innate_id]


def all_classes() -> List[ClassDef]:
    load_all()
    return list(_classes.values())
