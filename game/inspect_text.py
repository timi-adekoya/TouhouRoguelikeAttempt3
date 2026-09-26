from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from components.inventory import ItemInstance
from components.items import CollectibleDef, EquippableDef, UsableDef, WeaponDef
from components.skills import CATEGORY_ACTIVE, CATEGORY_PASSIVE, CATEGORY_PROFICIENCY, SkillInstance, skill_exp_to_next
from game import skill_data


def _format_formula(formula: Dict[str, Any]) -> str:
    base = formula.get("base", 0)
    scaling = formula.get("scaling", {})
    if not scaling:
        return str(base)
    return f"{base} + " + " + ".join(f"{coeff}x{attr}" for attr, coeff in scaling.items())


def skill_lines(instance: SkillInstance) -> List[str]:
    skill = instance.resolved
    lines: List[str] = []
    if skill.description:
        lines += [skill.description, ""]

    if instance.level < skill.max_level:
        lines.append(
            f"Level {instance.level}/{skill.max_level}  "
            f"({instance.exp:.0f}/{skill_exp_to_next(instance.level, skill.rarity):.0f} exp)"
        )
    else:
        lines.append(f"Level {instance.level}/{skill.max_level} (max)")

    if skill.cost:
        lines.append("Cost: " + ", ".join(f"{k.upper()} {v}" for k, v in skill.cost.items()))
    if skill.cooldown:
        lines.append(f"Cooldown: {skill.cooldown} turns")
    if skill.tags:
        lines.append("Tags: " + ", ".join(skill.tags))
    for effect in skill.effects:
        spec = (getattr(effect, "params", None) or {}).get("weapon_scaling")
        if spec:
            tag = spec.get("weapon_tag")
            lines.append(f"+{spec.get('percent', 100)}% weapon damage" + (f" ({tag} weapons)" if tag else ""))
    lines.append(f"Rarity: {skill.rarity}")
    if skill.tree is not None:
        lines.append(f"Self-upgrades claimed: {len(instance.claimed_nodes)}/{len(skill.tree.nodes)}")
    return lines


def item_lines(instance: ItemInstance) -> List[str]:
    definition = instance.definition
    lines: List[str] = []
    if definition.description:
        lines += [definition.description, ""]

    lines.append(f"Rarity: {definition.rarity}")
    if definition.tags:
        lines.append("Tags: " + ", ".join(definition.tags))

    if isinstance(definition, WeaponDef):
        lines.append(f"Slot: {definition.slot}")
        lines.append(f"Damage formula: {_format_formula(definition.damage_formula)}")
        lines.append(f"Crit rate formula: {_format_formula(definition.crit_rate_formula)}")
        lines.append(f"Crit damage formula: {_format_formula(definition.crit_damage_formula)}")
        if definition.is_ranged:
            lines.append(f"Range: {definition.range} (fire with F)")
        if definition.mp_cost:
            lines.append(f"MP cost per attack: {definition.mp_cost}")
        if definition.ammo_type:
            lines.append(f"Requires ammo: {definition.ammo_type}")
    elif isinstance(definition, EquippableDef):
        lines.append(f"Slot: {definition.slot}")
        if definition.defense:
            lines.append(f"Defense: {definition.defense}")
        if definition.magic_defense:
            lines.append(f"Magic Defense: {definition.magic_defense}")

    if isinstance(definition, EquippableDef):
        lines.append(f"Enchant slots: {len(instance.enchantments)}/{definition.enchant_slots}")
        if definition.granted_skills:
            lines.append(
                "Grants while equipped: "
                + ", ".join(skill_data.skill(s).name for s in definition.granted_skills)
            )

    if isinstance(definition, UsableDef):
        lines.append("Consumable" if definition.consumable else "Reusable")
        if instance.charges_remaining is not None:
            lines.append(f"Charges: {instance.charges_remaining}")

    if isinstance(definition, CollectibleDef) and definition.passive_effects:
        lines.append(f"Passive effects while held: {len(definition.passive_effects)}")

    if instance.quantity > 1:
        lines.append(f"Quantity: {instance.quantity}")
    return lines


def class_lines(class_def: "skill_data.ClassDef") -> List[str]:
    lines: List[str] = []
    if class_def.description:
        lines += [class_def.description, ""]
    lines.append(f"Max level: {class_def.max_level}")
    if class_def.unlock_cost_gold:
        lines.append(f"Unlock cost: {class_def.unlock_cost_gold} gold")
    lines.append(f"Tree nodes: {len(class_def.tree.nodes)}")
    return lines


def tree_node_lines(node) -> List[str]:
    lines: List[str] = []
    if node.description:
        lines += [node.description, ""]
    lines.append(f"Requires level {node.requires_level}")
    if node.prereqs:
        lines.append("Requires: " + ", ".join(node.prereqs))
    if node.grants_skills:
        lines.append(
            "Grants skills: " + ", ".join(skill_data.skill(s).name for s in node.grants_skills)
        )
    if node.grants_upgrades:
        lines.append("Grants upgrades: " + ", ".join(node.grants_upgrades))
    return lines


@dataclass
class SheetRow:
    """One line of the character sheet's progression column. Headers are
    plain labels; selectable rows (classes, skills) can be cursored onto and
    inspected — the character sheet reuses the same tooltip/inspect model
    as the DialogueMenus."""

    text: str
    header: bool = False
    selectable: bool = False
    inspect: Optional[Callable[[], List[str]]] = None


def character_sheet_rows(entity) -> List[SheetRow]:
    rows: List[SheetRow] = []

    progression = entity.progression
    if progression is not None:
        species = skill_data.species_def(progression.species_id).name
        rows.append(SheetRow("Progression", header=True))
        rows.append(SheetRow(f"Lv.{progression.species_level} {species}"))
        rows.append(SheetRow(f"Class slots {len(progression.classes)}/{progression.max_class_slots}"))
        for progress in progression.classes:
            rows.append(
                SheetRow(
                    f"{progress.definition.name} Lv.{progress.level}",
                    selectable=True,
                    inspect=lambda cd=progress.definition: class_lines(cd),
                )
            )
        rows.append(SheetRow(""))

    book = entity.skill_book
    if book is not None:
        for category, header in (
            (CATEGORY_ACTIVE, "Active Skills"),
            (CATEGORY_PASSIVE, "Passives"),
            (CATEGORY_PROFICIENCY, "Proficiencies"),
        ):
            instances = book.by_category(category)
            if not instances:
                continue
            rows.append(SheetRow(header, header=True))
            for instance in sorted(instances, key=lambda s: s.resolved.name):
                rows.append(
                    SheetRow(
                        instance.resolved.name,
                        selectable=True,
                        inspect=lambda i=instance: skill_lines(i),
                    )
                )
            rows.append(SheetRow(""))
    return rows
