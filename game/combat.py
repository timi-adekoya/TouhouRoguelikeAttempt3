from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Tuple

from components import progression as prog
from components.items import evaluate_formula
from components.stats import apply_stat_modifier
from game import equipment, message_log
from game.entity import Entity

if TYPE_CHECKING:
    from game.engine import Engine

UNARMED_DAMAGE = 2
UNARMED_NAME = "an unarmed strike"

# Recognized element tags — reused directly as skill/weapon `tags` entries
# (e.g. Flame Bullet is tagged ["magic", "fire"]). A `resist_<element>` combat
# modifier (whole percentage points, like crit_rate) scales both damage taken
# from that element and the chance to succumb to that element's status
# effects — 50 = 50% less damage/50% harder to inflict, -50 = weakness.
ELEMENTS = {"fire", "ice", "earth", "poison", "light", "lightning", "dark", "holy", "spiritual"}


def elemental_multiplier(target: Entity, tags: list) -> float:
    """Combined damage multiplier from every element tag present that the
    target has a resist/weakness modifier for. 1.0 (no change) if none do."""
    if target.stats is None:
        return 1.0
    multiplier = 1.0
    for tag in tags:
        if tag in ELEMENTS:
            resist = target.stats.modifiers.get(f"resist_{tag}", 0.0)
            multiplier *= max(0.0, 1 - resist / 100)
    return multiplier


def _race_damage_multiplier(attacker: Optional[Entity], target: Entity) -> float:
    """General racial-damage passives (e.g. Reimu's Youkai Buster): a flat
    percentage bonus keyed off the target's own species_id, granted via a
    `stat_modifier` on `damage_vs_<species_id>` — same convention as
    `holy_damage_percent`, just keyed by race instead of an element tag.
    1.0 (no change) if the attacker has no such modifier or the target has
    no species at all."""
    if attacker is None or attacker.stats is None or target.progression is None:
        return 1.0
    bonus = attacker.stats.modifiers.get(f"damage_vs_{target.progression.species_id}", 0.0)
    return 1 + bonus / 100 if bonus else 1.0


def is_alone(engine: "Engine") -> bool:
    """Parsee's Loner-style passives key off this: true whenever the whole
    squad is just the player, no recruited party members at all. Simple
    party-membership check rather than a live proximity/distance scan —
    there's currently no way for an individual party member to wander off
    on their own anyway, so "alone" and "solo run" are the same thing in
    practice. (A future distance-based version, once party members can
    separate from the player, should replace this — not built yet.)"""
    return len(engine.party) == 0


def _distance(a: Entity, b: Entity) -> int:
    """Chebyshev (grid) distance — same metric the rest of the engine uses
    for range/adjacency."""
    return max(abs(a.x - b.x), abs(a.y - b.y))


def _ranged_damage_multiplier(attacker: Optional[Entity], target: Entity) -> float:
    """Skirmisher (Ranger): rewards attacking from a distance. Minor perk —
    `ranged_bonus_percent` flat bonus once `ranged_bonus_distance` tiles of
    separation is reached. Capstone — `distance_bonus_per_tile` (capped by
    `distance_bonus_cap`) scales continuously with however far the attack
    was actually made from. Both read off the attacker; no-op (1.0) for
    anyone without the perk."""
    if attacker is None or attacker.stats is None:
        return 1.0
    multiplier = 1.0
    dist = _distance(attacker, target)

    threshold = attacker.stats.modifiers.get("ranged_bonus_distance", 0.0)
    ranged_bonus = attacker.stats.modifiers.get("ranged_bonus_percent", 0.0)
    if ranged_bonus > 0 and threshold > 0 and dist >= threshold:
        multiplier *= 1 + ranged_bonus / 100

    per_tile = attacker.stats.modifiers.get("distance_bonus_per_tile", 0.0)
    if per_tile > 0:
        bonus_percent = dist * per_tile
        cap = attacker.stats.modifiers.get("distance_bonus_cap", 0.0)
        if cap > 0:
            bonus_percent = min(bonus_percent, cap)
        multiplier *= 1 + bonus_percent / 100

    return multiplier


def _apply_crit_bonuses(engine: "Engine", attacker: Optional[Entity], target: Entity) -> int:
    """Marksman (Ranger), called once a hit has already rolled a crit.
    Capstone — flat bonus damage if `target` is still at full HP (rewards
    opening an engagement decisively); returned so the caller folds it into
    the hit like any other damage component. Mid perk — a chance to stack a
    decaying bonus-crit-damage buff ("Focus") on the attacker; each stack is
    its own independent timed stat_mod, so multiple procs naturally overlap
    and expire independently rather than needing separate stack-counting."""
    bonus = 0
    if attacker is None or attacker.stats is None:
        return bonus

    full_hp_bonus = attacker.stats.modifiers.get("full_hp_crit_bonus_flat", 0.0)
    if (
        full_hp_bonus > 0
        and target.stats is not None
        and target.stats.hp.current >= target.stats.hp.max_value
    ):
        bonus += int(full_hp_bonus)

    stack_chance = attacker.stats.modifiers.get("focus_stack_chance", 0.0) / 100
    if stack_chance > 0 and random.random() < stack_chance:
        from game.effects import TimedEffectInstance  # deferred: avoids a module cycle with effects.py

        stack_amount = attacker.stats.modifiers.get("focus_stack_amount", 0.0)
        duration = int(attacker.stats.modifiers.get("focus_stack_duration", 2))
        revert = apply_stat_modifier(attacker.stats, "crit_damage", stack_amount)
        engine.timed_effects.append(
            TimedEffectInstance(kind="stat_mod", target=attacker, remaining_turns=duration, payload={"revert": revert})
        )
        if attacker is engine.player:
            engine.message_log.add_message("Focus builds!", color=message_log.INFO_COLOR)

    return bonus


def weapon_contribution(caster: Entity, spec: Optional[dict]) -> float:
    """Extra skill damage from the caster's equipped weapon (or natural
    attack): `spec["percent"]` of the weapon's own damage formula, only if
    the weapon carries `spec["weapon_tag"]` when one is given (so Aimed Shot
    only benefits from a ranged weapon, Slash from a melee one). The weapon's
    crit formulas and ammo aren't involved — the skill's own crit applies."""
    if not spec or caster.stats is None:
        return 0.0
    weapon = equipment.equipped_weapon(caster)
    if weapon is None:
        return 0.0
    tag = spec.get("weapon_tag")
    if tag and tag not in weapon.tags:
        return 0.0
    return evaluate_formula(weapon.damage_formula, caster.stats.attributes) * spec.get("percent", 100) / 100


def _has_status(engine: "Engine", entity: Optional[Entity], status: str) -> bool:
    # Local duplicate of effects.has_status — effects.py already imports
    # from combat.py, so importing back would create a cycle.
    if entity is None:
        return False
    return any(t.kind == "status" and t.target is entity and t.payload.get("status") == status for t in engine.timed_effects)


def resolve_hit_chance(
    engine: "Engine",
    attacker: Optional[Entity],
    target: Entity,
    base_accuracy: float,
    hit_rate_formula: Optional[dict],
) -> float:
    """(base accuracy - target's evasion) * hit rate, floored at 1% so
    nothing is ever a guaranteed miss. Evasion is never innate — it only
    exists as a stat_modifier (skills/proficiencies/enchantments), same
    percentage-points convention as crit_rate. Blinded halves both sides:
    a blinded attacker's own accuracy, and a blinded target's evasion
    (hard to hit or dodge when you can't see)."""
    evasion = target.stats.modifiers.get("evasion", 0.0) / 100 if target.stats is not None else 0.0
    if _has_status(engine, target, "blinded"):
        evasion *= 0.5
    if _has_status(engine, attacker, "blinded"):
        base_accuracy *= 0.5
    hit_rate = 1.0
    if hit_rate_formula and attacker is not None and attacker.stats is not None:
        hit_rate = evaluate_formula(hit_rate_formula, attacker.stats.attributes)
    return max(0.01, (base_accuracy - evasion) * hit_rate)


def roll_hit(
    engine: "Engine",
    attacker: Optional[Entity],
    target: Entity,
    *,
    base_accuracy: float = 1.0,
    hit_rate_formula: Optional[dict] = None,
    ignore_evasion: bool = False,
) -> bool:
    if ignore_evasion:
        return True
    hit = random.random() < resolve_hit_chance(engine, attacker, target, base_accuracy, hit_rate_formula)
    if not hit:
        # General-purpose hook: any attack that misses this entity counts
        # as them "evading" it (no separate bookkeeping for exactly why —
        # low attacker accuracy and high target evasion both read the same
        # from here). Reusable well beyond Rogue's own capstone.
        engine.event_bus.emit("on_evaded", {"attacker": attacker, "target": target})
    return hit


def _consume_stealth_bonus(engine: "Engine", attacker: Optional[Entity]) -> float:
    """Sneak Attack (Rogue): if `attacker` is both stealthed and carries the
    sneak-attack passive (a flat `sneak_attack_multiplier` combat modifier,
    e.g. 0.5 for +50%), this attack gets that bonus and Stealth ends right
    now, win or miss — "attacking out of stealth" is the trigger, not
    landing the hit. Returns the damage multiplier to apply (1.0 = no
    bonus, stealth untouched if the attacker wasn't stealthed at all)."""
    if attacker is None or attacker.stats is None:
        return 1.0
    bonus = attacker.stats.modifiers.get("sneak_attack_multiplier", 0.0)
    if bonus <= 0:
        return 1.0
    from game.effects import break_status  # deferred: avoids a module cycle with effects.py

    if not break_status(engine, attacker, "stealthed"):
        return 1.0
    return 1.0 + bonus


def _consume_evasion_stacks(attacker: Optional[Entity]) -> float:
    """Rogue's Reflexes capstone: every evaded attack banks a stack (see
    Engine._on_evaded_stack), each worth `evasion_stack_percent` more
    damage on this attacker's own next hit — consumed here regardless of
    whether this particular attack lands, same "spend the buff on the
    attempt" rule Sneak Attack uses."""
    if attacker is None or attacker.stats is None:
        return 1.0
    stacks = attacker.stats.modifiers.get("evasion_stacks_current", 0)
    if stacks <= 0:
        return 1.0
    per_stack = attacker.stats.modifiers.get("evasion_stack_percent", 0.0)
    attacker.stats.modifiers["evasion_stacks_current"] = 0
    return 1.0 + (stacks * per_stack) / 100


def _apply_loot(engine: "Engine", source: Optional[Entity], target: Entity) -> None:
    """Gold + item drops rolled on a monster's death. Gold is a party-wide
    wallet (engine.gold), not tied to whoever landed the blow — it used to
    go to the killer's own Inventory, which silently stranded gold in an
    AI-controlled ally's inventory the shop never reads from."""
    loot = getattr(target, "loot", None)
    if loot is None:
        return

    gold_min, gold_max = loot.gold_range
    if gold_max > 0:
        amount = random.randint(gold_min, gold_max)
        if amount > 0:
            engine.gold += amount
            engine.message_log.add_message(f"Found {amount} gold.", color=message_log.INFO_COLOR)

    for item_id, chance in loot.drops:
        if random.random() < chance:
            engine.spawn_ground_item(target.x, target.y, item_id)


def _award_species_exp(engine: "Engine", entity: Optional[Entity], amount: float) -> None:
    if entity is None or entity.progression is None:
        return
    notices = entity.progression.gain_species_exp(amount)
    if notices and engine.should_log_progression(entity):
        for notice in notices:
            engine.message_log.add_message(notice, color=message_log.INFO_COLOR, stack=False)
    if notices:  # leveled up: linear species/innate trees may flow further
        engine.auto_claim_linear_trees(entity)
        engine.reapply_species_stat_growth(entity)


def _award_kill_exp(engine: "Engine", source: Optional[Entity], target: Entity, is_boss: bool) -> None:
    """Kill exp scales with the target's own max HP (a natural stand-in for
    "how tough was this" without a hand-tuned per-species table) instead of
    one flat amount for every kill, with an extra multiplier for bosses —
    defeating a boss is a bigger deal than its raw HP alone implies.
    Shared with the whole current party (player + engine.party), not just
    whoever landed the killing blow, so party members who didn't personally
    tag the kill still grow from a group effort."""
    if target.stats is None:
        return
    amount = target.stats.hp.max_value * prog.SPECIES_EXP_KILL_PER_MAX_HP
    if is_boss:
        amount *= prog.SPECIES_EXP_BOSS_KILL_MULTIPLIER
    recipients = {e for e in ([engine.player] + engine.party) if e.progression is not None}
    if source is not None:
        recipients.add(source)
    for entity in recipients:
        _award_species_exp(engine, entity, amount)


def resolve_weapon_attack(
    engine: "Engine", attacker: Entity, target: Entity, *, ranged: bool = False
) -> Optional[Tuple[int, str]]:
    """(damage, cause) a mainhand weapon attack deals, after a hit roll,
    crit roll, and the target's armor/elemental resistance. Returns None if
    the attack can't happen (out of ammo on a ranged shot, insufficient MP,
    or a miss) — the turn is still consumed either way. A melee bump with an
    ammo weapon that's out of ammo falls back to an unarmed strike."""
    weapon = equipment.equipped_weapon(attacker)
    if weapon is not None and not equipment.has_ammo(attacker, weapon):
        if ranged:
            engine.message_log.add_message(f"{attacker.name} is out of ammo!", color=message_log.INFO_COLOR)
            return None
        weapon = None
    stealth_bonus = _consume_stealth_bonus(engine, attacker)
    evasion_stack_bonus = _consume_evasion_stacks(attacker)
    if weapon is None or attacker.stats is None:
        if not roll_hit(engine, attacker, target):
            engine.message_log.add_message(
                f"{attacker.name}'s attack misses {target.name}!", color=message_log.INFO_COLOR
            )
            return None
        damage = float(UNARMED_DAMAGE)
        is_magic = False
        cause = UNARMED_NAME
        weapon_tags: list = []
    else:
        if weapon.ammo_type is not None:
            ammo = attacker.inventory.find_tagged(weapon.ammo_type)
            # Marksman (Ranger) minor perk: a chance to not consume the ammo
            # at all.
            save_chance = attacker.stats.modifiers.get("ammo_save_chance", 0.0) / 100
            if save_chance <= 0 or random.random() >= save_chance:
                attacker.inventory.consume_one(ammo)

        if weapon.mp_cost > 0:
            if attacker.stats.mp.current < weapon.mp_cost:
                engine.message_log.add_message(
                    f"{attacker.name} lacks the MP to attack with {weapon.name}.",
                    color=message_log.INFO_COLOR,
                )
                return None
            attacker.stats.mp.modify(-weapon.mp_cost)

        if not roll_hit(
            engine, attacker, target,
            base_accuracy=weapon.accuracy, hit_rate_formula=weapon.hit_rate_formula,
            ignore_evasion=weapon.ignore_evasion,
        ):
            engine.message_log.add_message(
                f"{attacker.name}'s attack misses {target.name}!", color=message_log.INFO_COLOR
            )
            return None

        damage = evaluate_formula(weapon.damage_formula, attacker.stats.attributes)
        # Stat-modifier-driven crit bonuses (skills, relics, enchants) stack
        # on top of the weapon's own formula, in percentage points.
        crit_chance = max(
            0.0,
            min(
                1.0,
                evaluate_formula(weapon.crit_rate_formula, attacker.stats.attributes)
                + attacker.stats.modifiers.get("crit_rate", 0.0) / 100,
            ),
        )
        if random.random() < crit_chance:
            damage *= evaluate_formula(
                weapon.crit_damage_formula, attacker.stats.attributes
            ) + attacker.stats.modifiers.get("crit_damage", 0.0) / 100
            damage += _apply_crit_bonuses(engine, attacker, target)
            engine.message_log.add_message("Critical hit!", color=message_log.DAMAGE_COLOR)
        is_magic = "magic" in weapon.tags
        cause = weapon.name
        weapon_tags = weapon.tags

    damage *= stealth_bonus
    damage *= evasion_stack_bonus
    damage *= _ranged_damage_multiplier(attacker, target)
    damage *= _race_damage_multiplier(attacker, target)
    if attacker.stats is not None:
        dealt_bonus = attacker.stats.modifiers.get("damage_dealt_percent", 0.0)
        if dealt_bonus:
            damage *= 1 + dealt_bonus / 100
    defense = equipment.total_defense(target, magic=is_magic)
    raw = max(1, int(damage) - defense)
    final = int(raw * elemental_multiplier(target, weapon_tags))
    return max(0, final), cause


def resolve_skill_damage(
    engine: "Engine", caster: Optional[Entity], target: Entity, params: dict, tags: list
) -> Optional[int]:
    """A damage-dealing skill effect that's been given its own
    damage_formula plays by the same rules a weapon attack does: a hit roll,
    scales off the caster's attributes, rolls its own (typically
    slower-scaling) crit, and comes off the target's defense/elemental
    resistance (magic_defense if tagged "magic"). Skills without a
    damage_formula just use their flat `amount` and skip the hit roll
    entirely — content that hasn't been converted yet keeps working
    unchanged. Returns None on a miss (distinct from 0 damage)."""
    damage_formula = params.get("damage_formula")
    if damage_formula is None or caster is None or caster.stats is None:
        return params.get("amount", 0)

    stealth_bonus = _consume_stealth_bonus(engine, caster)
    evasion_stack_bonus = _consume_evasion_stacks(caster)

    # A skill can grant itself a bonus conditioned on another of the
    # caster's own buffs currently being up (e.g. Aimed Shot reading
    # whether Bullseye's crit buff is active) — see effects.py's
    # _stat_modifier, which tags each buff's TimedEffectInstance with the
    # skill_id that granted it.
    conditional = params.get("conditional_bonus")
    conditional_active = False
    if conditional:
        required_source = conditional.get("requires_buff_source")
        conditional_active = any(
            t.kind == "stat_mod" and t.target is caster and t.payload.get("source_skill") == required_source
            for t in engine.timed_effects
        )

    if not roll_hit(
        engine, caster, target,
        base_accuracy=params.get("accuracy", 1.0),
        hit_rate_formula=params.get("hit_rate_formula"),
        ignore_evasion=params.get("ignore_evasion", False)
        or (conditional_active and conditional.get("guarantees_hit", False)),
    ):
        engine.message_log.add_message(
            f"{caster.name}'s attack misses {target.name}!", color=message_log.INFO_COLOR
        )
        return None

    damage = evaluate_formula(damage_formula, caster.stats.attributes)
    damage += weapon_contribution(caster, params.get("weapon_scaling"))
    crit_rate_formula = params.get("crit_rate_formula", {})
    crit_damage_formula = params.get("crit_damage_formula", {"base": 1.5})
    conditional_crit_bonus = conditional.get("crit_rate_bonus", 0.0) if conditional_active else 0.0

    # Per-skill (not general-passive) bonus against one specific species,
    # e.g. Persuasion Needles' own extra crit vs. youkai — stacks with a
    # general racial passive like Youkai Buster rather than replacing it.
    crit_vs_species = params.get("crit_vs_species")
    species_crit_bonus = 0.0
    if (
        crit_vs_species
        and target.progression is not None
        and target.progression.species_id == crit_vs_species.get("species")
    ):
        species_crit_bonus = crit_vs_species.get("percent", 0.0) / 100

    crit_chance = max(
        0.0,
        min(
            1.0,
            evaluate_formula(crit_rate_formula, caster.stats.attributes)
            + caster.stats.modifiers.get("crit_rate", 0.0) / 100
            + conditional_crit_bonus
            + species_crit_bonus,
        ),
    )
    if random.random() < crit_chance:
        damage *= evaluate_formula(
            crit_damage_formula, caster.stats.attributes
        ) + caster.stats.modifiers.get("crit_damage", 0.0) / 100
        damage += _apply_crit_bonuses(engine, caster, target)
        engine.message_log.add_message("Critical hit!", color=message_log.DAMAGE_COLOR)

    damage *= stealth_bonus
    damage *= evasion_stack_bonus
    damage *= _ranged_damage_multiplier(caster, target)
    damage *= _race_damage_multiplier(caster, target)

    # Per-skill (not general-passive) bonus damage against one specific
    # species, e.g. Purification Ofuda's own extra damage vs. youkai on top
    # of the general Youkai Buster passive above.
    bonus_vs_species = params.get("bonus_vs_species")
    if (
        bonus_vs_species
        and target.progression is not None
        and target.progression.species_id == bonus_vs_species.get("species")
    ):
        damage *= 1 + bonus_vs_species.get("percent", 0.0) / 100

    dealt_bonus = caster.stats.modifiers.get("damage_dealt_percent", 0.0)
    if dealt_bonus:
        damage *= 1 + dealt_bonus / 100

    # Parsee's Loner-style skill-damage passive: a much bigger bonus solo,
    # a token one in a party. Skills only (not weapon attacks), per her kit.
    loner_key = "skill_damage_bonus_alone" if is_alone(engine) else "skill_damage_bonus_party"
    loner_bonus = caster.stats.modifiers.get(loner_key, 0.0)
    if loner_bonus:
        damage *= 1 + loner_bonus / 100

    # Flat % damage boost to a specific element, e.g. Battle Cleric's
    # `holy_damage_percent` or Kisume's `fire_damage_percent` — reads
    # `<element>_damage_percent` generically off whichever element tags are
    # actually present, so any current or future elemental passive picks up
    # any matching skill automatically, no per-skill wiring needed.
    for tag in tags:
        if tag in ELEMENTS:
            element_bonus = caster.stats.modifiers.get(f"{tag}_damage_percent", 0.0)
            if element_bonus > 0:
                damage *= 1 + element_bonus / 100

    defense = equipment.total_defense(target, magic="magic" in tags)
    raw = max(1, int(damage) - defense)
    return max(0, int(raw * elemental_multiplier(target, tags)))


def apply_damage(
    engine: "Engine",
    source: Optional[Entity],
    target: Entity,
    amount: int,
    *,
    cause: Optional[str] = None,
    periodic: bool = False,
    tags: Optional[list] = None,
    _reflected: bool = False,
) -> int:
    """The one path all damage takes: HP loss, events, species exp for the
    contributor, and death handling (including the player's own death).
    Returns damage actually dealt.

    `cause` names what actually did the damage (a weapon, a skill, a status
    like "burning") so the log reads as "why", not just "who hit whom".
    `periodic` phrases DoT ticks as passive suffering rather than an active
    hit, since nothing is "swinging" on those turns. `tags` (e.g. "melee")
    lets a source-side on-hit effect (lifesteal) key off what kind of hit
    this was — DoT ticks never pass tags, so they never proc it. `_reflected`
    marks damage that's already a Warding reflection bounce, so it can't
    itself trigger another reflection back and forth forever.
    """
    if target.stats is None:
        return 0

    # Already dead (0 HP) before this hit even lands — a DoT tick, a
    # multi-effect skill's second effect, or a stale queued action can all
    # still reference an entity that died earlier in the same batch of
    # processing (removed from `engine.entities`, but nothing clears the
    # Python object itself). Without this, the hit would deal 0 damage
    # (nothing left to subtract) but still run the FULL death pipeline
    # again — a second "is defeated!" message, a second loot roll, a second
    # species-exp award, a second on_kill emission. Bail out first, no
    # further effects/events/logging at all.
    if target.stats.hp.is_empty:
        return 0

    # Warding (Mage) capstone: toggleable — bounces a portion of the RAW
    # incoming hit back at whoever dealt it, before any of the target's own
    # mitigation (Endurance, Reckless Stance's downside, Mana Shield, Ward)
    # touches it. Off by default (0%); flips on/off via toggle_passive.
    reflect_percent = target.stats.modifiers.get("reflect_percent", 0.0)
    if (
        not periodic
        and not _reflected
        and reflect_percent > 0
        and source is not None
        and source is not target
        and source.stats is not None
    ):
        reflected_amount = int(amount * reflect_percent / 100)
        if reflected_amount > 0:
            apply_damage(engine, target, source, reflected_amount, cause="reflected damage", _reflected=True)

    # Endurance (Warrior): mild, HP-scaling damage reduction — the lower
    # the target's own HP, the more of any incoming hit gets shaved off.
    # `hp_scaling_mitigation_max` is the reduction % at 0 HP; it scales
    # linearly with the fraction of HP already missing, 0% at full HP.
    mitigation_max = target.stats.modifiers.get("hp_scaling_mitigation_max", 0.0)
    if mitigation_max > 0 and target.stats.hp.max_value > 0:
        missing_fraction = 1 - (target.stats.hp.current / target.stats.hp.max_value)
        reduction = max(0.0, min(1.0, mitigation_max * missing_fraction / 100))
        amount = int(amount * (1 - reduction))

    # Aggression (Warrior) Reckless Stance: while active, take a flat %
    # more damage in exchange for damage_dealt_percent (see
    # resolve_weapon_attack/resolve_skill_damage) — a generic modifier, not
    # Warrior-exclusive, so anything else that wants a risk/reward toggle
    # can reuse it too.
    taken_bonus = target.stats.modifiers.get("damage_taken_percent", 0.0)
    if taken_bonus:
        amount = int(amount * (1 + taken_bonus / 100))

    # Warding (Mage) mid perk: temporary HP banked from an earlier hit (see
    # the ward proc below) absorbs before anything else, including Mana
    # Shield — it's a shield, not a resource conversion.
    temp_hp = target.stats.modifiers.get("temp_hp_current", 0)
    if temp_hp > 0:
        temp_absorbed = min(temp_hp, amount)
        target.stats.modifiers["temp_hp_current"] = temp_hp - temp_absorbed
        amount -= temp_absorbed

    # Mana Shield (Mage): while active, a percentage of incoming damage
    # comes out of MP instead of HP, capped by whatever MP is actually
    # available — running out of MP just means the rest falls through to
    # HP as normal, no separate failure state.
    shield_percent = target.stats.modifiers.get("mana_shield_percent", 0.0)
    hp_amount = amount
    mp_absorbed = 0
    if shield_percent > 0:
        shielded = int(amount * shield_percent / 100)
        mp_absorbed = min(shielded, target.stats.mp.current)
        hp_amount = amount - mp_absorbed
        if mp_absorbed > 0:
            target.stats.mp.modify(-mp_absorbed)

    dealt = -target.stats.hp.modify(-hp_amount)
    if mp_absorbed > 0:
        engine.message_log.add_message(
            f"{target.name}'s Mana Shield absorbs {mp_absorbed} damage as MP loss.",
            color=message_log.INFO_COLOR,
        )
    source_name = source.name if source is not None else "Something"
    if periodic:
        text = f"{target.name} suffers {dealt} damage from {cause or 'an affliction'}."
    elif cause:
        text = f"{source_name} hits {target.name} with {cause} for {dealt}."
    else:
        text = f"{source_name} hits {target.name} for {dealt}."
    engine.message_log.add_message(text, color=message_log.DAMAGE_COLOR)

    # Warding (Mage) mid perk: a portion of the damage that just landed
    # (post-mitigation, whatever actually got through) is banked as
    # temporary HP for `ward_duration` turns — absorbed first on the next
    # hit(s), see above. Unspent temp HP just expires, it doesn't carry over.
    ward_percent = target.stats.modifiers.get("ward_percent", 0.0)
    if not periodic and ward_percent > 0 and dealt > 0:
        from game.effects import TimedEffectInstance  # deferred: avoids a module cycle with effects.py

        shield_gain = int(dealt * ward_percent / 100)
        if shield_gain > 0:
            target.stats.modifiers["temp_hp_current"] = (
                target.stats.modifiers.get("temp_hp_current", 0) + shield_gain
            )
            duration = int(target.stats.modifiers.get("ward_duration", 2))
            engine.timed_effects.append(
                TimedEffectInstance(kind="temp_hp", target=target, remaining_turns=duration, payload={"amount": shield_gain})
            )

    engine.event_bus.emit(
        "on_damaged", {"source": source, "target": target, "amount": dealt, "cause": cause}
    )
    _award_species_exp(engine, source, dealt * prog.SPECIES_EXP_PER_DAMAGE)

    if target.stats.hp.is_empty:
        engine.event_bus.emit("on_kill", {"source": source, "target": target})
        if target is engine.player or target in engine.party:
            engine.handle_character_down(target)
        elif target is engine.pending_boss:
            _award_kill_exp(engine, source, target, is_boss=True)
            engine.handle_boss_defeated(target)
        else:
            engine.message_log.add_message(
                f"{target.name} is defeated!", color=message_log.DEATH_COLOR, stack=False
            )
            _award_kill_exp(engine, source, target, is_boss=False)
            _apply_loot(engine, source, target)
            engine.remove_entity(target)

    # Aggression (Warrior): lifesteal on melee hits — flat + a percentage
    # of damage actually dealt (post-mitigation, post-mana-shield), healed
    # back to the source. Gated on the "melee" tag so DoT ticks (which
    # never pass tags) and non-melee skills never proc it.
    if (
        not periodic
        and tags
        and "melee" in tags
        and source is not None
        and source.stats is not None
        and dealt > 0
    ):
        lifesteal_flat = source.stats.modifiers.get("lifesteal_flat", 0.0)
        lifesteal_percent = source.stats.modifiers.get("lifesteal_percent", 0.0)
        if lifesteal_flat > 0 or lifesteal_percent > 0:
            heal_amount = int(lifesteal_flat + dealt * lifesteal_percent / 100)
            if heal_amount > 0:
                apply_heal(engine, source, source, heal_amount, cause="lifesteal")

    # Battle Cleric: holy hits have a chance to weaken the target's evasion
    # and/or blind them outright. Both read generically off "holy" tags and
    # the source's own modifiers, same pattern as lifesteal above — no
    # per-skill wiring, any current or future holy-tagged skill picks it up.
    if not periodic and tags and "holy" in tags and source is not None and source.stats is not None and dealt > 0 and target.stats is not None:
        from game.effects import TimedEffectInstance  # deferred: effects.py imports from combat.py

        evasion_chance = source.stats.modifiers.get("holy_evasion_debuff_chance", 0.0) / 100
        if evasion_chance > 0 and random.random() < evasion_chance:
            debuff_amount = -abs(source.stats.modifiers.get("holy_evasion_debuff_amount", 0.0))
            duration = int(source.stats.modifiers.get("holy_evasion_debuff_duration", 1))
            revert = apply_stat_modifier(target.stats, "evasion", debuff_amount)
            engine.timed_effects.append(
                TimedEffectInstance(kind="stat_mod", target=target, remaining_turns=duration, payload={"revert": revert}, source=source)
            )
            engine.message_log.add_message(f"{target.name}'s guard falters!", color=message_log.INFO_COLOR)

        blind_chance = source.stats.modifiers.get("holy_blind_chance", 0.0) / 100
        if blind_chance > 0 and random.random() < blind_chance:
            duration = int(source.stats.modifiers.get("holy_blind_duration", 1))
            engine.timed_effects.append(
                TimedEffectInstance(kind="status", target=target, remaining_turns=duration, payload={"status": "blinded"}, source=source)
            )
            engine.message_log.add_message(f"{target.name} is blinded by holy light!", color=message_log.INFO_COLOR)

    return dealt


# Devout Cleric's own bonus procs (self-heal-on-heal, portion-of-max-hp
# overflow) call apply_heal recursively — tagging their own cause strings
# and skipping the Devout block on them is what stops that from looping.
_DEVOUT_BONUS_CAUSES = {"Restorative Echo", "Sacred Overflow"}


def apply_heal(
    engine: "Engine", source: Optional[Entity], target: Entity, amount: int, *, cause: Optional[str] = None
) -> int:
    """The one path all healing takes; heals give a small fixed species exp."""
    if target.stats is None:
        return 0

    is_devout_bonus_proc = cause in _DEVOUT_BONUS_CAUSES

    if not is_devout_bonus_proc and source is not None and source.stats is not None:
        healing_power_percent = source.stats.modifiers.get("healing_power_percent", 0.0)
        if healing_power_percent > 0:
            amount = int(amount * (1 + healing_power_percent / 100))

    healed = target.stats.hp.modify(amount)
    if healed > 0:
        text = (
            f"{target.name} recovers {healed} HP from {cause}."
            if cause
            else f"{target.name} recovers {healed} HP."
        )
        engine.message_log.add_message(text, color=message_log.INFO_COLOR)
        engine.event_bus.emit(
            "on_healed", {"source": source, "target": target, "amount": healed, "cause": cause}
        )
        _award_species_exp(engine, source, prog.SPECIES_EXP_HEAL)

        # Devout Cleric: healing anyone (including self) also heals the
        # caster a flat amount; the capstone additionally heals the target
        # for a portion of their own max HP, floored at 1.
        if not is_devout_bonus_proc and source is not None and source.stats is not None:
            self_heal_flat = source.stats.modifiers.get("devout_self_heal_flat", 0.0)
            if self_heal_flat > 0:
                apply_heal(engine, source, source, int(self_heal_flat), cause="Restorative Echo")

            portion_percent = source.stats.modifiers.get("devout_portion_percent", 0.0)
            if portion_percent > 0 and target.stats.hp.max_value > 0:
                bonus = max(1, int(target.stats.hp.max_value * portion_percent / 100))
                apply_heal(engine, source, target, bonus, cause="Sacred Overflow")

    return healed
