from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from components import progression as prog
from components.items import evaluate_formula
from components.skills import EffectDef, MovementEffectDef
from components.stats import apply_stat_modifier, revert_stat_modifier
from game import combat, message_log
from game.entity import Entity
from game.targeting import TargetPoint

if TYPE_CHECKING:
    from game.casting import CastContext
    from game.engine import Engine


class TimedEffectInstance:
    """Shared persistence container for DoT/HoT, temporary stat mods, and
    status conditions. One tick loop (TurnManager) handles all three kinds."""

    def __init__(
        self,
        kind: str,  # dot | hot | stat_mod | status
        target: Entity,
        remaining_turns: int,
        payload: Dict[str, Any],
        source: Optional[Entity] = None,
        stacks: int = 1,
    ):
        self.kind = kind
        self.target = target
        self.remaining_turns = remaining_turns
        self.payload = payload
        self.source = source
        self.stacks = stacks

    def tick(self, engine: "Engine") -> None:
        if self.kind == "dot":
            amount = self.payload["amount"] * self.stacks
            amount = int(amount * combat.elemental_multiplier(self.target, self.payload.get("tags", [])))
            combat.apply_damage(
                engine, self.source, self.target,
                amount,
                cause=self.payload.get("name") or "an affliction",
                periodic=True,
            )
        elif self.kind == "hot":
            combat.apply_heal(
                engine, self.source, self.target,
                self.payload["amount"] * self.stacks,
                cause=self.payload.get("name") or "regeneration",
            )
        self.remaining_turns -= 1

    def on_expire(self, engine: "Engine") -> None:
        if self.kind == "stat_mod":
            revert_stat_modifier(self.target.stats, self.payload["revert"])
        elif self.kind == "temp_hp":
            # Whatever's left of this ward's shield (it may have already
            # been partly/fully spent absorbing hits) just falls off.
            current = self.target.stats.modifiers.get("temp_hp_current", 0)
            self.target.stats.modifiers["temp_hp_current"] = max(0, current - self.payload["amount"])


# ---------------------------------------------------------------------------
# Effect executors. Adding a new effect behavior = one registration here.
# Signature: (engine, ctx, caster, target_point, effect) -> None
# ---------------------------------------------------------------------------

Executor = Callable[["Engine", "CastContext", Entity, TargetPoint, EffectDef], None]
EFFECT_EXECUTORS: Dict[str, Executor] = {}


def register(effect_type: str) -> Callable[[Executor], Executor]:
    def wrap(fn: Executor) -> Executor:
        EFFECT_EXECUTORS[effect_type] = fn
        return fn

    return wrap


@register("instant_damage_heal")
def _instant_damage_heal(engine, ctx, caster, tp, effect):
    if tp.entity is None:
        return
    if effect.params.get("mode", "damage") == "heal":
        combat.apply_heal(engine, caster, tp.entity, effect.params["amount"], cause=ctx.skill.name)
    else:
        # A detonation-style skill (Parsee's Grudge) only fires if the
        # target carries a specific status — consuming it in the process.
        # Without the mark, the cast is a clean whiff: no fallback damage,
        # so mistiming it is a real cost, not just a smaller hit. An
        # optional `min_stacks` additionally requires the status to have
        # built up to at least that many stacks (see `_status_condition`'s
        # `max_stacks`) before it can be detonated at all — and however
        # many stacks it actually had multiplies the damage.
        requires_status = effect.params.get("requires_status")
        stack_multiplier = 1
        if requires_status:
            match = next(
                (t for t in engine.timed_effects if t.kind == "status" and t.target is tp.entity and t.payload.get("status") == requires_status["status"]),
                None,
            )
            min_stacks = requires_status.get("min_stacks", 1)
            if match is None or match.stacks < min_stacks:
                engine.message_log.add_message(
                    f"{ctx.skill.name} finds nothing to detonate.", color=message_log.INFO_COLOR
                )
                return
            stack_multiplier = match.stacks
            if requires_status.get("consume", True):
                engine.timed_effects.remove(match)
                engine.message_log.add_message(
                    f"{tp.entity.name} is no longer {requires_status['status']}.", color=message_log.INFO_COLOR
                )

        resolved = combat.resolve_skill_damage(
            engine, caster, tp.entity, effect.params, ctx.skill.tags, cast_bonuses=ctx.damage_bonuses
        )
        if resolved is None:
            return  # missed — no damage, no on_hit chain
        amount, pre_defense = resolved
        amount = int(amount * stack_multiplier)
        dealt = combat.apply_damage(
            engine, caster, tp.entity, amount, cause=ctx.skill.name, tags=ctx.skill.tags,
            pre_defense=int(pre_defense * stack_multiplier),
        )
        if dealt > 0:
            ctx.emit("on_hit", tp)


def resolve_status_resist_chance(target: Entity, resist_formula: Optional[Dict[str, Any]], tags: list) -> float:
    """Percentage chance the target shrugs off a status/dot effect. Driven
    entirely by the effect's own `resist_formula` (evaluated against the
    target's own attributes — usually spirit/vitality/presence); there's no
    separate generic "spell resistance" stat to raise, matching the design:
    resistance is inherent to the effect, only specific counters (immunity
    skills, etc.) intervene. An elemental weakness/resistance tag shared
    with the effect nudges this directly — weakness lowers the resist
    chance (easier to succumb), resistance raises it."""
    base = 0.0
    if resist_formula and target.stats is not None:
        base = evaluate_formula(resist_formula, target.stats.attributes)
    elemental_adjust = 0.0
    if target.stats is not None:
        for tag in tags:
            if tag in combat.ELEMENTS:
                elemental_adjust += target.stats.modifiers.get(f"resist_{tag}", 0.0) / 100
    return max(0.0, min(1.0, base + elemental_adjust))


@register("dot_hot")
def _dot_hot(engine, ctx, caster, tp, effect):
    if tp.entity is None:
        return
    mode = effect.params.get("mode", "dot")
    if mode == "dot":
        resist_chance = resolve_status_resist_chance(tp.entity, effect.params.get("resist_formula"), ctx.skill.tags)
        if random.random() < resist_chance:
            engine.message_log.add_message(
                f"{tp.entity.name} resists the affliction!", color=message_log.INFO_COLOR
            )
            return
    duration = effect.params["duration"]
    stacks = 1
    if mode == "dot" and caster is not None and caster.stats is not None:
        # Rogue's Bleed aspect: a flat duration bonus on every DoT the
        # caster applies, and a % chance to land it as 2 stacks instead of
        # 1 (TimedEffectInstance.tick already multiplies tick damage by
        # `stacks`, so this needs no separate damage math).
        duration += int(caster.stats.modifiers.get("dot_duration_bonus", 0))
        double_chance = caster.stats.modifiers.get("double_stack_chance", 0.0) / 100
        if double_chance > 0 and random.random() < double_chance:
            stacks = 2

    engine.timed_effects.append(
        TimedEffectInstance(
            kind=mode,
            target=tp.entity,
            remaining_turns=duration,
            payload={
                "amount": effect.params["amount"], "name": effect.params.get("name", ""),
                "tags": ctx.skill.tags,
            },
            source=caster,
            stacks=stacks,
        )
    )
    name = effect.params.get("name")
    if name:
        engine.message_log.add_message(f"{tp.entity.name} is afflicted by {name}.")

    # Rogue capstone: whenever a caster with the "exposed wound" passive
    # (a flat bleed_mark_damage/bleed_mark_duration modifier pair) lands a
    # bleed, also mark the target — see Engine._on_damaged_bleed_mark for
    # the actual per-hit proc.
    if mode == "dot" and name == "bleeding" and caster is not None and caster.stats is not None:
        mark_duration = caster.stats.modifiers.get("bleed_mark_duration", 0)
        mark_damage = caster.stats.modifiers.get("bleed_mark_damage", 0)
        if mark_duration > 0 and mark_damage > 0:
            engine.timed_effects.append(
                TimedEffectInstance(
                    kind="status", target=tp.entity, remaining_turns=int(mark_duration),
                    payload={"status": "marked", "bonus_damage": mark_damage}, source=caster,
                )
            )
            engine.message_log.add_message(f"{tp.entity.name}'s wound is exposed!", color=message_log.INFO_COLOR)


@register("stat_modifier")
def _stat_modifier(engine, ctx, caster, tp, effect):
    if tp.entity is None or tp.entity.stats is None:
        return
    attribute = effect.params["attribute"]
    if "formula" in effect.params:
        amount = evaluate_formula(effect.params["formula"], tp.entity.stats.attributes)
    else:
        amount = effect.params["amount"]
    duration = effect.params.get("duration", -1)

    # `max_stacks` (optional): caps how many of THIS skill's own buffs on
    # this attribute/target can be active at once — recasting past the cap
    # refreshes (reverts the oldest, then applies fresh) instead of piling
    # on another independent instance. Unset (the default, every
    # pre-existing skill) means unlimited stacking, unchanged from before —
    # some buffs (Focus, evasion stacks) are *meant* to compound. Others
    # (Bullseye, Mana Shield) aren't: recasting before the previous cast
    # expired was silently doubling/tripling their effect, since nothing
    # stopped two live instances from the same skill both applying.
    max_stacks = effect.params.get("max_stacks")
    if max_stacks is not None and duration > 0:
        existing = [
            t
            for t in engine.timed_effects
            if t.kind == "stat_mod"
            and t.target is tp.entity
            and t.payload.get("source_skill") == ctx.skill.skill_id
            and t.payload.get("attribute") == attribute
        ]
        while len(existing) >= max_stacks:
            oldest = existing.pop(0)
            revert_stat_modifier(tp.entity.stats, oldest.payload["revert"])
            engine.timed_effects.remove(oldest)

    revert_payload = apply_stat_modifier(tp.entity.stats, attribute, amount)

    if duration > 0:
        engine.timed_effects.append(
            TimedEffectInstance(
                kind="stat_mod",
                target=tp.entity,
                remaining_turns=duration,
                # source_skill lets another skill's conditional_bonus (see
                # combat.resolve_skill_damage) check "is Bullseye's buff up
                # on me right now" without hardcoding a cross-skill link.
                payload={"revert": revert_payload, "source_skill": ctx.skill.skill_id, "attribute": attribute},
                source=caster,
            )
        )

    # Announce it — previously a cast buff/debuff (self or otherwise) never
    # logged anything at all, so e.g. Shield Wall going off was invisible.
    # Passive stat_modifier effects don't run through this executor (they're
    # applied directly by Engine._reapply_passive), so this only covers
    # active skill/item casts, not passive noise every level-up/turn.
    verb = "rises" if amount > 0 else "falls"
    label = attribute.replace("_", " ").title()
    engine.message_log.add_message(
        f"{tp.entity.name}'s {label} {verb} by {abs(amount)}.", color=message_log.INFO_COLOR, stack=False
    )

    # A positive modifier on someone else is a buff: small fixed species exp.
    if amount > 0 and tp.entity is not caster and caster is not None and caster.progression is not None:
        for notice in caster.progression.gain_species_exp(prog.SPECIES_EXP_BUFF):
            engine.message_log.add_message(notice, color=message_log.INFO_COLOR, stack=False)


@register("status_condition")
def _status_condition(engine, ctx, caster, tp, effect):
    if tp.entity is None:
        return
    resist_chance = resolve_status_resist_chance(tp.entity, effect.params.get("resist_formula"), ctx.skill.tags)
    if random.random() < resist_chance:
        engine.message_log.add_message(
            f"{tp.entity.name} resists the effect!", color=message_log.INFO_COLOR
        )
        return
    status = effect.params["status"]
    duration = effect.params["duration"]
    # `max_stacks` (optional, default 1 — every pre-existing status is
    # unaffected): reapplying past the default cap of 1 normally just adds
    # a second independent instance (fine for e.g. webbed/blinded, which
    # don't care about stacking). When set above 1, reapplying instead
    # deepens the SAME instance's stack count (capped) and refreshes its
    # duration, rather than piling up redundant copies.
    max_stacks = effect.params.get("max_stacks", 1)
    existing = None
    if max_stacks > 1:
        existing = next(
            (t for t in engine.timed_effects if t.kind == "status" and t.target is tp.entity and t.payload.get("status") == status),
            None,
        )

    if existing is not None:
        existing.stacks = min(max_stacks, existing.stacks + 1)
        existing.remaining_turns = duration
        engine.message_log.add_message(f"{tp.entity.name}'s {status} deepens ({existing.stacks}/{max_stacks}).")
    else:
        engine.timed_effects.append(
            TimedEffectInstance(
                kind="status",
                target=tp.entity,
                remaining_turns=duration,
                payload={"status": status},
                source=caster,
            )
        )
        engine.message_log.add_message(f"{tp.entity.name} is {status}!")

    if effect.params.get("interrupts_charge") and tp.entity.charging is not None:
        from game.casting import interrupt_charge  # deferred: avoids a module cycle with casting.py

        interrupt_charge(engine, tp.entity)

    # Rogue's Reflexes mid perk: gaining Stealth grants a % head start
    # toward the caster's own next turn (only ever fires on themself, since
    # Stealth is self-target, but keyed generically off any "stealthed"
    # application in case something else ever grants it to someone else).
    if status == "stealthed" and caster is not None and caster.stats is not None:
        from game.turn_queue import ACTION_COST  # deferred: avoids a module cycle with turn_queue's own imports

        tempo_percent = caster.stats.modifiers.get("stealth_tempo_percent", 0.0)
        if tempo_percent > 0:
            caster.energy += ACTION_COST * tempo_percent / 100


@register("steal_buff")
def _steal_buff(engine, ctx, caster, tp, effect):
    """Rip an active positive stat_mod buff off the target and re-apply the
    same attribute/amount to the caster instead, for its remaining
    duration. A no-op (not an error) if the target has no buff active right
    now — Parsee's kit accepts that some casts whiff."""
    if tp.entity is None or tp.entity.stats is None or caster.stats is None:
        return
    candidates = [
        t for t in engine.timed_effects
        if t.kind == "stat_mod" and t.target is tp.entity and t.payload["revert"].get("applied", 0) > 0
    ]
    if not candidates:
        return

    stolen = random.choice(candidates)
    engine.timed_effects.remove(stolen)
    revert_stat_modifier(tp.entity.stats, stolen.payload["revert"])

    name = stolen.payload["revert"]["name"]
    amount = stolen.payload["revert"]["applied"]
    new_payload = apply_stat_modifier(caster.stats, name, amount)
    engine.timed_effects.append(
        TimedEffectInstance(
            kind="stat_mod", target=caster, remaining_turns=stolen.remaining_turns,
            payload={"revert": new_payload}, source=caster,
        )
    )
    engine.message_log.add_message(
        f"{caster.name} steals {tp.entity.name}'s buff!", color=message_log.INFO_COLOR, stack=False
    )


@register("summon_clone")
def _summon_clone(engine, ctx, caster, tp, effect):
    """Spawn a weaker, hostile copy of the caster next to them — stats and
    attributes scaled by `params.fraction` (default half), and a skill_book
    copied from the caster's own (minus any skill that would let the clone
    summon further clones). No Character component, so it's plain-hostile
    like any other monster and isn't recruitable on defeat."""
    if caster.stats is None:
        return
    from game.ai import BehaviorTreeController, find_adjacent_free_tile  # deferred: ai -> casting -> effects would cycle
    from components.skills import SkillBook, SkillInstance
    from components.stats import Attributes, StatBlock

    spawn_pos = find_adjacent_free_tile(engine, caster)
    if spawn_pos is None:
        return

    fraction = effect.params.get("fraction", 0.5)
    src = caster.stats
    attributes = Attributes()
    for attr_name in ("power", "technique", "speed", "vitality", "spirit", "presence"):
        getattr(attributes, attr_name).modify(int(getattr(src.attributes, attr_name).current * fraction))
    clone_stats = StatBlock.create(
        max_hp=max(1, int(src.hp.max_value * fraction)),
        max_mp=max(0, int(src.mp.max_value * fraction)),
        max_sp=max(0, int(src.sp.max_value * fraction)),
        attributes=attributes,
    )

    clone_book = SkillBook()
    if caster.skill_book is not None:
        for instance in caster.skill_book.skills.values():
            if any(e.type == "summon_clone" for e in instance.resolved.effects):
                continue  # no clones summoning further clones
            clone_book.add(SkillInstance(instance.def_id, source=instance.source))

    clone = Entity(
        x=spawn_pos[0], y=spawn_pos[1], char=caster.char,
        color=tuple(max(0, c - 60) for c in caster.color),
        name=f"{caster.name} (Clone)", blocks_movement=True,
        stats=clone_stats, skill_book=clone_book,
    )
    clone.ai = BehaviorTreeController()
    clone.summoned_by = caster
    engine.entities.append(clone)
    engine.message_log.add_message(
        f"{caster.name} splits off a weaker copy of herself!", color=message_log.INFO_COLOR, stack=False
    )


@register("provide_value")
def _provide_value(engine, ctx, caster, tp, effect):
    # Proficiencies are queried on demand via SkillBook.get_proficiency_value,
    # not executed; this no-op exists so running one is never an error.
    pass


@register("resource_regen")
def _resource_regen(engine, ctx, caster, tp, effect):
    # Ambient passive regen (e.g. Human Regeneration) is ticked directly by
    # Engine._tick_player_passives every turn, not run through the cast
    # pipeline; this no-op exists so running one is never an error.
    pass


@register("revive_ally")
def _revive_ally(engine, ctx, caster, tp, effect):
    # Downed characters aren't on the map to target — Engine.begin_item_use
    # intercepts revive_ally items before targeting and opens a "who to
    # revive" menu instead; this no-op exists so running one is never an error.
    pass


@register("forced_movement")
def _forced_movement(engine, ctx, caster, tp, effect):
    assert isinstance(effect, MovementEffectDef)
    kind = effect.movement_kind

    if kind == "teleport":
        # Single-stage: teleports the caster to the target tile.
        if engine.game_map.is_walkable(tp.x, tp.y):
            caster.x, caster.y = tp.x, tp.y
            if caster is engine.player:
                engine.update_camera()
                engine.update_fov()
            engine.event_bus.emit("on_move_skill", {"caster": caster})
        return

    if kind == "teleport_adjacent":
        # Single-stage: teleports the CASTER to a free tile next to `tp`'s
        # entity (an ambush/vanish-and-reappear spellcard) — reuses the
        # skill's normal enemy-targeted stage, so it can sit alongside a
        # plain damage effect against the same target in one cast (e.g.
        # Kisume's Well Dive: reposition beside the player, then hit them).
        if tp.entity is None:
            return
        from game.ai import find_adjacent_free_tile  # deferred: ai -> casting -> effects would cycle otherwise

        dest = find_adjacent_free_tile(engine, tp.entity)
        if dest is None:
            return
        caster.x, caster.y = dest
        if caster is engine.player:
            engine.update_camera()
            engine.update_fov()
        engine.event_bus.emit("on_move_skill", {"caster": caster})
        return

    if kind == "teleport_other":
        # Two-stage: `tp` is the victim, picked by activation_targeting
        # stage 0; the destination tile comes from stage 1, resolved
        # separately by Engine's chained TargetSelector flow and stashed on
        # the CastContext (see `Engine._finish_multi_stage_cast`).
        if tp.entity is None:
            return
        destination = ctx.extra_targets.get(1)
        if not destination:
            return
        dest = destination[0]
        if not engine.game_map.is_walkable(dest.x, dest.y):
            return
        if any(e is not tp.entity and e.blocks_movement and e.x == dest.x and e.y == dest.y for e in engine.entities):
            return
        tp.entity.x, tp.entity.y = dest.x, dest.y
        if tp.entity is engine.player:
            engine.update_camera()
            engine.update_fov()
        return

    if tp.entity is None:
        return
    victim = tp.entity

    if kind in ("push", "pull"):
        # Direction derived from caster<->target; no shape.
        dx = (victim.x > caster.x) - (victim.x < caster.x)
        dy = (victim.y > caster.y) - (victim.y < caster.y)
        if kind == "pull":
            dx, dy = -dx, -dy
    else:  # directional: fixed vector from params
        dx, dy = effect.params.get("dx", 0), effect.params.get("dy", 0)

    for _ in range(effect.distance):
        nx, ny = victim.x + dx, victim.y + dy
        if not engine.game_map.is_walkable(nx, ny):
            break
        if any(e.blocks_movement and e.x == nx and e.y == ny for e in engine.entities):
            break
        victim.x, victim.y = nx, ny

    if victim is engine.player:
        engine.update_camera()


def has_status(engine: "Engine", entity: Entity, status: str) -> bool:
    return any(
        t.kind == "status" and t.target is entity and t.payload.get("status") == status
        for t in engine.timed_effects
    )


# Statuses considered debuffs by is_debuff() below — extend as new
# negative status_conditions are added. Anything not listed here (e.g.
# "stealthed") is treated as a buff and never eligible for a
# debuff-shortening/cleanse effect.
DEBUFF_STATUSES = {"webbed", "marked", "blinded", "silenced", "grudge"}


def is_debuff(timed: "TimedEffectInstance") -> bool:
    """Whether a TimedEffectInstance is something worth shortening/cleansing
    (Endurance's heal-triggered capstone, any future cleanse effect) —
    every DoT is a debuff, a stat_mod only counts if it actually lowered
    something, and a status only counts if it's in DEBUFF_STATUSES."""
    if timed.kind == "dot":
        return True
    if timed.kind == "status":
        return timed.payload.get("status") in DEBUFF_STATUSES
    if timed.kind == "stat_mod":
        return timed.payload.get("revert", {}).get("applied", 0) < 0
    return False


def break_status(engine: "Engine", entity: Entity, status: str) -> bool:
    """Remove a status early (e.g. Stealth ending the instant its owner
    attacks out of it) instead of waiting for its duration to expire.
    Returns True if something was actually removed."""
    match = next(
        (t for t in engine.timed_effects if t.kind == "status" and t.target is entity and t.payload.get("status") == status),
        None,
    )
    if match is None:
        return False
    engine.timed_effects.remove(match)
    engine.message_log.add_message(f"{entity.name} is no longer {status}.", color=message_log.INFO_COLOR)
    return True
