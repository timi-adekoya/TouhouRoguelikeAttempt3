from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from components import progression as prog
from components.skills import CATEGORY_ACTIVE, SKILL_EXP_PER_COOLDOWN, SKILL_EXP_PER_USE, SkillInstance
from game import combat, message_log
from game.effects import EFFECT_EXECUTORS
from game.entity import Entity
from game.targeting import TargetPoint

if TYPE_CHECKING:
    from game.engine import Engine


class ChargeState:
    """Lives on `Entity.charging` while a charge_turns skill is being
    channeled. `turns_remaining` counts down once per the caster's own
    turn (see Engine._advance_charge); at 0 the skill resolves against the
    targets locked in at cast time (stats/formulas are still evaluated
    live at resolution, since executors always read current stats — only
    *which* targets were chosen is frozen)."""

    def __init__(
        self,
        instance: SkillInstance,
        targets: List[TargetPoint],
        extra_targets: Dict[int, List[TargetPoint]],
        turns_remaining: int,
        lock_movement: bool,
        lock_actions: bool,
    ):
        self.instance = instance
        self.targets = targets
        self.extra_targets = extra_targets
        self.turns_remaining = turns_remaining
        self.lock_movement = lock_movement
        self.lock_actions = lock_actions
        # The turn the charge starts on doesn't count toward it; otherwise a
        # 1-turn charge would resolve before anyone else got to act.
        self.started_this_turn = True


class _ResolvedWrapper:
    """Minimal `.resolved` shim so CastContext can drive a UsableDef's effect
    list the same way it drives a SkillInstance's — items aren't skills, but
    they share the same EffectDef/executor pipeline."""

    def __init__(self, resolved):
        self.resolved = resolved


class CastContext:
    """Ephemeral, one per skill activation. Holds cast-local trigger wiring,
    tracks the last event payload for contextual retargeting, and bubbles
    events up to the global bus. Effect chaining runs here."""

    def __init__(
        self,
        engine: "Engine",
        caster: Entity,
        instance: SkillInstance,
        targets: List[TargetPoint],
        extra_targets: Optional[Dict[int, List[TargetPoint]]] = None,
    ):
        self.engine = engine
        self.caster = caster
        self.instance = instance
        self.skill = instance.resolved
        self.targets = targets
        # One-shot damage bonuses consumed by this cast's first damage roll,
        # shared by all its hits/targets (see combat.resolve_skill_damage).
        self.damage_bonuses: Dict[str, float] = {}
        # Resolved targets from any activation_targeting stage beyond the
        # first (e.g. stage 1's destination tile for a two-stage
        # teleport-other skill), keyed by stage index. Root effects always
        # run against `targets` (stage 0); an executor for a later-stage
        # effect (like forced_movement's "teleport_other") reads this
        # directly instead of being handed its own tp loop.
        self.extra_targets: Dict[int, List[TargetPoint]] = extra_targets or {}
        self.last_event_target: Optional[TargetPoint] = None
        self._current_effect_id: Optional[str] = None

    def emit(self, event: str, target: TargetPoint) -> None:
        """An executor reporting a cast-local event (e.g. on_hit). Wires on
        the currently-running effect route it to other effects in the skill;
        it also bubbles to the global bus."""
        self.last_event_target = target
        current = self.skill.get_effect(self._current_effect_id) if self._current_effect_id else None
        if current is not None:
            for wire in current.on_trigger:
                if wire.event != event:
                    continue
                if wire.delay > 0:
                    # Mutable list: the engine decrements the counter in place.
                    self.engine.delayed_triggers.append(
                        [wire.delay, self, wire.effect_id, target]
                    )
                else:
                    self.run_effect(wire.effect_id, [target])
        self.engine.event_bus.emit(
            event, {"caster": self.caster, "skill": self.skill, "target": target}
        )

    def run_effect(self, effect_id: str, targets: List[TargetPoint]) -> None:
        effect = self.skill.get_effect(effect_id)
        if effect is None:
            return
        executor = EFFECT_EXECUTORS.get(effect.type)
        if executor is None:
            raise ValueError(f"No executor registered for effect type {effect.type!r}")

        previous = self._current_effect_id
        self._current_effect_id = effect_id
        try:
            for tp in targets:
                executor(self.engine, self, self.caster, tp, effect)
        finally:
            self._current_effect_id = previous


def effective_cost(caster: Entity, skill) -> dict:
    """Skill cost after proficiency discounts (e.g. Healing Proficiency
    shaving MP off anything tagged "healing"), floored at 0 per resource.
    Tempo (Mage): a flat `magic_cost_reduction` further shaves MP off
    anything tagged "magic", stacking on top of any proficiency discount."""
    if caster.skill_book is None:
        return skill.cost
    cost = {
        resource: max(0, amount - caster.skill_book.get_cost_discount(skill.tags, resource))
        for resource, amount in skill.cost.items()
    }
    if "magic" in skill.tags and caster.stats is not None and "mp" in cost:
        reduction = caster.stats.modifiers.get("magic_cost_reduction", 0.0)
        if reduction > 0:
            cost["mp"] = max(0, int(cost["mp"] - reduction))
    return cost


def can_pay(caster: Entity, cost: dict) -> bool:
    stats = caster.stats
    if stats is None:
        return not cost
    for resource, amount in cost.items():
        pool = getattr(stats, resource, None)
        if pool is None or pool.current < amount:
            return False
    return True


def pay(caster: Entity, cost: dict) -> None:
    for resource, amount in cost.items():
        getattr(caster.stats, resource).modify(-amount)


def cast_skill(
    engine: "Engine",
    caster: Entity,
    instance: SkillInstance,
    targets: List[TargetPoint],
    extra_targets: Optional[Dict[int, List[TargetPoint]]] = None,
) -> bool:
    """Full activation pipeline for an active skill. `targets` come from
    TargetResolver (via the targeting UI, or directly for NPC/AI casts).
    `extra_targets` carries any additional activation_targeting stages
    beyond the first (e.g. a teleport-other's destination stage) — see
    `Engine._finish_multi_stage_cast`. Returns False (with a log message for
    the player) if the cast is illegal.
    """
    skill = instance.resolved
    if skill.category != CATEGORY_ACTIVE:
        return False

    def fail(reason: str) -> bool:
        if caster is engine.player:
            engine.message_log.add_message(reason, color=message_log.INFO_COLOR)
        return False

    if caster.charging is not None:
        return fail("You're already concentrating on a spell.")

    from game.effects import has_status  # deferred: avoids a module cycle with effects.py

    if has_status(engine, caster, "silenced"):
        return fail("You're silenced and can't cast!")

    cost = effective_cost(caster, skill)
    if not instance.ready:
        return fail(f"{skill.name} is on cooldown ({instance.cooldown_remaining} turns).")
    if not can_pay(caster, cost):
        return fail(f"Not enough resources for {skill.name}.")
    if not targets:
        return fail("No valid target.")

    # Tempo (Mage) mid perk: a chance for a magic skill's cost to simply not
    # be paid this cast, before cooldown/effects run.
    refund_chance = 0.0
    if "magic" in skill.tags and caster.stats is not None:
        refund_chance = caster.stats.modifiers.get("magic_cost_refund_chance", 0.0) / 100
    if cost and refund_chance > 0 and random.random() < refund_chance:
        if caster is engine.player:
            engine.message_log.add_message(
                f"{skill.name} costs nothing this time!", color=message_log.INFO_COLOR
            )
    elif cost:
        pay(caster, cost)
        engine.event_bus.emit("on_resource_spent", {"caster": caster, "skill": skill})

    # Tempo (Mage) capstone: flat cooldown reduction on every magic skill.
    cooldown = skill.cooldown
    if "magic" in skill.tags and caster.stats is not None:
        reduction = caster.stats.modifiers.get("magic_cooldown_reduction", 0.0)
        cooldown = max(0, int(cooldown - reduction))
    instance.cooldown_remaining = cooldown
    instance._skip_next_tick = True

    if skill.charge_turns > 0:
        caster.charging = ChargeState(
            instance=instance,
            targets=targets,
            extra_targets=extra_targets or {},
            turns_remaining=skill.charge_turns,
            lock_movement=skill.charge_lock_movement,
            lock_actions=skill.charge_lock_actions,
        )
        if caster is engine.player:
            engine.message_log.add_message(
                f"You begin channeling {skill.name}...", color=message_log.INFO_COLOR
            )
        return True

    ctx = CastContext(engine, caster, instance, targets, extra_targets)
    # Root effects run in priority order (higher first); contextual effects
    # only run when a trigger wire invokes them with a payload target.
    # "self" effects always apply to the caster regardless of what was
    # targeted — the mechanism a single-stage skill needs to e.g. damage an
    # enemy AND buff the caster in the same cast (a self-only buff skill
    # already gets this "for free" since its targets list is just the
    # caster, but a skill that targets something else needs this to also
    # touch the caster in the same activation).
    for effect in sorted(skill.effects, key=lambda e: -e.priority):
        if effect.targeting == "contextual":
            continue
        if effect.targeting == "self":
            ctx.run_effect(effect.effect_id, [TargetPoint(caster.x, caster.y, caster)])
        else:
            ctx.run_effect(effect.effect_id, targets)

    _award_skill_use_exp(engine, caster, instance)
    return True


def resolve_charge(engine: "Engine", caster: Entity, charge: ChargeState) -> None:
    """Fire off a completed channel. Cost/cooldown were already paid when
    the charge began; a target that died/left mid-channel just fizzles the
    whole cast rather than partially resolving."""
    instance = charge.instance
    skill = instance.resolved

    live_targets = [tp for tp in charge.targets if tp.entity is None or tp.entity in engine.entities]
    if charge.targets and not live_targets:
        if caster is engine.player:
            engine.message_log.add_message(
                f"{skill.name} fizzles — the target is gone.", color=message_log.INFO_COLOR
            )
        return

    ctx = CastContext(engine, caster, instance, live_targets, charge.extra_targets)
    for effect in sorted(skill.effects, key=lambda e: -e.priority):
        if effect.targeting == "contextual":
            continue
        if effect.targeting == "self":
            ctx.run_effect(effect.effect_id, [TargetPoint(caster.x, caster.y, caster)])
        else:
            ctx.run_effect(effect.effect_id, live_targets)

    _award_skill_use_exp(engine, caster, instance)


def interrupt_charge(engine: "Engine", caster: Entity) -> None:
    """Cancel an in-progress channel (e.g. from a stun tagged
    interrupts_charge). Cost/cooldown already spent are not refunded."""
    charge = caster.charging
    if charge is None:
        return
    caster.charging = None
    if caster is engine.player:
        engine.message_log.add_message(
            f"Your casting of {charge.instance.resolved.name} is interrupted!",
            color=message_log.INFO_COLOR,
        )


def _award_skill_use_exp(engine: "Engine", caster: Entity, instance: SkillInstance) -> None:
    """Class skills feed their class (affinity-scaled) AND the species level;
    species/innate skills feed the species level only. Independent of all of
    that, the skill itself also grows from its own use — any node in its own
    self-upgrade tree is claimed manually by the player, not automatically."""
    reward = SKILL_EXP_PER_USE + instance.resolved.cooldown * SKILL_EXP_PER_COOLDOWN
    if caster.stats is not None:
        # Parsee's Loner passive: her own skills level up much faster solo,
        # barely faster in a party.
        loner_key = "skill_exp_bonus_alone" if combat.is_alone(engine) else "skill_exp_bonus_party"
        loner_bonus = caster.stats.modifiers.get(loner_key, 0.0)
        if loner_bonus:
            reward *= 1 + loner_bonus / 100
    notices: List[str] = list(instance.gain_exp(reward))

    progression = caster.progression
    if progression is not None:
        kind, _, ident = instance.source.partition(":")
        if kind == "class":
            class_exp = prog.CLASS_EXP_SKILL_USE
            if caster.stats is not None:
                class_exp *= 1 + caster.stats.modifiers.get("class_exp_percent", 0.0) / 100
            notices += progression.gain_class_exp(ident, class_exp)
            notices += progression.gain_species_exp(prog.SPECIES_EXP_SKILL_USE)
        elif kind in ("species", "innate"):
            notices += progression.gain_species_exp(prog.SPECIES_EXP_SKILL_USE)

    if notices and engine.should_log_progression(caster):
        for notice in notices:
            engine.message_log.add_message(notice, color=message_log.INFO_COLOR, stack=False)
    if progression is not None and notices:
        engine.auto_claim_linear_trees(caster)
