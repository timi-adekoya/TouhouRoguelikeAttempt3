from __future__ import annotations

from typing import TYPE_CHECKING, List

from components.skills import SkillInstance
from game import combat, message_log
from game.entity import Entity
from game.targeting import TargetPoint

if TYPE_CHECKING:
    from game.engine import Engine

class Action:
    def __init__(self, entity: Entity):
        self.entity = entity

    def perform(self, engine: "Engine") -> None:
        raise NotImplementedError


class WaitAction(Action):
    def perform(self, engine: "Engine") -> None:
        pass


def _blocked_by_webbed(engine: "Engine", entity: Entity) -> bool:
    """Shared guard for anything that repositions an entity — a webbed
    target (Yamame's snare spellcard, etc.) can't move at all, not just
    slower, so this has to actually stop the action rather than debuff it."""
    from game.effects import has_status  # deferred: avoids a module cycle with effects.py

    if not has_status(engine, entity, "webbed"):
        return False
    if entity is engine.player:
        engine.message_log.add_message(
            "You're entangled and can't move!", color=message_log.INFO_COLOR
        )
    return True


def _blocked_by_charge(engine: "Engine", entity: Entity, kind: str) -> bool:
    """Shared guard for anything a charge-cast (see game/casting.py) locks
    out while channeling — `kind` is "movement" or "actions" (attacks/skills/
    items), matched against the ChargeState's own lock flags so a charge
    skill can opt out of either restriction."""
    charge = entity.charging
    if charge is None:
        return False
    locked = charge.lock_movement if kind == "movement" else charge.lock_actions
    if not locked:
        return False
    if entity is engine.player:
        verb = "move" if kind == "movement" else "act"
        engine.message_log.add_message(
            f"You're concentrating and can't {verb}!", color=message_log.INFO_COLOR
        )
    return True


class MovementAction(Action):
    def __init__(self, entity: Entity, dx: int, dy: int):
        super().__init__(entity)
        self.dx = dx
        self.dy = dy

    def perform(self, engine: "Engine") -> None:
        if _blocked_by_webbed(engine, self.entity):
            return
        if _blocked_by_charge(engine, self.entity, "movement"):
            return
        dest_x, dest_y = self.entity.x + self.dx, self.entity.y + self.dy
        if not engine.game_map.is_walkable(dest_x, dest_y):
            return

        self.entity.move(self.dx, self.dy)

        if self.entity is engine.player:
            engine.open_door_if_present(dest_x, dest_y)
            engine.update_camera()
            engine.update_fov()


class SwapPlacesAction(Action):
    """Trade tiles with a blocking ally instead of routing all the way
    around them — used both by the player bumping into a party member and
    by party AI pathing, so a crowded corridor/room doesn't force a long
    detour (which is what caused 3+ member parties to fall behind)."""

    def __init__(self, entity: Entity, other: Entity):
        super().__init__(entity)
        self.other = other

    def perform(self, engine: "Engine") -> None:
        if _blocked_by_webbed(engine, self.entity):
            return
        if _blocked_by_charge(engine, self.entity, "movement"):
            return
        self.entity.x, self.other.x = self.other.x, self.entity.x
        self.entity.y, self.other.y = self.other.y, self.entity.y
        if self.entity is engine.player or self.other is engine.player:
            engine.update_camera()
            engine.update_fov()


class MeleeAttackAction(Action):
    def __init__(self, entity: Entity, target: Entity):
        super().__init__(entity)
        self.target = target

    def perform(self, engine: "Engine") -> None:
        if _blocked_by_charge(engine, self.entity, "actions"):
            return
        result = combat.resolve_weapon_attack(engine, self.entity, self.target)
        if result is not None:
            damage, cause = result
            combat.apply_damage(engine, self.entity, self.target, damage, cause=cause, tags=["melee"])


class CastSkillAction(Action):
    """Cast a resolved skill at pre-resolved targets. Used by the player's
    skill menu and by AI controllers alike — any actor can drive a skill
    through this one entry point into the casting pipeline."""

    def __init__(self, entity: Entity, instance: SkillInstance, targets: List[TargetPoint]):
        super().__init__(entity)
        self.instance = instance
        self.targets = targets

    def perform(self, engine: "Engine") -> None:
        from game import casting

        casting.cast_skill(engine, self.entity, self.instance, self.targets)


class UseItemAction(Action):
    """Activate a UsableDef item (potion/wand/scroll) at pre-resolved
    targets. Mirrors Engine.use_item's effect-dispatch and charge/quantity
    bookkeeping, but generalized to any entity — Engine.use_item is
    hardcoded to self.player, so AI-controlled party members need this
    separate entry point to use items on their own."""

    def __init__(self, entity: Entity, instance, targets: List[TargetPoint]):
        super().__init__(entity)
        self.instance = instance
        self.targets = targets

    def perform(self, engine: "Engine") -> None:
        if _blocked_by_charge(engine, self.entity, "actions"):
            return
        from game import casting

        definition = self.instance.definition
        ctx = casting.CastContext(engine, self.entity, casting._ResolvedWrapper(definition), self.targets)
        for effect in sorted(definition.effects, key=lambda e: -e.priority):
            if effect.targeting == "contextual":
                continue
            if effect.targeting == "self":
                ctx.run_effect(effect.effect_id, [TargetPoint(self.entity.x, self.entity.y, self.entity)])
            else:
                ctx.run_effect(effect.effect_id, self.targets)

        if definition.consumable:
            self.entity.inventory.consume_one(self.instance)
        elif self.instance.charges_remaining is not None:
            self.instance.charges_remaining -= 1
