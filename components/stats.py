from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DEFAULT_ATTRIBUTE_MAX = 99
CORE_ATTRIBUTES = ("power", "technique", "speed", "vitality", "spirit", "presence")
RESOURCE_POOLS = ("hp", "mp", "sp", "faith")

# Same shape as Speed's pull on turn-queue energy rate (game/turn_queue.py's
# SPEED_ENERGY_FACTOR, now a much stronger effect than these two): Vitality/
# Spirit get an automatic passive tug on resource pools, on top of whatever
# a class/item/skill explicitly grants. Applies to any Vitality/Spirit
# change, enemy or player.
VITALITY_HP_FACTOR = 4  # +4 max HP per point of Vitality (raised from 3 2026-07-21 balance pass — sustain/HP investment was underwhelming)
SPIRIT_RESOURCE_FACTOR = 1  # +1 max MP and +1 max SP per point of Spirit (slight)


class Stat:
    """A clamped numeric value with a max (and optional min), e.g. HP or an attribute.

    Attributes themselves don't directly drive derived numbers (crit chance,
    accuracy, etc.) — that's left to class features/equipment/skills. This is
    just the clamped-value container both resources (HP/MP/SP/Faith) and
    attributes (Power/Technique/...) share.
    """

    def __init__(self, max_value: int, current: Optional[int] = None, min_value: int = 0):
        self.min_value = min_value
        self.max_value = max_value
        self.current = max_value if current is None else current

    def modify(self, amount: int) -> int:
        """Adjust current by amount, clamped to [min_value, max_value].

        Returns the actual delta applied (may differ from `amount` if clamped).
        """
        old = self.current
        self.current = max(self.min_value, min(self.max_value, self.current + amount))
        return self.current - old

    def set_max(self, new_max: int, clamp_current: bool = True) -> None:
        self.max_value = new_max
        if clamp_current:
            self.current = max(self.min_value, min(self.current, self.max_value))

    @property
    def is_empty(self) -> bool:
        return self.current <= self.min_value

    @property
    def is_full(self) -> bool:
        return self.current >= self.max_value

    def __repr__(self) -> str:
        return f"Stat({self.current}/{self.max_value})"


@dataclass
class Attributes:
    """The six core attributes. These are growth stats, not derived combat
    numbers — a class feature/equipment/skill decides how they translate
    into things like crit chance or accuracy."""

    power: Stat = field(default_factory=lambda: Stat(DEFAULT_ATTRIBUTE_MAX, 0))
    technique: Stat = field(default_factory=lambda: Stat(DEFAULT_ATTRIBUTE_MAX, 0))
    speed: Stat = field(default_factory=lambda: Stat(DEFAULT_ATTRIBUTE_MAX, 0))
    vitality: Stat = field(default_factory=lambda: Stat(DEFAULT_ATTRIBUTE_MAX, 0))
    spirit: Stat = field(default_factory=lambda: Stat(DEFAULT_ATTRIBUTE_MAX, 0))
    presence: Stat = field(default_factory=lambda: Stat(DEFAULT_ATTRIBUTE_MAX, 0))


@dataclass
class StatBlock:
    """HP/MP/SP resource pools, attributes, and an optional Faith pool.

    Attach to any Entity that needs stats (player characters, generic NPCs,
    enemies) — independent of whether that entity is a recruitable
    `Character`.
    """

    hp: Stat
    mp: Stat
    sp: Stat
    attributes: Attributes = field(default_factory=Attributes)
    faith: Optional[Stat] = None  # Only present for characters with Faith.
    # Free-form combat modifiers (crit_rate, crit_damage, defense,
    # magic_defense, ...) that don't belong to any Stat — combat formulas
    # read these directly at attack time. Percentage-based ones are stored
    # in whole percentage points, same convention as attribute amounts.
    modifiers: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        max_hp: int,
        max_mp: int,
        max_sp: int,
        attributes: Optional[Attributes] = None,
        max_faith: Optional[int] = None,
    ) -> "StatBlock":
        return cls(
            hp=Stat(max_hp),
            mp=Stat(max_mp),
            sp=Stat(max_sp),
            attributes=attributes if attributes is not None else Attributes(),
            faith=Stat(max_faith) if max_faith is not None else None,
        )


def _apply_resource_delta(pool: Stat, delta: float) -> Dict[str, Any]:
    """Raise a resource pool's max by `delta` and its current by the same
    amount, so a bonus is felt immediately instead of only on refill.

    Callers that resize an existing bonus on level-up (species growth,
    passive re-level — see Engine._reapply_stat_modifiers) must call this
    with the NET change from the old amount to the new one, not revert the
    old amount and then call this again with the full new amount: reverting
    a big old bonus while current HP is low clamps at the pool's floor and
    silently loses track of how much should've come back out, so a
    following full reapply lands on top of that artificially-low current —
    an unintended near-full heal on any level-up taken while below the old
    bonus amount. A single net-delta call never passes through that
    broken intermediate state."""
    delta = int(delta)
    # Bypass Stat.set_max's own clamp-current-to-new-max side effect —
    # applying it and then also calling modify(delta) below would apply a
    # negative delta twice (once via the clamp, once via modify), corrupting
    # `current` any time the new max is smaller than the old current.
    pool.max_value += delta
    current_applied = pool.modify(delta)
    return {"max_delta": delta, "current_applied": current_applied}


def _revert_resource_delta(pool: Stat, payload: Dict[str, Any]) -> None:
    pool.modify(-payload["current_applied"])
    pool.set_max(pool.max_value - payload["max_delta"])


def _linked_resource_deltas(stats: StatBlock, name: str, applied: int) -> list:
    """Vitality/Spirit each give a small automatic pull on a resource pool,
    same idea as Speed's mild pull on turn-queue energy rate — on top of
    whatever a class/item/skill explicitly grants."""
    if not applied:
        return []
    if name == "vitality":
        return [{"resource": "hp", **_apply_resource_delta(stats.hp, applied * VITALITY_HP_FACTOR)}]
    if name == "spirit":
        return [
            {"resource": "mp", **_apply_resource_delta(stats.mp, applied * SPIRIT_RESOURCE_FACTOR)},
            {"resource": "sp", **_apply_resource_delta(stats.sp, applied * SPIRIT_RESOURCE_FACTOR)},
        ]
    return []


def apply_stat_modifier(stats: StatBlock, name: str, amount: float) -> Dict[str, Any]:
    """Apply a named modifier to a StatBlock and return a payload describing
    exactly what changed, so `revert_stat_modifier` can undo it later.

    `name` may be:
      - a core attribute (power/technique/speed/vitality/spirit/presence):
        adjusts that Stat's current value. Vitality/Spirit changes also
        nudge HP/MP+SP respectively (see `_linked_resource_deltas`).
      - a resource pool (hp/mp/sp/faith): raises the pool's max by `amount`
        and its current by the same amount, so the bonus is felt immediately
        rather than only showing up once the pool is refilled.
      - anything else (crit_rate, crit_damage, defense, magic_defense, ...):
        a free-form combat modifier accumulated in `stats.modifiers`, read
        directly by combat formulas at attack time.
    """
    if name in CORE_ATTRIBUTES:
        stat = getattr(stats.attributes, name)
        applied = stat.modify(int(amount))
        payload: Dict[str, Any] = {"kind": "attribute", "name": name, "applied": applied}
        linked = _linked_resource_deltas(stats, name, applied)
        if linked:
            payload["linked"] = linked
        return payload

    if name in RESOURCE_POOLS:
        pool = getattr(stats, name, None)
        if pool is None:  # e.g. "faith" on a character with no Faith pool
            return {"kind": "noop"}
        return {"kind": "resource", "name": name, **_apply_resource_delta(pool, amount)}

    stats.modifiers[name] = stats.modifiers.get(name, 0.0) + amount
    return {"kind": "combat", "name": name, "applied": amount}


def revert_stat_modifier(stats: StatBlock, payload: Dict[str, Any]) -> None:
    kind = payload.get("kind")
    if kind == "attribute":
        getattr(stats.attributes, payload["name"]).modify(-payload["applied"])
        for linked in payload.get("linked", []):
            _revert_resource_delta(getattr(stats, linked["resource"]), linked)
    elif kind == "resource":
        _revert_resource_delta(getattr(stats, payload["name"]), payload)
    elif kind == "combat":
        stats.modifiers[payload["name"]] = stats.modifiers.get(payload["name"], 0.0) - payload["applied"]
