from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from game.skill_tree import SkillTreeDef, parse_tree

# Skill categories. Category gates whether cooldown/resource logic runs:
# only actives are cast; passives sit in the book and apply/tick; proficiencies
# are queried by ID for a value.
CATEGORY_ACTIVE = "active"
CATEGORY_PASSIVE = "passive"
CATEGORY_PROFICIENCY = "proficiency"

# A skill's own growth, independent of whatever class/species/innate tree
# granted it — every skill can level up on its own and (optionally) has a
# small tree of self-upgrades gated by its own level, not the granting
# tree's. Actives earn exp per successful cast; passives earn a small
# per-turn trickle just for being active (mirrors class-passive exp) UNLESS
# they have an `exp_trigger` set (see TRIGGER_EXP_REWARDS below).
SKILL_EXP_PER_USE = 4.0
# A high-cooldown active is cast far less often in real playtime than a
# low-cooldown one, same underlying problem as passive trigger frequency
# above -- without this, a legendary+ active (likely high-CD by design)
# would need an impractical number of real casts to max, while a 1-2 CD
# common skill hits the same total exp fine already. Gentler than the
# passive-trigger fix (a flat per-category reward) since this is
# continuous off the skill's own actual cooldown, not a lookup, and
# SKILL_EXP_PER_USE stays as the floor so a low-CD common skill's feel
# barely changes. Starting guess, not measured.
SKILL_EXP_PER_COOLDOWN = 1.0
SKILL_EXP_PASSIVE_TICK = 0.05
# Lowered from 0.3 on 2026-07-21 playtest feedback: at 0.3, a common-rarity
# passive (140 total exp to max) hit its cap in ~467 turns — well before
# depth 5. 0.05 stretches that to ~2800 turns; retune this one constant if
# it still feels off in either direction.

# Per-trigger exp reward for a passive with `exp_trigger` set (see
# Engine._award_trigger_exp) — NOT a flat rate. Different triggers fire at
# very different natural frequencies (on_damage_dealt nearly every combat
# turn; on_evaded only with real evasion investment), so a single flat
# reward would make some triggers max out almost immediately and others
# practically never. These are starting estimates pending real
# playtesting, not derived from any measurement.
TRIGGER_EXP_REWARDS: Dict[str, float] = {
    "on_damage_dealt": 1.5,
    "on_hp_change": 1.5,
    "on_resource_consumed": 2.5,
    "on_move_skill": 4.0,
    "on_heal_cast": 4.0,
    "on_evaded": 6.0,
}
DEFAULT_TRIGGER_EXP_REWARD = 1.5  # fallback for any future trigger name not yet tuned above

DEFAULT_SKILL_MAX_LEVEL = 5

# Rarity determines a skill's own depth: how many levels it can reach, and
# how steep the exp curve is getting there. Both axes grow geometrically
# per tier (bigger level-count jumps AND a steeper per-level cost further
# up), so total investment compounds rather than just extending linearly —
# by Transcendent it's roughly 46x a Common skill's total exp, not 5x.
# Only common/uncommon/rare have real content as of 2026-07-19 (every base
# class caps at rare, per design); epic and above are reserved for future
# class-advancement/boss/loot content. Numbers are a starting point to
# playtest against, not mathematically derived — expect these to move.
RARITY_MAX_LEVEL: Dict[str, int] = {
    "common": 5,
    "uncommon": 7,
    "rare": 9,
    "epic": 12,
    "legendary": 15,
    "mythic": 20,
    "transcendent": 25,
}
RARITY_EXP_CURVE: Dict[str, Tuple[float, float]] = {  # (base, coefficient) of `base + coef * level`
    "common": (15.0, 8.0),
    "uncommon": (19.0, 10.0),
    "rare": (24.0, 13.0),
    "epic": (30.0, 16.0),
    "legendary": (38.0, 20.0),
    "mythic": (48.0, 26.0),
    "transcendent": (60.0, 33.0),
}


def skill_exp_to_next(level: int, rarity: str = "common") -> float:
    base, coef = RARITY_EXP_CURVE.get(rarity, RARITY_EXP_CURVE["common"])
    return base + coef * level


@dataclass
class TriggerWire:
    """One on_trigger entry: when `event` fires during a cast, invoke the
    effect with `effect_id` after `delay` turns (0 = immediately)."""

    event: str
    effect_id: str
    delay: int = 0


@dataclass
class ActivationTargetingDef:
    """Player-facing selection spec for one targeting stage of a skill."""

    selection_mode: str = "single"  # single | multiple | shape
    range_shape: str = "self"  # self | point | circle | line
    shape_params: Dict[str, Any] = field(default_factory=dict)  # range, radius, length...
    restriction: str = "none"  # self | enemy | ally | all | none
    target_kind: str = "entity"  # entity | location | direction
    max_picks: int = 1


@dataclass
class EffectDef:
    """One node in a skill's flat effect list."""

    effect_id: str
    type: str  # key into the effect-executor registry
    params: Dict[str, Any] = field(default_factory=dict)
    targeting: str = "absolute"  # absolute (use cast targets) | contextual (from trigger payload)
    on_trigger: List[TriggerWire] = field(default_factory=list)
    priority: int = 0


@dataclass
class MovementEffectDef(EffectDef):
    """Effect that moves something. `movement_kind` is one of
    push | pull | directional | teleport."""

    movement_kind: str = "push"
    distance: int = 1


@dataclass
class SkillDef:
    """Static, immutable definition of a skill (loaded from JSON)."""

    skill_id: str
    name: str
    description: str = ""
    rarity: str = "common"
    category: str = CATEGORY_ACTIVE
    tags: List[str] = field(default_factory=list)  # e.g. "healing" — read by proficiency discounts
    cost: Dict[str, int] = field(default_factory=dict)  # e.g. {"mp": 5, "sp": 2}
    cooldown: int = 0
    max_level: int = DEFAULT_SKILL_MAX_LEVEL
    tree: Optional[SkillTreeDef] = None  # a skill's own (usually shallow) self-upgrade tree
    # Skill points earned per level of THIS skill (not the granting class/
    # species), spent on `tree`'s nodes via their own `cost`. Per-skill
    # rather than a global rate — a common 1-node skill and a legendary
    # 5-node skill should be free to tune independently. No scarcity is
    # intended here for now (unlike class trees): default 1/level comfortably
    # affords the typical 1-3 node tree.
    points_per_level: int = 1
    activation_targeting: List[ActivationTargetingDef] = field(default_factory=list)
    effects: List[EffectDef] = field(default_factory=list)
    # Charge-cast: 0 = resolves immediately (default, all pre-existing
    # skills). >0 = cost/cooldown are paid on activation but effects don't
    # run until this many of the caster's own turns have passed; see
    # game/casting.py's ChargeState/resolve_charge.
    charge_turns: int = 0
    charge_lock_movement: bool = False
    charge_lock_actions: bool = True
    # Passive-only: free, indefinite, player-flips-it-on/off (vs. a stance
    # that costs a resource and runs on a cooldown/duration). Off by default
    # so an untoggleable passive just always applies, as before.
    toggleable: bool = False
    # Passive-only: which event this skill's own exp comes from instead of
    # the default per-turn trickle. None (default, every pre-existing
    # passive) keeps the trickle unchanged. One of "on_damage_dealt",
    # "on_hp_change", "on_heal_cast", "on_evaded", "on_move_skill",
    # "on_resource_consumed" — see Engine._award_trigger_exp.
    exp_trigger: Optional[str] = None

    def get_effect(self, effect_id: str) -> Optional[EffectDef]:
        return next((e for e in self.effects if e.effect_id == effect_id), None)


@dataclass
class UpgradeDef:
    """A patch applied on top of a SkillDef's raw dict, gated by requirements.

    Ops are {"op": "set"|"add"|"append", "path": "...", "value": x} where
    path uses dots and [n] indexing, e.g. "effects[0].params.amount".
    """

    upgrade_id: str
    skill_id: str
    description: str = ""
    requires_level: int = 1  # level of the granting tree's owner (class/species)
    requires_rarity: Optional[str] = None
    ops: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing (raw JSON dict -> defs)
# ---------------------------------------------------------------------------


def parse_targeting(raw: Dict[str, Any]) -> ActivationTargetingDef:
    return ActivationTargetingDef(
        selection_mode=raw.get("selection_mode", "single"),
        range_shape=raw.get("range_shape", "self"),
        shape_params=raw.get("shape_params", {}),
        restriction=raw.get("restriction", "none"),
        target_kind=raw.get("target_kind", "entity"),
        max_picks=raw.get("max_picks", 1),
    )


def parse_effect(raw: Dict[str, Any]) -> EffectDef:
    wires = [
        TriggerWire(event=w["event"], effect_id=w["effect_id"], delay=w.get("delay", 0))
        for w in raw.get("on_trigger", [])
    ]
    common = dict(
        effect_id=raw["effect_id"],
        type=raw["type"],
        params=raw.get("params", {}),
        targeting=raw.get("targeting", "absolute"),
        on_trigger=wires,
        priority=raw.get("priority", 0),
    )
    if raw["type"] == "forced_movement":
        return MovementEffectDef(
            **common,
            movement_kind=raw.get("movement_kind", "push"),
            distance=raw.get("distance", 1),
        )
    return EffectDef(**common)


def parse_skill(raw: Dict[str, Any]) -> SkillDef:
    rarity = raw.get("rarity", "common")
    return SkillDef(
        skill_id=raw["skill_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        rarity=rarity,
        category=raw.get("category", CATEGORY_ACTIVE),
        tags=raw.get("tags", []),
        cost=raw.get("cost", {}),
        cooldown=raw.get("cooldown", 0),
        # Defaults from rarity (RARITY_MAX_LEVEL) unless a skill explicitly
        # overrides it — none currently do, but the escape hatch stays for
        # any future skill that genuinely needs a bespoke level cap.
        max_level=raw.get("max_level", RARITY_MAX_LEVEL.get(rarity, DEFAULT_SKILL_MAX_LEVEL)),
        tree=parse_tree(raw["skill_id"], raw["tree"]) if raw.get("tree") else None,
        activation_targeting=[parse_targeting(t) for t in raw.get("activation_targeting", [])],
        effects=[parse_effect(e) for e in raw.get("effects", [])],
        charge_turns=raw.get("charge_turns", 0),
        charge_lock_movement=raw.get("charge_lock_movement", False),
        charge_lock_actions=raw.get("charge_lock_actions", True),
        toggleable=raw.get("toggleable", False),
        points_per_level=raw.get("points_per_level", 1),
        exp_trigger=raw.get("exp_trigger"),
    )


def parse_upgrade(raw: Dict[str, Any]) -> UpgradeDef:
    return UpgradeDef(
        upgrade_id=raw["upgrade_id"],
        skill_id=raw["skill_id"],
        description=raw.get("description", ""),
        requires_level=raw.get("requires_level", 1),
        requires_rarity=raw.get("requires_rarity"),
        ops=raw.get("ops", []),
    )


# ---------------------------------------------------------------------------
# Upgrade patching (operates on raw dicts, before parsing)
# ---------------------------------------------------------------------------

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _walk_path(root: Any, path: str):
    """Yield (container, key) for the final segment of a dotted/[n] path."""
    tokens: List[Any] = []
    for name, index in _PATH_TOKEN.findall(path):
        tokens.append(int(index) if index else name)
    node = root
    for token in tokens[:-1]:
        node = node[token]
    return node, tokens[-1]


def apply_upgrade_ops(raw_skill: Dict[str, Any], ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    patched = copy.deepcopy(raw_skill)
    for op in ops:
        container, key = _walk_path(patched, op["path"])
        kind = op["op"]
        if kind == "set":
            container[key] = op["value"]
        elif kind == "add":
            container[key] = container.get(key, 0) + op["value"] if isinstance(container, dict) else container[key] + op["value"]
        elif kind == "append":
            container[key].append(op["value"])
        else:
            raise ValueError(f"Unknown upgrade op: {kind!r}")
    return patched


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class SkillInstance:
    """Runtime wrapper around a SkillDef: level, cooldown, chosen upgrades,
    and a cached resolved def (recomputed only when level/upgrades change).

    `source` records what granted the skill ("class:warrior",
    "species:human", "innate:reimu") — used for exp routing and for
    removing class skills when class levels reset at run end.
    """

    def __init__(self, def_id: str, source: str = "unknown", level: int = 1):
        self.def_id = def_id
        self.source = source
        self.level = level
        self.exp = 0.0
        self.claimed_nodes: Set[str] = set()
        # Only meaningful for toggleable passives; always True otherwise, so
        # an untoggleable passive's effects always apply as before.
        self.enabled = True
        self.cooldown_remaining = 0
        # Set the same turn a fresh cooldown starts, so that turn's own
        # end-of-turn tick_cooldowns() call doesn't immediately shave one off
        # a cooldown that was just set — otherwise "cooldown: 2" would only
        # ever hold the skill up for a single turn instead of two.
        self._skip_next_tick = False
        self.upgrades: List[str] = []
        # Revert payloads for this skill's own passive stat_modifier effects,
        # so leveling up (which changes the resolved amounts) can cleanly
        # swap the old bonus for the new one instead of stacking both.
        self.applied_deltas: List[Dict[str, Any]] = []
        # Per-effect turn counters for periodic passive effects (e.g. a
        # resource_regen that only fires every N turns instead of every one).
        self.tick_counters: Dict[str, int] = {}
        self._resolved: Optional[SkillDef] = None

    @property
    def resolved(self) -> SkillDef:
        if self._resolved is None:
            from game import skill_data  # deferred: registry lives game-side

            raw = copy.deepcopy(skill_data.raw_skill(self.def_id))
            for upgrade_id in self.upgrades:
                raw = apply_upgrade_ops(raw, skill_data.upgrade(upgrade_id).ops)
            self._resolved = parse_skill(raw)
        return self._resolved

    @property
    def category(self) -> str:
        return self.resolved.category

    def add_upgrade(self, upgrade_id: str) -> None:
        if upgrade_id not in self.upgrades:
            self.upgrades.append(upgrade_id)
            self._resolved = None

    def set_level(self, level: int) -> None:
        self.level = level
        self._resolved = None

    @property
    def ready(self) -> bool:
        return self.cooldown_remaining <= 0

    @property
    def at_max_level(self) -> bool:
        return self.level >= self.resolved.max_level

    def gain_exp(self, amount: float) -> List[str]:
        """This skill's own growth, independent of whatever class/species
        granted it. Returns level-up notices."""
        notices: List[str] = []
        if self.at_max_level:
            return notices
        rarity = self.resolved.rarity
        self.exp += amount
        while not self.at_max_level and self.exp >= skill_exp_to_next(self.level, rarity):
            self.exp -= skill_exp_to_next(self.level, rarity)
            self.level += 1
            self._resolved = None
            notices.append(f"{self.resolved.name} reached level {self.level}!")
        if self.at_max_level:
            self.exp = 0.0
        return notices

    def available_tree_nodes(self) -> List["TreeNodeDef"]:
        """Nodes in this skill's own tree claimable right now, given its
        current level. Manual — the player picks when/whether to claim
        them, same as class trees; nothing here auto-unlocks."""
        tree = self.resolved.tree
        if tree is None:
            return []
        return tree.available_nodes(self.level, self.claimed_nodes)

    @property
    def points_available(self) -> int:
        """This skill's own point pool: `level * points_per_level`, minus
        whatever's already spent on its own tree's claimed nodes."""
        tree = self.resolved.tree
        spent = sum(tree.nodes[n].cost for n in self.claimed_nodes if tree is not None and n in tree.nodes)
        return self.level * self.resolved.points_per_level - spent

    def claim_tree_node(self, node_id: str) -> List[str]:
        """Claim one node from this skill's own tree. Returns notices;
        empty means the claim failed (not available / already claimed /
        can't afford its cost)."""
        tree = self.resolved.tree
        if tree is None:
            return []
        node = tree.nodes.get(node_id)
        if node is None or node not in self.available_tree_nodes():
            return []
        if node.cost > self.points_available:
            return []

        from game import skill_data  # deferred: registry lives game-side

        self.claimed_nodes.add(node_id)
        notices: List[str] = [f"{self.resolved.name} learns {node.name}."]
        for upgrade_id in node.grants_upgrades:
            upgrade = skill_data.upgrade(upgrade_id)
            if self.level >= upgrade.requires_level:
                self.add_upgrade(upgrade_id)
        return notices

    def serialize(self) -> Dict[str, Any]:
        # Persist state only, never the resolved def; re-resolve at load.
        # applied_deltas/tick_counters are included too — without them a
        # reloaded passive would forget what it already applied and
        # double-stack its bonus (or lose periodic-regen timing) next tick.
        return {
            "def_id": self.def_id,
            "source": self.source,
            "level": self.level,
            "exp": self.exp,
            "claimed_nodes": list(self.claimed_nodes),
            "upgrades": list(self.upgrades),
            "enabled": self.enabled,
            "cooldown_remaining": self.cooldown_remaining,
            "applied_deltas": [dict(p) for p in self.applied_deltas],
            "tick_counters": dict(self.tick_counters),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "SkillInstance":
        instance = cls(data["def_id"], source=data.get("source", "unknown"), level=data.get("level", 1))
        instance.exp = data.get("exp", 0.0)
        instance.claimed_nodes = set(data.get("claimed_nodes", []))
        instance.upgrades = list(data.get("upgrades", []))
        instance.enabled = data.get("enabled", True)
        instance.cooldown_remaining = data.get("cooldown_remaining", 0)
        instance.applied_deltas = [dict(p) for p in data.get("applied_deltas", [])]
        instance.tick_counters = dict(data.get("tick_counters", {}))
        return instance


class SkillBook:
    """All of an actor's SkillInstances, uniformly (active/passive/proficiency)."""

    def __init__(self) -> None:
        self.skills: Dict[str, SkillInstance] = {}

    def add(self, instance: SkillInstance) -> None:
        self.skills[instance.def_id] = instance

    def remove(self, def_id: str) -> None:
        self.skills.pop(def_id, None)

    def get(self, def_id: str) -> Optional[SkillInstance]:
        return self.skills.get(def_id)

    def by_category(self, category: str) -> List[SkillInstance]:
        return [s for s in self.skills.values() if s.category == category]

    def by_source(self, source: str) -> List[SkillInstance]:
        return [s for s in self.skills.values() if s.source == source]

    def tick_cooldowns(self) -> None:
        for skill in self.skills.values():
            if skill._skip_next_tick:
                skill._skip_next_tick = False
                continue
            if skill.cooldown_remaining > 0:
                skill.cooldown_remaining -= 1

    def get_cost_discount(self, tags: List[str], resource: str) -> int:
        """Sum of proficiency discounts that apply to `resource` for a skill
        carrying any of `tags` (e.g. Healing Proficiency shaving MP off
        anything tagged "healing")."""
        total = 0
        for instance in self.by_category(CATEGORY_PROFICIENCY):
            for effect in instance.resolved.effects:
                if effect.type != "provide_value":
                    continue
                if effect.params.get("discount_resource") != resource:
                    continue
                if effect.params.get("discount_tag") not in tags:
                    continue
                total += effect.params.get("value", 0)
        return total

    def get_proficiency_value(self, def_id: str) -> int:
        """Sum of provide_value effects on the named proficiency, or 0."""
        instance = self.skills.get(def_id)
        if instance is None or instance.category != CATEGORY_PROFICIENCY:
            return 0
        return sum(
            e.params.get("value", 0)
            for e in instance.resolved.effects
            if e.type == "provide_value"
        )
