from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from components.skills import SkillBook, SkillInstance
from game import skill_data
from game.skill_tree import SkillTreeDef, TreeNodeDef

# --- Exp tuning -------------------------------------------------------------
# Species exp comes mainly from combat contribution; heals/buffs give small
# fixed amounts (intentionally inefficient long-term). Class exp comes from
# using that class's skills (active casts) AND from that class's passives
# actually proccing their own exp_trigger — no more automatic per-turn
# trickle (removed 2026-07-19; it let a class level itself up just from
# turns passing, same free-grind problem the skill-level trickle had before
# that got the trigger-based fix). Affinity scales class exp gain only — a
# gentle nudge, not a wall.
SPECIES_EXP_PER_DAMAGE = 1.0
# Kill exp scales off the target's own max HP (a natural stand-in for "how
# tough was this" without a hand-tuned per-species table) instead of one
# flat amount for every kill, plus an extra multiplier specifically for
# bosses — defeating a boss is a bigger deal than its raw HP alone implies.
# See combat._award_kill_exp, which also shares this among the whole
# current party, not just whoever landed the killing blow.
SPECIES_EXP_KILL_PER_MAX_HP = 1.5
SPECIES_EXP_BOSS_KILL_MULTIPLIER = 2.0
SPECIES_EXP_HEAL = 3.0
SPECIES_EXP_BUFF = 3.0
SPECIES_EXP_SKILL_USE = 3.0
CLASS_EXP_SKILL_USE = 5.0
# Awarded to a class whenever one of ITS passives procs its own
# exp_trigger (see Engine._award_trigger_exp) — the passive-side
# replacement for the removed per-turn trickle, gated on actually doing
# something instead of existing.
CLASS_EXP_PASSIVE_TRIGGER = 2.0

# Automatic baseline HP/MP/SP growth purely from species level, independent
# of Vitality/Spirit investment or any picked-up passive — stacks on top of
# those, doesn't replace them. Without this, a character whose build didn't
# happen to grant any HP-flavored passive just never grows past their
# starting pools at all (confirmed 2026-07-19: StatBlock has zero automatic
# growth otherwise). Level 1 nets zero, same "no bonus at the starting
# level" convention as skill points — see Engine._reapply_species_stat_growth.
SPECIES_HP_GROWTH_PER_LEVEL = 3  # raised from 2 2026-07-21 balance pass, alongside VITALITY_HP_FACTOR 3->4
SPECIES_MP_GROWTH_PER_LEVEL = 1
SPECIES_SP_GROWTH_PER_LEVEL = 1

SPECIES_MAX_LEVEL = 50
DEFAULT_AFFINITY = 1.0

# Species-level milestones that each open one more class slot.
CLASS_SLOT_MILESTONES = [1, 10, 20, 35]


def species_exp_to_next(level: int) -> float:
    # Was purely linear (40 + 20*level, 460 total to level 10) — playtesting
    # showed species level badly outpacing dungeon pacing (species level 20+
    # reached by floor 11, vs. class level only 10 over the same stretch).
    # The quadratic term makes each successive level cost noticeably more
    # than the last instead of a flat, ever-easier-relative-to-total climb —
    # early levels barely move, but by level 20 total exp needed is roughly
    # 3.7x the old curve's. First-pass coefficient, not measured; expect
    # this to move after a fresh playtest, same as every other exp number
    # touched this session (2026-07-22).
    return 40.0 + 20.0 * level + 5.0 * level * level


def class_exp_to_next(level: int) -> float:
    # Lowered 2026-07-19 (was 30 + 15*level, 945 total to max level 10) —
    # with the free passive trickle removed, the old total would've
    # required far more deliberate active-use/passive-trigger engagement
    # than intended. Starting guess, not measured; expect this to move
    # after playtesting, same as every other exp number touched this pass.
    return 20.0 + 9.0 * level


class ClassProgress:
    """One active class on a character: level, exp, claimed tree nodes.
    Class levels reset at the end of a dungeon run; unlocks don't."""

    def __init__(self, class_id: str):
        self.class_id = class_id
        self.level = 1
        self.exp = 0.0
        self.claimed_nodes: Set[str] = set()

    @property
    def definition(self) -> skill_data.ClassDef:
        return skill_data.class_def(self.class_id)

    @property
    def at_max_level(self) -> bool:
        return self.level >= self.definition.max_level

    def gain_exp(self, amount: float, affinity: float) -> List[str]:
        """Apply exp (scaled by affinity); returns level-up notices."""
        notices: List[str] = []
        if self.at_max_level:
            return notices
        self.exp += amount * affinity
        while not self.at_max_level and self.exp >= class_exp_to_next(self.level):
            self.exp -= class_exp_to_next(self.level)
            self.level += 1
            notices.append(f"{self.definition.name} class reached level {self.level}!")
        if self.at_max_level:
            self.exp = 0.0
        return notices


class Progression:
    """A character's growth state: immutable species, species level (the
    "main" level), innate + species tree progress, and active classes.
    """

    def __init__(
        self,
        species_id: str,
        innate_id: Optional[str] = None,
        affinities: Optional[Dict[str, float]] = None,
    ):
        self.species_id = species_id
        self.innate_id = innate_id
        self.species_level = 1
        self.species_exp = 0.0
        self.species_claimed: Set[str] = set()
        self.innate_claimed: Set[str] = set()
        self.classes: List[ClassProgress] = []
        # class_id -> exp multiplier; missing means neutral (1.0).
        self.affinities: Dict[str, float] = affinities or {}
        # Revert payloads for the automatic per-species-level HP/MP/SP
        # growth (see Engine._reapply_species_stat_growth) — tracked here,
        # same "hold onto what was applied so it can be cleanly swapped
        # instead of stacked" pattern SkillInstance uses for its own
        # passive bonus.
        self.species_stat_growth: List[Dict[str, Any]] = []

    # --- species level ------------------------------------------------------

    @property
    def max_class_slots(self) -> int:
        return sum(1 for m in CLASS_SLOT_MILESTONES if self.species_level >= m)

    def gain_species_exp(self, amount: float) -> List[str]:
        notices: List[str] = []
        if self.species_level >= SPECIES_MAX_LEVEL:
            return notices
        self.species_exp += amount
        while (
            self.species_level < SPECIES_MAX_LEVEL
            and self.species_exp >= species_exp_to_next(self.species_level)
        ):
            self.species_exp -= species_exp_to_next(self.species_level)
            self.species_level += 1
            notices.append(f"Reached level {self.species_level}!")
        return notices

    # --- classes --------------------------------------------------------------

    def affinity_for(self, class_id: str) -> float:
        return self.affinities.get(class_id, DEFAULT_AFFINITY)

    def get_class(self, class_id: str) -> Optional[ClassProgress]:
        return next((c for c in self.classes if c.class_id == class_id), None)

    def can_add_class(self, class_id: str, unlocked_classes: Set[str]) -> bool:
        return (
            class_id in unlocked_classes
            and self.get_class(class_id) is None
            and len(self.classes) < self.max_class_slots
        )

    def add_class(self, class_id: str, unlocked_classes: Set[str]) -> Optional[ClassProgress]:
        """Free if unlocked and a slot is open; None if not allowed."""
        if not self.can_add_class(class_id, unlocked_classes):
            return None
        progress = ClassProgress(class_id)
        self.classes.append(progress)
        return progress

    def gain_class_exp(self, class_id: str, amount: float) -> List[str]:
        progress = self.get_class(class_id)
        if progress is None:
            return []
        return progress.gain_exp(amount, self.affinity_for(class_id))

    # --- skill trees ----------------------------------------------------------

    def tree_state(self, source: str):
        """(tree, owner_level, claimed_set, points_per_level) for a source
        like "class:warrior". Species trees are always auto-claimed linear
        rivers (every node cost 0 by convention, points_per_level a fixed 1
        — never actually a binding constraint, just keeps the formula
        uniform). Innate trees read their own `InnateDef.points_per_level`
        — still 1 with cost-0 nodes for most characters (an auto-claimed
        river, same as species), but a character flagged `manual=True`
        (see `_linear_sources`) can have real per-node costs and its own
        curve, claimed manually via the same menu class trees use."""
        kind, _, ident = source.partition(":")
        if kind == "species":
            return skill_data.species_def(ident).tree, self.species_level, self.species_claimed, 1
        if kind == "innate":
            innate_def = skill_data.innate_def(ident)
            return innate_def.tree, self.species_level, self.innate_claimed, innate_def.points_per_level
        if kind == "class":
            progress = self.get_class(ident)
            if progress is None:
                return None, 0, set(), 1
            return progress.definition.tree, progress.level, progress.claimed_nodes, progress.definition.points_per_level
        raise ValueError(f"Unknown tree source: {source!r}")

    def available_nodes(self, source: str) -> List[TreeNodeDef]:
        """Nodes reachable by level/prereqs/excludes, regardless of whether
        they're currently affordable — see `points_available` for that."""
        tree, level, claimed, _ = self.tree_state(source)
        if tree is None:
            return []
        return tree.available_nodes(level, claimed)

    def points_available(self, source: str) -> int:
        """This tree's current spendable pool: `(level - 1) * points_per_level`
        minus whatever's already spent on its own claimed nodes' costs.
        Level 1 itself nets zero points — there's nothing to spend one on
        yet (only the starter node is reachable), so the curve starts
        accruing from level 2 instead of banking an unusable point at 1."""
        tree, level, claimed, points_per_level = self.tree_state(source)
        if tree is None:
            return 0
        spent = sum(tree.nodes[n].cost for n in claimed if n in tree.nodes)
        return max(0, level - 1) * points_per_level - spent

    def claim_node(self, source: str, node_id: str, book: SkillBook) -> List[str]:
        """Claim a tree node: grant its skills/upgrades into `book`.
        Returns message-log notices; empty list means the claim failed
        (not available / already claimed / can't afford its cost)."""
        tree, level, claimed, points_per_level = self.tree_state(source)
        if tree is None:
            return []
        node = tree.nodes.get(node_id)
        if node is None or node not in tree.available_nodes(level, claimed):
            return []
        if node.cost > self.points_available(source):
            return []

        claimed.add(node_id)
        notices: List[str] = []
        for skill_id in node.grants_skills:
            # If the character already has this skill from another source
            # (e.g. a weapon that happens to grant the same skill_id a
            # class tree node also grants), don't overwrite it — SkillBook
            # keys purely by def_id, so blindly adding here would replace
            # the existing SkillInstance (losing its level/upgrades) and
            # leave it misattributed to this source, which would then wrongly
            # survive/vanish on this source's own reset/unequip logic.
            if book.get(skill_id) is None:
                book.add(SkillInstance(skill_id, source=source))
            notices.append(f"Learned {skill_data.skill(skill_id).name}.")
        for upgrade_id in node.grants_upgrades:
            upgrade = skill_data.upgrade(upgrade_id)
            instance = book.get(upgrade.skill_id)
            if instance is not None and level >= upgrade.requires_level:
                instance.add_upgrade(upgrade_id)
                notices.append(f"{instance.resolved.name} upgraded.")
        return notices

    def auto_claim_linear(self, book: SkillBook) -> List[str]:
        """Species/innate trees flow like a river: claim everything currently
        reachable and affordable. Class trees stay manual (they branch).

        `progressed` only flips true on an actual successful claim — a node
        that's level/prereq-reachable but too expensive right now must NOT
        count as progress, or this loops forever (it never gets claimed, so
        it never leaves `available_nodes`, and nothing else changes between
        passes). A river tree can have real per-node costs (see Kisume's/
        Yamame's, added to line up exactly with points earned by level 5)
        as long as the total never outpaces total points available — this
        just makes an over-budget node get silently skipped instead of
        freezing the game."""
        notices: List[str] = []
        for source in self._linear_sources():
            progressed = True
            while progressed:
                progressed = False
                for node in self.available_nodes(source):
                    claimed_notices = self.claim_node(source, node.node_id, book)
                    if claimed_notices:
                        notices += claimed_notices
                        progressed = True
        return notices

    def _linear_sources(self) -> List[str]:
        sources = [f"species:{self.species_id}"]
        if self.innate_id is not None and not skill_data.innate_def(self.innate_id).manual:
            sources.append(f"innate:{self.innate_id}")
        return sources

    # --- run lifecycle ----------------------------------------------------------

    def reset_all(self) -> None:
        """Full reset on returning to town (death or voluntary exit): species/
        innate progress and all class assignments revert, as if freshly
        recruited. Unlocked classes (account-wide) are untouched — the
        character just has to invest in them again next run."""
        self.species_level = 1
        self.species_exp = 0.0
        self.species_claimed.clear()
        self.innate_claimed.clear()
        self.classes.clear()
