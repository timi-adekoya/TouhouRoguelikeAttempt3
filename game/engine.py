from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tcod.console import Console
from tcod.context import Context

from components import progression as prog
from components.inventory import Inventory, ItemInstance
from components.items import EquippableDef, UsableDef
from components.skills import (
    CATEGORY_ACTIVE,
    CATEGORY_PASSIVE,
    DEFAULT_TRIGGER_EXP_REWARD,
    SKILL_EXP_PASSIVE_TICK,
    TRIGGER_EXP_REWARDS,
    ActivationTargetingDef,
    SkillBook,
    SkillInstance,
)
from components.stats import RESOURCE_POOLS, Attributes, StatBlock, apply_stat_modifier, revert_stat_modifier
from game import bestiary, casting, combat, equipment, fov, inspect_text, item_data, location_data, message_log as message_log_module, save as save_module, skill_data, targeting, tile_types, ui
from game.actions import Action, MeleeAttackAction, MovementAction, SwapPlacesAction, WaitAction
from game import ai
from game.ai import UtilityAIController
from game import effects as effects_module
from game.effects import TimedEffectInstance
from game.entity import Entity
from game.event_bus import EventBus
from game.game_map import VIEWPORT_HEIGHT, VIEWPORT_WIDTH, GameMap
from game.menu import DialogueMenu, MenuOption
from game.message_log import MessageLog
from game.procgen import (
    generate_boss_arena,
    generate_bridge_arena,
    generate_cave,
    generate_dungeon,
    generate_tunnels,
)
from game.targeting import TargetPoint, TargetSelector
from game.turn_queue import TurnQueue


class Engine:
    def __init__(self, entities: Iterable[Entity], game_map: GameMap, player: Entity):
        self.entities = list(entities)
        self.game_map = game_map
        self.player = player
        self.camera_x = 0
        self.camera_y = 0
        # A single party-wide wallet, not per-character — a kill's gold used
        # to go to whichever entity (player or ally) actually landed the
        # blow, silently piling up in an AI-controlled ally's own inventory
        # that the shop/UI never reads from (only self.player's was ever
        # shown/spendable). Every gold source (loot, shop purchases, class
        # unlocks) reads/writes this instead now.
        self.gold = 0
        # Debug-only: see should_log_progression — off by default so every
        # monster's own species/passive leveling doesn't spam the log.
        self.debug_log_all_progression = False
        self.active_menu: Optional[DialogueMenu] = None
        self.pause_menu_open = False
        self.character_screen_open = False
        self.character_screen_entity: Optional[Entity] = None
        self.character_sheet_cursor = 0
        self.target_selector: Optional[TargetSelector] = None
        self.turn_queue = TurnQueue()
        self.message_log = MessageLog()
        self.message_log_expanded = False
        self.log_scroll_offset = 0
        self.inspect_open = False
        self.inspect_title = ""
        self.inspect_lines: List[str] = []
        self.event_bus = EventBus()
        self.event_bus.subscribe("on_healed", self._on_healed_endurance_cleanse)
        self.event_bus.subscribe("on_damaged", self._on_damaged_bleed_mark)
        self.event_bus.subscribe("on_evaded", self._on_evaded_stack)
        self.event_bus.subscribe("on_move_skill", self._on_move_skill_buff)
        self.event_bus.subscribe("on_damaged", self._on_damaged_trigger_exp)
        self.event_bus.subscribe("on_healed", self._on_healed_trigger_exp)
        self.event_bus.subscribe("on_evaded", self._on_evaded_trigger_exp)
        self.event_bus.subscribe("on_move_skill", self._on_move_skill_trigger_exp)
        self.event_bus.subscribe("on_resource_spent", self._on_resource_spent_trigger_exp)
        # Runtime containers the TurnManager duties operate on.
        self.timed_effects: List[TimedEffectInstance] = []
        # (turns_remaining, cast_context, effect_id, target_point)
        self.delayed_triggers: List[list] = []
        # Account-wide class unlocks: Warrior and Mage are free/starting, the
        # rest are gold-gated via the shop. Persistent across runs, unlike
        # class levels.
        self.unlocked_classes: set = {"warrior", "mage"}
        # Non-player characters that travel with the player across floors,
        # acted by Utility AI. Distinct from `self.player` (the one the user
        # directly controls) and from monsters (no Character component).
        self.party: List[Entity] = []
        # Player/party characters at 0 HP: pulled off the map entirely (not
        # in `entities`, not in `party`) but never removed for good — always
        # revivable with their levels intact as long as someone still stands.
        self.downed: List[Entity] = []
        # Same-call-stack signal only: set True the instant a run-ending
        # reset happens (total party wipe) so an in-progress turn loop stops
        # immediately; cleared as soon as it's observed. Not a persistent
        # "game over" mode — the player is back in town by the time control
        # returns to input handling.
        self.game_over = False
        # Indexed floor history (index 0 = town) + a depth pointer, so
        # ascending/descending revisits the exact floor left behind instead
        # of regenerating it. A new floor is only generated the first time
        # depth goes past what's been visited.
        self.floor_history: List[Tuple[GameMap, List[Entity], Tuple[int, int]]] = [
            (self.game_map, self.entities, (self.player.x, self.player.y))
        ]
        self.floor_depth = 0

        # Location/gap state. `current_location` is None while in the
        # tutorial (floors 1-5) or town. `location_entry_depth` is the depth
        # the current location's floor 1 sits at — ascending past it is
        # blocked (locations are one-way; only death/give-up gets you out).
        self.current_location: Optional[str] = None
        self.location_floor_index = 0
        self.location_floor_target = 0
        self.location_entry_depth = 0
        self.pending_boss: Optional[Entity] = None
        self.pending_boss_id: Optional[str] = None
        # Permanent, town-level: which locations/bosses have ever been seen.
        # Used to obscure an unseen boss's name in the gap menu as "???"
        # (location names are never obscured).
        self.knowledge: Dict[str, Set[str]] = {"locations": set(), "bosses": set(), "recruited_bosses": set()}
        # Player-designated focus-fire target: party AI prioritizes this
        # over its own nearest-hostile pick (see `UtilityAIController`),
        # and the player's own targeting reticle defaults to it too. Cleared
        # whenever it dies/leaves or a new floor is generated.
        self.focus_target: Optional[Entity] = None
        # Set while auto-explore is driving the player automatically —
        # main.py's loop checks this to advance one step per frame (via
        # `auto_explore_step`) instead of blocking on real input, and any
        # keypress cancels it. See `start_auto_explore`.
        self.auto_exploring: bool = False
        self.save_slot: Optional[int] = None
        # Set by the pause menu; main.py's loop autosaves and goes back to
        # the title screen when it sees this.
        self.return_to_title = False

        skill_data.load_all()
        item_data.load_all()
        location_data.load_all()
        self.update_fov()

    TUTORIAL_FLOOR_COUNT = 5

    @property
    def in_dungeon(self) -> bool:
        return self.floor_depth > 0

    def try_move_player(self, dx: int, dy: int) -> None:
        dest_x, dest_y = self.player.x + dx, self.player.y + dy
        if not self.game_map.is_walkable(dest_x, dest_y):
            return

        blocker = next(
            (
                e
                for e in self.entities
                if e.blocks_movement and e.x == dest_x and e.y == dest_y
            ),
            None,
        )
        if blocker is not None:
            if blocker in self.party:
                # Already-recruited party members are reachable via the
                # roster menu ('o') for management, so a bump just swaps
                # formation instead of interrupting movement with a menu —
                # this also avoids the long detours 3+ member parties took
                # trying to route around each other in tight corridors.
                self.perform_player_action(SwapPlacesAction(self.player, blocker))
            elif targeting.is_hostile_to(self.player, blocker) and blocker.stats is not None:
                # Hostility takes priority over the Character check below —
                # an unrecruited boss still carries a Character component
                # (so it *can* be recruited on defeat), but bumping into one
                # should attack it, not pop a "What do you need?" menu.
                self.perform_player_action(MeleeAttackAction(self.player, blocker))
            elif blocker.character is not None:
                self.open_interaction_menu(blocker)
            elif blocker.stats is not None:
                self.perform_player_action(MeleeAttackAction(self.player, blocker))
            return

        self.perform_player_action(MovementAction(self.player, dx, dy))

    def start_auto_explore(self) -> None:
        """Kick off auto-explore: walk the player toward the nearest
        unexplored reachable tile, one step (one full turn, enemies included)
        per frame, until there's nothing left to explore, a hostile comes
        into view, or the player cancels with any keypress. Refuses to start
        at all (no-op, no message spam) if any menu/targeting/charge state
        is already active, or there's simply nowhere left to explore."""
        if (
            self.active_menu is not None
            or self.target_selector is not None
            or self.pause_menu_open
            or self.player.charging is not None
        ):
            return
        if ai.find_frontier_tile(self, self.player) is None:
            self.message_log.add_message("Nothing left to explore.", color=message_log_module.INFO_COLOR)
            return
        self.auto_exploring = True

    def auto_explore_step(self) -> None:
        """One step of an in-progress auto-explore — called once per frame
        by main.py while `auto_exploring` is set. Re-checks for a visible
        hostile and a valid frontier fresh every step (not cached), since
        both can change out from under a multi-turn walk (an enemy wanders
        into view, or the frontier tile gets explored by the walk itself)."""
        if not self.auto_exploring:
            return
        if ai.find_nearest_hostile(self, self.player) is not None:
            self.auto_exploring = False
            self.message_log.add_message("You spot something nearby!", color=message_log_module.INFO_COLOR)
            return
        frontier = ai.find_frontier_tile(self, self.player)
        if frontier is None:
            self.auto_exploring = False
            self.message_log.add_message("Exploration complete.", color=message_log_module.INFO_COLOR)
            return
        step = ai.pathfind_step(
            self, self.player, frontier[0], frontier[1], stop_short=False, only_visible_hostiles=True
        )
        if step is None:
            self.auto_exploring = False
            self.message_log.add_message("Can't find a way there.", color=message_log_module.INFO_COLOR)
            return
        self.perform_player_action(step)

    def perform_player_action(self, action: Action) -> None:
        action.perform(self)
        self._end_player_turn()

    def try_player_cast(self, def_id: str, chosen: Optional[Tuple[int, int]] = None) -> bool:
        """Cast one of the player's skills at a chosen tile. Only a successful
        cast spends the turn. Single-stage only — a skill with more than one
        activation_targeting stage goes through `_begin_multi_stage_cast`/
        `_finish_multi_stage_cast` instead (see `begin_cast`)."""
        book = self.player.skill_book
        instance = book.get(def_id) if book is not None else None
        if instance is None:
            return False

        stages = instance.resolved.activation_targeting
        targets = (
            targeting.resolve_targets(self, self.player, stages[0], chosen)
            if stages
            else [TargetPoint(self.player.x, self.player.y, self.player)]
        )
        if not casting.cast_skill(self, self.player, instance, targets):
            return False
        self._end_player_turn()
        return True

    # --- skill casting UI ----------------------------------------------------

    def open_skill_menu(self) -> None:
        """List the player's active skills as a cast menu, plus any
        toggleable passives at the bottom (selecting one flips it on/off
        in place and refreshes this same menu, rather than casting)."""
        book = self.player.skill_book
        actives = sorted(
            book.by_category(CATEGORY_ACTIVE) if book is not None else [],
            key=lambda s: s.resolved.name,
        )
        toggleables = sorted(
            (s for s in (book.by_category(CATEGORY_PASSIVE) if book is not None else []) if s.resolved.toggleable),
            key=lambda s: s.resolved.name,
        )
        options = [
            MenuOption(
                self._skill_label(instance),
                lambda i=instance: self._select_skill(i),
                description=instance.resolved.description,
                inspect=lambda i=instance: inspect_text.skill_lines(i),
            )
            for instance in actives
        ]
        options += [
            MenuOption(
                f"{instance.resolved.name} [{'ON' if instance.enabled else 'OFF'}]",
                lambda i=instance: self._menu_toggle_passive_skill(i),
                description=instance.resolved.description,
                inspect=lambda i=instance: inspect_text.skill_lines(i),
            )
            for instance in toggleables
        ]
        options.append(MenuOption("Cancel", self.close_menu))
        prompt = "Use which skill?" if actives or toggleables else "You have no active skills."
        self.active_menu = DialogueMenu(
            speaker=self.player.name, prompt=prompt, options=options
        )

    def _menu_toggle_passive_skill(self, instance: SkillInstance) -> None:
        self.toggle_passive(self.player, instance)
        self.open_skill_menu()  # refresh so the [ON]/[OFF] label updates

    def _skill_label(self, instance: SkillInstance) -> str:
        skill = instance.resolved
        parts = [skill.name]
        cost = casting.effective_cost(self.player, skill)
        if cost:
            parts.append("[" + " ".join(f"{k.upper()} {v}" for k, v in cost.items()) + "]")
        if instance.cooldown_remaining > 0:
            parts.append(f"(CD {instance.cooldown_remaining})")
        return "  ".join(parts)

    def _select_skill(self, instance: SkillInstance) -> None:
        self.close_menu()
        self.begin_cast(instance)

    def begin_cast(self, instance: SkillInstance) -> None:
        """Cast a self-targeting skill immediately, otherwise enter the
        map-reticle targeting mode. A revive skill bypasses targeting
        entirely, same as a revive item — downed characters aren't on the
        map to target, so it opens a "who to revive" menu instead. A skill
        with more than one activation_targeting stage (e.g. teleport-other:
        pick the victim, then pick a destination) chains reticles instead of
        firing after the first."""
        skill = instance.resolved
        if any(effect.type == "revive_ally" for effect in skill.effects):
            self.open_revive_menu(instance)
            return

        stages = skill.activation_targeting
        if not stages or stages[0].range_shape == "self":
            self.try_player_cast(instance.def_id)
            return
        if len(stages) > 1:
            self._begin_multi_stage_cast(instance, 0, {})
            return
        self.target_selector = TargetSelector(
            self, self.player, stages[0], lambda pos: self.try_player_cast(instance.def_id, pos)
        )

    def _begin_multi_stage_cast(
        self, instance: SkillInstance, stage_index: int, prior_targets: Dict[int, List[TargetPoint]]
    ) -> None:
        stages = instance.resolved.activation_targeting
        stage = stages[stage_index]

        def on_confirm(pos: Tuple[int, int]) -> bool:
            resolved = targeting.resolve_targets(self, self.player, stage, pos)
            if not resolved:
                self.message_log.add_message("No valid target.", color=message_log_module.INFO_COLOR)
                return False
            next_targets = dict(prior_targets)
            next_targets[stage_index] = resolved
            if stage_index + 1 < len(stages):
                # Advance to the next stage's reticle; returning False keeps
                # target_selector as whatever this just assigned instead of
                # letting confirm_targeting() null it back out.
                self._begin_multi_stage_cast(instance, stage_index + 1, next_targets)
                return False
            return self._finish_multi_stage_cast(instance, next_targets)

        self.target_selector = TargetSelector(self, self.player, stage, on_confirm)

    def _finish_multi_stage_cast(
        self, instance: SkillInstance, stage_targets: Dict[int, List[TargetPoint]]
    ) -> bool:
        primary = stage_targets.get(0, [])
        extra = {k: v for k, v in stage_targets.items() if k != 0}
        if not casting.cast_skill(self, self.player, instance, primary, extra):
            return False
        self._end_player_turn()
        return True

    def move_reticle(self, dx: int, dy: int) -> None:
        if self.target_selector is not None:
            self.target_selector.move(dx, dy, self)

    def confirm_targeting(self) -> None:
        if self.target_selector is not None and self.target_selector.confirm(self):
            self.target_selector = None  # a legal cast ends targeting

    def cancel_targeting(self) -> None:
        self.target_selector = None

    def open_mark_target_selector(self) -> None:
        """Free (no turn cost) focus-fire designation: opens the same map
        reticle used for skill/item targeting, but just records who was
        picked instead of casting anything. Party AI (`UtilityAIController`)
        prioritizes this over its own nearest-hostile pick, and the
        targeting reticle itself defaults to it too. Picking the entity
        already marked toggles the mark off."""
        stage = ActivationTargetingDef(
            selection_mode="single", range_shape="point", shape_params={"range": 20},
            restriction="enemy", target_kind="entity",
        )

        def on_confirm(pos: Tuple[int, int]) -> bool:
            resolved = targeting.resolve_targets(self, self.player, stage, pos)
            target = resolved[0].entity if resolved else None
            if target is None:
                self.message_log.add_message("No valid target.", color=message_log_module.INFO_COLOR)
                return False
            if self.focus_target is target:
                self.focus_target = None
                self.message_log.add_message(
                    f"No longer focusing {target.name}.", color=message_log_module.INFO_COLOR, stack=False
                )
            else:
                self.focus_target = target
                self.message_log.add_message(
                    f"Focusing {target.name}!", color=message_log_module.INFO_COLOR, stack=False
                )
            return True

        self.target_selector = TargetSelector(self, self.player, stage, on_confirm)

    def _end_player_turn(self) -> None:
        """TurnManager duties, at player-turn granularity: cooldowns, timed
        effect ticks/pruning, delayed triggers, class passive exp trickle.

        A DoT tick or delayed effect can itself kill the player and trigger
        a run-ending reset (`end_run`, via `game_over` as a same-call-stack
        signal) — if that happens mid-loop, stop touching the now-stale
        `timed_effects`/`delayed_triggers` lists and bail out entirely.
        """
        self.turn_queue.spend(self.player)
        self._advance_charge(self.player)

        if self.player.skill_book is not None:
            self.player.skill_book.tick_cooldowns()

        for timed in list(self.timed_effects):
            timed.tick(self)
            if self.game_over:
                break
            if timed.remaining_turns <= 0 or timed.target not in self.entities:
                timed.on_expire(self)
                self.timed_effects.remove(timed)
        if self.game_over:
            self.game_over = False
            return

        for entry in list(self.delayed_triggers):
            entry[0] -= 1
            if entry[0] <= 0:
                self.delayed_triggers.remove(entry)
                _, ctx, effect_id, target = entry
                ctx.run_effect(effect_id, [target])
            if self.game_over:
                break
        if self.game_over:
            self.game_over = False
            return

        # Every entity currently on the map grows from its own passives each
        # turn, not just the controlled player — otherwise party members (and
        # monsters) sit still, XP-wise, whenever they aren't the one you're
        # driving. (Class exp no longer trickles here — see
        # _award_trigger_exp, which feeds it from actual passive procs.)
        for entity in list(self.entities):
            self._tick_entity_passives(entity)
        self.event_bus.emit("on_turn_end", {"entity": self.player})
        self._process_enemy_turns()

    def _on_healed_endurance_cleanse(self, payload: dict) -> None:
        """Endurance (Warrior) capstone: whenever this entity gains HP from
        any source (Battle Fortitude's regen tick included), shorten a
        handful of its own active debuffs. Gated on two flat combat
        modifiers (`endurance_cleanse_count`/`endurance_cleanse_turns`) so
        it's a no-op for anyone who doesn't have the capstone passive."""
        target = payload.get("target")
        if target is None or target.stats is None:
            return
        count = int(target.stats.modifiers.get("endurance_cleanse_count", 0))
        turns = int(target.stats.modifiers.get("endurance_cleanse_turns", 0))
        if count <= 0 or turns <= 0:
            return

        candidates = [t for t in self.timed_effects if t.target is target and effects_module.is_debuff(t)]
        if not candidates:
            return

        for timed in random.sample(candidates, min(count, len(candidates))):
            timed.remaining_turns -= turns
            if timed.remaining_turns <= 0:
                timed.on_expire(self)
                if timed in self.timed_effects:
                    self.timed_effects.remove(timed)
        self.message_log.add_message(
            f"{target.name}'s resolve shortens a lingering affliction.", color=message_log_module.INFO_COLOR, stack=False
        )

    def _on_damaged_bleed_mark(self, payload: dict) -> None:
        """Rogue capstone: a target marked by Exposed Wound (applied
        alongside Stab's bleed — see effects._dot_hot) takes a small extra
        hit of damage from ANY attack that lands on them, from any source,
        as many times per turn as they get hit. `bonus_damage` is snapshot
        onto the mark's own payload when it's applied, not read live off
        the original caster, so the proc still works even if that caster
        is gone. Guarded by `cause` to avoid the proc re-triggering itself."""
        if payload.get("cause") == "Exposed Wound":
            return
        target = payload.get("target")
        if target is None or target.stats is None:
            return
        mark = next(
            (t for t in self.timed_effects if t.kind == "status" and t.target is target and t.payload.get("status") == "marked"),
            None,
        )
        if mark is None:
            return
        bonus = mark.payload.get("bonus_damage", 0)
        if bonus > 0:
            combat.apply_damage(self, payload.get("source"), target, bonus, cause="Exposed Wound", periodic=True)

    def _on_evaded_stack(self, payload: dict) -> None:
        """Rogue's Reflexes capstone: every attack the entity evades banks
        a stack (capped by `evasion_stack_cap`), each worth
        `evasion_stack_percent` more damage on their own next hit — see
        combat._consume_evasion_stacks for where it's spent. A no-op for
        anyone without the capstone (cap defaults to 0)."""
        target = payload.get("target")
        if target is None or target.stats is None:
            return
        cap = target.stats.modifiers.get("evasion_stack_cap", 0)
        if cap <= 0:
            return
        current = target.stats.modifiers.get("evasion_stacks_current", 0)
        if current < cap:
            target.stats.modifiers["evasion_stacks_current"] = current + 1

    def _on_move_skill_buff(self, payload: dict) -> None:
        """Skirmisher (Ranger) mid perk: using any movement skill (Disengage,
        or any future one — the trigger is "the caster relocated via a
        skill", not any one named ability) grants a brief technique/speed
        buff. A no-op for anyone without the perk (amount defaults to 0)."""
        caster = payload.get("caster")
        if caster is None or caster.stats is None:
            return
        amount = caster.stats.modifiers.get("movement_buff_amount", 0.0)
        if amount <= 0:
            return
        duration = int(caster.stats.modifiers.get("movement_buff_duration", 2))
        for attribute in ("technique", "speed"):
            revert = apply_stat_modifier(caster.stats, attribute, amount)
            self.timed_effects.append(
                TimedEffectInstance(kind="stat_mod", target=caster, remaining_turns=duration, payload={"revert": revert})
            )
        if caster is self.player:
            self.message_log.add_message("Quickened by the retreat!", color=message_log_module.INFO_COLOR)

    def _advance_charge(self, entity: Entity) -> None:
        """Count down one of `entity`'s own turns against an in-progress
        charge-cast (see game/casting.py); auto-resolves at 0. Called once
        per turn taken by the charging entity — for the player that's the
        end of `_end_player_turn`, for AI actors it's right after they
        spend energy in `_process_enemy_turns`."""
        charge = entity.charging
        if charge is None:
            return
        charge.turns_remaining -= 1
        if charge.turns_remaining <= 0:
            entity.charging = None
            casting.resolve_charge(self, entity, charge)

    def _tick_entity_passives(self, entity: Entity) -> None:
        """Per-turn duties for the entity's own passive skills: ambient
        effects (regen) always run (on whatever period they declare); each
        passive without its own `exp_trigger` also earns a little exp just
        from being active (a plain stat-stick passive, e.g. Warrior's
        Toughness — nothing it "does" to hook an event to). A passive WITH
        an `exp_trigger` set skips this trickle entirely — it only earns
        exp via `_award_trigger_exp`, from actually doing its thing. Either
        way, leveling up reapplies its stat_modifier bonus to match."""
        if entity.skill_book is None or entity.stats is None:
            return
        for instance in entity.skill_book.by_category(CATEGORY_PASSIVE):
            for effect in instance.resolved.effects:
                if effect.type == "resource_regen":
                    period = effect.params.get("period", 1)
                    elapsed = instance.tick_counters.get(effect.effect_id, 0) + 1
                    if elapsed >= period:
                        elapsed = 0
                        pool = getattr(entity.stats, effect.params["resource"], None)
                        if pool is not None:
                            pool.modify(effect.params["amount"])
                    instance.tick_counters[effect.effect_id] = elapsed

            if instance.resolved.exp_trigger is not None:
                continue
            notices = instance.gain_exp(SKILL_EXP_PASSIVE_TICK)
            if notices:
                if self.should_log_progression(entity):
                    for notice in notices:
                        self.message_log.add_message(
                            notice, color=message_log_module.INFO_COLOR, stack=False
                        )
                self._reapply_passive(entity, instance)

    def _reapply_stat_modifiers(self, stats: StatBlock, wanted: List[Tuple[str, str, float]], old_deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Move a set of named stat_modifier bonuses from whatever was
        applied last time (`old_deltas`) to `wanted` (list of (key,
        attribute, amount)) — the shared "swap not stack" primitive both
        `reapply_species_stat_growth` and `_reapply_passive` use so a
        level-up/re-level resizes a bonus instead of stacking it.

        A resource-pool attribute (hp/mp/sp/faith) is moved with a single
        net-delta `apply_stat_modifier` call instead of a separate
        revert-then-reapply pair: reverting a big old bonus while current HP
        is already low clamps at the pool's floor and silently loses track
        of how much should've come back out, so the following reapply's
        full new amount would land on top of that artificially-low current
        — an unintended near-full heal on any level-up taken while below
        the old bonus. A single net call never passes through that broken
        intermediate state. Non-resource attributes/modifiers don't have
        this failure mode, so they keep the simpler revert-then-reapply."""
        old_by_key = {p["key"]: p for p in old_deltas if "key" in p}
        new_deltas: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()
        for key, attribute, amount in wanted:
            seen_keys.add(key)
            old_payload = old_by_key.get(key)
            if attribute in RESOURCE_POOLS and old_payload is not None:
                net = amount - old_payload.get("amount", 0)
                payload = apply_stat_modifier(stats, attribute, net) if net else dict(old_payload)
            else:
                if old_payload is not None:
                    revert_stat_modifier(stats, old_payload)
                payload = apply_stat_modifier(stats, attribute, amount)
            payload["key"] = key
            payload["amount"] = amount
            new_deltas.append(payload)
        for key, payload in old_by_key.items():
            if key not in seen_keys:
                revert_stat_modifier(stats, payload)
        return new_deltas

    def reapply_species_stat_growth(self, entity: Entity) -> None:
        """(Re)apply the automatic per-species-level HP/MP/SP baseline for
        the entity's current species_level. Level 1 nets zero bonus levels,
        so a freshly reset/created character just ends up with nothing
        applied. See `_reapply_stat_modifiers` for why this resizes the
        bonus in place instead of a plain revert-then-reapply."""
        if entity.progression is None or entity.stats is None:
            return
        bonus_levels = max(0, entity.progression.species_level - 1)
        wanted = [
            ("hp", "hp", bonus_levels * prog.SPECIES_HP_GROWTH_PER_LEVEL),
            ("mp", "mp", bonus_levels * prog.SPECIES_MP_GROWTH_PER_LEVEL),
            ("sp", "sp", bonus_levels * prog.SPECIES_SP_GROWTH_PER_LEVEL),
        ]
        entity.progression.species_stat_growth = self._reapply_stat_modifiers(
            entity.stats, wanted, entity.progression.species_stat_growth
        )

    def _award_trigger_exp(self, entity: Optional[Entity], trigger: str) -> None:
        """Give every one of `entity`'s passives whose `exp_trigger` matches
        `trigger` a chunk of exp (TRIGGER_EXP_REWARDS[trigger] — varies per
        trigger, since some fire far more often than others), reapplying
        any that level up. A class-sourced passive also feeds its class's
        own level (CLASS_EXP_PASSIVE_TRIGGER) — the replacement for the
        removed per-turn class-passive trickle, gated on the passive
        actually doing something instead of existing. Shared tail end for
        all the event-bus listeners below — a no-op for anyone with no
        matching passive, so it's safe to call unconditionally from a
        generic combat/heal/evade hook."""
        if entity is None or entity.skill_book is None:
            return
        reward = TRIGGER_EXP_REWARDS.get(trigger, DEFAULT_TRIGGER_EXP_REWARD)
        for instance in entity.skill_book.by_category(CATEGORY_PASSIVE):
            if instance.resolved.exp_trigger != trigger:
                continue
            notices = instance.gain_exp(reward)
            if notices:
                if self.should_log_progression(entity):
                    for notice in notices:
                        self.message_log.add_message(
                            notice, color=message_log_module.INFO_COLOR, stack=False
                        )
                self._reapply_passive(entity, instance)

            if entity.progression is not None:
                kind, _, ident = instance.source.partition(":")
                if kind == "class":
                    class_exp = prog.CLASS_EXP_PASSIVE_TRIGGER
                    if entity.stats is not None:
                        class_exp *= 1 + entity.stats.modifiers.get("class_exp_percent", 0.0) / 100
                    for notice in entity.progression.gain_class_exp(ident, class_exp):
                        self.message_log.add_message(
                            notice, color=message_log_module.INFO_COLOR, stack=False
                        )

    def _on_damaged_trigger_exp(self, payload: dict) -> None:
        amount = payload.get("amount", 0)
        if amount <= 0:
            return
        self._award_trigger_exp(payload.get("source"), "on_damage_dealt")
        self._award_trigger_exp(payload.get("target"), "on_hp_change")

    def _on_healed_trigger_exp(self, payload: dict) -> None:
        amount = payload.get("amount", 0)
        if amount <= 0:
            return
        self._award_trigger_exp(payload.get("source"), "on_heal_cast")
        self._award_trigger_exp(payload.get("target"), "on_hp_change")

    def _on_evaded_trigger_exp(self, payload: dict) -> None:
        self._award_trigger_exp(payload.get("target"), "on_evaded")

    def _on_move_skill_trigger_exp(self, payload: dict) -> None:
        self._award_trigger_exp(payload.get("caster"), "on_move_skill")

    def _on_resource_spent_trigger_exp(self, payload: dict) -> None:
        self._award_trigger_exp(payload.get("caster"), "on_resource_consumed")

    def _process_enemy_turns(self) -> None:
        others = [e for e in self.entities if e is not self.player]
        if not others:
            return

        actors = [self.player] + others
        while True:
            # Re-filter every iteration: an earlier actor this same call may
            # have killed another one still sitting in `actors` — without
            # this, a dead entity (already gone from `self.entities`) could
            # still get picked and act on its now-stale turn.
            actors = [a for a in actors if a is self.player or a in self.entities]
            actor = self.turn_queue.next_ready(actors)
            if actor is self.player:
                return
            if actor.skill_book is not None:
                actor.skill_book.tick_cooldowns()
            action = actor.ai.decide(self, actor) if actor.ai is not None else WaitAction(actor)
            action.perform(self)
            if self.game_over:
                self.game_over = False
                return
            self.turn_queue.spend(actor)
            self._advance_charge(actor)
            if self.game_over:
                self.game_over = False
                return

    def remove_entity(self, entity: Entity) -> None:
        self.entities = [e for e in self.entities if e is not entity]
        if entity in self.party:
            self.party.remove(entity)
        if self.focus_target is entity:
            self.focus_target = None

    # --- downed / revival ------------------------------------------------------

    def mark_downed(self, entity: Entity) -> None:
        """Pull a player/party character off the map at 0 HP. Not removed for
        good — stays revivable with progression intact as long as someone
        else still stands."""
        self.remove_entity(entity)
        if entity not in self.downed:
            self.downed.append(entity)

    def handle_character_down(self, entity: Entity) -> None:
        """A player- or party-affiliated character hit 0 HP. If it's the
        controlled player and a living party member remains, hand control
        to them instead of ending anything. Only a true wipe (no one left
        standing) ends the run."""
        was_player = entity is self.player
        self.message_log.add_message(
            f"{entity.name} falls!", color=message_log_module.DEATH_COLOR, stack=False
        )
        self.mark_downed(entity)

        if not was_player:
            return

        survivor = next(
            (m for m in self.party if m.stats is not None and not m.stats.hp.is_empty), None
        )
        if survivor is None:
            # Only meaningful mid-turn-loop: signals _end_player_turn /
            # _process_enemy_turns to bail instead of touching now-stale state.
            self.game_over = True
            self.end_run(defeated=True)
            return

        self.player = survivor
        self.party.remove(survivor)
        survivor.ai = None
        self.update_fov()
        self.message_log.add_message(
            f"{survivor.name} takes command!", color=message_log_module.INFO_COLOR, stack=False
        )

    def end_run(self, defeated: bool) -> None:
        """Return to town, resetting everything the dungeon run itself
        granted — species/innate level, class assignments, non-keepsake loot
        — for every character who went in. Triggered by a total wipe or by
        voluntarily ascending back to town from floor 1. Persistent stuff
        (roster, unlocked classes, gold, town state) is untouched."""
        participants = [self.player] + self.party + self.downed
        for character in participants:
            self._reset_run_progress(character)
            self._strip_run_loot(character)
            if character.stats is not None:
                character.stats.hp.current = character.stats.hp.max_value
                character.stats.mp.current = character.stats.mp.max_value
                character.stats.sp.current = character.stats.sp.max_value

        self.game_map, base_entities, (self.player.x, self.player.y) = self.floor_history[0]
        # Merge every participant back in — including anyone downed, since
        # they've just been healed and un-downed above. Every participant
        # (even one already present in the base town snapshot, like a
        # starting party member) gets repositioned: their x/y is whatever
        # they were standing at in the dungeon, which may be out of bounds
        # for the town map entirely (confirmed crash — a stale coordinate
        # from a bigger location's map indexing past the smaller town map's
        # bounds on render).
        for character in participants:
            if character is not self.player:
                character.x, character.y = self.player.x, self.player.y
        self.entities = [e for e in base_entities if e not in participants] + participants
        self.floor_history = self.floor_history[:1]
        self.floor_depth = 0
        self.update_camera()
        self.party = []
        self.downed = []
        self.timed_effects = []
        self.delayed_triggers = []
        self.current_location = None
        self.location_floor_index = 0
        self.location_floor_target = 0
        self.location_entry_depth = 0
        self.pending_boss = None
        self.pending_boss_id = None
        self.focus_target = None
        self.update_fov()

        if defeated:
            self.message_log.add_message(
                "Your party has fallen... returning to town.", color=message_log_module.DEATH_COLOR, stack=False
            )
        else:
            self.message_log.add_message(
                "You return to town.", color=message_log_module.INFO_COLOR, stack=False
            )
        # Overwrites any mid-run save, which is what makes a lost run final.
        self.save_game()

    def _reset_run_progress(self, entity: Entity) -> None:
        if entity.inventory is not None:
            for slot in list(entity.inventory.equipped):
                equipment.unequip(self, entity, slot)
        if entity.skill_book is not None:
            if entity.stats is not None:
                for instance in entity.skill_book.skills.values():
                    for payload in instance.applied_deltas:
                        revert_stat_modifier(entity.stats, payload)
            entity.skill_book.skills.clear()
        if entity.stats is not None:
            entity.stats.modifiers.clear()
        if entity.progression is not None:
            entity.progression.reset_all()
            self.auto_claim_linear_trees(entity)
            self.reapply_species_stat_growth(entity)

    def _strip_run_loot(self, entity: Entity) -> None:
        """Everything found this run is lost except gold and items tagged
        "keepsake" — the escape hatch for relics meant to survive death."""
        if entity.inventory is None:
            return
        for instance in list(entity.inventory.items):
            if "keepsake" not in instance.definition.tags:
                entity.inventory.remove(instance)

    def add_to_party(self, entity: Entity) -> bool:
        """Recruit an NPC as an AI-controlled teammate that travels with the
        player across floors, without making them the controlled character."""
        if entity is self.player or entity in self.party:
            return False
        if entity.inventory is None:
            entity.inventory = Inventory()
        entity.ai = UtilityAIController()
        self.party.append(entity)
        self.message_log.add_message(
            f"{entity.name} joins the party.", color=message_log_module.INFO_COLOR, stack=False
        )
        return True

    # --- items ---------------------------------------------------------------

    def spawn_ground_item(
        self, x: int, y: int, item_id: str, charges: Optional[int] = None, quantity: int = 1
    ) -> None:
        instance = ItemInstance(item_id, charges=charges, quantity=quantity)
        self.entities.append(
            Entity(
                x=x, y=y, char="!", color=(200, 200, 80), name=instance.definition.name,
                ground_item=instance,
            )
        )

    def try_pickup(self) -> None:
        ground = next(
            (e for e in self.entities if e.ground_item is not None and e.x == self.player.x and e.y == self.player.y),
            None,
        )
        if ground is None:
            self.message_log.add_message("Nothing here to pick up.", color=message_log_module.INFO_COLOR)
            return
        if self.player.inventory is None or not equipment.pick_up(self, self.player, ground.ground_item):
            self.message_log.add_message("Inventory is full.", color=message_log_module.INFO_COLOR)
            return
        self.message_log.add_message(
            f"Picked up {ground.ground_item.definition.name}.", color=message_log_module.INFO_COLOR, stack=False
        )
        self.remove_entity(ground)

    def open_inventory_menu(self, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        inventory = entity.inventory
        options: List[MenuOption] = []
        if inventory is not None:
            for instance in inventory.items:
                label = instance.definition.name
                if instance.uid in {i.uid for i in inventory.equipped.values()}:
                    label += " (equipped)"
                elif instance.quantity > 1:
                    label += f" x{instance.quantity}"
                elif instance.charges_remaining is not None:
                    label += f" ({instance.charges_remaining} charges)"
                options.append(
                    MenuOption(
                        label,
                        lambda i=instance: self._build_item_action_menu(i, entity),
                        description=instance.definition.description,
                        inspect=lambda i=instance: inspect_text.item_lines(i),
                    )
                )
        prompt = f"Inventory (Gold: {self.gold}):" if options else f"Inventory is empty. (Gold: {self.gold})"
        options.append(MenuOption("Leave", self.close_menu))
        self.active_menu = DialogueMenu(speaker=entity.name, prompt=prompt, options=options)

    def _build_item_action_menu(self, instance: ItemInstance, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        definition = instance.definition
        inventory = entity.inventory
        options: List[MenuOption] = []

        if isinstance(definition, EquippableDef):
            is_equipped = inventory.equipped.get(definition.slot) is instance
            if is_equipped:
                options.append(MenuOption("Unequip", lambda: self._menu_unequip(definition.slot, entity)))
            else:
                options.append(MenuOption("Equip", lambda: self._menu_equip(instance, entity)))
        elif isinstance(definition, UsableDef) and entity is self.player:
            # Item activation drives a targeting reticle hardcoded to the
            # active character; managing a non-active party member's kit
            # from the party screen is limited to equip/unequip/drop.
            options.append(MenuOption("Use", lambda: self._menu_use_item(instance)))

        options.append(MenuOption("Drop", lambda: self._menu_drop(instance, entity)))
        options.append(MenuOption("Back", lambda: self.open_inventory_menu(entity)))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=definition.name, options=options
        )

    def _menu_equip(self, instance: ItemInstance, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        equipment.equip(self, entity, instance)
        self.open_inventory_menu(entity)

    def _menu_unequip(self, slot: str, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        equipment.unequip(self, entity, slot)
        self.open_inventory_menu(entity)

    def _menu_drop(self, instance: ItemInstance, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        equipment.drop(self, entity, instance)
        self.spawn_ground_item(
            entity.x, entity.y, instance.item_id,
            charges=instance.charges_remaining, quantity=instance.quantity,
        )
        self.open_inventory_menu(entity)

    def _menu_use_item(self, instance: ItemInstance) -> None:
        self.close_menu()
        self.begin_item_use(instance)

    def begin_item_use(self, instance: ItemInstance) -> bool:
        """Use a usable item: self-targeting fires immediately, otherwise
        enters the same map-reticle targeting mode skills use. A revive item
        bypasses targeting entirely — downed characters aren't on the map to
        target, so it opens a "who to revive" menu instead."""
        definition = instance.definition
        if any(effect.type == "revive_ally" for effect in definition.effects):
            self.open_revive_menu(instance)
            return True

        stages = definition.activation_targeting
        if not stages or stages[0].range_shape == "self":
            return self.use_item(instance)
        self.target_selector = TargetSelector(
            self, self.player, stages[0], lambda pos: self.use_item(instance, pos)
        )
        return True

    def open_revive_menu(self, source) -> None:
        """`source` is whatever carries the revive_ally effect — an
        ItemInstance or a SkillInstance; both funnel through _do_revive."""
        options = [
            MenuOption(entity.name, lambda e=entity: self._do_revive(source, e))
            for entity in self.downed
        ]
        prompt = "Revive who?" if options else "No one to revive."
        options.append(MenuOption("Cancel", self.close_menu))
        self.active_menu = DialogueMenu(speaker=self.player.name, prompt=prompt, options=options)

    def _do_revive(self, source, entity: Entity) -> None:
        is_skill = isinstance(source, SkillInstance)

        if is_skill:
            skill = source.resolved
            cost = casting.effective_cost(self.player, skill)
            if not source.ready:
                self.message_log.add_message(
                    f"{skill.name} is on cooldown ({source.cooldown_remaining} turns).",
                    color=message_log_module.INFO_COLOR,
                )
                return
            if not casting.can_pay(self.player, cost):
                self.message_log.add_message(
                    f"Not enough resources for {skill.name}.", color=message_log_module.INFO_COLOR
                )
                return
            casting.pay(self.player, cost)
            source.cooldown_remaining = skill.cooldown
            revive_effect = next(e for e in skill.effects if e.type == "revive_ally")
        else:
            revive_effect = next(e for e in source.definition.effects if e.type == "revive_ally")

        self.close_menu()
        fraction = revive_effect.params.get("hp_fraction", 0.5)
        if entity.stats is not None:
            entity.stats.hp.current = max(1, int(entity.stats.hp.max_value * fraction))
        entity.x, entity.y = self.player.x, self.player.y
        self.downed.remove(entity)
        self.entities.append(entity)
        self.add_to_party(entity)
        self.message_log.add_message(
            f"{entity.name} is revived!", color=message_log_module.INFO_COLOR, stack=False
        )

        if not is_skill:
            definition = source.definition
            if definition.consumable:
                self.player.inventory.consume_one(source)
            elif source.charges_remaining is not None:
                source.charges_remaining -= 1

        self._end_player_turn()

    def use_item(self, instance: ItemInstance, chosen: Optional[Tuple[int, int]] = None) -> bool:
        definition = instance.definition
        if not isinstance(definition, UsableDef):
            return False
        if definition.consumable and instance.quantity <= 0:
            self.message_log.add_message(f"No {definition.name} left.", color=message_log_module.INFO_COLOR)
            return False
        if not definition.consumable and instance.charges_remaining is not None and instance.charges_remaining <= 0:
            self.message_log.add_message(f"{definition.name} has no charges left.", color=message_log_module.INFO_COLOR)
            return False

        stages = definition.activation_targeting
        targets = (
            targeting.resolve_targets(self, self.player, stages[0], chosen)
            if stages
            else [TargetPoint(self.player.x, self.player.y, self.player)]
        )
        if not targets:
            self.message_log.add_message("No valid target.", color=message_log_module.INFO_COLOR)
            return False

        ctx = casting.CastContext(self, self.player, casting._ResolvedWrapper(definition), targets)
        for effect in sorted(definition.effects, key=lambda e: -e.priority):
            if effect.targeting == "contextual":
                continue
            if effect.targeting == "self":
                ctx.run_effect(effect.effect_id, [TargetPoint(self.player.x, self.player.y, self.player)])
            else:
                ctx.run_effect(effect.effect_id, targets)

        if definition.consumable:
            self.player.inventory.consume_one(instance)
        elif instance.charges_remaining is not None:
            instance.charges_remaining -= 1

        self._end_player_turn()
        return True

    # --- progression management ---------------------------------------------

    def add_class_to(self, entity: Entity, class_id: str) -> bool:
        """Attach an unlocked class to a character (free, slot-gated by
        species level). Available mid-dungeon via the class menu."""
        if entity.progression is None:
            return False
        progress = entity.progression.add_class(class_id, self.unlocked_classes)
        if progress is None:
            return False
        self.message_log.add_message(
            f"{entity.name} takes up the {progress.definition.name} class.",
            color=message_log_module.INFO_COLOR,
            stack=False,
        )
        return True

    def claim_tree_node(self, entity: Entity, source: str, node_id: str) -> bool:
        """Claim a skill-tree node and immediately apply any newly granted
        passives (their effects run once, on self, permanent)."""
        if entity.progression is None or entity.skill_book is None:
            return False
        before = set(entity.skill_book.skills)
        notices = entity.progression.claim_node(source, node_id, entity.skill_book)
        if not notices:
            return False
        if self.should_log_progression(entity):
            for notice in notices:
                self.message_log.add_message(notice, color=message_log_module.INFO_COLOR, stack=False)
        for def_id in set(entity.skill_book.skills) - before:
            instance = entity.skill_book.get(def_id)
            if instance.category == CATEGORY_PASSIVE:
                self._reapply_passive(entity, instance)
        return True

    def auto_claim_linear_trees(self, entity: Entity) -> None:
        """Species/innate trees are near-linear rivers: claim whatever the
        current species level reaches. Called at spawn and on level up."""
        if entity.progression is None or entity.skill_book is None:
            return
        before = set(entity.skill_book.skills)
        notices = entity.progression.auto_claim_linear(entity.skill_book)
        if self.should_log_progression(entity):
            for notice in notices:
                self.message_log.add_message(notice, color=message_log_module.INFO_COLOR, stack=False)
        for def_id in set(entity.skill_book.skills) - before:
            instance = entity.skill_book.get(def_id)
            if instance.category == CATEGORY_PASSIVE:
                self._reapply_passive(entity, instance)

    def claim_skill_upgrade(self, entity: Entity, instance: SkillInstance, node_id: str) -> bool:
        """Manually claim a node in a skill's own (self) upgrade tree — a
        class passive getting a bigger stat bonus, an active getting a
        stronger formula, etc. Distinct from `claim_tree_node`, which claims
        nodes in the class/species/innate tree that granted the skill."""
        notices = instance.claim_tree_node(node_id)
        if not notices:
            return False
        for notice in notices:
            self.message_log.add_message(notice, color=message_log_module.INFO_COLOR, stack=False)
        if instance.category == CATEGORY_PASSIVE:
            self._reapply_passive(entity, instance)
        return True

    def toggle_passive(self, entity: Entity, instance: SkillInstance) -> bool:
        """Flip a toggleable passive on/off. Free and instant — no cost, no
        cooldown, no turn spent; just swaps whether its stat_modifier
        effects are currently applied."""
        if instance.category != CATEGORY_PASSIVE or not instance.resolved.toggleable:
            return False
        instance.enabled = not instance.enabled
        self._reapply_passive(entity, instance)
        state = "on" if instance.enabled else "off"
        self.message_log.add_message(
            f"{instance.resolved.name} is now {state}.",
            color=message_log_module.INFO_COLOR,
            stack=False,
        )
        return True

    def _reapply_passive(self, entity: Entity, instance: SkillInstance) -> None:
        """(Re)apply a passive skill's stat_modifier effects so leveling up
        (or a scale_with_level bump) resizes the old bonus into the new one
        instead of stacking both — see `_reapply_stat_modifiers` for why a
        resource-pool bonus (hp/mp/sp/faith) is resized via a single net
        delta rather than a revert-then-reapply pair. Other effect types on
        a passive (status/movement/etc.) don't fit this permanent-hold model
        and are ignored. A disabled passive wants nothing applied, same as
        before.

        `"scale_with_level": true` makes an innate passive's bonus grow with
        the skill's own level automatically (amount is a per-level
        coefficient) — no manual upgrade needed, unlike class passives,
        which only get bigger via a manually-claimed tree node."""
        if entity.stats is None:
            return
        wanted: List[Tuple[str, str, float]] = []
        if instance.enabled:
            for effect in instance.resolved.effects:
                if effect.type != "stat_modifier":
                    continue
                amount = effect.params["amount"]
                if effect.params.get("scale_with_level"):
                    amount *= instance.level
                wanted.append((effect.effect_id, effect.params["attribute"], amount))
        instance.applied_deltas = self._reapply_stat_modifiers(entity.stats, wanted, instance.applied_deltas)

    def open_door_if_present(self, x: int, y: int) -> None:
        if (x, y) in self.game_map.door_positions and not self.game_map.tiles["transparent"][x, y]:
            self.game_map.tiles[x, y] = tile_types.door_open

    def open_interaction_menu(self, entity: Entity) -> None:
        character = entity.character
        assert character is not None

        options = []
        if entity is not self.player and character.recruited:
            options.append(
                MenuOption(
                    "Set as active character",
                    lambda: self._set_active_character(entity),
                )
            )
            if entity not in self.party:
                options.append(MenuOption("Add to party", lambda: self._menu_add_to_party(entity)))
            else:
                options.append(MenuOption("Manage", lambda: self._build_party_member_menu(entity)))
        if entity.is_shop:
            options.append(MenuOption("Shop", self.open_shop_menu))
        options.append(MenuOption("Leave", self.close_menu))

        self.active_menu = DialogueMenu(
            speaker=character.name, prompt="What do you need?", options=options
        )

    # --- party roster ---------------------------------------------------------

    def open_party_menu(self) -> None:
        """Manage anyone in the party — inventory, character sheet, class/
        skill progression — without making them the active/controlled
        character first."""
        members = [self.player] + self.party
        options: List[MenuOption] = []
        for member in members:
            label = member.name
            if member.stats is not None:
                label += f" ({member.stats.hp.current}/{member.stats.hp.max_value} HP)"
            options.append(MenuOption(label, lambda m=member: self._build_party_member_menu(m)))
        prompt = "Manage who?" if options else "No one in the party yet."
        options.append(MenuOption("Leave", self.close_menu))
        self.active_menu = DialogueMenu(speaker="Party", prompt=prompt, options=options)

    def _build_party_member_menu(self, entity: Entity) -> None:
        options = [
            MenuOption("Character Sheet", lambda: self._menu_open_character_sheet(entity)),
            MenuOption("Inventory", lambda: self.open_inventory_menu(entity)),
            MenuOption("Progression & Skills", lambda: self.open_progression_menu(entity)),
        ]
        if entity is not self.player:
            options.append(MenuOption("Set as active character", lambda: self._set_active_character(entity)))
        options.append(MenuOption("Back", self.open_party_menu))
        self.active_menu = DialogueMenu(speaker=entity.name, prompt="What do you need?", options=options)

    def _menu_open_character_sheet(self, entity: Entity) -> None:
        self.close_menu()
        self.open_character_screen(entity)

    def close_menu(self) -> None:
        self.active_menu = None

    # --- shop --------------------------------------------------------------
    # Temporary/testing functionality — expect this to be locked back down
    # (e.g. Cleric's free Revive) once the game is more developed.

    def open_shop_menu(self) -> None:
        options = [
            MenuOption("Buy Items", self._build_shop_items_menu),
            MenuOption("Unlock Classes", self._build_shop_classes_menu),
            MenuOption("Leave", self.close_menu),
        ]
        self.active_menu = DialogueMenu(
            speaker="Shop", prompt=f"You have {self.gold} gold.", options=options
        )

    def _build_shop_items_menu(self) -> None:
        options = [
            MenuOption(
                f"{definition.name} - {definition.price}g",
                lambda item_id=definition.item_id: self._menu_buy_item(item_id),
                description=definition.description,
            )
            for definition in item_data.all_items()
            if definition.price > 0
        ]
        prompt = (
            f"Buy what? (Gold: {self.gold})" if options else "Nothing for sale."
        )
        options.append(MenuOption("Back", self.open_shop_menu))
        self.active_menu = DialogueMenu(speaker="Shop", prompt=prompt, options=options)

    def _menu_buy_item(self, item_id: str) -> None:
        definition = item_data.item(item_id)
        inventory = self.player.inventory
        if self.gold < definition.price:
            self.message_log.add_message("Not enough gold.", color=message_log_module.INFO_COLOR)
        elif not inventory.add(ItemInstance(item_id)):
            self.message_log.add_message("Inventory is full.", color=message_log_module.INFO_COLOR)
        else:
            self.gold -= definition.price
            self.message_log.add_message(
                f"Bought {definition.name}.", color=message_log_module.INFO_COLOR, stack=False
            )
        self._build_shop_items_menu()

    def _build_shop_classes_menu(self) -> None:
        options = [
            MenuOption(
                f"{class_def.name} - {class_def.unlock_cost_gold}g",
                lambda cid=class_def.class_id: self._menu_buy_class(cid),
                description=class_def.description,
            )
            for class_def in skill_data.all_classes()
            if class_def.class_id not in self.unlocked_classes
        ]
        prompt = (
            f"Unlock which class? (Gold: {self.gold})"
            if options
            else "Nothing left to unlock."
        )
        options.append(MenuOption("Back", self.open_shop_menu))
        self.active_menu = DialogueMenu(speaker="Shop", prompt=prompt, options=options)

    def _menu_buy_class(self, class_id: str) -> None:
        class_def = skill_data.class_def(class_id)
        if self.gold < class_def.unlock_cost_gold:
            self.message_log.add_message("Not enough gold.", color=message_log_module.INFO_COLOR)
        else:
            self.gold -= class_def.unlock_cost_gold
            self.unlocked_classes.add(class_id)
            self.message_log.add_message(
                f"Unlocked the {class_def.name} class.", color=message_log_module.INFO_COLOR, stack=False
            )
        self._build_shop_classes_menu()

    # --- pause menu ------------------------------------------------------------

    def open_pause_menu(self) -> None:
        self.pause_menu_open = True
        options = [MenuOption("Resume", self.close_pause_menu)]
        if self.in_dungeon:
            options.append(MenuOption("Give Up (Return to Town)", self._confirm_give_up))
        if self.save_slot is not None:
            options.append(MenuOption("Save Game", self._menu_save_game))
            if self.can_load:
                options.append(
                    MenuOption(
                        "Load Last Save",
                        self._confirm_load_game,
                        description="Rewind to this playthrough's last save.",
                    )
                )
        options.append(
            MenuOption(
                "Return to Main Menu",
                self._request_main_menu,
                description="Saves automatically before leaving.",
            )
        )
        self.active_menu = DialogueMenu(
            speaker="Paused", prompt="Press Q to save and quit the game.", options=options
        )

    def close_pause_menu(self) -> None:
        self.pause_menu_open = False
        self.active_menu = None

    def _confirm_give_up(self) -> None:
        self.active_menu = DialogueMenu(
            speaker="Paused",
            prompt="Give up and return to town? This ends your run.",
            options=[
                MenuOption("Yes, give up", self._do_give_up),
                MenuOption("No", self.open_pause_menu),
            ],
        )

    def _do_give_up(self) -> None:
        self.close_pause_menu()
        if self.in_dungeon:
            self.end_run(defeated=False)

    # --- save / load -----------------------------------------------------------
    # One slot per playthrough. Loading is a mid-run rewind only: in town it's
    # disabled, and every return to town autosaves, so a lost run can't be
    # undone (see end_run).

    @property
    def can_load(self) -> bool:
        return self.in_dungeon and self.save_slot is not None and save_module.slot_exists(self.save_slot)

    def save_game(self) -> None:
        if self.save_slot is None:
            return
        save_module.write_slot(save_module.serialize_engine(self), self.save_slot)

    def load_game(self) -> None:
        save_module.apply_engine_state(self, save_module.read_slot(self.save_slot))

    def _menu_save_game(self) -> None:
        self.save_game()
        self.close_pause_menu()
        self.message_log.add_message("Game saved.", color=message_log_module.INFO_COLOR, stack=False)

    def _confirm_load_game(self) -> None:
        self.active_menu = DialogueMenu(
            speaker="Paused",
            prompt="Load your last save? Progress since then is lost.",
            options=[
                MenuOption("Yes, load", self._menu_load_game),
                MenuOption("No", self.open_pause_menu),
            ],
        )

    def _menu_load_game(self) -> None:
        self.load_game()
        self.close_pause_menu()
        self.message_log.add_message("Game loaded.", color=message_log_module.INFO_COLOR, stack=False)

    def _request_main_menu(self) -> None:
        self.close_pause_menu()
        self.return_to_title = True

    # --- character sheet -----------------------------------------------------

    def open_character_screen(self, entity: Optional[Entity] = None) -> None:
        self.character_screen_entity = entity or self.player
        self.character_screen_open = True
        self.character_sheet_cursor = 0

    def close_character_screen(self) -> None:
        self.character_screen_open = False

    def _character_sheet_selectable(self) -> List["inspect_text.SheetRow"]:
        return [row for row in inspect_text.character_sheet_rows(self.character_screen_entity) if row.selectable]

    def move_character_sheet_cursor(self, delta: int) -> None:
        selectable = self._character_sheet_selectable()
        if selectable:
            self.character_sheet_cursor = (self.character_sheet_cursor + delta) % len(selectable)

    def inspect_character_sheet_entry(self) -> None:
        selectable = self._character_sheet_selectable()
        if not selectable:
            return
        row = selectable[self.character_sheet_cursor % len(selectable)]
        if row.inspect is not None:
            self.open_inspect(row.text, row.inspect())

    # --- message log --------------------------------------------------------

    def toggle_message_log(self) -> None:
        self.message_log_expanded = not self.message_log_expanded
        self.log_scroll_offset = 0

    def close_message_log(self) -> None:
        self.message_log_expanded = False

    def scroll_message_log(self, delta: int) -> None:
        self.log_scroll_offset = max(0, self.log_scroll_offset + delta)

    # --- inspect overlay ------------------------------------------------------

    def open_inspect(self, title: str, lines: List[str]) -> None:
        self.inspect_title = title
        self.inspect_lines = lines
        self.inspect_open = True

    def close_inspect(self) -> None:
        self.inspect_open = False

    def _menu_add_to_party(self, entity: Entity) -> None:
        self.add_to_party(entity)
        self.close_menu()

    # --- progression menu (nested DialogueMenus) -----------------------------

    def open_progression_menu(self, entity: Optional[Entity] = None) -> None:
        self._build_progression_root(entity or self.player)

    def _build_progression_root(self, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        progression = entity.progression
        options: List[MenuOption] = []
        if progression is not None:
            options.append(MenuOption("Add a class", lambda: self._build_add_class_menu(entity)))
            options.append(MenuOption("Class skill trees", lambda: self._build_tree_source_menu(entity)))
            if progression.innate_id is not None:
                # Shown regardless of whether this innate is manually-claimed
                # (Reimu/Marisa) or a plain auto-claimed river (Kisume/Yamame/
                # Parsee) — the auto-claimed ones used to have no menu entry
                # at all, so there was no way to even see their tree/points
                # once it started carrying real per-node costs. Harmless for
                # an auto tree: everything reachable is already claimed by
                # the time you open it, so this is read-only in practice.
                source = f"innate:{progression.innate_id}"
                innate_points = progression.points_available(source)
                options.append(
                    MenuOption(
                        f"Innate skill tree - {innate_points} points",
                        lambda: self._build_tree_node_menu(source, entity),
                    )
                )
            options.append(MenuOption("Skill Upgrades", lambda: self._build_skill_upgrade_source_menu(entity)))
            species = skill_data.species_def(progression.species_id).name
            prompt = (
                f"Lv.{progression.species_level} {species}  "
                f"({len(progression.classes)}/{progression.max_class_slots} class slots)"
            )
        else:
            prompt = "No progression to manage."
        options.append(MenuOption("Leave", self.close_menu))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=prompt, options=options
        )

    def _build_add_class_menu(self, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        progression = entity.progression
        options: List[MenuOption] = []
        for class_def in skill_data.all_classes():
            if progression.can_add_class(class_def.class_id, self.unlocked_classes):
                options.append(
                    MenuOption(
                        class_def.name,
                        lambda cid=class_def.class_id: self._menu_add_class(cid, entity),
                        description=class_def.description,
                        inspect=lambda cd=class_def: inspect_text.class_lines(cd),
                    )
                )
        prompt = "Take up which class?" if options else "No classes available (slots full or locked)."
        options.append(MenuOption("Back", lambda: self._build_progression_root(entity)))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=prompt, options=options
        )

    def _menu_add_class(self, class_id: str, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        self.add_class_to(entity, class_id)
        self._build_progression_root(entity)

    def _build_tree_source_menu(self, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        progression = entity.progression
        options: List[MenuOption] = []
        for progress in progression.classes:
            source = f"class:{progress.class_id}"
            points = progression.points_available(source)
            label = f"{progress.definition.name} (Lv.{progress.level}) - {points} points"
            options.append(
                MenuOption(
                    label,
                    lambda s=source: self._build_tree_node_menu(s, entity),
                    description=progress.definition.description,
                    inspect=lambda cd=progress.definition: inspect_text.class_lines(cd),
                )
            )
        prompt = "Advance which class tree?" if options else "No classes yet — add one first."
        options.append(MenuOption("Back", lambda: self._build_progression_root(entity)))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=prompt, options=options
        )

    def _build_tree_node_menu(self, source: str, entity: Optional[Entity] = None) -> None:
        """Every node in the tree is listed (not just currently-claimable
        ones) so the whole tree's shape is visible up front: already-learned
        nodes are dimmed green, locked ones (level/prereqs/excludes unmet)
        are dark grey, reachable-but-can't-afford-yet ones are red, and
        anything actually claimable right now is left at the plain default
        color. Selecting a non-claimable node just re-opens this same menu
        (claim_tree_node no-ops on failure), no separate lockout needed."""
        entity = entity or self.player
        progression = entity.progression
        tree, level, claimed, points_per_level = progression.tree_state(source)
        available = {n.node_id for n in progression.available_nodes(source)}
        points = progression.points_available(source)

        options = []
        if tree is not None:
            for node in tree.nodes.values():
                label = f"{node.name} (req Lv.{node.requires_level}, cost {node.cost})"
                if node.node_id in claimed:
                    label += " [learned]"
                    color = (90, 160, 90)
                elif node.node_id not in available:
                    color = (100, 100, 100)
                elif node.cost > points:
                    color = (200, 70, 70)
                else:
                    color = None
                options.append(
                    MenuOption(
                        label,
                        lambda n=node.node_id: self._menu_claim_node(source, n, entity),
                        description=node.description,
                        inspect=lambda n=node: inspect_text.tree_node_lines(n),
                        color=color,
                    )
                )
        prompt = f"Learn which node? (Points: {points})" if options else "No tree here."
        # Innate trees have no source-selection submenu (each character has
        # only one), so "Back" from here goes straight to the progression
        # root instead of the class-tree source list.
        back = (
            (lambda: self._build_progression_root(entity))
            if source.startswith("innate:")
            else (lambda: self._build_tree_source_menu(entity))
        )
        options.append(MenuOption("Back", back))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=prompt, options=options
        )

    def _menu_claim_node(self, source: str, node_id: str, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        self.claim_tree_node(entity, source, node_id)
        self._build_tree_node_menu(source, entity)  # refresh remaining nodes

    def _build_skill_upgrade_source_menu(self, entity: Optional[Entity] = None) -> None:
        """Every learned skill that has its own self-upgrade tree — separate
        from the class/species/innate tree that granted the skill."""
        entity = entity or self.player
        book = entity.skill_book
        options: List[MenuOption] = []
        if book is not None:
            for instance in sorted(book.skills.values(), key=lambda i: i.resolved.name):
                if instance.resolved.tree is None:
                    continue
                available = len(instance.available_tree_nodes())
                label = f"{instance.resolved.name} (Lv.{instance.level})"
                if available:
                    label += f" - {available} available"
                options.append(
                    MenuOption(label, lambda i=instance: self._build_skill_upgrade_node_menu(i, entity))
                )
        prompt = "Upgrade which skill?" if options else "No skills with their own upgrades yet."
        options.append(MenuOption("Back", lambda: self._build_progression_root(entity)))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=prompt, options=options
        )

    def _build_skill_upgrade_node_menu(self, instance: SkillInstance, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        nodes = instance.available_tree_nodes()
        options = [
            MenuOption(
                f"{node.name} (req Lv.{node.requires_level})",
                lambda n=node.node_id: self._menu_claim_skill_upgrade(instance, n, entity),
                description=node.description,
                inspect=lambda n=node: inspect_text.tree_node_lines(n),
            )
            for node in nodes
        ]
        prompt = "Learn which upgrade?" if nodes else "Nothing available at this level yet."
        options.append(MenuOption("Back", lambda: self._build_skill_upgrade_source_menu(entity)))
        self.active_menu = DialogueMenu(
            speaker=entity.name, prompt=prompt, options=options
        )

    def _menu_claim_skill_upgrade(self, instance: SkillInstance, node_id: str, entity: Optional[Entity] = None) -> None:
        entity = entity or self.player
        self.claim_skill_upgrade(entity, instance, node_id)
        self._build_skill_upgrade_node_menu(instance, entity)  # refresh remaining nodes

    def _save_current_floor(self) -> None:
        self.floor_history[self.floor_depth] = (
            self.game_map, self.entities, (self.player.x, self.player.y)
        )

    def _sync_party_entities(self, entities: List[Entity]) -> List[Entity]:
        """A restored floor snapshot can be stale about party composition —
        recruiting mid-run (or swapping active character) changes who's in
        `party` after that floor was last saved. Patch in anyone missing
        (at the player's restored position, since we don't know where
        they'd have stood on a floor they weren't part of yet)."""
        missing = [e for e in [self.player] + self.party if e not in entities]
        for member in missing:
            member.x, member.y = self.player.x, self.player.y
        return entities + missing if missing else entities

    def _reset_turn_energy(self) -> None:
        """Zero every current entity's turn-queue energy. Called on every
        floor entry (fresh generation or restoration from `floor_history`).

        Without this, entering a floor could hand out a burst of enemy
        turns before the player's own comes up: a freshly generated floor's
        monsters start at 0 energy but the player carries over whatever
        they had left from the previous floor (using stairs never spends a
        turn), and a *restored* floor's entities carry back whatever energy
        they had frozen at the exact moment that floor was last saved —
        either way, nothing guarantees the player and the floor's entities
        start from a comparable baseline otherwise.
        """
        for entity in self.entities:
            entity.energy = 0.0

    def _place_player_party(self, start: Tuple[int, int]) -> None:
        self.player.x, self.player.y = start
        for member in self.party:
            member.x, member.y = start

    def _place_gap(self, game_map: GameMap, pos: Tuple[int, int]) -> None:
        game_map.stairs_down = None
        game_map.gap = pos
        game_map.tiles[pos] = tile_types.gap

    def try_descend(self) -> None:
        """The descend key: an active gap opens the location-choice menu;
        otherwise it's a normal floor descent."""
        if self.game_map.gap is not None and self.game_map.gap == (self.player.x, self.player.y):
            self.open_gap_menu()
            return
        self.try_use_stairs_down()

    def try_use_stairs_down(self) -> None:
        if self.game_map.stairs_down != (self.player.x, self.player.y):
            return

        self.focus_target = None
        self._save_current_floor()
        self.floor_depth += 1

        if self.floor_depth < len(self.floor_history):
            # Already generated (we're returning to a floor left via stairs
            # up) — restore it exactly, don't regenerate.
            self.game_map, self.entities, (self.player.x, self.player.y) = (
                self.floor_history[self.floor_depth]
            )
            self.entities = self._sync_party_entities(self.entities)
        elif self.current_location is None:
            self._generate_tutorial_floor()
        else:
            self._generate_location_floor()

        self._reset_turn_energy()
        self.update_camera()
        self.update_fov()
        self.message_log.add_message("You descend the stairs.")

    def _generate_dungeon_floor(self):
        return generate_dungeon(
            map_width=self.game_map.width,
            map_height=self.game_map.height,
            max_rooms=20,
            room_min_size=6,
            room_max_size=10,
        )

    def _generate_location_dungeon_floor(self, location: "location_data.LocationDef"):
        """Non-boss floor generation for a location, picking the layout by
        `location.generator` — plain dungeon by default, or a cave/tunnels
        variant for locations that opt in."""
        map_width = location.map_width or self.game_map.width
        map_height = location.map_height or self.game_map.height
        if location.generator == "cave":
            # num_spawns 12 -> 18 (+50%): enemy count AND item count both
            # derive directly from this same candidate list (one monster
            # per point, each also independently rolling for loot), so
            # bumping it is a single lever for both asks at once.
            # num_rooms scaled with map area vs. the default 80x54 canvas so
            # bigger maps (e.g. Fantastic Blowhole's 130x90) still get a
            # proportionally full cavern instead of the same 5 blobs lost in
            # more empty space.
            default_area = VIEWPORT_WIDTH * VIEWPORT_HEIGHT
            num_rooms = max(5, round(5 * (map_width * map_height) / default_area))
            return generate_cave(map_width, map_height, num_spawns=18, num_rooms=num_rooms)
        if location.generator == "tunnels":
            # Same area-scaling idea as the cave branch above — a bigger
            # canvas needs proportionally more tunnel segments, or the
            # snake just wanders through a fraction of the available space
            # and everything else sits empty.
            default_area = VIEWPORT_WIDTH * VIEWPORT_HEIGHT
            area_ratio = (map_width * map_height) / default_area
            min_segments = max(6, round(6 * area_ratio))
            max_segments = max(9, round(9 * area_ratio))
            return generate_tunnels(
                map_width, map_height, num_spawns=18,
                min_segments=min_segments, max_segments=max_segments,
            )
        return generate_dungeon(
            map_width=map_width,
            map_height=map_height,
            max_rooms=20,
            room_min_size=6,
            room_max_size=10,
        )

    def _generate_tutorial_floor(self) -> None:
        """Floors 1-5: no location assigned yet, the plain default dungeon.
        The 5th floor's stairs_down becomes a gap instead of continuing —
        that's where the player picks their first location."""
        new_map, start, enemy_spawns = self._generate_dungeon_floor()
        self._place_player_party(start)

        monsters = bestiary.populate_dungeon_floor(enemy_spawns, level=self.floor_depth)
        for monster in monsters:
            self._finalize_monster_progression(monster)
        loot = bestiary.scatter_room_loot(enemy_spawns)

        if self.floor_depth >= self.TUTORIAL_FLOOR_COUNT:
            self._place_gap(new_map, new_map.stairs_down)

        self.game_map = new_map
        self.update_camera()
        self.entities = [self.player] + self.party + monsters + loot
        self.floor_history.append((self.game_map, self.entities, (self.player.x, self.player.y)))

    def _generate_location_floor(self) -> None:
        """A location's own floor: its own enemy roster, and on the final
        floor (count chosen when the location was picked), a dedicated boss
        arena instead of a normal generated floor — no fodder monsters, no
        stairs down (only the gap, revealed once the boss falls)."""
        location = location_data.location(self.current_location)
        self.location_floor_index += 1
        is_boss_floor = self.location_floor_index >= self.location_floor_target

        if is_boss_floor:
            arena_fn = {"open": generate_boss_arena, "bridge": generate_bridge_arena}.get(
                location.boss_arena, generate_boss_arena
            )
            new_map, start, boss_pos = arena_fn(
                location.map_width or self.game_map.width,
                location.map_height or self.game_map.height,
            )
            self._place_player_party(start)

            boss_id = random.choice(location.boss_ids) if location.boss_ids else location.boss_id
            boss_def = location_data.boss(boss_id)
            boss = bestiary.spawn_boss(boss_def, *boss_pos)
            self._finalize_boss_progression(boss, boss_def)
            monsters = [boss]
            self.pending_boss = boss
            self.pending_boss_id = boss_def.boss_id
            # Encountering a boss reveals their identity for future gap
            # menus regardless of how the fight goes — losing still counts.
            already_known = boss_def.boss_id in self.knowledge["bosses"]
            self.knowledge["bosses"].add(boss_def.boss_id)
            name = boss_def.name if already_known else "A powerful presence"
            self.message_log.add_message(
                f"{name} blocks your path!", color=message_log_module.DEATH_COLOR, stack=False
            )
        else:
            new_map, start, enemy_spawns = self._generate_location_dungeon_floor(location)
            self._place_player_party(start)
            monsters = bestiary.populate_location_floor(
                location, enemy_spawns,
                current_depth=self.floor_depth,
                min_depth=self.location_entry_depth,
                max_depth=self.location_entry_depth + self.location_floor_target - 1,
            )
            for monster in monsters:
                self._finalize_monster_progression(monster)
            monsters = monsters + bestiary.scatter_room_loot(enemy_spawns)

        self.game_map = new_map
        self.update_camera()
        self.entities = [self.player] + self.party + monsters
        self.floor_history.append((self.game_map, self.entities, (self.player.x, self.player.y)))

    def _finalize_monster_progression(self, monster: Entity) -> None:
        """Monsters go through the same species/skill machinery as playable
        characters. Goblins additionally pick up the Warrior class, to
        exercise the class system on an enemy too (bosses will lean on this
        more; trash mobs mostly just need their species tree)."""
        self.auto_claim_linear_trees(monster)
        self.reapply_species_stat_growth(monster)
        if monster.progression is not None and monster.progression.species_id == "goblin":
            self.add_class_to(monster, "warrior")
        self._level_up_monster_skills_and_classes(monster)

    def _level_up_monster_skills_and_classes(self, monster: Entity) -> None:
        """Floor-scaled enemies otherwise only grow via their species tree
        — every skill instance they're granted (species/innate/class) stays
        at level 1, and a classed monster (Goblins' Warrior) stays at class
        level 1, forever: a monster almost never survives long enough to
        accrue meaningful in-combat exp before dying, so none of that
        normal growth ever actually happens. This directly sets skill/class
        levels to match species level (capped at each one's own max) and
        randomly climbs whatever tree each has — including picking a random
        branch on a fork like Warrior's Aggression vs. Endurance — so kit
        strength actually keeps pace with floor depth instead of flatlining
        at whatever a level-1 kit happens to do."""
        if monster.progression is None:
            return
        species_level = monster.progression.species_level

        # Class tree first: this is what actually grants a classed
        # monster's active/passive skills (e.g. Goblin's Warrior kit) into
        # its skill_book in the first place — leveling skill instances
        # first would miss every one of them, since they wouldn't exist
        # yet.
        for progress in monster.progression.classes:
            target_level = min(species_level, progress.definition.max_level)
            if target_level > progress.level:
                progress.level = target_level
            source = f"class:{progress.class_id}"
            self._auto_claim_randomized(
                lambda s=source: monster.progression.available_nodes(s),
                lambda s=source: monster.progression.points_available(s),
                lambda node_id, s=source: self.claim_tree_node(monster, s, node_id),
            )

        if monster.skill_book is not None:
            for instance in list(monster.skill_book.skills.values()):
                target_level = min(species_level, instance.resolved.max_level)
                if target_level > instance.level:
                    instance.set_level(target_level)
                    if instance.category == CATEGORY_PASSIVE:
                        self._reapply_passive(monster, instance)
                self._auto_claim_randomized(
                    instance.available_tree_nodes,
                    lambda i=instance: i.points_available,
                    instance.claim_tree_node,
                )
                if instance.category == CATEGORY_PASSIVE:
                    self._reapply_passive(monster, instance)

    def _auto_claim_randomized(self, available_fn, points_fn, claim_fn) -> None:
        """Repeatedly claim a random currently-affordable node, until
        nothing affordable is left — the "give a level-scaled monster/class
        a plausible loadout" primitive `_level_up_monster_skills_and_classes`
        uses for both a skill's own upgrade tree and a class tree. Random
        instead of "claim everything reachable" specifically so a branching
        tree with mutually-exclusive forks (excludes) gets a genuinely
        random pick per spawn instead of always favoring whichever branch
        happens to be listed/iterated first."""
        while True:
            nodes = [n for n in available_fn() if n.cost <= points_fn()]
            if not nodes:
                return
            node = random.choice(nodes)
            if not claim_fn(node.node_id):
                return  # shouldn't happen given the cost filter, but stop safely

    def _finalize_boss_progression(self, boss: Entity, boss_def: "location_data.BossDef") -> None:
        """Scale a freshly-spawned boss up to a real fight-worthy character
        instead of a hand-tuned fixed statblock — the same underlying growth
        every level-scaled monster now gets (species tree, attribute
        investment, skill/class leveling), plus a boss-only toughness
        multiplier on top since she still needs to be a credible solo
        threat against a multi-person party.

        `boss_def.base_hp/mp/sp` (set in bestiary.spawn_boss) are just her
        starting baseline now, same idea as a player's starting stats —
        everything else stacks on top here. No revert bookkeeping needed
        for any of this: a boss fight Entity is one-off, discarded on
        defeat (recruiting resets her to a completely fresh baseline
        anyway, see handle_boss_defeated)."""
        if boss.progression is None or boss.stats is None:
            return

        fight_level = max(self.floor_depth, boss_def.min_depth)
        boss.progression.species_level = fight_level
        self.auto_claim_linear_trees(boss)
        self.reapply_species_stat_growth(boss)

        for attr_name, amount in boss_def.attributes.items():
            apply_stat_modifier(boss.stats, attr_name, amount)

        extra_hp = int(boss.stats.hp.max_value * (boss_def.hp_multiplier - 1))
        if extra_hp > 0:
            apply_stat_modifier(boss.stats, "hp", extra_hp)

        self._level_up_monster_skills_and_classes(boss)

    def handle_boss_defeated(self, boss_entity: Entity) -> None:
        """First-time defeat permanently recruits the boss (same rules as
        any Character-bearing NPC) and reveals a gap where they fell. A
        future re-encounter of the same location's boss is still a fresh
        clone Entity (naturally, since a new one is spawned each time the
        location's final floor is generated) — but it's discarded rather
        than recruited again if `boss_id` is already in the roster, tracked
        independently of the party/floor state via
        `knowledge["recruited_bosses"]` so it survives town resets/saves.

        Recruiting (the first time) resets them to a normal party-member
        baseline — their boss-fight statblock and boss-only skills (base
        kit + spellcards) were for the fight only and are discarded
        entirely, not kept."""
        boss_def = location_data.boss(self.pending_boss_id) if self.pending_boss_id else None
        # Anything the boss summoned (e.g. Parsee's clone) doesn't outlive
        # her — otherwise it's left standing, still hostile, with no way to
        # end the fight.
        for clone in [e for e in self.entities if e.summoned_by is boss_entity]:
            self.remove_entity(clone)
        self.remove_entity(boss_entity)

        # Gold on every defeat, first-time recruit or a later re-encounter
        # kill alike — recruiting is the headline reward, but a boss fight
        # shouldn't otherwise be worth less gold than the fodder along the way.
        if boss_def is not None:
            gold_min, gold_max = boss_def.loot_gold_range
            if gold_max > 0:
                amount = random.randint(gold_min, gold_max)
                if amount > 0:
                    self.gold += amount
                    self.message_log.add_message(f"Found {amount} gold.", color=message_log_module.INFO_COLOR)

        if boss_def is not None and boss_def.boss_id in self.knowledge["recruited_bosses"]:
            self._place_gap(self.game_map, (boss_entity.x, boss_entity.y))
            self.pending_boss = None
            self.pending_boss_id = None
            self.message_log.add_message(
                f"{boss_entity.name} is defeated again, but you've already recruited them.",
                color=message_log_module.DEATH_COLOR, stack=False,
            )
            return

        if boss_entity.character is not None:
            boss_entity.character.recruited = True

        # Recruited bosses read as playable characters visually (same
        # glyph Reimu/Marisa use) while keeping their own boss color as an
        # identifying tint, instead of staying the generic boss glyph.
        boss_entity.char = "@"

        if boss_def is not None and boss_entity.stats is not None:
            boss_entity.stats = StatBlock.create(
                max_hp=boss_def.recruit_hp, max_mp=boss_def.recruit_mp, max_sp=boss_def.recruit_sp,
                attributes=Attributes(),
            )
            # The fight's own species growth (see _finalize_boss_progression)
            # was tracked against the StatBlock just discarded above — left
            # in place, reapply_species_stat_growth below would compute a
            # net delta against stale numbers that don't apply to this brand
            # new object at all (usually landing on a no-op net of 0, so the
            # fresh StatBlock never actually gets its growth applied).
            if boss_entity.progression is not None:
                boss_entity.progression.species_stat_growth = []
        boss_entity.skill_book = SkillBook()

        if boss_entity.progression is not None:
            # Recruited mid-run: starts at the party's current species level
            # (no free class levels though — those are earned like anyone else).
            levels = [
                e.progression.species_level for e in [self.player] + self.party if e.progression is not None
            ]
            boss_entity.progression.reset_all()
            boss_entity.progression.species_level = max(levels) if levels else 1
            self.auto_claim_linear_trees(boss_entity)
            self.reapply_species_stat_growth(boss_entity)

        if boss_def is not None:
            self.knowledge["recruited_bosses"].add(boss_def.boss_id)

        self._place_gap(self.game_map, (boss_entity.x, boss_entity.y))
        self.pending_boss = None
        self.pending_boss_id = None
        self.message_log.add_message(
            f"{boss_entity.name} is defeated! A gap opens in the dungeon...",
            color=message_log_module.DEATH_COLOR, stack=False,
        )
        self.add_to_party(boss_entity)

    # --- gap / location choice -------------------------------------------------

    def open_gap_menu(self) -> None:
        choices = self._roll_location_choices()
        options = [
            MenuOption(
                f"{loc.name} ({self._boss_label(loc)})",
                lambda l=loc: self._choose_location(l),
                description=loc.description,
            )
            for loc in choices
        ]
        self.active_menu = DialogueMenu(
            speaker="???", prompt="A gap opens. Where will you go?", options=options
        )

    def _boss_label(self, location: "location_data.LocationDef") -> str:
        boss_ids = location.boss_ids or [location.boss_id]
        known = [bid for bid in boss_ids if bid in self.knowledge["bosses"]]
        if not known:
            return "???"
        label = " / ".join(location_data.boss(bid).name for bid in known)
        if len(known) < len(boss_ids):
            label += " / ???"
        return label

    def _roll_location_choices(self, count: int = 2) -> List["location_data.LocationDef"]:
        pool = [loc for loc in location_data.all_locations() if loc.min_depth <= self.floor_depth]
        if not pool:
            pool = location_data.all_locations()
        return random.sample(pool, min(count, len(pool)))

    def _choose_location(self, location: "location_data.LocationDef") -> None:
        self.close_menu()
        self.knowledge["locations"].add(location.location_id)
        self.current_location = location.location_id
        self.location_floor_index = 0
        self.location_floor_target = random.randint(*location.floor_range)

        self._save_current_floor()
        self.floor_depth += 1
        self.location_entry_depth = self.floor_depth
        self._generate_location_floor()
        self._reset_turn_energy()
        self.update_fov()
        self.message_log.add_message(
            f"You step through the gap into {location.name}.",
            color=message_log_module.INFO_COLOR, stack=False,
        )

    def try_use_stairs_up(self) -> None:
        if self.game_map.stairs_up != (self.player.x, self.player.y):
            return
        if self.floor_depth <= 0:
            return
        if self.current_location is not None and self.floor_depth <= self.location_entry_depth:
            self.message_log.add_message(
                "The way back has sealed shut.", color=message_log_module.INFO_COLOR
            )
            return

        self.focus_target = None
        self._save_current_floor()
        self.floor_depth -= 1
        self.game_map, self.entities, (self.player.x, self.player.y) = (
            self.floor_history[self.floor_depth]
        )
        self.entities = self._sync_party_entities(self.entities)
        self._reset_turn_energy()
        self.update_camera()
        self.update_fov()

        if self.floor_depth == 0:
            self.end_run(defeated=False)  # voluntarily leaving the dungeon ends the run
        else:
            self.message_log.add_message("You ascend the stairs.")

    def _set_active_character(self, entity: Entity) -> None:
        previous = self.player
        # In town, picking someone from OUTSIDE the current party (not a
        # mid-dungeon reshuffle among existing party members) is a full
        # swap: `entity` becomes the new solo leader and `previous` is left
        # out of the party entirely, rather than being auto-added — that
        # auto-add used to make a true solo run impossible (switching to
        # Marisa always dragged Reimu along as a forced 2-person party;
        # the only workaround was recruiting Marisa, entering the dungeon,
        # then giving up so the party actually became solo).
        entity_already_in_party = entity in self.party
        keep_previous_in_party = entity_already_in_party or self.in_dungeon

        self.player = entity
        if entity.inventory is None:
            entity.inventory = Inventory()

        # The newly controlled character is played directly, not by AI.
        if entity in self.party:
            self.party.remove(entity)
        entity.ai = None

        # The character stepping down keeps acting instead of freezing —
        # but only when this was a reshuffle within the existing party (or
        # we're mid-dungeon), not a town swap to someone outside it.
        if keep_previous_in_party and previous is not entity and previous not in self.party:
            self.add_to_party(previous)

        self.active_menu = None
        self.update_fov()

    def toggle_debug_log_all_progression(self) -> None:
        """F1: bypass should_log_progression's player/party-only filter, so
        every monster's own leveling/skill-learn notices print too — for
        verifying the underlying machinery actually runs on non-party
        entities, without permanently spamming the log during normal play."""
        self.debug_log_all_progression = not self.debug_log_all_progression
        state = "ON" if self.debug_log_all_progression else "OFF"
        self.message_log.add_message(
            f"[Debug] Log all progression: {state}", color=message_log_module.INFO_COLOR
        )

    def should_log_progression(self, entity: Optional[Entity]) -> bool:
        """Whether a level-up/skill-learned notice for `entity` should
        actually be printed — the player and active party only, by
        default. Every generated monster runs through the exact same
        auto_claim_linear_trees/gain_exp/reapply_passive machinery
        (species tree claims, passive exp ticks, trigger exp) as any
        playable character, so without this every monster spawned/leveled
        on the current floor spammed the log with its own "Learned X" /
        "Reached level N!" lines. `debug_log_all_progression` (toggled via
        the debug menu) bypasses the filter for actually verifying this
        machinery works on non-party entities."""
        if self.debug_log_all_progression:
            return True
        return entity is not None and (entity is self.player or entity in self.party)

    def update_camera(self) -> None:
        """Re-center the viewport on the player, clamped so it never scrolls
        past the map edge. A no-op whenever the map fits inside the fixed
        console viewport (every map until Fantastic Blowhole's cave)."""
        max_x = max(0, self.game_map.width - VIEWPORT_WIDTH)
        max_y = max(0, self.game_map.height - VIEWPORT_HEIGHT)
        self.camera_x = max(0, min(max_x, self.player.x - VIEWPORT_WIDTH // 2))
        self.camera_y = max(0, min(max_y, self.player.y - VIEWPORT_HEIGHT // 2))

    def update_fov(self) -> None:
        other_entities = [e for e in self.entities if e is not self.player]
        self.game_map.visible = fov.compute_visibility(
            self.game_map,
            self.player,
            self.game_map.ambient_light_radius,
            other_entities,
        )

    def render(self, console: Console, context: Context) -> None:
        self.game_map.render(console, self.camera_x, self.camera_y)
        cam_x, cam_y = self.camera_x, self.camera_y

        def in_viewport(x: int, y: int) -> bool:
            return 0 <= x - cam_x < VIEWPORT_WIDTH and 0 <= y - cam_y < VIEWPORT_HEIGHT

        # Highlight the targeting footprint under the entities, so entity
        # glyphs still draw on top of the tinted tiles.
        if self.target_selector is not None:
            in_range, tiles = self.target_selector.preview(self)
            tint = (50, 70, 120) if in_range else (120, 40, 40)
            for tx, ty in tiles:
                if self.game_map.in_bounds(tx, ty) and in_viewport(tx, ty):
                    console.rgb["bg"][tx - cam_x, ty - cam_y] = tint

        # Focus-fire mark: a distinct background tint under the target so
        # it's clear at a glance who the party is prioritizing.
        if (
            self.focus_target is not None
            and self.game_map.visible[self.focus_target.x, self.focus_target.y]
            and in_viewport(self.focus_target.x, self.focus_target.y)
        ):
            console.rgb["bg"][self.focus_target.x - cam_x, self.focus_target.y - cam_y] = (90, 20, 20)

        # Ground items draw first so any creature/player standing on the same
        # tile is drawn on top of the item glyph, not hidden underneath it.
        for entity in sorted(self.entities, key=lambda e: e.ground_item is None):
            if self.game_map.visible[entity.x, entity.y] and in_viewport(entity.x, entity.y):
                console.print(entity.x - cam_x, entity.y - cam_y, entity.char, fg=entity.color)

        if self.target_selector is not None:
            if in_viewport(self.target_selector.x, self.target_selector.y):
                console.rgb["bg"][self.target_selector.x - cam_x, self.target_selector.y - cam_y] = (200, 200, 80)
            console.print(
                0, 0,
                "Target: move to aim, Enter to confirm, Esc to cancel",
                fg=(255, 255, 0),
            )

        self.message_log.render(
            console,
            x=0,
            y=VIEWPORT_HEIGHT,
            width=console.width,
            height=ui.MESSAGE_LOG_HEIGHT,
        )
        if self.current_location is not None:
            location_name = location_data.location(self.current_location).name
            floor_label = f"{location_name} - Floor {self.floor_depth}"
        else:
            floor_label = f"Floor {self.floor_depth}" if self.floor_depth > 0 else "Town"
        ui.render_hud(
            console, self.player, hud_y=VIEWPORT_HEIGHT + ui.MESSAGE_LOG_HEIGHT,
            floor_label=floor_label, party=tuple(self.party),
        )

        if self.character_screen_open:
            ui.render_character_sheet(console, self.character_screen_entity, self.character_sheet_cursor)
        elif self.active_menu is not None:
            self.active_menu.render(console, bottom_margin=console.height - VIEWPORT_HEIGHT)

        if self.message_log_expanded:
            x0, y0 = 2, 1
            self.log_scroll_offset = self.message_log.render_expanded(
                console, x0, y0, console.width - 4, console.height - 2, self.log_scroll_offset
            )

        if self.inspect_open:
            ui.render_inspect(console, self.inspect_title, self.inspect_lines)

        context.present(console)
        console.clear()
