from __future__ import annotations

from typing import Optional, Tuple

from components.character import Character
from components.inventory import Inventory, ItemInstance
from components.progression import Progression
from components.skills import SkillBook
from components.stats import StatBlock


class Entity:
    """A generic object on the map: player, enemy, item, etc."""

    def __init__(
        self,
        x: int,
        y: int,
        char: str,
        color: Tuple[int, int, int],
        name: str = "<Unnamed>",
        blocks_movement: bool = False,
        light_radius: Optional[int] = None,
        sees_through_darkness: bool = False,
        character: Optional[Character] = None,
        stats: Optional[StatBlock] = None,
        progression: Optional[Progression] = None,
        skill_book: Optional[SkillBook] = None,
        inventory: Optional[Inventory] = None,
        ground_item: Optional[ItemInstance] = None,
        ai: Optional[object] = None,
        is_shop: bool = False,
        loot: Optional[object] = None,
    ):
        self.x = x
        self.y = y
        self.char = char
        self.color = color
        self.name = name
        self.blocks_movement = blocks_movement
        # Radius of light this entity carries/emits, or None if it emits none.
        self.light_radius = light_radius
        # If True, darkness sources never block this entity's vision.
        self.sees_through_darkness = sees_through_darkness
        # Present if this entity is a playable/recruitable character.
        self.character = character
        # Present if this entity has HP/MP/SP/attributes (NPCs, enemies,
        # and playable characters alike; independent of `character`).
        self.stats = stats
        # Growth state (species/class levels, tree progress) and learned
        # skills. Progression implies a skill_book; enemies may have a
        # skill_book with no progression.
        self.progression = progression
        self.skill_book = skill_book if skill_book is not None else (SkillBook() if progression else None)
        # Held/equipped items; independent of stats/progression (a chest full
        # of loot has no stats, a player has both).
        self.inventory = inventory
        # Present if this entity is an item sitting on the floor (non-blocking,
        # no stats/character), picked up by walking onto it.
        self.ground_item = ground_item
        # Duck-typed controller with `.decide(engine, entity) -> Action`, one
        # of FSMController/BehaviorTreeController/UtilityAIController. None
        # for the player and for idle/undirected entities (they just wait).
        self.ai = ai
        # True for a shopkeeper NPC — adds a "Shop" option to the
        # interaction menu regardless of recruited state.
        self.is_shop = is_shop
        # bestiary.LootTable: gold range + item drop chances, rolled on
        # death by whoever killed this entity (monsters only).
        self.loot = loot
        # Accrued turn-queue energy; only meaningful for entities that act.
        self.energy: float = 0.0
        # Set while mid-channel on a charge_turns skill (game.casting.ChargeState),
        # None otherwise. See game/casting.py for the full lifecycle.
        self.charging: Optional[object] = None
        # Set on a summoned clone (game/effects.py's summon_clone) to the
        # entity that summoned it, so its summoner's own death/defeat can
        # clean it up too. None for everyone else.
        self.summoned_by: Optional["Entity"] = None

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
