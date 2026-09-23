from __future__ import annotations

from typing import Optional, Tuple

import tcod.event
from tcod.console import Console

from game import save as save_module
from game.input_handlers import CONFIRM_KEYS
from game.menu import DialogueMenu, MenuOption

TITLE = "Touhou Roguelike"

# ("new" | "load", slot) or ("quit", None)
TitleChoice = Tuple[str, Optional[int]]


class TitleScreen(tcod.event.EventDispatch[None]):
    def __init__(self) -> None:
        self.choice: Optional[TitleChoice] = None
        self.menu: DialogueMenu
        self._open_root()

    def _open_root(self) -> None:
        options = []
        recent = save_module.most_recent_slot()
        if recent is not None:
            info = save_module.slot_info(recent)
            options.append(MenuOption("Continue", lambda: self._choose("load", recent), description=info.label))
        options.append(MenuOption("New Game", self._open_new_game))
        if recent is not None:
            options.append(MenuOption("Load Game", self._open_load_game))
        options.append(MenuOption("Quit", lambda: self._choose("quit", None)))
        self.menu = DialogueMenu(speaker="Main Menu", prompt="", options=options)

    def _open_new_game(self) -> None:
        options = []
        for info in save_module.all_slots():
            if info.exists:
                options.append(MenuOption(info.label, lambda s=info.slot: self._confirm_overwrite(s)))
            else:
                options.append(MenuOption(info.label, lambda s=info.slot: self._choose("new", s)))
        options.append(MenuOption("Back", self._open_root))
        self.menu = DialogueMenu(speaker="New Game", prompt="Choose a slot.", options=options)

    def _confirm_overwrite(self, slot: int) -> None:
        self.menu = DialogueMenu(
            speaker="New Game",
            prompt=f"Overwrite slot {slot}? That playthrough will be lost.",
            options=[
                MenuOption("Yes, overwrite", lambda: self._choose("new", slot)),
                MenuOption("No", self._open_new_game),
            ],
        )

    def _open_load_game(self) -> None:
        options = [
            MenuOption(info.label, lambda s=info.slot: self._choose("load", s))
            for info in save_module.all_slots()
            if info.exists
        ]
        options.append(MenuOption("Back", self._open_root))
        self.menu = DialogueMenu(speaker="Load Game", prompt="Choose a playthrough.", options=options)

    def _choose(self, kind: str, slot: Optional[int]) -> None:
        self.choice = (kind, slot)

    def reset(self) -> None:
        self.choice = None
        self._open_root()

    def ev_quit(self, event: tcod.event.Quit) -> None:
        self.choice = ("quit", None)

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        if event.sym in (tcod.event.KeySym.UP, tcod.event.KeySym.k):
            self.menu.move_cursor(-1)
        elif event.sym in (tcod.event.KeySym.DOWN, tcod.event.KeySym.j):
            self.menu.move_cursor(1)
        elif event.sym in CONFIRM_KEYS:
            self.menu.confirm()
        elif event.sym == tcod.event.KeySym.ESCAPE:
            self._open_root()

    def render(self, console: Console) -> None:
        console.clear()
        console.print((console.width - len(TITLE)) // 2, console.height // 3, TITLE, fg=(255, 80, 80))
        self.menu.render(console)
