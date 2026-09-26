from __future__ import annotations

from typing import Optional

import tcod.event

MOVE_KEYS = {
    tcod.event.KeySym.UP: (0, -1),
    tcod.event.KeySym.DOWN: (0, 1),
    tcod.event.KeySym.LEFT: (-1, 0),
    tcod.event.KeySym.RIGHT: (1, 0),
    tcod.event.KeySym.k: (0, -1),
    tcod.event.KeySym.j: (0, 1),
    tcod.event.KeySym.h: (-1, 0),
    tcod.event.KeySym.l: (1, 0),
    tcod.event.KeySym.y: (-1, -1),
    tcod.event.KeySym.u: (1, -1),
    tcod.event.KeySym.b: (-1, 1),
    tcod.event.KeySym.n: (1, 1),
}

CONFIRM_KEYS = {tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER, tcod.event.KeySym.SPACE}


class EventHandler(tcod.event.EventDispatch[None]):
    def __init__(self, engine: "Engine"):
        self.engine = engine
        self.quit = False

    def ev_quit(self, event: tcod.event.Quit) -> Optional[None]:
        self.quit = True

    def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[None]:
        if self.engine.pause_menu_open:
            if event.sym == tcod.event.KeySym.q:
                self.quit = True
            elif event.sym == tcod.event.KeySym.ESCAPE:
                self.engine.close_pause_menu()
            else:
                self._handle_menu_key(event)
            return

        if self.engine.inspect_open:
            if event.sym in (tcod.event.KeySym.ESCAPE, tcod.event.KeySym.x):
                self.engine.close_inspect()
            return

        if self.engine.character_screen_open:
            if event.sym in (tcod.event.KeySym.c, tcod.event.KeySym.ESCAPE):
                self.engine.close_character_screen()
            elif event.sym in (tcod.event.KeySym.UP, tcod.event.KeySym.k):
                self.engine.move_character_sheet_cursor(-1)
            elif event.sym in (tcod.event.KeySym.DOWN, tcod.event.KeySym.j):
                self.engine.move_character_sheet_cursor(1)
            elif event.sym == tcod.event.KeySym.x:
                self.engine.inspect_character_sheet_entry()
            return

        if self.engine.message_log_expanded:
            self._handle_log_key(event)
            return

        if self.engine.target_selector is not None:
            self._handle_targeting_key(event)
            return

        if self.engine.active_menu is not None:
            self._handle_menu_key(event)
            return

        if event.sym == tcod.event.KeySym.ESCAPE:
            self.engine.open_pause_menu()
            return

        if event.sym == tcod.event.KeySym.c:
            self.engine.open_character_screen()
            return

        if event.sym == tcod.event.KeySym.z:
            self.engine.open_skill_menu()
            return

        if event.sym == tcod.event.KeySym.p:
            self.engine.open_progression_menu()
            return

        if event.sym == tcod.event.KeySym.i:
            self.engine.open_inventory_menu()
            return

        if event.sym == tcod.event.KeySym.o:
            self.engine.open_party_menu()
            return

        if event.sym == tcod.event.KeySym.g:
            self.engine.try_pickup()
            return

        if event.sym == tcod.event.KeySym.f:
            self.engine.open_fire_selector()
            return

        if event.sym == tcod.event.KeySym.t:
            self.engine.open_mark_target_selector()
            return

        if event.sym == tcod.event.KeySym.m:
            self.engine.toggle_message_log()
            return

        if event.sym == tcod.event.KeySym.e:
            self.engine.start_auto_explore()
            return

        if event.sym == tcod.event.KeySym.F1:
            self.engine.toggle_debug_log_all_progression()
            return

        shift_held = bool(event.mod & tcod.event.Modifier.SHIFT)

        if event.sym == tcod.event.KeySym.KP_GREATER or (
            event.sym == tcod.event.KeySym.PERIOD and shift_held
        ):
            self.engine.try_descend()
            return

        if event.sym == tcod.event.KeySym.KP_LESS or (
            event.sym == tcod.event.KeySym.COMMA and shift_held
        ):
            self.engine.try_use_stairs_up()
            return

        move = MOVE_KEYS.get(event.sym)
        if move is not None:
            dx, dy = move
            self.engine.try_move_player(dx, dy)

    def _handle_log_key(self, event: tcod.event.KeyDown) -> None:
        if event.sym in (tcod.event.KeySym.ESCAPE, tcod.event.KeySym.m):
            self.engine.close_message_log()
        elif event.sym in (tcod.event.KeySym.UP, tcod.event.KeySym.k):
            self.engine.scroll_message_log(1)
        elif event.sym in (tcod.event.KeySym.DOWN, tcod.event.KeySym.j):
            self.engine.scroll_message_log(-1)
        elif event.sym == tcod.event.KeySym.PAGEUP:
            self.engine.scroll_message_log(10)
        elif event.sym == tcod.event.KeySym.PAGEDOWN:
            self.engine.scroll_message_log(-10)

    def _handle_targeting_key(self, event: tcod.event.KeyDown) -> None:
        if event.sym == tcod.event.KeySym.ESCAPE:
            self.engine.cancel_targeting()
        elif event.sym in CONFIRM_KEYS:
            self.engine.confirm_targeting()
        else:
            move = MOVE_KEYS.get(event.sym)
            if move is not None:
                self.engine.move_reticle(*move)

    def _handle_menu_key(self, event: tcod.event.KeyDown) -> None:
        menu = self.engine.active_menu
        assert menu is not None

        if event.sym == tcod.event.KeySym.ESCAPE:
            self.engine.close_menu()
        elif event.sym in (tcod.event.KeySym.UP, tcod.event.KeySym.k):
            menu.move_cursor(-1)
        elif event.sym in (tcod.event.KeySym.DOWN, tcod.event.KeySym.j):
            menu.move_cursor(1)
        elif event.sym in CONFIRM_KEYS:
            menu.confirm()
        elif event.sym == tcod.event.KeySym.x:
            option = menu.current_option
            if option is not None and option.inspect is not None:
                self.engine.open_inspect(option.label, option.inspect())
