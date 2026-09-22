from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from tcod.console import Console


@dataclass
class MenuOption:
    label: str
    on_select: Callable[[], None]
    # Shown live as a tooltip under the option list while this option is
    # highlighted. Empty means no tooltip.
    description: str = ""
    # If set, pressing [X] on this option opens a full detail overlay built
    # by calling this (lazily, so building the detail text is only paid for
    # when actually inspected).
    inspect: Optional[Callable[[], List[str]]] = None
    # Overrides the default cursor/non-cursor color (e.g. a skill tree node
    # that's locked/unaffordable/already learned) — see
    # Engine._build_tree_node_menu. None keeps the plain white/grey
    # highlight-vs-not behavior every other menu already relies on.
    color: Optional[Tuple[int, int, int]] = None


class DialogueMenu:
    """A speaker/prompt/options overlay, styled like a VN dialogue box.

    Reusable for any interaction (NPC talk, shops, blacksmith, etc.) — the
    caller just supplies a speaker name, a prompt line, and options.
    """

    def __init__(self, speaker: str, prompt: str, options: List[MenuOption]):
        self.speaker = speaker
        self.prompt = prompt
        self.options = options
        self.cursor = 0

    def move_cursor(self, delta: int) -> None:
        self.cursor = (self.cursor + delta) % len(self.options)

    def confirm(self) -> None:
        self.options[self.cursor].on_select()

    @property
    def current_option(self) -> Optional[MenuOption]:
        return self.options[self.cursor] if self.options else None

    def render(self, console: Console, bottom_margin: int = 0) -> None:
        box_width = console.width - 4
        current = self.current_option

        tooltip_lines: List[str] = []
        if current is not None and current.description:
            tooltip_lines = textwrap.wrap(current.description, box_width - 4)[:3]
        show_inspect_hint = current is not None and current.inspect is not None

        extra = (len(tooltip_lines) + 1 if tooltip_lines else 0) + (1 if show_inspect_hint else 0)
        box_height = 3 + len(self.options) + extra
        y0 = console.height - bottom_margin - box_height - 1

        console.draw_frame(
            2, y0, box_width, box_height, fg=(255, 255, 255), bg=(0, 0, 0)
        )
        console.print(4, y0 + 1, f"{self.speaker}: {self.prompt}", fg=(255, 255, 0))
        for i, option in enumerate(self.options):
            prefix = "> " if i == self.cursor else "  "
            if option.color is not None:
                fg = option.color
            else:
                fg = (255, 255, 255) if i == self.cursor else (170, 170, 170)
            console.print(4, y0 + 2 + i, f"{prefix}{option.label}", fg=fg)

        next_y = y0 + 2 + len(self.options)
        if tooltip_lines:
            console.print(4, next_y, "-" * (box_width - 4), fg=(90, 90, 90))
            for j, line in enumerate(tooltip_lines):
                console.print(4, next_y + 1 + j, line, fg=(150, 150, 200))
            next_y += 1 + len(tooltip_lines)
        if show_inspect_hint:
            console.print(4, next_y, "[X] Inspect", fg=(120, 120, 120))
