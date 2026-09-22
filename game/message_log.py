from __future__ import annotations

import textwrap
from typing import List, Tuple

from tcod.console import Console

WHITE = (255, 255, 255)
DAMAGE_COLOR = (255, 120, 120)
DEATH_COLOR = (255, 60, 60)
INFO_COLOR = (170, 170, 255)


class Message:
    def __init__(self, text: str, color: Tuple[int, int, int]):
        self.plain_text = text
        self.color = color
        self.count = 1

    @property
    def full_text(self) -> str:
        if self.count > 1:
            return f"{self.plain_text} (x{self.count})"
        return self.plain_text


class MessageLog:
    def __init__(self) -> None:
        self.messages: List[Message] = []

    def add_message(self, text: str, color: Tuple[int, int, int] = WHITE, *, stack: bool = True) -> None:
        if stack and self.messages and self.messages[-1].plain_text == text:
            self.messages[-1].count += 1
            return
        self.messages.append(Message(text, color))

    def wrapped_lines(self, width: int) -> List[Tuple[str, Tuple[int, int, int]]]:
        """All messages word-wrapped to `width`, most recent first."""
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        for message in reversed(self.messages):
            for line in reversed(textwrap.wrap(message.full_text, width)):
                lines.append((line, message.color))
        return lines

    def render(self, console: Console, x: int, y: int, width: int, height: int) -> None:
        """The compact HUD strip: always shows the most recent lines."""
        console.draw_rect(x=x, y=y, width=width, height=height, ch=ord(" "), bg=(0, 0, 0))
        lines = self.wrapped_lines(width)[:height]
        for i, (line, color) in enumerate(lines):
            console.print(x, y + height - 1 - i, line, fg=color)

    def render_expanded(
        self, console: Console, x: int, y: int, width: int, height: int, scroll_offset: int
    ) -> int:
        """Full scrollback view. `scroll_offset` counts lines back from the
        most recent (0 = latest). Returns the offset actually used, clamped
        to the available history, so the caller can keep its state in sync."""
        console.draw_frame(x, y, width, height, title="Message Log", fg=(255, 255, 255), bg=(0, 0, 0), clear=True)

        inner_width, inner_height = width - 2, height - 3
        all_lines = self.wrapped_lines(inner_width)
        max_offset = max(0, len(all_lines) - inner_height)
        scroll_offset = max(0, min(scroll_offset, max_offset))

        window = all_lines[scroll_offset : scroll_offset + inner_height]
        for i, (line, color) in enumerate(window):
            console.print(x + 1, y + 1 + inner_height - 1 - i, line, fg=color)

        footer = "[Up/Down] Scroll  [Esc/M] Close"
        if scroll_offset > 0:
            footer = f"({scroll_offset} older)  " + footer
        console.print(x + 1, y + height - 2, footer[: width - 2], fg=(170, 170, 170))
        return scroll_offset
