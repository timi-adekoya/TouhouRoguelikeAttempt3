from __future__ import annotations

from typing import List, Tuple

from tcod.console import Console

from game import inspect_text
from game.entity import Entity

HUD_HEIGHT = 9
MESSAGE_LOG_HEIGHT = 5


def render_inspect(console: Console, title: str, lines: List[str]) -> None:
    """A modal detail overlay for a single skill/item/class/tree node,
    opened via [X] on a menu option that supports it."""
    width = min(60, console.width - 6)
    height = min(len(lines) + 4, console.height - 4)
    x0 = (console.width - width) // 2
    y0 = (console.height - height) // 2

    console.draw_frame(x0, y0, width, height, title=title, fg=(255, 255, 0), bg=(0, 0, 0), clear=True)
    for i, line in enumerate(lines[: height - 3]):
        console.print(x0 + 2, y0 + 1 + i, line[: width - 4], fg=(255, 255, 255))
    console.print(x0 + 2, y0 + height - 2, "[Esc/X] Close", fg=(170, 170, 170))


ATTRIBUTE_LABELS = (
    ("Power", lambda a: a.power),
    ("Technique", lambda a: a.technique),
    ("Speed", lambda a: a.speed),
    ("Vitality", lambda a: a.vitality),
    ("Spirit", lambda a: a.spirit),
    ("Presence", lambda a: a.presence),
)


def _render_bar(
    console: Console,
    x: int,
    y: int,
    width: int,
    current: int,
    maximum: int,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    filled = int(width * current / maximum) if maximum > 0 else 0
    console.draw_rect(x=x, y=y, width=width, height=1, ch=ord(" "), bg=(40, 40, 40))
    if filled > 0:
        console.draw_rect(x=x, y=y, width=filled, height=1, ch=ord(" "), bg=color)
    console.print(x, y, f"{label} {current}/{maximum}", fg=(255, 255, 255))


def render_hud(
    console: Console,
    entity: Entity,
    hud_y: int,
    floor_label: str = "",
    party: Tuple[Entity, ...] = (),
) -> None:
    """Draw the HP/MP/SP(/Faith) bars for `entity` (the active character) in
    the reserved bottom strip, plus a floor-depth label and a compact
    HP-only line per other party member — so neither requires opening the
    party menu just to check."""
    console.draw_rect(
        x=0, y=hud_y, width=console.width, height=console.height - hud_y, ch=ord(" "), bg=(0, 0, 0)
    )
    console.print(1, hud_y, entity.name, fg=(255, 255, 255))
    if floor_label:
        console.print(console.width - 1 - len(floor_label), hud_y, floor_label, fg=(200, 200, 200))

    if entity.stats is not None:
        stats = entity.stats
        bar_width = console.width - 2
        _render_bar(console, 1, hud_y + 1, bar_width, stats.hp.current, stats.hp.max_value, "HP", (150, 30, 30))
        _render_bar(console, 1, hud_y + 2, bar_width, stats.mp.current, stats.mp.max_value, "MP", (30, 30, 150))
        _render_bar(console, 1, hud_y + 3, bar_width, stats.sp.current, stats.sp.max_value, "SP", (30, 140, 30))
        if stats.faith is not None:
            _render_bar(
                console, 1, hud_y + 4, bar_width, stats.faith.current, stats.faith.max_value, "Faith", (150, 130, 20)
            )

    party_row = hud_y + 5
    max_party_rows = console.height - party_row
    for member in party[:max_party_rows]:
        if member.stats is None:
            continue
        _render_bar(
            console, 1, party_row, console.width - 2,
            member.stats.hp.current, member.stats.hp.max_value, member.name, (110, 40, 40),
        )
        party_row += 1


def render_character_sheet(console: Console, entity: Entity, cursor: int = 0) -> None:
    """A full-screen overlay with detailed stats/attributes for `entity`.
    The progression column (classes/skills) is cursor-navigable and
    inspectable, same as the DialogueMenus."""
    x0, y0 = 4, 4
    width = console.width - 8
    height = console.height - 8
    console.draw_frame(
        x0, y0, width, height, title=entity.name, fg=(255, 255, 255), bg=(0, 0, 0), clear=True
    )

    if entity.stats is None:
        console.print(x0 + 2, y0 + 2, "No stats.", fg=(200, 200, 200))
        return

    stats = entity.stats
    y = y0 + 2
    console.print(x0 + 2, y, f"HP    {stats.hp.current}/{stats.hp.max_value}", fg=(255, 255, 255))
    y += 1
    console.print(x0 + 2, y, f"MP    {stats.mp.current}/{stats.mp.max_value}", fg=(255, 255, 255))
    y += 1
    console.print(x0 + 2, y, f"SP    {stats.sp.current}/{stats.sp.max_value}", fg=(255, 255, 255))
    y += 1
    if stats.faith is not None:
        console.print(x0 + 2, y, f"Faith {stats.faith.current}/{stats.faith.max_value}", fg=(255, 255, 255))
        y += 1

    y += 1
    console.print(x0 + 2, y, "Attributes", fg=(255, 255, 0))
    y += 1
    for label, getter in ATTRIBUTE_LABELS:
        stat = getter(stats.attributes)
        console.print(x0 + 2, y, f"{label:<10}{stat.current}/{stat.max_value}", fg=(255, 255, 255))
        y += 1

    rows = inspect_text.character_sheet_rows(entity)
    has_selectable = _render_sheet_rows(console, rows, x0 + width // 2, y0 + 2, cursor)

    hint = "[C/Esc] Close  [Up/Down] Select  [X] Inspect" if has_selectable else "[C/Esc] Close"
    console.print(x0 + 2, y0 + height - 2, hint, fg=(170, 170, 170))


def _render_sheet_rows(
    console: Console, rows: List["inspect_text.SheetRow"], x: int, y: int, cursor: int
) -> bool:
    selectable_index = 0
    has_selectable = False
    for row in rows:
        if row.selectable:
            has_selectable = True
            is_current = selectable_index == cursor
            prefix = "> " if is_current else "  "
            color = (255, 255, 255) if is_current else (170, 170, 170)
            selectable_index += 1
        else:
            prefix = ""
            color = (255, 255, 0) if row.header else (200, 200, 200)
        console.print(x, y, f"{prefix}{row.text}", fg=color)
        y += 1
    return has_selectable
