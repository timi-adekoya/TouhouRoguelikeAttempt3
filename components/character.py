from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Character:
    """Game-facing identity/progression data for a playable character.

    Kept separate from Entity (which only knows about world position and
    rendering) so that stats, classes, and skills can be added here later
    without touching Entity, the interaction menu, or party-swap code.
    """

    name: str
    recruited: bool = True
