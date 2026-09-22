from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LightSource:
    """A static, map-placed source of light (torch, crystal, etc.)."""

    x: int
    y: int
    radius: int


@dataclass
class DarknessSource:
    """A static source that blocks vision entirely within its radius,
    regardless of any light source, unless the viewer sees through darkness.
    """

    x: int
    y: int
    radius: int
