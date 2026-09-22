from __future__ import annotations

from typing import List

from game.entity import Entity

ACTION_COST = 100.0
BASE_ENERGY_RATE = 10.0
# At max Speed (99), this nearly doubles the base rate (10 -> ~19.9), i.e.
# almost 2 actions for every 1 a Speed-0 entity gets. A real, meaningful
# investment rather than a slight nudge, without letting Speed alone
# triple someone's turn frequency.
SPEED_ENERGY_FACTOR = 0.1


class TurnQueue:
    """Energy-based turn scheduler.

    Every participating entity accrues energy each tick at a mostly-fixed
    rate; Speed only contributes a small bonus. Once an entity's energy
    reaches ACTION_COST it's eligible to act; ties/eligibility go to whoever
    has the most energy. Energy carries over past the threshold, so a
    slightly faster entity gradually earns extra turns rather than always
    going first.
    """

    def _rate(self, entity: Entity) -> float:
        rate = BASE_ENERGY_RATE
        if entity.stats is not None:
            rate += entity.stats.attributes.speed.current * SPEED_ENERGY_FACTOR
        return rate

    def tick(self, actors: List[Entity]) -> None:
        for entity in actors:
            entity.energy += self._rate(entity)

    def spend(self, entity: Entity) -> None:
        entity.energy -= ACTION_COST

    def next_ready(self, actors: List[Entity]) -> Entity:
        """Advance ticks until at least one actor can act, then return the
        one with the most energy."""
        while True:
            ready = [e for e in actors if e.energy >= ACTION_COST]
            if ready:
                return max(ready, key=lambda e: e.energy)
            self.tick(actors)
