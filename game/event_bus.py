from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

Listener = Callable[[Dict[str, Any]], None]


class EventBus:
    """Global listener registry for true passives (on_damaged, on_kill,
    on_turn_start, ...). Listeners fire in priority order (higher first)."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Tuple[int, Listener]]] = {}

    def subscribe(self, event: str, listener: Listener, priority: int = 0) -> None:
        self._listeners.setdefault(event, []).append((priority, listener))
        self._listeners[event].sort(key=lambda pair: -pair[0])

    def unsubscribe(self, event: str, listener: Listener) -> None:
        self._listeners[event] = [
            (p, l) for p, l in self._listeners.get(event, []) if l is not listener
        ]

    def emit(self, event: str, payload: Dict[str, Any]) -> None:
        for _, listener in list(self._listeners.get(event, [])):
            listener(payload)
