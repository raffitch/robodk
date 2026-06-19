"""A tiny pub/sub event bus that bridges worker threads to asyncio consumers.

Long jobs (robot moving through calibration poses) run in a background thread;
the web layer consumes their progress over a WebSocket on the asyncio loop.
:meth:`EventBus.publish` is therefore thread-safe and hops onto the loop via
``call_soon_threadsafe`` so subscribers (asyncio queues) receive events safely.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JobEvent:
    """One thing that happened during a job, fan-out to all subscribers."""

    type: str                       # progress | log | frame | status | result | error
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the asyncio loop that worker-thread events should hop onto."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, event: JobEvent) -> None:
        """Fan ``event`` out to every subscriber. Safe to call from any thread."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: JobEvent) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop on a slow/stuck consumer rather than block the producer;
                # frames are the only high-rate event and dropping them is fine.
                pass
