"""Async min-interval rate limiter shared by the agent and customer LLM requests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class RateLimiter:
    """Allow at most `rpm` requests per minute by spacing them >= 60/rpm seconds apart.

    Min-interval (not a burst bucket) keeps the request stream smooth, which is what free-tier
    rate limits reward. `clock` and `sleep` are injectable for tests. `rpm <= 0` disables it.
    """

    def __init__(
        self,
        rpm: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next: float | None = None
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            if self._next is not None and now < self._next:
                await self._sleep(self._next - now)
                now = self._next
            self._next = now + self.interval

    __call__ = wait
