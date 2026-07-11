"""TurnManager: sole issuer of turn ids; cancellation authority (§A4.4, behavior B6.1).

Every cancellable artifact carries a turn id; `is_obsolete()` answers for every
layer. Late artifacts from cancelled turns are dropped at scheduler dispatch
(cancel_turn kills the turn's jobs), at orchestrator emission (on_delta checks
is_obsolete before broadcasting), and at client render (is_turn_obsolete).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..store import CoreStore

log = logging.getLogger(__name__)


@dataclass
class Turn:
    id: int
    origin: str
    owner_session: str
    status: str = "active"
    task: asyncio.Task | None = field(default=None, repr=False)


class TurnManager:
    def __init__(self, store: CoreStore,
                 notify: Callable[[Turn, str, str | None], Awaitable[None]]) -> None:
        """`notify(turn, event, reason)` publishes the `turn` protocol message."""
        self._store = store
        self._notify = notify
        self._current: Turn | None = None

    @property
    def current(self) -> Turn | None:
        return self._current

    def is_obsolete(self, turn_id: int) -> bool:
        """A turn id is live iff it is the current active turn — there is exactly
        one foreground (behavior B1-1). Finished, cancelled, and never-issued ids
        are all equally obsolete: none of them may render."""
        return not (self._current is not None and self._current.id == turn_id)

    async def begin(self, origin: str, owner_session: str) -> Turn:
        """Start a turn. An active turn is superseded: sending a new message
        while the companion responds is an implicit barge-in (behavior B6)."""
        if self._current is not None and self._current.status == "active":
            await self.cancel(self._current.id, reason="superseded", by_session=None)
        turn = Turn(id=self._store.next_turn_id(), origin=origin, owner_session=owner_session)
        self._store.record_turn(turn.id, origin, owner_session)
        self._current = turn
        await self._notify(turn, "started", None)
        return turn

    async def cancel(self, turn_id: int, reason: str, by_session: str | None) -> bool:
        """Cancel by id. Ownership (§A7.6): a non-owner session may cancel only
        proactive turns; by_session=None is core-internal authority.
        Returns False if the turn is already finished (idempotent)."""
        turn = self._current
        if turn is None or turn.id != turn_id:
            return False
        if by_session is not None and by_session != turn.owner_session \
                and turn.origin != "proactive":
            raise PermissionError(
                f"session {by_session} does not own turn {turn_id}")
        turn.status = "cancelled"
        self._store.finish_turn(turn_id, "cancelled", reason)
        self._current = None
        if turn.task is not None and not turn.task.done():
            turn.task.cancel()
        await self._notify(turn, "cancelled", reason)
        log.info("turn %d cancelled (%s)", turn_id, reason)
        return True

    async def finish(self, turn_id: int, status: str) -> None:
        """Mark the current turn done/error. No-op if it was cancelled meanwhile
        (the cancel already notified — a late finish must not resurrect it)."""
        turn = self._current
        if turn is None or turn.id != turn_id:
            return
        turn.status = status
        self._store.finish_turn(turn_id, status)
        self._current = None
        await self._notify(turn, "done", None if status == "done" else status)
