"""Priority scheduler in front of llama-server (§A4.4).

llama.cpp's own queue is FIFO with no preemption, so priority lives here:
  0 chat > 1 perception > 2 embeddings > 3 consolidation
- Ordering ages: waiting jobs gain one level per aging_s so background work
  is never starved. Aging affects ORDER only — admission always treats a
  background job as background (a starved consolidation job must not eat the
  reserved interactive slot).
- Admission: a background job (nominal prio >= 1) never takes the last free
  slot; that slot is reserved for interactive chat.
- Preemption = cancellation: an arriving chat job with no free slot cancels
  the newest in-flight PURE background job and silently re-enqueues it.
  Effectful jobs (tool calls / writes downstream) are never preempted and
  never auto-re-enqueued (§A4.4); in M0 no effectful jobs exist yet, but the
  contract is enforced now so M5 cannot violate it by accident.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

PRIO_CHAT = 0
PRIO_PERCEPTION = 1
PRIO_EMBEDDINGS = 2
PRIO_CONSOLIDATION = 3


@dataclass(eq=False)  # identity equality — Job lives in the _preempting set
class Job:
    prio: int
    factory: Callable[[], Awaitable[object]]
    pure: bool
    turn_id: int | None
    label: str
    future: asyncio.Future = field(repr=False)
    enqueued_at: float = field(default_factory=time.monotonic)
    seq: int = 0
    preempt_count: int = 0

    def effective_prio(self, now: float, aging_s: float) -> int:
        aged = int((now - self.enqueued_at) / aging_s)
        return max(0, self.prio - aged)


class LlmScheduler:
    def __init__(self, slots: int = 2, reserved_interactive: int = 1,
                 aging_s: float = 60.0) -> None:
        if reserved_interactive >= slots:
            raise ValueError("reserved_interactive must leave at least one shared slot")
        self._slots = slots
        self._reserved = reserved_interactive
        self._aging_s = aging_s
        self._queue: list[Job] = []
        self._running: dict[asyncio.Task, Job] = {}
        self._preempting: set[Job] = set()
        self._wake = asyncio.Event()
        self._seq = itertools.count()
        self._worker: asyncio.Task | None = None
        self._fatal: BaseException | None = None

    def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("scheduler already started")
        self._worker = asyncio.get_running_loop().create_task(
            self._run(), name="llm-scheduler")
        self._worker.add_done_callback(self._on_worker_dead)

    def _on_worker_dead(self, task: asyncio.Task) -> None:
        """The worker loop must never exit on its own. If it dies, every pending
        future would otherwise hang forever — fail them all loudly instead."""
        if task.cancelled():
            return  # normal stop() path
        exc = task.exception() or RuntimeError("llm-scheduler worker exited unexpectedly")
        self._fatal = exc
        log.critical("llm-scheduler worker died; failing all pending jobs", exc_info=exc)
        for job in [*self._queue, *self._running.values()]:
            if not job.future.done():
                job.future.set_exception(exc)
        self._queue.clear()
        for running_task in list(self._running):
            running_task.cancel()
        self._running.clear()

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None
        for task in list(self._running):
            task.cancel()
        for job in self._queue:
            if not job.future.done():
                job.future.cancel()
        self._queue.clear()

    async def submit(self, prio: int, factory: Callable[[], Awaitable[object]], *,
                     pure: bool, turn_id: int | None = None, label: str = "") -> object:
        """Enqueue and await the job's result. Cancelling this await cancels the job."""
        if self._fatal is not None:
            raise RuntimeError("llm-scheduler is dead") from self._fatal
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        job = Job(prio=prio, factory=factory, pure=pure, turn_id=turn_id,
                  label=label or getattr(factory, "__qualname__", type(factory).__name__),
                  future=future, seq=next(self._seq))
        self._queue.append(job)
        self._wake.set()
        try:
            return await future
        except asyncio.CancelledError:
            self._remove(job)
            raise

    def cancel_turn(self, turn_id: int) -> int:
        """Cancel queued + in-flight jobs of a turn. Returns count cancelled."""
        count = 0
        for job in [j for j in self._queue if j.turn_id == turn_id]:
            self._queue.remove(job)
            if not job.future.done():
                job.future.cancel()
            count += 1
        for task, job in list(self._running.items()):
            if job.turn_id == turn_id:
                task.cancel()
                count += 1
        return count

    # -- internals -------------------------------------------------------------

    def _remove(self, job: Job) -> None:
        if job in self._queue:
            self._queue.remove(job)
        for task, running in list(self._running.items()):
            if running is job:
                task.cancel()

    async def _run(self) -> None:
        while True:
            self._dispatch()
            waiters = [asyncio.ensure_future(self._wake.wait())]
            waiters.extend(self._running)
            done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            self._wake.clear()
            for fut in pending:
                if fut not in self._running:  # the wake waiter
                    fut.cancel()
            for fut in done:
                if fut in self._running:
                    self._running.pop(fut)

    def _dispatch(self) -> None:
        now = time.monotonic()
        while self._queue:
            free = self._slots - len(self._running)
            candidates = sorted(
                self._queue, key=lambda j: (j.effective_prio(now, self._aging_s), j.seq))
            job = None
            for cand in candidates:
                is_interactive = cand.prio == PRIO_CHAT
                if free >= 1 and is_interactive:
                    job = cand
                    break
                if free > self._reserved and not is_interactive:
                    job = cand
                    break
                if free < 1 and is_interactive and self._try_preempt():
                    # A slot is being vacated; re-dispatch on the wake that follows.
                    return
            if job is None:
                return
            self._queue.remove(job)
            if job.future.done():
                continue  # cancelled while queued
            task = asyncio.get_running_loop().create_task(
                self._execute(job), name=f"llm-job-{job.label}")
            self._running[task] = job

    def _try_preempt(self) -> bool:
        """Cancel the newest in-flight pure background job to free a slot for chat."""
        victims = sorted(
            (j for t, j in self._running.items()
             if j.pure and j.prio >= PRIO_EMBEDDINGS and j not in self._preempting),
            key=lambda j: -j.seq)
        if not victims:
            return False
        victim = victims[0]
        self._preempting.add(victim)
        for task, job in self._running.items():
            if job is victim:
                log.info("preempting %s for interactive chat", victim.label)
                task.cancel()
                return True
        raise RuntimeError("preemption victim vanished from the running set")  # invariant

    async def _execute(self, job: Job) -> None:
        try:
            result = await job.factory()
        except asyncio.CancelledError:
            if job in self._preempting:
                # Preempted, not user-cancelled: silently requeue, future stays pending.
                self._preempting.discard(job)
                job.preempt_count += 1
                job.enqueued_at = time.monotonic()  # aging restarts; seq keeps order fairness
                self._queue.append(job)
                self._wake.set()
            elif not job.future.done():
                job.future.cancel()
            return
        except Exception as exc:  # propagate to the submitter — never swallowed
            if not job.future.done():
                job.future.set_exception(exc)
            return
        if not job.future.done():
            job.future.set_result(result)


async def stream_via_scheduler(scheduler: LlmScheduler, prio: int, turn_id: int | None,
                               make_stream: Callable[[], AsyncIterator[str]],
                               on_delta: Callable[[str], Awaitable[None]],
                               label: str = "chat-stream") -> None:
    """Run a streaming LLM call as one scheduler job, pushing deltas out as they
    arrive. The whole stream occupies one slot for its duration (chat turns are
    the interactive foreground; chunking applies to background jobs, not these).
    """
    async def run() -> None:
        async for delta in make_stream():
            await on_delta(delta)

    await scheduler.submit(prio, run, pure=True, turn_id=turn_id, label=label)
