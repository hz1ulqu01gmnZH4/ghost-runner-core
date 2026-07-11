"""LlmScheduler contract tests (§A4.4): priority order, reserved-slot
admission, preemption-by-cancellation of pure background work, effectful jobs
never preempted, turn cancellation, aging."""

import asyncio

import pytest

from ghost_runner_core.llm.scheduler import (
    PRIO_CHAT,
    PRIO_CONSOLIDATION,
    PRIO_EMBEDDINGS,
    LlmScheduler,
)


class Gate:
    """A job body that parks until released, recording its lifecycle."""

    def __init__(self, name: str, journal: list):
        self.name = name
        self.journal = journal
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def __call__(self):
        self.journal.append(("start", self.name))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.journal.append(("cancelled", self.name))
            raise
        self.journal.append(("done", self.name))
        return self.name


@pytest.fixture
async def sched():
    s = LlmScheduler(slots=2, reserved_interactive=1, aging_s=3600)
    s.start()
    yield s
    await s.stop()


async def _settle():
    for _ in range(10):
        await asyncio.sleep(0)


async def test_background_never_takes_last_slot(sched):
    journal = []
    bg1, bg2 = Gate("bg1", journal), Gate("bg2", journal)
    t1 = asyncio.ensure_future(sched.submit(PRIO_EMBEDDINGS, bg1, pure=True))
    t2 = asyncio.ensure_future(sched.submit(PRIO_EMBEDDINGS, bg2, pure=True))
    await bg1.started.wait()
    await _settle()
    # slots=2, reserved=1: the second background job must wait even though a slot is free
    assert not bg2.started.is_set()
    chat = Gate("chat", journal)
    t3 = asyncio.ensure_future(sched.submit(PRIO_CHAT, chat, pure=True))
    await chat.started.wait()  # chat takes the reserved slot immediately
    chat.release.set()
    bg1.release.set()
    await t3
    await t1
    await bg2.started.wait()   # bg1's slot frees → bg2 finally runs
    bg2.release.set()
    await t2


async def test_chat_preempts_newest_pure_background(sched):
    journal = []
    bg = Gate("bg", journal)
    chat1, chat2 = Gate("chat1", journal), Gate("chat2", journal)
    # bg must hold its slot BEFORE chat1 is enqueued: admission never gives a
    # background job the last free slot, so enqueueing both at once would let
    # chat1 win the race and starve bg (the test would deadlock, not preempt).
    tb = asyncio.ensure_future(sched.submit(PRIO_CONSOLIDATION, bg, pure=True))
    await bg.started.wait()
    tc1 = asyncio.ensure_future(sched.submit(PRIO_CHAT, chat1, pure=True))
    await chat1.started.wait()
    # Both slots busy; second chat arrives → the pure background job is preempted.
    tc2 = asyncio.ensure_future(sched.submit(PRIO_CHAT, chat2, pure=True))
    await chat2.started.wait()
    assert ("cancelled", "bg") in journal
    assert not tb.done()  # preemption re-enqueues silently: submitter still waiting
    chat1.release.set()
    chat2.release.set()
    await tc1
    await tc2
    await bg.started.wait()  # re-enqueued job runs again once a slot frees
    bg.release.set()
    assert await tb == "bg"


async def test_effectful_job_never_preempted(sched):
    journal = []
    eff = Gate("eff", journal)
    chat1, chat2 = Gate("chat1", journal), Gate("chat2", journal)
    # Same admission-order note as above: eff must be running before chat1 exists.
    te = asyncio.ensure_future(sched.submit(PRIO_CONSOLIDATION, eff, pure=False))
    await eff.started.wait()
    tc1 = asyncio.ensure_future(sched.submit(PRIO_CHAT, chat1, pure=True))
    await chat1.started.wait()
    tc2 = asyncio.ensure_future(sched.submit(PRIO_CHAT, chat2, pure=True))
    await _settle()
    assert not chat2.started.is_set()          # no preemptable victim → chat2 queues
    assert ("cancelled", "eff") not in journal
    chat1.release.set()
    await tc1
    await chat2.started.wait()                 # runs when a slot frees normally
    chat2.release.set()
    eff.release.set()
    await tc2
    await te


async def test_cancel_turn_kills_queued_and_running(sched):
    journal = []
    running = Gate("running", journal)
    queued = Gate("queued", journal)
    blocker1, blocker2 = Gate("b1", journal), Gate("b2", journal)
    tr = asyncio.ensure_future(sched.submit(PRIO_CHAT, running, pure=True, turn_id=7))
    await running.started.wait()
    t1 = asyncio.ensure_future(sched.submit(PRIO_CHAT, blocker1, pure=True))
    await blocker1.started.wait()
    tq = asyncio.ensure_future(sched.submit(PRIO_CHAT, queued, pure=True, turn_id=7))
    t2 = asyncio.ensure_future(sched.submit(PRIO_CHAT, blocker2, pure=True))
    await _settle()
    assert sched.cancel_turn(7) == 2
    with pytest.raises(asyncio.CancelledError):
        await tr
    with pytest.raises(asyncio.CancelledError):
        await tq
    assert ("start", "queued") not in journal  # never ran
    blocker1.release.set()
    blocker2.release.set()
    await t1
    await t2


async def test_aging_reorders_background_classes():
    s = LlmScheduler(slots=2, reserved_interactive=1, aging_s=0.01)
    s.start()
    try:
        journal = []
        hog = Gate("hog", journal)
        th = asyncio.ensure_future(s.submit(PRIO_EMBEDDINGS, hog, pure=True))
        await hog.started.wait()
        # Old consolidation vs fresh embeddings: aging must rank the old one first.
        old = Gate("old", journal)
        to = asyncio.ensure_future(s.submit(PRIO_CONSOLIDATION, old, pure=True))
        await asyncio.sleep(0.05)  # old ages past embeddings priority
        fresh = Gate("fresh", journal)
        tf = asyncio.ensure_future(s.submit(PRIO_EMBEDDINGS, fresh, pure=True))
        hog.release.set()
        await th
        await old.started.wait()
        assert not fresh.started.is_set()  # aged job won the freed slot
        old.release.set()
        await to
        await fresh.started.wait()
        fresh.release.set()
        await tf
    finally:
        await s.stop()
