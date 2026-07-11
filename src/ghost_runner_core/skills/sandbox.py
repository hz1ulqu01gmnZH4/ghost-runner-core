"""Deny-by-default skill execution under bubblewrap.

The tool-pipeline research picks Wasm/Wassette as the end state; this first
slice uses bubblewrap (Claude Code's sandbox model) because core runs on
Linux/WSL and bwrap gives ms-startup, no-root, capability-style isolation
today. Inside the sandbox a skill sees:

  - read-only /usr (+ /bin /lib /lib64 symlinks), its own dir read-only at /skill
  - a private size-capped /tmp, /proc, /dev; nothing else — no /home, no core state
  - no network (fresh namespaces via --unshare-all), cleared environment, no caps
  - python3 -I (isolated mode: no user site-packages, no cwd on sys.path)
  - rlimits via prlimit: address space, CPU seconds, process count — a skill
    can waste at most its own timeout window, not the host

There is deliberately NO unsandboxed fallback: if bwrap is missing the core
refuses to start (PRD §6.1 — a fallback must be a deliberate, louder decision,
never silent).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from .library import Skill, SkillError, SkillLibrary

log = logging.getLogger(__name__)

MAX_RESULT_BYTES = 1_000_000  # a skill result is a small JSON object, not a data dump
LIMIT_AS_BYTES = 512 * 1024 * 1024  # address space per skill process
LIMIT_NPROC = 64                    # fork-bomb guard (fresh userns, so count starts near 0)
TMPFS_BYTES = 64 * 1024 * 1024      # the skill's private /tmp
_PRLIMIT = "/usr/bin/prlimit"
_STDERR_CAP = 65536
_STDERR_EXCERPT = 400
_READ_CHUNK = 65536


class SkillExecutionError(Exception):
    """The skill ran and failed (bad exit, bad output, timeout). Per-request,
    not fatal to the core."""


class _StdoutOverflow(Exception):
    """Internal: skill exceeded MAX_RESULT_BYTES while still running."""


def check_sandbox_available() -> str:
    """Startup fail-fast (§A9): no bwrap, no skills — never run unsandboxed."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SkillError(
            "[skills] is configured but bwrap (bubblewrap) is not installed; "
            "refusing to run skills without a sandbox")
    if not Path(_PRLIMIT).is_file():
        raise SkillError(
            f"[skills] is configured but {_PRLIMIT} (util-linux) is missing; "
            "refusing to run skills without resource limits")
    return bwrap


def _bwrap_argv(bwrap: str, skill_dir: str, timeout_s: float) -> list[str]:
    # RLIMIT_CPU is the busy-loop backstop; wall-clock timeout is the primary
    # kill. +2 s so a legitimately CPU-bound skill hits the wall clock first.
    cpu_s = int(timeout_s) + 2
    return [
        bwrap,
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--size", str(TMPFS_BYTES), "--tmpfs", "/tmp",
        "--ro-bind", skill_dir, "/skill",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop", "ALL",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "LANG", "C.UTF-8",
        "--chdir", "/skill",
        "--",
        _PRLIMIT, f"--as={LIMIT_AS_BYTES}", f"--cpu={cpu_s}", f"--nproc={LIMIT_NPROC}",
        "--", "/usr/bin/python3", "-I", "/skill/skill.py",
    ]


async def _drain(stream: asyncio.StreamReader, cap: int, *,
                 discard_overflow: bool) -> bytes:
    """Read a pipe to EOF, keeping at most `cap` bytes in memory.

    discard_overflow=True (stderr): keep reading so the skill never blocks on
    a full pipe, but stop storing. discard_overflow=False (stdout): past the
    cap the result is already invalid — raise so the caller kills the skill
    instead of buffering its flood.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total <= cap:
            chunks.append(chunk)
        elif not discard_overflow:
            raise _StdoutOverflow


_REAP_TIMEOUT_S = 5.0


async def _reap(proc: asyncio.subprocess.Process, skill_name: str) -> None:
    """SIGKILL and fully collect the sandbox. asyncio's Process.wait() only
    completes once the process's pipes hit EOF, so a kill path that stopped
    reading them must discard the dead sandbox's buffered output or wait()
    blocks forever (observed, not theoretical)."""
    try:
        proc.kill()
    except ProcessLookupError:
        pass  # already exited on its own; pipes still need draining below
    try:
        async with asyncio.timeout(_REAP_TIMEOUT_S):
            await asyncio.gather(
                _drain(proc.stdout, 0, discard_overflow=True),
                _drain(proc.stderr, 0, discard_overflow=True))
            await proc.wait()
    except TimeoutError:
        raise SkillError(
            f"skill {skill_name}: sandbox process did not exit after SIGKILL") from None


async def run_skill(library: SkillLibrary, skill: Skill, args: dict) -> dict:
    """Run one skill invocation: JSON args on stdin, JSON object on stdout.

    Raises SkillError on integrity/sandbox problems (hash drift, bwrap broken)
    and SkillExecutionError on runtime failure. Never returns partial results.
    """
    library.verify_hash(skill)
    bwrap = check_sandbox_available()
    argv = _bwrap_argv(bwrap, str(skill.path), skill.timeout_s)

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise SkillError(f"cannot start skill sandbox: {exc}") from exc

    async def feed_stdin() -> None:
        try:
            proc.stdin.write(json.dumps(args, ensure_ascii=False).encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass  # skill died before reading its input; the exit-code path reports it

    tasks = (asyncio.create_task(feed_stdin()),
             asyncio.create_task(_drain(proc.stdout, MAX_RESULT_BYTES,
                                        discard_overflow=False)),
             asyncio.create_task(_drain(proc.stderr, _STDERR_CAP,
                                        discard_overflow=True)))
    try:
        async with asyncio.timeout(skill.timeout_s):
            _, stdout, stderr = await asyncio.gather(*tasks)
            await proc.wait()
    except (TimeoutError, _StdoutOverflow) as exc:
        # A child exception leaves sibling gather tasks running; two readers on
        # one StreamReader is illegal, so settle them before _reap drains.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _reap(proc, skill.name)
        if isinstance(exc, TimeoutError):
            raise SkillExecutionError(
                f"skill {skill.name} timed out after {skill.timeout_s}s") from None
        raise SkillExecutionError(
            f"skill {skill.name} output capped at {MAX_RESULT_BYTES} bytes; "
            "results must be small JSON objects") from None

    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        # bwrap reports its own setup failures (e.g. user namespaces disabled)
        # as "bwrap: ..." on stderr — that is a broken sandbox, not a broken
        # skill, and must surface as an operator problem (heuristic: a skill
        # could print the same prefix, and would then be miscategorized loudly).
        if detail.startswith("bwrap:"):
            raise SkillError(f"sandbox failed for skill {skill.name}: {detail[:_STDERR_EXCERPT]}")
        raise SkillExecutionError(
            f"skill {skill.name} exited {proc.returncode}: "
            f"{detail[:_STDERR_EXCERPT] or '(no stderr)'}")
    try:
        result = json.loads(stdout)
    except ValueError as exc:
        raise SkillExecutionError(
            f"skill {skill.name} wrote invalid JSON to stdout: {exc}") from exc
    if not isinstance(result, dict):
        raise SkillExecutionError(
            f"skill {skill.name} must output a JSON object, got {type(result).__name__}")
    return result
