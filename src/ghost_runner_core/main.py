"""Process entry point: wiring + fail-fast startup checks (§A9)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .asr.client import AsrError, WhisperClient
from .config import ConfigError, CoreConfig, load_config
from .llm.client import LlamaClient, LlmError
from .llm.scheduler import LlmScheduler
from .orchestrator.orchestrator import Orchestrator
from .server.session import SessionManager
from .server.ws_server import WsServer
from .skills.library import SkillError, SkillLibrary
from .skills.sandbox import check_sandbox_available
from .store import CoreStore
from .tts import IrodoriClient, TtsClient, TtsError

log = logging.getLogger(__name__)


async def run(cfg: CoreConfig) -> None:
    llama = LlamaClient(cfg.llm.base_url, cfg.llm.model)
    available = await llama.check_reachable()  # raises LlmError with the exact dependency
    if cfg.llm.model not in available:
        raise LlmError(
            f"configured model {cfg.llm.model!r} not served by {cfg.llm.base_url} "
            f"(available: {available})")
    log.info("llama-server ok at %s, model %s", cfg.llm.base_url, cfg.llm.model)

    asr: WhisperClient | None = None
    if cfg.asr is not None:
        asr = WhisperClient(cfg.asr.server_url, cfg.asr.language)
        await asr.check_reachable()  # raises AsrError with the exact dependency
        log.info("whisper-server ok at %s, language %s", cfg.asr.server_url, cfg.asr.language)

    skills: SkillLibrary | None = None
    if cfg.skills is not None:
        check_sandbox_available()  # raises SkillError: no bwrap, no skills
        skills = SkillLibrary(Path(cfg.skills.dir).expanduser())
        skills.load()  # raises SkillError with the exact broken manifest
        log.info("skill library ok at %s (%d skills: %s)", skills.root,
                 len(skills.list()), ", ".join(s.name for s in skills.list()) or "none")

    tts: TtsClient | IrodoriClient | None = None
    if cfg.tts is not None:
        if cfg.tts.engine == "irodori":
            tts = IrodoriClient(cfg.tts.server_url, voice=cfg.tts.voice)
        else:
            tts = TtsClient(cfg.tts.server_url)
        await tts.check_reachable()  # raises TtsError with the exact dependency
        await tts.probe_format()     # pins pcm_s16le/rate or refuses to start
        log.info("tts-server ok at %s (engine %s, pcm_s16le %d Hz mono)",
                 cfg.tts.server_url, cfg.tts.engine, tts.sample_rate)

    db_path = Path(cfg.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = CoreStore(db_path)

    auth_token: str | None = None
    if cfg.server.auth_token_file is not None:
        token_path = Path(cfg.server.auth_token_file).expanduser()
        try:
            auth_token = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"cannot read auth token file {token_path}: {exc}") from exc
        if not auth_token:
            raise ConfigError(f"auth token file {token_path} is empty")

    sessions = SessionManager()
    scheduler = LlmScheduler()
    scheduler.start()
    orchestrator = Orchestrator(store, sessions, scheduler, llama, cfg.llm,
                                asr=asr, skills=skills, tts=tts)
    server = WsServer(cfg.server.bind, cfg.server.port, sessions,
                      orchestrator.handle_command, auth_token,
                      resync_streams=orchestrator.resync_streams)

    await server.start()
    await orchestrator.startup()
    log.info("ghost-runner-core up — protocol v1 on ws://%s:%d",
             cfg.server.bind, cfg.server.port)
    try:
        await asyncio.Event().wait()  # run until cancelled (SIGINT)
    finally:
        await server.stop()
        await scheduler.stop()
        if asr is not None:
            await asr.aclose()
        if tts is not None:
            await tts.aclose()
        store.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ghost Runner core (backend brain)")
    parser.add_argument("--config", required=True, help="path to config.toml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        asyncio.run(run(cfg))
    except (LlmError, AsrError, SkillError, TtsError, ConfigError) as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    cli()
