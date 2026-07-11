"""Typed config loader. Unknown or missing keys are startup failures with the
exact key path (§A4.4) — no silent defaults for security-relevant settings.
"""

from __future__ import annotations

import ipaddress
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    pass


@dataclass(slots=True, frozen=True)
class ServerConfig:
    bind: str
    port: int
    auth_token_file: str | None


@dataclass(slots=True, frozen=True)
class LlmConfig:
    base_url: str
    model: str
    system_prompt: str
    history_messages: int


@dataclass(slots=True, frozen=True)
class AsrConfig:
    server_url: str
    language: str


@dataclass(slots=True, frozen=True)
class SkillsConfig:
    dir: str


@dataclass(slots=True, frozen=True)
class TtsConfig:
    server_url: str
    engine: str  # "fish" (S2 Pro api_server) or "irodori" (Irodori-TTS-Server)
    voice: str   # irodori only: reference-voice id ("none" = model default)


@dataclass(slots=True, frozen=True)
class CoreConfig:
    server: ServerConfig
    llm: LlmConfig
    db_path: str
    asr: AsrConfig | None
    skills: SkillsConfig | None
    tts: TtsConfig | None


_KNOWN = {
    "server": {"bind", "port", "auth_token_file"},
    "llm": {"base_url", "model", "system_prompt", "history_messages"},
    "memory": {"db"},
    "asr": {"server_url", "language"},
    "skills": {"dir"},
    "tts": {"server_url", "engine", "voice"},
}

# Japanese-first (PRD §2): the companion speaks JA by default. Honest-embodiment
# rules from behavior design B10 baked into the prompt: no fabricated feelings,
# no manipulation, concise spoken-register replies.
_DEFAULT_SYSTEM_PROMPT = (
    "あなたはデスクトップに常駐するアシスタント「ゴースト」です。"
    "日本語で簡潔に、話し言葉で答えてください。"
    "分からないことは分からないと正直に言い、感情を持っているかのような表現"
    "(「嬉しい」「寂しい」など)は使わないでください。"
    "結果に基づく表現(「うまくいきました」「よかったです」)は使えます。"
)


def _is_loopback(bind: str) -> bool:
    try:
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        return bind == "localhost"


def load_config(path: str | Path) -> CoreConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, "rb") as f:
        raw = tomllib.load(f)

    for section, keys in raw.items():
        if section not in _KNOWN:
            raise ConfigError(f"unknown config section [{section}]")
        if not isinstance(keys, dict):
            raise ConfigError(f"[{section}] must be a table")
        for key in keys:
            if key not in _KNOWN[section]:
                raise ConfigError(f"unknown config key {section}.{key}")

    def require(section: str, key: str) -> object:
        try:
            return raw[section][key]
        except KeyError:
            raise ConfigError(f"missing required config key {section}.{key}") from None

    bind = require("server", "bind")
    port = require("server", "port")
    if not isinstance(bind, str):
        raise ConfigError("server.bind must be a string")
    if not isinstance(port, int):
        raise ConfigError("server.port must be an integer")

    auth_token_file = raw["server"].get("auth_token_file")
    if auth_token_file is not None and not isinstance(auth_token_file, str):
        raise ConfigError("server.auth_token_file must be a string path")
    if not _is_loopback(bind) and auth_token_file is None:
        raise ConfigError(
            f"server.bind={bind!r} is not loopback: server.auth_token_file is required "
            "(plain unauthenticated ws is allowed only on 127.0.0.1 — §A7.6)"
        )

    base_url = require("llm", "base_url")
    model = require("llm", "model")
    if not isinstance(base_url, str) or not isinstance(model, str):
        raise ConfigError("llm.base_url and llm.model must be strings")

    system_prompt = raw["llm"].get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str):
        raise ConfigError("llm.system_prompt must be a string")
    history_messages = raw["llm"].get("history_messages", 20)
    if not isinstance(history_messages, int) or history_messages < 0:
        raise ConfigError("llm.history_messages must be a non-negative integer")

    db_path = raw.get("memory", {}).get("db")
    if db_path is None:
        raise ConfigError("missing required config key memory.db")
    if not isinstance(db_path, str):
        raise ConfigError("memory.db must be a string path")

    asr: AsrConfig | None = None
    if "asr" in raw:
        asr_url = require("asr", "server_url")
        if not isinstance(asr_url, str):
            raise ConfigError("asr.server_url must be a string")
        asr_language = raw["asr"].get("language", "ja")  # Japanese-first (PRD §2)
        if not isinstance(asr_language, str) or not asr_language:
            raise ConfigError("asr.language must be a non-empty string")
        asr = AsrConfig(server_url=asr_url, language=asr_language)

    skills: SkillsConfig | None = None
    if "skills" in raw:
        skills_dir = require("skills", "dir")
        if not isinstance(skills_dir, str) or not skills_dir:
            raise ConfigError("skills.dir must be a non-empty string path")
        skills = SkillsConfig(dir=skills_dir)

    tts: TtsConfig | None = None
    if "tts" in raw:
        tts_url = require("tts", "server_url")
        if not isinstance(tts_url, str) or not tts_url:
            raise ConfigError("tts.server_url must be a non-empty string")
        tts_engine = raw["tts"].get("engine", "fish")  # the first engine shipped
        if tts_engine not in ("fish", "irodori"):
            raise ConfigError(
                f"tts.engine must be 'fish' or 'irodori', got {tts_engine!r}")
        tts_voice = raw["tts"].get("voice", "none")
        if not isinstance(tts_voice, str) or not tts_voice:
            raise ConfigError("tts.voice must be a non-empty string")
        if tts_voice != "none" and tts_engine != "irodori":
            raise ConfigError("tts.voice is only supported by the irodori engine")
        tts = TtsConfig(server_url=tts_url, engine=tts_engine, voice=tts_voice)

    return CoreConfig(
        server=ServerConfig(bind=bind, port=port, auth_token_file=auth_token_file),
        llm=LlmConfig(base_url=base_url, model=model, system_prompt=system_prompt,
                      history_messages=history_messages),
        db_path=db_path,
        asr=asr,
        skills=skills,
        tts=tts,
    )
