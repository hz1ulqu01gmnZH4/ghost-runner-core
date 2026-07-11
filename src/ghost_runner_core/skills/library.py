"""Skill library: git-tracked directory of sandboxed, content-hashed tools.

First slice of F3 (self-extending tools, PRD §6). A skill is a directory:

    <library>/<name>/manifest.toml   name, version, description, timeout_s,
                                     sha256 (of skill.py), provenance
    <library>/<name>/skill.py        reads JSON args on stdin, writes a JSON
                                     object result on stdout, exits 0

The sha256 in the manifest is the rug-pull guard from the tool-pipeline
research: a skill whose code no longer matches its manifest never runs.
Loading is strict and startup-fatal (§A4.4): one bad manifest means the
operator fixes the library, not a core that silently serves a subset.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# skill.run is awaited inline in the session recv loop (like asr.transcribe),
# so a skill must fail before the client's heartbeat gives up on the link
# (2 missed pings × 15 s).
MAX_TIMEOUT_S = 20.0

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {"name", "version", "description", "timeout_s", "sha256", "provenance"}


class SkillError(Exception):
    """Library is invalid (bad manifest, hash mismatch, missing sandbox)."""


@dataclass(slots=True, frozen=True)
class Skill:
    name: str
    version: str
    description: str
    timeout_s: float
    sha256: str
    provenance: str
    path: Path  # directory containing manifest.toml + skill.py

    @property
    def code_path(self) -> Path:
        return self.path / "skill.py"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(skill_dir: Path) -> Skill:
    manifest_path = skill_dir / "manifest.toml"
    where = f"skill manifest {manifest_path}"
    try:
        with open(manifest_path, "rb") as f:
            raw = tomllib.load(f)
    except OSError as exc:
        raise SkillError(f"{where}: cannot read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SkillError(f"{where}: invalid TOML: {exc}") from exc

    unknown = set(raw) - _MANIFEST_KEYS
    if unknown:
        raise SkillError(f"{where}: unknown keys {sorted(unknown)}")
    missing = _MANIFEST_KEYS - set(raw)
    if missing:
        raise SkillError(f"{where}: missing keys {sorted(missing)}")

    for key in ("name", "version", "description", "sha256", "provenance"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise SkillError(f"{where}: {key} must be a non-empty string")
    timeout_s = raw["timeout_s"]
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise SkillError(f"{where}: timeout_s must be a number")
    timeout_s = float(timeout_s)
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise SkillError(
            f"{where}: timeout_s must be in (0, {MAX_TIMEOUT_S}] (got {timeout_s})")

    sha256 = raw["sha256"].lower()
    if not _SHA256_RE.match(sha256):
        raise SkillError(f"{where}: sha256 must be 64 hex characters, got {raw['sha256']!r}")

    name = raw["name"]
    if not _NAME_RE.match(name):
        raise SkillError(f"{where}: name {name!r} must match {_NAME_RE.pattern}")
    if name != skill_dir.name:
        raise SkillError(
            f"{where}: name {name!r} does not match directory name {skill_dir.name!r}")

    skill = Skill(name=name, version=raw["version"], description=raw["description"],
                  timeout_s=timeout_s, sha256=sha256,
                  provenance=raw["provenance"], path=skill_dir)

    if not skill.code_path.is_file():
        raise SkillError(f"{where}: {skill.code_path} does not exist")
    actual = _sha256_file(skill.code_path)
    if actual != skill.sha256:
        raise SkillError(
            f"{where}: sha256 mismatch — manifest says {skill.sha256}, "
            f"skill.py is {actual}; refusing to load tampered/stale skill")
    return skill


class SkillLibrary:
    """All skills under one root directory, validated eagerly at load()."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._skills: dict[str, Skill] = {}

    @property
    def root(self) -> Path:
        return self._root

    def load(self) -> None:
        if not self._root.is_dir():
            raise SkillError(f"skill library {self._root} is not a directory")
        skills: dict[str, Skill] = {}
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue  # stray files (e.g. README.md) beside skill dirs are fine
            skills[entry.name] = _load_manifest(entry)
        self._skills = skills

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def verify_hash(self, skill: Skill) -> None:
        """Re-check right before every run: the library sits on disk and could
        have been rewritten since load() (rug-pull guard, tool-pipeline research §5)."""
        try:
            actual = _sha256_file(skill.code_path)
        except OSError as exc:
            raise SkillError(f"skill {skill.name}: cannot read {skill.code_path}: {exc}") from exc
        if actual != skill.sha256:
            raise SkillError(
                f"skill {skill.name}: skill.py changed on disk since load "
                f"(manifest {skill.sha256}, now {actual}); refusing to run")
