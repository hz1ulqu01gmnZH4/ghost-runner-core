"""Skill library + bwrap sandbox tests (F3 slice 1).

These run the REAL bubblewrap sandbox — bwrap starts in milliseconds and the
sandbox's deny-by-default properties are exactly what must not regress. If
bwrap is missing these tests fail loudly rather than skip: an environment
that can't run the sandbox can't validate the core.
"""

import hashlib
import json
from pathlib import Path

import pytest

from ghost_runner_core.config import ConfigError, load_config
from ghost_runner_core.skills.library import MAX_TIMEOUT_S, SkillError, SkillLibrary
from ghost_runner_core.skills.sandbox import (
    LIMIT_AS_BYTES,
    LIMIT_NPROC,
    SkillExecutionError,
    run_skill,
)

REPO_SKILLS = Path(__file__).resolve().parent.parent / "skills"

ECHO_SKILL = """\
import json, sys
print(json.dumps({"echo": json.load(sys.stdin)}, ensure_ascii=False))
"""


def write_skill(root: Path, name: str, code: str, *, timeout_s: float = 5.0,
                sha256: str | None = None, manifest_name: str | None = None,
                drop_key: str | None = None, extra: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "skill.py").write_text(code, encoding="utf-8")
    digest = sha256 if sha256 is not None else hashlib.sha256(code.encode()).hexdigest()
    lines = {
        "name": f'name = "{manifest_name or name}"',
        "version": 'version = "0.1.0"',
        "description": 'description = "test skill"',
        "timeout_s": f"timeout_s = {timeout_s}",
        "sha256": f'sha256 = "{digest}"',
        "provenance": 'provenance = "test"',
    }
    if drop_key is not None:
        del lines[drop_key]
    (d / "manifest.toml").write_text("\n".join(lines.values()) + "\n" + extra,
                                     encoding="utf-8")
    return d


def load_library(root: Path) -> SkillLibrary:
    lib = SkillLibrary(root)
    lib.load()
    return lib


# -- library loading ---------------------------------------------------------


def test_load_and_list(tmp_path):
    write_skill(tmp_path, "alpha", ECHO_SKILL)
    write_skill(tmp_path, "beta", ECHO_SKILL)
    (tmp_path / "README.md").write_text("stray file is fine")
    lib = load_library(tmp_path)
    assert [s.name for s in lib.list()] == ["alpha", "beta"]
    assert lib.get("alpha").timeout_s == 5.0
    assert lib.get("nope") is None


def test_missing_library_dir_fails(tmp_path):
    lib = SkillLibrary(tmp_path / "nonexistent")
    with pytest.raises(SkillError, match="not a directory"):
        lib.load()


def test_hash_mismatch_refuses_to_load(tmp_path):
    write_skill(tmp_path, "evil", ECHO_SKILL, sha256="0" * 64)
    with pytest.raises(SkillError, match="sha256 mismatch"):
        load_library(tmp_path)


def test_unknown_manifest_key_fails(tmp_path):
    write_skill(tmp_path, "s", ECHO_SKILL, extra='network = "all"\n')
    with pytest.raises(SkillError, match="unknown keys.*network"):
        load_library(tmp_path)


def test_missing_manifest_key_fails(tmp_path):
    write_skill(tmp_path, "s", ECHO_SKILL, drop_key="sha256")
    with pytest.raises(SkillError, match="missing keys.*sha256"):
        load_library(tmp_path)


def test_name_directory_mismatch_fails(tmp_path):
    write_skill(tmp_path, "actual_dir", ECHO_SKILL, manifest_name="other_name")
    with pytest.raises(SkillError, match="does not match directory"):
        load_library(tmp_path)


@pytest.mark.parametrize("timeout_s", [0, -1, MAX_TIMEOUT_S + 0.1])
def test_timeout_out_of_bounds_fails(tmp_path, timeout_s):
    write_skill(tmp_path, "s", ECHO_SKILL, timeout_s=timeout_s)
    with pytest.raises(SkillError, match="timeout_s"):
        load_library(tmp_path)


def test_malformed_sha256_fails(tmp_path):
    write_skill(tmp_path, "s", ECHO_SKILL, sha256="not-a-hash")
    with pytest.raises(SkillError, match="64 hex"):
        load_library(tmp_path)


def test_missing_code_file_fails(tmp_path):
    d = write_skill(tmp_path, "s", ECHO_SKILL)
    (d / "skill.py").rename(d / "gone.py")
    with pytest.raises(SkillError, match="does not exist"):
        load_library(tmp_path)


# -- sandbox execution -------------------------------------------------------


async def test_run_roundtrips_japanese_args(tmp_path):
    write_skill(tmp_path, "echo", ECHO_SKILL)
    lib = load_library(tmp_path)
    result = await run_skill(lib, lib.get("echo"), {"text": "こんにちは、世界"})
    assert result == {"echo": {"text": "こんにちは、世界"}}


async def test_tamper_after_load_refuses_to_run(tmp_path):
    d = write_skill(tmp_path, "echo", ECHO_SKILL)
    lib = load_library(tmp_path)
    (d / "skill.py").write_text(ECHO_SKILL + "# rug-pull\n", encoding="utf-8")
    with pytest.raises(SkillError, match="changed on disk"):
        await run_skill(lib, lib.get("echo"), {})


async def test_nonzero_exit_surfaces_stderr(tmp_path):
    code = 'import sys; print("boom reason", file=sys.stderr); sys.exit(3)\n'
    write_skill(tmp_path, "dies", code)
    lib = load_library(tmp_path)
    with pytest.raises(SkillExecutionError, match="exited 3.*boom reason"):
        await run_skill(lib, lib.get("dies"), {})


async def test_bwrap_stderr_is_operator_error_not_skill_error(tmp_path):
    """stderr starting with "bwrap:" means the sandbox itself failed to set up
    (an operator problem), and must surface as SkillError (-> unavailable),
    never as a skill crash (-> internal)."""
    code = 'import sys; print("bwrap: setting up uid map: ...", file=sys.stderr); sys.exit(1)\n'
    write_skill(tmp_path, "brokenbox", code)
    lib = load_library(tmp_path)
    with pytest.raises(SkillError, match="sandbox failed"):
        await run_skill(lib, lib.get("brokenbox"), {})


async def test_non_json_output_fails(tmp_path):
    write_skill(tmp_path, "garbage", 'print("not json")\n')
    lib = load_library(tmp_path)
    with pytest.raises(SkillExecutionError, match="invalid JSON"):
        await run_skill(lib, lib.get("garbage"), {})


async def test_non_object_output_fails(tmp_path):
    write_skill(tmp_path, "listy", 'print("[1, 2]")\n')
    lib = load_library(tmp_path)
    with pytest.raises(SkillExecutionError, match="JSON object"):
        await run_skill(lib, lib.get("listy"), {})


async def test_oversized_output_fails(tmp_path):
    code = 'import json; print(json.dumps({"pad": "x" * 2_000_000}))\n'
    write_skill(tmp_path, "bloat", code)
    lib = load_library(tmp_path)
    with pytest.raises(SkillExecutionError, match="capped"):
        await run_skill(lib, lib.get("bloat"), {})


async def test_hung_skill_times_out(tmp_path):
    write_skill(tmp_path, "hangs", "import time; time.sleep(60)\n", timeout_s=0.5)
    lib = load_library(tmp_path)
    with pytest.raises(SkillExecutionError, match="timed out"):
        await run_skill(lib, lib.get("hangs"), {})


async def test_sandbox_denies_network_home_and_writes(tmp_path):
    """The deny-by-default contract itself: a skill must not reach the network,
    see the real filesystem, or write outside its private /tmp."""
    probe = """\
import json, socket, os
report = {}
try:
    socket.create_connection(("127.0.0.1", 8080), timeout=2)
    report["network"] = "REACHABLE"
except OSError:
    report["network"] = "blocked"
report["home_visible"] = os.path.exists(os.path.expanduser("~ak")) or os.path.exists("/home")
try:
    open("/skill/pwned", "w")
    report["skill_dir_writable"] = True
except OSError:
    report["skill_dir_writable"] = False
# PATH/HOME/LANG are set by our sandbox; PWD by bwrap --chdir. Anything else leaked from the host.
report["env_leak"] = [k for k in os.environ if k not in ("PATH", "HOME", "LANG", "PWD")]
print(json.dumps(report))
"""
    write_skill(tmp_path, "probe", probe)
    lib = load_library(tmp_path)
    report = await run_skill(lib, lib.get("probe"), {})
    assert report == {"network": "blocked", "home_visible": False,
                      "skill_dir_writable": False, "env_leak": []}


async def test_rlimits_are_applied_inside_sandbox(tmp_path):
    """prlimit must actually constrain the skill process (QA finding: bwrap
    isolates namespaces, not resource consumption)."""
    probe = """\
import json, resource
print(json.dumps({
    "as": resource.getrlimit(resource.RLIMIT_AS)[0],
    "nproc": resource.getrlimit(resource.RLIMIT_NPROC)[0],
    "cpu": resource.getrlimit(resource.RLIMIT_CPU)[0],
}))
"""
    write_skill(tmp_path, "limits", probe, timeout_s=5.0)
    lib = load_library(tmp_path)
    report = await run_skill(lib, lib.get("limits"), {})
    assert report == {"as": LIMIT_AS_BYTES, "nproc": LIMIT_NPROC, "cpu": 7}  # int(5)+2


async def test_allocation_bomb_dies_inside_sandbox(tmp_path):
    """A skill trying to allocate past RLIMIT_AS must die with MemoryError in
    its own process, not grow the host."""
    code = "big = bytearray(600 * 1024 * 1024)\n"
    write_skill(tmp_path, "bomb", code)
    lib = load_library(tmp_path)
    with pytest.raises(SkillExecutionError, match="(?s)exited 1.*MemoryError"):
        await run_skill(lib, lib.get("bomb"), {})


async def test_stderr_flood_does_not_hang_or_fail(tmp_path):
    """stderr past its cap is discarded (never buffered, never blocks the
    skill on a full pipe); the result still comes back."""
    code = ('import json, sys\n'
            'sys.stderr.write("x" * 5_000_000)\n'
            'print(json.dumps({"ok": True}))\n')
    write_skill(tmp_path, "noisy", code)
    lib = load_library(tmp_path)
    assert await run_skill(lib, lib.get("noisy"), {}) == {"ok": True}


# -- the shipped demo skills -------------------------------------------------


async def test_repo_skills_load_and_hashes_match():
    """Guards the checked-in library: editing a skill.py without updating its
    manifest sha256 must fail here, not at core startup."""
    lib = load_library(REPO_SKILLS)
    assert {s.name for s in lib.list()} == {"jp_calendar", "text_stats"}


async def test_jp_calendar_is_correct():
    lib = load_library(REPO_SKILLS)
    result = await run_skill(lib, lib.get("jp_calendar"), {"date": "2026-07-04"})
    assert result == {"date": "2026-07-04", "era": "令和", "era_year": 8,
                      "wareki": "令和8年7月4日", "weekday": "土曜日"}
    first = await run_skill(lib, lib.get("jp_calendar"), {"date": "2019-05-01"})
    assert first["wareki"] == "令和元年5月1日"
    heisei = await run_skill(lib, lib.get("jp_calendar"), {"date": "1989-01-08"})
    assert heisei["era"] == "平成" and heisei["era_year"] == 1
    with pytest.raises(SkillExecutionError, match="exited 1"):
        await run_skill(lib, lib.get("jp_calendar"), {"date": "not-a-date"})


async def test_text_stats_is_correct():
    lib = load_library(REPO_SKILLS)
    result = await run_skill(lib, lib.get("text_stats"),
                             {"text": "猫がニャーと鳴いた。\nCat!"})
    assert result == {"chars": 15, "chars_no_space": 14, "lines": 2,
                      "hiragana": 4, "katakana": 3, "kanji": 2}


# -- config ------------------------------------------------------------------


BASE_CONFIG = """\
[server]
bind = "127.0.0.1"
port = 8790

[llm]
base_url = "http://127.0.0.1:8080/v1"
model = "test"

[memory]
db = "/tmp/test.db"
"""


def test_config_without_skills_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG)
    assert load_config(p).skills is None


def test_config_with_skills_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[skills]\ndir = "/somewhere/skills"\n')
    assert load_config(p).skills.dir == "/somewhere/skills"


def test_config_skills_missing_dir_fails(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + "\n[skills]\n")
    with pytest.raises(ConfigError, match="skills.dir"):
        load_config(p)


def test_config_skills_unknown_key_fails(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(BASE_CONFIG + '\n[skills]\ndir = "/x"\nallow_network = true\n')
    with pytest.raises(ConfigError, match="unknown config key skills.allow_network"):
        load_config(p)
