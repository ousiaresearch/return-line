#!/usr/bin/env python3
"""Self-test for The Return Line: proves the plugin works standalone, on a neutral agent.

Builds a throwaway agent directory with a real brain-state (using the mind kit's generator when
it is present next to this kit), points the plugin at it, and exercises the three things the kit
promises: the state block arrives, the record is searchable, and the turn ledger is written.

    python3 scripts/self_test.py
"""
from __future__ import annotations
import importlib.util, json, os, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
PLUG = KIT / "return-line"
MIND_KIT = KIT.parent / "ousia-mind-kit"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    results.append((bool(ok), label))


def leaked_identity(text: str, names: list[str]) -> str | None:
    """The two failure modes worth asserting on a shipped kit.

    1. an absolute user path from the machine the kit was lifted out of
    2. this agent's own name baked into source rather than read from config
    """
    for probe in ("/Users/", "/home/"):
        if probe in text:
            return f"absolute path {probe!r}"
    low = text.lower()
    for name in names:
        if name and name.lower() in low:
            return f"baked-in agent name {name!r}"
    return None


def build_agent(tmp: Path) -> Path:
    """A neutral agent with a state file, a record, and a journal."""
    agent = tmp / "example-agent"
    (agent / "record" / "texts").mkdir(parents=True)
    (agent / "discord-memory").mkdir(parents=True)

    if (MIND_KIT / "scripts/generate-brain-state.py").exists():
        (agent / "brain").mkdir(parents=True, exist_ok=True)
        for example in (MIND_KIT / "brain").rglob("*.example.json"):
            dest = agent / "brain" / str(example.relative_to(MIND_KIT / "brain")).replace(".example.json", ".json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(example, dest)
        subprocess.run([sys.executable, str(MIND_KIT / "scripts/generate-brain-state.py")],
                       env={**os.environ, "MIND_AGENT_DIR": str(agent)},
                       capture_output=True, text=True, timeout=120)
    if not (agent / "brain-state.json").exists():   # mind kit absent: a minimal honest state
        (agent / "brain-state.json").write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00",
            "health_summary": {"healthy": 2, "total": 2},
            "systems": {"somatic": {"valence": 0.6, "energy": 0.5, "tension": 0.2, "gut": "go_ahead",
                                    "age_h": 0.1, "max_age_h": 8},
                        "fatigue": {"level": 0.2, "sustainable": True, "age_h": 0.1, "max_age_h": 24},
                        "SCN": {"phase": "morning", "age_h": 0.1, "max_age_h": 24}}}, indent=2))

    (agent / "record" / "texts" / "2026-01-01.md").write_text(
        "## Morning\nWe agreed the ledger matters more than the prose, because a ledger can be audited and prose cannot.\n\n"
        "## Evening\nReread the route notes and found the same three questions still sitting unanswered.\n")
    (agent / "record" / "dreams").mkdir(parents=True, exist_ok=True)
    (agent / "record" / "dreams" / "a-dream.md").write_text(
        "## 03:10\nA bridge made of receipts, and someone on the far side checking each one against the crossing list.\n")

    j = sqlite3.connect(agent / "discord-memory" / "journal.sqlite")
    j.execute("create table messages (created_at text, author_name text, content text)")
    j.executemany("insert into messages values (?,?,?)", [
        ("2026-01-01T10:00:00+00:00", "operator", "does the kit actually run without me?"),
        ("2026-01-01T10:01:00+00:00", "agent", "it does; the ledger says so."),
    ])
    j.commit(); j.close()

    (PLUG / "config.json").write_text(json.dumps({
        "agent_dir": str(agent), "record_dir": str(agent / "record"),
        "journal_db": str(agent / "discord-memory" / "journal.sqlite"),
        "session_db": str(agent / "no-such-state.db"),
        "self_names": ["example-agent"],
    }, indent=2))
    return agent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="return-line-selftest-"))
    cfg = PLUG / "config.json"
    cfg_backup = cfg.read_text() if cfg.exists() else None
    try:
        agent = build_agent(tmp)

        # 1. the plugin imports without a host, and resolves the configured agent
        mod = load_module("return_line_under_test", PLUG / "__init__.py")
        check(getattr(mod, "AGENT", None) == agent, "plugin resolved agent_dir from config.json")

        # 2. the state block is built, and carries the generic marker
        block = mod.build_state_block()
        check(block.startswith("<agent-state>") and block.rstrip().endswith("</agent-state>"),
              "state block is emitted with the <agent-state> marker")
        check("body:" in block and "phase" in block,
              "state block carries body readings and circadian phase")

        # 3. the record is indexed and searchable
        rec = load_module("return_line_recall_under_test", PLUG / "recall.py")
        stats = rec.build_index(force=True)
        check(stats.get("journal", 0) >= 2 and stats.get("texts", 0) >= 2,
              f"index built over journal + record (got {stats})")
        hits = rec.search("ledger matters", k=3, min_score=0.01)
        check(bool(hits), f"search returns the record for a related query (got {len(hits)} hits)")
        check(leaked_identity(json.dumps(hits), []) is None,
              "retrieved excerpts carry no absolute paths")

        # 4. the hook contract: context is returned as a dict, and never raises
        out = mod.on_pre_llm_call(platform="discord",
                                  user_message="what did we agree about the ledger?")
        check(isinstance(out, dict) and "context" in out, "pre_llm_call returns {'context': ...}")
        check(out is None or leaked_identity(json.dumps(out), []) is None,
              "injected context carries no absolute paths")
        check(mod.on_pre_llm_call(platform="subagent", user_message="ignored") is None,
              "subagent turns are skipped")

        # 5. the write-back: one raw line per turn, silence recorded as silence
        mod.on_post_llm_call(platform="discord", turn_id="t1", session_id="s1",
                             user_message="hello", assistant_response="present.", model="test")
        mod.on_post_llm_call(platform="discord", turn_id="t2", session_id="s1",
                             user_message="anyone there?", assistant_response="SILENT", model="test")
        ledger = agent / "turn-ledger.jsonl"
        rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
        check(len(rows) == 2, f"turn ledger wrote one line per turn (got {len(rows)})")
        check(rows and rows[0]["outbound"] == "present." and rows[1]["silent"] is True,
              "ledger records the reply and marks a silence token as silence")
        mod.on_post_llm_call(platform="discord", turn_id="t1", session_id="s1",
                             user_message="hello", assistant_response="present.")
        check(len(ledger.read_text().splitlines()) == 2, "a repeated turn id is not double-written")

        # 6. the shipped source carries no residue from the tree it came from
        blob = "\n".join(p.read_text(errors="replace") for p in PLUG.glob("*.py"))
        names = []
        if cfg.exists():
            try:
                names = json.loads(cfg.read_text()).get("self_names") or []
            except Exception:
                names = []
        leak = leaked_identity(blob, [])
        check(leak is None,
              f"shipped plugin source carries no identity residue{'' if leak is None else f' ({leak})'}")
        check(all(not n or n.lower() not in blob.lower() for n in names),
              "the configured agent name is not baked into the source")
    finally:
        if cfg_backup is not None:
            cfg.write_text(cfg_backup)
        elif cfg.exists():
            cfg.unlink()
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [label for ok, label in results if not ok]
    for ok, label in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
