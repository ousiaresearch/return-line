"""Configuration for The Return Line.

Nothing in this kit assumes an agent name or a home directory. Resolution order for every value:
the plugin's own ``config.json`` (beside this file), then the environment variable, then a default
relative to the agent directory.

``config.json``::

    {
      "agent_dir": "~/.hermes/agents/your-agent",
      "record_dir": "~/agent-record",
      "journal_db": "~/.hermes/agents/your-agent/discord-memory/journal.sqlite",
      "session_db": "~/.hermes/state.db",
      "record_globs": ["texts/*.md", "dreams/*.md", "Story/*.md"],
      "self_names": ["my-agent", "my-agent-alias"],
      "max_state_chars": 1100,
      "max_record_chars": 900,
      "max_record_hits": 4
    }
"""
from __future__ import annotations
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else None


def load() -> dict:
    cfg: dict = {}
    path = HERE / "config.json"
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except Exception as exc:  # a broken config must not break a reply
            cfg = {}
            print(f"[return-line] config.json unreadable ({exc}); using defaults")
    agent = _env("RETURN_LINE_AGENT_DIR") or cfg.get("agent_dir") or str(Path.home() / ".hermes/agents/agent")
    record = _env("RETURN_LINE_RECORD_DIR") or cfg.get("record_dir") or ""
    journal = _env("RETURN_LINE_JOURNAL_DB") or cfg.get("journal_db") or ""
    session = _env("RETURN_LINE_SESSION_DB") or cfg.get("session_db") or str(Path.home() / ".hermes/state.db")
    return {
        "agent_dir": Path(agent).expanduser(),
        "record_dir": Path(record).expanduser() if record else None,
        "journal_db": Path(journal).expanduser() if journal else None,
        "session_db": Path(session).expanduser(),
        "record_globs": cfg.get("record_globs") or ["texts/*.md", "dreams/*.md", "Story/*.md"],
        "self_names": [s.lower() for s in (cfg.get("self_names") or [])],
        "max_state_chars": int(cfg.get("max_state_chars") or 1100),
        "max_record_chars": int(cfg.get("max_record_chars") or 900),
        "max_record_hits": int(cfg.get("max_record_hits") or 4),
    }


CONFIG = load()
