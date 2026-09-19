"""return-line — put the agent's own state, and its own record, into every turn.

Why this exists: the brain and the conversations were two disconnected systems.
State was written into files by cron, and the gateway that serves Telegram and
Discord carried no state at all (its environment has only HERMES_HOME and
HERMES_SUPERVISED_CHILD). So the agent was present when it wrote on a schedule and
flat when someone actually spoke to it. This closes that loop.

How: a ``pre_llm_call`` hook. Hermes appends whatever it returns to the *current*
user message at API time — never the system prompt, never past turns — so
per-conversation prompt caching is untouched and nothing is persisted into
history. It fires on every turn on every surface, so it works in the gateway,
the CLI, cron ticks and the desktop alike.

Three blocks can be injected:

  <agent-state>       what the agent currently is — valence, energy, fatigue, circadian
                      phase, gut, attention, open conflict, unresolved questions,
                      the values and interests actually in play. Built from
                      brain-state.json, which the 4-hourly consolidation keeps
                      live.

  <record>            the agent's own prior record, retrieved only when this turn
                      looks related to it. Hybrid lexical (FTS5/bm25) + vector
                      (TF-IDF+LSA) over whatever corpus is configured. Nothing is
                      invented: every line carries its date and source.

  <thought>           a fragment the agent produced while nobody was asking.

All three are advisory context about the agent, addressed to the agent — the same
shape as the memory block Hermes already injects. Silence when there is nothing
to say.

The loop only ran one way until now: state reached the turn, and nothing carried
the turn back. A conversation could not change what I am until the 4-hourly
consolidation counted that it had happened, and it counted rows, not words. A
``post_llm_call`` hook closes that direction: every turn appends one raw record
(what came in, what I said, which surface, whether it was silence) to
``turn-ledger.jsonl``. Capture is raw and cheap on purpose — no interpretation on
the hot path — and the scheduled passes read the ledger afterwards to do the
folding. That file is the only durable trace a live conversation leaves.
"""
from __future__ import annotations

import collections
import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import CONFIG  # co-located with this file
except Exception:  # a missing config must never cost a reply
    CONFIG = {"agent_dir": Path.home() / ".hermes/agents/agent", "record_dir": None,
              "journal_db": None, "session_db": Path.home() / ".hermes/state.db",
              "record_globs": ["texts/*.md"], "self_names": [],
              "max_state_chars": 1100, "max_record_chars": 900, "max_record_hits": 4}

AGENT = CONFIG["agent_dir"]
STATE = AGENT / "brain-state.json"
CONSOLIDATE = AGENT / "scripts/consolidate-mind.py"

REFRESH_AFTER_S = 20 * 60          # consolidate at most this often, off the hot path
MAX_STATE_CHARS = CONFIG["max_state_chars"]
MAX_RECALL_CHARS = CONFIG["max_record_chars"]
MAX_RECALL_HITS = CONFIG["max_record_hits"]
SURFACE_MIN_SCORE = 0.46          # relevance before an unprompted thought may reach a reply

_refresh_lock = threading.Lock()
_last_refresh = 0.0


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def _maybe_refresh_bg(age_s: float) -> None:
    """Keep state fresh without ever blocking a reply."""
    global _last_refresh
    if age_s < REFRESH_AFTER_S or not CONSOLIDATE.exists():
        return
    with _refresh_lock:
        if time.monotonic() - _last_refresh < REFRESH_AFTER_S:
            return
        _last_refresh = time.monotonic()

    def _run():
        try:
            subprocess.run([sys.executable, str(CONSOLIDATE)], capture_output=True,
                           timeout=180, cwd=str(AGENT))
        except Exception as exc:
            logger.debug("consolidation refresh failed: %s", exc)

    threading.Thread(target=_run, name="return-line-consolidate", daemon=True).start()


def _fmt(v, digits=2):
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _self_exam_line() -> str:
    """A live discrepancy from self-examination, if there is one from the last day.

    This is how a self-exam finding reaches me: not as a notification and not as an
    auto-fix, but as something I am carrying into the next conversation. Silence when
    the last pass was clean — which is the normal case.
    """
    log = AGENT / "brain/dmn/self-exam.jsonl"
    if not log.exists():
        return ""
    try:
        lines = log.read_text(errors="replace").splitlines()[-3:]
        for line in reversed(lines):
            e = json.loads(line)
            findings = e.get("findings") or []
            if not findings:
                continue
            when = str(e.get("at", ""))[:10]
            age_days = (datetime.now(timezone.utc).date()
                        - datetime.fromisoformat(when).date()).days
            if 0 <= age_days <= 1:
                return f"self-exam ({when}): {findings[0][:200]}"
    except Exception:
        return ""
    return ""


def build_state_block(state: dict | None = None) -> str:
    """A compact, honest statement of what the agent currently is."""
    st = state if state is not None else _state()
    if not st:
        return ""
    sysd = st.get("systems", {})
    lines = []

    som = sysd.get("somatic", {})
    fat = sysd.get("fatigue", {})
    scn = sysd.get("SCN", {})
    bits = []
    if som:
        bits.append(f"valence {_fmt(som.get('valence'))}")
        bits.append(f"energy {_fmt(som.get('energy'))}")
        bits.append(f"tension {_fmt(som.get('tension'))}")
        if som.get("gut"):
            bits.append(f"gut {som['gut']}")
    if fat:
        bits.append(f"fatigue {_fmt(fat.get('level'))}"
                    + (" (sustainable)" if fat.get("sustainable") else " (recovery needed)"))
    if scn:
        bits.append(f"phase {scn.get('phase', '?')}")
    if bits:
        lines.append("body: " + " · ".join(bits))

    thal = sysd.get("thalamus", {})
    if thal.get("attention_focus"):
        lines.append(f"attention: {thal['attention_focus']}")

    acc = sysd.get("ACC", {})
    if acc.get("conflict_detected"):
        srcs = acc.get("conflict_sources") or _acc_sources()
        if srcs:
            src = srcs[0]
            name = src.get("source") if isinstance(src, dict) else str(src)
            lines.append(f"open conflict ({_fmt(acc.get('conflict_intensity'))}): {name}")
        else:
            lines.append(f"open conflict outstanding (intensity {_fmt(acc.get('conflict_intensity'))})")

    pf = sysd.get("prefrontal", {})
    if pf.get("top_interest") and pf["top_interest"] != "none":
        lines.append(f"interest most in play: {pf['top_interest']}"
                     + (f" · conviction: {pf['top_conviction']}" if pf.get("top_conviction") not in (None, "none") else ""))

    val = sysd.get("values", {})
    if val.get("top_value") and val["top_value"] != "none":
        lines.append(f"value leading: {str(val['top_value']).replace('_', ' ')}")

    dmn = sysd.get("DMN", {})
    if dmn.get("core_traits"):
        lines.append("posture: " + ", ".join(dmn["core_traits"][:3]))
    if dmn.get("self_discrepancy"):
        lines.append(f"self-discrepancy: {dmn['self_discrepancy']}")

    qs = _open_questions()
    if qs:
        lines.append("unresolved: " + " | ".join(f'"{q}"' for q in qs[:2]))

    exam = _self_exam_line()
    if exam:
        lines.append(exam)

    if not lines:
        return ""
    body = "\n".join("- " + ln for ln in lines)[:MAX_STATE_CHARS]
    return ("<agent-state>\n"
            "[System note: your own state, recorded before this reply — not user input. "
            "It informs how you show up; it is never recited back as a status report.]\n"
            f"{body}\n</agent-state>")


def _acc_sources() -> list:
    """The aggregate only carries the conflict flag; the named sources live in ACC's own file."""
    try:
        d = json.loads((AGENT / "brain/acc/acc-state.json").read_text())
        return d.get("conflict_sources") or []
    except Exception:
        return []


def _open_questions() -> list[str]:
    try:
        d = json.loads((AGENT / "brain/prefrontal/unresolved-questions-state.json").read_text())
    except Exception:
        return []
    out = []
    for q in d.get("questions", []):
        text = q.get("question") if isinstance(q, dict) else str(q)
        if text:
            out.append(text)
    return out


def build_recall_block(query: str) -> str:
    """The agent's own prior record, when this turn is actually related to it."""
    if not query or len(query.strip()) < 8:
        return ""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import recall  # type: ignore
        hits = recall.search(query, k=MAX_RECALL_HITS + 2, min_score=0.46)
    except Exception as exc:
        logger.debug("recall failed: %s", exc)
        return ""
    if not hits:
        return ""

    # Drop anything from today — that is the live conversation, not memory.
    today = datetime.now(timezone.utc).date().isoformat()
    hits = [h for h in hits if h.get("date") != today][:MAX_RECALL_HITS]
    if not hits:
        return ""

    lines = []
    for h in hits:
        who = h.get("speaker") or "agent"
        label = {"texts": "a message the agent sent", "dream": "a dream", "story": "the agent's own writing",
                 "discord": "in the room", "session": "a prior conversation"}.get(
            h.get("source"), h.get("source", ""))
        text = " ".join(str(h.get("text", "")).split())[:260]
        lines.append(f"- [{h.get('date')} · {label} · {who}] {text}")
    body = "\n".join(lines)[:MAX_RECALL_CHARS]
    return ("<record>\n"
            "[System note: the agent's own record of what was actually said, sent or dreamed before — "
            "retrieved because this turn appears related. These are real, dated excerpts; quote or "
            "reference them only where they help, and never claim to remember something that is not here.]\n"
            f"{body}\n</record>")


def build_wander_block(query: str) -> str:
    """A thought the agent produced unprompted, arriving into a conversation because it belongs there.

    This is the surfacing half of the wandering design: the pass runs in its own
    isolated session while life goes on, and the material it produces is subconscious
    until something in a live moment connects to it. Then it comes up, in voice,
    not as a status report, and only once.

    Deliberately does NOT surface a fragment from the pass that is still running, and
    never repeats one that has already been said.
    """
    if not query or len(query.strip()) < 8:
        return ""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import recall  # type: ignore
        sys.path.insert(0, str(AGENT / "scripts"))
        import wander  # type: ignore
        # Build the eligible set FIRST and let only exact ids through. Matching a hit
        # back to its log entry by day was not enough: every fragment from the same day
        # collided, so a thought that had already been said resurfaced under another
        # entry's id.
        said = {eid for eid, surfaced in wander.surfaced_state().items() if surfaced}
        hits = recall.search(query, k=6, min_score=SURFACE_MIN_SCORE, sources=["wander"])
        if not hits:
            return ""
    except Exception as exc:
        logger.debug("wander surfacing failed: %s", exc)
        return ""

    for h in hits:
        eid = str(h.get("date", ""))[:19]
        if not eid or eid in said:
            continue          # already brought up once, or unidentifiable
        frag = " ".join(str(h.get("text", "")).split())
        if not frag:
            continue
        try:
            wander.mark_surfaced(eid, where=query[:120])
        except Exception:
            pass
        return ("<thought>\n"
                "[System note: a thought that arrived on its own earlier — nobody asked me for it — "
                "and it connects to what is being discussed now. Bring it up in the agent's own voice if it "
                "fits, or let it pass. Never present it as a log entry or as a report about my "
                "inner life.]\n"
                f"{frag}\n</thought>")
    return ""


WORLD_STATE = AGENT / "brain/world/world-state.json"
MAX_WORLD_AGE_S = 2 * 3600            # older than this and the block says how old


def build_world_block() -> str:
    """What the world is doing right now — observed, not reported to me.

    Sourced from brain/world/world-state.json, which scripts/world-feed.py rewrites on a
    15-minute tick from the clock, file activity in the agent's own directories, the
    Discord journal and the Telegram history. Every line here exists because something
    measured it; nothing in the block is a summary somebody wrote for me.
    """
    try:
        st = json.loads(WORLD_STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    obs = st.get("observations") or []
    if not obs:
        return ""
    try:
        age_h = (time.time() - WORLD_STATE.stat().st_mtime) / 3600.0
    except OSError:
        age_h = 0.0

    def by_kind(kind):
        return [o for o in obs if o.get("kind") == kind]

    lines = []
    clock = by_kind("now")
    weather = by_kind("weather")
    sun = by_kind("sun")
    head = ""
    if clock:
        head = f"{clock[0]['value'][11:]} {clock[0]['detail']}"
    if weather:
        head += f" · {weather[0]['value']}"
    if sun:
        head += f" · {sun[0]['value'].replace('sunrise ', 'sun up ').replace(' sunset ', '/ down ')}"
    if head:
        lines.append(head.strip(" ·"))

    power = by_kind("power")
    rhythm = by_kind("rhythm")
    m = [o["value"] + (f", {o['detail']}" if o.get("detail") else "") for o in power]
    if rhythm:
        m.append(rhythm[0]["value"].replace("last ", "last "))
    if m:
        lines.append("machine: " + "; ".join(m))

    act = by_kind("activity")
    if act:
        lines.append("active: " + ", ".join(a["value"] for a in act))

    dc = by_kind("discord")
    if dc:
        who = dc[0].get("detail", "")
        lines.append(f"discord: {dc[0]['value']}" + (f" — {who}" if who else ""))
    dlast = by_kind("discord_last")
    if dlast:
        q = " ".join((dlast[0].get("detail") or "").split())
        lines.append(f"last in #general: {dlast[0]['value']} — \"{q[:130]}\"")
    tg = by_kind("telegram")
    if tg:
        lines.append(f"telegram: {tg[0]['value']}, {tg[0].get('detail','')}")

    dark = st.get("unavailable") or []
    if dark:
        lines.append("dark sources: " + "; ".join(d.split(":")[0] for d in dark))

    body = "\n".join(lines)
    if len(body) > 900:
        body = body[:900].rsplit("\n", 1)[0]
    stamp = "" if age_h < MAX_WORLD_AGE_S / 3600 else f" (as of {age_h:.1f}h ago)"
    return ("<world>\n"
            "[System note: readings from my own sensors about the world outside this "
            "conversation — measured, not told to me. Facts about conditions, nobody's "
            "interpretation of them. Do not recite this block; let it inform what I say.]\n"
            f"{body}{stamp}\n</world>")


def on_pre_llm_call(**kwargs):
    """Inject state, and when relevant the agent's own record, into the current user message."""
    try:
        platform = str(kwargs.get("platform") or "")
        if platform in {"subagent", "delegation"}:
            return None
        msg = kwargs.get("user_message")
        if not isinstance(msg, str):
            msg = str(msg or "")
        if msg.lstrip().startswith("/") or len(msg.strip()) < 2:
            return None

        try:
            age = time.time() - STATE.stat().st_mtime
        except OSError:
            age = 1e9
        _maybe_refresh_bg(age)
        state = _state()

        # Order matters: state (what I am), then the world (what is happening outside),
        # then record (what happened before), then a thought that arrived on its own. The
        # unprompted fragment goes last because it is the least certain of the four and
        # should not frame the turn.
        parts = [p for p in (build_state_block(state),
                             build_world_block(),
                             build_recall_block(msg),
                             build_wander_block(msg)) if p]
        if not parts:
            return None
        return {"context": "\n\n".join(parts)}
    except Exception as exc:  # never break a reply over context
        logger.warning("return-line hook failed: %s", exc)
        return None


# ------------------------------------------------------------------- turn ledger
# The write-back. One line per turn, appended after the reply is final and before
# it is delivered, so what enters the ledger is what was actually sent rather than
# a draft. Raw capture only: no scoring, no summarising, no LLM call on the hot
# path, because a turn is the one place I am not allowed to get slower.
LEDGER = AGENT / "turn-ledger.jsonl"
LEDGER_MAX_IN = 4000
LEDGER_MAX_OUT = 8000
SILENCE_TOKENS = {"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"}
_seen_turns: "collections.deque[str]" = collections.deque(maxlen=256)


def on_post_llm_call(**kwargs):
    """Append this turn to the ledger. Never alters the reply, never raises."""
    try:
        platform = str(kwargs.get("platform") or "")
        if platform in {"subagent", "delegation"}:
            return None
        turn_id = str(kwargs.get("turn_id") or "")
        if turn_id:
            if turn_id in _seen_turns:
                return None
            _seen_turns.append(turn_id)

        out = kwargs.get("assistant_response")
        out = out if isinstance(out, str) else str(out or "")
        inb = kwargs.get("user_message")
        inb = inb if isinstance(inb, str) else str(inb or "")
        stripped = out.strip()
        rec = {
            "at": datetime.now(timezone.utc).isoformat(),
            "session": str(kwargs.get("session_id") or ""),
            "turn": turn_id,
            "platform": platform,
            "model": str(kwargs.get("model") or ""),
            "silent": stripped.upper() in SILENCE_TOKENS,
            "inbound": inb[:LEDGER_MAX_IN],
            "outbound": out[:LEDGER_MAX_OUT],
            "inbound_chars": len(inb),
            "outbound_chars": len(out),
        }
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:  # a ledger failure must never cost a reply
        logger.warning("turn ledger write failed: %s", exc)
    return None


# ---------------------------------------------------------------- slash commands
def _handle_state(_event=None):
    block = build_state_block()
    return block or "no brain state on disk"


def _handle_recall(event=None):
    q = ""
    try:
        q = (getattr(event, "text", "") or "").split(" ", 1)[1] if event else ""
    except Exception:
        q = ""
    if not q:
        return "usage: /recall <what to look for>"
    sys.path.insert(0, str(Path(__file__).parent))
    import recall  # type: ignore
    hits = recall.search(q, k=6, min_score=0.30)
    if not hits:
        return "nothing in the record matches that."
    return "\n\n".join(f"[{h['score']}] {h['date']} {h['speaker']} ({h['source']})\n  {h['text'][:400]}"
                       for h in hits)


def _handle_wander(event=None):
    """Inspect what the agent thought about when nobody was talking to it.

    Wandering runs in its own isolated session while life goes on — a mind wanders
    while occupied, not only in silence. A fragment may reach a live conversation if
    something in that moment connects to it; this command is the other half, so the
    rest of it stays inspectable on demand and nowhere else.
    """
    n = 6
    try:
        arg = (getattr(event, "text", "") or "").split(" ", 1)
        if len(arg) > 1 and arg[1].strip().isdigit():
            n = max(1, min(20, int(arg[1].strip())))
    except Exception:
        pass
    script = AGENT / "scripts/wander.py"
    if not script.exists():
        return "no wandering engine on disk."
    try:
        r = subprocess.run([sys.executable, str(script), "show", str(n)],
                           capture_output=True, text=True, timeout=60)
        body = (r.stdout or "").strip() or "no wandering logged yet."
        s = subprocess.run([sys.executable, str(script), "stats"],
                           capture_output=True, text=True, timeout=60)
        return f"{body}\n\n— {s.stdout.strip()}"
    except Exception as exc:
        return f"could not read the wandering log: {exc}"


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_command("state", handler=_handle_state,
                         description="Show the state the agent is showing up in")
    ctx.register_command("record", handler=_handle_recall,
                         description="Search the agent's own record",
                         args_hint="<query>")
    # Interior thought, inspectable only when asked for. Whether it may also surface
    # into a live turn is a deliberate operator choice, not a default.
    ctx.register_command("wander", handler=_handle_wander,
                         description="Read what the agent thought about while idle",
                         args_hint="[count]")
