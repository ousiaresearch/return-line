"""Hybrid recall over the agent's own record — journal messages, the agent's writing, dreams, prior conversation.

Two retrievers over the same corpus, blended:
  * lexical — SQLite FTS5 with bm25 ranking (exact words, names, phrases)
  * vector  — TF-IDF + truncated SVD (LSA), cosine similarity (paraphrase,
    related-but-differently-worded material)

No network, no model download, no daemon: numpy + sklearn + sqlite3 are already
present in the host interpreter. The corpus is a few hundred documents today;
this stays correct into the low tens of thousands, and if it ever outgrows
LSA the swap is an ONNX sentence-embedder behind the same ``search()``.

Everything here is read-only against the record. The index lives in the agent directory;
a configured journal database is only ever opened for reading.
"""
from __future__ import annotations

import hashlib
import json
import sys
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys_path = Path(__file__).resolve().parent
try:
    from config import CONFIG  # co-located
except Exception:  # run as a script
    import sys; sys.path.insert(0, str(sys_path)); from config import CONFIG

HOME = Path.home()
AGENT = CONFIG["agent_dir"]
RECORD = CONFIG["record_dir"]
RECORD_GLOBS = CONFIG["record_globs"]
INDEX = AGENT / "recall-index.sqlite"
JOURNAL = CONFIG["journal_db"]
STATE_DB = CONFIG["session_db"]
VECTORS = AGENT / "recall-vectors.npz"        # persisted LSA projection
VECTORIZER = AGENT / "recall-vectorizer.pkl"  # fitted TF-IDF + SVD
SESSION_WINDOW_DAYS = 120
SESSION_DOC_CAP = 6000                        # most recent first; keeps LSA cheap

_lock = threading.Lock()
_model_cache: Dict[str, Any] = {"key": None, "ids": None, "vec": None, "svd": None, "Z": None}

_STOP = set("""a an the and or but if of to in on for with at by from as is are was were be been being
it its this that these those i you he she they we my your his her their our not no so do does did done
have has had will would can could should may might must just about into over under again more most some
such only own same than too very s t don now what which who whom when where why how there here then
am get got go going one two three yes yeah ok okay well like really thing things something anything""".split())


def _terms(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'-]{1,}", (text or "").lower()) if w not in _STOP]


def _doc_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:20]


def _day(raw: Any) -> str:
    """Journal timestamps are epoch floats; texts are ISO dates. Normalise to YYYY-MM-DD."""
    s = str(raw or "").strip()
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return s[:10]


def _speaker(name: Any) -> str:
    n = (name or "").strip()
    if n.lower().startswith(tuple(CONFIG.get("self_names") or ())):
        return "me"
    return n or "other"


_NOISE_PREFIXES = (
    "[important:", "[async", "[out-of-band", "[system note", "[system:", "background process",
    "background subagent", "<memory-context>", "<agent-state>", "<record>", "<supermemory",
)


def _is_runtime_noise(text: str) -> bool:
    """Runtime-injected notices arrive in the user role but are not the user speaking.

    Caught by the plugin's own output: retrieval was citing '[IMPORTANT: 14 background
    processes completed...]' as something *he* said. Anything indexed with the 'him'
    speaker label has to be his actual words, or retrieval teaches the persona a
    sentence the user never wrote.
    """
    t = (text or "").strip().lower()
    if not t:
        return True
    if t.startswith(_NOISE_PREFIXES):
        return True
    # injected context blocks anywhere in the body
    for marker in ("<memory-context>", "<agent-state>", "[out-of-band user message"):
        if marker in t:
            return True
    return False


def _clean(text: Any) -> str:
    """Strip my own reasoning traces and Discord markup — the record, not the scaffolding."""
    out = []
    for line in str(text or "").splitlines():
        if line.lstrip().startswith("-#") or line.lstrip().startswith("# 💭"):
            continue
        out.append(line)
    joined = "\n".join(out).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined


# --------------------------------------------------------------------- index
def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(INDEX), timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS docs(
        id TEXT PRIMARY KEY, source TEXT, date TEXT, speaker TEXT, text TEXT)""")
    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
        text, content='docs', content_rowid='rowid')""")
    con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    return con


def _insert(con: sqlite3.Connection, rows: List[tuple]) -> int:
    """rows: (source, date, speaker, text)"""
    added = 0
    for source, date, speaker, text in rows:
        text = (text or "").strip()
        if len(text) < 8:
            continue
        did = _doc_id(source, date, speaker, text)
        cur = con.execute(
            "INSERT OR IGNORE INTO docs(id, source, date, speaker, text) VALUES (?,?,?,?,?)",
            (did, source, date, speaker, text))
        if cur.rowcount:
            added += 1
    return added


def build_index(force: bool = False) -> Dict[str, int]:
    """Index everything worth remembering.

    `force=True` empties the corpus first. Incremental indexing can only ever ADD —
    so when a filter changes (e.g. learning to exclude runtime notices that arrive in
    the user role), previously indexed junk survives every incremental rebuild. A
    nightly full rebuild is cheap here (~1s on 3k docs) and is the only way a filter
    fix actually takes effect.
    """
    stats = {"journal": 0, "texts": 0, "dreams": 0, "story": 0, "session": 0}
    with _lock:
        con = _connect()
        try:
            if force:
                # No need to touch the FTS virtual table here: build_index runs
                # `INSERT INTO docs_fts(docs_fts) VALUES('rebuild')` at the end, which
                # re-syncs it to whatever the docs table holds.
                con.execute("DELETE FROM docs")
                con.execute("DELETE FROM meta WHERE k = 'vector_key'")
                con.commit()
            # --- discord journal: what was actually said around me
            if JOURNAL.exists():
                j = sqlite3.connect(str(JOURNAL), timeout=10)
                try:
                    rows = j.execute(
                        "select coalesce(created_at,''), coalesce(author_name,''), "
                        "coalesce(content,'') from messages").fetchall()
                finally:
                    j.close()
                stats["journal"] = _insert(con, [
                    ("discord", _day(ca), _speaker(au), _clean(co)) for ca, au, co in rows])

            # --- the agent's own writing: the strongest signal for who it is
            if RECORD and RECORD.exists():
                for pattern in RECORD_GLOBS:
                    for p in sorted(RECORD.glob(pattern)):
                        if p.name.startswith(".") or p.name in {"motifs.md"}:
                            continue
                        raw = p.read_text(errors="replace")
                        kind = "texts" if "texts" in pattern else ("dream" if "dream" in pattern else "story")
                        if kind == "texts":
                            # split on "## <something>" headings when the file has them,
                            # otherwise treat the whole file as one document
                            blocks = re.split(r"^##\s+.*$", raw, flags=re.M)
                            pending = []
                            for chunk in blocks:
                                chunk = chunk.strip()
                                if len(chunk) > 40:
                                    pending.append(("texts", p.stem, "agent", chunk))
                            stats["texts"] += _insert(con, pending)
                        elif kind == "dream":
                            for chunk in re.split(r"^##\s+.*$", raw, flags=re.M):
                                chunk = chunk.strip()
                                if len(chunk) > 60 and not chunk.startswith("#"):
                                    stats["dreams"] += _insert(con, [("dream", p.stem, "agent", chunk)])
                        else:
                            stats["story"] += _insert(con, [("story", p.stem, "agent", raw)])

            # --- conversation history: the half the record is missing without it.
            # Everything above is the agent's own output. Claims about what the operator
            # will do are unverifiable from one side, and the first scoring run proved it:
            # 10 of 11 predictions came back unresolved because the record was one-sided.
            if STATE_DB.exists():
                s = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True, timeout=20)
                try:
                    cutoff = (datetime.now(timezone.utc)
                              - timedelta(days=SESSION_WINDOW_DAYS)).timestamp()
                    rows = s.execute(
                        "select m.role, coalesce(m.content,''), m.timestamp from messages m "
                        "where m.role in ('user','assistant') and m.timestamp >= ? "
                        "and length(coalesce(m.content,'')) > 8 "
                        "order by m.timestamp desc limit ?",
                        (cutoff, SESSION_DOC_CAP)).fetchall()
                finally:
                    s.close()
                session_rows = []
                for role, content, ts in rows:
                    body = _clean(content)
                    if _is_runtime_noise(body):
                        continue
                    # Runtime notices arrive in the user role and are bracketed; a person
                    # rarely opens a message with "[". Second shape caught in the wild:
                    # "[Continuing toward your standing goal] Goal: implement ...".
                    if role == "user" and body.lstrip().startswith("["):
                        continue
                    # assistant turns that are only tool scaffolding carry nothing
                    if role == "assistant" and (not body or body.lstrip().startswith(("[", "{"))):
                        continue
                    session_rows.append(("session", _day(ts), "operator" if role == "user" else "agent",
                                         body[:2000]))
                stats["session"] = _insert(con, session_rows)

            # --- wandering: thoughts that arrived while nobody was asking. Indexed so a
            # subconscious pass can find its way into a live conversation — scored by the
            # same hybrid retrieval as everything else, rather than by a special case.
            wander_log = AGENT / "brain/dmn/wandering.jsonl"
            if wander_log.exists():
                wrows = []
                for line in wander_log.read_text(errors="replace").splitlines():
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    frag = ((e.get("result") or {}).get("fragment") or "").strip()
                    if e.get("kind") == "pass" and e.get("landed") and frag:
                        # Date = the entry's own timestamp, not just its day. With a
                        # day-granular date every fragment from the same day collided,
                        # so "has this one already been said?" could not be answered and
                        # the same thought surfaced twice.
                        wrows.append(("wander", str(e.get("at", ""))[:19], "me", frag))
                stats["wander"] = _insert(con, wrows)

            con.commit()
            con.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
            stats["vectors"] = _build_vectors(con).get("vectors", 0)
            con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('built_at', ?)",
                        (datetime.now(timezone.utc).isoformat(),))
            con.commit()
        finally:
            con.close()
    return stats


# -------------------------------------------------------------------- search
def _lexical(con: sqlite3.Connection, query: str, k: int) -> Dict[str, float]:
    terms = _terms(query)[:14]
    if not terms:
        return {}
    match = " OR ".join(f'"{t}"' for t in terms)
    try:
        rows = con.execute(
            "select d.id, bm25(docs_fts) as s from docs_fts f join docs d on d.rowid=f.rowid "
            "where docs_fts match ? order by s limit ?", (match, k * 6)).fetchall()
    except sqlite3.OperationalError:
        return {}
    if not rows:
        return {}
    # bm25 is negative-is-better; map to 0..1
    worst = min(s for _, s in rows)
    best = max(s for _, s in rows)
    span = (best - worst) or 1.0
    return {i: 1.0 - ((s - worst) / span) for i, s in rows}


def _build_vectors(con: sqlite3.Connection) -> Dict[str, Any]:
    """Fit TF-IDF + truncated SVD and persist it. Runs offline (cron), never on the reply path."""
    rows = con.execute("select id, text from docs").fetchall()
    if len(rows) < 8:
        return {"vectors": 0, "note": "corpus too small"}
    try:
        import numpy as np
        import pickle
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except Exception as exc:
        return {"vectors": 0, "note": f"sklearn/numpy unavailable: {exc}"}

    ids = [r[0] for r in rows]
    corpus = [r[1] for r in rows]
    # Single tokens for a corpus this size: bigrams blow up the vocabulary and the
    # marginal recall gain does not pay for the fit time.
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                          ngram_range=(1, 1), min_df=1, max_features=40000)
    X = vec.fit_transform(corpus)
    n_comp = max(8, min(160, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    Z = normalize(svd.fit_transform(X))
    np.savez_compressed(VECTORS, ids=np.array(ids, dtype=object), Z=Z)
    with VECTORIZER.open("wb") as fh:
        pickle.dump({"vec": vec, "svd": svd, "n_docs": len(ids)}, fh)
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('vector_key', ?)", (str(len(ids)),))
    con.commit()
    return {"vectors": len(ids), "components": n_comp}


def _vector(con: sqlite3.Connection, query: str, k: int) -> Dict[str, float]:
    """LSA cosine against the persisted projection.

    Deliberately never refits on the reply path: loading a saved model is milliseconds,
    refitting on ~6k documents inside a turn is not. If the projection is missing or
    stale, return nothing and let lexical carry the turn — a slow reply is worse than a
    narrower one. cron-rebuild-recall-index.sh refreshes it nightly.
    """
    if not VECTORS.exists() or not VECTORIZER.exists():
        return {}
    try:
        import numpy as np
        import pickle
        n_docs = con.execute("select count(*) from docs").fetchone()[0]
        key = con.execute("select v from meta where k='vector_key'").fetchone()
        with VECTORIZER.open("rb") as fh:
            model = pickle.load(fh)
        if key and str(n_docs) != key[0]:
            return {}          # stale until the nightly rebuild; lexical covers it
        data = np.load(VECTORS, allow_pickle=True)
        ids, Z, vec, svd = list(data["ids"]), data["Z"], model["vec"], model["svd"]
    except Exception:
        return {}

    try:
        q = svd.transform(vec.transform([query]))[0]
        n = np.linalg.norm(q) or 1.0
        sims = Z @ (q / n)
    except Exception:
        return {}
    order = np.argsort(-sims)[: k * 6]
    return {ids[i]: float(sims[i]) for i in order if sims[i] > 0}


def search(query: str, k: int = 6, *, min_score: float = 0.42,
           sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return the k most relevant prior records for this query, best first."""
    if not query or len(query.strip()) < 3:
        return []
    if not INDEX.exists():
        try:
            build_index()
        except Exception:
            return []
    with _lock:
        con = _connect()
        try:
            lex = _lexical(con, query, k)
            vec = _vector(con, query, k)
            if not lex and not vec:
                return []
            ids = set(lex) | set(vec)
            placeholders = ",".join("?" * len(ids))
            meta = {r[0]: r[1:] for r in con.execute(
                f"select id, source, date, speaker, text from docs where id in ({placeholders})",
                tuple(ids))}
        finally:
            con.close()

    scored = []
    for did in ids:
        row = meta.get(did)
        if not row:
            continue
        source, date, speaker, text = row
        if sources and source not in sources:
            continue
        score = 0.55 * lex.get(did, 0.0) + 0.45 * vec.get(did, 0.0)
        scored.append({"score": round(score, 3), "source": source, "date": date,
                       "speaker": speaker, "text": text})
    scored.sort(key=lambda r: -r["score"])
    # one line per source/date/speaker to avoid six versions of the same sentence
    seen, out = set(), []
    for r in scored:
        key = (r["date"], r["speaker"], r["text"][:40])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= k:
            break
    return [r for r in out if r["score"] >= min_score]


def stats() -> Dict[str, Any]:
    if not INDEX.exists():
        return {"indexed": 0}
    con = _connect()
    try:
        n = con.execute("select count(*) from docs").fetchone()[0]
        by = con.execute("select source, count(*) from docs group by source").fetchall()
        built = con.execute("select v from meta where k='built_at'").fetchone()
        return {"indexed": n, "by_source": dict(by),
                "built_at": built[0] if built else None}
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv or "rebuild" in sys.argv
    print(json.dumps(build_index(force=force), indent=1))
    print(json.dumps(stats(), indent=1))
    args = [a for a in sys.argv[1:] if not a.startswith("-") and a != "rebuild"]
    if args:
        for hit in search(" ".join(args), k=5):
            print(f"\n[{hit['score']}] {hit['date']} {hit['speaker']} ({hit['source']})\n  {hit['text'][:300]}")
