# SETUP GUIDE — The Return Line (Ousia Context Kit)

Written for one human and their agent to walk through together. If you are the agent: read the whole file first, then run the validation.

Ten minutes, most of it deciding what belongs in your record directory.

---

## 0. What you are installing

A Hermes plugin with two hooks: one that appends context to the turn before the model sees it, one that writes the finished turn to a ledger afterwards. No service, no daemon, no network calls. Nothing is sent anywhere.

The plugin reads two things: your agent's state file, and its own record. Both are configurable, and either can be absent without breaking anything — you will just get less context.

---

## 1. Prerequisites

- Hermes with plugin support (`~/.hermes/plugins/`).
- Optional but recommended: **the mind kit** ([`biomimetic-brain`](https://github.com/ousiaresearch/biomimetic-brain), shipped earlier as `ousia-mind-kit`) installed first, so `brain-state.json` exists. Without it the state block simply returns empty, and the self-test says so in a `NOTE` line rather than passing silently.
- The host interpreter needs `sqlite3` (stdlib) and, for the vector half of retrieval, `numpy` and `scikit-learn`. If those are missing, search still works lexically; the kit does not fail.

---

## 2. Install the plugin

```bash
cp -r return-line ~/.hermes/plugins/return-line
cp ~/.hermes/plugins/return-line/config.example.json ~/.hermes/plugins/return-line/config.json
```

Then edit `config.json`:

| Key | What it means |
|---|---|
| `agent_dir` | Where the agent's `brain-state.json` and `turn-ledger.jsonl` live |
| `record_dir` | A directory of the agent's own writing — anything the globs below match. Optional. |
| `record_globs` | Which files inside `record_dir` are indexed (default `texts/*.md`, `dreams/*.md`, `Story/*.md`) |
| `journal_db` | Optional SQLite file with a `messages(created_at, author_name, content)` table — what the Discord heartbeat kit produces. Optional. |
| `session_db` | The Hermes session store, indexed so the record is not one-sided. Default `~/.hermes/state.db` |
| `self_names` | Names that mark a document as the agent's own identity boilerplate; those are skipped |
| `max_state_chars` / `max_record_chars` / `max_record_hits` | Caps. Raise them only if you have measured why |

Environment variables override the file (`RETURN_LINE_AGENT_DIR`, `RETURN_LINE_RECORD_DIR`, `RETURN_LINE_JOURNAL_DB`, `RETURN_LINE_SESSION_DB`), which is how you run two agents off one plugin copy.

---

## 3. Prove it before you trust it

```bash
cd return-line && python3 scripts/self_test.py
```

Expect 14/14. The test builds a neutral agent from scratch — its own state, record and journal — and checks that state reaches the turn, the record is searchable, and the ledger records both a reply and a silence token. It restores your `config.json` afterwards.

---

## 4. Turn it on

Restart the gateway (or the host process) so the plugin loads. Then, in a live conversation:

```
/state      show what the agent is currently showing up in
/record <query>   search its own record
/wander [n]       read what it thought about while idle (needs a wandering pass)
```

If `/state` returns *no state on disk*, the plugin loaded and `brain-state.json` is missing — finish the mind kit.

---

## 5. Tune it honestly

- **The blocks are advisory and addressed to the agent.** Keep the injected system notes. They are what stops a model from turning its own readings into conversation fodder.
- **Start with the caps where they are.** A 1100-character state block is about 275 tokens. A record block that grows into a document changes the behaviour of the whole conversation.
- **Watch the first hundred turns before trusting retrieval.** Retrieval that is never wrong is retrieval nobody checks. Read the ledger and ask whether the `<record>` excerpts were actually relevant; if they are not, raise `min_score` in the recall call.
- **The ledger is the honest artefact.** If the agent claims continuity your record cannot support, the ledger is where you find out.

---

## 6. Removing it

Delete the plugin directory and restart. The ledger and index files stay in `agent_dir`; delete those too if you want the record gone. Nothing else references them.
