# VALIDATION — The Return Line (Ousia Context Kit)

## The one command

```bash
python3 scripts/self_test.py
```

Expected on a healthy tree:

```
  PASS  plugin resolved agent_dir from config.json
  PASS  state block is emitted with the <agent-state> marker
  PASS  state block carries body readings and circadian phase
  PASS  index built over journal + record
  PASS  search returns the record for a related query
  PASS  retrieved excerpts carry no absolute paths
  PASS  pre_llm_call returns {'context': ...}
  PASS  injected context carries no absolute paths
  PASS  subagent turns are skipped
  PASS  turn ledger wrote one line per turn (got 2)
  PASS  ledger records the reply and marks a silence token as silence
  PASS  a repeated turn id is not double-written
  PASS  shipped plugin source carries no identity residue
  PASS  the configured agent name is not baked into the source

14/14 checks passed
```

## What each check is really testing

| Check | The failure it catches |
|---|---|
| Agent dir resolved from config | A copy of the kit that still points at the machine it came from |
| `<agent-state>` marker emitted | The block built but never delimited, so the model reads it as user text |
| Body readings + phase present | A state block that renders as an empty shell against a real aggregate |
| Index over journal + record | Retrieval that silently indexes nothing and reports success |
| Search returns the record | FTS or vector path broken on a clean machine |
| No persona names in excerpts | Someone else's history shipped inside the bundle |
| `pre_llm_call` returns a dict | A hook that returns a bare string, which the host will not treat as context |
| Subagent turns skipped | Context injected into delegated children, where it is noise and cost |
| Ledger one line per turn | Write-back that drops turns, or writes the draft rather than the reply |
| Silence token recorded as silence | The one signal you cannot reconstruct later from the record itself |
| Repeated turn id not re-written | Duplicate ledger rows inflating whatever reads them |
| No identity residue in source | An identity-coupled copy going out under someone else's name |

## Checks worth running once by hand

1. **Turn the state file off.** Rename `brain-state.json` and send a message. The turn must proceed normally with no state block and no error. A context plugin that can cost you a reply is worse than no context plugin.
2. **Point `record_dir` at an empty directory.** Same expectation: normal turn, empty `<record>`, no exception.
3. **Read the injected block yourself.** Put a raw API call in front of the turn (or log what the hook returns) and confirm the block carries the system note that tells the agent this is its own state, not something to report to the human.

## What a passing test does not prove

That retrieval is *good*. The test asserts the machinery moves; it cannot tell you whether the excerpts the retriever chose were the right ones. That judgement lives in the ledger — see `SETUP-GUIDE.md` §5.
