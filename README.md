# The Return Line — make your agent's state show up in conversation

**Your agent can hold a state file all day and still sound identical in every conversation. This plugin closes that gap.**

It is a small Hermes plugin with two jobs. Before each reply, it hands the agent its current state and, when relevant, excerpts from its own past — appended to the message being answered, not to the system prompt. After each reply, it writes that turn to a ledger.

That is the whole thing. Two hooks, one file each direction.

## Why not just put it in the system prompt?

Because of prompt caching. Changing the system prompt mid-conversation throws away the cached prefix and makes every turn more expensive. Hermes' `pre_llm_call` hook avoids that: whatever it returns is attached to the current message at API time — never the system prompt, never past turns — so a long conversation stays cheap and cached, and the state still arrives.

It fires on every surface too, so one install covers the gateway, the CLI, a scheduled tick, or the desktop app.

## What the agent receives

**Its own state** — the readings from [Biomimetic Brain](https://github.com/ousiaresearch/biomimetic-brain): energy, fatigue, time of day, attention, what it is conflicted about, what it is carrying. Capped at about 1100 characters, so it can never crowd out the actual conversation.

**Its own record** — dated excerpts of things it actually said, wrote, or dreamed before, retrieved only when the incoming message looks related. Search is hybrid: exact words via SQLite full-text, and meaning via TF-IDF + SVD, blended. Every line carries a date and a source. Nothing is invented, and nothing from today is served back as memory.

**Optionally, a thought** — a fragment the agent produced while nobody was asking, surfaced only when the current turn connects to it, only once, in its own voice. Off unless you run a wandering pass that produces the log.

## What goes the other way

One raw line per turn into `turn-ledger.jsonl`: what came in, what was sent, on which surface, with which model, and whether the reply was a silence token. No scoring, no summarizing, no model call — a turn is the one place an agent is not allowed to get slower. The scheduled passes in the mind kit read that ledger afterwards.

Without it the loop only runs one way: state reaches the conversation, and nothing carries the conversation back.

## Install

```bash
cp -r return-line ~/.hermes/plugins/return-line
cp ~/.hermes/plugins/return-line/config.example.json ~/.hermes/plugins/return-line/config.json
# edit config.json: where your agent lives, and optionally where its writing lives
python3 scripts/self_test.py     # 14 checks, expect all green
```

Then restart the host so the plugin loads. In a live conversation: `/state` shows what the agent is currently showing up in, `/record <query>` searches its own record.

No dependencies beyond the host interpreter. `sqlite3` is stdlib; `numpy` and `scikit-learn` enable the meaning-based half of search, and if they are missing the plugin quietly falls back to keyword search instead of failing. That quiet fallback is deliberate: a context plugin must never be the reason a reply does not happen.

## Files

```
return-line/
  plugin.yaml           manifest (pre_llm_call + post_llm_call)
  __init__.py           the state block, the record block, the thought block, the ledger
  recall.py             hybrid retrieval over the agent's own record
  config.py             configuration: config.json, then environment, then defaults
  config.example.json   copy to config.json and edit
SETUP-GUIDE.md          install, step by step, including the knobs worth leaving alone
VALIDATION.md           what each check catches, and what a passing test does not prove
NOTES.md                what was changed to make this adoptable, and its limits
scripts/self_test.py    builds a neutral agent and proves all three promises
```

## The Ousia framing, kept short

*Receive, calibrate, repair, return* — read before deciding, keep an honest record, correct what drifted, and leave the work usable afterwards. This plugin is the return half: the loop that carries a live conversation back into the agent's own record, and the agent's record back into the conversation.

## What this is not

Not a memory provider, not a personality, and not a claim that the agent is present in any sense beyond *these readings reached the turn*. Plumbing with an honest label.

MIT licensed. See [LICENSE](LICENSE).
