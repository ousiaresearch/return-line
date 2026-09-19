# NOTES — what moved to make this adoptable

This kit is the operating half of the Ousia build, lifted out of the tree it ran in. The logic is the same; the coupling is gone. What changed, honestly:

## De-identification

- Every hardcoded path became configuration: `agent_dir`, `record_dir`, `journal_db`, `session_db`, plus environment overrides so two agents can run off one copy.
- The injected markers are generic — `<agent-state>`, `<record>`, `<thought>` — and the system notes are written to *the agent*, not from one named agent.
- Speaker labels in the corpus are `operator` / `agent` instead of a named pair.
- Noise filters that dropped documents beginning with a specific agent's name now read the names from `self_names` in config.
- The corpus walk was three hardcoded vault globs; it is now `record_globs` over whatever record directory you point it at, and it splits documents on `##` headings when they have them rather than assuming one author's file format.

## What was deliberately left behind

Live state values, the original agent's record directory, its journal database, its identity documents, and every log line that referenced a person by name. The kit is the instrument; the readings stayed in the tree.

## What a reader should know about its limits

- **Retrieval quality is unproven at the boundaries.** The hybrid retriever is correct and cheap; whether it surfaces the *right* excerpt for a given turn is a judgement that only the ledger can settle. Treat the first hundred turns as calibration.
- **The vector half degrades quietly.** Without `numpy`/`scikit-learn` the index still builds and search still works lexically. Quiet degradation is intentional — a context plugin must never be the reason a reply fails — but it means a machine missing those packages will not tell you it is doing less.
- **`session_db` is read-only and optional.** If the configured session store does not exist, that source is skipped rather than created. Nothing in this kit writes to it.
- **Prompt caching is the constraint that shaped the design.** If you move this logic into a system prompt or a memory provider, expect to pay for it on every turn.
