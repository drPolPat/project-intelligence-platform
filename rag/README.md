# RAG Pipeline (Phase 3)

Citation-grounded Q&A over the fabricated GFPA standards corpus (see
[`knowledge_base/README.md`](../knowledge_base/README.md)).

## Components

- [`chunk_documents.py`](chunk_documents.py) — clause-level chunker. Splits
  each document into citable units (numbered clauses, whole non-clause
  sections, and — as of the fix below — per-topic sub-chunks within a
  bulleted clause). Each chunk carries an `embedding_text` used only for
  retrieval, separate from the exact `text` used for citation checking.
- [`build_index.py`](build_index.py) — embeds every chunk with fastembed
  (`BAAI/bge-small-en-v1.5`) and stores them in a local, persistent ChromaDB
  collection (`chroma_db/`, gitignored — rebuild locally).
- [`query_pipeline.py`](query_pipeline.py) — retrieval + refusal
  classification + Claude-generated, citation-structured answers.
- [`eval/`](eval) — hand-verified 20-question reference set and the eval
  harness scoring retrieval accuracy, citation validity, LLM-judged
  faithfulness, and refusal correctness.

## Closed: facility-type → classification-bridge retrieval gap

**Mechanism (background):** `bge-small`'s embeddings score on a chunk's
overall semantic gist, not keyword presence. `GFPA-201 Section 2` (the
Class A/B/C definitions) competes for retrieval rank against chunks that
state the actual numeric coverage/staffing requirement, and used to lose
that competition whenever a query named a concrete facility type (museum,
mall, office tower, utility substation) without also using the standard's
own vocabulary (Class A, Class B, Class C) — or whenever the query's
dominant topic pulled toward a different document entirely (e.g. inspection
frequency, `GFPA-240`).

An embedding-level fix (`EMBEDDING_ENRICHMENT` in
[`chunk_documents.py`](chunk_documents.py) — prepending a query-phrased
bridge sentence to that chunk's `embedding_text`) helped but didn't
generalize: it depended on the query's phrasing landing close enough to the
enrichment text, and testing found real cases where it still didn't —
notably when the query's dominant topic belonged to a different document
entirely (a "how often does an office building need inspecting" query is
topically about `GFPA-240`, not `GFPA-201`), and when an *agent* calling
this pipeline reformulated the user's question before passing it in, adding
its own sampling-dependent variance on top of the retrieval gap (confirmed
directly: the same "staffing minimums for malls" request produced a
correctly-cited answer in some runs and a refusal in others, depending on
how the calling agent happened to paraphrase it that time).

**The actual fix — deterministic, at the retrieval layer, not another
embedding trick:** `_ensure_classification_bridge()` in
[`query_pipeline.py`](query_pipeline.py) checks every query for a concrete
facility-type mention (word-boundary matched against `FACILITY_TYPE_KEYWORDS`,
plurals included) and, if `GFPA-201 Section 2` isn't already in the
retrieved set, fetches that exact chunk **by ID** — not a second approximate
search, which could fail the same way the first one did — and appends it
with an honestly-computed cosine similarity (not a placeholder). This closes
the gap once, for every caller (eval harness, agent, anything else built on
`query_pipeline.answer_question`), rather than relying on a calling agent to
notice and compensate — which was itself diagnosed as an unreliable strategy
(the T2 finding above).

This closes **two** of the three previously-documented instances of the
general mechanism:
- The original `museum`/`mall`/`substation` → `Class A/B/C` gap.
- The `GFPA-201 Section 2` vs. `GFPA-240` cross-document crowd-out (the
  "office building" inspection-frequency case) — confirmed fixed directly,
  since "office building" is itself a facility-type mention this fix
  triggers on.

Two real bugs were caught and fixed during implementation, not just assumed
away: a naive substring check on `"mall"` false-triggered on `"small"`/
`"smaller"`; and the initial word-boundary regex fix then missed the plural
`"malls"` (the actual word in the request that started this investigation),
since `\bmall\b` has no boundary before a trailing `s`. Both confirmed fixed
via a 13-case test matrix before shipping.

## Known limitation: dense retrieval underweights plain-language ↔ internal-label bridges (still open elsewhere)

The general mechanism above still applies to classification/definition
chunks **other than** `GFPA-201 Section 2` — this fix is scoped to facility
type → Class A/B/C specifically, not a general "any definition chunk" fix,
for the same reason the original embedding-level attempt wasn't generalized:
each instance needs the same diagnose-before-patch treatment, not a
reflexive one-size-fixes-all rule. Two instances remain open and accepted:

1. **`GFPA-230 Section 2.5` ("Critical" incident severity) — never patched.**
   A plain-language query about a "life-threatening" or "urgent" incident
   does not reliably retrieve the chunk that actually says "Critical" means
   activating emergency response (eval case Q09). The model correctly
   declined to guess the severity-label mapping and answered only from what
   it had — appropriately conservative, but it means the user doesn't get
   the direct answer a query like this should be able to produce. The
   agent-reformulation non-determinism described above is still a live risk
   here specifically, since this gap has no deterministic fix yet: the same
   severity-related request could still succeed or fail depending on how an
   agent happens to phrase its internal `query_standards` call.
2. **Topic dilution inside a multi-subject chunk (`GFPA-240 Section 3.1`) —
   partially mitigated, not fully closed.** The original checklist clause
   listed five unrelated inspection topics in one chunk, diluting each
   topic's contribution to the shared embedding. Splitting it into five
   per-topic sub-chunks (`3.1.1`–`3.1.5`, see `_make_clause_chunks` in
   `chunk_documents.py`) measurably improved the structural-condition
   sub-chunk's similarity against a real structural-vulnerability query
   (0.680 → 0.704), and the sub-chunk is now retrievable and citable on its
   own for other phrasings. But for the specific eval question that
   surfaced this (Q14), it still didn't crack a competitive top-5 (settling
   at rank 9, similarity 0.701 vs. a ~0.714 cutoff) once GFPA-215's own
   remediation-timeline chunks — also relevant to the same question — filled
   the top slots. The chunking-strategy fix is real and reusable; it is not
   a guarantee of top-k inclusion when a query legitimately spans several
   competitive documents.

**Why these two are documented rather than patched the same way:** the
facility-type fix works because there is exactly one well-defined target
chunk and a closed, enumerable set of trigger keywords (museum, mall, office
tower, utility substation, and their synonyms) — a deterministic "does the
query mention X, is chunk Y present, if not fetch it" rule is cheap and
unambiguous to write. Neither remaining case has that shape as cleanly:
severity-label synonyms ("critical," "urgent," "life-threatening," ...) are
a much fuzzier, open-ended set than four facility types, and the `GFPA-240
Section 3.1` case is about a chunk being *outcompeted*, not *absent from the
corpus*, so there's no single missing chunk to deterministically re-inject.
Treat these two as a standing, evidenced characteristic of the retrieval
layer, not a bug list to clear to zero — the eval harness
(`eval/run_eval.py`) and the agent tool-call logs
(`agent/eval_tool_selection_run.json`) are what surface new instances as the
corpus or query patterns evolve, and each new instance should get the same
diagnose-before-patch treatment rather than a reflexive fix.
