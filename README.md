# Project Intelligence Platform

A facility security risk-assessment platform combining an ML risk-scoring
model, a citation-grounded RAG Q&A system over a synthetic standards corpus,
and a Claude-based agent that orchestrates both — plus a dashboard on top of
all three. Built as an end-to-end portfolio demonstration of the full
pipeline a real risk-intelligence product would need: data generation, model
evaluation with an honest model card, retrieval-grounded generation with a
documented failure mode, agentic tool orchestration with a caught grounding
regression, a FastAPI backend, and a deployed React dashboard.

**Live demo:** [project-intelligence-platform frontend](https://frontend-three-pi-43.vercel.app) · [backend health check](https://project-intelligence-platform-production-90f0.up.railway.app/api/health)

**All data in this repository is synthetic and fabricated.** No real company,
client, facility, or incident data is used anywhere. The "GFPA" standards
framework the RAG layer queries is fictional — see
[`knowledge_base/README.md`](knowledge_base/README.md).

---

## Architecture: a 6-phase pipeline

Each phase was built and verified independently before the next one
consumed it. Full detail lives in each phase's own doc — this section links
out rather than repeating it.

| Phase | What | Detail |
|---|---|---|
| 1. Data generation | Three seeded, reproducible synthetic datasets — 200 facility risk assessments (120 facilities), 300 safety inspections, 505 incident/near-miss reports — deliberately cross-correlated (inspection failure and incident rate both driven by the same underlying risk factors) but not derived from each other, to avoid leaking the prediction target into itself | [`data_generation/`](data_generation/) |
| 2. ML risk model | Ridge vs. Random Forest predicting `computed_risk_score`, evaluated with a `GroupShuffleSplit`-by-facility and a 10-split robustness check | [`MODEL_CARD.md`](MODEL_CARD.md) |
| 3. RAG standards Q&A | Clause-level chunking, fastembed + ChromaDB retrieval, citation-structured Claude generation, two-layer refusal classification, 20-question hand-verified eval harness | [`rag/README.md`](rag/README.md) |
| 4. Agent layer | Claude tool-calling agent orchestrating `predict_risk_score`, `query_standards`, and `draft_risk_report`, with a 9-case tool-selection + grounding eval | [`agent/eval_tool_selection.py`](agent/eval_tool_selection.py) |
| 5. API layer | FastAPI backend, 6 endpoints, deliberately no `response_model` on the two polymorphic-response endpoints (see inline comments in [`api/main.py`](api/main.py)) | [`api/main.py`](api/main.py) |
| 6. Dashboard | React + Recharts dashboard: facility risk form, full report view, standards search, trend/overview charts, chat panel | [`frontend/`](frontend/) |

**Deployment:** FastAPI backend on Railway (Docker build regenerates the
CSVs, trained model, and Chroma vector index from their seeded/tracked
sources at build time — see [`api/Dockerfile`](api/Dockerfile) — rather than
committing generated binaries); React frontend on Vercel, configured via
`VITE_API_BASE_URL`.

**MCP server (additive, not part of the 6-phase pipeline above):**
[`mcp_server/`](mcp_server/) exposes the same three agent tools
(`predict_risk_score`, `query_standards`, `draft_risk_report`) over the Model
Context Protocol via FastMCP, so any MCP-compatible client — Claude Desktop,
Claude Code, another agent framework — can call them without going through
`agent/agent_loop.py`'s Messages-API loop, which keeps running unchanged for
the deployed chat endpoint. It imports the tool functions and their
Messages-API tool definitions directly from `agent/tools.py` and
`agent/draft_risk_report.py` — same single-source-of-truth discipline as
everywhere else in this project, not a second implementation.

It runs in its own venv (`mcp_server/.venv`, `mcp_server/requirements.txt`,
`mcp_server/.env`), independent of the top-level `requirements.txt`. This
isn't incidental: installing `fastmcp` into the shared venv upgrades
`starlette` to a version incompatible with the deployed API's
`fastapi==0.115.6` pin, confirmed by direct reproduction — `FastAPI()` itself
raises `TypeError: Router.__init__() got an unexpected keyword argument
'on_startup'` once that happens. The two servers are independently
deployable and never need to share a runtime, so isolating the environment
was the fix, not pinning `starlette` down.

---

## Notable engineering findings

These are the findings worth a technical reader's time — stated with the
evidence, not as a feature list. Each links to its full writeup.

**1. The headline ML result reversed under a robustness check, and that reversal is the finding.**
A single train/test split (`random_state=0`) showed Random Forest beating
Ridge on both RMSE (6.09 vs. 7.46) and MAE (4.73 vs. 5.45). Re-running the
identical pipeline across 10 `GroupShuffleSplit` folds reversed it: Ridge is
comparable-or-better on mean RMSE/MAE (7.02 vs. 7.51) with meaningfully
lower variance. What *didn't* reverse — and got stronger — is Critical-tier
recall: Ridge caught **zero** Critical-tier facilities in **10 out of 10**
held-out folds, because the generator's compounding-risk interaction term is
structurally invisible to a linear model; Random Forest averaged 0.57
recall on the same rows. → [`MODEL_CARD.md` §2, §5](MODEL_CARD.md)

**2. A raw correlation near zero hid a real, recoverable effect.**
`entry_points`' pairwise correlation with `computed_risk_score` is **−0.047**
— indistinguishable from noise in isolation, because `facility_type`
confounds it (each type has both a fixed `entry_points` range and a
different baseline risk). Once the other four risk factors are controlled
for in Ridge's multiple regression, it recovers a correctly-signed
coefficient of 6.6 (true weight: 15.0). The four factors that *do* drive the
compounding-risk bonus are all *over*-recovered (coefficients sum to 112.1
against a true 100) for the same underlying reason. → [`MODEL_CARD.md` §3](MODEL_CARD.md#3-coefficient-recovery-ridge-vs-the-known-risk_weights)

**3. Two independent refusal mechanisms in RAG — proven non-redundant by a hard-negative test.**
The pipeline refuses two structurally different ways: a deterministic
pre-generation check (similarity threshold / known-missing-document match)
and a generation-time model judgment (`can_answer: false` even when
retrieval passed). A hard-negative question about fire-suppression design
standards retrieves at 0.712 cosine similarity — comfortably *above* the
0.55 refusal threshold, because "fire suppression" is a literal keyword
match to an unrelated checklist bullet — so the similarity gate cannot catch
it. Only the model's own grounding-discipline judgment can, and live testing
confirmed it does (`model_declined`, with reasoning). → [`rag/eval/reference_questions.json` Q20](rag/eval/reference_questions.json)

**4. Keyword stuffing measurably hurt retrieval; a question-shaped sentence fixed it — and even that didn't fully generalize.**
Plain-language facility-type queries ("what surveillance does a museum
need") often failed to retrieve the corpus's Class A/B/C definition chunk,
because dense embeddings score on overall semantic gist, not keyword
presence. The first fix attempted — appending a plain list of facility-type
synonyms to the chunk's embedding text — was measured directly rather than
assumed to work, and it made things *worse*: similarity against a
representative query dropped from 0.655 to 0.616, still outside the top-5.
Replacing the noun list with a short, question-mimicking sentence carrying
real definitional content lifted the same query to 0.75+. Even that
qualitatively-correct fix didn't fully generalize, though: cross-document
topic crowd-out and agent-introduced reformulation variance remained
(the same request producing a cited answer in some runs and a refusal in
others). The fix that actually closed it: a deterministic rule that detects
a facility-type mention and fetches the classification chunk **by ID** if
it's missing from the retrieved set — no second approximate search that
could fail the same way. Two real regressions were caught writing this fix
(a `"mall"` substring match false-triggering on `"small"`; a word-boundary
regex then missing the plural `"malls"`), both closed with a 13-case test
matrix before shipping. → [`rag/README.md`](rag/README.md#closed-facility-type--classification-bridge-retrieval-gap)

**5. An agent passed a tool-selection eval while silently violating grounding — caught by a stricter check, not the original one.**
A "give me a full report" request correctly called `draft_risk_report` alone
(tool-selection: PASS) — but the agent then appended a self-generated
closing sentence after relaying the tool's supposedly-verbatim narrative,
synthesizing a new claim by connecting two separate citations. Tool-name
matching can never catch this class of bug, since the wrong tool was never
called — nothing was structurally wrong except the words the agent added on
top. Fixed by adding an exact-string comparison between the agent's final
response and the tool's own narrative field (`eval_tool_selection.py`'s T9),
not by trusting that correct tool selection implies correct output. → [`agent/eval_tool_selection.py`](agent/eval_tool_selection.py)

**6. A JSON parsing bug was traced to the model's own generation, not our code, by comparing live outputs byte-for-byte.**
Grounding-check failures kept surfacing in composite report generation,
always on an em-dash inside a citation quote, with the *same* source text
correct in some calls and corrupted in others within a single run. Root
cause: the model occasionally double-escapes a backslash when writing a
non-ASCII character into generated JSON, emitting the 7-character literal
`\u2014` instead of a proper 6-character JSON escape — standards-compliant
JSON decoding then produces the *literal text* `\u2014`, not an em-dash.
Confirmed 4/4 on real Phase 4 output via `repr()` diffing against the true
corpus text, not assumed from a stack trace. Fixed once at the single point
raw model output enters the system, not patched separately in every
downstream consumer. → [`rag/query_pipeline.py`](rag/query_pipeline.py) (`_fix_literal_unicode_escapes`)

---

## Tech stack

- **Data / ML:** Python, pandas, numpy, scikit-learn (Ridge, Random Forest), joblib
- **RAG:** fastembed (`BAAI/bge-small-en-v1.5`, local, no API calls for retrieval), ChromaDB (persistent local vector store)
- **Agent / generation:** Anthropic Claude (Messages API, tool calling, structured output)
- **Backend:** FastAPI, Pydantic, uvicorn
- **Frontend:** React 19, Vite, Recharts
- **Deployment:** Docker (Railway), Vercel

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (required for the
RAG, agent, and chat endpoints — not for pure ML prediction or the raw
`/api/facilities` dataset endpoint).

Regenerate the three build artifacts `.gitignore` deliberately excludes,
**in this order** (each seeded for reproducibility):

```bash
python data_generation/generate_risk_assessments.py
python data_generation/generate_inspection_reports.py    # reads the CSV above
python data_generation/generate_incident_reports.py      # reads it too
python ml/train_final_model.py                           # reads facility_risk_assessments.csv
cd rag && python build_index.py                          # embeds knowledge_base/*.md into chroma_db/
```

Run the backend and frontend:

```bash
uvicorn api.main:app --reload --port 8000

cd frontend
npm install
npm run dev          # reads VITE_API_BASE_URL from .env, defaults to localhost:8000
```

## Limitations

This is a synthetic-data demonstration project. It is not, and should not be
read as, a production-ready risk-assessment tool.

- **100% synthetic ground truth throughout.** `computed_risk_score` is
  generated by a known, disclosed formula, not observed from real incidents.
  Every ML/RAG result demonstrates that these methods can recover a *known*
  synthetic structure — it says nothing about real-world facility risk.
- **Small-N statistics.** 120 unique facilities behind 200 assessment rows;
  Critical-tier test folds carry 1–5 rows. Point estimates (e.g. RF's 0.57
  mean recall) are directional, not precise — see [`MODEL_CARD.md` §6](MODEL_CARD.md#6-limitations).
- **The RAG similarity-refusal threshold is a placeholder**, calibrated from
  two ad hoc data points, not a real score distribution — see
  [`rag/README.md`](rag/README.md) and the docstring in
  [`rag/query_pipeline.py`](rag/query_pipeline.py).
- **Dense retrieval still underweights some plain-language ↔ internal-label
  bridges**, beyond the one case (facility-type → Class A/B/C) that got a
  deterministic fix. Two known instances remain open and documented rather
  than patched — see [`rag/README.md`](rag/README.md#known-limitation-dense-retrieval-underweights-plain-language--internal-label-bridges-still-open-elsewhere).
- **The fictional GFPA standards framework** is invented for this project
  and must not be mistaken for real regulatory or compliance guidance — see
  [`knowledge_base/README.md`](knowledge_base/README.md).
- **No production hardening.** No auth, no rate limiting, no persistent chat
  history (in-memory only, dropped on backend restart), no monitoring beyond
  a basic `/api/health` check.
