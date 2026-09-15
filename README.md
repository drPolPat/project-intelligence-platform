# Project Intelligence Platform

Portfolio project demonstrating an end-to-end AI/ML/data engineering pipeline for
engineering & infrastructure project management, inspired by security risk
assessment and project coordination work on infrastructure projects (museums,
malls, towers, utility infrastructure).

**All data in this repository is synthetic and fabricated.** No real company,
client, facility, or incident data is used anywhere in this project.

**Fictional standards body and documents, created for demonstration purposes
only — not real regulatory or compliance guidance.** See
[`knowledge_base/README.md`](knowledge_base/README.md) for details on the
fabricated "GFPA" standards framework used by the RAG layer.

## Pipeline (built incrementally)

1. **Data generation** — Python scripts generating synthetic facility risk
   assessments, safety inspection reports, and incident/near-miss reports.
2. **Structured storage** — PostgreSQL (SQLite for local dev).
3. **Document store** — ChromaDB/fastembed vector store over a small synthetic
   knowledge base of fabricated security/safety standards documents.
4. **ML layer** — Interpretable scikit-learn model predicting risk score /
   incident likelihood, with feature importance analysis and a model card.
5. **RAG layer** — Citation-backed Q&A over the synthetic standards docs, with
   a retrieval/citation/refusal eval harness.
6. **Agent layer** — Claude API tool-calling agent orchestrating the ML
   risk-scoring tool and RAG Q&A tool, drafting first-pass risk summaries.
7. **API layer** — FastAPI backend.
8. **Frontend** — React dashboard with an AI assistant chat interface.
9. **Deployment** — Railway (backend) + Vercel (frontend).

## Status

**Phase 1 (complete):** synthetic data generation.
- [`data_generation/generate_risk_assessments.py`](data_generation/generate_risk_assessments.py) —
  facility risk assessments. Builds a fixed 120-facility registry (stable
  `facility_id` -> type/name/city) and generates assessment snapshots against it.
- [`data_generation/generate_inspection_reports.py`](data_generation/generate_inspection_reports.py) —
  safety inspection reports. Reuses the same facility_id pool; checklist
  failure rates are driven by each facility's latest risk-assessment factors
  (surveillance coverage, structural vulnerabilities, staffing), so the two
  datasets join into one coherent story.
- [`data_generation/generate_incident_reports.py`](data_generation/generate_incident_reports.py) —
  incident/near-miss reports. Also reuses the same facility_id pool; incident
  frequency, category mix, and severity are all driven by each facility's raw
  risk factors — but via an independent Poisson process with its own
  sensitivity constants and RNG seed, deliberately NOT derived from (or
  numerically matching) the risk assessments' `past_incident_count`, to avoid
  leaking that value into any model that would later predict incidents from
  risk-assessment features.

**Phase 2 (complete):** ML risk-prediction model.
- [`ml/train_risk_model.py`](ml/train_risk_model.py) — Ridge (normalized
  factors) and Random Forest (raw factors) predicting `computed_risk_score`,
  evaluated with a `GroupShuffleSplit` by `facility_id`.
- [`ml/robustness_check.py`](ml/robustness_check.py) — reruns the same
  pipeline across 10 splits to check result stability.
- [`MODEL_CARD.md`](MODEL_CARD.md) — full write-up: headline finding is that
  Ridge and Random Forest have comparable overall regression accuracy, but
  Random Forest has a robust, mechanistically-explained advantage at
  detecting Critical-tier (compounding-risk) facilities that Ridge cannot
  represent at all.

**Phase 3 (in progress):** RAG / synthetic standards knowledge base.
- [`knowledge_base/`](knowledge_base/) — 6 fabricated "GFPA" standards
  documents mapped to the structured datasets' risk factors. See its own
  README for the fictional-framework disclaimer and constraints.
- [`rag/`](rag/) — clause-level chunking, fastembed + ChromaDB retrieval,
  citation-structured Claude generation, refusal classification, and a
  20-question hand-verified eval harness (retrieval accuracy, citation
  validity, LLM-judged faithfulness, refusal correctness). See its own
  README for a documented, evidenced limitation of the retrieval layer
  (dense embeddings underweighting plain-language ↔ internal-label bridges).

Not yet built: the agent layer, the API layer, the frontend, and everything
from structured storage onward.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Generating synthetic data

```bash
python data_generation/generate_risk_assessments.py
python data_generation/generate_inspection_reports.py   # reads the CSV above
python data_generation/generate_incident_reports.py     # also reads it (imports latest_risk_profile from the inspections script)
```

Writes `facility_risk_assessments.csv`, `safety_inspection_reports.csv`, and
`incident_reports.csv` under `data/synthetic/` (gitignored — regenerate
locally; all three scripts are seeded for reproducibility). Regenerate the
risk assessments first — the other two both read that CSV to build their
facility risk profiles.
