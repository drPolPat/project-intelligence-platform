"""
Phase 4: agent tools, wired up and tested one at a time (per the phase plan)
before any tool-calling loop exists to orchestrate them.

predict_risk_score wraps the deployed Random Forest model artifact from
ml/train_final_model.py.

query_standards wraps the existing RAG pipeline (rag/query_pipeline.py).
"""
import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
import joblib
import pandas as pd
from fastembed import TextEmbedding

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
GEN_DIR = Path(__file__).resolve().parent.parent / "data_generation"
RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
sys.path.insert(0, str(GEN_DIR))
sys.path.insert(0, str(RAG_DIR))
from generate_risk_assessments import FACILITY_TYPES, RISK_LEVEL_BINS, RISK_LEVEL_LABELS  # noqa: E402
from build_index import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL  # noqa: E402
from query_pipeline import answer_question as _answer_question  # noqa: E402

MODEL_PATH = ML_DIR / "model_artifacts" / "random_forest_risk_model.joblib"
METADATA_PATH = ML_DIR / "model_artifacts" / "random_forest_risk_model.meta.json"

_model = None
_metadata = None


def _load_model():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model artifact at {MODEL_PATH}. Run `python ml/train_final_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
        _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return _model, _metadata


# Tool definition in Anthropic Messages API format (see agent/run_agent.py,
# added alongside the other tools, for how this gets passed to client.messages.create).
PREDICT_RISK_SCORE_TOOL = {
    "name": "predict_risk_score",
    "description": (
        "Predicts a facility's risk score (0-100) and risk level (Low/Medium/High/Critical) from its "
        "raw security risk factors, using a Random Forest model trained on 200 synthetic facility risk "
        "assessments. Random Forest is used here rather than the project's alternative Ridge regression "
        "model specifically because it has a documented Critical-tier detection advantage: across 10 "
        "held-out test splits, Ridge caught 0% of Critical-tier facilities while this Random Forest "
        "averaged 57% recall on the same rows (see MODEL_CARD.md) — Ridge is structurally unable to "
        "represent the nonlinear 'multiple risk factors compounding at once' interaction that actually "
        "drives the Critical tier, so it is the wrong tool for a risk-FLAGGING use case even though the "
        "two models are comparable on average-case accuracy. This model is trained entirely on synthetic "
        "data with no real-world validation (see MODEL_CARD.md Limitations) — treat its output as a "
        "demonstration, not a real risk assessment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entry_points": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of distinct entry points (doors, gates, access points) at the facility.",
            },
            "surveillance_coverage_pct": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Percentage of the facility under active video surveillance coverage.",
            },
            "staffing_level": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of on-site security staff during a typical shift.",
            },
            "past_incident_count": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of security/safety incidents logged at the facility in roughly the past 12 months.",
            },
            "structural_vulnerabilities": {
                "type": "integer",
                "minimum": 0,
                "description": "Count of currently open/unresolved structural vulnerability findings at the facility.",
            },
            "facility_type": {
                "type": "string",
                "enum": FACILITY_TYPES,
                "description": "The facility's type classification.",
            },
        },
        "required": [
            "entry_points", "surveillance_coverage_pct", "staffing_level",
            "past_incident_count", "structural_vulnerabilities", "facility_type",
        ],
        "additionalProperties": False,
    },
}


def predict_risk_score(facility_features: dict) -> dict:
    """Executes the predict_risk_score tool call. facility_features must
    contain the six fields declared in PREDICT_RISK_SCORE_TOOL['input_schema']
    (already validated against that schema if called via strict tool use)."""
    model, metadata = _load_model()

    if facility_features["facility_type"] not in metadata["facility_types"]:
        raise ValueError(
            f"Unknown facility_type {facility_features['facility_type']!r}; "
            f"must be one of {metadata['facility_types']}"
        )

    row = {feat: facility_features[feat] for feat in metadata["raw_features"]}
    for ftype in metadata["facility_types"]:
        row[f"type_{ftype}"] = facility_features["facility_type"] == ftype
    X = pd.DataFrame([row])[metadata["feature_columns"]]

    predicted_score = float(model.predict(X)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))
    risk_level = str(pd.cut([predicted_score], bins=RISK_LEVEL_BINS, labels=RISK_LEVEL_LABELS, right=False)[0])

    return {
        "predicted_risk_score": round(predicted_score, 1),
        "predicted_risk_level": risk_level,
        "model": "Random Forest (see MODEL_CARD.md)",
        "global_feature_importances": metadata["feature_importances"],
        "input_features": facility_features,
        "caveat": "Synthetic-data model, no real-world validation — see MODEL_CARD.md Limitations.",
    }


_embed_model = None
_collection = None
_claude_client = None
_claude_client_checked = False


def _load_rag_dependencies():
    global _embed_model, _collection, _claude_client, _claude_client_checked
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = chroma_client.get_collection(COLLECTION_NAME)
    if not _claude_client_checked:
        _claude_client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
        _claude_client_checked = True
    return _embed_model, _collection, _claude_client


QUERY_STANDARDS_TOOL = {
    "name": "query_standards",
    "description": (
        "Answers a question against the fabricated GFPA security/safety standards corpus, with "
        "citations back to document_id + section_id for every claim. This tool does NOT always "
        "return an answer — the response's `status` field can also be one of three distinct "
        "refusal types, which the caller MUST handle explicitly rather than assuming an answer is "
        "always present:\n"
        "  - referenced_but_missing: the question names a specific GFPA document ID that the corpus "
        "mentions but does not actually include (e.g. GFPA-001, GFPA-002, GFPA-999).\n"
        "  - low_similarity: the question is unrelated to anything in the corpus.\n"
        "  - model_declined: retrieval found plausibly-relevant excerpts, but they don't actually "
        "contain enough to answer responsibly (e.g. a topic that sounds in-scope, like fire "
        "suppression, where the corpus mentions the topic but never states an actual requirement "
        "for it).\n"
        "Never fabricate standards content to fill a refusal — when this tool refuses, relay that "
        "refusal and its reason, don't answer the underlying question from outside knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A natural-language question about the GFPA standards corpus.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}


def query_standards(question: str) -> dict:
    """Executes the query_standards tool call. Returns a dict whose `status`
    is one of: "answered", "refused", or "generation_skipped" (no API key
    configured — retrieval still ran, but no answer was generated). Callers
    must branch on `status`, not assume `answer`/`citations` are present."""
    embed_model, collection, claude_client = _load_rag_dependencies()
    result = _answer_question(question, embed_model, collection, claude_client)

    top_retrieved = [c["citation"] for c in result.retrieved[:3]]

    if result.refused:
        return {
            "status": "refused",
            "refusal_reason": result.refusal_reason,  # referenced_but_missing | low_similarity | model_declined
            "model_reasoning": result.answer,  # populated for model_declined, else None
            "top_retrieved": top_retrieved,
        }
    if result.generation_skipped:
        return {
            "status": "generation_skipped",
            "reason": "No ANTHROPIC_API_KEY configured — retrieval succeeded but no answer was generated.",
            "top_retrieved": top_retrieved,
        }
    return {
        "status": "answered",
        "answer": result.answer,
        "citations": result.citations,  # [{document_id, section_id, supporting_quote}, ...]
    }


# --- Isolated smoke test: hand-picked profiles, including a Critical case ---
TEST_PROFILES = [
    {
        "label": "Low-risk Museum",
        "features": {
            "entry_points": 5, "surveillance_coverage_pct": 90, "staffing_level": 8,
            "past_incident_count": 0, "structural_vulnerabilities": 1, "facility_type": "Museum",
        },
    },
    {
        "label": "Medium-risk Mall",
        "features": {
            "entry_points": 12, "surveillance_coverage_pct": 60, "staffing_level": 5,
            "past_incident_count": 2, "structural_vulnerabilities": 2, "facility_type": "Mall",
        },
    },
    {
        "label": "High-risk Office Tower",
        "features": {
            "entry_points": 6, "surveillance_coverage_pct": 55, "staffing_level": 3,
            "past_incident_count": 3, "structural_vulnerabilities": 2, "facility_type": "Office Tower",
        },
    },
    {
        # Deliberately built to trigger the generator's compounding condition
        # (2+ of: surveillance<35, staffing<=1, incidents>=3, vulnerabilities>=5)
        # so at least one test case should land Critical.
        "label": "Critical-tier Utility Substation (compounding)",
        "features": {
            "entry_points": 2, "surveillance_coverage_pct": 25, "staffing_level": 0,
            "past_incident_count": 4, "structural_vulnerabilities": 6, "facility_type": "Utility Substation",
        },
    },
]


TEST_QUERIES = [
    {"label": "Clean in-corpus hit (standard vocabulary)",
     "question": "Within what timeframe must a Tier 2 vulnerability be remediated?"},
    {"label": "Plain-language query (tests the retrieval-enrichment fix)",
     "question": "What surveillance coverage is required for a museum?"},
    {"label": "Referenced-but-missing",
     "question": "What does GFPA-999 say about combining multiple minor vulnerabilities?"},
    {"label": "Fire-suppression hard negative",
     "question": "What fire suppression system design standards must a facility comply with, such as sprinkler coverage or extinguisher placement density?"},
]


def main():
    print("### Tool 1: predict_risk_score ###\n")
    for case in TEST_PROFILES:
        result = predict_risk_score(case["features"])
        print(f"=== {case['label']} ===")
        print(f"  input: {case['features']}")
        print(f"  predicted_risk_score: {result['predicted_risk_score']}")
        print(f"  predicted_risk_level: {result['predicted_risk_level']}")
        print()

    print("\n### Tool 2: query_standards ###\n")
    for case in TEST_QUERIES:
        result = query_standards(case["question"])
        print(f"=== {case['label']} ===")
        print(f"  question: {case['question']!r}")
        print(f"  status: {result['status']}")
        if result["status"] == "answered":
            print(f"  answer: {result['answer']}")
            for c in result["citations"]:
                print(f"    cite: {c['document_id']} Section {c['section_id']} — \"{c['supporting_quote']}\"")
        elif result["status"] == "refused":
            print(f"  refusal_reason: {result['refusal_reason']}")
            if result["model_reasoning"]:
                print(f"  model_reasoning: {result['model_reasoning']}")
            print(f"  top_retrieved: {result['top_retrieved']}")
        else:
            print(f"  reason: {result['reason']}")
            print(f"  top_retrieved: {result['top_retrieved']}")
        print()


if __name__ == "__main__":
    main()
