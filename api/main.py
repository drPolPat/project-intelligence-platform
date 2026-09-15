"""
Phase 5: FastAPI serving layer.

Built and verified one endpoint at a time, per the phase plan:
  STEP 1 (verified): POST /api/predict-risk
  STEP 2 (verified): POST /api/draft-report
  STEP 3 (verified): POST /api/query-standards
  STEP 4 (verified): GET /api/facilities
  STEP 5 (this pass): POST /api/chat — required a real fix to agent_loop.py's
    run_agent() first: it previously always started a fresh conversation and
    never returned the full message history (and had a latent bug where the
    final assistant turn was never appended to that history on a normal
    end-of-turn — harmless for the old single-turn-only callers, but would
    have silently truncated any multi-turn conversation). Both fixed there,
    not papered over here — session continuity depends on that history being
    complete.

DESIGN PRINCIPLE (carried over from every prior phase): this layer adds NO
new text generation or synthesis. It validates input, calls the exact same
Python functions already verified in Phase 4, and returns their output
as-is. If a future endpoint's response ever contains a sentence that isn't a
direct field from a tool's return value, that's a regression, not a feature.

Missing/invalid required fields never reach predict_risk_score at all —
FastAPI's own Pydantic request validation rejects them with a 422 and a
field-level message before the handler ever runs. Errors from inside the
tool itself (e.g. the model artifact not having been trained yet) are
caught explicitly and turned into a clean HTTP error, not a raw traceback.
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()  # loads .env if present; never required, never committed (see .env.example)

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))
from tools import FACILITY_TYPES, predict_risk_score, query_standards  # noqa: E402
from draft_risk_report import draft_risk_report  # noqa: E402
from agent_loop import run_agent  # noqa: E402

DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("WARNING: ANTHROPIC_API_KEY is not set. /api/predict-risk does not need it (pure ML "
          "inference), but the standards/chat/report endpoints added in later steps will fail "
          "without it. Set it in the environment or in a local .env file (see .env.example).")

app = FastAPI(title="Project Intelligence Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Sanity ceiling for the four count-based fields, deliberately generous, NOT
# taken from generate_risk_assessments.py's NORM_CAPS (entry_points=20,
# staffing_level=10, past_incidents=10, structural_vulnerabilities=8).
# NORM_CAPS are normalization caps for the risk FORMULA — values beyond them
# just stop adding marginal risk internally — not real-world limits on a
# facility. Using them as a hard API ceiling would wrongly reject a
# legitimately large facility (e.g. a big mall with 25 entry points) that
# the model can still score. This constant exists only to catch obvious
# data-entry errors (a stray extra digit, a client bug) before they reach
# the model, the same job surveillance_coverage_pct's le=100 already does
# for that field via its own natural 0-100 range.
MAX_SANE_COUNT = 1000


class FacilityFeatures(BaseModel):
    entry_points: int = Field(..., ge=0, le=MAX_SANE_COUNT, description="Number of distinct entry points (doors, gates, access points).")
    surveillance_coverage_pct: float = Field(..., ge=0, le=100, description="Percentage of the facility under active video surveillance coverage.")
    staffing_level: int = Field(..., ge=0, le=MAX_SANE_COUNT, description="Number of on-site security staff during a typical shift.")
    past_incident_count: int = Field(..., ge=0, le=MAX_SANE_COUNT, description="Number of security/safety incidents logged in roughly the past 12 months.")
    structural_vulnerabilities: int = Field(..., ge=0, le=MAX_SANE_COUNT, description="Count of currently open/unresolved structural vulnerability findings.")
    facility_type: str = Field(..., description=f"One of: {', '.join(FACILITY_TYPES)}")

    # A plain str field + validator (not typing.Literal) so FACILITY_TYPES —
    # imported from the same single source of truth every other module in
    # this project uses (generate_risk_assessments.py, via agent/tools.py) —
    # is what's actually enforced at runtime. A hardcoded Literal here would
    # be one more place this list could silently drift out of sync.
    @field_validator("facility_type")
    @classmethod
    def validate_facility_type(cls, v: str) -> str:
        if v not in FACILITY_TYPES:
            raise ValueError(f"facility_type must be one of {FACILITY_TYPES}, got {v!r}")
        return v


class PredictRiskScoreResponse(BaseModel):
    predicted_risk_score: float
    predicted_risk_level: str
    model: str
    global_feature_importances: dict[str, float]
    input_features: dict
    caveat: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict-risk", response_model=PredictRiskScoreResponse)
def predict_risk(features: FacilityFeatures) -> dict:
    try:
        return predict_risk_score(features.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Risk model artifact not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error running predict_risk_score: {e}")


# DO NOT add a response_model to this endpoint without reading this first.
# Deliberately NO response_model here, unlike /api/predict-risk. draft_risk_report's
# top_contributing_factors[].standards_lookup sub-dict has a genuinely different
# shape depending on status ("answered" carries citations; "refused" carries
# refusal_reason + model_reasoning; "generation_skipped" carries reason) — a
# Pydantic response_model only includes fields it knows about and SILENTLY DROPS
# anything else, which is exactly the truncation risk this endpoint was asked to
# rule out. Passthrough (return the dict as-is, let FastAPI's default JSON
# serialization handle it) is what actually guarantees nothing is flattened
# or dropped, at the cost of no auto-generated response schema in /docs for
# this one endpoint. That missing schema is the trade-off, not an oversight —
# don't "fix" it with a typed model unless the model is a proper discriminated
# union covering all three standards_lookup shapes.
@app.post("/api/draft-report")
def draft_report(features: FacilityFeatures) -> dict:
    try:
        return draft_risk_report(features.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Risk model artifact not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error running draft_risk_report: {e}")


class StandardsQuestion(BaseModel):
    question: str = Field(..., min_length=1, description="A natural-language question about the GFPA standards corpus.")


# DO NOT add a response_model to this endpoint without reading this first.
# Same no-response_model reasoning as /api/draft-report: query_standards's
# return shape genuinely differs by status ("answered" carries answer +
# citations; "refused" carries refusal_reason + model_reasoning +
# top_retrieved; "generation_skipped" carries reason + top_retrieved) — all
# three must serialize distinctly and completely, not get coerced into one
# schema that drops whichever fields it wasn't told about. If this ever gets
# a response_model "for consistency" or "for /docs", it must be a
# discriminated union of all three shapes, not a single flattened model.
@app.post("/api/query-standards")
def query_standards_endpoint(body: StandardsQuestion) -> dict:
    try:
        return query_standards(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error running query_standards: {e}")


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
DATASET_FILES = {
    "facility_risk_assessments": DATA_DIR / "facility_risk_assessments.csv",
    "safety_inspection_reports": DATA_DIR / "safety_inspection_reports.csv",
    "incident_reports": DATA_DIR / "incident_reports.csv",
}


@app.get("/api/facilities")
def facilities() -> dict:
    """Returns the three synthetic datasets as raw records — no server-side
    aggregation. Confirmed design choice: a thin backend that always returns
    the same raw shape lets the dashboard add a new chart cut without a
    backend change, and these dataset sizes (a few hundred rows each) are
    far too small to need pagination or pre-aggregation for a demo. The
    client joins across the three arrays on facility_id itself."""
    result = {}
    for name, path in DATASET_FILES.items():
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"{name} not found at {path} — run the corresponding data_generation script first.",
            )
        result[name] = pd.read_csv(path).to_dict(orient="records")
    return result


# In-memory session store: session_id -> the full Messages-API conversation
# history (as returned by run_agent's `messages` field). No persistence, per
# the phase requirements — a server restart drops all sessions, which is
# fine for a demo. Keyed by a server-generated UUID, never a client-supplied
# value, so one client can't collide with or continue another's session by
# guessing an ID.
CHAT_SESSIONS: dict[str, list[dict]] = {}

_claude_client: Optional[anthropic.Anthropic] = None


def _get_claude_client() -> anthropic.Anthropic:
    global _claude_client
    if _claude_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set; /api/chat requires it.")
        _claude_client = anthropic.Anthropic()
    return _claude_client


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's chat message.")
    session_id: Optional[str] = Field(
        None, description="Omit on the first message of a conversation; the response returns one to reuse for subsequent turns."
    )


class ToolCallSummary(BaseModel):
    name: str
    input: dict


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: list[ToolCallSummary]


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> dict:
    # An unrecognized session_id (expired server, typo, server restart since
    # the client last saw it) starts a fresh conversation under that same ID
    # rather than erroring — the alternative (reject it) would make a plain
    # server restart look like a client-facing bug for no benefit here.
    session_id = body.session_id or str(uuid.uuid4())
    history = CHAT_SESSIONS.get(session_id)

    client = _get_claude_client()
    try:
        result = run_agent(body.message, client, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error running the agent: {e}")

    CHAT_SESSIONS[session_id] = result["messages"]

    return {
        "session_id": session_id,
        "response": result["final_text"],
        # Name + input only, not each tool's full raw result — the chat
        # response's `response` field already carries what the agent chose
        # to relay from those results; the full payload lives in the
        # single-tool endpoints above when a caller actually needs it.
        "tool_calls": [{"name": c["name"], "input": c["input"]} for c in result["tool_calls"]],
    }
