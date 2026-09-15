"""
MCP server exposing this project's three agent tools (predict_risk_score,
query_standards, draft_risk_report) to any MCP-compatible client (Claude
Desktop, Claude Code, or another agent framework) — additive, not a
replacement for agent/agent_loop.py's Messages-API tool-calling loop, which
keeps orchestrating these same functions for the deployed chat endpoint.

Single-source-of-truth discipline, same as the rest of this project: this
module imports each tool function and its Messages-API tool definition
(PREDICT_RISK_SCORE_TOOL, QUERY_STANDARDS_TOOL, DRAFT_RISK_REPORT_TOOL)
directly from agent/tools.py and agent/draft_risk_report.py rather than
reimplementing any logic or rewriting a single description. The RF-over-Ridge
rationale, the three-way refusal contract, and the grounding/verbatim-relay
constraint on draft_risk_report's output are all reused verbatim as these MCP
tools' descriptions too.

query_standards and draft_risk_report need ANTHROPIC_API_KEY. Per the
isolated-environment design, that comes from THIS directory's own .env
(mcp_server/.env, gitignored, not the main project's) — see .env.example.

Runs in its own venv (mcp_server/.venv, mcp_server/requirements.txt) —
fastmcp's MCP SDK dependency requires a newer starlette than the deployed
API's fastapi==0.115.6 pin tolerates; installing it into the shared venv
broke FastAPI app construction outright (see mcp_server/requirements.txt for
the reproduction). Isolating the venv, not pinning starlette down, is the
fix, since the two servers are independently deployable and never need to
share a runtime.
"""
import sys
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field

# Isolated-environment design: this server's own .env (mcp_server/.env, never
# the main project's) supplies ANTHROPIC_API_KEY for query_standards/
# draft_risk_report. Loaded before importing tools.py so its module-level
# RAG-client check (os.environ.get("ANTHROPIC_API_KEY")) sees it.
load_dotenv(Path(__file__).resolve().parent / ".env")

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))
from tools import PREDICT_RISK_SCORE_TOOL, QUERY_STANDARDS_TOOL, predict_risk_score, query_standards  # noqa: E402
from draft_risk_report import DRAFT_RISK_REPORT_TOOL, draft_risk_report  # noqa: E402

mcp = FastMCP(name="project-intelligence-platform")

_PROPS = PREDICT_RISK_SCORE_TOOL["input_schema"]["properties"]
_FACILITY_TYPES = tuple(_PROPS["facility_type"]["enum"])


@mcp.tool(name="predict_risk_score", description=PREDICT_RISK_SCORE_TOOL["description"])
def predict_risk_score_tool(
    entry_points: Annotated[int, Field(description=_PROPS["entry_points"]["description"], ge=0)],
    surveillance_coverage_pct: Annotated[
        float, Field(description=_PROPS["surveillance_coverage_pct"]["description"], ge=0, le=100)
    ],
    staffing_level: Annotated[int, Field(description=_PROPS["staffing_level"]["description"], ge=0)],
    past_incident_count: Annotated[int, Field(description=_PROPS["past_incident_count"]["description"], ge=0)],
    structural_vulnerabilities: Annotated[
        int, Field(description=_PROPS["structural_vulnerabilities"]["description"], ge=0)
    ],
    facility_type: Annotated[Literal[_FACILITY_TYPES], Field(description=_PROPS["facility_type"]["description"])],
) -> dict:
    return predict_risk_score({
        "entry_points": entry_points,
        "surveillance_coverage_pct": surveillance_coverage_pct,
        "staffing_level": staffing_level,
        "past_incident_count": past_incident_count,
        "structural_vulnerabilities": structural_vulnerabilities,
        "facility_type": facility_type,
    })


@mcp.tool(name="query_standards", description=QUERY_STANDARDS_TOOL["description"])
def query_standards_tool(
    question: Annotated[
        str, Field(description=QUERY_STANDARDS_TOOL["input_schema"]["properties"]["question"]["description"])
    ],
) -> dict:
    return query_standards(question)


# draft_risk_report takes the same 6 facility fields as predict_risk_score
# (DRAFT_RISK_REPORT_TOOL["input_schema"] IS PREDICT_RISK_SCORE_TOOL's schema
# object, not a copy — see agent/draft_risk_report.py). FastMCP builds each
# tool's schema from its own function signature, so the parameter list is
# repeated here rather than shared; the descriptions and enum values below
# still come from that same _PROPS dict, not retyped.
@mcp.tool(name="draft_risk_report", description=DRAFT_RISK_REPORT_TOOL["description"])
def draft_risk_report_tool(
    entry_points: Annotated[int, Field(description=_PROPS["entry_points"]["description"], ge=0)],
    surveillance_coverage_pct: Annotated[
        float, Field(description=_PROPS["surveillance_coverage_pct"]["description"], ge=0, le=100)
    ],
    staffing_level: Annotated[int, Field(description=_PROPS["staffing_level"]["description"], ge=0)],
    past_incident_count: Annotated[int, Field(description=_PROPS["past_incident_count"]["description"], ge=0)],
    structural_vulnerabilities: Annotated[
        int, Field(description=_PROPS["structural_vulnerabilities"]["description"], ge=0)
    ],
    facility_type: Annotated[Literal[_FACILITY_TYPES], Field(description=_PROPS["facility_type"]["description"])],
) -> dict:
    return draft_risk_report({
        "entry_points": entry_points,
        "surveillance_coverage_pct": surveillance_coverage_pct,
        "staffing_level": staffing_level,
        "past_incident_count": past_incident_count,
        "structural_vulnerabilities": structural_vulnerabilities,
        "facility_type": facility_type,
    })


if __name__ == "__main__":
    mcp.run()
