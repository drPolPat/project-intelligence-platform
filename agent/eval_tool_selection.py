"""
Small eval: does the agent select the right tool(s) for each request type?

T1-T6: stage 1 (predict_risk_score + query_standards). T7-T8: stage 2 adds
draft_risk_report and checks the boundary against predict_risk_score. T9:
a grounding regression check, not a tool-selection check — added after a
real failure was found in an earlier run, where the agent correctly called
draft_risk_report alone (tool selection: PASS) but then appended a
self-generated closing note after relaying the tool's narrative. That note
was accurate-sounding but was a genuine grounding violation (synthesizing a
new claim by connecting two separate citations), which tool-selection
matching alone can never catch — hence T9's stricter exact-string check.

Tool-selection correctness (which tool names got called) is checked
automatically by comparing against expected_tools. Whether a clarifying
question or a decline is actually WORDED well (T5/T6) is flagged for manual
review instead of automated — judging that well would need its own
LLM-judge call, which is more machinery than a small smoke eval warrants.
A general grounding checker for the agent's own free-text responses
(T2/T3/T4-style answers the agent composes around tool results, not just
the T9 draft_risk_report passthrough case) is scoped as separate follow-up
work, not built here.
"""
import json
import os
import sys
from pathlib import Path

import anthropic

from agent_loop import run_agent

RUN_OUTPUT_PATH = Path(__file__).resolve().parent / "eval_tool_selection_run.json"

TEST_CASES = [
    {
        "id": "T1",
        "label": "Pure predict_risk_score",
        "request": "What's the risk score for a Mall with entry_points=10, surveillance_coverage_pct=50, "
                   "staffing_level=4, past_incident_count=2, structural_vulnerabilities=2?",
        "expected_tools": {"predict_risk_score"},
        "expect_clarification": False,
    },
    {
        "id": "T2",
        "label": "Pure query_standards (standard's own vocabulary)",
        "request": "What does the standard say about staffing minimums for malls?",
        "expected_tools": {"query_standards"},
        "expect_clarification": False,
    },
    {
        "id": "T3",
        "label": "Pure query_standards (plain-language phrasing)",
        "request": "How quickly must a critical structural vulnerability be fixed?",
        "expected_tools": {"query_standards"},
        "expect_clarification": False,
    },
    {
        "id": "T4",
        "label": "Combination: both tools needed in one request",
        "request": "What surveillance coverage does a museum need, and also give me the risk score for a "
                   "museum with surveillance_coverage_pct=90, staffing_level=8, entry_points=5, "
                   "past_incident_count=0, structural_vulnerabilities=1?",
        "expected_tools": {"predict_risk_score", "query_standards"},
        "expect_clarification": False,
    },
    {
        "id": "T5",
        "label": "Ambiguous/incomplete data -> should ask a clarifying question, not guess",
        "request": "What's the risk score for a facility?",
        "expected_tools": set(),
        "expect_clarification": True,
    },
    {
        "id": "T6",
        "label": "Out of scope -> should decline gracefully, not force a tool call",
        "request": "What's the weather in Paris?",
        "expected_tools": set(),
        "expect_clarification": False,
    },
    {
        "id": "T7",
        "label": "Full report request -> draft_risk_report, NOT predict_risk_score directly",
        "request": "Give me a full risk assessment for a Utility Substation with entry_points=2, "
                   "surveillance_coverage_pct=25, staffing_level=0, past_incident_count=4, "
                   "structural_vulnerabilities=6.",
        "expected_tools": {"draft_risk_report"},
        "expect_clarification": False,
    },
    {
        "id": "T8",
        "label": "Regression check: bare score request still uses predict_risk_score alone, "
                 "not the heavier draft_risk_report, now that a 3rd tool exists",
        "request": "What's the risk score for an Office Tower with entry_points=6, "
                   "surveillance_coverage_pct=55, staffing_level=3, past_incident_count=3, "
                   "structural_vulnerabilities=2?",
        "expected_tools": {"predict_risk_score"},
        "expect_clarification": False,
    },
    {
        "id": "T9",
        "label": "Grounding regression check: agent's final_text for a draft_risk_report-only "
                 "request must be an EXACT match to the tool's own narrative — nothing added",
        "request": "Give me a full risk assessment for a Utility Substation with entry_points=2, "
                   "surveillance_coverage_pct=25, staffing_level=0, past_incident_count=4, "
                   "structural_vulnerabilities=6.",
        "expected_tools": {"draft_risk_report"},
        "expect_clarification": False,
        "verify_exact_narrative_match": True,
    },
]


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set — this eval needs the real model making real "
              "tool-selection decisions; there's no meaningful way to fake this.")
        return
    client = anthropic.Anthropic()

    n_correct = 0
    all_results = []
    for case in TEST_CASES:
        result = run_agent(case["request"], client)
        actual_tools = {c["name"] for c in result["tool_calls"]}
        tools_correct = actual_tools == case["expected_tools"]

        print(f"=== {case['id']}: {case['label']} ===")
        print(f"  request: {case['request']!r}")
        print(f"  expected_tools: {case['expected_tools'] or '(none)'}")
        print(f"  actual_tools (unique names): {actual_tools or '(none)'}")
        # Full per-call sequence, not deduplicated — a set of unique tool
        # names can't distinguish "called query_standards once" from "called
        # it three times with reformulated questions before answering," and
        # that distinction turned out to matter for a real result (T2).
        print(f"  actual tool calls in order ({len(result['tool_calls'])} total):")
        for i, c in enumerate(result["tool_calls"], start=1):
            print(f"    {i}. {c['name']}(input={c['input']!r}, is_error={c['is_error']})")
        print(f"  tool selection: {'PASS' if tools_correct else 'FAIL'}")
        print(f"  final_text: {result['final_text']}")
        if case["expect_clarification"]:
            print("  [manual check] does final_text actually ask which fields are missing, "
                  "rather than guessing values or refusing outright?")

        if case.get("verify_exact_narrative_match"):
            # Compare against what THIS call's draft_risk_report actually
            # returned to the agent — never a freshly recomputed call, since
            # query_standards makes live, non-deterministic LLM calls
            # internally and a fresh call could legitimately produce
            # different (but equally valid) phrasing.
            report_calls = [c for c in result["tool_calls"] if c["name"] == "draft_risk_report"]
            if not report_calls:
                print("  grounding (exact narrative match): FAIL - draft_risk_report was never actually called")
                tools_correct = False
            else:
                actual_narrative = json.loads(report_calls[0]["result"])["narrative"]
                exact_match = result["final_text"] == actual_narrative
                print(f"  grounding (exact narrative match): {'PASS' if exact_match else 'FAIL'}")
                if not exact_match:
                    print(f"    tool's own narrative (what the agent actually received):\n{actual_narrative}")
                    print(f"    agent's final_text (what it actually sent the user):\n{result['final_text']}")
                tools_correct = tools_correct and exact_match

        print()

        n_correct += tools_correct
        # Only the fields this script (or a reader of its saved JSON) ever
        # actually uses — not the full run_agent() result. result["messages"]
        # in particular holds raw Anthropic SDK content-block objects
        # (see agent_loop.py's run_agent: `"content": response.content`),
        # which are not JSON-serializable; api/main.py never runs into this
        # because it only ever extracts final_text/tool_calls for its HTTP
        # response, but this eval was passing the whole dict through to
        # json.dumps() and crashing before it could ever write output or
        # report a real pass/fail count.
        all_results.append({
            "id": case["id"],
            "label": case["label"],
            "request": case["request"],
            "result": {
                "final_text": result["final_text"],
                "tool_calls": result["tool_calls"],
                "stop_reason": result["stop_reason"],
            },
        })

    print(f"Tool-selection accuracy: {n_correct}/{len(TEST_CASES)}")

    RUN_OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full run output (every tool call, in order, with inputs and results) to {RUN_OUTPUT_PATH}")

    # Gate for CI (.github/workflows/ci.yml): this eval has run at 9/9 since
    # T9 was added; a regression below that baseline should fail the build,
    # not just get printed and ignored. Manual local runs get the same
    # signal via exit code, which this script never surfaced before.
    if n_correct < len(TEST_CASES):
        print(f"FAIL: tool-selection accuracy {n_correct}/{len(TEST_CASES)} is below the {len(TEST_CASES)}/{len(TEST_CASES)} baseline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
