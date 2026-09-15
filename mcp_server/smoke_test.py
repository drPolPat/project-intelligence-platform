"""
Ad hoc verification script, not part of the server itself: connects an
in-memory FastMCP Client directly to the `mcp` server object (no subprocess,
no transport) to confirm tool registration and run the 4 hand-picked
facility profiles from agent/tools.py's own TEST_PROFILES through the MCP
tool, exactly as specified by the current step's test plan.
"""
import asyncio
import json

from fastmcp import Client

from server import mcp
from tools import TEST_PROFILES, TEST_QUERIES  # agent/tools.py; already on sys.path via server.py

# The Critical-tier profile, reused for draft_risk_report — same convention
# as View 2's testing throughout this project (richest case: multiple
# contributing factors, real chance of a mix of answered/refused lookups).
_CRITICAL_PROFILE = TEST_PROFILES[3]


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print(f"Registered tools ({len(tools)}):")
        for t in tools:
            print(f"  - {t.name}")
        print()
        print("Full input schema for predict_risk_score:")
        predict_tool = next(t for t in tools if t.name == "predict_risk_score")
        print(json.dumps(predict_tool.inputSchema, indent=2))
        print()

        print("=== 4 hand-picked facility profiles (agent/tools.py TEST_PROFILES) ===\n")
        for case in TEST_PROFILES:
            result = await client.call_tool("predict_risk_score", case["features"])
            data = result.data
            print(f"=== {case['label']} ===")
            print(f"  input: {case['features']}")
            print(f"  predicted_risk_score: {data['predicted_risk_score']}")
            print(f"  predicted_risk_level: {data['predicted_risk_level']}")
            print()

        print("=== 4 canonical standards queries (agent/tools.py TEST_QUERIES) ===\n")
        for case in TEST_QUERIES:
            result = await client.call_tool("query_standards", {"question": case["question"]})
            data = result.data
            print(f"=== {case['label']} ===")
            print(f"  question: {case['question']!r}")
            print(f"  status: {data['status']}")
            if data["status"] == "answered":
                print(f"  answer: {data['answer']}")
                for c in data["citations"]:
                    print(f"    cite: {c['document_id']} Section {c['section_id']} — \"{c['supporting_quote']}\"")
            elif data["status"] == "refused":
                print(f"  refusal_reason: {data['refusal_reason']}")
                if data["model_reasoning"]:
                    print(f"  model_reasoning: {data['model_reasoning']}")
            else:
                print(f"  reason: {data['reason']}")
            print()

        print(f"=== draft_risk_report: {_CRITICAL_PROFILE['label']} ===\n")
        result = await client.call_tool("draft_risk_report", _CRITICAL_PROFILE["features"])
        data = result.data
        print(f"  ml_result: score={data['ml_result']['predicted_risk_score']}, "
              f"level={data['ml_result']['predicted_risk_level']}")
        print(f"  top_contributing_factors ({len(data['top_contributing_factors'])}):")
        for factor in data["top_contributing_factors"]:
            lookup = factor["standards_lookup"]
            print(f"    - {factor['factor']} (value={factor['value']}, "
                  f"contributing_score={factor['contributing_score']}): standards_lookup.status={lookup['status']}")
        print()
        print("  narrative:")
        print(data["narrative"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
