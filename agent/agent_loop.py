"""
Phase 4: manual tool-calling loop (not the SDK's beta tool runner, not
LangGraph — a plain while-loop, matching how SteelSense AI's first pass
worked, per the user's explicit request to compare approaches sequentially).

STAGE 1 was predict_risk_score + query_standards only, confirmed at 6/6 on
the tool-selection eval (agent/eval_tool_selection.py) before this file was
touched again. STAGE 2 (this version) adds draft_risk_report as a third
option now that the two-tool distinction is solid — the risk this stage
specifically has to guard against is the agent reaching for the heavier
composite tool on a request a single predict_risk_score call would have
answered just as well; see the system prompt's tool-selection rule and
eval_tool_selection.py's new T7/T8 cases, which test exactly that boundary.

GROUNDING DISCIPLINE (extended from draft_risk_report to the agent's own
framing text): the agent's final response, when it incorporates tool
results, must not add claims beyond what predict_risk_score or
query_standards actually returned. This is enforced via the system prompt
below (an instruction, not a code-level guarantee like draft_risk_report's
deterministic template) — because at this layer the agent IS the one
composing free text, unlike draft_risk_report's Python-templated narrative.
That's a real, weaker guarantee than draft_risk_report's, and worth being
honest about rather than claiming equivalent strictness. draft_risk_report's
OWN narrative is the one exception where the agent should relay text
verbatim rather than compose it, precisely because that text already carries
the stronger, code-enforced guarantee — re-paraphrasing it would trade a
strong guarantee for a weak one.
"""
import json
import os
from typing import Optional

import anthropic

from draft_risk_report import DRAFT_RISK_REPORT_TOOL, draft_risk_report
from tools import PREDICT_RISK_SCORE_TOOL, QUERY_STANDARDS_TOOL, predict_risk_score, query_standards

MODEL_NAME = "claude-opus-5"
MAX_TOOL_ITERATIONS = 5  # hard ceiling so a stuck loop can't run away

TOOLS = [PREDICT_RISK_SCORE_TOOL, QUERY_STANDARDS_TOOL, DRAFT_RISK_REPORT_TOOL]

TOOL_REGISTRY = {
    "predict_risk_score": lambda tool_input: predict_risk_score(tool_input),
    "query_standards": lambda tool_input: query_standards(tool_input["question"]),
    "draft_risk_report": lambda tool_input: draft_risk_report(tool_input),
}

SYSTEM_PROMPT = """You are a facility risk-assessment assistant with access to three tools:

- predict_risk_score: predicts a numeric risk score/level for a facility from its raw risk factors (entry_points, surveillance_coverage_pct, staffing_level, past_incident_count, structural_vulnerabilities, facility_type). ALL SIX fields are required. Use this for a bare "what's the risk score" request.
- query_standards: answers a question against a fabricated security/safety standards corpus, with citations. It can also refuse (referenced_but_missing, low_similarity, or model_declined) — when it does, relay that refusal and its reason plainly. Never answer a standards question yourself from outside knowledge when this tool is available or has refused.
- draft_risk_report: a composite tool taking the same six facility fields as predict_risk_score, but producing a full report (score, top contributing factors, and grounded standards context for each). Use this ONLY when the user asks for a full/comprehensive assessment or report — not for a bare score request, which predict_risk_score alone answers faster and without extra standards lookups the user didn't ask for. draft_risk_report's returned narrative is already strictly grounded by code, not just by prompting — relay it as-is rather than paraphrasing it; re-summarizing it in your own words could reintroduce exactly the unsupported-claim risk it exists to prevent.

Rules:
1. GROUNDING: when you incorporate a tool's result into your response, state only what the tool actually returned — the numeric score/level from predict_risk_score, the exact answer/citations (or refusal) from query_standards, or draft_risk_report's narrative relayed as-is. Do not add your own judgment, advice, or "common sense" framing connecting facts beyond stating them side by side. Do not invent standards content to fill a query_standards refusal.
2. MISSING DATA: if a request needs predict_risk_score or draft_risk_report but is missing one or more of the six required fields, do NOT guess or invent plausible-sounding values. Ask the user a clarifying question naming exactly which fields are missing, and do not call either tool until you have them.
3. OUT OF SCOPE: if a request has nothing to do with facility risk assessment or the security/safety standards corpus, say so directly and do not call any tool just to produce some kind of answer.
4. Call query_standards for questions about requirements, standards, or "what does the standard say" — even when phrased in plain language (a facility type name, or words like "critical"/"urgent") rather than the standard's own vocabulary (Class A, Tier 1).
5. Choosing between predict_risk_score and draft_risk_report: a request for "the risk score," "a risk rating," or similar wants predict_risk_score alone. A request for "a full assessment," "a risk report," "an evaluation with context," or similar wants draft_risk_report. When genuinely unsure which the user wants, ask rather than picking the heavier tool by default.
6. A single request may legitimately need more than one tool (e.g. predict_risk_score plus a separate query_standards question) — call whichever are actually needed. But never call BOTH predict_risk_score and draft_risk_report for the same facility in the same request — draft_risk_report already includes a risk score, so that combination is always redundant.
7. When draft_risk_report is the only tool a request needs, your ENTIRE final response must be its returned narrative field, copied character-for-character — nothing added before it, nothing added after it. No closing note, no caveat, no "one thing worth mentioning," no transition sentence, even if it would be accurate. That narrative already carries a code-enforced grounding guarantee; one sentence of your own next to it does not, and turns a fully-grounded report into a partially-grounded one."""


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Returns (content_string, is_error)."""
    if name not in TOOL_REGISTRY:
        return f"Error: unknown tool {name!r}", True
    try:
        result = TOOL_REGISTRY[name](tool_input)
        return json.dumps(result), False
    except Exception as e:
        return f"Error executing {name}: {e}", True


def run_agent(user_message: str, client: anthropic.Anthropic, history: Optional[list[dict]] = None) -> dict:
    """Runs the manual tool-calling loop for one user message. `history`, if
    given, is a prior call's returned `messages` list — passing it back in
    is what makes multi-turn conversation possible (used by /api/chat's
    session store); omitting it starts a fresh conversation, unchanged from
    every prior single-turn caller (eval_tool_selection.py, this module's
    own demo). Returns a dict with `final_text`, `tool_calls` (a log of
    every tool invoked, its input, whether it errored, AND its raw result
    content — captured so a grounding check can compare the agent's
    final_text against what THIS call's tool actually returned, never a
    freshly-recomputed "ground truth", since query_standards makes live,
    non-deterministic LLM calls internally), and `messages` (the full
    updated conversation, including the final assistant turn — this used to
    NOT be appended before returning on a non-tool-use stop, which silently
    truncated the history; fixed here since a multi-turn caller needs the
    complete conversation to replay next turn, not just the tool-use ones)."""
    messages = list(history) if history else []
    messages.append({"role": "user", "content": user_message})
    tool_calls_log = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            return {"final_text": final_text, "tool_calls": tool_calls_log, "stop_reason": response.stop_reason, "messages": messages}

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            content, is_error = execute_tool(block.name, block.input)
            tool_calls_log.append({"name": block.name, "input": block.input, "is_error": is_error, "result": content})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})

    return {"final_text": "(gave up after MAX_TOOL_ITERATIONS)", "tool_calls": tool_calls_log, "stop_reason": "max_iterations", "messages": messages}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set. The tool-calling loop needs a real model to "
              "decide which tools to call — there's no meaningful way to smoke-test this without it.")
        return
    client = anthropic.Anthropic()

    demo_requests = [
        "What's the risk score for a Mall with entry_points=10, surveillance_coverage_pct=50, "
        "staffing_level=4, past_incident_count=2, structural_vulnerabilities=2?",
        "What does the standard say about staffing minimums for malls?",
    ]
    for req in demo_requests:
        print(f"=== Request: {req!r} ===")
        result = run_agent(req, client)
        print(f"tool_calls: {[(c['name'], c['input']) for c in result['tool_calls']]}")
        print(f"final_text: {result['final_text']}")
        print()


if __name__ == "__main__":
    main()
