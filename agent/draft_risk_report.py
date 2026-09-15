"""
Phase 4, tool 3: draft_risk_report — the composite action.

Step 1 (this file, built first per the phase plan): identify_top_contributing_factors.
Pure computation, no API calls. Combines each raw risk factor's GLOBAL feature
importance (from the trained model's metadata — same for every facility,
since it's a property of the model, not the instance) with a PER-FACILITY
"gap risk" (0-1, how far that facility's own value is from a good one) to
rank which factors are driving THIS facility's score specifically.

The gap-risk normalization is deliberately NOT a new invented heuristic — it
reuses the exact same entry_risk / surveillance_gap_risk / staffing_gap_risk /
incident_risk / vulnerability_risk transforms from
generate_risk_assessments.py (NORM_CAPS) and ml/train_risk_model.py
(FACTOR_MAP), which are already the basis for RISK_WEIGHTS and the Ridge
coefficient comparison in MODEL_CARD.md. Reusing it here means
"contributing factor" means the same thing everywhere in this project.

contributing_score(feature) = global_importance(feature) * gap_risk(feature)

A factor with high importance but a good value (low gap_risk) contributes
little; a factor with a bad value but low importance also contributes
little. Only factors that are BOTH influential AND currently bad for this
facility rank highly — which is the intent of "top contributing factors,"
not just "worst factor" or "most important factor" in isolation.
"""
import json
import sys
from pathlib import Path

RUN_OUTPUT_PATH = Path(__file__).resolve().parent / "draft_risk_report_run.json"

GEN_DIR = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.insert(0, str(GEN_DIR))
from generate_risk_assessments import NORM_CAPS  # noqa: E402

from tools import PREDICT_RISK_SCORE_TOOL, TEST_PROFILES, predict_risk_score, query_standards  # noqa: E402

RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))
from chunk_documents import load_all_chunks  # noqa: E402
from citation_utils import normalize_section_id  # noqa: E402

# feature -> how to compute its 0-1 "gap risk" from a raw value, mirroring
# generate_risk_assessments.py's own normalized risk factors exactly.
_GAP_RISK_CAPS = {
    "entry_points": NORM_CAPS["entry_points"],           # higher is worse
    "staffing_level": NORM_CAPS["staffing_level"],        # LOWER is worse (inverted below)
    "past_incident_count": NORM_CAPS["past_incidents"],   # higher is worse
    "structural_vulnerabilities": NORM_CAPS["structural_vulnerabilities"],  # higher is worse
}
_LOWER_IS_WORSE = {"staffing_level"}  # surveillance_coverage_pct handled separately (0-100 scale, not a count)


def _gap_risk(feature: str, value: float) -> float:
    if feature == "surveillance_coverage_pct":
        return max(0.0, min(1.0, 1 - value / 100))
    cap = _GAP_RISK_CAPS[feature]
    normalized = max(0.0, min(1.0, value / cap))
    return (1 - normalized) if feature in _LOWER_IS_WORSE else normalized


def identify_top_contributing_factors(facility_features: dict, feature_importances: dict, top_n: int = 3) -> list[dict]:
    """facility_features: the same six fields predict_risk_score takes.
    feature_importances: predict_risk_score's returned global_feature_importances
    (raw-feature-keyed; facility_type dummies are excluded here since MODEL_CARD.md
    already established type has no direct contribution once raw factors are
    controlled for — it isn't an actionable "contributing factor" in this sense)."""
    scored = []
    for feature in _GAP_RISK_CAPS.keys() | {"surveillance_coverage_pct"}:
        value = facility_features[feature]
        gap_risk = _gap_risk(feature, value)
        importance = feature_importances.get(feature, 0.0)
        scored.append({
            "factor": feature,
            "value": value,
            "gap_risk": round(gap_risk, 3),
            "feature_importance": round(importance, 4),
            "contributing_score": round(gap_risk * importance, 4),
        })
    scored.sort(key=lambda s: s["contributing_score"], reverse=True)
    return scored[:top_n]


_FACTOR_LABELS = {
    "surveillance_coverage_pct": "Surveillance coverage",
    "staffing_level": "Staffing level",
    "structural_vulnerabilities": "Structural vulnerabilities",
    "past_incident_count": "Past incident count",
    "entry_points": "Entry point count",
}

# One templated standards question per factor. Each targets the specific
# GFPA topic that factor maps to, phrased in plain language deliberately
# (facility type name, not "Class A") to exercise the retrieval-enrichment
# fix rather than sidestep it.
_FACTOR_QUESTION_TEMPLATES = {
    "surveillance_coverage_pct": lambda f: f"What surveillance coverage is required for a {f['facility_type']}?",
    "staffing_level": lambda f: f"What staffing is required for a {f['facility_type']}?",
    "structural_vulnerabilities": lambda f: "How quickly must structural vulnerabilities be remediated, and how does the standard treat multiple concurrent vulnerabilities?",
    "past_incident_count": lambda f: "What does the standard say about handling a facility with a pattern of multiple past incidents?",
    "entry_points": lambda f: "What additional requirements apply to a facility with a higher number of entry points?",
}


def _narrative_for_factor(factor: dict) -> str:
    label = _FACTOR_LABELS[factor["factor"]]
    lookup = factor["standards_lookup"]
    header = f"- {label} (value: {factor['value']}, contributing_score: {factor['contributing_score']}):"

    if lookup["status"] == "answered":
        cite_lines = "\n".join(
            f"    Per {c['document_id']} Section {normalize_section_id(c['section_id'])}: \"{c['supporting_quote']}\""
            for c in lookup["citations"]
        )
        return f"{header}\n{cite_lines}" if cite_lines else f"{header}\n    (answered, but no citations were returned)"

    if lookup["status"] == "refused":
        return (f"{header}\n    No standards guidance found for this factor "
                f"(query_standards refused: {lookup['refusal_reason']}).")

    return (f"{header}\n    Standards lookup unavailable: {lookup['reason']}")


def _build_narrative(facility_features: dict, ml_result: dict, top_factors: list[dict]) -> str:
    lines = [
        f"Facility type: {facility_features['facility_type']}",
        f"Predicted risk score: {ml_result['predicted_risk_score']} "
        f"(risk level: {ml_result['predicted_risk_level']}) — {ml_result['model']}",
    ]

    if ml_result["predicted_risk_level"] == "Low":
        top_score = top_factors[0]["contributing_score"] if top_factors else 0.0
        lines.append(
            f"This facility's risk level is Low. The highest-ranked contributing factor's "
            f"contributing_score is {top_score} — no single factor meaningfully drives this "
            f"facility's risk score."
        )
    else:
        lines.append("Top contributing factors (ranked by feature importance x how far each factor's "
                      "value is from a low-risk value):")

    lines.append("")
    for factor in top_factors:
        lines.append(_narrative_for_factor(factor))
        lines.append("")

    lines.append(ml_result["caveat"])
    return "\n".join(lines)


# Reuses PREDICT_RISK_SCORE_TOOL's input_schema verbatim (same six facility
# fields) rather than redefining them a third time — this tool takes exactly
# the same input, it just does more with it.
DRAFT_RISK_REPORT_TOOL = {
    "name": "draft_risk_report",
    "description": (
        "Produces a full grounded risk assessment report for a facility: predicted risk score/level, "
        "the top 3 contributing risk factors (ranked by feature importance x how far this facility's own "
        "value is from a good one), and for each factor either the relevant standards citation or an "
        "explicit statement that no standards guidance was found. Use this when the user wants a "
        "comprehensive report or full assessment, not just a bare number — for a plain 'what's the risk "
        "score' request, use predict_risk_score alone instead, which is faster and doesn't run several "
        "standards lookups the user didn't ask for. The returned narrative text is ALREADY strictly "
        "grounded (every claim traces to the ML model's output or an exact standards citation, enforced "
        "by code, not by prompting). When you use this tool, your ENTIRE final response to the user must "
        "be that narrative text, character-for-character, with NOTHING added before or after it — no "
        "closing notes, caveats, transitional commentary, or synthesis connecting facts across it, even if "
        "what you'd add sounds accurate or helpful. Anything you add yourself carries none of this tool's "
        "grounding guarantee and defeats the reason it exists."
    ),
    "input_schema": PREDICT_RISK_SCORE_TOOL["input_schema"],
}


def draft_risk_report(facility_features: dict) -> dict:
    """The composite tool. Calls predict_risk_score once, then query_standards
    once per top contributing factor (always top_n=3, regardless of magnitude
    — see the decision recorded in this project's history: uniform behavior
    over a second uncalibrated threshold). Every sentence in the resulting
    narrative is either a field from ml_result, a field from a factor's own
    computation, or an exact citation/refusal-reason from query_standards —
    nothing else is added at this layer."""
    ml_result = predict_risk_score(facility_features)
    top_factors = identify_top_contributing_factors(facility_features, ml_result["global_feature_importances"])

    for factor in top_factors:
        question = _FACTOR_QUESTION_TEMPLATES[factor["factor"]](facility_features)
        factor["standards_question"] = question
        factor["standards_lookup"] = query_standards(question)

    narrative = _build_narrative(facility_features, ml_result, top_factors)

    return {
        "facility_features": facility_features,
        "ml_result": ml_result,
        "top_contributing_factors": top_factors,
        "narrative": narrative,
    }


def verify_report_grounding(report: dict) -> list[str]:
    """The sixth eval check: walks a drafted report and confirms every claim
    traces back to real evidence. Returns a list of problems found (empty =
    fully grounded). Checks:
      1. predicted_risk_score / predicted_risk_level in the narrative match
         ml_result's actual fields (byte-for-byte, not just "look similar").
      2. Every factor's value/contributing_score printed in the narrative
         matches its own computed record.
      3. Every citation's supporting_quote actually appears verbatim in that
         (document_id, section_id)'s real corpus text — the same check
         rag/eval/run_eval.py uses, applied here to the composite tool's
         own output rather than to eval reference questions.
      4. Every refused factor states its real refusal_reason, not a
         fabricated or omitted one.
    """
    problems = []
    ml_result = report["ml_result"]
    narrative = report["narrative"]

    if str(ml_result["predicted_risk_score"]) not in narrative:
        problems.append(f"predicted_risk_score {ml_result['predicted_risk_score']} not found verbatim in narrative")
    if ml_result["predicted_risk_level"] not in narrative:
        problems.append(f"predicted_risk_level {ml_result['predicted_risk_level']!r} not found verbatim in narrative")

    corpus_lookup = {(c.document_id, c.section_id): c.text for c in load_all_chunks()}
    normalize = lambda s: " ".join(s.replace("**", "").split())

    for factor in report["top_contributing_factors"]:
        if str(factor["value"]) not in narrative:
            problems.append(f"{factor['factor']}: value {factor['value']} not found verbatim in narrative")
        if str(factor["contributing_score"]) not in narrative:
            problems.append(f"{factor['factor']}: contributing_score {factor['contributing_score']} not found verbatim in narrative")

        lookup = factor["standards_lookup"]
        if lookup["status"] == "answered":
            for c in lookup["citations"]:
                key = (c["document_id"], normalize_section_id(c["section_id"]))
                true_text = corpus_lookup.get(key)
                if true_text is None:
                    problems.append(f"{factor['factor']}: cited {key} does not exist in the corpus at all")
                elif normalize(c["supporting_quote"]) not in normalize(true_text):
                    problems.append(f"{factor['factor']}: supporting_quote for {key} not found verbatim in that section's real text")
                if c["document_id"] not in narrative or c["supporting_quote"] not in narrative:
                    problems.append(f"{factor['factor']}: citation for {key} not actually printed in the narrative text")
        elif lookup["status"] == "refused":
            if lookup["refusal_reason"] not in narrative:
                problems.append(f"{factor['factor']}: refusal_reason {lookup['refusal_reason']!r} not stated in narrative")
        elif lookup["status"] == "generation_skipped":
            if lookup["reason"] not in narrative:
                problems.append(f"{factor['factor']}: generation_skipped reason not stated in narrative")

    return problems


def main():
    print("### Step 1 only: identify_top_contributing_factors on all 4 profiles ###\n")
    for case in TEST_PROFILES:
        result = predict_risk_score(case["features"])
        top_factors = identify_top_contributing_factors(case["features"], result["global_feature_importances"])
        print(f"=== {case['label']} (score={result['predicted_risk_score']}, level={result['predicted_risk_level']}) ===")
        for f in top_factors:
            print(f"  {f['factor']:28s} value={f['value']!s:6s} gap_risk={f['gap_risk']:.3f}  "
                  f"importance={f['feature_importance']:.4f}  contributing_score={f['contributing_score']:.4f}")
        print()

    print("\n### Full composite: draft_risk_report on all 4 profiles ###\n")
    all_reports = []
    for case in TEST_PROFILES:
        report = draft_risk_report(case["features"])
        print(f"=== {case['label']} ===")
        print(report["narrative"])
        problems = verify_report_grounding(report)
        print(f"[grounding check: {'PASS' if not problems else 'FAIL'}{'' if not problems else ' - ' + '; '.join(problems)}]")
        print()
        all_reports.append({"label": case["label"], "report": report, "grounding_problems": problems})

    # Persist raw output (including exact citation dicts as returned by the
    # model, before any display formatting) so a failing grounding check can
    # be diagnosed byte-for-byte after the fact, not just re-described from
    # memory of the terminal output.
    RUN_OUTPUT_PATH.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full run output (raw citations included) to {RUN_OUTPUT_PATH}")

    print("\n### Citation repr() diagnostic (checks for literal escape-sequence text, not just display garbling) ###\n")
    corpus_lookup = {(c.document_id, c.section_id): c.text for c in load_all_chunks()}
    for entry in all_reports:
        for factor in entry["report"]["top_contributing_factors"]:
            lookup = factor["standards_lookup"]
            if lookup["status"] != "answered":
                continue
            for c in lookup["citations"]:
                key = (c["document_id"], normalize_section_id(c["section_id"]))
                true_text = corpus_lookup.get(key, "<NOT FOUND IN CORPUS>")
                quote = c["supporting_quote"]
                has_literal_escape = "\\u" in quote  # a REAL em-dash char never contains the 2 ASCII chars "\" + "u"
                print(f"[{entry['label']}] {key}  literal_escape_sequence_present={has_literal_escape}")
                print(f"  supporting_quote repr(): {quote!r}")
                print(f"  corpus text repr():      {true_text!r}")
                print()

    print("\n### Dedicated refusal-handling test (real refusal injected, not mocked) ###\n")
    print("This sandbox has no ANTHROPIC_API_KEY, so a naturally-occurring model_declined or")
    print("low_similarity refusal can't be produced live right now. referenced_but_missing is")
    print("the one refusal type that's fully deterministic (caught before any API call), so it's")
    print("used here to exercise the refusal-rendering code path with REAL pipeline output —")
    print("this is a real refused query_standards() call, injected into one factor slot,")
    print("not a fabricated/mocked result.\n")
    case = TEST_PROFILES[3]  # Critical-tier Utility Substation
    report = draft_risk_report(case["features"])
    real_refusal = query_standards("What does GFPA-999 say about combining multiple minor vulnerabilities?")
    assert real_refusal["status"] == "refused" and real_refusal["refusal_reason"] == "referenced_but_missing", \
        f"expected a real referenced_but_missing refusal, got {real_refusal}"
    report["top_contributing_factors"][0]["standards_lookup"] = real_refusal
    report["narrative"] = _build_narrative(case["features"], report["ml_result"], report["top_contributing_factors"])
    print(report["narrative"])
    problems = verify_report_grounding(report)
    print(f"[grounding check: {'PASS' if not problems else 'FAIL'}{'' if not problems else ' - ' + '; '.join(problems)}]")


if __name__ == "__main__":
    main()
