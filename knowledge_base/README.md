# Synthetic Standards Knowledge Base

**Fictional standards body and documents, created for demonstration purposes only — not real regulatory or compliance guidance.**

This is a small corpus of fabricated "security/safety standards" documents used as the RAG layer's source material for Phase 3 of this project. None of it is real regulatory or compliance text — see the constraints below.

## Fictional framework

- **Issuing body:** Global Facility Protection Alliance (GFPA) — an invented organization. It does not exist and is not modeled on any specific real standards body.
- **Numbering:** documents are numbered `GFPA-NNN`, an invented scheme. Any resemblance of the numbering style to real standards numbering (e.g. a body name + hyphen + number) is coincidental to how such documents are conventionally formatted in general, not a reference to any specific real standard, series, or publisher.
- A few documents cross-reference companion standards that are **not included** in this demonstration corpus (e.g. GFPA-001, GFPA-002, GFPA-999) — mentioned only for narrative realism (real standards frameworks cross-reference each other extensively); the RAG system's refusal behavior is expected to correctly decline to answer questions that would require one of these unwritten documents.

## Documents

| ID | Title | Maps to structured-data factor |
|---|---|---|
| [GFPA-201](standards/GFPA-201_surveillance-coverage.md) | Perimeter and Interior Surveillance Coverage Requirements | `surveillance_coverage_pct` |
| [GFPA-210](standards/GFPA-210_staffing-ratios.md) | Minimum Staffing Ratios for Physical Security Personnel | `staffing_level` |
| [GFPA-215](standards/GFPA-215_structural-vulnerability-remediation.md) | Structural Vulnerability Identification and Remediation Timelines | `structural_vulnerabilities` |
| [GFPA-230](standards/GFPA-230_incident-reporting-escalation.md) | Incident and Near-Miss Reporting, Classification, and Escalation Procedures | `incident_reports.csv` (severity/category) |
| [GFPA-240](standards/GFPA-240_inspection-frequency.md) | Facility Safety Inspection Frequency, Scope, and Checklist Requirements | `safety_inspection_reports.csv` (overall_result) |
| [GFPA-250](standards/GFPA-250_access-point-control.md) | Access Point and Entry Control Management | `entry_points` |

Two intentional consistency points with the synthetic structured data (Phases 1-2), useful for later cross-referencing demos:
- GFPA-230's severity scale (Near-Miss / Minor / Moderate / Major / Critical) matches `incident_reports.csv`'s `severity` field exactly.
- GFPA-240's Pass / Pass with Corrective Actions / Fail thresholds (0 / 1-2 / 3+ deficiencies) match `safety_inspection_reports.csv`'s `overall_result` logic exactly.
- GFPA-215 Section 5 and GFPA-201/210's cross-references explicitly describe the same "compounding risk from multiple simultaneous gaps" concept that drives the Critical tier in `facility_risk_assessments.csv` and the ML model card's headline finding — without stating the generator's exact formula or thresholds.

## Constraints (do not violate when adding documents)

- No real standards body names, acronyms, or numbering schemes (no NFPA, ISO, EN, ANSI, OSHA, UL, IEC, ASTM, CSA, or similar).
- No text paraphrased closely enough from a real standard to be recognizable as derived from it.
- No real jurisdiction, law, or regulatory citation.
- Every document must carry the disclaimer line at the top, verbatim.
