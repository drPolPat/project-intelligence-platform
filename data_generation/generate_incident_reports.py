"""
Synthetic incident / near-miss report generator.

Reuses the same 120-facility pool as facility_risk_assessments.csv, the same
way generate_inspection_reports.py does.

Anti-leakage design note (important): incident frequency and severity here
are generated from each facility's RAW risk factors (surveillance_coverage_pct,
staffing_level, structural_vulnerabilities, facility_type) — never from
past_incident_count or computed_risk_score in the risk-assessment table.
Both datasets are independent noisy realizations of the same underlying
facility risk propensity, drawn with their own Poisson processes, own
sensitivity constants, and their own RNG stream/seed. Concretely, this
dataset's per-facility incident count will generally NOT equal that
facility's past_incident_count — that's intentional. If a future model uses
risk-assessment features to predict incidents, the "past_incident_count"
feature must not be a disguised copy of what it's predicting.
"""
import numpy as np
import pandas as pd

from generate_inspection_reports import latest_risk_profile

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

SEED = 44  # independent of both the risk-assessment (42) and inspection (43) seeds
RISK_ASSESSMENTS_PATH = "data/synthetic/facility_risk_assessments.csv"
OUTPUT_PATH = "data/synthetic/incident_reports.csv"

NORM_CAPS = {"staffing_level": 10, "structural_vulnerabilities": 8}

# Expected incident/near-miss count per facility over a ~24-month window.
# Deliberately higher than the risk assessments' past-12-month incident
# lambdas (Museum 0.8 / Mall 2.5 / Office 1.0 / Utility 1.2) since this
# dataset also captures near-misses, not just realized incidents — another
# reason the two datasets' counts won't line up 1:1.
BASE_LAMBDA_BY_TYPE = {
    "Museum": 1.5,
    "Mall": 4.0,
    "Office Tower": 1.8,
    "Utility Substation": 2.2,
}

# How much a facility's own risk gaps inflate its expected incident count,
# on top of the facility-type baseline. Own tuned constants, distinct from
# the sensitivities used for past_incident_count in generate_risk_assessments.py.
FREQUENCY_SENSITIVITY = {
    "surveillance": 0.8,
    "staffing": 0.5,
    "structural": 0.6,
}

# (category, driver risk factor, base selection weight)
# Categories skew toward whichever risk factor plausibly causes them; the
# selection weight is inflated by that facility's level of the driver factor,
# so poorly-surveilled facilities see proportionally more access/theft/
# vandalism reports, structurally-compromised ones more facility-hazard
# reports, and understaffed ones more procedural-lapse reports.
CATEGORIES = [
    ("Unauthorized Access", "surveillance", 1.0),
    ("Theft / Property Loss", "surveillance", 1.0),
    ("Vandalism", "surveillance", 0.7),
    ("Structural / Facility Hazard", "structural", 1.0),
    ("Fire / Life-Safety Hazard", "structural", 0.8),
    ("Staffing / Procedural Lapse", "staffing", 0.9),
]
CATEGORY_RISK_SENSITIVITY = 2.5  # how strongly the driver factor skews category odds

SEVERITY_LEVELS = ["Near-Miss", "Minor", "Moderate", "Major", "Critical"]
# Tuned (see module docstring-adjacent sweep in dev notes) so the population-
# wide shape is a realistic safety pyramid — Near-Miss/Minor dominant, a
# meaningful Moderate band, and Major/Critical reserved for the highest-risk
# facilities — rather than every facility landing in the middle bands.
SEVERITY_BASE_MEAN = -0.5        # latent index at zero combined risk (clips to Near-Miss)
SEVERITY_RISK_SENSITIVITY = 3.2  # added to the latent index at worst-case combined risk (bias = 1.0)
SEVERITY_NOISE_STD = 1.1
SEVERITY_WEIGHTS = {"surveillance": 0.40, "staffing": 0.25, "structural": 0.35}  # sums to 1.0

DESCRIPTION_TEMPLATES = {
    "Unauthorized Access": "An unidentified individual was observed {detail} without authorization. {response}",
    "Theft / Property Loss": "Property was reported missing from {area}. {response}",
    "Vandalism": "Signs of vandalism ({detail}) were discovered in {area}. {response}",
    "Structural / Facility Hazard": "A structural or facility hazard ({detail}) was identified in {area}. {response}",
    "Fire / Life-Safety Hazard": "A fire/life-safety hazard involving {detail} was identified in {area}. {response}",
    "Staffing / Procedural Lapse": "A procedural lapse ({detail}) was noted, attributed to {area}. {response}",
}
DETAILS = [
    "an unmarked service entrance", "a rear loading area", "an unattended access point",
    "a stairwell", "a maintenance corridor", "a public-facing entrance", "a rooftop access hatch",
]
AREAS = [
    "the main lobby", "the east wing", "a storage area", "the loading dock",
    "a mechanical room", "the parking structure", "a public corridor",
]
RESPONSES = {
    "Near-Miss": "No injury or loss occurred; logged for trend monitoring.",
    "Minor": "Addressed on-site with no further escalation required.",
    "Moderate": "Facility management notified; corrective follow-up scheduled.",
    "Major": "Incident escalated to facility leadership; formal review opened.",
    "Critical": "Emergency response activated; full incident review and reporting initiated.",
}


def risk_factors(profile: pd.Series) -> dict:
    return {
        "surveillance": 1 - (profile["surveillance_coverage_pct"] / 100),
        "staffing": 1 - (min(profile["staffing_level"], NORM_CAPS["staffing_level"]) / NORM_CAPS["staffing_level"]),
        "structural": min(profile["structural_vulnerabilities"], NORM_CAPS["structural_vulnerabilities"]) / NORM_CAPS["structural_vulnerabilities"],
    }


def generate(seed: int, risk_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = latest_risk_profile(risk_df)

    rows = []
    incident_counter = 0
    for _, profile in profiles.iterrows():
        factors = risk_factors(profile)

        frequency_multiplier = (
            1
            + FREQUENCY_SENSITIVITY["surveillance"] * factors["surveillance"]
            + FREQUENCY_SENSITIVITY["staffing"] * factors["staffing"]
            + FREQUENCY_SENSITIVITY["structural"] * factors["structural"]
        )
        lam = BASE_LAMBDA_BY_TYPE[profile["facility_type"]] * frequency_multiplier
        n_events = int(rng.poisson(lam))

        combined_severity_bias = (
            SEVERITY_WEIGHTS["surveillance"] * factors["surveillance"]
            + SEVERITY_WEIGHTS["staffing"] * factors["staffing"]
            + SEVERITY_WEIGHTS["structural"] * factors["structural"]
        )

        # Category odds: base weight inflated by that category's driver factor.
        category_weights = np.array([
            base_w * (1 + CATEGORY_RISK_SENSITIVITY * factors[driver])
            for _, driver, base_w in CATEGORIES
        ])
        category_probs = category_weights / category_weights.sum()

        for _ in range(n_events):
            incident_counter += 1
            category, driver, _ = CATEGORIES[rng.choice(len(CATEGORIES), p=category_probs)]

            severity_latent = SEVERITY_BASE_MEAN + SEVERITY_RISK_SENSITIVITY * combined_severity_bias + rng.normal(0, SEVERITY_NOISE_STD)
            severity_idx = int(np.clip(round(severity_latent), 0, len(SEVERITY_LEVELS) - 1))
            severity = SEVERITY_LEVELS[severity_idx]

            incident_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 730)))

            template = DESCRIPTION_TEMPLATES[category]
            description = template.format(
                detail=rng.choice(DETAILS),
                area=rng.choice(AREAS),
                response=RESPONSES[severity],
            )

            rows.append({
                "incident_id": f"INC-{incident_counter:05d}",
                "facility_id": profile["facility_id"],
                "facility_name": profile["facility_name"],
                "facility_type": profile["facility_type"],
                "city": profile["city"],
                "incident_date": incident_date.date().isoformat(),
                "category": category,
                "severity": severity,
                "is_near_miss": severity == "Near-Miss",
                "description": description,
            })

    df = pd.DataFrame(rows).sort_values("incident_date").reset_index(drop=True)
    df["incident_id"] = [f"INC-{i + 1:05d}" for i in range(len(df))]
    return df


def main():
    risk_df = pd.read_csv(RISK_ASSESSMENTS_PATH)
    df = generate(SEED, risk_df)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}\n")
    print("Sample rows:")
    print(df.head(8).to_string(index=False))

    print("\nSeverity distribution:")
    print(df["severity"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nIncidents per facility, by facility type:")
    counts_by_facility = df.groupby("facility_id").size()
    profiles = latest_risk_profile(risk_df).set_index("facility_id")
    profiles["incident_count"] = counts_by_facility.reindex(profiles.index).fillna(0).astype(int)
    print(profiles.groupby("facility_type")["incident_count"].agg(["mean", "std", "count"]).round(2))

    # Verify independence from past_incident_count: correlated (same
    # underlying cause) but not identical or trivially derived.
    risk_latest = risk_df.sort_values("assessment_date").groupby("facility_id").last()
    compare = profiles.join(risk_latest[["past_incident_count"]])
    exact_matches = (compare["incident_count"] == compare["past_incident_count"]).mean() * 100
    print(f"\nRows where new incident_count exactly equals risk-assessment past_incident_count: "
          f"{exact_matches:.1f}% (expect low, NOT ~100% - they must stay independent draws)")

    print("\nCorrelation: this dataset's per-facility incident_count vs. risk-assessment raw factors")
    print("(expect negative for surveillance/staffing, positive for vulnerabilities/risk score,")
    print("and clearly less than a perfect correlation with past_incident_count itself):")
    corr_cols = [
        "surveillance_coverage_pct", "staffing_level", "structural_vulnerabilities",
        "computed_risk_score", "past_incident_count", "incident_count",
    ]
    print(compare[corr_cols].corr().round(2)["incident_count"])

    print("\nMean combined severity index (0=Near-Miss..4=Critical) vs. facility type:")
    severity_rank = {lvl: i for i, lvl in enumerate(SEVERITY_LEVELS)}
    df["_severity_rank"] = df["severity"].map(severity_rank)
    print(df.groupby("facility_type")["_severity_rank"].mean().round(2))


if __name__ == "__main__":
    main()
