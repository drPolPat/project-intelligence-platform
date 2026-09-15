"""
Synthetic facility risk assessment data generator.

Produces one row per assessment with:
  - facility identity/type (fully fabricated, no real locations or companies)
  - raw risk factors: entry points, surveillance coverage, staffing, past
    incidents, structural vulnerabilities
  - a computed_risk_score derived from a documented, transparent weighted
    formula plus Gaussian noise (see RISK_WEIGHTS below)

The formula is intentionally simple and disclosed so that the ML phase can
demonstrate recovering it (feature importances should roughly track
RISK_WEIGHTS), and so a model card can honestly describe what the model is
and isn't learning from real-world signal.

All facility names, cities, and identifiers are synthetic placeholders.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

SEED = 42
N_ASSESSMENTS = 200
N_FACILITIES = 120
OUTPUT_PATH = "data/synthetic/facility_risk_assessments.csv"

FACILITY_TYPES = ["Museum", "Mall", "Office Tower", "Utility Substation"]

# Fully fictional place names — not tied to any real city or facility.
SYNTHETIC_CITIES = [
    "Meridian Falls", "Ashford Bay", "Northgate Crossing", "Rivermoor",
    "Castellan Heights", "Fenbrook", "Draymoor", "Silverpine Junction",
    "Harlow Vale", "Kettering Reach",
]

# Per-facility-type distribution parameters. These encode plausible baseline
# differences (e.g. utility substations are typically unmanned/remote with
# lower surveillance and staffing than a staffed museum or mall) without
# referencing any real facility, company, or incident.
TYPE_PROFILES = {
    "Museum": {
        "entry_points": (3, 8),          # (min, max) staffed public entrances
        "surveillance_pct": (60, 97),    # % of premises under camera coverage
        "staffing_level": (4, 10),       # security staff on duty, peak shift
        "incident_lambda": 0.8,          # Poisson mean, past-12-month incidents
        "vulnerability_max": 3,          # max structural vulnerabilities flagged
    },
    "Mall": {
        "entry_points": (6, 20),
        "surveillance_pct": (40, 88),
        "staffing_level": (3, 9),
        "incident_lambda": 2.5,
        "vulnerability_max": 4,
    },
    "Office Tower": {
        "entry_points": (2, 10),
        "surveillance_pct": (50, 92),
        "staffing_level": (2, 8),
        "incident_lambda": 1.0,
        "vulnerability_max": 3,
    },
    "Utility Substation": {
        "entry_points": (1, 4),
        "surveillance_pct": (15, 70),    # often remote/unmanned, sparse cameras
        "staffing_level": (0, 3),        # frequently unstaffed, remote-monitored
        "incident_lambda": 1.2,          # e.g. copper theft, perimeter breaches
        "vulnerability_max": 8,          # critical infrastructure, higher exposure
    },
}

# Weighted risk formula. Each raw factor is normalized to [0, 1] (higher =
# riskier) and combined with these weights, then scaled to 0-100. Weights
# sum to 1.0 by construction.
RISK_WEIGHTS = {
    "entry_points": 0.15,
    "surveillance_gap": 0.30,   # inverse of surveillance coverage
    "staffing_gap": 0.20,       # inverse of staffing level
    "past_incidents": 0.20,
    "structural_vulnerabilities": 0.15,
}
NOISE_STD = 5.0  # Gaussian noise added to the 0-100 risk score

# Compounding-risk bonus: when several bad conditions co-occur at one facility
# (e.g. a near-unstaffed, poorly-surveilled substation with a history of
# incidents), real-world risk tends to escalate faster than a purely additive
# linear formula predicts. This models that nonlinearity directly, rather
# than relying on noise, so the small critical-risk tail comes from genuine
# compounding cases (and stays meaningful for the RAG/agent demo) instead of
# being an artifact of a wide noise distribution.
COMPOUND_THRESHOLDS = {
    "surveillance_low": 35,     # coverage % below this counts as "very low"
    "staffing_low": 1,          # staffing at/below this counts as "very low"
    "incidents_high": 3,        # past incidents at/above this counts as "elevated"
    "vulnerabilities_high": 5,  # structural vulnerabilities at/above counts as "high"
}
COMPOUND_MIN_CONDITIONS = 1   # bonus kicks in once 2+ conditions co-occur
COMPOUND_BONUS_PER_EXTRA = 11.5  # added per condition beyond COMPOUND_MIN_CONDITIONS
# Tuned by sweeping thresholds/bonus against this seed so ~4-5% of rows land
# Critical, via genuine multi-factor compounding rather than a wider noise term.

# Normalization caps for each raw factor (value at/above cap -> risk of 1.0)
NORM_CAPS = {
    "entry_points": 20,
    "staffing_level": 10,
    "past_incidents": 10,
    "structural_vulnerabilities": 8,
}

RISK_LEVEL_BINS = [0, 25, 50, 75, 100.0001]
RISK_LEVEL_LABELS = ["Low", "Medium", "High", "Critical"]

# How much a facility's own surveillance/staffing gaps inflate its expected
# past-incident count, on top of the facility-type baseline lambda. Without
# this, incidents are independent of surveillance/staffing within a type and
# the low-surveillance-correlates-with-more-incidents relationship barely
# shows up pooled across types. Multiplier ranges from 1x (full surveillance,
# max staffing) up to 1 + INCIDENT_SURVEILLANCE_SENSITIVITY +
# INCIDENT_STAFFING_SENSITIVITY (zero surveillance, zero staffing).
INCIDENT_SURVEILLANCE_SENSITIVITY = 0.9
INCIDENT_STAFFING_SENSITIVITY = 0.4


def build_facility_registry(rng, n_facilities: int) -> list[dict]:
    """Assigns each facility_id a fixed type/name/city, once. Assessments for
    the same facility_id must always report the same facility identity — only
    the point-in-time risk factors vary between assessments of that facility."""
    registry = []
    for idx in range(n_facilities):
        facility_type = rng.choice(FACILITY_TYPES)
        city = rng.choice(SYNTHETIC_CITIES)
        registry.append({
            "facility_id": f"FAC-{idx + 1:04d}",
            "facility_type": facility_type,
            "facility_name": f"{city} {facility_type}",
            "city": city,
        })
    return registry


def generate(n: int, seed: int, n_facilities: int = N_FACILITIES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    registry = build_facility_registry(rng, n_facilities)
    rows = []

    for i in range(n):
        facility = registry[i % n_facilities]
        facility_type = facility["facility_type"]
        profile = TYPE_PROFILES[facility_type]

        entry_points = int(rng.integers(profile["entry_points"][0], profile["entry_points"][1] + 1))
        surveillance_pct = round(float(rng.uniform(*profile["surveillance_pct"])), 1)
        staffing_level = int(rng.integers(profile["staffing_level"][0], profile["staffing_level"][1] + 1))

        incident_multiplier = (
            1
            + INCIDENT_SURVEILLANCE_SENSITIVITY * (1 - surveillance_pct / 100)
            + INCIDENT_STAFFING_SENSITIVITY * (1 - min(staffing_level, NORM_CAPS["staffing_level"]) / NORM_CAPS["staffing_level"])
        )
        past_incident_count = int(rng.poisson(profile["incident_lambda"] * incident_multiplier))
        structural_vulnerabilities = int(rng.integers(0, profile["vulnerability_max"] + 1))

        entry_risk = min(entry_points, NORM_CAPS["entry_points"]) / NORM_CAPS["entry_points"]
        surveillance_gap_risk = 1 - (surveillance_pct / 100)
        staffing_gap_risk = 1 - (min(staffing_level, NORM_CAPS["staffing_level"]) / NORM_CAPS["staffing_level"])
        incident_risk = min(past_incident_count, NORM_CAPS["past_incidents"]) / NORM_CAPS["past_incidents"]
        vulnerability_risk = min(structural_vulnerabilities, NORM_CAPS["structural_vulnerabilities"]) / NORM_CAPS["structural_vulnerabilities"]

        raw_score = 100 * (
            RISK_WEIGHTS["entry_points"] * entry_risk
            + RISK_WEIGHTS["surveillance_gap"] * surveillance_gap_risk
            + RISK_WEIGHTS["staffing_gap"] * staffing_gap_risk
            + RISK_WEIGHTS["past_incidents"] * incident_risk
            + RISK_WEIGHTS["structural_vulnerabilities"] * vulnerability_risk
        )

        compound_count = sum([
            surveillance_pct < COMPOUND_THRESHOLDS["surveillance_low"],
            staffing_level <= COMPOUND_THRESHOLDS["staffing_low"],
            past_incident_count >= COMPOUND_THRESHOLDS["incidents_high"],
            structural_vulnerabilities >= COMPOUND_THRESHOLDS["vulnerabilities_high"],
        ])
        compound_bonus = max(0, compound_count - COMPOUND_MIN_CONDITIONS) * COMPOUND_BONUS_PER_EXTRA

        noisy_score = float(np.clip(raw_score + compound_bonus + rng.normal(0, NOISE_STD), 0, 100))

        assessment_date = pd.Timestamp("2024-01-01") + pd.Timedelta(
            days=int(rng.integers(0, 730))
        )

        rows.append({
            "assessment_id": f"FRA-{i + 1:05d}",
            "facility_id": facility["facility_id"],
            "facility_name": facility["facility_name"],
            "facility_type": facility_type,
            "city": facility["city"],
            "assessment_date": assessment_date.date().isoformat(),
            "entry_points": entry_points,
            "surveillance_coverage_pct": surveillance_pct,
            "staffing_level": staffing_level,
            "past_incident_count": past_incident_count,
            "structural_vulnerabilities": structural_vulnerabilities,
            "computed_risk_score": round(noisy_score, 1),
        })

    df = pd.DataFrame(rows)
    df["risk_level"] = pd.cut(
        df["computed_risk_score"], bins=RISK_LEVEL_BINS, labels=RISK_LEVEL_LABELS, right=False
    )
    return df


def main():
    df = generate(N_ASSESSMENTS, SEED)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}\n")
    print("Sample rows:")
    print(df.head(10).to_string(index=False))

    print("\nRisk score by facility type:")
    print(df.groupby("facility_type")["computed_risk_score"].agg(["mean", "std", "count"]).round(1))

    print("\nRisk level distribution:")
    print(df["risk_level"].value_counts())

    print(f"\nCritical rows: {(df['risk_level'] == 'Critical').sum()} / {len(df)} "
          f"({(df['risk_level'] == 'Critical').mean() * 100:.1f}%)")

    factor_cols = [
        "entry_points", "surveillance_coverage_pct", "staffing_level",
        "past_incident_count", "structural_vulnerabilities", "computed_risk_score",
    ]
    print("\nCorrelation matrix, raw risk factors + score (pooled across facility types -")
    print("note some of this reflects type-level baselines, e.g. substations skew low")
    print("staffing/surveillance by construction, not just a pairwise factor relationship):")
    print(df[factor_cols].corr().round(2))


if __name__ == "__main__":
    main()
