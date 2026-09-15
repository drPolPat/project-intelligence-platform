"""
Phase 2: interpretable risk-score model.

Trains two models on facility_risk_assessments.csv and compares their
recovered feature signal against the KNOWN generative formula in
generate_risk_assessments.py (RISK_WEIGHTS) — that comparison is the point
of this script, not raw predictive accuracy.

  1. Ridge regression on the generator's own normalized risk factors
     (entry_risk, surveillance_gap_risk, staffing_gap_risk, incident_risk,
     vulnerability_risk) + facility_type dummies. Coefficients on those five
     terms are directly comparable to RISK_WEIGHTS * 100 (the target is a
     0-100 score), because the features are on the same 0-1 scale the
     generator itself used.
  2. Random Forest on the raw (un-normalized) features. The generator adds a
     nonlinear "compounding bonus" when 2+ risk factors are simultaneously
     bad (see COMPOUND_* in generate_risk_assessments.py) — Ridge can't
     represent that interaction, a tree ensemble can.

risk_level is NOT modeled separately: it's a deterministic bucketing of
computed_risk_score in the generator (pd.cut), so a second classifier would
just re-learn the same signal with less information and its own inconsistent
boundary. Instead we bucket each model's predicted score through the exact
same thresholds and report classification metrics on that derived label.

Split: GroupShuffleSplit by facility_id (not a plain random row split) so a
facility's multiple assessment rows never straddle train/test — required
once facility-level joined features (inspections/incidents) enter the
picture in a later experiment, and harmless here. We additionally search
random states for a split that holds out a handful of Critical-tier rows,
since with only 9/200 rows in that tier a naive split can easily zero it out
and make the compounding-effect case unassessable.

load_and_prepare() and evaluate_split() are also imported by
robustness_check.py, which reruns this exact same pipeline across many
random states — keeping one implementation means both scripts are
guaranteed to be evaluating the same methodology.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_generation")
sys.path.insert(0, GEN_DIR)
from generate_risk_assessments import NORM_CAPS, RISK_LEVEL_BINS, RISK_LEVEL_LABELS, RISK_WEIGHTS  # noqa: E402

DATA_PATH = "data/synthetic/facility_risk_assessments.csv"
TARGET = "computed_risk_score"

RAW_FEATURES = [
    "entry_points", "surveillance_coverage_pct", "staffing_level",
    "past_incident_count", "structural_vulnerabilities",
]
NORMALIZED_FEATURES = [
    "entry_risk", "surveillance_gap_risk", "staffing_gap_risk",
    "incident_risk", "vulnerability_risk",
]
# normalized feature name -> (RISK_WEIGHTS key, matching raw feature name)
FACTOR_MAP = {
    "entry_risk": ("entry_points", "entry_points"),
    "surveillance_gap_risk": ("surveillance_gap", "surveillance_coverage_pct"),
    "staffing_gap_risk": ("staffing_gap", "staffing_level"),
    "incident_risk": ("past_incidents", "past_incident_count"),
    "vulnerability_risk": ("structural_vulnerabilities", "structural_vulnerabilities"),
}

MIN_CRITICAL_IN_TEST = 2
SPLIT_RANDOM_STATES = range(200)


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_risk"] = df["entry_points"].clip(upper=NORM_CAPS["entry_points"]) / NORM_CAPS["entry_points"]
    df["surveillance_gap_risk"] = 1 - df["surveillance_coverage_pct"] / 100
    df["staffing_gap_risk"] = 1 - df["staffing_level"].clip(upper=NORM_CAPS["staffing_level"]) / NORM_CAPS["staffing_level"]
    df["incident_risk"] = df["past_incident_count"].clip(upper=NORM_CAPS["past_incidents"]) / NORM_CAPS["past_incidents"]
    df["vulnerability_risk"] = (
        df["structural_vulnerabilities"].clip(upper=NORM_CAPS["structural_vulnerabilities"]) / NORM_CAPS["structural_vulnerabilities"]
    )
    return df


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = add_normalized_features(df)

    type_dummies_full = pd.get_dummies(df["facility_type"], prefix="type")
    type_dummies_dropfirst = pd.get_dummies(df["facility_type"], prefix="type", drop_first=True)

    X_ridge = pd.concat([df[NORMALIZED_FEATURES], type_dummies_dropfirst], axis=1)
    X_rf = pd.concat([df[RAW_FEATURES], type_dummies_full], axis=1)
    y = df[TARGET]
    groups = df["facility_id"]
    risk_level = df["risk_level"]
    return df, X_ridge, X_rf, y, groups, risk_level


def find_group_split(groups: pd.Series, risk_level: pd.Series):
    """Group-aware train/test split, searched over random states so the test
    fold actually contains a handful of Critical-tier rows (else the
    compounding-effect story can't be evaluated at all)."""
    best = None
    for rs in SPLIT_RANDOM_STATES:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
        train_idx, test_idx = next(gss.split(groups, groups=groups))
        n_crit = int((risk_level.iloc[test_idx] == "Critical").sum())
        if best is None or n_crit > best[2]:
            best = (train_idx, test_idx, n_crit, rs)
        if n_crit >= MIN_CRITICAL_IN_TEST:
            return train_idx, test_idx, rs, n_crit
    print(f"WARNING: no split among {len(SPLIT_RANDOM_STATES)} random states reached "
          f"{MIN_CRITICAL_IN_TEST} Critical-tier test rows; using the best found.")
    train_idx, test_idx, n_crit, rs = best
    return train_idx, test_idx, rs, n_crit


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bucket_scores(scores) -> np.ndarray:
    clipped = np.clip(scores, 0, 100)
    return np.asarray(pd.cut(clipped, bins=RISK_LEVEL_BINS, labels=RISK_LEVEL_LABELS, right=False)).astype(str)


def fit_ridge(X_train, y_train, groups_train):
    n_train_groups = groups_train.nunique()
    gkf = GroupKFold(n_splits=min(5, n_train_groups))
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=gkf.split(X_train, y_train, groups_train))
    ridge.fit(X_train, y_train)
    return ridge


def fit_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def evaluate_split(X_ridge, X_rf, y, groups, risk_level, train_idx, test_idx):
    """Fits both models on train_idx, evaluates on test_idx. Returns
    (metrics, artifacts): metrics is scalars only (safe to tabulate across
    many splits in robustness_check.py); artifacts carries the fitted models
    and predictions, used only by this module's detailed headline report."""
    X_ridge_train, X_ridge_test = X_ridge.iloc[train_idx], X_ridge.iloc[test_idx]
    X_rf_train, X_rf_test = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    risk_level_test = risk_level.iloc[test_idx].astype(str)

    ridge = fit_ridge(X_ridge_train, y_train, groups_train)
    rf = fit_rf(X_rf_train, y_train)
    ridge_pred = ridge.predict(X_ridge_test)
    rf_pred = rf.predict(X_rf_test)

    risk_level_test_arr = risk_level_test.to_numpy()
    crit_mask = risk_level_test_arr == "Critical"
    n_crit_test = int(crit_mask.sum())

    def critical_recall(pred_bucket_arr):
        if n_crit_test == 0:
            return float("nan")
        return float((pred_bucket_arr[crit_mask] == "Critical").mean())

    metrics = {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_facilities": int(groups.iloc[train_idx].nunique()),
        "n_test_facilities": int(groups.iloc[test_idx].nunique()),
        "n_crit_test": n_crit_test,
        "ridge_alpha": float(ridge.alpha_),
        "rmse_ridge": rmse(y_test, ridge_pred),
        "mae_ridge": mean_absolute_error(y_test, ridge_pred),
        "rmse_rf": rmse(y_test, rf_pred),
        "mae_rf": mean_absolute_error(y_test, rf_pred),
        "rf_train_r2": rf.score(X_rf_train, y_train),
        "rf_test_r2": rf.score(X_rf_test, y_test),
        "critical_recall_ridge": critical_recall(bucket_scores(ridge_pred)),
        "critical_recall_rf": critical_recall(bucket_scores(rf_pred)),
    }
    artifacts = {
        "ridge": ridge, "rf": rf,
        "ridge_pred": ridge_pred, "rf_pred": rf_pred,
        "X_ridge_test": X_ridge_test, "X_rf_test": X_rf_test, "y_test": y_test,
        "risk_level_test": risk_level_test,
    }
    return metrics, artifacts


def main():
    df, X_ridge, X_rf, y, groups, risk_level = load_and_prepare()

    train_idx, test_idx, split_rs, n_crit_test = find_group_split(groups, risk_level)
    print(f"Split: GroupShuffleSplit(random_state={split_rs}), "
          f"train={len(train_idx)} rows / {groups.iloc[train_idx].nunique()} facilities, "
          f"test={len(test_idx)} rows / {groups.iloc[test_idx].nunique()} facilities")
    print(f"Critical-tier rows in test set: {n_crit_test} "
          f"(of {int((risk_level == 'Critical').sum())} total)")

    metrics, artifacts = evaluate_split(X_ridge, X_rf, y, groups, risk_level, train_idx, test_idx)
    ridge, rf = artifacts["ridge"], artifacts["rf"]
    ridge_pred_test, rf_pred_test = artifacts["ridge_pred"], artifacts["rf_pred"]
    X_rf_test, y_test = artifacts["X_rf_test"], artifacts["y_test"]
    risk_level_test = artifacts["risk_level_test"]

    print(f"\nRidge chosen alpha (via GroupKFold CV on train): {metrics['ridge_alpha']:.4g}")
    print("\n=== Regression metrics (held-out test set) ===")
    print(f"Ridge (normalized features):    RMSE={metrics['rmse_ridge']:.2f}  MAE={metrics['mae_ridge']:.2f}")
    print(f"Random Forest (raw features):   RMSE={metrics['rmse_rf']:.2f}  MAE={metrics['mae_rf']:.2f}")
    print(f"Random Forest train R2={metrics['rf_train_r2']:.3f}  test R2={metrics['rf_test_r2']:.3f}  "
          f"(gap flags overfitting on this small dataset)")

    # --- Ridge coefficients vs. RISK_WEIGHTS -----------------------------
    coef_map = dict(zip(X_ridge.columns, ridge.coef_))
    print("\n=== Ridge coefficients vs. RISK_WEIGHTS (both on a 0-100-point scale) ===")
    rows = []
    for feat, (weight_key, raw_feat) in FACTOR_MAP.items():
        true_points = RISK_WEIGHTS[weight_key] * 100
        rows.append({
            "factor": feat,
            "true_weight_x100": round(true_points, 1),
            "ridge_coefficient": round(coef_map[feat], 1),
        })
    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))
    print(f"\nRidge intercept: {ridge.intercept_:.2f} (generator has no additive constant - expect near 0)")

    type_coefs = {k: round(v, 2) for k, v in coef_map.items() if k.startswith("type_")}
    print(f"facility_type dummy coefficients (generator has no direct type effect once raw "
          f"factors are controlled for - expect near 0): {type_coefs}")

    entry_pairwise_corr = df[["entry_points", "computed_risk_score"]].corr().iloc[0, 1]
    print(f"\nentry_points: pairwise correlation with computed_risk_score = {entry_pairwise_corr:.3f} "
          f"(near zero, confounded by facility_type)")
    print(f"entry_points: Ridge coefficient (controlling for other factors) = {coef_map['entry_risk']:.1f} "
          f"vs. true {RISK_WEIGHTS['entry_points'] * 100:.1f} -- "
          f"{'RECOVERED despite near-zero pairwise correlation' if coef_map['entry_risk'] > 5 else 'still muted even after controlling for confounders'}")

    # --- Random Forest importances ---------------------------------------
    print("\n=== Random Forest feature importances ===")
    impurity_importance = pd.Series(rf.feature_importances_, index=X_rf.columns).sort_values(ascending=False)
    print("Impurity-based:")
    print(impurity_importance.round(3).to_string())

    perm = permutation_importance(rf, X_rf_test, y_test, n_repeats=30, random_state=42)
    perm_importance = pd.Series(perm.importances_mean, index=X_rf.columns).sort_values(ascending=False)
    print("\nPermutation importance on held-out test set (more reliable under correlated features):")
    print(perm_importance.round(3).to_string())

    print("\n=== Ridge vs. Random Forest: do the two mechanisms rank factors similarly? ===")
    rank_rows = []
    for norm_feat, (_, raw_feat) in FACTOR_MAP.items():
        rank_rows.append({
            "factor": norm_feat,
            "ridge_coef_abs": abs(coef_map[norm_feat]),
            "rf_permutation_importance": perm_importance.get(raw_feat, float("nan")),
        })
    rank_df = pd.DataFrame(rank_rows)
    rank_df["ridge_rank"] = rank_df["ridge_coef_abs"].rank(ascending=False)
    rank_df["rf_rank"] = rank_df["rf_permutation_importance"].rank(ascending=False)
    print(rank_df.to_string(index=False))
    rho, pval = spearmanr(rank_df["ridge_rank"], rank_df["rf_rank"])
    print(f"Spearman rank correlation between the two models' factor ordering: {rho:.2f} (p={pval:.2f})")

    # --- Derived risk_level classification metrics ------------------------
    for name, pred_scores in [("Ridge", ridge_pred_test), ("Random Forest", rf_pred_test)]:
        pred_bucket = bucket_scores(pred_scores)
        print(f"\n=== Derived risk_level metrics on test set ({name}) ===")
        print(f"Accuracy: {accuracy_score(risk_level_test, pred_bucket):.3f}")
        print(classification_report(
            risk_level_test, pred_bucket, labels=RISK_LEVEL_LABELS, zero_division=0,
        ))
        cm = confusion_matrix(risk_level_test, pred_bucket, labels=RISK_LEVEL_LABELS)
        print(f"Confusion matrix (rows=true, cols=predicted), labels={RISK_LEVEL_LABELS}:")
        print(pd.DataFrame(cm, index=RISK_LEVEL_LABELS, columns=RISK_LEVEL_LABELS).to_string())


if __name__ == "__main__":
    main()
