"""
Trains and persists the Random Forest model that the agent layer (Phase 4)
actually calls — a separate step from ml/train_risk_model.py's held-out
evaluation, which exists to answer "how good is this model / how does it
compare to Ridge" (see MODEL_CARD.md), not to produce a deployable artifact.

Why Random Forest, not Ridge, for the deployed tool: MODEL_CARD.md's
headline finding is that the two models are comparable on overall regression
accuracy (Ridge RMSE 7.02 vs. RF 7.51 mean over 10 splits) but NOT on
Critical-tier detection — Ridge caught 0/9 Critical-tier rows across every
one of 10 held-out test folds, while Random Forest averaged 57% recall on
the same rows, because the generator's compounding-risk interaction term is
structurally invisible to a linear model. A risk-flagging tool's whole
purpose is catching the highest-stakes cases, so Random Forest is the
correct choice here even though it isn't the uniformly "better" model.

Why train on the FULL 200 rows here (unlike train_risk_model.py's holdout
split): the held-out evaluation already answered the model-selection and
capability questions above. A shipped tool should use all available signal;
re-holding-out data for a model whose architecture and hyperparameters are
already validated would only make its predictions noisier for no evaluative
benefit. Hyperparameters are unchanged from the validated run
(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42).
"""
import json
import os
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from train_risk_model import RAW_FEATURES, load_and_prepare  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "random_forest_risk_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "random_forest_risk_model.meta.json"


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df, _, X_rf, y, _, _ = load_and_prepare()

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42)
    rf.fit(X_rf, y)

    joblib.dump(rf, MODEL_PATH)

    metadata = {
        "feature_columns": list(X_rf.columns),
        "raw_features": RAW_FEATURES,
        "facility_types": sorted(df["facility_type"].unique().tolist()),
        "trained_on_rows": len(df),
        "hyperparameters": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3, "random_state": 42},
        "note": "Trained on the FULL dataset (not a holdout split) - see docstring. "
                "Held-out validation results are in ../MODEL_CARD.md.",
        "feature_importances": dict(zip(X_rf.columns, [round(v, 4) for v in rf.feature_importances_])),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Trained on {len(df)} rows, {len(X_rf.columns)} features: {list(X_rf.columns)}")
    print("\nFeature importances (full-data fit):")
    for feat, imp in sorted(metadata["feature_importances"].items(), key=lambda kv: -kv[1]):
        print(f"  {feat:30s} {imp:.4f}")


if __name__ == "__main__":
    main()
