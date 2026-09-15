"""
Phase 2 robustness check.

The headline result in train_risk_model.py (random_state=0) showed Random
Forest catching 1/2 Critical-tier test rows vs. Ridge catching 0/2. With
only 2 rows, that's barely more than a coin flip either way. This script
reruns the EXACT SAME pipeline (imported from train_risk_model.py, not
reimplemented) across a range of GroupShuffleSplit random states — not
searched or cherry-picked for Critical-tier presence, just 0..N-1 in order —
to see whether the RF > Ridge pattern on Critical recall holds up, or was a
lucky single draw.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from train_risk_model import evaluate_split, load_and_prepare

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

N_SPLITS = 10
RANDOM_STATES = list(range(N_SPLITS))  # not filtered for Critical-tier presence, by design


def main():
    df, X_ridge, X_rf, y, groups, risk_level = load_and_prepare()

    records = []
    for rs in RANDOM_STATES:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
        train_idx, test_idx = next(gss.split(groups, groups=groups))
        metrics, _ = evaluate_split(X_ridge, X_rf, y, groups, risk_level, train_idx, test_idx)
        metrics["random_state"] = rs
        records.append(metrics)

    results = pd.DataFrame(records).set_index("random_state")

    print(f"=== Per-split results across {N_SPLITS} GroupShuffleSplit random states (0-{N_SPLITS - 1}) ===")
    print(results[[
        "n_test", "n_test_facilities", "n_crit_test",
        "rmse_ridge", "rmse_rf", "mae_ridge", "mae_rf",
        "critical_recall_ridge", "critical_recall_rf",
    ]].round(2).to_string())

    print("\n=== Regression metrics: mean +/- std across all splits ===")
    for col, label in [
        ("rmse_ridge", "Ridge RMSE"), ("rmse_rf", "RF RMSE"),
        ("mae_ridge", "Ridge MAE"), ("mae_rf", "RF MAE"),
    ]:
        print(f"{label}: {results[col].mean():.2f} +/- {results[col].std():.2f}")

    n_zero_crit = int((results["n_crit_test"] == 0).sum())
    print(f"\nSplits with zero Critical-tier rows in test: {n_zero_crit} / {N_SPLITS} "
          f"(Critical recall is undefined -- NaN -- for those; illustrates how fragile "
          f"a single random split is for evaluating a 9/200-row tier)")

    evaluable = results[results["n_crit_test"] > 0]
    print(f"\n=== Critical-tier recall, splits with >=1 Critical row in test (n={len(evaluable)}/{N_SPLITS}) ===")
    for col, label in [("critical_recall_ridge", "Ridge"), ("critical_recall_rf", "Random Forest")]:
        vals = evaluable[col]
        if len(vals) == 0:
            print(f"{label}: no evaluable splits")
            continue
        print(f"{label}: mean={vals.mean():.2f}  range=[{vals.min():.2f}, {vals.max():.2f}]  "
              f"n_crit_test per split={evaluable['n_crit_test'].tolist()}")

    if len(evaluable) > 0:
        rf_better = int((evaluable["critical_recall_rf"] > evaluable["critical_recall_ridge"]).sum())
        rf_tied = int((evaluable["critical_recall_rf"] == evaluable["critical_recall_ridge"]).sum())
        rf_worse = int((evaluable["critical_recall_rf"] < evaluable["critical_recall_ridge"]).sum())
        print(f"\nAcross the {len(evaluable)} evaluable splits: RF beat Ridge on Critical recall in "
              f"{rf_better}, tied in {rf_tied}, lost in {rf_worse}.")
        if rf_better > rf_worse and rf_better >= len(evaluable) / 2:
            print("-> Pattern holds up: RF's Critical-tier edge is not just a random_state=0 fluke.")
        elif rf_better == rf_worse:
            print("-> Mixed: RF and Ridge split Critical-tier wins roughly evenly across draws -- "
                  "the random_state=0 result should NOT be read as a general RF advantage on its own.")
        else:
            print("-> The random_state=0 result does NOT generalize: Ridge matches or beats RF on "
                  "Critical recall as often or more often across other splits.")


if __name__ == "__main__":
    main()
