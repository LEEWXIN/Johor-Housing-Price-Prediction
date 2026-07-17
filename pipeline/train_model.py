"""
train_model.py
================
BDS23124 - Johor Property DSS
Trains and evaluates three data mining models: Linear Regression, Decision Tree, Random Forest.

[NOTE] This script was originally missing from the appendix and has been added back in:
  there was previously no code in the project that could reproduce or prove how
  model_scores.json and house_model.pkl were produced - this script fills that gap.

Pipeline:
  1. Load johor_final_clean.csv, already deduplicated by build_database.py
     (if this file doesn't exist, falls back to reading the raw johor_final.csv
      and deduplicating on the spot, so this script can also run standalone)
  2. Use GroupShuffleSplit to split train/test by "property profile" (grouping
     rows with identical Area+Type+Size+Beds+Baths) so the same property profile
     never appears on both sides of the split, preventing data leakage that
     would inflate the metrics
  3. Train Linear Regression / Decision Tree / Random Forest and evaluate
     R2 / RMSE / MAE on the held-out test set
  4. Write model_scores.json (used by app.py's Data Dashboard)
  5. Refit the final Random Forest on the FULL deduplicated dataset and
     write house_model.pkl (used by app.py's Price Check)

Run:
    python train_model.py
"""
import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES  = ["Area", "Property_Type", "Size_SQFT", "Bedrooms", "Bathrooms"]
CAT_COLS  = ["Area", "Property_Type"]
DEDUP_KEYS = ["Area", "Property_Type", "Size_SQFT", "Bedrooms", "Bathrooms", "Price_RM"]


def load_clean_data():
    """Prefer the data already deduplicated by build_database.py; otherwise
    deduplicate on the spot so this script can also run standalone."""
    if os.path.exists("johor_final_clean.csv"):
        df = pd.read_csv("johor_final_clean.csv")
        print(f"[LOAD] Loaded deduplicated data johor_final_clean.csv: {len(df):,} rows")
        return df
    df = pd.read_csv("johor_final.csv")
    n0 = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS).reset_index(drop=True)
    print(f"[LOAD] johor_final_clean.csv not found, deduplicating johor_final.csv on the spot: "
          f"{n0:,} -> {len(df):,} rows ({n0 - len(df):,} exact duplicate rows removed)")
    return df


def make_pipeline(estimator):
    pre = ColumnTransformer(
        [("c", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def main():
    df = load_clean_data()

    # ── Split grouped by property profile, to prevent data leakage ──────────
    df["profile_id"] = (
        df["Area"].astype(str) + "|" + df["Property_Type"].astype(str) + "|" +
        df["Size_SQFT"].astype(str) + "|" + df["Bedrooms"].astype(str) + "|" +
        df["Bathrooms"].astype(str)
    )
    X, y = df[FEATURES], df["Price_RM"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["profile_id"]))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    print(f"[SPLIT] Train {len(Xtr):,} rows / Test {len(Xte):,} rows "
          f"(GroupShuffleSplit, grouped by property profile, no group crosses train/test)")

    # ── Train and evaluate the three models ──────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
    }

    scores = {}
    print("\n" + "=" * 70)
    print(f"{'Model':<20s} | {'R2':>6s} | {'RMSE (RM)':>12s} | {'MAE (RM)':>12s}")
    print("-" * 70)
    for name, est in models.items():
        pipe = make_pipeline(est)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        r2   = r2_score(yte, pred)
        rmse = mean_squared_error(yte, pred) ** 0.5
        mae  = mean_absolute_error(yte, pred)
        scores[name] = {"R2": round(r2, 3), "RMSE": round(rmse), "MAE": round(mae)}
        print(f"{name:<20s} | {r2:6.3f} | {rmse:12,.0f} | {mae:12,.0f}")
    print("=" * 70)

    with open("model_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print("\n[SAVE] model_scores.json written")

    # ── Final deployed model: refit on the full deduplicated dataset ──────
    #     (grouped split is used at evaluation time for honest metrics;
    #      training the deployed model on all the data is standard practice
    #      so the model used inside app.py performs as well as possible)
    final_pipe = make_pipeline(RandomForestRegressor(n_estimators=200, random_state=42))
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, "house_model.pkl")
    print("[SAVE] house_model.pkl written (Random Forest, trained on all deduplicated data)")


if __name__ == "__main__":
    main()