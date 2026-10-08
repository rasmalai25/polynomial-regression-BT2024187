"""
Polynomial Regression Assignment - BT2024187
============================================
Problem 1 (var1): Power Plant Steam Turbine Optimization
  - Features: x1..x6, Target: Net Power Score (y), Max degree: 10

Problem 2 (var2): Subterranean Thermal Reservoir Mapping
  - Features: x1..x3, Target: Thermal Anomaly Score (y), Max degree: 20

Usage (run from the repo root):
  1. Put the four CSVs (BT2024187_{train,test}_var{1,2}.csv) in ./data/
  2. python polynomial_regression.py
  3. Predictions are written to ./output/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

ROLL_NO  = "BT2024187"
DATA_DIR = Path("data")      # train/test CSVs go here
OUT_DIR  = Path("output")    # prediction CSVs are written here
OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------
# Helper: find best degree + alpha via cross-validation
# --------------------------------------------------------
def find_best_params(X_train, y_train, max_degree, alphas, cv=5, label=""):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    print(f"\n{'='*55}")
    print(f"  {label}: Tuning degree (1 to {max_degree}) and Ridge alpha")
    print(f"{'='*55}")

    for degree in range(1, max_degree + 1):
        for alpha in alphas:
            pipe = Pipeline([
                ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("ridge",  Ridge(alpha=alpha))
            ])
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=kf, scoring="neg_mean_squared_error"
            )
            mse_cv = -scores.mean()
            results.append({"degree": degree, "alpha": alpha, "cv_mse": mse_cv})
        print(f"  degree={degree:2d} complete | best so far: {min(r['cv_mse'] for r in results):.6f}", flush=True)

    df_res = pd.DataFrame(results).sort_values("cv_mse")
    best = df_res.iloc[0]
    print(f"\n  Best degree : {int(best['degree'])}")
    print(f"  Best alpha  : {best['alpha']}")
    print(f"  CV MSE      : {best['cv_mse']:.6f}")

    # Exact 5-fold CV R2 for the chosen (degree, alpha), same folds as above
    best_pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=int(best["degree"]), include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=best["alpha"]))
    ])
    r2_cv = cross_val_score(best_pipe, X_train, y_train, cv=kf, scoring="r2").mean()
    print(f"  CV R2       : {r2_cv:.6f}")

    return int(best["degree"]), best["alpha"], df_res


# --------------------------------------------------------
# Helper: train final model and report training metrics
# --------------------------------------------------------
def train_final_model(X_train, y_train, degree, alpha, label=""):
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha))
    ])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    mse  = mean_squared_error(y_train, y_pred_train)
    r2   = r2_score(y_train, y_pred_train)
    n_features = pipe.named_steps["poly"].n_output_features_

    print(f"\n  [{label}] Final model metrics on training data:")
    print(f"    Polynomial features : {n_features}")
    print(f"    MSE (train)         : {mse:.6f}")
    print(f"    R2  (train)         : {r2:.6f}")
    return pipe


# ========================================================
# PROBLEM 1  --  var1
# ========================================================
print("\n" + "#"*55)
print("  PROBLEM 1: Power Plant Steam Turbine (var1)")
print("#"*55)

train1 = pd.read_csv(DATA_DIR / f"{ROLL_NO}_train_var1.csv")
test1  = pd.read_csv(DATA_DIR / f"{ROLL_NO}_test_var1.csv")

X_train1 = train1.drop("y", axis=1).values
y_train1 = train1["y"].values
X_test1  = test1.values

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

best_deg1, best_alpha1, results1 = find_best_params(
    X_train1, y_train1, max_degree=10, alphas=alphas, label="var1"
)
model1 = train_final_model(X_train1, y_train1, best_deg1, best_alpha1, label="var1")

pred1 = model1.predict(X_test1)
out1 = OUT_DIR / f"{ROLL_NO}_pred_var1.csv"
pd.DataFrame({"y": pred1}).to_csv(out1, index=False)
print(f"\n  Saved: {out1}  ({len(pred1)} rows)")


# ========================================================
# PROBLEM 2  --  var2
# ========================================================
print("\n" + "#"*55)
print("  PROBLEM 2: Thermal Reservoir Mapping (var2)")
print("#"*55)

train2 = pd.read_csv(DATA_DIR / f"{ROLL_NO}_train_var2.csv")
test2  = pd.read_csv(DATA_DIR / f"{ROLL_NO}_test_var2.csv")

X_train2 = train2.drop("y", axis=1).values
y_train2 = train2["y"].values
X_test2  = test2.values

best_deg2, best_alpha2, results2 = find_best_params(
    X_train2, y_train2, max_degree=20, alphas=alphas, label="var2"
)
model2 = train_final_model(X_train2, y_train2, best_deg2, best_alpha2, label="var2")

pred2 = model2.predict(X_test2)
out2 = OUT_DIR / f"{ROLL_NO}_pred_var2.csv"
pd.DataFrame({"y": pred2}).to_csv(out2, index=False)
print(f"\n  Saved: {out2}  ({len(pred2)} rows)")


# ========================================================
# FINAL SUMMARY
# ========================================================
print("\n" + "="*55)
print("  DONE - SUMMARY")
print("="*55)
print(f"  var1 -> Best degree={best_deg1}, Best alpha={best_alpha1}")
print(f"  var2 -> Best degree={best_deg2}, Best alpha={best_alpha2}")
print(f"\n  Output files written to: {OUT_DIR.resolve()}")
print("\n  All done!")