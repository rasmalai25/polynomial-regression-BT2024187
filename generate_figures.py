import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings, os
warnings.filterwarnings("ignore")

ROLL_NO  = "BT2024187"
DATA_DIR = r"D:\College\Semester_5\ML\Assignment\BT2024187\BT2024187"
OUT_DIR  = r"D:\College\Semester_5\ML\Assignment\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Known CV results
cv1 = {1:9.5890,2:3.2626,3:0.9694,4:0.7444,5:0.4520,
       6:0.6350,7:0.6825,8:0.8517,9:1.0550,10:1.2114}
cv2 = {1:33.0915,2:20.9886,3:11.9245,4:3.9956,5:1.8467,
       6:0.6322,7:0.3635,8:0.2728,9:0.2704,10:0.2533,
       11:0.2497,12:0.2452,13:0.2501,14:0.2557,15:0.2609,
       16:0.2608,17:0.2626,18:0.2672,19:0.2708,20:0.2823}

best_deg1, best_alpha1 = 5, 10.0
best_deg2, best_alpha2 = 12, 1.0

train1 = pd.read_csv(rf"{DATA_DIR}\{ROLL_NO}_train_var1.csv")
train2 = pd.read_csv(rf"{DATA_DIR}\{ROLL_NO}_train_var2.csv")
X1, y1 = train1.drop("y", axis=1).values, train1["y"].values
X2, y2 = train2.drop("y", axis=1).values, train2["y"].values

def fit_model(X, y, degree, alpha):
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha))
    ])
    pipe.fit(X, y)
    return pipe

model1 = fit_model(X1, y1, best_deg1, best_alpha1)
model2 = fit_model(X2, y2, best_deg2, best_alpha2)
yhat1  = model1.predict(X1);  res1 = y1 - yhat1
yhat2  = model2.predict(X2);  res2 = y2 - yhat2
mse1   = mean_squared_error(y1, yhat1);  r2_1 = r2_score(y1, yhat1)
mse2   = mean_squared_error(y2, yhat2);  r2_2 = r2_score(y2, yhat2)

plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# ── Figure 1: CV MSE curves ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Cross-Validation MSE vs Polynomial Degree", fontsize=13, fontweight="bold")

d1, m1 = zip(*sorted(cv1.items()))
ax1.plot(d1, m1, "o-", color="#0f3460", lw=2, markersize=6)
ax1.axvline(best_deg1, color="#e94560", lw=2, linestyle="--", label=f"Optimal degree = {best_deg1}")
ax1.fill_between(d1, m1, alpha=0.08, color="#0f3460")
ax1.set_title("Problem 1 — var1 (6 features, max deg 10)", fontsize=10)
ax1.set_xlabel("Polynomial Degree"); ax1.set_ylabel("CV MSE")
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_facecolor("#f8f9fc")

d2, m2 = zip(*sorted(cv2.items()))
ax2.plot(d2, m2, "s-", color="#e94560", lw=2, markersize=6)
ax2.axvline(best_deg2, color="#0f3460", lw=2, linestyle="--", label=f"Optimal degree = {best_deg2}")
ax2.fill_between(d2, m2, alpha=0.08, color="#e94560")
ax2.set_title("Problem 2 — var2 (3 features, max deg 20)", fontsize=10)
ax2.set_xlabel("Polynomial Degree"); ax2.set_ylabel("CV MSE")
ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_facecolor("#f8f9fc")

fig.tight_layout()
fig.savefig(rf"{OUT_DIR}\fig_cv_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_cv_curves.png")

# ── Figure 2: Actual vs Predicted ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Actual vs Predicted (Training Data)", fontsize=13, fontweight="bold")

for ax, y_true, y_pred, r2, mse, color, lbl in [
    (axes[0], y1, yhat1, r2_1, mse1, "#0f3460", "var1"),
    (axes[1], y2, yhat2, r2_2, mse2, "#e94560", "var2"),
]:
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=color)
    mn, mx = y_true.min(), y_true.max()
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="Perfect fit ($y=\\hat{y}$)")
    ax.set_title(f"{lbl}:  $R^2$ = {r2:.4f},  MSE = {mse:.4f}", fontsize=10)
    ax.set_xlabel("Actual $y$"); ax.set_ylabel("Predicted $\\hat{y}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_facecolor("#f8f9fc")

fig.tight_layout()
fig.savefig(rf"{OUT_DIR}\fig_actual_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_actual_vs_pred.png")

# ── Figure 3: Residuals ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Residual Plots (Training Data)", fontsize=13, fontweight="bold")

for ax, y_pred, res, color, lbl in [
    (axes[0], yhat1, res1, "#0f3460", "var1"),
    (axes[1], yhat2, res2, "#e94560", "var2"),
]:
    ax.scatter(y_pred, res, alpha=0.3, s=8, color=color)
    ax.axhline(0, color="black", lw=1.5, linestyle="--")
    ax.set_title(f"{lbl}:  Std = {res.std():.4f},  Mean = {res.mean():.5f}", fontsize=10)
    ax.set_xlabel("Predicted $\\hat{y}$"); ax.set_ylabel("Residual $(y - \\hat{y})$")
    ax.grid(True, alpha=0.3); ax.set_facecolor("#f8f9fc")

fig.tight_layout()
fig.savefig(rf"{OUT_DIR}\fig_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_residuals.png")

# ── Figure 4: Feature count bar charts ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Number of Polynomial Features vs Degree", fontsize=13, fontweight="bold")

fc1 = []
for d in range(1, 11):
    pf = PolynomialFeatures(degree=d, include_bias=False); pf.fit(X1[:1])
    fc1.append(pf.n_output_features_)
colors1 = ["#e94560" if d == best_deg1 else "#0f3460" for d in range(1,11)]
ax1.bar(range(1, 11), fc1, color=colors1, alpha=0.85, edgecolor="white")
ax1.set_title("var1 (6 features)", fontsize=10)
ax1.set_xlabel("Polynomial Degree"); ax1.set_ylabel("Feature Count")
ax1.grid(True, alpha=0.3, axis="y"); ax1.set_facecolor("#f8f9fc")
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color="#e94560", label=f"Chosen (deg={best_deg1}): {fc1[best_deg1-1]} features"),
                    Patch(color="#0f3460", label="Others")], fontsize=8)

fc2 = []
for d in range(1, 21):
    pf = PolynomialFeatures(degree=d, include_bias=False); pf.fit(X2[:1])
    fc2.append(pf.n_output_features_)
colors2 = ["#0f3460" if d == best_deg2 else "#e94560" for d in range(1,21)]
ax2.bar(range(1, 21), fc2, color=colors2, alpha=0.85, edgecolor="white")
ax2.set_title("var2 (3 features)", fontsize=10)
ax2.set_xlabel("Polynomial Degree"); ax2.set_ylabel("Feature Count")
ax2.grid(True, alpha=0.3, axis="y"); ax2.set_facecolor("#f8f9fc")
ax2.legend(handles=[Patch(color="#0f3460", label=f"Chosen (deg={best_deg2}): {fc2[best_deg2-1]} features"),
                    Patch(color="#e94560", label="Others")], fontsize=8)

fig.tight_layout()
fig.savefig(rf"{OUT_DIR}\fig_feature_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_feature_counts.png")

print("\nAll figures saved to:", OUT_DIR)
print(f"var1 -> MSE={mse1:.4f}, R2={r2_1:.4f}, nfeatures={model1.named_steps['poly'].n_output_features_}")
print(f"var2 -> MSE={mse2:.4f}, R2={r2_2:.4f}, nfeatures={model2.named_steps['poly'].n_output_features_}")
