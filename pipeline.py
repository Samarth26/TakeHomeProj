import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
import os; os.makedirs("images", exist_ok=True)
np.random.seed(42)


# ── 1. Load ────────────────────────────────────────────────────────────────────
with open("census-bureau.columns", "r", encoding="utf-8") as f:
    cols = [line.strip() for line in f if line.strip()]

df = pd.read_csv("census-bureau.data", header=None, names=cols, sep=",", skipinitialspace=True)
print(f"Loaded: {df.shape}")

# ── 2. Duplicates ──────────────────────────────────────────────────────────────
n_before = len(df)
df = df.drop_duplicates()
print(f"Dropped {n_before - len(df)} exact duplicates → {df.shape}")

# ── 3. Labels ──────────────────────────────────────────────────────────────────
y = df["label"].map({"- 50000.": 0, "50000+.": 1})
print(f"Label distribution:\n{y.value_counts()}\n")

# ── 4. Weights ─────────────────────────────────────────────────────────────────
# Sampling weight: each row represents this many people in the US population.
# Normalise around 1.0 so loss magnitude stays comparable to an unweighted run.
raw_weights = df["weight"].copy()
sample_weights = raw_weights / raw_weights.mean()

X = df.drop(columns=["label", "weight"])

# ── 5. Type coercion – low-cardinality int cols → categorical ──────────────────
age_nunique = X["age"].nunique()
for col in X.select_dtypes(include="number").columns:
    if X[col].nunique() < age_nunique:
        X[col] = X[col].astype("object")

# Recode columns are integer codes with no meaningful order — stringify so
# XGBoost treats them as nominal, not ordinal.
X["detailed occupation recode"] = X["detailed occupation recode"].astype(str)
X["detailed industry recode"]   = X["detailed industry recode"].astype(str)

# ── 6. Normalise NA variants ───────────────────────────────────────────────────
# "Not in universe" is a meaningful category (person doesn't meet the question's
# criteria) so we keep it. Only true unknowns become NA.
na_values = ["?", "Not identifiable", np.nan]
X = X.replace(na_values, pd.NA)
X = X.replace({"All other": "Other"})

# ── 7. Drop columns with >50 % true NaN ───────────────────────────────────────
high_missing = X.columns[X.isnull().mean() > 0.5].tolist()
print(f"Dropping high-missing columns: {high_missing}")
X = X.drop(columns=high_missing)
print(f"X shape after drop: {X.shape}\n")

# ── 8. Engineered features ─────────────────────────────────────────────────────
# Wage per year: Census defines full-time as 35 hrs/wk; halve for part-time.
weeks = pd.to_numeric(X["weeks worked in year"], errors="coerce").fillna(0)
wage_per_year = (X["wage per hour"] * 35 * weeks).astype(float)
pt_mask = X["full or part time employment stat"].str.contains("PT", na=False)
wage_per_year[pt_mask] /= 2

X["total_income"]  = wage_per_year

# ── 9. EDA – numerical feature distributions ──────────────────────────────────
num_cols_plot = ["age", "wage per hour", "capital gains", "capital losses",
                 "dividends from stocks", "weeks worked in year", "weight"]
num_data = df[num_cols_plot].copy()

n_cols = 3
n_rows = (len(num_cols_plot) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols_plot):
    axes[i].boxplot(num_data[col].dropna(), vert=False, flierprops=dict(marker="o", markersize=2, alpha=0.3))
    axes[i].set_title(col)
    axes[i].set_xlabel(col)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Distribution of Numerical Features", y=1.02)
plt.tight_layout()
plt.savefig("images/numerical_distributions.png", dpi=150)

# ── 10. EDA – total income by label ───────────────────────────────────────────
plot_df = pd.DataFrame({"total_income": wage_per_year, "label": y.map({0: "≤50K", 1: ">50K"})})

plt.figure(figsize=(8, 6))
sns.boxplot(data=plot_df, x="label", y="total_income", showfliers=False)
plt.title("Total Income Distribution by Label (outliers hidden)")
plt.xlabel("Label")
plt.ylabel("Total Income ($)")
plt.tight_layout()
plt.savefig("images/total_income_by_label.png", dpi=150)

# ── 10. Cross-validation helper ────────────────────────────────────────────────
def cross_validate(X, y, weights, n_splits=5, label="model", **xgb_kwargs):
    X_cv = X.copy()
    for col in X_cv.select_dtypes(include=["object", "str"]).columns:
        X_cv[col] = X_cv[col].astype("category")

    spw = (y == 0).sum() / (y == 1).sum()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    base_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        enable_categorical=True,
        scale_pos_weight=spw,
        device="cpu",
        random_state=42,
    )
    base_params.update(xgb_kwargs)

    fold_train_f1, fold_test_f1, fold_acc = [], [], []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_w    = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X_cv, y):
        X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train, w_test = weights.iloc[train_idx], weights.iloc[test_idx]

        model = xgb.XGBClassifier(**base_params)
        model.fit(X_train, y_train, sample_weight=w_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        oof_pred[test_idx] = y_pred_test
        oof_w[test_idx]    = w_test.to_numpy()

        fold_train_f1.append(f1_score(y_train, y_pred_train, average="macro", sample_weight=w_train))
        fold_test_f1.append(f1_score(y_test,  y_pred_test,  average="macro", sample_weight=w_test))
        fold_acc.append(accuracy_score(y_test, y_pred_test, sample_weight=w_test))

    print(f"\n── {label} ({n_splits}-fold CV, scale_pos_weight={spw:.1f}) ──")
    print(f"  Train Macro F1 : {np.mean(fold_train_f1):.4f} ± {np.std(fold_train_f1):.4f}")
    print(f"  Test  Macro F1 : {np.mean(fold_test_f1):.4f} ± {np.std(fold_test_f1):.4f}")
    print(f"  Test  Accuracy : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(classification_report(y, oof_pred, sample_weight=oof_w))

    model_full = xgb.XGBClassifier(**base_params)
    model_full.fit(X_cv, y, sample_weight=weights)
    return model_full, np.mean(fold_train_f1), np.mean(fold_test_f1)

# ── 11. Optuna tuning ─────────────────────────────────────────────────────────
GAP_PENALTY = 0.5  # penalise overfitting: score = test_f1 - 0.5 * gap

def optuna_objective(trial, X, y, weights):
    params = dict(
        max_depth    = trial.suggest_int("max_depth",    6, 10),
        max_leaves   = trial.suggest_int("max_leaves",   20, 40),
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50),
    )
    _, train_f1, test_f1 = cross_validate(X, y, weights, n_splits=3, label="optuna", **params)
    gap = train_f1 - test_f1
    trial.set_user_attr("train_f1", train_f1)
    trial.set_user_attr("gap", gap)
    return test_f1 - GAP_PENALTY * gap

def print_trials(study):
    df = (
        study.trials_dataframe()
        .sort_values(["value", "user_attrs_gap"], ascending=[False, True])
    )
    df["test_f1"] = df["value"] + GAP_PENALTY * df["user_attrs_gap"]
    print(df[["number", "value", "test_f1", "user_attrs_train_f1", "user_attrs_gap",
              "params_max_depth", "params_max_leaves", "params_n_estimators"]]
          .rename(columns={"value": "score", "user_attrs_train_f1": "train_f1",
                            "user_attrs_gap": "gap"})
          .to_string(index=False, float_format="{:.4f}".format))

print("Running optuna hyperparameter search…")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda trial: optuna_objective(trial, X, y, sample_weights), n_trials=30)

best_params = study.best_params
best_gap = study.best_trial.user_attrs["gap"]
print(f"\nBest trial — score: {study.best_value:.4f}  test_f1: {study.best_value + GAP_PENALTY * best_gap:.4f}  gap: {best_gap:.4f}")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print("\nAll trials:")
print_trials(study)

# ── 12. Baseline – all features ───────────────────────────────────────────────
print(f"\nTraining baseline on {X.shape}")
baseline_model, base_train_f1, base_test_f1 = cross_validate(X, y, sample_weights, label="Baseline (all features)", **best_params)

# ── 12. Feature importance from baseline ───────────────────────────────────────
importance_type = "gain"
raw_imp = baseline_model.get_booster().get_score(importance_type=importance_type)
feature_importance = (
    pd.Series(raw_imp)
    .reindex(X.columns, fill_value=0)
    .sort_values(ascending=False)
)

print(f"\nTop features by {importance_type}:")
print(feature_importance.to_frame(name=importance_type).to_string())

plt.figure(figsize=(10, 8))
feature_importance.sort_values().plot(kind="barh")
plt.title(f"Baseline Feature Importance ({importance_type})")
plt.xlabel(importance_type)
plt.tight_layout()
plt.savefig("images/feature_importance_baseline.png", dpi=150)


# -- 13. Colinearity Comparison Major Occupation Code and Major Industry Code 

X_col_check = X.drop(columns=["detailed occupation recode", "detailed industry recode"])
model_col_check, col_check_train_f1, col_check_test_f1 = cross_validate(X_col_check, y, sample_weights, label="Colinearity check", **best_params)
# Feature Importance for colinearity check model
raw_imp_col_check = model_col_check.get_booster().get_score(importance_type=importance_type)
feature_importance_col_check = (
    pd.Series(raw_imp_col_check)
    .reindex(X_col_check.columns, fill_value=0)
    .sort_values(ascending=False)
)
print(f"\nColinearity check model — top features by {importance_type}:")
print(feature_importance_col_check.to_frame(name=importance_type).to_string())

# Plotting feature importance for colinearity check model
plt.figure(figsize=(10, 8))
feature_importance_col_check.sort_values().plot(kind="barh")
plt.title(f"Colinearity Check Model Importance ({importance_type})")
plt.xlabel(importance_type)
plt.tight_layout()
plt.savefig("images/feature_importance_colinearity_check.png", dpi=150)

# ── 14. Selected-feature model ────────────────────────────────────────────────
top8 = feature_importance.iloc[:-9].index.tolist()
extra = ["total_income"]
selected = list(dict.fromkeys(top8 + extra))
selected = [c for c in selected if c in X.columns]
print(f"\nSelected features ({len(selected)}): {selected}")

X_sel = X[selected]

print("Running optuna hyperparameter search for selected-feature model…")
study_sel = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_sel.optimize(lambda trial: optuna_objective(trial, X_sel, y, sample_weights), n_trials=30)

best_params_sel = study_sel.best_params
best_gap_sel = study_sel.best_trial.user_attrs["gap"]
print(f"\nBest trial — score: {study_sel.best_value:.4f}  test_f1: {study_sel.best_value + GAP_PENALTY * best_gap_sel:.4f}  gap: {best_gap_sel:.4f}")
for param, value in best_params_sel.items():
    print(f"  {param}: {value}")
print("\nAll trials:")
print_trials(study_sel)

sel_model, sel_train_f1, sel_test_f1 = cross_validate(X_sel, y, sample_weights, label="Selected features", **best_params_sel)

raw_imp_sel = sel_model.get_booster().get_score(importance_type=importance_type)
feature_importance_sel = (
    pd.Series(raw_imp_sel)
    .reindex(X_sel.columns, fill_value=0)
    .sort_values(ascending=False)
)

print(f"\nSelected model — top features by {importance_type}:")
print(feature_importance_sel.to_frame(name=importance_type).to_string())

plt.figure(figsize=(10, 8))
feature_importance_sel.sort_values().plot(kind="barh")
plt.title(f"Selected-Feature Model Importance ({importance_type})")
plt.xlabel(importance_type)
plt.tight_layout()
plt.savefig("images/feature_importance_selected.png", dpi=150)



# ── 15. Model comparison ──────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"model": "Baseline (all features)", "n_features": X.shape[1],
     "train_f1": base_train_f1, "test_f1": base_test_f1,
     "gap": base_train_f1 - base_test_f1},
    {"model": "Selected features",       "n_features": X_sel.shape[1],
     "train_f1": sel_train_f1,  "test_f1": sel_test_f1,
     "gap": sel_train_f1 - sel_test_f1},
]).sort_values(["test_f1", "gap"], ascending=[False, True])

print("\n── Model Comparison ──")
print(summary.to_string(index=False, float_format="{:.4f}".format))