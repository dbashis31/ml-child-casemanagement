"""
CaseAI Complete Training Pipeline with Visualizations
=====================================================
Self-contained script: generates synthetic data in-memory, trains all models,
and produces publication-ready graphs (AUC-ROC, confusion matrices, feature
importance, SHAP, bias analysis, survival curves, routing performance, etc.)

Output: ./output/graphs/  (PNG files for paper)
        ./output/models/  (trained model artifacts)

Usage:
    python run_training_with_graphs.py
"""
import json
import os
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = Path("/Users/dpatra/Documents/tikrampaper1/workspace/training/output")
GRAPH_DIR = OUTPUT_DIR / "graphs"
MODEL_DIR = OUTPUT_DIR / "models"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
})
sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION (in-memory)
# ---------------------------------------------------------------------------
print("=" * 70)
print("  STEP 1: Generating Synthetic Data")
print("=" * 70)

N_SAMPLES = 5000

CONCERNS = [
    "possible neglect", "suspected physical abuse", "educational neglect",
    "inadequate supervision", "substance abuse in the home",
    "domestic violence exposure", "emotional abuse concerns",
    "failure to thrive", "medical neglect", "housing instability",
]
DETAILS = [
    "child frequently appears unkempt and hungry at school",
    "unexplained bruising observed on arms and legs",
    "child has missed over 30 days of school this semester",
    "child left unsupervised for extended periods",
    "parent appears intoxicated during pickups",
    "loud arguments and sounds of violence heard regularly",
    "child exhibits extreme anxiety and withdrawal behaviors",
    "child has not gained weight in 6 months despite no medical cause",
    "parent refused prescribed medical treatment for child",
    "family evicted and currently homeless",
]
HISTORIES = [
    "no prior CPS involvement", "one prior referral (unsubstantiated)",
    "two prior substantiated findings", "family previously received services",
    "prior case closed after reunification", "no known history",
    "sibling had prior open case", "multiple referrals over past 3 years",
]

def generate_note():
    return (
        f"Referral received regarding {random.choice(CONCERNS)} for child. "
        f"Reporter indicates {random.choice(DETAILS)}. "
        f"Family history: {random.choice(HISTORIES)}."
    )

# Generate base data
data = {
    "child_age": np.random.randint(0, 18, N_SAMPLES),
    "gender": np.random.choice(["MALE", "FEMALE", "NON_BINARY"], N_SAMPLES, p=[0.48, 0.48, 0.04]),
    "prior_cases": np.random.poisson(0.8, N_SAMPLES),
    "prior_referrals": np.random.poisson(1.2, N_SAMPLES),
    "income_bracket": np.random.choice(["LOW", "MEDIUM", "HIGH"], N_SAMPLES, p=[0.45, 0.35, 0.20]),
    "jurisdiction": np.random.choice(["Halifax", "Dartmouth", "Sydney", "Truro", "Kentville"], N_SAMPLES),
    "contact_count": np.random.poisson(3, N_SAMPLES),
    "days_since_last_contact": np.random.exponential(30, N_SAMPLES).astype(int),
    "housing_stable": np.random.choice([0, 1], N_SAMPLES, p=[0.3, 0.7]),
    "substance_concern": np.random.choice([0, 1], N_SAMPLES, p=[0.75, 0.25]),
    "domestic_violence": np.random.choice([0, 1], N_SAMPLES, p=[0.8, 0.2]),
    "mental_health_concern": np.random.choice([0, 1], N_SAMPLES, p=[0.7, 0.3]),
    "note_text": [generate_note() for _ in range(N_SAMPLES)],
}
df = pd.DataFrame(data)

# Generate risk labels with realistic correlations
risk_score = (
    0.3 * df["prior_cases"] +
    0.2 * df["substance_concern"] +
    0.2 * df["domestic_violence"] +
    0.15 * df["mental_health_concern"] +
    0.1 * (1 - df["housing_stable"]) +
    0.05 * (df["child_age"] < 3).astype(int) +
    np.random.normal(0, 0.2, N_SAMPLES)
)
df["risk_label"] = pd.cut(risk_score, bins=[-np.inf, 0.3, 0.7, np.inf], labels=["LOW", "MEDIUM", "HIGH"])

# Eligibility
df["eligible"] = ((df["income_bracket"] != "HIGH") & (df["child_age"] < 13)).astype(int)
df["eligible"] = (df["eligible"] | (np.random.random(N_SAMPLES) < 0.1)).astype(int)

# Case outcome
df["case_closed"] = np.random.choice([0, 1], N_SAMPLES, p=[0.35, 0.65])
df["case_duration_days"] = np.random.exponential(90, N_SAMPLES).astype(int).clip(1, 730)
df["escalated"] = (risk_score > 0.7).astype(int) | (np.random.random(N_SAMPLES) < 0.08).astype(int)

# Routing features
df["caseworker_id"] = np.random.randint(0, 50, N_SAMPLES)
df["caseworker_load"] = np.random.randint(5, 30, N_SAMPLES)
df["caseworker_experience_years"] = np.random.exponential(5, N_SAMPLES).clip(0.5, 25)
df["routing_reward"] = (
    0.5 * df["case_closed"] +
    0.3 * (1 - df["caseworker_load"] / 30) +
    0.2 * np.random.random(N_SAMPLES)
)

# Entity resolution pairs
n_entity_pairs = 3000
entity_pairs = {
    "first_name_sim": np.random.beta(2, 5, n_entity_pairs),
    "last_name_sim": np.random.beta(2, 5, n_entity_pairs),
    "fn_soundex_match": np.random.choice([0, 1], n_entity_pairs, p=[0.7, 0.3]),
    "ln_soundex_match": np.random.choice([0, 1], n_entity_pairs, p=[0.6, 0.4]),
    "dob_exact": np.random.choice([0, 1], n_entity_pairs, p=[0.8, 0.2]),
    "dob_close": np.random.choice([0, 1], n_entity_pairs, p=[0.6, 0.4]),
    "dob_diff_norm": np.random.exponential(0.3, n_entity_pairs).clip(0, 1),
    "city_match": np.random.choice([0, 1], n_entity_pairs, p=[0.5, 0.5]),
    "postal_sim": np.random.beta(3, 3, n_entity_pairs),
}
entity_df = pd.DataFrame(entity_pairs)
# Positive pairs: high similarity
n_pos = n_entity_pairs // 2
entity_df.iloc[:n_pos, 0] = np.random.beta(8, 2, n_pos)  # fn_sim high
entity_df.iloc[:n_pos, 1] = np.random.beta(8, 2, n_pos)  # ln_sim high
entity_df.iloc[:n_pos, 4] = np.random.choice([0, 1], n_pos, p=[0.15, 0.85])  # dob_exact
entity_df["is_match"] = 0
entity_df.iloc[:n_pos, -1] = 1

print(f"  Generated {N_SAMPLES} case samples, {n_entity_pairs} entity pairs")
print(f"  Risk distribution: {df['risk_label'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 2. INTAKE RISK SCORING MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 2: Training Intake Risk Scoring Model")
print("=" * 70)

label_enc = LabelEncoder()
df["risk_encoded"] = label_enc.fit_transform(df["risk_label"])
risk_classes = label_enc.classes_

# Tabular features
tab_features = ["child_age", "prior_cases", "prior_referrals", "contact_count",
                "days_since_last_contact", "housing_stable", "substance_concern",
                "domestic_violence", "mental_health_concern", "caseworker_load"]

# Add encoded categoricals
for col in ["gender", "income_bracket", "jurisdiction"]:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    tab_features.extend(dummies.columns.tolist())

X_tab = df[tab_features].values
y_risk = df["risk_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(X_tab, y_risk, test_size=0.2, stratify=y_risk, random_state=SEED)

# --- XGBoost Tabular Model ---
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", random_state=SEED,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
xgb_proba_train = xgb_model.predict_proba(X_train)
xgb_proba_test = xgb_model.predict_proba(X_test)
xgb_pred = xgb_model.predict(X_test)

# --- NLP TF-IDF Model ---
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
X_nlp_train = tfidf.fit_transform(df.iloc[X_train.shape[0]:]["note_text"].values.tolist()[:X_train.shape[0]]
    if X_train.shape[0] < len(df) else tfidf.fit_transform(df["note_text"].values))
# Refit properly
all_indices = np.arange(len(df))
train_idx, test_idx = train_test_split(all_indices, test_size=0.2, stratify=y_risk, random_state=SEED)
X_nlp_all = tfidf.fit_transform(df["note_text"].values)
X_nlp_train = X_nlp_all[train_idx]
X_nlp_test = X_nlp_all[test_idx]

nlp_model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0, random_state=SEED)
nlp_model.fit(X_nlp_train, y_train)
nlp_proba_train = nlp_model.predict_proba(X_nlp_train)
nlp_proba_test = nlp_model.predict_proba(X_nlp_test)

# --- Meta-Learner (Fusion) ---
X_meta_train = np.hstack([xgb_proba_train, nlp_proba_train])
X_meta_test = np.hstack([xgb_proba_test, nlp_proba_test])

meta_base = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=SEED)
meta_model = CalibratedClassifierCV(meta_base, cv=3, method="isotonic")
meta_model.fit(X_meta_train, y_train)
meta_proba = meta_model.predict_proba(X_meta_test)
meta_pred = meta_model.predict(X_meta_test)

print(f"  XGBoost Accuracy:     {accuracy_score(y_test, xgb_pred):.4f}")
print(f"  NLP (TF-IDF) Accuracy: {accuracy_score(y_test, nlp_model.predict(X_nlp_test)):.4f}")
print(f"  Meta-Learner Accuracy: {accuracy_score(y_test, meta_pred):.4f}")

# ===================== RISK MODEL GRAPHS =====================

# --- Graph 1: Confusion Matrix (Meta-Learner) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, preds) in enumerate([
    ("XGBoost (Tabular)", xgb_pred),
    ("NLP (TF-IDF+LogReg)", nlp_model.predict(X_nlp_test)),
    ("Meta-Learner (Fused)", meta_pred),
]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=risk_classes,
                yticklabels=risk_classes, ax=axes[idx])
    axes[idx].set_title(f"{name}\nAcc: {accuracy_score(y_test, preds):.3f}")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
fig.suptitle("Risk Scoring: Confusion Matrices", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "01_risk_confusion_matrices.png")
plt.close()
print("  [Saved] 01_risk_confusion_matrices.png")

# --- Graph 2: Multi-class ROC-AUC ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, probas) in enumerate([
    ("XGBoost", xgb_proba_test),
    ("NLP", nlp_proba_test),
    ("Meta-Learner", meta_proba),
]):
    for i, cls in enumerate(risk_classes):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probas[:, i])
        auc_val = roc_auc_score(y_bin, probas[:, i])
        axes[idx].plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})", linewidth=2)
    axes[idx].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[idx].set_title(f"{name}")
    axes[idx].set_xlabel("False Positive Rate")
    axes[idx].set_ylabel("True Positive Rate")
    axes[idx].legend(loc="lower right")
fig.suptitle("Risk Scoring: ROC-AUC Curves (One-vs-Rest)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "02_risk_roc_auc.png")
plt.close()
print("  [Saved] 02_risk_roc_auc.png")

# --- Graph 3: Feature Importance (XGBoost) ---
importance = xgb_model.feature_importances_
feat_imp_df = pd.DataFrame({"feature": tab_features, "importance": importance})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(feat_imp_df["feature"], feat_imp_df["importance"], color=sns.color_palette("viridis", 15))
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Risk Model: Top 15 Feature Importances (XGBoost)", fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "03_risk_feature_importance.png")
plt.close()
print("  [Saved] 03_risk_feature_importance.png")

# --- Graph 4: SHAP Summary ---
try:
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:500])
    fig, ax = plt.subplots(figsize=(10, 8))
    # For multi-class, take class with highest risk (index 2 = HIGH)
    if isinstance(shap_values, list):
        sv = shap_values[2]
    else:
        sv = shap_values[:, :, 2] if shap_values.ndim == 3 else shap_values
    shap.summary_plot(sv, X_test[:500], feature_names=tab_features, show=False, max_display=15)
    plt.title("SHAP Values: HIGH Risk Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "04_risk_shap_summary.png")
    plt.close()
    print("  [Saved] 04_risk_shap_summary.png")
except Exception as e:
    print(f"  [SKIP] SHAP plot: {e}")

# --- Graph 5: Abstention Analysis ---
thresholds = np.arange(0.3, 0.96, 0.02)
abstention_rates = []
confident_accuracies = []
for t in thresholds:
    max_conf = meta_proba.max(axis=1)
    mask = max_conf >= t
    abstention_rates.append(1 - mask.mean())
    if mask.sum() > 0:
        confident_accuracies.append(accuracy_score(y_test[mask], meta_pred[mask]))
    else:
        confident_accuracies.append(np.nan)

fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = "#2196F3"
color2 = "#FF5722"
ax1.plot(thresholds, [a * 100 for a in abstention_rates], color=color1, linewidth=2.5, label="Abstention Rate (%)")
ax1.set_xlabel("Confidence Threshold")
ax1.set_ylabel("Abstention Rate (%)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.axvline(x=0.65, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.65)")

ax2 = ax1.twinx()
ax2.plot(thresholds, [a * 100 if not np.isnan(a) else np.nan for a in confident_accuracies],
         color=color2, linewidth=2.5, label="Confident Accuracy (%)")
ax2.set_ylabel("Accuracy on Confident Predictions (%)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
plt.title("Abstention Analysis: Coverage vs Accuracy Trade-off", fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "05_abstention_analysis.png")
plt.close()
print("  [Saved] 05_abstention_analysis.png")

# --- Graph 6: Calibration Curves ---
fig, ax = plt.subplots(figsize=(8, 8))
for i, cls in enumerate(risk_classes):
    y_bin = (y_test == i).astype(int)
    prob_true, prob_pred = calibration_curve(y_bin, meta_proba[:, i], n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", label=f"{cls}", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Risk Model: Calibration Curves (Meta-Learner)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "06_calibration_curves.png")
plt.close()
print("  [Saved] 06_calibration_curves.png")

# --- Graph 7: Learning Curves ---
fig, ax = plt.subplots(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric="mlogloss"),
    X_tab, y_risk, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
)
ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color="#2196F3")
ax.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1, color="#FF5722")
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2196F3", label="Training Score", linewidth=2)
ax.plot(train_sizes, test_scores.mean(axis=1), "o-", color="#FF5722", label="Cross-Validation Score", linewidth=2)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Risk Model: Learning Curves", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "07_learning_curves.png")
plt.close()
print("  [Saved] 07_learning_curves.png")

# ---------------------------------------------------------------------------
# 3. ELIGIBILITY PREDICTION MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 3: Training Eligibility Prediction Model")
print("=" * 70)

elig_features = ["child_age", "prior_cases", "income_bracket_MEDIUM", "income_bracket_LOW",
                 "housing_stable", "substance_concern", "contact_count"]
# Ensure columns exist
for c in elig_features:
    if c not in df.columns:
        if "income_bracket" in c:
            df[c] = 0
        else:
            df[c] = 0

# Fix: create income dummies if needed
if "income_bracket_LOW" not in df.columns:
    inc_dum = pd.get_dummies(df["income_bracket"], prefix="income_bracket", drop_first=False)
    for c in inc_dum.columns:
        if c not in df.columns:
            df[c] = inc_dum[c]

elig_features_actual = [c for c in elig_features if c in df.columns]
X_elig = df[elig_features_actual].values
y_elig = df["eligible"].values

Xe_train, Xe_test, ye_train, ye_test = train_test_split(X_elig, y_elig, test_size=0.2, stratify=y_elig, random_state=SEED)

elig_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    eval_metric="logloss", random_state=SEED,
)
elig_model.fit(Xe_train, ye_train, eval_set=[(Xe_train, ye_train), (Xe_test, ye_test)], verbose=False)
ye_pred = elig_model.predict(Xe_test)
ye_proba = elig_model.predict_proba(Xe_test)[:, 1]

print(f"  Eligibility Accuracy: {accuracy_score(ye_test, ye_pred):.4f}")
print(f"  Eligibility AUC-ROC:  {roc_auc_score(ye_test, ye_proba):.4f}")
print(f"  Eligibility F1:       {f1_score(ye_test, ye_pred):.4f}")

# --- Graph 8: Eligibility ROC + PR Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(ye_test, ye_proba)
roc_auc_val = roc_auc_score(ye_test, ye_proba)
ax1.plot(fpr, tpr, color="#4CAF50", linewidth=2.5, label=f"AUC = {roc_auc_val:.3f}")
ax1.fill_between(fpr, tpr, alpha=0.15, color="#4CAF50")
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("Eligibility: ROC Curve")
ax1.legend(loc="lower right", fontsize=12)

prec, rec, _ = precision_recall_curve(ye_test, ye_proba)
pr_auc = auc(rec, prec)
ax2.plot(rec, prec, color="#FF9800", linewidth=2.5, label=f"PR-AUC = {pr_auc:.3f}")
ax2.fill_between(rec, prec, alpha=0.15, color="#FF9800")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Eligibility: Precision-Recall Curve")
ax2.legend(loc="lower left", fontsize=12)

fig.suptitle("Eligibility Prediction Model Performance", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "08_eligibility_roc_pr.png")
plt.close()
print("  [Saved] 08_eligibility_roc_pr.png")

# --- Graph 9: Eligibility Confusion Matrix + SHAP ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

cm = confusion_matrix(ye_test, ye_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Not Eligible", "Eligible"],
            yticklabels=["Not Eligible", "Eligible"], ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title(f"Confusion Matrix (Acc: {accuracy_score(ye_test, ye_pred):.3f})")

# SHAP importance bar
try:
    import shap
    elig_explainer = shap.TreeExplainer(elig_model)
    elig_shap = elig_explainer.shap_values(Xe_test[:300])
    mean_shap = np.abs(elig_shap).mean(axis=0)
    shap_df = pd.DataFrame({"feature": elig_features_actual, "shap": mean_shap}).sort_values("shap", ascending=True)
    ax2.barh(shap_df["feature"], shap_df["shap"], color=sns.color_palette("YlOrRd", len(shap_df)))
    ax2.set_xlabel("Mean |SHAP Value|")
    ax2.set_title("SHAP Feature Importance")
except Exception as e:
    eimp = elig_model.feature_importances_
    eimp_df = pd.DataFrame({"feature": elig_features_actual, "importance": eimp}).sort_values("importance", ascending=True)
    ax2.barh(eimp_df["feature"], eimp_df["importance"], color=sns.color_palette("YlOrRd", len(eimp_df)))
    ax2.set_xlabel("Feature Importance")
    ax2.set_title("Gini Feature Importance")

fig.suptitle("Eligibility Prediction: Evaluation", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "09_eligibility_cm_shap.png")
plt.close()
print("  [Saved] 09_eligibility_cm_shap.png")

# ---------------------------------------------------------------------------
# 4. ENTITY RESOLUTION MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 4: Training Entity Resolution Model")
print("=" * 70)

entity_features = list(entity_df.columns[:-1])
X_ent = entity_df[entity_features].values
y_ent = entity_df["is_match"].values

Xen_train, Xen_test, yen_train, yen_test = train_test_split(X_ent, y_ent, test_size=0.2, stratify=y_ent, random_state=SEED)

ent_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                eval_metric="logloss", random_state=SEED)
ent_model.fit(Xen_train, yen_train, eval_set=[(Xen_train, yen_train), (Xen_test, yen_test)], verbose=False)
yen_pred = ent_model.predict(Xen_test)
yen_proba = ent_model.predict_proba(Xen_test)[:, 1]

print(f"  Entity Resolution Accuracy: {accuracy_score(yen_test, yen_pred):.4f}")
print(f"  Entity Resolution AUC-ROC:  {roc_auc_score(yen_test, yen_proba):.4f}")

# --- Graph 10: Entity Resolution ROC + Threshold Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr_e, tpr_e, _ = roc_curve(yen_test, yen_proba)
ax1.plot(fpr_e, tpr_e, color="#9C27B0", linewidth=2.5, label=f"AUC = {roc_auc_score(yen_test, yen_proba):.3f}")
ax1.fill_between(fpr_e, tpr_e, alpha=0.15, color="#9C27B0")
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(fontsize=12)

# Threshold vs Precision/Recall
prec_e, rec_e, thresh_e = precision_recall_curve(yen_test, yen_proba)
ax2.plot(thresh_e, prec_e[:-1], color="#2196F3", linewidth=2, label="Precision")
ax2.plot(thresh_e, rec_e[:-1], color="#FF5722", linewidth=2, label="Recall")
ax2.axvline(x=0.85, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.85)")
ax2.set_xlabel("Classification Threshold")
ax2.set_ylabel("Score")
ax2.set_title("Precision-Recall vs Threshold")
ax2.legend(fontsize=11)

fig.suptitle("Entity Resolution (Duplicate Detection) Performance", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "10_entity_resolution.png")
plt.close()
print("  [Saved] 10_entity_resolution.png")

# ---------------------------------------------------------------------------
# 5. ROUTING OPTIMIZATION (LinUCB Bandit)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 5: Training Routing Optimization Model (LinUCB)")
print("=" * 70)

routing_features = ["child_age", "prior_cases", "caseworker_load", "contact_count",
                    "housing_stable", "substance_concern", "domestic_violence"]
X_route = df[routing_features].values
scaler = StandardScaler()
X_route_scaled = scaler.fit_transform(X_route)

n_arms = 50  # caseworkers
n_features = X_route_scaled.shape[1]
alpha = 0.25

# LinUCB simulation
A_mats = [np.eye(n_features) for _ in range(n_arms)]
b_vecs = [np.zeros(n_features) for _ in range(n_arms)]
cumulative_rewards = []
running_avg = []
arm_pulls = np.zeros(n_arms)
total_reward = 0

for t in range(len(df)):
    ctx = X_route_scaled[t]
    logged_arm = df["caseworker_id"].iloc[t]
    reward = df["routing_reward"].iloc[t]

    # UCB scores
    ucb_scores = np.zeros(n_arms)
    for a in range(n_arms):
        A_inv = np.linalg.inv(A_mats[a])
        theta = A_inv @ b_vecs[a]
        ucb_scores[a] = theta @ ctx + alpha * np.sqrt(ctx @ A_inv @ ctx)

    selected = int(np.argmax(ucb_scores))

    # Replay: update if match
    if selected == logged_arm:
        A_mats[selected] += np.outer(ctx, ctx)
        b_vecs[selected] += reward * ctx
        total_reward += reward
        arm_pulls[selected] += 1

    cumulative_rewards.append(total_reward)
    running_avg.append(total_reward / max(arm_pulls.sum(), 1))

print(f"  LinUCB Total Reward:    {total_reward:.2f}")
print(f"  LinUCB Match Rate:      {arm_pulls.sum() / len(df) * 100:.1f}%")
print(f"  LinUCB Avg Reward:      {total_reward / max(arm_pulls.sum(), 1):.4f}")

# --- Graph 11: Routing - Cumulative Reward & Arm Distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

steps = np.arange(len(cumulative_rewards))
ax1.plot(steps[::50], running_avg[::50], color="#009688", linewidth=2)
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Running Average Reward")
ax1.set_title("LinUCB: Running Average Reward over Time")
ax1.axhline(y=np.mean(df["routing_reward"]), color="red", linestyle="--", alpha=0.7, label="Random Baseline")
ax1.legend()

# Arm pull distribution
nonzero_pulls = arm_pulls[arm_pulls > 0]
ax2.hist(nonzero_pulls, bins=20, color="#3F51B5", edgecolor="white", alpha=0.8)
ax2.set_xlabel("Number of Pulls")
ax2.set_ylabel("Number of Caseworkers")
ax2.set_title("LinUCB: Caseworker Selection Distribution")
ax2.axvline(nonzero_pulls.mean(), color="red", linestyle="--", label=f"Mean={nonzero_pulls.mean():.0f}")
ax2.legend()

fig.suptitle("Routing Optimization: LinUCB Contextual Bandit", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "11_routing_linucb.png")
plt.close()
print("  [Saved] 11_routing_linucb.png")

# ---------------------------------------------------------------------------
# 6. OUTCOME PREDICTION MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 6: Training Outcome Prediction Models")
print("=" * 70)

outcome_features = ["child_age", "prior_cases", "prior_referrals", "contact_count",
                    "housing_stable", "substance_concern", "domestic_violence",
                    "mental_health_concern", "caseworker_load", "caseworker_experience_years"]

X_out = df[outcome_features].values

# --- 6a: Escalation Risk ---
y_esc = df["escalated"].values
Xo_train, Xo_test, yo_train, yo_test = train_test_split(X_out, y_esc, test_size=0.2, stratify=y_esc, random_state=SEED)

esc_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                eval_metric="logloss", random_state=SEED)
esc_model.fit(Xo_train, yo_train, eval_set=[(Xo_train, yo_train), (Xo_test, yo_test)], verbose=False)
esc_pred = esc_model.predict(Xo_test)
esc_proba = esc_model.predict_proba(Xo_test)[:, 1]

# --- 6b: Case Closure ---
y_close = df["case_closed"].values
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_out, y_close, test_size=0.2, stratify=y_close, random_state=SEED)

close_model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                  eval_metric="logloss", random_state=SEED)
close_model.fit(Xc_train, yc_train, eval_set=[(Xc_train, yc_train), (Xc_test, yc_test)], verbose=False)
close_pred = close_model.predict(Xc_test)
close_proba = close_model.predict_proba(Xc_test)[:, 1]

# --- 6c: Time-to-Closure Regression ---
closed_mask = df["case_closed"] == 1
X_surv = df.loc[closed_mask, outcome_features].values
y_dur = df.loc[closed_mask, "case_duration_days"].values
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_surv, y_dur, test_size=0.2, random_state=SEED)

dur_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED)
dur_model.fit(Xs_train, ys_train)
dur_pred = dur_model.predict(Xs_test)

print(f"  Escalation AUC:       {roc_auc_score(yo_test, esc_proba):.4f}")
print(f"  Closure AUC:          {roc_auc_score(yc_test, close_proba):.4f}")
print(f"  Duration MAE:         {mean_absolute_error(ys_test, dur_pred):.1f} days")
print(f"  Duration R2:          {r2_score(ys_test, dur_pred):.4f}")

# --- Graph 12: Outcome Models ROC ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, name, yt, yp in [
    (axes[0], "Escalation Risk", yo_test, esc_proba),
    (axes[1], "Case Closure", yc_test, close_proba),
]:
    fpr_o, tpr_o, _ = roc_curve(yt, yp)
    auc_o = roc_auc_score(yt, yp)
    ax.plot(fpr_o, tpr_o, linewidth=2.5, label=f"AUC = {auc_o:.3f}")
    ax.fill_between(fpr_o, tpr_o, alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name)
    ax.legend(fontsize=12)

# Actual vs Predicted duration
axes[2].scatter(ys_test, dur_pred, alpha=0.3, s=15, color="#FF5722")
axes[2].plot([0, 730], [0, 730], "k--", alpha=0.5)
axes[2].set_xlabel("Actual Duration (days)")
axes[2].set_ylabel("Predicted Duration (days)")
axes[2].set_title(f"Time-to-Closure (MAE={mean_absolute_error(ys_test, dur_pred):.0f}d, R\u00b2={r2_score(ys_test, dur_pred):.3f})")

fig.suptitle("Outcome Prediction Models", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "12_outcome_models.png")
plt.close()
print("  [Saved] 12_outcome_models.png")

# --- Graph 13: Survival Curve (Kaplan-Meier Style) ---
fig, ax = plt.subplots(figsize=(10, 6))
for priority, color, label in [
    ("CRITICAL", "#F44336", "Critical"), ("HIGH", "#FF9800", "High"),
    ("MEDIUM", "#2196F3", "Medium"), ("LOW", "#4CAF50", "Low"),
]:
    mask = df["risk_label"] == (priority if priority != "CRITICAL" else "HIGH")
    if priority == "CRITICAL":
        mask = df["escalated"] == 1
        label = "Escalated"
    durations = df.loc[mask, "case_duration_days"].sort_values().values
    if len(durations) == 0:
        continue
    survival = 1 - np.arange(1, len(durations) + 1) / len(durations)
    ax.step(durations, survival, where="post", linewidth=2, color=color, label=label)

ax.set_xlabel("Days Since Case Opening")
ax.set_ylabel("Proportion of Cases Still Open")
ax.set_title("Kaplan-Meier Style Survival Curves by Risk Category", fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0, 500)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "13_survival_curves.png")
plt.close()
print("  [Saved] 13_survival_curves.png")

# ---------------------------------------------------------------------------
# 7. BIAS MONITORING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 7: Bias Monitoring Analysis")
print("=" * 70)

# Simulate risk predictions per group
df["predicted_risk"] = label_enc.inverse_transform(meta_model.predict(X_meta_test).tolist()[:len(df)]
    if len(X_meta_test) >= len(df) else
    label_enc.inverse_transform(
        meta_model.predict(
            np.hstack([
                xgb_model.predict_proba(X_tab),
                nlp_model.predict_proba(X_nlp_all)
            ])
        )
    )
)
df["pred_high_risk"] = (df["predicted_risk"] == "HIGH").astype(int)

# --- Graph 14: Bias - Disparate Impact Across Demographics ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (group_col, title) in enumerate([
    ("gender", "Gender"), ("income_bracket", "Income Bracket"), ("jurisdiction", "Jurisdiction")
]):
    rates = df.groupby(group_col)["pred_high_risk"].mean().sort_values(ascending=True)
    ref_rate = rates.max()
    dir_vals = rates / ref_rate

    colors = ["#F44336" if d < 0.8 or d > 1.25 else "#4CAF50" for d in dir_vals]
    bars = axes[idx].barh(rates.index, dir_vals.values, color=colors, edgecolor="white")
    axes[idx].axvline(x=0.8, color="red", linestyle="--", alpha=0.7, label="4/5 Rule Lower")
    axes[idx].axvline(x=1.0, color="gray", linestyle="-", alpha=0.3)
    axes[idx].axvline(x=1.25, color="red", linestyle="--", alpha=0.7, label="4/5 Rule Upper")
    axes[idx].set_xlabel("Disparate Impact Ratio")
    axes[idx].set_title(f"By {title}")
    if idx == 0:
        axes[idx].legend(fontsize=9)

fig.suptitle("Bias Monitoring: Disparate Impact Analysis (HIGH Risk Predictions)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "14_bias_disparate_impact.png")
plt.close()
print("  [Saved] 14_bias_disparate_impact.png")

# --- Graph 15: Bias - Positive Rate Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (group_col, title) in enumerate([
    ("gender", "Gender"), ("income_bracket", "Income Bracket"), ("jurisdiction", "Jurisdiction")
]):
    group_rates = df.groupby(group_col).agg(
        high_risk_rate=("pred_high_risk", "mean"),
        count=("pred_high_risk", "size"),
    ).sort_values("high_risk_rate", ascending=True)

    bars = axes[idx].barh(group_rates.index, group_rates["high_risk_rate"] * 100,
                          color=sns.color_palette("coolwarm", len(group_rates)), edgecolor="white")
    for bar, cnt in zip(bars, group_rates["count"]):
        axes[idx].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      f"n={cnt}", va="center", fontsize=9)
    axes[idx].set_xlabel("High Risk Rate (%)")
    axes[idx].set_title(f"By {title}")

fig.suptitle("Bias Monitoring: High Risk Prediction Rate by Demographic Group", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "15_bias_positive_rates.png")
plt.close()
print("  [Saved] 15_bias_positive_rates.png")

# ---------------------------------------------------------------------------
# 8. MODEL COMPARISON DASHBOARD
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 8: Generating Summary Dashboard")
print("=" * 70)

# --- Graph 16: Overall Model Comparison ---
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

# (0,0) - Risk model accuracies
ax = fig.add_subplot(gs[0, 0])
models_names = ["XGBoost\n(Tabular)", "NLP\n(TF-IDF)", "Meta-Learner\n(Fused)"]
accuracies = [
    accuracy_score(y_test, xgb_pred),
    accuracy_score(y_test, nlp_model.predict(X_nlp_test)),
    accuracy_score(y_test, meta_pred),
]
bars = ax.bar(models_names, [a * 100 for a in accuracies],
              color=["#2196F3", "#FF9800", "#4CAF50"], edgecolor="white")
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.1%}", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Risk Scoring: Model Comparison")
ax.set_ylim(0, 100)

# (0,1) - AUC comparison across all models
ax = fig.add_subplot(gs[0, 1])
all_aucs = {
    "Risk\n(XGBoost)": roc_auc_score(y_test, xgb_proba_test, multi_class="ovr"),
    "Risk\n(Meta)": roc_auc_score(y_test, meta_proba, multi_class="ovr"),
    "Eligibility": roc_auc_score(ye_test, ye_proba),
    "Entity\nResolution": roc_auc_score(yen_test, yen_proba),
    "Escalation": roc_auc_score(yo_test, esc_proba),
    "Closure": roc_auc_score(yc_test, close_proba),
}
colors_auc = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
bars = ax.bar(all_aucs.keys(), all_aucs.values(), color=colors_auc, edgecolor="white")
for bar, val in zip(bars, all_aucs.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC-ROC Across All Models")
ax.set_ylim(0.5, 1.05)
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random")
ax.tick_params(axis="x", rotation=15)

# (0,2) - Abstention impact
ax = fig.add_subplot(gs[0, 2])
threshold = 0.65
max_conf = meta_proba.max(axis=1)
confident_mask = max_conf >= threshold
abs_rate = 1 - confident_mask.mean()
conf_acc = accuracy_score(y_test[confident_mask], meta_pred[confident_mask]) if confident_mask.sum() > 0 else 0
full_acc = accuracy_score(y_test, meta_pred)

categories = ["Full\nDataset", "Confident\nOnly", "Abstained\n(Human Review)"]
values = [full_acc * 100, conf_acc * 100, abs_rate * 100]
colors_abs = ["#2196F3", "#4CAF50", "#FF5722"]
bars = ax.bar(categories, values, color=colors_abs, edgecolor="white")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Percentage")
ax.set_title(f"Abstention Impact (Threshold={threshold})")

# (1,0) - Risk distribution
ax = fig.add_subplot(gs[1, 0])
risk_dist = df["risk_label"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"])
ax.pie(risk_dist.values, labels=risk_dist.index, autopct="%1.1f%%",
       colors=["#4CAF50", "#FF9800", "#F44336"], startangle=90,
       textprops={"fontsize": 11})
ax.set_title("Risk Label Distribution")

# (1,1) - Training data volume
ax = fig.add_subplot(gs[1, 1])
data_volumes = {
    "Risk\nScoring": N_SAMPLES,
    "Eligibility": N_SAMPLES,
    "Entity\nResolution": n_entity_pairs,
    "Routing": N_SAMPLES,
    "Outcome\nPrediction": N_SAMPLES,
}
ax.bar(data_volumes.keys(), data_volumes.values(), color="#78909C", edgecolor="white")
for i, (k, v) in enumerate(data_volumes.items()):
    ax.text(i, v + 50, f"{v:,}", ha="center", fontsize=9)
ax.set_ylabel("Number of Samples")
ax.set_title("Training Data Volume")

# (1,2) - Time-to-closure distribution
ax = fig.add_subplot(gs[1, 2])
for risk_cat, color in [("LOW", "#4CAF50"), ("MEDIUM", "#FF9800"), ("HIGH", "#F44336")]:
    durations = df.loc[df["risk_label"] == risk_cat, "case_duration_days"]
    ax.hist(durations, bins=30, alpha=0.5, color=color, label=risk_cat, edgecolor="white")
ax.set_xlabel("Duration (days)")
ax.set_ylabel("Count")
ax.set_title("Case Duration by Risk Level")
ax.legend()

fig.suptitle("CaseAI: Model Performance Dashboard", fontsize=16, fontweight="bold", y=1.02)
plt.savefig(GRAPH_DIR / "16_model_dashboard.png")
plt.close()
print("  [Saved] 16_model_dashboard.png")

# --- Graph 17: Training Loss Curves (XGBoost) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model_obj, metric_name) in enumerate([
    ("Risk Scoring", xgb_model, "mlogloss"),
    ("Eligibility", elig_model, "logloss"),
    ("Escalation", esc_model, "logloss"),
]):
    results = model_obj.evals_result()
    keys = list(results.keys())
    if len(keys) >= 2:
        train_loss = results[keys[0]][metric_name]
        val_loss = results[keys[1]][metric_name]
        epochs = range(1, len(train_loss) + 1)
        axes[idx].plot(epochs, train_loss, label="Train", linewidth=2, color="#2196F3")
        axes[idx].plot(epochs, val_loss, label="Validation", linewidth=2, color="#FF5722")
        axes[idx].set_xlabel("Boosting Round")
        axes[idx].set_ylabel(f"Loss ({metric_name})")
        axes[idx].set_title(name)
        axes[idx].legend()

fig.suptitle("Training Loss Curves (XGBoost)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "17_training_loss_curves.png")
plt.close()
print("  [Saved] 17_training_loss_curves.png")

# --- Graph 18: Correlation Heatmap of Features ---
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols = ["child_age", "prior_cases", "prior_referrals", "contact_count",
             "housing_stable", "substance_concern", "domestic_violence",
             "mental_health_concern", "caseworker_load", "caseworker_experience_years",
             "case_duration_days", "escalated", "case_closed"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "18_correlation_heatmap.png")
plt.close()
print("  [Saved] 18_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 9. SAVE ALL METRICS SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  STEP 9: Saving Metrics Summary")
print("=" * 70)

metrics_summary = {
    "generated_at": datetime.now().isoformat(),
    "data": {"n_samples": N_SAMPLES, "n_entity_pairs": n_entity_pairs},
    "risk_scoring": {
        "xgboost_accuracy": round(accuracy_score(y_test, xgb_pred), 4),
        "nlp_accuracy": round(accuracy_score(y_test, nlp_model.predict(X_nlp_test)), 4),
        "meta_learner_accuracy": round(accuracy_score(y_test, meta_pred), 4),
        "meta_learner_auc_ovr": round(roc_auc_score(y_test, meta_proba, multi_class="ovr"), 4),
        "abstention_rate_at_065": round(abs_rate, 4),
        "confident_accuracy_at_065": round(conf_acc, 4),
    },
    "eligibility": {
        "accuracy": round(accuracy_score(ye_test, ye_pred), 4),
        "auc_roc": round(roc_auc_score(ye_test, ye_proba), 4),
        "f1_score": round(f1_score(ye_test, ye_pred), 4),
    },
    "entity_resolution": {
        "accuracy": round(accuracy_score(yen_test, yen_pred), 4),
        "auc_roc": round(roc_auc_score(yen_test, yen_proba), 4),
    },
    "routing": {
        "linucb_avg_reward": round(total_reward / max(arm_pulls.sum(), 1), 4),
        "match_rate": round(arm_pulls.sum() / len(df) * 100, 2),
    },
    "outcome_prediction": {
        "escalation_auc": round(roc_auc_score(yo_test, esc_proba), 4),
        "closure_auc": round(roc_auc_score(yc_test, close_proba), 4),
        "duration_mae_days": round(mean_absolute_error(ys_test, dur_pred), 1),
        "duration_r2": round(r2_score(ys_test, dur_pred), 4),
    },
}

with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
print(f"  Metrics saved to {OUTPUT_DIR / 'metrics_summary.json'}")

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  TRAINING COMPLETE - ALL GRAPHS GENERATED")
print("=" * 70)
print(f"\n  Output directory: {GRAPH_DIR}")
print(f"  Total graphs: 18")
print(f"\n  Graphs generated:")
for f in sorted(GRAPH_DIR.glob("*.png")):
    print(f"    - {f.name}")
print(f"\n  Key Metrics:")
print(f"    Risk Model (Meta-Learner) Accuracy:  {metrics_summary['risk_scoring']['meta_learner_accuracy']:.1%}")
print(f"    Risk Model AUC-ROC (OVR):            {metrics_summary['risk_scoring']['meta_learner_auc_ovr']:.4f}")
print(f"    Eligibility AUC-ROC:                 {metrics_summary['eligibility']['auc_roc']:.4f}")
print(f"    Entity Resolution AUC-ROC:           {metrics_summary['entity_resolution']['auc_roc']:.4f}")
print(f"    Escalation AUC-ROC:                  {metrics_summary['outcome_prediction']['escalation_auc']:.4f}")
print(f"    Closure AUC-ROC:                     {metrics_summary['outcome_prediction']['closure_auc']:.4f}")
print(f"    Duration MAE:                        {metrics_summary['outcome_prediction']['duration_mae_days']:.1f} days")
print(f"    Abstention Rate (t=0.65):            {metrics_summary['risk_scoring']['abstention_rate_at_065']:.1%}")
print("=" * 70)
