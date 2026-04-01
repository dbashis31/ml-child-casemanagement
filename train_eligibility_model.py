"""
CaseAI Eligibility Prediction Model Training
Trains a gradient boosting classifier with SHAP explainability
for the Eligibility Decision Agent.

Usage:
    python train_eligibility_model.py
    python train_eligibility_model.py --threshold 0.75
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import TrainingConfig
from data_extractor import CaseAIDataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_eligibility")


class EligibilityModelTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.model_dir = Path(cfg.model_output_dir) / "eligibility_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        extractor = CaseAIDataExtractor(self.cfg)
        df = extractor.extract_eligibility_training_data()

        # Filter valid rows
        df = df.dropna(subset=["eligible_label"])
        if len(df) < 50:
            logger.error("Insufficient data (%d rows). Run synthetic_data_generator.py first.", len(df))
            return

        logger.info("Eligibility dataset: %d rows", len(df))
        logger.info("Label distribution:\n%s", df["eligible_label"].value_counts().to_string())

        # Features
        feature_cols = []

        # Numeric features
        for col in ["child_age", "income", "duration_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                feature_cols.append(col)

        # Gender encoding
        if "gender_code" in df.columns:
            dummies = pd.get_dummies(df["gender_code"].fillna("UNKNOWN"), prefix="gender", drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            feature_cols.extend(dummies.columns.tolist())

        X = df[feature_cols].values
        y = df["eligible_label"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        # Train model
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=self.cfg.eligibility.n_estimators,
                max_depth=self.cfg.eligibility.max_depth,
                learning_rate=self.cfg.eligibility.learning_rate,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
            )
        except ImportError:
            logger.warning("xgboost not installed, using sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=self.cfg.eligibility.n_estimators,
                max_depth=self.cfg.eligibility.max_depth,
                learning_rate=self.cfg.eligibility.learning_rate,
                random_state=42,
            )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        logger.info("Eligibility Model Results:")
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Not Eligible", "Eligible"]))
        logger.info("AUC-ROC: %.4f", roc_auc_score(y_test, y_proba))

        # Abstention analysis
        max_conf = np.maximum(y_proba, 1 - y_proba)
        threshold = self.cfg.eligibility.abstention_threshold
        abstain_mask = max_conf < threshold

        if (~abstain_mask).sum() > 0:
            confident_acc = accuracy_score(y_test[~abstain_mask], y_pred[~abstain_mask])
        else:
            confident_acc = 0.0

        logger.info("\nAbstention Analysis (threshold=%.2f):", threshold)
        logger.info("  Abstention rate: %.1f%%", abstain_mask.mean() * 100)
        logger.info("  Accuracy on confident: %.3f", confident_acc)

        # SHAP explanations
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test[:100])

            # Mean absolute SHAP importance
            if isinstance(shap_values, list):
                shap_importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)

            feature_importance = dict(zip(feature_cols, shap_importance.tolist()))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("\nSHAP Feature Importance:")
            for feat, imp in sorted_importance[:10]:
                logger.info("  %s: %.4f", feat, imp)

            with open(self.model_dir / "shap_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=2)
        except ImportError:
            logger.info("shap not installed, skipping SHAP explanations")
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
                logger.info("Gini Feature Importance: %s",
                            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

        # Save
        with open(self.model_dir / "eligibility_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(self.model_dir / "feature_columns.json", "w") as f:
            json.dump(feature_cols, f)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "abstention_rate": float(abstain_mask.mean()),
            "confident_accuracy": float(confident_acc),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        with open(self.model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Model saved to %s", self.model_dir)


def main():
    parser = argparse.ArgumentParser(description="Train CaseAI Eligibility Model")
    parser.add_argument("--threshold", type=float, default=None, help="Abstention threshold")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.threshold:
        cfg.eligibility.abstention_threshold = args.threshold

    trainer = EligibilityModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
