"""
CaseAI Outcome Prediction Model Training
Trains models for three outcome dimensions:
  1. Case escalation risk (gradient boosting classifier)
  2. Re-entry probability (Cox proportional hazards / survival)
  3. Time-to-closure (random survival forest)

Usage:
    python train_outcome_model.py
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from config import TrainingConfig
from data_extractor import CaseAIDataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_outcome")


class OutcomeModelTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.model_dir = Path(cfg.model_output_dir) / "outcome_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        extractor = CaseAIDataExtractor(self.cfg)
        df = extractor.extract_outcome_training_data()

        if len(df) < 50:
            logger.error("Insufficient data (%d rows). Run synthetic_data_generator.py first.", len(df))
            return

        logger.info("Outcome dataset: %d rows", len(df))

        # Feature preparation
        feature_cols = [
            "service_count", "completed_services", "referral_count",
            "completed_referrals", "declined_referrals",
            "contact_count", "contact_type_variety", "duration_days",
        ]

        # Encode categoricals
        for col in ["priority_code", "jurisdiction_code", "eligibility_status"]:
            if col in df.columns:
                dummies = pd.get_dummies(df[col].fillna("UNKNOWN"), prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())

        df[feature_cols] = df[feature_cols].fillna(0)

        # 1. Escalation Risk Model
        self._train_escalation_model(df, feature_cols)

        # 2. Closure Prediction Model
        self._train_closure_model(df, feature_cols)

        # 3. Survival Analysis (Time-to-Event)
        self._train_survival_model(df, feature_cols)

        logger.info("All outcome models saved to %s", self.model_dir)

    def _train_escalation_model(self, df: pd.DataFrame, feature_cols: list):
        """Predict whether a case will escalate to HIGH/CRITICAL priority."""
        logger.info("\n--- Escalation Risk Model ---")

        X = df[feature_cols].values
        y = df["is_escalated"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=self.cfg.outcome.gbm_n_estimators,
                max_depth=5, learning_rate=0.1,
                eval_metric="logloss", use_label_encoder=False, random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=self.cfg.outcome.gbm_n_estimators,
                max_depth=5, learning_rate=0.1, random_state=42,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        logger.info("Escalation Model:")
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No Escalation", "Escalated"]))
        if len(np.unique(y_test)) > 1:
            logger.info("AUC-ROC: %.4f", roc_auc_score(y_test, y_proba))

        with open(self.model_dir / "escalation_model.pkl", "wb") as f:
            pickle.dump(model, f)

    def _train_closure_model(self, df: pd.DataFrame, feature_cols: list):
        """Predict whether a case will close (binary)."""
        logger.info("\n--- Closure Prediction Model ---")

        X = df[feature_cols].values
        y = df["is_closed"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", use_label_encoder=False, random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        logger.info("Closure Model:")
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Open", "Closed"]))
        if len(np.unique(y_test)) > 1:
            logger.info("AUC-ROC: %.4f", roc_auc_score(y_test, y_proba))

        with open(self.model_dir / "closure_model.pkl", "wb") as f:
            pickle.dump(model, f)

    def _train_survival_model(self, df: pd.DataFrame, feature_cols: list):
        """Train survival model for time-to-closure estimation."""
        logger.info("\n--- Survival Analysis (Time-to-Closure) ---")

        try:
            from lifelines import CoxPHFitter

            surv_df = df[feature_cols + ["duration_days", "is_closed"]].copy()
            surv_df = surv_df.replace([np.inf, -np.inf], np.nan).dropna()
            surv_df["duration_days"] = surv_df["duration_days"].clip(lower=1)

            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(surv_df, duration_col="duration_days", event_col="is_closed")

            logger.info("Cox PH Model Summary:")
            logger.info("\n%s", cph.summary[["coef", "exp(coef)", "p"]].to_string())

            # Concordance
            logger.info("Concordance index: %.4f", cph.concordance_index_)

            cph.summary.to_csv(self.model_dir / "cox_summary.csv")
            with open(self.model_dir / "cox_model.pkl", "wb") as f:
                pickle.dump(cph, f)

        except ImportError:
            logger.info("lifelines not installed. Training regression approximation for time-to-closure...")

            from sklearn.ensemble import GradientBoostingRegressor

            valid = df[df["is_closed"] == 1].copy()
            if len(valid) < 20:
                logger.warning("Too few closed cases for time-to-closure model")
                return

            X = valid[feature_cols].values
            y = valid["duration_days"].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
            )
            model.fit(X_train, y_train)

            from sklearn.metrics import mean_absolute_error, r2_score
            y_pred = model.predict(X_test)
            logger.info("Time-to-Closure Regression:")
            logger.info("  MAE: %.1f days", mean_absolute_error(y_test, y_pred))
            logger.info("  R2: %.3f", r2_score(y_test, y_pred))

            with open(self.model_dir / "time_to_closure_model.pkl", "wb") as f:
                pickle.dump(model, f)

        # Save feature columns
        with open(self.model_dir / "feature_columns.json", "w") as f:
            json.dump(feature_cols, f)


def main():
    cfg = TrainingConfig()
    trainer = OutcomeModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
