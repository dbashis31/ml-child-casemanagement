"""
CaseAI Intake Risk Scoring Model Training
Trains a multi-modal risk scoring model combining:
  1. ClinicalBERT for NLP-based text analysis of intake notes
  2. XGBoost/LightGBM for tabular feature processing
  3. Calibrated meta-learner for score fusion with abstention

Usage:
    python train_risk_model.py                          # Train with defaults
    python train_risk_model.py --skip-nlp               # Tabular only (faster)
    python train_risk_model.py --epochs 10 --threshold 0.7
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import TrainingConfig
from data_extractor import CaseAIDataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_risk")


class IntakeRiskTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.model_dir = Path(cfg.model_output_dir) / "risk_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = LabelEncoder()

    def train(self, skip_nlp: bool = False):
        # Extract data
        extractor = CaseAIDataExtractor(self.cfg)
        df = extractor.extract_risk_training_data()

        if len(df) < 50:
            logger.error("Insufficient data for training (%d rows). Run synthetic_data_generator.py first.", len(df))
            return

        # Encode labels
        df["risk_label_encoded"] = self.label_encoder.fit_transform(df["risk_label"])
        logger.info("Label distribution:\n%s", df["risk_label"].value_counts().to_string())

        # Split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["risk_label"], random_state=42)
        logger.info("Train: %d, Test: %d", len(train_df), len(test_df))

        # ---- Tabular Model ----
        tabular_features = self._train_tabular(train_df, test_df)

        # ---- NLP Model ----
        nlp_features = None
        if not skip_nlp:
            nlp_features = self._train_nlp(train_df, test_df)

        # ---- Meta-Learner (Fusion) ----
        self._train_meta_learner(train_df, test_df, tabular_features, nlp_features)

        # Save label encoder
        with open(self.model_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        logger.info("All models saved to %s", self.model_dir)

    def _get_tabular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare tabular features from structured data."""
        features = pd.DataFrame()
        features["child_age_days"] = pd.to_numeric(df["child_age_days"], errors="coerce").fillna(0)
        features["prior_case_count"] = df["prior_case_count"].fillna(0).astype(int)
        features["contact_count"] = df["contact_count"].fillna(0).astype(int)
        features["note_count"] = df["note_count"].fillna(0).astype(int)

        # Encode categoricals
        for col in ["gender_code", "jurisdiction_code", "intake_status", "eligibility_status"]:
            if col in df.columns:
                features[col] = df[col].fillna("UNKNOWN").astype(str)
                dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
                features = pd.concat([features.drop(columns=[col]), dummies], axis=1)

        # Text length as feature
        features["text_length"] = df["all_text"].fillna("").str.len()

        return features

    def _train_tabular(self, train_df, test_df) -> dict:
        """Train XGBoost model on tabular features."""
        logger.info("Training tabular model (XGBoost)...")
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed, falling back to sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingClassifier
            xgb = None

        X_train = self._get_tabular_features(train_df)
        X_test = self._get_tabular_features(test_df)
        y_train = train_df["risk_label_encoded"].values
        y_test = test_df["risk_label_encoded"].values

        # Align columns
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

        if xgb:
            model = xgb.XGBClassifier(
                n_estimators=self.cfg.risk.tabular_n_estimators,
                max_depth=self.cfg.risk.tabular_max_depth,
                learning_rate=self.cfg.risk.tabular_learning_rate,
                objective="multi:softprob",
                num_class=len(self.label_encoder.classes_),
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=self.cfg.risk.tabular_n_estimators,
                max_depth=self.cfg.risk.tabular_max_depth,
                learning_rate=self.cfg.risk.tabular_learning_rate,
                random_state=42,
            )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        logger.info("Tabular Model Results:")
        logger.info("\n%s", classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_
        ))

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(X_train.columns, model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("Top features: %s", top_features)

        # Save
        with open(self.model_dir / "tabular_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(self.model_dir / "tabular_columns.json", "w") as f:
            json.dump(list(X_train.columns), f)

        # Return predictions for meta-learner
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        return {"train": train_proba, "test": test_proba}

    def _train_nlp(self, train_df, test_df) -> dict:
        """Train ClinicalBERT-based text classifier on intake notes."""
        logger.info("Training NLP model (ClinicalBERT)...")
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            logger.warning("transformers/torch not installed. Falling back to TF-IDF + LogReg.")
            return self._train_nlp_fallback(train_df, test_df)

        model_name = self.cfg.risk.nlp_model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.label_encoder.classes_)
        )

        class NoteDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx], truncation=True, padding="max_length",
                    max_length=self.max_len, return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                }

        train_texts = train_df["all_text"].fillna("").tolist()
        test_texts = test_df["all_text"].fillna("").tolist()
        train_labels = train_df["risk_label_encoded"].values
        test_labels = test_df["risk_label_encoded"].values

        train_dataset = NoteDataset(train_texts, train_labels, tokenizer, self.cfg.risk.max_seq_length)
        test_dataset = NoteDataset(test_texts, test_labels, tokenizer, self.cfg.risk.max_seq_length)

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.risk.nlp_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.risk.nlp_batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.risk.nlp_learning_rate)

        # Training loop
        for epoch in range(self.cfg.risk.nlp_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()
            logger.info("  Epoch %d/%d - Loss: %.4f", epoch + 1, self.cfg.risk.nlp_epochs, total_loss / len(train_loader))

        # Evaluate
        model.eval()
        all_proba = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_proba.append(proba)

        test_proba = np.vstack(all_proba)
        y_pred = test_proba.argmax(axis=1)
        logger.info("NLP Model Results:")
        logger.info("\n%s", classification_report(
            test_labels, y_pred, target_names=self.label_encoder.classes_
        ))

        # Save
        model.save_pretrained(str(self.model_dir / "nlp_model"))
        tokenizer.save_pretrained(str(self.model_dir / "nlp_model"))

        # Get train predictions for meta-learner
        train_proba_list = []
        model.eval()
        train_eval_loader = DataLoader(train_dataset, batch_size=self.cfg.risk.nlp_batch_size)
        with torch.no_grad():
            for batch in train_eval_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                train_proba_list.append(proba)

        train_proba = np.vstack(train_proba_list)
        return {"train": train_proba, "test": test_proba}

    def _train_nlp_fallback(self, train_df, test_df) -> dict:
        """Fallback NLP using TF-IDF + Logistic Regression when transformers unavailable."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        logger.info("Training TF-IDF + LogReg fallback NLP model...")

        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X_train = tfidf.fit_transform(train_df["all_text"].fillna(""))
        X_test = tfidf.transform(test_df["all_text"].fillna(""))
        y_train = train_df["risk_label_encoded"].values
        y_test = test_df["risk_label_encoded"].values

        model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        logger.info("TF-IDF NLP Fallback Results:")
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        with open(self.model_dir / "nlp_tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        with open(self.model_dir / "nlp_logreg.pkl", "wb") as f:
            pickle.dump(model, f)

        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        return {"train": train_proba, "test": test_proba}

    def _train_meta_learner(self, train_df, test_df, tabular_feats, nlp_feats):
        """Train calibrated meta-learner that fuses tabular + NLP predictions."""
        logger.info("Training meta-learner with temperature-scaled calibration...")

        y_train = train_df["risk_label_encoded"].values
        y_test = test_df["risk_label_encoded"].values

        # Stack predictions from sub-models
        if nlp_feats is not None:
            X_meta_train = np.hstack([tabular_feats["train"], nlp_feats["train"]])
            X_meta_test = np.hstack([tabular_feats["test"], nlp_feats["test"]])
        else:
            X_meta_train = tabular_feats["train"]
            X_meta_test = tabular_feats["test"]

        # Calibrated logistic regression as meta-learner
        base_meta = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
        meta_model = CalibratedClassifierCV(base_meta, cv=3, method="isotonic")
        meta_model.fit(X_meta_train, y_train)

        y_pred = meta_model.predict(X_meta_test)
        y_proba = meta_model.predict_proba(X_meta_test)

        logger.info("Meta-Learner (Fused) Results:")
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Abstention analysis
        max_confidence = y_proba.max(axis=1)
        threshold = self.cfg.risk.abstention_threshold
        abstain_mask = max_confidence < threshold
        abstention_rate = abstain_mask.mean()

        if (~abstain_mask).sum() > 0:
            confident_acc = (y_pred[~abstain_mask] == y_test[~abstain_mask]).mean()
        else:
            confident_acc = 0.0

        logger.info("Abstention Analysis (threshold=%.2f):", threshold)
        logger.info("  Abstention rate: %.1f%%", abstention_rate * 100)
        logger.info("  Accuracy on confident predictions: %.3f", confident_acc)
        logger.info("  Cases requiring human review: %d / %d", abstain_mask.sum(), len(abstain_mask))

        # Save
        with open(self.model_dir / "meta_learner.pkl", "wb") as f:
            pickle.dump(meta_model, f)

        # Save evaluation metrics
        metrics = {
            "test_accuracy": float((y_pred == y_test).mean()),
            "abstention_rate": float(abstention_rate),
            "confident_accuracy": float(confident_acc),
            "abstention_threshold": threshold,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "label_distribution": dict(zip(
                self.label_encoder.classes_.tolist(),
                np.bincount(y_test, minlength=len(self.label_encoder.classes_)).tolist(),
            )),
        }
        with open(self.model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Metrics saved to %s/metrics.json", self.model_dir)


def main():
    parser = argparse.ArgumentParser(description="Train CaseAI Intake Risk Model")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP model (tabular only)")
    parser.add_argument("--epochs", type=int, default=None, help="Override NLP training epochs")
    parser.add_argument("--threshold", type=float, default=None, help="Override abstention threshold")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.epochs:
        cfg.risk.nlp_epochs = args.epochs
    if args.threshold:
        cfg.risk.abstention_threshold = args.threshold

    trainer = IntakeRiskTrainer(cfg)
    trainer.train(skip_nlp=args.skip_nlp)


if __name__ == "__main__":
    main()
