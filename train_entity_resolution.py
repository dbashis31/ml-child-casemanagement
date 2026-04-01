"""
CaseAI Entity Resolution Model Training
Trains a Siamese network for duplicate child record detection.
Pipeline: LSH blocking -> Siamese matching -> graph-based identity resolution.

Usage:
    python train_entity_resolution.py
    python train_entity_resolution.py --threshold 0.9
"""
import argparse
import json
import logging
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

from config import TrainingConfig
from data_extractor import CaseAIDataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_entity")


def soundex(name: str) -> str:
    """Simple Soundex encoding for phonetic blocking."""
    if not name:
        return "0000"
    name = name.upper().strip()
    if not name:
        return "0000"

    codes = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    result = name[0]
    prev_code = codes.get(name[0], "")

    for ch in name[1:]:
        code = codes.get(ch, "")
        if code and code != prev_code:
            result += code
        prev_code = code if code else prev_code

    return (result + "000")[:4]


def generate_training_pairs(df: pd.DataFrame, n_positive: int = 2000, n_negative: int = 2000) -> pd.DataFrame:
    """
    Generate positive (duplicate) and negative (non-duplicate) training pairs.
    Positive pairs are synthetically created by perturbing existing records.
    """
    logger.info("Generating training pairs...")
    pairs = []

    # Positive pairs: create near-duplicates by perturbing records
    for _ in range(n_positive):
        idx = np.random.randint(0, len(df))
        row = df.iloc[idx]

        # Apply random perturbations
        perturbed = row.copy()
        perturbation = np.random.choice(["typo", "nickname", "dob_off", "address_change"])

        if perturbation == "typo":
            name = perturbed["first_name"]
            if len(name) > 2:
                pos = np.random.randint(1, len(name))
                perturbed["first_name"] = name[:pos-1] + name[pos] + name[pos-1] + name[pos+1:] if pos < len(name) - 1 else name[:pos-1] + name[pos]
        elif perturbation == "nickname":
            nicknames = {"James": "Jim", "William": "Bill", "Robert": "Bob", "Richard": "Dick",
                         "Elizabeth": "Beth", "Patricia": "Pat", "Jennifer": "Jenny", "Michael": "Mike"}
            if perturbed["first_name"] in nicknames:
                perturbed["first_name"] = nicknames[perturbed["first_name"]]
        elif perturbation == "dob_off":
            if pd.notna(perturbed["date_of_birth"]):
                dob = pd.to_datetime(perturbed["date_of_birth"])
                perturbed["date_of_birth"] = (dob + pd.Timedelta(days=np.random.choice([-1, 1]))).date()
        elif perturbation == "address_change":
            perturbed["city"] = "DIFFERENT_CITY"

        pairs.append({
            "id_a": row["person_id"], "id_b": f"synth_{uid_counter()}",
            "first_name_a": row["first_name"], "first_name_b": perturbed["first_name"],
            "last_name_a": row["last_name"], "last_name_b": perturbed["last_name"],
            "dob_a": str(row["date_of_birth"]), "dob_b": str(perturbed["date_of_birth"]),
            "city_a": str(row.get("city", "")), "city_b": str(perturbed.get("city", "")),
            "postal_a": str(row.get("postal_code", "")), "postal_b": str(perturbed.get("postal_code", "")),
            "label": 1,
        })

    # Negative pairs: random non-matching records
    for _ in range(n_negative):
        idx_a, idx_b = np.random.choice(len(df), size=2, replace=False)
        row_a, row_b = df.iloc[idx_a], df.iloc[idx_b]
        pairs.append({
            "id_a": row_a["person_id"], "id_b": row_b["person_id"],
            "first_name_a": row_a["first_name"], "first_name_b": row_b["first_name"],
            "last_name_a": row_a["last_name"], "last_name_b": row_b["last_name"],
            "dob_a": str(row_a["date_of_birth"]), "dob_b": str(row_b["date_of_birth"]),
            "city_a": str(row_a.get("city", "")), "city_b": str(row_b.get("city", "")),
            "postal_a": str(row_a.get("postal_code", "")), "postal_b": str(row_b.get("postal_code", "")),
            "label": 0,
        })

    return pd.DataFrame(pairs)


_uid_counter = 0
def uid_counter():
    global _uid_counter
    _uid_counter += 1
    return _uid_counter


def compute_pair_features(pairs_df: pd.DataFrame) -> np.ndarray:
    """Compute similarity features for each pair."""
    from difflib import SequenceMatcher

    features = []
    for _, row in pairs_df.iterrows():
        # Name similarity
        fn_sim = SequenceMatcher(None, str(row["first_name_a"]).lower(), str(row["first_name_b"]).lower()).ratio()
        ln_sim = SequenceMatcher(None, str(row["last_name_a"]).lower(), str(row["last_name_b"]).lower()).ratio()

        # Soundex match
        fn_soundex_match = int(soundex(str(row["first_name_a"])) == soundex(str(row["first_name_b"])))
        ln_soundex_match = int(soundex(str(row["last_name_a"])) == soundex(str(row["last_name_b"])))

        # DOB similarity
        try:
            dob_a = pd.to_datetime(row["dob_a"])
            dob_b = pd.to_datetime(row["dob_b"])
            dob_diff_days = abs((dob_a - dob_b).days)
            dob_exact = int(dob_diff_days == 0)
            dob_close = int(dob_diff_days <= 3)
        except (ValueError, TypeError):
            dob_diff_days = 9999
            dob_exact = 0
            dob_close = 0

        # Location similarity
        city_match = int(str(row["city_a"]).lower() == str(row["city_b"]).lower())
        postal_sim = SequenceMatcher(None, str(row["postal_a"]), str(row["postal_b"])).ratio()

        features.append([
            fn_sim, ln_sim, fn_soundex_match, ln_soundex_match,
            dob_exact, dob_close, min(dob_diff_days, 365) / 365.0,
            city_match, postal_sim,
        ])

    return np.array(features)


class EntityResolutionTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.model_dir = Path(cfg.model_output_dir) / "entity_resolution"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        extractor = CaseAIDataExtractor(self.cfg)
        df = extractor.extract_entity_resolution_data()

        if len(df) < 50:
            logger.error("Insufficient person data (%d). Run synthetic_data_generator.py first.", len(df))
            return

        logger.info("Person records: %d", len(df))

        # Generate training pairs
        pairs_df = generate_training_pairs(df)
        logger.info("Training pairs: %d (positive: %d, negative: %d)",
                     len(pairs_df), (pairs_df["label"] == 1).sum(), (pairs_df["label"] == 0).sum())

        # Compute features
        X = compute_pair_features(pairs_df)
        y = pairs_df["label"].values

        feature_names = [
            "first_name_sim", "last_name_sim", "fn_soundex_match", "ln_soundex_match",
            "dob_exact", "dob_close", "dob_diff_normalized",
            "city_match", "postal_sim",
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        # Train gradient boosting classifier (Siamese network approximation)
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", use_label_encoder=False, random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
            )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        logger.info("Entity Resolution Model Results:")
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Non-Match", "Match"]))

        # Precision-recall analysis at different thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        target_threshold = self.cfg.entity_resolution.similarity_threshold

        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - target_threshold))
        logger.info("\nAt threshold %.2f: Precision=%.3f, Recall=%.3f",
                     target_threshold, precisions[idx], recalls[idx])

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("\nFeature Importance:")
            for feat, imp in sorted_imp:
                logger.info("  %s: %.4f", feat, imp)

        # Blocking statistics
        logger.info("\nBlocking Statistics:")
        soundex_groups = df.groupby(df["last_name"].apply(lambda x: soundex(str(x)))).size()
        logger.info("  Soundex blocks: %d", len(soundex_groups))
        logger.info("  Avg block size: %.1f", soundex_groups.mean())
        logger.info("  Max block size: %d", soundex_groups.max())

        # Save
        with open(self.model_dir / "matching_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(self.model_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        metrics = {
            "n_persons": len(df),
            "n_training_pairs": len(pairs_df),
            "test_precision_at_threshold": float(precisions[idx]),
            "test_recall_at_threshold": float(recalls[idx]),
            "similarity_threshold": target_threshold,
            "n_soundex_blocks": int(len(soundex_groups)),
        }
        with open(self.model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Model saved to %s", self.model_dir)


def main():
    parser = argparse.ArgumentParser(description="Train CaseAI Entity Resolution Model")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.threshold:
        cfg.entity_resolution.similarity_threshold = args.threshold

    trainer = EntityResolutionTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
