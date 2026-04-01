"""
CaseAI Routing Optimization Model Training
Trains a LinUCB contextual bandit for case-to-caseworker assignment.

Usage:
    python train_routing_model.py
    python train_routing_model.py --alpha 0.5 --epochs 200
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import TrainingConfig
from data_extractor import CaseAIDataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_routing")


class LinUCBAgent:
    """LinUCB with disjoint linear models per arm (caseworker)."""

    def __init__(self, n_features: int, n_arms: int, alpha: float = 0.25):
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = alpha

        # Per-arm parameters
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        self.theta = [np.zeros(n_features) for _ in range(n_arms)]

        # Tracking
        self.total_reward = 0.0
        self.n_pulls = np.zeros(n_arms)
        self.history = []

    def select_arm(self, context: np.ndarray) -> tuple[int, float]:
        """Select arm using UCB criterion. Returns (arm_index, ucb_score)."""
        ucb_scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            self.theta[a] = A_inv @ self.b[a]
            ucb = self.theta[a] @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_scores[a] = ucb

        best_arm = int(np.argmax(ucb_scores))
        return best_arm, float(ucb_scores[best_arm])

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update arm parameters with observed reward."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.total_reward += reward
        self.n_pulls[arm] += 1

    def get_top_k(self, context: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
        """Return top-k arms with scores for recommendation."""
        scores = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            scores.append((a, float(ucb)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class RoutingModelTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.model_dir = Path(cfg.model_output_dir) / "routing_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        extractor = CaseAIDataExtractor(self.cfg)
        df = extractor.extract_routing_training_data()

        if len(df) < 20:
            logger.error("Insufficient routing data (%d rows). Run synthetic_data_generator.py first.", len(df))
            return

        # Encode caseworkers as arm indices
        cw_encoder = LabelEncoder()
        df["arm_index"] = cw_encoder.fit_transform(df["caseworker_id"].fillna("UNKNOWN"))
        n_arms = len(cw_encoder.classes_)
        logger.info("Number of caseworkers (arms): %d", n_arms)

        # Prepare context features
        feature_cols = ["prior_case_count", "contact_count", "total_assignments", "recent_assignments"]

        # Add encoded categoricals
        for col in ["priority_code", "jurisdiction_code", "case_status"]:
            if col in df.columns:
                dummies = pd.get_dummies(df[col].fillna("UNKNOWN"), prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())

        # Fill NAs and scale
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df[feature_cols] = df[feature_cols].fillna(0)

        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols].values)
        n_features = X.shape[1]

        arms = df["arm_index"].values
        rewards = df["reward"].fillna(0).values

        logger.info("Context dimension: %d, Training samples: %d", n_features, len(df))
        logger.info("Reward statistics: mean=%.3f, std=%.3f", rewards.mean(), rewards.std())

        # Offline policy evaluation via replay method
        agent = LinUCBAgent(n_features=n_features, n_arms=n_arms, alpha=self.cfg.routing.alpha)

        matched = 0
        cumulative_reward = 0.0
        reward_history = []

        # Shuffle data for replay
        indices = np.random.permutation(len(df))

        for i, idx in enumerate(indices):
            context = X[idx]
            logged_arm = arms[idx]
            logged_reward = rewards[idx]

            # Agent selects arm
            selected_arm, ucb_score = agent.select_arm(context)

            # Replay: only update if agent's choice matches logged action
            if selected_arm == logged_arm:
                agent.update(selected_arm, context, logged_reward)
                cumulative_reward += logged_reward
                matched += 1

            if (i + 1) % 500 == 0:
                match_rate = matched / (i + 1) * 100
                avg_reward = cumulative_reward / max(matched, 1)
                logger.info(
                    "  Step %d/%d - Match rate: %.1f%%, Avg reward: %.3f",
                    i + 1, len(indices), match_rate, avg_reward,
                )
                reward_history.append({
                    "step": i + 1,
                    "match_rate": round(match_rate, 2),
                    "avg_reward": round(avg_reward, 4),
                })

        # Final evaluation
        logger.info("\n=== Routing Model Training Complete ===")
        logger.info("Total matches (replay): %d / %d (%.1f%%)", matched, len(indices), matched / len(indices) * 100)
        logger.info("Average reward (matched): %.4f", cumulative_reward / max(matched, 1))

        # Arm pull distribution
        pull_dist = agent.n_pulls
        logger.info("Arm pull distribution: min=%d, max=%d, mean=%.1f",
                     pull_dist.min(), pull_dist.max(), pull_dist.mean())

        # Save model
        model_data = {
            "A": [a.tolist() for a in agent.A],
            "b": [b.tolist() for b in agent.b],
            "theta": [t.tolist() for t in agent.theta],
            "n_features": n_features,
            "n_arms": n_arms,
            "alpha": self.cfg.routing.alpha,
            "n_pulls": agent.n_pulls.tolist(),
        }
        with open(self.model_dir / "linucb_model.json", "w") as f:
            json.dump(model_data, f)

        with open(self.model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(self.model_dir / "caseworker_encoder.pkl", "wb") as f:
            pickle.dump(cw_encoder, f)
        with open(self.model_dir / "feature_columns.json", "w") as f:
            json.dump(feature_cols, f)

        metrics = {
            "n_arms": n_arms,
            "n_features": n_features,
            "n_training_samples": len(df),
            "replay_match_rate": round(matched / len(indices) * 100, 2),
            "avg_reward": round(cumulative_reward / max(matched, 1), 4),
            "alpha": self.cfg.routing.alpha,
            "reward_history": reward_history,
        }
        with open(self.model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Model saved to %s", self.model_dir)


def main():
    parser = argparse.ArgumentParser(description="Train CaseAI Routing Model")
    parser.add_argument("--alpha", type=float, default=None, help="UCB exploration parameter")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.alpha:
        cfg.routing.alpha = args.alpha

    trainer = RoutingModelTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
