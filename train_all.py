"""
CaseAI Master Training Pipeline
Orchestrates the full ML training pipeline: DB check -> synthetic data -> model training -> bias audit.

Usage:
    python train_all.py                        # Full pipeline
    python train_all.py --skip-synthetic       # Skip data generation
    python train_all.py --models risk,routing  # Train specific models only
    python train_all.py --db-check-only        # Only run DB checks
"""
import argparse
import logging
import sys
import time

from config import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_all")


def run_db_check(cfg: TrainingConfig) -> bool:
    logger.info("=" * 60)
    logger.info("STEP 1: Database Readiness Check")
    logger.info("=" * 60)
    from db_check import run_all_checks
    return run_all_checks(cfg, verbose=True)


def run_synthetic_generation(cfg: TrainingConfig):
    logger.info("=" * 60)
    logger.info("STEP 2: Synthetic Data Generation")
    logger.info("=" * 60)
    from synthetic_data_generator import SyntheticDataGenerator
    generator = SyntheticDataGenerator(cfg)
    generator.generate_all()


def run_model_training(cfg: TrainingConfig, models: list[str]):
    logger.info("=" * 60)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 60)

    results = {}

    if "risk" in models:
        logger.info("\n--- Training: Intake Risk Model ---")
        start = time.time()
        from train_risk_model import IntakeRiskTrainer
        trainer = IntakeRiskTrainer(cfg)
        trainer.train(skip_nlp=True)  # Default to fast mode (TF-IDF fallback)
        results["risk"] = {"status": "complete", "duration_s": round(time.time() - start, 1)}

    if "routing" in models:
        logger.info("\n--- Training: Routing Optimization Model ---")
        start = time.time()
        from train_routing_model import RoutingModelTrainer
        trainer = RoutingModelTrainer(cfg)
        trainer.train()
        results["routing"] = {"status": "complete", "duration_s": round(time.time() - start, 1)}

    if "eligibility" in models:
        logger.info("\n--- Training: Eligibility Prediction Model ---")
        start = time.time()
        from train_eligibility_model import EligibilityModelTrainer
        trainer = EligibilityModelTrainer(cfg)
        trainer.train()
        results["eligibility"] = {"status": "complete", "duration_s": round(time.time() - start, 1)}

    if "entity" in models:
        logger.info("\n--- Training: Entity Resolution Model ---")
        start = time.time()
        from train_entity_resolution import EntityResolutionTrainer
        trainer = EntityResolutionTrainer(cfg)
        trainer.train()
        results["entity"] = {"status": "complete", "duration_s": round(time.time() - start, 1)}

    if "outcome" in models:
        logger.info("\n--- Training: Outcome Prediction Models ---")
        start = time.time()
        from train_outcome_model import OutcomeModelTrainer
        trainer = OutcomeModelTrainer(cfg)
        trainer.train()
        results["outcome"] = {"status": "complete", "duration_s": round(time.time() - start, 1)}

    return results


def run_bias_audit(cfg: TrainingConfig):
    logger.info("=" * 60)
    logger.info("STEP 4: Bias Monitoring Audit")
    logger.info("=" * 60)
    from bias_monitor import BiasMonitor
    monitor = BiasMonitor(cfg)
    return monitor.run_audit()


def main():
    parser = argparse.ArgumentParser(description="CaseAI Master Training Pipeline")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--skip-bias", action="store_true", help="Skip bias audit")
    parser.add_argument("--db-check-only", action="store_true", help="Only run DB checks")
    parser.add_argument(
        "--models", type=str, default="risk,routing,eligibility,entity,outcome",
        help="Comma-separated list of models to train",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()

    logger.info("CaseAI Training Pipeline")
    logger.info("Model output: %s", cfg.model_output_dir)
    pipeline_start = time.time()

    # Step 1: DB Check
    db_ready = run_db_check(cfg)

    if args.db_check_only:
        sys.exit(0 if db_ready else 1)

    # Step 2: Synthetic Data (if DB is empty or not ready)
    if not args.skip_synthetic:
        run_synthetic_generation(cfg)

    # Step 3: Model Training
    models = [m.strip() for m in args.models.split(",")]
    training_results = run_model_training(cfg, models)

    # Step 4: Bias Audit
    if not args.skip_bias:
        run_bias_audit(cfg)

    # Summary
    total_time = round(time.time() - pipeline_start, 1)
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE (%.1f seconds)", total_time)
    logger.info("=" * 60)
    for model, result in training_results.items():
        logger.info("  %s: %s (%.1fs)", model, result["status"], result["duration_s"])
    logger.info("Models saved to: %s", cfg.model_output_dir)


if __name__ == "__main__":
    main()
