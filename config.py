"""
CaseAI Training Configuration
Centralized configuration for database connections, model parameters, and training settings.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    # Database names matching the workspace SQL scripts
    case_db: str = "case_service"
    person_db: str = "person_service"
    routing_db: str = "routing_service"
    eligibility_db: str = "eligibility_orchestrator"
    provider_db: str = "provider_service"
    provider_assignment_db: str = "provider_assignment_service"


@dataclass
class RiskModelConfig:
    nlp_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_seq_length: int = 256
    nlp_learning_rate: float = 2e-5
    nlp_epochs: int = 5
    nlp_batch_size: int = 16
    tabular_n_estimators: int = 300
    tabular_max_depth: int = 6
    tabular_learning_rate: float = 0.1
    meta_learner_calibration: str = "temperature_scaling"
    abstention_threshold: float = 0.65
    risk_labels: tuple = ("LOW", "MEDIUM", "HIGH")


@dataclass
class RoutingModelConfig:
    algorithm: str = "LinUCB"
    alpha: float = 0.25  # exploration parameter
    context_dim: int = 32
    reward_window_days: int = 90
    min_exploration_rounds: int = 100
    fallback_strategy: str = "ROUND_ROBIN"


@dataclass
class EligibilityModelConfig:
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.1
    shap_max_display: int = 10
    abstention_threshold: float = 0.70


@dataclass
class EntityResolutionConfig:
    embedding_dim: int = 128
    similarity_threshold: float = 0.85
    blocking_window_size: int = 3  # days for DOB proximity
    siamese_learning_rate: float = 1e-4
    siamese_epochs: int = 20
    siamese_batch_size: int = 64


@dataclass
class OutcomeModelConfig:
    escalation_windows_days: tuple = (30, 60, 90)
    survival_n_estimators: int = 100
    gbm_n_estimators: int = 200


@dataclass
class BiasMonitorConfig:
    disparate_impact_lower: float = 0.8
    disparate_impact_upper: float = 1.25
    fpr_tolerance: float = 0.05
    demographic_axes: tuple = ("race", "income_bracket", "geography")
    monitoring_interval_minutes: int = 60


@dataclass
class SyntheticDataConfig:
    n_persons: int = 5000
    n_intakes: int = 3000
    n_cases: int = 2000
    n_providers: int = 200
    n_caseworkers: int = 100
    seed: int = 42


@dataclass
class TrainingConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    risk: RiskModelConfig = field(default_factory=RiskModelConfig)
    routing: RoutingModelConfig = field(default_factory=RoutingModelConfig)
    eligibility: EligibilityModelConfig = field(default_factory=EligibilityModelConfig)
    entity_resolution: EntityResolutionConfig = field(default_factory=EntityResolutionConfig)
    outcome: OutcomeModelConfig = field(default_factory=OutcomeModelConfig)
    bias: BiasMonitorConfig = field(default_factory=BiasMonitorConfig)
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    model_output_dir: str = os.getenv("MODEL_OUTPUT_DIR", "./models")
    log_level: str = "INFO"
