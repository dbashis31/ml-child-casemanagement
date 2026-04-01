"""
CaseAI Data Extractor
Extracts and joins data across microservice databases to produce ML-ready feature sets.
Each extractor produces a pandas DataFrame suitable for model training.

Usage:
    from data_extractor import CaseAIDataExtractor
    extractor = CaseAIDataExtractor(cfg)
    risk_df = extractor.extract_risk_training_data()
    routing_df = extractor.extract_routing_training_data()
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

from config import TrainingConfig

logger = logging.getLogger("data_extractor")


class CaseAIDataExtractor:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    def _conn(self, dbname: str):
        return psycopg2.connect(
            host=self.cfg.db.host, port=self.cfg.db.port,
            user=self.cfg.db.user, password=self.cfg.db.password,
            dbname=dbname,
        )

    def _query(self, dbname: str, query: str) -> pd.DataFrame:
        conn = self._conn(dbname)
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    # ---------------------------------------------------------------
    # 1. Risk Scoring Training Data
    # ---------------------------------------------------------------
    def extract_risk_training_data(self) -> pd.DataFrame:
        """
        Joins intake, case, person, and contact data to produce features for
        the Intake Risk Agent. Outcome label derived from case priority.
        """
        logger.info("Extracting risk training data...")

        # Intake + Case (outcome label = priority)
        cases_df = self._query(self.cfg.db.case_db, """
            SELECT
                i.intake_id, i.child_person_id, i.status_code AS intake_status,
                i.eligibility_status, i.created_at AS intake_created_at,
                cr.case_id, cr.priority_code, cr.status_code AS case_status,
                cr.jurisdiction_code, cr.assigned_at
            FROM intake i
            LEFT JOIN case_record cr ON i.intake_id = cr.intake_id
            WHERE i.status_code IN ('SUBMITTED', 'CLOSED')
        """)

        # Person demographics
        persons_df = self._query(self.cfg.db.person_db, """
            SELECT
                p.person_id, p.date_of_birth, p.gender_code,
                a.city, a.state_province, a.postal_code
            FROM person p
            LEFT JOIN person_address pa ON p.person_id = pa.person_id AND pa.is_current = true
            LEFT JOIN address a ON pa.address_id = a.address_id
        """)

        # Case contact notes (NLP features)
        notes_df = self._query(self.cfg.db.case_db, """
            SELECT case_id, STRING_AGG(notes, ' ') AS combined_notes,
                   COUNT(*) AS contact_count
            FROM case_contact
            WHERE notes IS NOT NULL
            GROUP BY case_id
        """)

        # Person notes
        person_notes_df = self._query(self.cfg.db.person_db, """
            SELECT person_id, STRING_AGG(note_text, ' ') AS person_notes,
                   COUNT(*) AS note_count
            FROM person_note
            WHERE note_text IS NOT NULL
            GROUP BY person_id
        """)

        # Prior case history (count of previous cases per child)
        history_df = self._query(self.cfg.db.case_db, """
            SELECT cp.person_id AS child_person_id,
                   COUNT(DISTINCT cp.case_id) AS prior_case_count
            FROM case_participant cp
            WHERE cp.role_code = 'CHILD'
            GROUP BY cp.person_id
        """)

        # Join everything
        df = cases_df.merge(persons_df, left_on="child_person_id", right_on="person_id", how="left")
        df = df.merge(notes_df, on="case_id", how="left")
        df = df.merge(person_notes_df, left_on="child_person_id", right_on="person_id", how="left", suffixes=("", "_person"))
        df = df.merge(history_df, on="child_person_id", how="left")

        # Derived features
        df["child_age_days"] = (pd.Timestamp.now() - pd.to_datetime(df["date_of_birth"])).dt.days
        df["prior_case_count"] = df["prior_case_count"].fillna(0).astype(int)
        df["contact_count"] = df["contact_count"].fillna(0).astype(int)
        df["note_count"] = df["note_count"].fillna(0).astype(int)

        # Risk label from priority (training target)
        priority_map = {"CRITICAL": "HIGH", "HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
        df["risk_label"] = df["priority_code"].map(priority_map).fillna("LOW")

        # Combine text features
        df["all_text"] = (
            df["combined_notes"].fillna("") + " " + df["person_notes"].fillna("")
        ).str.strip()

        logger.info("  Risk dataset: %d rows, %d features", len(df), len(df.columns))
        return df

    # ---------------------------------------------------------------
    # 2. Routing Training Data
    # ---------------------------------------------------------------
    def extract_routing_training_data(self) -> pd.DataFrame:
        """
        Extracts case assignment history with outcomes for the Routing Optimization Agent.
        Context = case features + caseworker state. Reward = outcome quality.
        """
        logger.info("Extracting routing training data...")

        assignments_df = self._query(self.cfg.db.routing_db, """
            SELECT
                ca.assignment_id, ca.case_id, ca.assigned_to_id AS caseworker_id,
                ca.team_id, ca.routing_strategy, ca.assigned_reason,
                ca.model_score, ca.assigned_at
            FROM case_assignment ca
        """)

        # Case details for context
        cases_df = self._query(self.cfg.db.case_db, """
            SELECT cr.case_id, cr.priority_code, cr.jurisdiction_code,
                   cr.status_code AS case_status,
                   cr.created_at AS case_created_at,
                   i.child_person_id
            FROM case_record cr
            JOIN intake i ON cr.intake_id = i.intake_id
        """)

        # Caseworker load at assignment time (approximated)
        load_df = self._query(self.cfg.db.routing_db, """
            SELECT assigned_to_id AS caseworker_id,
                   COUNT(*) AS total_assignments,
                   COUNT(CASE WHEN assigned_at > NOW() - INTERVAL '30 days' THEN 1 END) AS recent_assignments
            FROM case_assignment
            GROUP BY assigned_to_id
        """)

        # Outcome proxy: case closure status and time-to-close
        outcome_df = self._query(self.cfg.db.case_db, """
            SELECT cr.case_id,
                   cr.status_code AS final_status,
                   EXTRACT(EPOCH FROM (cr.created_at - cr.assigned_at)) / 86400.0 AS days_to_assign
            FROM case_record cr
            WHERE cr.assigned_at IS NOT NULL
        """)

        df = assignments_df.merge(cases_df, on="case_id", how="left")
        df = df.merge(load_df, on="caseworker_id", how="left")
        df = df.merge(outcome_df, on="case_id", how="left")

        # Reward signal: composite of closure + speed
        df["total_assignments"] = df["total_assignments"].fillna(0)
        df["recent_assignments"] = df["recent_assignments"].fillna(0)
        df["is_closed"] = (df["final_status"] == "CLOSED").astype(int)
        df["reward"] = df["is_closed"] * 0.7 + (1 - df["recent_assignments"].clip(0, 20) / 20) * 0.3

        logger.info("  Routing dataset: %d rows", len(df))
        return df

    # ---------------------------------------------------------------
    # 3. Eligibility Training Data
    # ---------------------------------------------------------------
    def extract_eligibility_training_data(self) -> pd.DataFrame:
        """
        Extracts eligibility execution history with decision outcomes
        for the Eligibility Decision Agent.
        """
        logger.info("Extracting eligibility training data...")

        elig_df = self._query(self.cfg.db.eligibility_db, """
            SELECT
                ee.execution_id, ee.subject_person_id, ee.intake_id,
                ee.facts_json, ee.decision_json, ee.status_code,
                ee.duration_ms, ee.started_at
            FROM eligibility_execution ee
            WHERE ee.status_code = 'SUCCESS'
        """)

        # Parse JSON fields
        def parse_facts(facts_str):
            if facts_str and isinstance(facts_str, str):
                try:
                    return json.loads(facts_str)
                except json.JSONDecodeError:
                    return {}
            elif isinstance(facts_str, dict):
                return facts_str
            return {}

        def parse_decision(dec_str):
            if dec_str and isinstance(dec_str, str):
                try:
                    return json.loads(dec_str)
                except json.JSONDecodeError:
                    return {}
            elif isinstance(dec_str, dict):
                return dec_str
            return {}

        elig_df["facts"] = elig_df["facts_json"].apply(parse_facts)
        elig_df["decision"] = elig_df["decision_json"].apply(parse_decision)

        # Extract features from facts
        elig_df["child_age"] = elig_df["facts"].apply(lambda f: f.get("child_age", None))
        elig_df["income"] = elig_df["facts"].apply(lambda f: f.get("income", None))

        # Extract label from decision
        elig_df["eligible"] = elig_df["decision"].apply(lambda d: d.get("eligible", None))
        elig_df["confidence"] = elig_df["decision"].apply(lambda d: d.get("confidence", None))

        # Person demographics
        persons_df = self._query(self.cfg.db.person_db, """
            SELECT person_id, date_of_birth, gender_code
            FROM person
        """)

        df = elig_df.merge(
            persons_df, left_on="subject_person_id", right_on="person_id", how="left"
        )

        df["eligible_label"] = df["eligible"].astype(float)

        logger.info("  Eligibility dataset: %d rows", len(df))
        return df

    # ---------------------------------------------------------------
    # 4. Entity Resolution Training Data
    # ---------------------------------------------------------------
    def extract_entity_resolution_data(self) -> pd.DataFrame:
        """
        Extracts person records for duplicate detection training.
        Returns all persons with their identifying fields.
        """
        logger.info("Extracting entity resolution data...")

        df = self._query(self.cfg.db.person_db, """
            SELECT
                p.person_id, p.first_name, p.middle_name, p.last_name,
                p.date_of_birth, p.gender_code,
                a.line1 AS address_line1, a.city, a.state_province,
                a.postal_code
            FROM person p
            LEFT JOIN person_address pa ON p.person_id = pa.person_id AND pa.is_current = true
            LEFT JOIN address a ON pa.address_id = a.address_id
        """)

        logger.info("  Entity resolution dataset: %d persons", len(df))
        return df

    # ---------------------------------------------------------------
    # 5. Outcome Prediction Training Data
    # ---------------------------------------------------------------
    def extract_outcome_training_data(self) -> pd.DataFrame:
        """
        Extracts case trajectories for outcome prediction models
        (escalation, re-entry, time-to-closure).
        """
        logger.info("Extracting outcome prediction data...")

        cases_df = self._query(self.cfg.db.case_db, """
            SELECT
                cr.case_id, cr.priority_code, cr.status_code, cr.jurisdiction_code,
                cr.assigned_at, cr.created_at,
                i.child_person_id, i.eligibility_status
            FROM case_record cr
            JOIN intake i ON cr.intake_id = i.intake_id
        """)

        # Services received
        services_df = self._query(self.cfg.db.case_db, """
            SELECT case_id, COUNT(*) AS service_count,
                   COUNT(CASE WHEN status_code = 'COMPLETED' THEN 1 END) AS completed_services
            FROM case_provider_service
            GROUP BY case_id
        """)

        # Referral outcomes
        referrals_df = self._query(self.cfg.db.provider_db, """
            SELECT case_id,
                   COUNT(*) AS referral_count,
                   COUNT(CASE WHEN status_code = 'COMPLETED' THEN 1 END) AS completed_referrals,
                   COUNT(CASE WHEN status_code = 'DECLINED' THEN 1 END) AS declined_referrals
            FROM referral
            GROUP BY case_id
        """)

        # Contact intensity
        contacts_df = self._query(self.cfg.db.case_db, """
            SELECT case_id, COUNT(*) AS contact_count,
                   COUNT(DISTINCT contact_type) AS contact_type_variety
            FROM case_contact
            GROUP BY case_id
        """)

        df = cases_df.merge(services_df, on="case_id", how="left")
        df = df.merge(referrals_df, on="case_id", how="left")
        df = df.merge(contacts_df, on="case_id", how="left")

        # Fill NAs
        for col in ["service_count", "completed_services", "referral_count",
                     "completed_referrals", "declined_referrals",
                     "contact_count", "contact_type_variety"]:
            df[col] = df[col].fillna(0).astype(int)

        # Outcome labels
        df["is_closed"] = (df["status_code"] == "CLOSED").astype(int)
        df["is_escalated"] = (df["priority_code"].isin(["CRITICAL", "HIGH"])).astype(int)

        # Duration (days since creation, censored if not closed)
        df["duration_days"] = (
            pd.Timestamp.now() - pd.to_datetime(df["created_at"])
        ).dt.days

        logger.info("  Outcome dataset: %d rows, %d features", len(df), len(df.columns))
        return df

    # ---------------------------------------------------------------
    # 6. Bias Monitoring Data
    # ---------------------------------------------------------------
    def extract_bias_monitoring_data(self) -> pd.DataFrame:
        """
        Extracts demographic + decision data for fairness analysis.
        Joins person demographics with risk scores and eligibility decisions.
        """
        logger.info("Extracting bias monitoring data...")

        # Demographics
        demo_df = self._query(self.cfg.db.person_db, """
            SELECT p.person_id, p.gender_code, p.date_of_birth,
                   a.city, a.state_province, a.postal_code
            FROM person p
            LEFT JOIN person_address pa ON p.person_id = pa.person_id AND pa.is_current = true
            LEFT JOIN address a ON pa.address_id = a.address_id
        """)

        # Case decisions
        decisions_df = self._query(self.cfg.db.case_db, """
            SELECT i.child_person_id AS person_id,
                   cr.priority_code AS risk_decision,
                   i.eligibility_status AS elig_decision
            FROM intake i
            LEFT JOIN case_record cr ON i.intake_id = cr.intake_id
            WHERE i.status_code IN ('SUBMITTED', 'CLOSED')
        """)

        df = demo_df.merge(decisions_df, on="person_id", how="inner")

        # Derive geographic group
        df["geo_group"] = df["city"].fillna("UNKNOWN")

        # Age group
        df["age_days"] = (pd.Timestamp.now() - pd.to_datetime(df["date_of_birth"])).dt.days
        df["age_group"] = pd.cut(
            df["age_days"] / 365.25,
            bins=[0, 2, 5, 12, 18, 200],
            labels=["infant", "toddler", "child", "adolescent", "adult"],
        )

        logger.info("  Bias monitoring dataset: %d rows", len(df))
        return df
