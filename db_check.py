"""
CaseAI Database Readiness Checker
Validates database connectivity, schema presence, table structures, data volume,
and ML-readiness for all microservice databases in the case management platform.

Usage:
    python db_check.py                  # Run all checks
    python db_check.py --db case        # Check specific database
    python db_check.py --verbose        # Detailed output
    python db_check.py --fix            # Attempt to create missing schemas
"""
import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2 import sql

from config import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("db_check")

# ---------------------------------------------------------------------------
# Expected schema definitions (derived from workspace SQL files)
# ---------------------------------------------------------------------------
EXPECTED_SCHEMAS = {
    "case_service": {
        "tables": {
            "intake": [
                "intake_id", "created_by_user_id", "child_person_id",
                "status_code", "validation_status", "eligibility_status",
                "eligibility_reason_code", "draft_payload_json",
                "submitted_at", "created_at", "updated_at",
            ],
            "case_record": [
                "case_id", "case_number", "intake_id", "status_code",
                "priority_code", "jurisdiction_code", "assigned_to_user_id",
                "assigned_at", "created_at", "created_by",
            ],
            "case_participant": [
                "case_participant_id", "case_id", "person_id",
                "role_code", "is_primary", "start_date", "end_date",
                "created_at", "created_by",
            ],
            "case_contact": [
                "case_contact_id", "case_id", "contact_type",
                "contact_datetime", "contacted_person_id", "summary",
                "notes", "created_at", "created_by_user_id",
            ],
            "case_provider_service": [
                "case_provider_service_id", "case_id", "provider_person_id",
                "service_type_code", "status_code", "referral_date",
                "start_date", "end_date", "notes",
                "created_at", "created_by_user_id",
            ],
        },
        "ml_critical_tables": ["intake", "case_record", "case_participant", "case_contact"],
    },
    "person_service": {
        "tables": {
            "person": [
                "person_id", "first_name", "middle_name", "last_name",
                "date_of_birth", "gender_code", "is_active",
                "created_at", "created_by", "updated_at", "updated_by",
            ],
            "person_identifier": [
                "person_identifier_id", "person_id", "identifier_type",
                "identifier_value_enc", "last4", "is_primary",
                "verified_status", "created_at",
            ],
            "person_contact_method": [
                "person_contact_method_id", "person_id", "contact_type",
                "contact_value", "contact_label", "is_primary",
                "is_verified", "created_at",
            ],
            "address": [
                "address_id", "line1", "city", "state_province",
                "postal_code", "country_code", "created_at",
            ],
            "person_address": [
                "person_address_id", "person_id", "address_id",
                "address_type", "is_current", "start_date", "end_date",
                "created_at",
            ],
            "person_note": [
                "person_note_id", "person_id", "note_text",
                "note_type", "is_confidential", "created_at",
            ],
        },
        "ml_critical_tables": ["person", "person_address", "person_note"],
    },
    "routing_service": {
        "tables": {
            "routing_team": [
                "team_id", "team_key", "team_name", "is_active", "created_at",
            ],
            "routing_team_member": [
                "team_member_id", "team_id", "user_id", "is_active", "joined_at",
            ],
            "routing_policy": [
                "policy_id", "team_id", "strategy", "is_active",
                "config_json", "created_at",
            ],
            "routing_round_robin_state": [
                "team_id", "last_assigned_user_id", "last_assigned_at",
            ],
            "routing_model_registry": [
                "model_id", "model_name", "model_version", "is_active",
                "endpoint_url", "metadata_json", "created_at",
            ],
            "case_assignment": [
                "assignment_id", "case_id", "assigned_type",
                "assigned_to_id", "team_id", "routing_strategy",
                "assigned_reason", "model_name", "model_version",
                "model_score", "decision_details_json", "assigned_at",
            ],
        },
        "ml_critical_tables": ["case_assignment", "routing_team_member", "routing_policy"],
    },
    "eligibility_orchestrator": {
        "tables": {
            "brm_provider_registry": [
                "brm_provider_key", "display_name", "endpoint_base_url",
                "auth_type", "is_active",
            ],
            "rule_definition": [
                "rule_id", "rule_key", "brm_provider_key",
                "brm_ruleset_name", "is_active",
            ],
            "rule_lookup_step": [
                "lookup_step_id", "rule_id", "sequence_no", "step_key",
                "service_key", "http_method", "path_template",
            ],
            "eligibility_execution": [
                "execution_id", "rule_key", "subject_person_id",
                "intake_id", "decision_json", "status_code",
                "started_at", "completed_at", "duration_ms",
            ],
            "eligibility_execution_step": [
                "execution_step_id", "execution_id", "sequence_no",
                "step_key", "status_code", "duration_ms",
            ],
        },
        "ml_critical_tables": ["eligibility_execution"],
    },
    "provider_service": {
        "tables": {
            "provider_org": [
                "provider_org_id", "org_name", "org_type_code",
                "status_code", "created_at",
            ],
            "provider_location": [
                "provider_location_id", "provider_org_id", "city",
                "state_province", "postal_code", "latitude", "longitude",
                "is_active",
            ],
            "service_catalog": [
                "service_code", "service_name", "is_active",
            ],
            "provider_org_service": [
                "provider_org_service_id", "provider_org_id",
                "service_code", "status_code",
            ],
            "provider_staff": [
                "provider_staff_id", "provider_org_id", "person_id",
                "staff_role_code", "is_active",
            ],
            "referral": [
                "referral_id", "case_id", "child_person_id",
                "provider_org_id", "service_code", "priority_code",
                "status_code", "created_at",
            ],
            "referral_status_history": [
                "referral_status_history_id", "referral_id",
                "old_status_code", "new_status_code", "changed_at",
            ],
            "provider_capacity": [
                "provider_capacity_id", "provider_org_id", "service_code",
                "capacity_status_code", "max_active_referrals",
                "current_active_referrals",
            ],
        },
        "ml_critical_tables": ["referral", "provider_org", "provider_capacity"],
    },
    "provider_assignment_service": {
        "tables": {
            "assignment_policy": [
                "policy_id", "policy_key", "strategy",
                "jurisdiction_code", "program_code", "service_code",
                "is_active",
            ],
            "assignment_model_registry": [
                "model_id", "model_name", "model_version",
                "is_active", "endpoint_url",
            ],
            "assignment_request": [
                "request_id", "case_id", "child_person_id",
                "service_code", "strategy_used", "status_code",
                "started_at", "duration_ms",
            ],
            "assignment_recommendation": [
                "recommendation_id", "request_id", "rank_no",
                "provider_org_id", "score", "scoring_breakdown_json",
            ],
            "assignment_selection": [
                "selection_id", "request_id",
                "selected_provider_org_id", "selection_method",
                "selected_at",
            ],
            "provider_performance_snapshot": [
                "snapshot_id", "provider_org_id", "service_code",
                "acceptance_rate", "completion_rate",
                "avg_time_to_complete_days",
            ],
        },
        "ml_critical_tables": [
            "assignment_request", "assignment_recommendation",
            "provider_performance_snapshot",
        ],
    },
}

# Minimum row counts for ML training readiness
MIN_ROWS_FOR_TRAINING = {
    "intake": 500,
    "case_record": 500,
    "case_contact": 1000,
    "person": 1000,
    "person_note": 500,
    "case_assignment": 200,
    "eligibility_execution": 200,
    "referral": 200,
    "assignment_request": 100,
}


class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details: list[str] = []

    def ok(self, msg: str):
        self.passed += 1
        self.details.append(f"  [PASS] {msg}")

    def fail(self, msg: str):
        self.failed += 1
        self.details.append(f"  [FAIL] {msg}")

    def warn(self, msg: str):
        self.warnings += 1
        self.details.append(f"  [WARN] {msg}")

    @property
    def status(self) -> str:
        if self.failed > 0:
            return "FAILED"
        if self.warnings > 0:
            return "WARNING"
        return "PASSED"

    def summary(self) -> str:
        return (
            f"[{self.status}] {self.name}: "
            f"{self.passed} passed, {self.failed} failed, {self.warnings} warnings"
        )


def get_connection(cfg: TrainingConfig, dbname: str):
    return psycopg2.connect(
        host=cfg.db.host,
        port=cfg.db.port,
        user=cfg.db.user,
        password=cfg.db.password,
        dbname=dbname,
        connect_timeout=10,
    )


def check_connectivity(cfg: TrainingConfig) -> CheckResult:
    """Test basic connectivity to each database."""
    result = CheckResult("Database Connectivity")
    databases = [
        cfg.db.case_db, cfg.db.person_db, cfg.db.routing_db,
        cfg.db.eligibility_db, cfg.db.provider_db, cfg.db.provider_assignment_db,
    ]
    for dbname in databases:
        try:
            conn = get_connection(cfg, dbname)
            conn.close()
            result.ok(f"Connected to '{dbname}'")
        except psycopg2.OperationalError as e:
            result.fail(f"Cannot connect to '{dbname}': {e}")
        except Exception as e:
            result.fail(f"Unexpected error for '{dbname}': {e}")
    return result


def check_tables(cfg: TrainingConfig, db_filter: Optional[str] = None) -> CheckResult:
    """Verify all expected tables exist in each database."""
    result = CheckResult("Table Existence")

    for dbname, schema_def in EXPECTED_SCHEMAS.items():
        if db_filter and db_filter not in dbname:
            continue
        try:
            conn = get_connection(cfg, dbname)
            cur = conn.cursor()
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            existing = {row[0] for row in cur.fetchall()}
            for table in schema_def["tables"]:
                if table in existing:
                    result.ok(f"{dbname}.{table} exists")
                else:
                    result.fail(f"{dbname}.{table} MISSING")
            cur.close()
            conn.close()
        except psycopg2.OperationalError:
            result.fail(f"Cannot connect to '{dbname}' to check tables")
    return result


def check_columns(cfg: TrainingConfig, db_filter: Optional[str] = None) -> CheckResult:
    """Verify expected columns exist in each table."""
    result = CheckResult("Column Validation")

    for dbname, schema_def in EXPECTED_SCHEMAS.items():
        if db_filter and db_filter not in dbname:
            continue
        try:
            conn = get_connection(cfg, dbname)
            cur = conn.cursor()
            for table, expected_cols in schema_def["tables"].items():
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """, (table,))
                existing_cols = {row[0] for row in cur.fetchall()}
                if not existing_cols:
                    result.fail(f"{dbname}.{table}: table not found or has no columns")
                    continue

                missing = [c for c in expected_cols if c not in existing_cols]
                if missing:
                    result.fail(f"{dbname}.{table} missing columns: {missing}")
                else:
                    result.ok(f"{dbname}.{table}: all {len(expected_cols)} expected columns present")
            cur.close()
            conn.close()
        except psycopg2.OperationalError:
            result.fail(f"Cannot connect to '{dbname}' for column check")
    return result


def check_row_counts(cfg: TrainingConfig, db_filter: Optional[str] = None) -> CheckResult:
    """Check row counts for ML-critical tables and assess training readiness."""
    result = CheckResult("Data Volume (ML Readiness)")

    db_table_map = {
        "case_service": ["intake", "case_record", "case_contact"],
        "person_service": ["person", "person_note"],
        "routing_service": ["case_assignment"],
        "eligibility_orchestrator": ["eligibility_execution"],
        "provider_service": ["referral"],
        "provider_assignment_service": ["assignment_request"],
    }

    for dbname, tables in db_table_map.items():
        if db_filter and db_filter not in dbname:
            continue
        try:
            conn = get_connection(cfg, dbname)
            cur = conn.cursor()
            for table in tables:
                try:
                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table)
                    ))
                    count = cur.fetchone()[0]
                    min_req = MIN_ROWS_FOR_TRAINING.get(table, 100)
                    if count >= min_req:
                        result.ok(f"{dbname}.{table}: {count:,} rows (min: {min_req})")
                    elif count > 0:
                        result.warn(
                            f"{dbname}.{table}: {count:,} rows "
                            f"(below min {min_req} for training)"
                        )
                    else:
                        result.warn(
                            f"{dbname}.{table}: EMPTY "
                            f"(need {min_req}+ rows; use synthetic data generator)"
                        )
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()
                    result.fail(f"{dbname}.{table}: table does not exist")
            cur.close()
            conn.close()
        except psycopg2.OperationalError:
            result.fail(f"Cannot connect to '{dbname}' for row count check")
    return result


def check_data_quality(cfg: TrainingConfig, db_filter: Optional[str] = None) -> CheckResult:
    """Check data quality: nulls in critical columns, date ranges, status distributions."""
    result = CheckResult("Data Quality")

    quality_checks = [
        {
            "db": "case_service",
            "desc": "Intake status distribution",
            "query": """
                SELECT status_code, COUNT(*) as cnt
                FROM intake GROUP BY status_code ORDER BY cnt DESC
            """,
        },
        {
            "db": "case_service",
            "desc": "Case priority distribution",
            "query": """
                SELECT priority_code, COUNT(*) as cnt
                FROM case_record GROUP BY priority_code ORDER BY cnt DESC
            """,
        },
        {
            "db": "case_service",
            "desc": "Null child_person_id in intake",
            "query": """
                SELECT COUNT(*) as null_count
                FROM intake WHERE child_person_id IS NULL
            """,
            "expect_zero": True,
        },
        {
            "db": "person_service",
            "desc": "Person records with NULL DOB",
            "query": """
                SELECT COUNT(*) as null_count
                FROM person WHERE date_of_birth IS NULL
            """,
            "expect_zero": True,
        },
        {
            "db": "person_service",
            "desc": "Gender code distribution",
            "query": """
                SELECT gender_code, COUNT(*) as cnt
                FROM person GROUP BY gender_code ORDER BY cnt DESC
            """,
        },
        {
            "db": "routing_service",
            "desc": "Routing strategy distribution",
            "query": """
                SELECT routing_strategy, COUNT(*) as cnt
                FROM case_assignment GROUP BY routing_strategy ORDER BY cnt DESC
            """,
        },
        {
            "db": "eligibility_orchestrator",
            "desc": "Eligibility execution status distribution",
            "query": """
                SELECT status_code, COUNT(*) as cnt
                FROM eligibility_execution GROUP BY status_code ORDER BY cnt DESC
            """,
        },
        {
            "db": "provider_service",
            "desc": "Referral status distribution",
            "query": """
                SELECT status_code, COUNT(*) as cnt
                FROM referral GROUP BY status_code ORDER BY cnt DESC
            """,
        },
    ]

    for check in quality_checks:
        dbname = check["db"]
        if db_filter and db_filter not in dbname:
            continue
        try:
            conn = get_connection(cfg, dbname)
            cur = conn.cursor()
            cur.execute(check["query"])
            rows = cur.fetchall()

            if check.get("expect_zero"):
                count = rows[0][0] if rows else 0
                if count == 0:
                    result.ok(f"{check['desc']}: 0 (clean)")
                else:
                    result.warn(f"{check['desc']}: {count} problematic rows")
            else:
                if rows:
                    dist = ", ".join(f"{r[0]}={r[1]}" for r in rows[:5])
                    result.ok(f"{check['desc']}: {dist}")
                else:
                    result.warn(f"{check['desc']}: no data")

            cur.close()
            conn.close()
        except psycopg2.errors.UndefinedTable:
            result.warn(f"{check['desc']}: table not found (skipped)")
        except psycopg2.OperationalError:
            result.fail(f"Cannot connect to '{dbname}' for quality check")
        except Exception as e:
            result.warn(f"{check['desc']}: {e}")
    return result


def check_indexes(cfg: TrainingConfig, db_filter: Optional[str] = None) -> CheckResult:
    """Verify key indexes exist for ML query performance."""
    result = CheckResult("Index Verification")

    critical_indexes = {
        "case_service": [
            ("intake", "idx_intake_child_status"),
            ("case_record", "idx_case_status"),
            ("case_record", "idx_case_assignee"),
        ],
        "person_service": [
            ("person", "idx_person_name_dob"),
        ],
        "routing_service": [
            ("case_assignment", "idx_case_assignment_case"),
        ],
        "eligibility_orchestrator": [
            ("eligibility_execution", "idx_exec_rule_time"),
            ("eligibility_execution", "idx_exec_subject"),
        ],
        "provider_service": [
            ("referral", "idx_ref_case"),
            ("referral", "idx_ref_status"),
        ],
    }

    for dbname, indexes in critical_indexes.items():
        if db_filter and db_filter not in dbname:
            continue
        try:
            conn = get_connection(cfg, dbname)
            cur = conn.cursor()
            cur.execute("""
                SELECT indexname FROM pg_indexes WHERE schemaname = 'public'
            """)
            existing = {row[0] for row in cur.fetchall()}
            for table, idx_name in indexes:
                if idx_name in existing:
                    result.ok(f"{dbname}.{table}.{idx_name}")
                else:
                    result.warn(f"{dbname}.{table}.{idx_name} MISSING (may slow ML queries)")
            cur.close()
            conn.close()
        except psycopg2.OperationalError:
            result.fail(f"Cannot connect to '{dbname}' for index check")
    return result


def check_cross_service_integrity(cfg: TrainingConfig) -> CheckResult:
    """Check referential integrity across microservice boundaries."""
    result = CheckResult("Cross-Service Integrity")

    try:
        # Check that case_record.intake_id references valid intakes
        conn = get_connection(cfg, cfg.db.case_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM case_record cr
            LEFT JOIN intake i ON cr.intake_id = i.intake_id
            WHERE i.intake_id IS NULL
        """)
        orphan_cases = cur.fetchone()[0]
        if orphan_cases == 0:
            result.ok("All case_records reference valid intakes")
        else:
            result.warn(f"{orphan_cases} case_records with orphaned intake_id")
        cur.close()
        conn.close()
    except Exception as e:
        result.warn(f"Case-intake integrity check: {e}")

    try:
        # Check person_id references from case_participant exist in person_service
        case_conn = get_connection(cfg, cfg.db.case_db)
        case_cur = case_conn.cursor()
        case_cur.execute("SELECT DISTINCT person_id FROM case_participant")
        case_person_ids = {row[0] for row in case_cur.fetchall()}
        case_cur.close()
        case_conn.close()

        if case_person_ids:
            person_conn = get_connection(cfg, cfg.db.person_db)
            person_cur = person_conn.cursor()
            person_cur.execute("SELECT person_id FROM person")
            person_ids = {row[0] for row in person_cur.fetchall()}
            person_cur.close()
            person_conn.close()

            orphans = case_person_ids - person_ids
            if not orphans:
                result.ok(
                    f"All {len(case_person_ids)} case_participant person_ids "
                    f"exist in person_service"
                )
            else:
                result.warn(
                    f"{len(orphans)}/{len(case_person_ids)} case_participant "
                    f"person_ids not found in person_service"
                )
        else:
            result.warn("No case_participant records to check")
    except Exception as e:
        result.warn(f"Person cross-reference check: {e}")

    return result


def check_ml_feature_availability(cfg: TrainingConfig) -> CheckResult:
    """Check that key features needed for ML models are populated."""
    result = CheckResult("ML Feature Availability")

    feature_checks = [
        {
            "db": "case_service",
            "desc": "Case contacts with notes (NLP input)",
            "query": "SELECT COUNT(*) FROM case_contact WHERE notes IS NOT NULL AND notes != ''",
            "min_count": 100,
        },
        {
            "db": "person_service",
            "desc": "Person notes for risk NLP",
            "query": "SELECT COUNT(*) FROM person_note WHERE note_text IS NOT NULL AND note_text != ''",
            "min_count": 100,
        },
        {
            "db": "person_service",
            "desc": "Persons with addresses (geography features)",
            "query": """
                SELECT COUNT(DISTINCT pa.person_id)
                FROM person_address pa JOIN address a ON pa.address_id = a.address_id
                WHERE a.postal_code IS NOT NULL
            """,
            "min_count": 200,
        },
        {
            "db": "routing_service",
            "desc": "Assignments with ML scores (for reward signal)",
            "query": "SELECT COUNT(*) FROM case_assignment WHERE model_score IS NOT NULL",
            "min_count": 0,  # may be zero initially
        },
        {
            "db": "provider_service",
            "desc": "Providers with geo-coordinates",
            "query": "SELECT COUNT(*) FROM provider_location WHERE latitude IS NOT NULL",
            "min_count": 10,
        },
        {
            "db": "provider_assignment_service",
            "desc": "Provider performance snapshots",
            "query": "SELECT COUNT(*) FROM provider_performance_snapshot",
            "min_count": 10,
        },
    ]

    for check in feature_checks:
        try:
            conn = get_connection(cfg, check["db"])
            cur = conn.cursor()
            cur.execute(check["query"])
            count = cur.fetchone()[0]
            if count >= check["min_count"]:
                result.ok(f"{check['desc']}: {count:,} records")
            else:
                result.warn(
                    f"{check['desc']}: {count:,} records "
                    f"(need {check['min_count']}+ for training)"
                )
            cur.close()
            conn.close()
        except Exception as e:
            result.warn(f"{check['desc']}: {e}")

    return result


def run_all_checks(cfg: TrainingConfig, db_filter: Optional[str] = None, verbose: bool = False):
    """Execute all database checks and print a summary report."""
    print("=" * 70)
    print(f"  CaseAI Database Readiness Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Host: {cfg.db.host}:{cfg.db.port}")
    if db_filter:
        print(f"  Filter: {db_filter}")
    print("=" * 70)

    checks = [
        check_connectivity(cfg),
        check_tables(cfg, db_filter),
        check_columns(cfg, db_filter),
        check_row_counts(cfg, db_filter),
        check_data_quality(cfg, db_filter),
        check_indexes(cfg, db_filter),
        check_cross_service_integrity(cfg),
        check_ml_feature_availability(cfg),
    ]

    total_pass = 0
    total_fail = 0
    total_warn = 0

    for check in checks:
        print(f"\n{check.summary()}")
        if verbose:
            for detail in check.details:
                print(detail)
        total_pass += check.passed
        total_fail += check.failed
        total_warn += check.warnings

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed, {total_warn} warnings")
    if total_fail > 0:
        print("  STATUS: NOT READY for ML training")
        print("  ACTION: Fix failed checks above, or run synthetic data generator")
    elif total_warn > 5:
        print("  STATUS: PARTIALLY READY (warnings should be reviewed)")
        print("  ACTION: Consider generating synthetic data for low-volume tables")
    else:
        print("  STATUS: READY for ML training")
    print("=" * 70)

    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(description="CaseAI Database Readiness Checker")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Filter to specific database (e.g., 'case', 'person', 'routing')",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    args = parser.parse_args()

    cfg = TrainingConfig()
    success = run_all_checks(cfg, db_filter=args.db, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
