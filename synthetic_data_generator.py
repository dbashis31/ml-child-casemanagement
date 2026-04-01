"""
CaseAI Synthetic Data Generator
Generates realistic synthetic child welfare case management data for ML training.
Populates all microservice databases with correlated, demographically representative data.

Usage:
    python synthetic_data_generator.py              # Generate with defaults
    python synthetic_data_generator.py --scale 2.0  # Double the default volume
    python synthetic_data_generator.py --seed 123   # Custom random seed
"""
import argparse
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from config import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthetic_gen")

# ---------------------------------------------------------------------------
# Reference data for realistic generation
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Daniel", "Lisa", "Matthew", "Nancy",
    "Anthony", "Betty", "Mark", "Margaret", "Sofia", "Liam", "Olivia", "Noah",
    "Emma", "Aiden", "Ava", "Lucas", "Mia", "Ethan", "Isabella", "Mason",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]

CITIES = [
    ("Halifax", "NS", "B3H"), ("Dartmouth", "NS", "B2W"), ("Sydney", "NS", "B1P"),
    ("Truro", "NS", "B2N"), ("New Glasgow", "NS", "B2H"), ("Kentville", "NS", "B4N"),
    ("Bridgewater", "NS", "B4V"), ("Yarmouth", "NS", "B5A"), ("Amherst", "NS", "B4H"),
    ("Antigonish", "NS", "B2G"),
]

STREETS = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine Rd", "Elm St",
    "Spring Garden Rd", "Barrington St", "Robie St", "Quinpool Rd",
    "Portland St", "Prince St", "George St", "King St", "Queen St",
]

INTAKE_NOTES_TEMPLATES = [
    "Referral received regarding concerns about {concern} for child age {age}. "
    "Reporter indicates {detail}. Family has {history}.",
    "Anonymous call reporting {concern}. Child is {age} years old. "
    "Caller states {detail}. Previous involvement: {history}.",
    "School counselor reported {concern} for {age}-year-old student. "
    "Teacher observations include {detail}. {history}.",
    "Medical professional flagged {concern} during routine visit for child age {age}. "
    "Clinical notes indicate {detail}. Family background: {history}.",
    "Neighbor reports {concern}. Child approximately {age} years old. "
    "Observations: {detail}. {history}.",
]

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
    "parent has refused to follow through on prescribed medical treatment",
    "family has been evicted and is living in a vehicle",
]

HISTORIES = [
    "no prior CPS involvement", "one prior referral 2 years ago (unsubstantiated)",
    "two prior substantiated findings", "family previously received voluntary services",
    "prior case closed 18 months ago with successful reunification",
    "no known history in this jurisdiction",
    "sibling had prior open case for similar concerns",
    "multiple referrals over past 3 years with varied outcomes",
]

SERVICE_TYPES = [
    ("COUNSELING", "Family Counseling"),
    ("SPEECH_THERAPY", "Speech Therapy"),
    ("FINANCIAL_AID", "Financial Assistance"),
    ("PARENTING_CLASSES", "Parenting Education"),
    ("SUBSTANCE_TREATMENT", "Substance Abuse Treatment"),
    ("MENTAL_HEALTH", "Mental Health Services"),
    ("HOUSING_SUPPORT", "Housing Support"),
    ("DAYCARE", "Child Day Care"),
    ("MEDICAL", "Medical Services"),
    ("LEGAL_AID", "Legal Aid Services"),
]

ORG_TYPES = ["CLINIC", "NGO", "DAYCARE", "THERAPY_CENTER", "GOVERNMENT", "HOSPITAL"]
STAFF_ROLES = ["THERAPIST", "COUNSELOR", "ADMIN", "SOCIAL_WORKER", "NURSE", "PSYCHOLOGIST"]


def uid() -> str:
    return str(uuid.uuid4())


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_intake_note(age: int) -> str:
    template = random.choice(INTAKE_NOTES_TEMPLATES)
    return template.format(
        concern=random.choice(CONCERNS),
        age=age,
        detail=random.choice(DETAILS),
        history=random.choice(HISTORIES),
    )


class SyntheticDataGenerator:
    def __init__(self, cfg: TrainingConfig, scale: float = 1.0):
        self.cfg = cfg
        self.scale = scale
        self.rng = np.random.default_rng(cfg.synthetic.seed)
        random.seed(cfg.synthetic.seed)

        # Scaled counts
        self.n_persons = int(cfg.synthetic.n_persons * scale)
        self.n_intakes = int(cfg.synthetic.n_intakes * scale)
        self.n_cases = int(cfg.synthetic.n_cases * scale)
        self.n_providers = int(cfg.synthetic.n_providers * scale)
        self.n_caseworkers = int(cfg.synthetic.n_caseworkers * scale)

        # Generated ID pools (populated during generation)
        self.person_ids: list[str] = []
        self.child_person_ids: list[str] = []
        self.caseworker_user_ids: list[str] = []
        self.intake_ids: list[str] = []
        self.case_ids: list[str] = []
        self.provider_org_ids: list[str] = []
        self.team_ids: list[str] = []

    def _conn(self, dbname: str):
        return psycopg2.connect(
            host=self.cfg.db.host, port=self.cfg.db.port,
            user=self.cfg.db.user, password=self.cfg.db.password,
            dbname=dbname,
        )

    def generate_all(self):
        logger.info("Starting synthetic data generation (scale=%.1f)", self.scale)
        self.generate_persons()
        self.generate_intakes_and_cases()
        self.generate_routing()
        self.generate_eligibility()
        self.generate_providers()
        self.generate_provider_assignments()
        logger.info("Synthetic data generation complete!")

    # ----- Person Service -----

    def generate_persons(self):
        logger.info("Generating %d persons...", self.n_persons)
        conn = self._conn(self.cfg.db.person_db)
        cur = conn.cursor()

        persons = []
        addresses = []
        person_addresses = []
        person_notes = []
        now = datetime.now()

        for i in range(self.n_persons):
            pid = uid()
            self.person_ids.append(pid)

            is_child = i < self.n_intakes  # first N persons are children
            if is_child:
                self.child_person_ids.append(pid)
                dob = random_date(now - timedelta(days=17*365), now - timedelta(days=30))
            else:
                dob = random_date(now - timedelta(days=65*365), now - timedelta(days=18*365))

            gender = random.choice(["MALE", "FEMALE", "NON_BINARY"])
            persons.append((
                pid, random.choice(FIRST_NAMES), random.choice(LAST_NAMES),
                dob.date(), gender, True, now, "SYSTEM", now, "SYSTEM",
            ))

            # Address
            aid = uid()
            city, state, postal_prefix = random.choice(CITIES)
            postal = f"{postal_prefix} {random.randint(1,9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1,9)}"
            addresses.append((
                aid, f"{random.randint(1,999)} {random.choice(STREETS)}",
                city, state, postal, "CA", now,
            ))
            person_addresses.append((
                uid(), pid, aid, random.choice(["HOME", "MAILING"]),
                True, dob.date(), now, "SYSTEM",
            ))

            # Notes for ~40% of persons
            if random.random() < 0.4:
                age = (now.date() - dob.date()).days // 365
                note_text = generate_intake_note(max(1, age))
                person_notes.append((
                    uid(), pid, note_text,
                    random.choice(["GENERAL", "RISK", "ASSESSMENT", "OBSERVATION"]),
                    random.random() < 0.1, now, "SYSTEM",
                ))

        execute_values(cur, """
            INSERT INTO person (person_id, first_name, last_name, date_of_birth,
                gender_code, is_active, created_at, created_by, updated_at, updated_by)
            VALUES %s ON CONFLICT DO NOTHING
        """, persons)

        execute_values(cur, """
            INSERT INTO address (address_id, line1, city, state_province,
                postal_code, country_code, created_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, addresses)

        execute_values(cur, """
            INSERT INTO person_address (person_address_id, person_id, address_id,
                address_type, is_current, start_date, created_at, created_by)
            VALUES %s ON CONFLICT DO NOTHING
        """, person_addresses)

        execute_values(cur, """
            INSERT INTO person_note (person_note_id, person_id, note_text,
                note_type, is_confidential, created_at, created_by)
            VALUES %s ON CONFLICT DO NOTHING
        """, person_notes)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("  Persons: %d, Addresses: %d, Notes: %d",
                     len(persons), len(addresses), len(person_notes))

    # ----- Case Service -----

    def generate_intakes_and_cases(self):
        logger.info("Generating %d intakes and %d cases...", self.n_intakes, self.n_cases)
        conn = self._conn(self.cfg.db.case_db)
        cur = conn.cursor()

        # Generate caseworker user IDs
        self.caseworker_user_ids = [uid() for _ in range(self.n_caseworkers)]

        intakes = []
        cases = []
        participants = []
        contacts = []
        now = datetime.now()

        statuses = ["DRAFT", "VALIDATED", "ELIGIBLE", "SUBMITTED", "CLOSED"]
        priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        eligibility_statuses = ["ELIGIBLE", "NOT_ELIGIBLE", "PENDING"]

        for i in range(self.n_intakes):
            iid = uid()
            self.intake_ids.append(iid)
            child_pid = self.child_person_ids[i % len(self.child_person_ids)]
            caseworker = random.choice(self.caseworker_user_ids)
            created = random_date(now - timedelta(days=730), now)

            status = random.choices(statuses, weights=[0.05, 0.1, 0.15, 0.5, 0.2])[0]
            elig = random.choice(eligibility_statuses) if status != "DRAFT" else None
            submitted = created + timedelta(hours=random.randint(1, 72)) if status == "SUBMITTED" else None

            intakes.append((
                iid, caseworker, child_pid, status,
                "PASSED" if status != "DRAFT" else None,
                elig, None, None, submitted, created, created,
            ))

            # Create case for submitted/closed intakes
            if status in ("SUBMITTED", "CLOSED") and len(cases) < self.n_cases:
                cid = uid()
                self.case_ids.append(cid)
                case_status = random.choice(["ASSIGNED", "ACTIVE", "HOLD", "CLOSED"])
                priority = random.choices(priorities, weights=[0.1, 0.25, 0.4, 0.25])[0]
                assigned_cw = random.choice(self.caseworker_user_ids)

                cases.append((
                    cid, f"CASE-{len(cases)+1:06d}", iid, case_status,
                    priority, random.choice(["NS", "NB", "PE"]),
                    assigned_cw, created + timedelta(hours=random.randint(1, 48)),
                    created, "SYSTEM",
                ))

                # Participants: child + guardian + sometimes caseworker
                participants.append((
                    uid(), cid, child_pid, "CHILD", True,
                    created.date(), None, created, "SYSTEM",
                ))
                guardian_pid = random.choice(self.person_ids)
                participants.append((
                    uid(), cid, guardian_pid, "GUARDIAN", False,
                    created.date(), None, created, "SYSTEM",
                ))

                # 1-5 contacts per case
                for _ in range(random.randint(1, 5)):
                    contact_dt = created + timedelta(days=random.randint(1, 180))
                    age = max(1, random.randint(1, 17))
                    contacts.append((
                        uid(), cid, random.choice(["CALL", "VISIT", "EMAIL", "MEETING"]),
                        contact_dt, child_pid,
                        f"Follow-up contact regarding case progress",
                        generate_intake_note(age),
                        contact_dt, caseworker,
                    ))

        execute_values(cur, """
            INSERT INTO intake (intake_id, created_by_user_id, child_person_id,
                status_code, validation_status, eligibility_status,
                eligibility_reason_code, draft_payload_json, submitted_at,
                created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, intakes)

        execute_values(cur, """
            INSERT INTO case_record (case_id, case_number, intake_id, status_code,
                priority_code, jurisdiction_code, assigned_to_user_id,
                assigned_at, created_at, created_by)
            VALUES %s ON CONFLICT DO NOTHING
        """, cases)

        execute_values(cur, """
            INSERT INTO case_participant (case_participant_id, case_id, person_id,
                role_code, is_primary, start_date, end_date, created_at, created_by)
            VALUES %s ON CONFLICT DO NOTHING
        """, participants)

        execute_values(cur, """
            INSERT INTO case_contact (case_contact_id, case_id, contact_type,
                contact_datetime, contacted_person_id, summary, notes,
                created_at, created_by_user_id)
            VALUES %s ON CONFLICT DO NOTHING
        """, contacts)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("  Intakes: %d, Cases: %d, Participants: %d, Contacts: %d",
                     len(intakes), len(cases), len(participants), len(contacts))

    # ----- Routing Service -----

    def generate_routing(self):
        logger.info("Generating routing data...")
        conn = self._conn(self.cfg.db.routing_db)
        cur = conn.cursor()

        teams = []
        members = []
        policies = []
        assignments = []
        now = datetime.now()

        # Create teams
        team_names = [
            "Halifax Child Care Team", "Dartmouth Family Services",
            "Cape Breton Intake Unit", "Valley Region CPS",
            "South Shore Support Team",
        ]
        for name in team_names:
            tid = uid()
            self.team_ids.append(tid)
            key = name.upper().replace(" ", "_")
            teams.append((tid, key, name, True, now, now))

            # Assign caseworkers to teams
            team_cws = random.sample(
                self.caseworker_user_ids,
                min(len(self.caseworker_user_ids), random.randint(5, 20)),
            )
            for cw_id in team_cws:
                members.append((uid(), tid, cw_id, True, now))

            # Policy per team
            strategy = random.choice(["ROUND_ROBIN", "ROUND_ROBIN", "XGBOOST"])
            policies.append((uid(), tid, strategy, True, None, now, now))

        # Case assignments
        strategies = ["ROUND_ROBIN", "XGBOOST", "MANUAL", "FALLBACK_RR"]
        for cid in self.case_ids:
            team = random.choice(self.team_ids)
            cw = random.choice(self.caseworker_user_ids)
            strategy = random.choices(strategies, weights=[0.5, 0.2, 0.2, 0.1])[0]
            score = round(random.uniform(0.3, 0.95), 6) if strategy == "XGBOOST" else None

            assignments.append((
                uid(), cid, "USER", cw, team, strategy,
                random.choice(["WORKLOAD", "JURISDICTION", "SPECIALIZATION", "PROXIMITY"]),
                None, None, None, score,
                json.dumps({"top_features": ["workload", "proximity", "specialization"]}) if score else None,
                random_date(now - timedelta(days=365), now),
            ))

        execute_values(cur, """
            INSERT INTO routing_team (team_id, team_key, team_name, is_active,
                created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, teams)

        execute_values(cur, """
            INSERT INTO routing_team_member (team_member_id, team_id, user_id,
                is_active, joined_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, members)

        execute_values(cur, """
            INSERT INTO routing_policy (policy_id, team_id, strategy, is_active,
                config_json, created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, policies)

        execute_values(cur, """
            INSERT INTO case_assignment (assignment_id, case_id, assigned_type,
                assigned_to_id, team_id, routing_strategy, assigned_reason,
                assigned_by_user_id, correlation_id, model_name, model_score,
                decision_details_json, assigned_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, assignments)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("  Teams: %d, Members: %d, Assignments: %d",
                     len(teams), len(members), len(assignments))

    # ----- Eligibility Service -----

    def generate_eligibility(self):
        logger.info("Generating eligibility data...")
        conn = self._conn(self.cfg.db.eligibility_db)
        cur = conn.cursor()

        # BRM provider
        cur.execute("""
            INSERT INTO brm_provider_registry (brm_provider_key, display_name,
                endpoint_base_url, auth_type, is_active)
            VALUES ('INRULE', 'InRule Decision Engine', 'https://inrule.internal/api', 'M2M_JWT', 1)
            ON CONFLICT DO NOTHING
        """)

        # Lookup service
        cur.execute("""
            INSERT INTO lookup_service_registry (service_key, display_name,
                base_url, auth_type, is_active)
            VALUES ('PERSON_API', 'Person Service API', 'http://person-service:8080', 'M2M_JWT', 1)
            ON CONFLICT DO NOTHING
        """)

        # Rule definition
        rule_id = uid()
        cur.execute("""
            INSERT INTO rule_definition (rule_id, rule_key, description,
                brm_provider_key, brm_ruleset_name, is_active)
            VALUES (%s, 'ELIGIBILITY_CHILD_CARE', 'Child care eligibility determination',
                'INRULE', 'ChildCareEligibility', 1)
            ON CONFLICT DO NOTHING
        """, (rule_id,))

        # Eligibility executions
        executions = []
        now = datetime.now()
        for i, iid in enumerate(self.intake_ids):
            child_pid = self.child_person_ids[i % len(self.child_person_ids)]
            started = random_date(now - timedelta(days=365), now)
            duration = random.randint(200, 5000)
            status = random.choices(["SUCCESS", "FAILED", "PARTIAL"], weights=[0.85, 0.1, 0.05])[0]

            decision = {
                "eligible": random.random() > 0.3,
                "program": "CHILD_CARE",
                "confidence": round(random.uniform(0.4, 0.99), 3),
                "reasons": random.sample([
                    "income_below_threshold", "age_eligible",
                    "jurisdiction_match", "no_duplicate_case",
                    "program_capacity_available",
                ], k=random.randint(1, 3)),
            }

            executions.append((
                uid(), "ELIGIBILITY_CHILD_CARE", uid(), None,
                child_pid, iid,
                json.dumps({"childPersonId": child_pid, "intakeId": iid}),
                json.dumps({"child_age": random.randint(1, 17), "income": random.randint(15000, 80000)}),
                "INRULE", "ChildCareEligibility", "v1.0",
                None, None,
                json.dumps(decision), status, None, None,
                started, started + timedelta(milliseconds=duration), duration,
            ))

        execute_values(cur, """
            INSERT INTO eligibility_execution (execution_id, rule_key, correlation_id,
                idempotency_key, subject_person_id, intake_id, request_json, facts_json,
                brm_provider_key, brm_ruleset_name, brm_ruleset_ver,
                brm_request_json, brm_response_json,
                decision_json, status_code, error_code, error_message,
                started_at, completed_at, duration_ms)
            VALUES %s ON CONFLICT DO NOTHING
        """, executions)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("  Eligibility executions: %d", len(executions))

    # ----- Provider Service -----

    def generate_providers(self):
        logger.info("Generating %d providers...", self.n_providers)
        conn = self._conn(self.cfg.db.provider_db)
        cur = conn.cursor()

        orgs = []
        locations = []
        services_catalog = []
        org_services = []
        staff = []
        referrals = []
        capacities = []
        now = datetime.now()

        # Service catalog
        for code, name in SERVICE_TYPES:
            services_catalog.append((code, name, None, True, now, now))

        execute_values(cur, """
            INSERT INTO service_catalog (service_code, service_name, description,
                is_active, created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, services_catalog)

        # Provider orgs
        for i in range(self.n_providers):
            oid = uid()
            self.provider_org_ids.append(oid)
            org_type = random.choice(ORG_TYPES)
            orgs.append((
                oid, f"{random.choice(LAST_NAMES)} {org_type.title()} Services",
                org_type, "ACTIVE",
                f"902-{random.randint(200,999)}-{random.randint(1000,9999)}",
                f"info@provider{i}.ca", None, now, "SYSTEM", now, "SYSTEM",
            ))

            # Location
            city, state, postal_prefix = random.choice(CITIES)
            lid = uid()
            locations.append((
                lid, oid, f"Main Office",
                f"{random.randint(1,999)} {random.choice(STREETS)}",
                None, city, state,
                f"{postal_prefix} {random.randint(1,9)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1,9)}",
                "CA",
                round(44.6 + random.uniform(-1, 1), 7),
                round(-63.5 + random.uniform(-2, 2), 7),
                None, None, True, True, now, now,
            ))

            # Services offered (1-4 per org)
            offered = random.sample(SERVICE_TYPES, k=random.randint(1, 4))
            for svc_code, _ in offered:
                org_services.append((uid(), oid, svc_code, "ACTIVE", None, now, now))
                capacities.append((
                    uid(), oid, svc_code,
                    random.choice(["OPEN", "OPEN", "LIMITED", "CLOSED"]),
                    random.randint(5, 50),
                    random.randint(0, 30),
                    now,
                ))

            # Staff (1-5 per org)
            for _ in range(random.randint(1, 5)):
                staff_person = random.choice(self.person_ids)
                staff.append((
                    uid(), oid, staff_person,
                    random.choice(STAFF_ROLES), True, now, now,
                ))

        execute_values(cur, """
            INSERT INTO provider_org (provider_org_id, org_name, org_type_code,
                status_code, phone, email, website, created_at, created_by,
                updated_at, updated_by)
            VALUES %s ON CONFLICT DO NOTHING
        """, orgs)

        execute_values(cur, """
            INSERT INTO provider_location (provider_location_id, provider_org_id,
                location_name, line1, line2, city, state_province, postal_code,
                country_code, latitude, longitude, phone, email,
                is_primary, is_active, created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, locations)

        execute_values(cur, """
            INSERT INTO provider_org_service (provider_org_service_id, provider_org_id,
                service_code, status_code, notes, created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, org_services)

        execute_values(cur, """
            INSERT INTO provider_staff (provider_staff_id, provider_org_id,
                person_id, staff_role_code, is_active, created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, staff)

        execute_values(cur, """
            INSERT INTO provider_capacity (provider_capacity_id, provider_org_id,
                service_code, capacity_status_code, max_active_referrals,
                current_active_referrals, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, capacities)

        # Referrals from cases to providers
        ref_statuses = ["SENT", "ACCEPTED", "DECLINED", "SCHEDULED", "IN_PROGRESS", "COMPLETED", "CANCELLED"]
        for cid in self.case_ids[:int(len(self.case_ids) * 0.7)]:
            child_pid = random.choice(self.child_person_ids)
            prov = random.choice(self.provider_org_ids)
            svc_code = random.choice(SERVICE_TYPES)[0]
            created = random_date(now - timedelta(days=365), now)
            status = random.choices(
                ref_statuses, weights=[0.1, 0.2, 0.05, 0.15, 0.2, 0.25, 0.05]
            )[0]

            referrals.append((
                uid(), cid, child_pid, prov, None, svc_code,
                random.choice(["HIGH", "MEDIUM", "LOW"]),
                status, None, created.date(),
                (created + timedelta(days=7)).date() if status not in ("SENT", "DECLINED") else None,
                (created + timedelta(days=90)).date() if status == "COMPLETED" else None,
                None, random.choice(self.caseworker_user_ids), created, created,
            ))

        execute_values(cur, """
            INSERT INTO referral (referral_id, case_id, child_person_id,
                provider_org_id, provider_location_id, service_code,
                priority_code, status_code, status_reason_code,
                requested_start_date, scheduled_date, completed_date,
                notes, created_by_user_id, created_at, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, referrals)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("  Orgs: %d, Locations: %d, Services: %d, Staff: %d, Referrals: %d",
                     len(orgs), len(locations), len(org_services), len(staff), len(referrals))

    # ----- Provider Assignment Service -----

    def generate_provider_assignments(self):
        logger.info("Generating provider assignment data...")
        conn = self._conn(self.cfg.db.provider_assignment_db)
        cur = conn.cursor()

        # Policy
        policy_id = uid()
        cur.execute("""
            INSERT INTO assignment_policy (policy_id, policy_key, is_active,
                jurisdiction_code, program_code, service_code, strategy,
                fallback_strategy, config_json, created_at, created_by,
                updated_at, updated_by)
            VALUES (%s, 'DEFAULT_CHILDCARE_COUNSELING_NS', 1, 'NS', 'CHILD_CARE',
                'COUNSELING', 'RULE_BASED', 'RULE_BASED', NULL,
                NOW(), 'SYSTEM', NOW(), 'SYSTEM')
            ON CONFLICT DO NOTHING
        """, (policy_id,))

        requests = []
        recommendations = []
        selections = []
        snapshots = []
        now = datetime.now()

        for cid in self.case_ids[:int(len(self.case_ids) * 0.5)]:
            rid = uid()
            child_pid = random.choice(self.child_person_ids)
            svc = random.choice(SERVICE_TYPES)[0]
            started = random_date(now - timedelta(days=365), now)
            duration = random.randint(100, 3000)
            strategy = random.choice(["RULE_BASED", "XGBOOST", "FALLBACK_RULE_BASED"])

            requests.append((
                rid, uid(), None, cid, child_pid, svc, "NS", "CHILD_CARE",
                random.choice(["HIGH", "MEDIUM", "LOW"]),
                random.choice(self.caseworker_user_ids),
                json.dumps({"postalCode": "B3H", "radius_km": 25}),
                json.dumps({"child_age": random.randint(1, 17)}),
                policy_id, strategy, "SUCCESS", None, None,
                started, started + timedelta(milliseconds=duration), duration,
            ))

            # Top 3 recommendations per request
            top_providers = random.sample(
                self.provider_org_ids, min(3, len(self.provider_org_ids))
            )
            for rank, prov_id in enumerate(top_providers, 1):
                score = round(random.uniform(0.3, 0.98), 6)
                recommendations.append((
                    uid(), rid, rank, prov_id, None, None,
                    score,
                    json.dumps({
                        "distanceScore": round(random.uniform(0, 1), 3),
                        "loadScore": round(random.uniform(0, 1), 3),
                        "capacityScore": round(random.uniform(0, 1), 3),
                    }),
                    json.dumps(["closest match", "capacity available"]),
                    None, None, None, None, started,
                ))

            # Selection (pick rank 1)
            selections.append((
                uid(), rid, top_providers[0], None, None,
                random.choice(["CASEWORKER_PICK", "AUTO_SELECT"]),
                random.choice(self.caseworker_user_ids),
                None, None, started,
            ))

        execute_values(cur, """
            INSERT INTO assignment_request (request_id, correlation_id,
                idempotency_key, case_id, child_person_id, service_code,
                jurisdiction_code, program_code, priority_code,
                requested_by_user_id, request_context_json, facts_json,
                policy_id, strategy_used, status_code, error_code, error_message,
                started_at, completed_at, duration_ms)
            VALUES %s ON CONFLICT DO NOTHING
        """, requests)

        execute_values(cur, """
            INSERT INTO assignment_recommendation (recommendation_id, request_id,
                rank_no, provider_org_id, provider_location_id,
                provider_staff_person_id, score, scoring_breakdown_json,
                reasons_json, model_name, model_version, model_score,
                decision_details_json, created_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, recommendations)

        execute_values(cur, """
            INSERT INTO assignment_selection (selection_id, request_id,
                selected_provider_org_id, selected_provider_location_id,
                selected_provider_staff_person_id, selection_method,
                selected_by_user_id, selection_notes, referral_id, selected_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, selections)

        # Performance snapshots
        for prov_id in self.provider_org_ids:
            for svc_code, _ in random.sample(SERVICE_TYPES, k=random.randint(1, 3)):
                snapshots.append((
                    uid(), prov_id, svc_code,
                    round(random.uniform(0.5, 0.98), 4),
                    round(random.uniform(2, 72), 2),
                    round(random.uniform(0.4, 0.95), 4),
                    round(random.uniform(7, 120), 2),
                    random.randint(0, 25),
                    random.choice(["OPEN", "LIMITED", "CLOSED"]),
                    now,
                ))

        execute_values(cur, """
            INSERT INTO provider_performance_snapshot (snapshot_id, provider_org_id,
                service_code, acceptance_rate, avg_time_to_accept_hours,
                completion_rate, avg_time_to_complete_days,
                current_active_referrals, capacity_status_code, updated_at)
            VALUES %s ON CONFLICT DO NOTHING
        """, snapshots)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("  Requests: %d, Recommendations: %d, Selections: %d, Snapshots: %d",
                     len(requests), len(recommendations), len(selections), len(snapshots))


def main():
    parser = argparse.ArgumentParser(description="CaseAI Synthetic Data Generator")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for data volume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = TrainingConfig()
    cfg.synthetic.seed = args.seed
    generator = SyntheticDataGenerator(cfg, scale=args.scale)
    generator.generate_all()


if __name__ == "__main__":
    main()
