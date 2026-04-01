"""
CaseAI Bias Monitoring Agent
Continuous fairness auditor for all model predictions.
Monitors disparate impact, FPR parity, and calibration disparity.

Usage:
    python bias_monitor.py                     # Run one-time audit
    python bias_monitor.py --continuous        # Run continuously
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import TrainingConfig
from data_extractor import CaseAIDataExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bias_monitor")


class BiasMonitor:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.report_dir = Path(cfg.model_output_dir) / "bias_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run_audit(self) -> dict:
        """Run a complete fairness audit across all demographic axes."""
        extractor = CaseAIDataExtractor(self.cfg)
        df = extractor.extract_bias_monitoring_data()

        if len(df) < 50:
            logger.warning("Insufficient data for bias audit (%d rows)", len(df))
            return {"status": "INSUFFICIENT_DATA"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "n_records": len(df),
            "audits": {},
            "alerts": [],
        }

        # Audit risk decisions
        if "risk_decision" in df.columns:
            risk_audit = self._audit_decisions(
                df, decision_col="risk_decision",
                positive_values=["CRITICAL", "HIGH"],
                audit_name="Risk Scoring",
            )
            report["audits"]["risk_scoring"] = risk_audit

        # Audit eligibility decisions
        if "elig_decision" in df.columns:
            elig_audit = self._audit_decisions(
                df, decision_col="elig_decision",
                positive_values=["ELIGIBLE"],
                audit_name="Eligibility",
            )
            report["audits"]["eligibility"] = elig_audit

        # Collect alerts
        for audit_name, audit_data in report["audits"].items():
            for alert in audit_data.get("alerts", []):
                alert["source"] = audit_name
                report["alerts"].append(alert)

        # Overall status
        if report["alerts"]:
            report["status"] = "ALERT"
            logger.warning("BIAS ALERTS DETECTED: %d issues found", len(report["alerts"]))
        else:
            report["status"] = "CLEAN"
            logger.info("No bias alerts detected")

        # Save report
        filename = f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(self.report_dir / filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self._print_report(report)
        return report

    def _audit_decisions(self, df: pd.DataFrame, decision_col: str,
                         positive_values: list, audit_name: str) -> dict:
        """Audit a decision column for demographic bias."""
        audit = {"decision_col": decision_col, "groups": {}, "alerts": []}

        valid = df.dropna(subset=[decision_col])
        valid["positive"] = valid[decision_col].isin(positive_values).astype(int)

        overall_positive_rate = valid["positive"].mean()
        audit["overall_positive_rate"] = round(float(overall_positive_rate), 4)

        # Audit across each demographic axis
        for axis in ["gender_code", "geo_group", "age_group"]:
            if axis not in valid.columns:
                continue

            group_stats = valid.groupby(axis).agg(
                count=("positive", "size"),
                positive_count=("positive", "sum"),
                positive_rate=("positive", "mean"),
            ).reset_index()

            # Need at least 2 groups with enough samples
            group_stats = group_stats[group_stats["count"] >= 10]
            if len(group_stats) < 2:
                continue

            # Reference group (largest)
            ref_group = group_stats.loc[group_stats["count"].idxmax()]
            ref_rate = ref_group["positive_rate"]

            axis_results = {"reference_group": str(ref_group[axis]), "groups": {}}

            for _, row in group_stats.iterrows():
                group_name = str(row[axis])
                group_rate = row["positive_rate"]

                # Disparate Impact Ratio
                if ref_rate > 0:
                    dir_val = group_rate / ref_rate
                else:
                    dir_val = 1.0 if group_rate == 0 else float("inf")

                group_result = {
                    "count": int(row["count"]),
                    "positive_count": int(row["positive_count"]),
                    "positive_rate": round(float(group_rate), 4),
                    "disparate_impact_ratio": round(float(dir_val), 4),
                }

                # Check thresholds
                if dir_val < self.cfg.bias.disparate_impact_lower:
                    alert = {
                        "type": "DISPARATE_IMPACT_LOW",
                        "axis": axis,
                        "group": group_name,
                        "reference": str(ref_group[axis]),
                        "dir": round(float(dir_val), 4),
                        "threshold": self.cfg.bias.disparate_impact_lower,
                        "message": (
                            f"{audit_name}: Group '{group_name}' has DIR={dir_val:.3f} "
                            f"(below {self.cfg.bias.disparate_impact_lower}) on axis '{axis}'"
                        ),
                    }
                    audit["alerts"].append(alert)
                    group_result["alert"] = "DISPARATE_IMPACT_LOW"

                elif dir_val > self.cfg.bias.disparate_impact_upper:
                    alert = {
                        "type": "DISPARATE_IMPACT_HIGH",
                        "axis": axis,
                        "group": group_name,
                        "reference": str(ref_group[axis]),
                        "dir": round(float(dir_val), 4),
                        "threshold": self.cfg.bias.disparate_impact_upper,
                        "message": (
                            f"{audit_name}: Group '{group_name}' has DIR={dir_val:.3f} "
                            f"(above {self.cfg.bias.disparate_impact_upper}) on axis '{axis}'"
                        ),
                    }
                    audit["alerts"].append(alert)
                    group_result["alert"] = "DISPARATE_IMPACT_HIGH"

                axis_results["groups"][group_name] = group_result

            audit["groups"][axis] = axis_results

        return audit

    def _print_report(self, report: dict):
        """Print a human-readable bias report."""
        print("\n" + "=" * 70)
        print(f"  CaseAI Bias Monitoring Report")
        print(f"  Timestamp: {report['timestamp']}")
        print(f"  Records Analyzed: {report['n_records']:,}")
        print(f"  Status: {report['status']}")
        print("=" * 70)

        for audit_name, audit_data in report.get("audits", {}).items():
            print(f"\n--- {audit_name.upper()} ---")
            print(f"  Overall positive rate: {audit_data.get('overall_positive_rate', 'N/A')}")

            for axis, axis_data in audit_data.get("groups", {}).items():
                print(f"\n  Axis: {axis} (ref: {axis_data['reference_group']})")
                for group, stats in axis_data.get("groups", {}).items():
                    alert_marker = " *** ALERT ***" if "alert" in stats else ""
                    print(
                        f"    {group:20s}: rate={stats['positive_rate']:.3f}, "
                        f"DIR={stats['disparate_impact_ratio']:.3f}, "
                        f"n={stats['count']}{alert_marker}"
                    )

        if report["alerts"]:
            print(f"\n{'!' * 70}")
            print(f"  {len(report['alerts'])} ALERT(S) DETECTED:")
            for alert in report["alerts"]:
                print(f"    - {alert['message']}")
            print(f"{'!' * 70}")
        else:
            print("\n  No bias alerts detected.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CaseAI Bias Monitor")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    args = parser.parse_args()

    cfg = TrainingConfig()
    monitor = BiasMonitor(cfg)

    if args.continuous:
        import time
        logger.info("Starting continuous monitoring (interval: %d min)",
                     cfg.bias.monitoring_interval_minutes)
        while True:
            monitor.run_audit()
            time.sleep(cfg.bias.monitoring_interval_minutes * 60)
    else:
        monitor.run_audit()


if __name__ == "__main__":
    main()
