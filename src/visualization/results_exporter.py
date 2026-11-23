"""Deterministic JSON serialization of the complete SCLDO analytical report."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.core.contracts import ModelEvaluationRecord, ScldoReportPayload


def _serialize_model_record(record: ModelEvaluationRecord) -> dict[str, Any]:
    return {
        "regime": record.regime,
        "model_name": record.model_name,
        "calibration_method": record.calibration_method,
        "cv_metrics": asdict(record.cv_metrics),
        "oot_metrics": asdict(record.oot_metrics),
        "oot_reliability_bins": [asdict(b) for b in record.oot_reliability_bins],
    }


def serialize_scldo_payload(payload: ScldoReportPayload) -> dict[str, Any]:
    """Convert ScldoReportPayload into a JSON-serializable dictionary."""
    return {
        "audit": asdict(payload.audit),
        "temporal_benchmark": {
            "best_clean_model_name": payload.benchmark.best_clean_model_name,
            "best_clean_calibration": payload.benchmark.best_clean_calibration,
            "cv_fold_summaries": list(payload.benchmark.cv_fold_summaries),
            "records": [
                _serialize_model_record(rec) for rec in payload.benchmark.records
            ],
        },
        "shap_drivers": {
            "sample_size": payload.shap_summary.sample_size,
            "drivers": [asdict(d) for d in payload.shap_summary.drivers],
        },
        "causal_review_impact": {
            "sample_size": payload.causal_impact.sample_size,
            "bootstrap_replicates": payload.causal_impact.bootstrap_replicates,
            "confounders_used": list(payload.causal_impact.confounders_used),
            "buckets": [asdict(b) for b in payload.causal_impact.buckets],
        },
        "sla_policy": {
            "cost_ratio_fn_to_fp": payload.sla_policy.cost_ratio_fn_to_fp,
            "proactive_cost": payload.sla_policy.proactive_cost,
            "unmitigated_loss": payload.sla_policy.unmitigated_loss,
            "theoretical_bayes_threshold": payload.sla_policy.theoretical_bayes_threshold,
            "no_action_cost_per_order": payload.sla_policy.no_action_cost_per_order,
            "default_policy": asdict(payload.sla_policy.default_policy),
            "bayes_policy": asdict(payload.sla_policy.bayes_policy),
            "empirical_optimal_policy": asdict(
                payload.sla_policy.empirical_optimal_policy
            ),
            "sweep": [asdict(pt) for pt in payload.sla_policy.sweep],
        },
    }


def export_results_json(payload: ScldoReportPayload, output_path: Path) -> Path:
    """Write the complete serialized SCLDO metrics report to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_scldo_payload(payload)
    output_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
