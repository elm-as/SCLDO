"""Asymmetric 8:1 proactive SLA intervention threshold optimizer."""

import numpy as np

from src.core.config import SlaEconomicsConfig
from src.core.contracts import SlaPolicySummary, SlaThresholdPoint


def evaluate_sla_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, params: tuple[float, float, float]
) -> SlaThresholdPoint:
    """Evaluate operational metrics and expected SLA cost at one decision threshold.

    params is a 3-tuple of (threshold, proactive_intervention_cost, unmitigated_late_loss).
    Every flagged order incurs proactive_intervention_cost; every unflagged late order
    incurs unmitigated_late_loss.
    """
    threshold, c_proactive, c_late_loss = params
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(probabilities, dtype=float)
    flagged = p_arr >= threshold

    n_total = len(y_arr)
    tp = int(np.sum(flagged & (y_arr == 1)))
    fp = int(np.sum(flagged & (y_arr == 0)))
    fn = int(np.sum((~flagged) & (y_arr == 1)))
    tn = int(np.sum((~flagged) & (y_arr == 0)))

    recall = float(tp / max(tp + fn, 1))
    precision = float(tp / max(tp + fp, 1))
    specificity = float(tn / max(tn + fp, 1))
    flagged_rate = float((tp + fp) / max(n_total, 1))

    # Proactive intervention avoids unmitigated churn/1-star loss on true positives
    # while incurring the intervention cost on every flagged shipment (TP + FP).
    total_cost = (tp + fp) * c_proactive + fn * c_late_loss
    expected_cost = float(total_cost / max(n_total, 1))
    no_action_cost = float(np.mean(y_arr) * c_late_loss)
    cost_10k = expected_cost * 10_000.0
    savings_10k = (no_action_cost - expected_cost) * 10_000.0

    return SlaThresholdPoint(
        threshold=float(round(threshold, 4)),
        recall=recall,
        precision=precision,
        specificity=specificity,
        flagged_rate=flagged_rate,
        expected_cost_per_order=expected_cost,
        cost_per_10k_orders=cost_10k,
        savings_vs_no_action_per_10k=savings_10k,
    )


def optimize_sla_intervention_policy(
    y_true: np.ndarray, probabilities: np.ndarray, config: SlaEconomicsConfig
) -> SlaPolicySummary:
    """Sweep thresholds to find the optimal proactive SLA policy under asymmetric costs.

    Under cost C_proactive for intervening and C_loss for an unmitigated late delivery,
    intervening is optimal whenever p * C_loss >= C_proactive, i.e. p >= C_proactive / C_loss.
    """
    c_fp = config.proactive_intervention_cost
    c_fn = config.unmitigated_late_loss
    if c_fp <= 0.0 or c_fn <= c_fp:
        raise ValueError("Require 0 < proactive_intervention_cost < unmitigated_late_loss.")

    # Intervene when expected avoided loss p * C_fn exceeds intervention cost C_fp.
    bayes_tau = float(c_fp / c_fn)
    grid = np.linspace(0.02, 0.80, config.threshold_grid_points)
    sweep = tuple(
        evaluate_sla_threshold(y_true, probabilities, (float(tau), c_fp, c_fn))
        for tau in grid
    )
    empirical_best = min(sweep, key=lambda pt: pt.expected_cost_per_order)
    default_pt = evaluate_sla_threshold(y_true, probabilities, (0.50, c_fp, c_fn))
    bayes_pt = evaluate_sla_threshold(y_true, probabilities, (bayes_tau, c_fp, c_fn))
    no_action_cost = float(np.mean(np.asarray(y_true, dtype=float)) * c_fn)

    return SlaPolicySummary(
        cost_ratio_fn_to_fp=float(c_fn / c_fp),
        proactive_cost=c_fp,
        unmitigated_loss=c_fn,
        theoretical_bayes_threshold=bayes_tau,
        no_action_cost_per_order=no_action_cost,
        default_policy=default_pt,
        bayes_policy=bayes_pt,
        empirical_optimal_policy=empirical_best,
        sweep=sweep,
    )
