"""Estimator factory for logistic, HistGradientBoosting, and LightGBM models."""

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.core.config import ModelHyperparamConfig


def create_supply_chain_estimator(
    model_name: str, hyperparams: ModelHyperparamConfig
) -> Any:
    """Instantiate an unfitted binary classifier for the requested model_name."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=hyperparams.logreg_max_iter,
            random_state=hyperparams.random_state,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=hyperparams.hgb_max_iter,
            learning_rate=hyperparams.hgb_learning_rate,
            max_leaf_nodes=hyperparams.hgb_max_leaf_nodes,
            min_samples_leaf=40,
            random_state=hyperparams.random_state,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=hyperparams.lgbm_n_estimators,
            learning_rate=hyperparams.lgbm_learning_rate,
            num_leaves=hyperparams.lgbm_num_leaves,
            min_child_samples=hyperparams.lgbm_min_child_samples,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            random_state=hyperparams.random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model name: {model_name}")
