"""Strictly intra-fold feature selection and preprocessing for clean vs leaky regimes."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import FeatureSchemaConfig

CLEAN_REGIME = "pre_dispatch_clean"
LEAKY_REGIME = "post_carrier_leaky"


def get_feature_columns_for_regime(
    schema: FeatureSchemaConfig, regime: str
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) strictly enforcing regime boundaries."""
    if regime == CLEAN_REGIME:
        return list(schema.pre_dispatch_numeric), list(schema.pre_dispatch_categorical)
    if regime == LEAKY_REGIME:
        numeric = list(schema.pre_dispatch_numeric) + list(
            schema.post_carrier_leaky_numeric
        )
        return numeric, list(schema.pre_dispatch_categorical)
    raise ValueError(f"Unsupported feature regime: {regime}")


def build_supply_chain_preprocessor(
    schema: FeatureSchemaConfig, regime: str, scale_numeric: bool = False
) -> ColumnTransformer:
    """Create an unfitted ColumnTransformer for intra-fold imputation and encoding."""
    numeric_cols, categorical_cols = get_feature_columns_for_regime(schema, regime)
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def extract_clean_feature_frame(
    orders_df: pd.DataFrame, schema: FeatureSchemaConfig, regime: str
) -> pd.DataFrame:
    """Return a defensive copy containing only columns permitted by the active regime."""
    numeric_cols, categorical_cols = get_feature_columns_for_regime(schema, regime)
    selected = numeric_cols + categorical_cols
    missing = set(selected) - set(orders_df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    return orders_df[selected].copy()
