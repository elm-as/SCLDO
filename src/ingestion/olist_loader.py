"""Validated loader for all nine Olist Brazilian e-commerce relational CSV tables."""

from pathlib import Path

import pandas as pd

from src.core.contracts import OlistRawTables

_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "olist_orders_dataset.csv": (
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ),
    "olist_order_items_dataset.csv": (
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ),
    "olist_order_payments_dataset.csv": (
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ),
    "olist_order_reviews_dataset.csv": (
        "review_id",
        "order_id",
        "review_score",
        "review_creation_date",
    ),
    "olist_products_dataset.csv": (
        "product_id",
        "product_category_name",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ),
    "olist_sellers_dataset.csv": (
        "seller_id",
        "seller_zip_code_prefix",
        "seller_state",
    ),
    "olist_customers_dataset.csv": (
        "customer_id",
        "customer_zip_code_prefix",
        "customer_state",
    ),
    "olist_geolocation_dataset.csv": (
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    ),
    "product_category_name_translation.csv": (
        "product_category_name",
        "product_category_name_english",
    ),
}


def _load_validated_csv(raw_dir: Path, filename: str) -> pd.DataFrame:
    file_path = raw_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required Olist table: {file_path}")
    frame = pd.read_csv(file_path)
    missing_cols = set(_REQUIRED_FILES[filename]) - set(frame.columns)
    if missing_cols:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing_cols)}")
    return frame


def load_olist_raw_tables(raw_data_dir: Path) -> OlistRawTables:
    """Load and schema-validate all nine raw Olist CSV files from raw_data_dir."""
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    return OlistRawTables(
        orders=_load_validated_csv(raw_data_dir, "olist_orders_dataset.csv"),
        order_items=_load_validated_csv(raw_data_dir, "olist_order_items_dataset.csv"),
        order_payments=_load_validated_csv(
            raw_data_dir, "olist_order_payments_dataset.csv"
        ),
        order_reviews=_load_validated_csv(
            raw_data_dir, "olist_order_reviews_dataset.csv"
        ),
        products=_load_validated_csv(raw_data_dir, "olist_products_dataset.csv"),
        sellers=_load_validated_csv(raw_data_dir, "olist_sellers_dataset.csv"),
        customers=_load_validated_csv(raw_data_dir, "olist_customers_dataset.csv"),
        geolocation=_load_validated_csv(raw_data_dir, "olist_geolocation_dataset.csv"),
        category_translation=_load_validated_csv(
            raw_data_dir, "product_category_name_translation.csv"
        ),
    )
