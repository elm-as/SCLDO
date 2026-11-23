"""Relational assembly of all nine Olist tables into a leak-free order dataset."""

import numpy as np
import pandas as pd

from src.core.config import TemporalSplitConfig
from src.core.contracts import IngestionAuditStats, OlistRawTables
from src.ingestion.geospatial_builder import (
    CoordinatePairs,
    build_zip_median_coordinates,
    compute_haversine_km,
)
from src.ingestion.seller_history_builder import build_strict_past_seller_features

_TIMESTAMP_COLS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)


def _aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    items = order_items.copy()
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    sorted_items = items.sort_values(
        ["order_id", "price", "order_item_id"], ascending=[True, False, True]
    )
    primary = sorted_items.drop_duplicates(subset=["order_id"], keep="first")[
        ["order_id", "seller_id", "product_id", "shipping_limit_date"]
    ].rename(
        columns={
            "seller_id": "primary_seller_id",
            "product_id": "primary_product_id",
        }
    )
    agg = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        unique_product_count=("product_id", "nunique"),
        unique_seller_count=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )
    merged = agg.merge(primary, on="order_id", how="inner")
    merged["freight_ratio"] = merged["total_freight"] / (merged["total_price"] + 1.0)
    return merged


def _aggregate_order_payments(order_payments: pd.DataFrame) -> pd.DataFrame:
    payments = order_payments.copy()
    primary = (
        payments.sort_values(
            ["order_id", "payment_value", "payment_sequential"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["order_id"], keep="first")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "primary_payment_type"})
    )
    agg = payments.groupby("order_id", as_index=False).agg(
        payment_value_sum=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_count=("payment_sequential", "count"),
    )
    return agg.merge(primary, on="order_id", how="left")


def _aggregate_order_reviews(order_reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = order_reviews.copy()
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])
    return (
        reviews.sort_values(["order_id", "review_creation_date"])
        .drop_duplicates(subset=["order_id"], keep="first")[["order_id", "review_score"]]
    )


def _enrich_products(
    products: pd.DataFrame, translation: pd.DataFrame
) -> pd.DataFrame:
    merged = products.merge(translation, on="product_category_name", how="left")
    merged["product_category_english"] = (
        merged["product_category_name_english"]
        .fillna(merged["product_category_name"])
        .fillna("unknown")
    )
    merged["product_volume_cm3"] = (
        merged["product_length_cm"]
        * merged["product_height_cm"]
        * merged["product_width_cm"]
    )
    return merged[
        [
            "product_id",
            "product_category_english",
            "product_weight_g",
            "product_volume_cm3",
            "product_photos_qty",
            "product_description_lenght",
        ]
    ].rename(
        columns={
            "product_id": "primary_product_id",
            "product_description_lenght": "product_description_length",
        }
    )


def _attach_geospatial_features(
    frame: pd.DataFrame, zip_coords: pd.DataFrame
) -> pd.DataFrame:
    enriched = frame.merge(
        zip_coords.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "geo_lat": "seller_lat",
                "geo_lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    ).merge(
        zip_coords.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "geo_lat": "customer_lat",
                "geo_lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    coords = CoordinatePairs(
        origin_lat=enriched["seller_lat"].to_numpy(),
        origin_lng=enriched["seller_lng"].to_numpy(),
        dest_lat=enriched["customer_lat"].to_numpy(),
        dest_lng=enriched["customer_lng"].to_numpy(),
    )
    enriched["distance_km"] = compute_haversine_km(coords)
    enriched["is_same_state"] = (
        enriched["seller_state"] == enriched["customer_state"]
    ).astype(float)
    return enriched


def _attach_temporal_and_ablation_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    purchase = enriched["order_purchase_timestamp"]
    estimated = enriched["order_estimated_delivery_date"]
    delivered = enriched["order_delivered_customer_date"]
    approved = enriched["order_approved_at"]
    carrier = enriched["order_delivered_carrier_date"]

    enriched["delay_days"] = (delivered - estimated).dt.total_seconds() / 86400.0
    enriched["is_late"] = (delivered > estimated).astype(int)
    enriched["promised_lead_days"] = (estimated - purchase).dt.total_seconds() / 86400.0
    enriched["shipping_limit_lead_days"] = (
        enriched["shipping_limit_date"] - purchase
    ).dt.total_seconds() / 86400.0
    enriched["approval_lag_hours"] = (
        ((approved - purchase).dt.total_seconds() / 3600.0).clip(lower=0.0)
    )
    enriched["purchase_month"] = purchase.dt.month.astype(float)
    enriched["purchase_dayofweek"] = purchase.dt.dayofweek.astype(float)
    enriched["purchase_hour"] = purchase.dt.hour.astype(float)
    enriched["is_weekend_purchase"] = purchase.dt.dayofweek.isin([5, 6]).astype(float)
    enriched["is_q4_peak"] = purchase.dt.month.isin([11, 12]).astype(float)

    # Explicitly segregated post-carrier columns used ONLY in the leakage ablation study.
    enriched["carrier_handoff_days"] = (
        (carrier - purchase).dt.total_seconds() / 86400.0
    )
    enriched["carrier_delay_vs_limit_days"] = (
        (carrier - enriched["shipping_limit_date"]).dt.total_seconds() / 86400.0
    )
    enriched["remaining_promised_days_after_carrier"] = (
        (estimated - carrier).dt.total_seconds() / 86400.0
    )
    return enriched


def build_olist_order_dataset(
    tables: OlistRawTables, temporal_config: TemporalSplitConfig
) -> tuple[pd.DataFrame, IngestionAuditStats]:
    """Assemble all nine Olist tables into a chronologically sorted order-level frame.

    Filters strictly to delivered orders with valid delivery and estimated dates,
    attaches Haversine logistics distance and strictly past-only seller history, and
    returns both the modeling DataFrame and an IngestionAuditStats integrity record.
    """
    raw_orders = tables.orders.copy()
    total_raw_orders = len(raw_orders)
    delivered = raw_orders.loc[raw_orders["order_status"] == "delivered"].copy()
    delivered_count = len(delivered)

    for col in _TIMESTAMP_COLS:
        delivered[col] = pd.to_datetime(delivered[col])
    valid_orders = delivered.dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    ).copy()

    items_agg = _aggregate_order_items(tables.order_items)
    payments_agg = _aggregate_order_payments(tables.order_payments)
    reviews_agg = _aggregate_order_reviews(tables.order_reviews)
    products_clean = _enrich_products(tables.products, tables.category_translation)
    zip_coords = build_zip_median_coordinates(tables.geolocation)

    frame = (
        valid_orders.merge(items_agg, on="order_id", how="inner")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews_agg, on="order_id", how="left")
        .merge(products_clean, on="primary_product_id", how="left")
        .merge(
            tables.sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
            left_on="primary_seller_id",
            right_on="seller_id",
            how="left",
        )
        .merge(
            tables.customers[
                ["customer_id", "customer_zip_code_prefix", "customer_state"]
            ],
            on="customer_id",
            how="left",
        )
    )
    frame = _attach_geospatial_features(frame, zip_coords)
    frame = _attach_temporal_and_ablation_features(frame)
    frame = build_strict_past_seller_features(frame, temporal_config)

    cutoff_ts = pd.Timestamp(temporal_config.oot_cutoff_iso)
    train_mask = frame["order_purchase_timestamp"] <= cutoff_ts
    test_mask = ~train_mask

    audit = IngestionAuditStats(
        total_raw_orders=total_raw_orders,
        delivered_orders=delivered_count,
        valid_delivered_orders=len(frame),
        orders_with_reviews=int(frame["review_score"].notna().sum()),
        overall_late_count=int(frame["is_late"].sum()),
        overall_late_rate=float(frame["is_late"].mean()),
        train_orders=int(train_mask.sum()),
        train_late_rate=float(frame.loc[train_mask, "is_late"].mean()),
        oot_test_orders=int(test_mask.sum()),
        oot_test_late_rate=float(frame.loc[test_mask, "is_late"].mean()),
        oot_cutoff_iso=temporal_config.oot_cutoff_iso,
        min_purchase_iso=str(frame["order_purchase_timestamp"].min()),
        max_purchase_iso=str(frame["order_purchase_timestamp"].max()),
        unique_sellers=int(frame["primary_seller_id"].nunique()),
        unique_customer_states=int(frame["customer_state"].nunique()),
        valid_geo_zip_prefixes=len(zip_coords),
        distance_coverage_rate=float(frame["distance_km"].notna().mean()),
    )
    return frame, audit
