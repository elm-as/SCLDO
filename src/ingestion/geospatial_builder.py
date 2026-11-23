"""Brazil-bounded geolocation deduplication and Haversine logistics distances."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

_BRAZIL_LAT_MIN = -34.5
_BRAZIL_LAT_MAX = 6.0
_BRAZIL_LNG_MIN = -74.5
_BRAZIL_LNG_MAX = -34.0
_EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class CoordinatePairs:
    """Origin and destination geographic coordinates in decimal degrees."""

    origin_lat: np.ndarray
    origin_lng: np.ndarray
    dest_lat: np.ndarray
    dest_lng: np.ndarray


def build_zip_median_coordinates(geolocation_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate Olist geolocation records into Brazil-bounded median lat/lng per zip."""
    geo = geolocation_df[
        ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
    ].copy()
    valid_lat = geo["geolocation_lat"].between(_BRAZIL_LAT_MIN, _BRAZIL_LAT_MAX)
    valid_lng = geo["geolocation_lng"].between(_BRAZIL_LNG_MIN, _BRAZIL_LNG_MAX)
    cleaned = geo.loc[valid_lat & valid_lng].dropna()
    grouped = (
        cleaned.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(geo_lat=("geolocation_lat", "median"), geo_lng=("geolocation_lng", "median"))
    )
    grouped["geolocation_zip_code_prefix"] = grouped[
        "geolocation_zip_code_prefix"
    ].astype(int)
    return grouped


def compute_haversine_km(coords: CoordinatePairs) -> np.ndarray:
    """Compute vectorized great-circle Haversine distance in kilometers."""
    lat1 = np.radians(coords.origin_lat.astype(float))
    lng1 = np.radians(coords.origin_lng.astype(float))
    lat2 = np.radians(coords.dest_lat.astype(float))
    lng2 = np.radians(coords.dest_lng.astype(float))

    dlat = lat2 - lat1
    dlng = lng2 - lng1
    sin_half_lat = np.sin(dlat / 2.0)
    sin_half_lng = np.sin(dlng / 2.0)
    hav = (
        sin_half_lat * sin_half_lat
        + np.cos(lat1) * np.cos(lat2) * sin_half_lng * sin_half_lng
    )
    clipped = np.clip(hav, 0.0, 1.0)
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(clipped))
