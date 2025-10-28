"""Feature engineering pipeline for Hydro raw material forecasting."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANCHOR_DATE_DEFAULT = pd.Timestamp("2024-12-31")
ROLLING_WINDOWS = [7, 14, 28, 56, 84, 112, 168, 224]
TREND_WINDOWS = [30, 60, 90]
EWM_SPANS = [7, 14, 30, 90]
PURCHASE_BUCKETS: Sequence[tuple[int, int]] = (
    (1, 7),
    (8, 14),
    (15, 30),
    (31, 60),
    (61, 90),
    (91, 120),
    (121, 180),
    (181, 365),
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RawData:
    receivals: pd.DataFrame
    purchase_orders: pd.DataFrame
    materials: pd.DataFrame
    transportation: pd.DataFrame
    prediction_mapping: pd.DataFrame


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_raw_data(base_path: Path) -> RawData:
    """Load all CSV inputs into dataframes."""
    data_dir = base_path / "data"
    kernel_dir = data_dir / "kernel"
    extended_dir = data_dir / "extended"

    receivals = pd.read_csv(
        kernel_dir / "receivals.csv",
        parse_dates=["date_arrival"],
        dtype={"rm_id": "Int64", "supplier_id": "Int64"},
    )
    receivals["date_arrival"] = (
        pd.to_datetime(receivals["date_arrival"], utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
    )
    receivals["arrival_date"] = receivals["date_arrival"].dt.normalize()
    receivals["net_weight"] = receivals["net_weight"].fillna(0.0)
    receivals = receivals.dropna(subset=["rm_id", "arrival_date"])

    purchase_orders = pd.read_csv(
        kernel_dir / "purchase_orders.csv",
        parse_dates=["delivery_date", "created_date_time", "modified_date_time"],
    )
    purchase_orders["delivery_date"] = (
        pd.to_datetime(purchase_orders["delivery_date"], utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
    )
    purchase_orders["created_date_time"] = (
        pd.to_datetime(purchase_orders["created_date_time"], utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
    )

    materials = pd.read_csv(extended_dir / "materials.csv")
    transportation = pd.read_csv(
        extended_dir / "transportation.csv",
        dtype={"rm_id": "Int64"},
    )

    prediction_mapping = pd.read_csv(
        data_dir / "prediction_mapping.csv",
        parse_dates=["forecast_start_date", "forecast_end_date"],
    )

    return RawData(
        receivals=receivals,
        purchase_orders=purchase_orders,
        materials=materials,
        transportation=transportation,
        prediction_mapping=prediction_mapping,
    )


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def build_daily_receivals(
    receivals: pd.DataFrame,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate receivals to daily level and ensure continuous calendar per rm_id."""
    if start_date is None:
        start_date = min(receivals["arrival_date"].min(), end_date)

    rm_ids = receivals["rm_id"].unique()
    calendar = pd.date_range(start=start_date, end=end_date, freq="D")
    full_index = pd.MultiIndex.from_product(
        (rm_ids, calendar), names=["rm_id", "date"]
    )

    daily = (
        receivals.groupby(["rm_id", "arrival_date"], as_index=False)["net_weight"]
        .sum()
        .rename(columns={"arrival_date": "date"})
    )
    daily = (
        daily.set_index(["rm_id", "date"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .sort_values(["rm_id", "date"])
        .rename(columns={"level_0": "rm_id", "level_1": "date"})
    )
    daily["has_delivery"] = (daily["net_weight"] > 0).astype(int)
    daily["cumulative_net_weight"] = (
        daily.groupby("rm_id")["net_weight"].cumsum()
    )
    return daily


def _linear_trend(values: np.ndarray) -> float:
    """Compute slope of values against evenly spaced time index."""
    n_obs = values.size
    if n_obs == 0:
        return np.nan
    x = np.arange(n_obs, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((x - x_mean) * (values - y_mean))
    return numer / denom


def _days_since_last_delivery(series: pd.Series) -> pd.Series:
    days = []
    counter = np.nan
    for value in series.values:
        if value > 0:
            counter = 0
        else:
            counter = counter + 1 if not np.isnan(counter) else np.nan
        days.append(counter)
    return pd.Series(days, index=series.index, dtype=float)


def engineer_temporal_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics, frequency, and trend signals."""
    df = daily.set_index(["rm_id", "date"]).sort_index()
    group = df.groupby(level=0)

    # Rolling aggregations on net_weight
    for window in ROLLING_WINDOWS:
        rolling = group["net_weight"].rolling(window=window, min_periods=1)
        df[f"roll_sum_{window}d"] = rolling.sum().values
        df[f"roll_mean_{window}d"] = rolling.mean().values
        df[f"roll_std_{window}d"] = rolling.std().values
        df[f"roll_p10_{window}d"] = rolling.quantile(0.10).values
        df[f"roll_p20_{window}d"] = rolling.quantile(0.20).values
        df[f"roll_p50_{window}d"] = rolling.quantile(0.50).values

        freq_roll = group["has_delivery"].rolling(window=window, min_periods=1)
        df[f"delivery_freq_{window}d"] = freq_roll.mean().values

    # Coefficient of variation and skewness/kurtosis on mid-range window
    window_mid = 56
    df["roll_cv_56d"] = df[f"roll_std_{window_mid}d"] / (df[f"roll_mean_{window_mid}d"] + 1e-3)
    df["roll_skew_84d"] = (
        group["net_weight"].rolling(window=84, min_periods=10).skew().values
    )
    df["roll_kurt_84d"] = (
        group["net_weight"].rolling(window=84, min_periods=10).kurt().values
    )

    # Exponential moving averages
    for span in EWM_SPANS:
        df[f"ewm_mean_span{span}"] = (
            group["net_weight"].transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )
        df[f"ewm_std_span{span}"] = (
            group["net_weight"].transform(lambda s: s.ewm(span=span, adjust=False).std())
        )

    # Trend features
    for window in TREND_WINDOWS:
        df[f"trend_slope_{window}d"] = group["net_weight"].transform(
            lambda s: s.rolling(window=window, min_periods=max(5, window // 3)).apply(
                _linear_trend, raw=True
            )
        )

    df["trend_acceleration"] = df["trend_slope_30d"] - df["trend_slope_60d"]

    # Days since last delivery and zero streaks
    df["days_since_last_delivery"] = group["net_weight"].transform(
        _days_since_last_delivery
    )
    df["zero_streak_length"] = group["has_delivery"].transform(
        lambda s: s.groupby((s != 0).cumsum()).cumcount().astype(float)
    )

    df = df.reset_index()
    return df


def engineer_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar-derived features from the date index."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # Cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Period markers
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
    df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
    return df


def engineer_purchase_order_features(
    purchase_orders: pd.DataFrame,
    receivals: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame:
    """Summaries of future purchase orders and historical reliability."""
    # Map product_id to rm_id using receivals information
    mapping_source = (
        receivals.dropna(subset=["product_id", "rm_id"])[["product_id", "rm_id"]]
        .copy()
    )
    mapping_source["product_id"] = pd.to_numeric(
        mapping_source["product_id"], errors="coerce"
    )
    mapping_source = mapping_source.dropna(subset=["product_id"])

    def _mode_rm_id(values: pd.Series) -> int:
        mode = values.mode()
        return int(mode.iloc[0]) if not mode.empty else int(values.iloc[0])

    product_map = (
        mapping_source.groupby("product_id")["rm_id"].agg(_mode_rm_id).to_dict()
    )

    po = purchase_orders.copy()
    po["product_id"] = pd.to_numeric(po["product_id"], errors="coerce")
    po["rm_id"] = po["product_id"].map(product_map)
    po = po.dropna(subset=["rm_id"])
    po["rm_id"] = po["rm_id"].astype(int)

    # Lead time statistics from historical orders
    po["lead_time_days"] = (
        (po["delivery_date"] - po["created_date_time"]).dt.days.astype(float)
    )

    historical_mask = po["delivery_date"] <= anchor_date
    historical_orders = po.loc[historical_mask].copy()

    delivered = (
        receivals.loc[receivals["arrival_date"] <= anchor_date]
        .groupby(["purchase_order_id", "purchase_order_item_no", "rm_id"], as_index=False)[
            "net_weight"
        ]
        .sum()
    )

    historical_orders = historical_orders.merge(
        delivered,
        on=["purchase_order_id", "purchase_order_item_no", "rm_id"],
        how="left",
        suffixes=("_ordered", "_delivered"),
    )
    historical_orders["net_weight"] = historical_orders["net_weight"].fillna(0.0)
    historical_orders["reliability_ratio"] = (
        historical_orders["net_weight"]
        / historical_orders["quantity"].replace(0, np.nan)
    )

    reliability = historical_orders.groupby("rm_id").agg(
        po_reliability_mean=("reliability_ratio", "mean"),
        po_reliability_median=("reliability_ratio", "median"),
        po_reliability_std=("reliability_ratio", "std"),
        po_lead_time_mean=("lead_time_days", "mean"),
        po_lead_time_std=("lead_time_days", "std"),
    )

    # Future purchase summary
    future_mask = po["delivery_date"] > anchor_date
    future_orders = po.loc[future_mask].copy()
    future_orders["days_until_delivery"] = (
        (future_orders["delivery_date"] - anchor_date).dt.days.astype(int)
    )

    bucket_frames: List[pd.DataFrame] = []
    for lower, upper in PURCHASE_BUCKETS:
        mask = (
            future_orders["days_until_delivery"].between(lower, upper, inclusive="both")
        )
        bucket = (
            future_orders.loc[mask]
            .groupby("rm_id")
            .agg(
                **{
                    f"po_weight_{lower}_{upper}": ("quantity", "sum"),
                    f"po_count_{lower}_{upper}": ("quantity", "count"),
                }
            )
        )
        bucket_frames.append(bucket)

    if bucket_frames:
        future_summary = pd.concat(bucket_frames, axis=1).fillna(0.0)
    else:
        future_summary = pd.DataFrame()

    future_overall = future_orders.groupby("rm_id").agg(
        po_weight_future_total=("quantity", "sum"),
        po_future_orders=("quantity", "count"),
        po_future_weight_mean=("quantity", "mean"),
    )

    features = reliability.join(future_summary, how="outer").join(
        future_overall, how="outer"
    )
    return features.reset_index()


def engineer_metadata_features(
    materials: pd.DataFrame,
    receivals: pd.DataFrame,
    transportation: pd.DataFrame,
) -> pd.DataFrame:
    """Static features derived from metadata and supplier diversity."""
    # Materials metadata (mode + diversity)
    materials_clean = materials.dropna(subset=["rm_id"]).copy()
    materials_clean["rm_id"] = materials_clean["rm_id"].astype(int)

    def _mode(series: pd.Series) -> str:
        series = series.dropna()
        if series.empty:
            return "unknown"
        mode = series.mode()
        return str(mode.iloc[0]) if not mode.empty else "unknown"

    material_meta = materials_clean.groupby("rm_id").agg(
        raw_material_alloy_mode=("raw_material_alloy", _mode),
        raw_material_format_mode=("raw_material_format_type", _mode),
        stock_location_mode=("stock_location", _mode),
        raw_material_alloy_nunique=("raw_material_alloy", pd.Series.nunique),
        raw_material_format_nunique=(
            "raw_material_format_type",
            pd.Series.nunique,
        ),
    )

    # Supplier diversity
    supplier_stats = receivals.groupby("rm_id").agg(
        supplier_unique_count=("supplier_id", pd.Series.nunique),
        supplier_delivery_count=("supplier_id", "count"),
        delivery_total_weight=("net_weight", "sum"),
        delivery_avg_weight=("net_weight", "mean"),
    )

    # Transporter diversity
    transport_stats = transportation.groupby("rm_id").agg(
        transporter_unique_count=("transporter_name", pd.Series.nunique),
        transporter_vehicle_unique=("vehicle_no", pd.Series.nunique),
    )

    features = material_meta.join(supplier_stats, how="outer").join(
        transport_stats, how="outer"
    )

    # Encode categorical modes numerically for tree models
    for col in [
        "raw_material_alloy_mode",
        "raw_material_format_mode",
        "stock_location_mode",
    ]:
        codes, uniques = pd.factorize(features[col].fillna("unknown"))
        features[f"{col}_code"] = codes

    features = features.reset_index()
    return features


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_feature_snapshot(
    temporal_features: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame:
    """Pick the feature row corresponding to the anchor date for each rm_id."""
    snapshot = temporal_features.loc[
        temporal_features["date"] == anchor_date
    ].copy()
    snapshot = snapshot.drop(columns=["date", "has_delivery"], errors="ignore")
    snapshot = snapshot.fillna(0.0)
    return snapshot


def assemble_prediction_matrix(
    raw: RawData,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame:
    """Combine temporal, calendar, PO, and metadata features for predictions."""
    daily = build_daily_receivals(raw.receivals, end_date=anchor_date)
    temporal = engineer_temporal_features(daily)
    temporal = engineer_calendar_features(temporal)
    temporal_snapshot = build_feature_snapshot(temporal, anchor_date)

    po_features = engineer_purchase_order_features(
        raw.purchase_orders, raw.receivals, anchor_date
    )
    meta_features = engineer_metadata_features(
        raw.materials, raw.receivals, raw.transportation
    )

    master = (
        temporal_snapshot.merge(po_features, on="rm_id", how="left")
        .merge(meta_features, on="rm_id", how="left")
    )

    # Join with prediction mapping (horizon-specific features)
    mapping = raw.prediction_mapping.copy()
    mapping["horizon_days"] = (
        (mapping["forecast_end_date"] - mapping["forecast_start_date"]).dt.days + 1
    )
    mapping["horizon_weeks"] = mapping["horizon_days"] / 7.0
    mapping["forecast_end_month"] = mapping["forecast_end_date"].dt.month
    mapping["forecast_end_quarter"] = mapping["forecast_end_date"].dt.quarter

    dataset = mapping.merge(master, on="rm_id", how="left")

    # Conservative scaling heuristics suggestions
    dataset["suggested_shrink_factor"] = np.clip(
        1.0
        - (dataset.get("trend_slope_30d", 0.0) > 0).astype(float) * 0.05
        - dataset.get("roll_cv_56d", 0.0) * 0.02,
        0.7,
        1.0,
    )

    # Fill numeric NaNs with zeros, preserve categorical codes as integers
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(0.0)
    dataset["anchor_date"] = anchor_date
    return dataset


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate feature matrix for Hydro forecasting submissions.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=".",
        help="Project base directory containing the data folder.",
    )
    parser.add_argument(
        "--anchor-date",
        type=str,
        default=str(ANCHOR_DATE_DEFAULT.date()),
        help="Date (YYYY-MM-DD) to anchor features on (should be <= 2024-12-31).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/prediction_features.csv",
        help="Path to save the engineered feature matrix (CSV).",
    )
    return parser.parse_args(args=args)


def main(args: Sequence[str] | None = None) -> None:
    options = parse_args(args)
    base_path = Path(options.base_path).resolve()
    anchor_date = pd.Timestamp(options.anchor_date)

    raw = load_raw_data(base_path)
    dataset = assemble_prediction_matrix(raw, anchor_date)

    output_path = base_path / options.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(
        f"Feature matrix created for anchor {anchor_date.date()} "
        f"with shape {dataset.shape} → {output_path}"
    )


if __name__ == "__main__":
    main()
