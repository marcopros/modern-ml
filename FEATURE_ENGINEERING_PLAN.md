# 🧪 Feature Engineering Blueprint

## 🎯 Objectives
- Boost Kaggle score by extracting richer temporal and cross-source signals while staying conservative (quantile 0.2).
- Produce modular feature sets that can feed LightGBM/CatBoost/XGBoost pipelines and ensembles.
- Preserve robustness: avoid leakage, control sparsity, and keep reproducible transformations.

## 📝 Data Sanity Checklist (Run Before Feature Work)
- Validate schema and datatypes for `receivals`, `purchase_orders`, `materials`, `transportation`.
- Confirm no duplicated `(rm_id, receival_date)` pairs after aggregation.
- Fill missing weights/dates with explicit zeros; flag inferred imputations for later use.
- Generate a continuous calendar per `rm_id` from 2018-01-01 to 2024-12-31 to simplify rolling calculations.

## 🧱 Core Feature Families

### 1. Temporal Aggregates (Receivals)
- Rolling windows per `rm_id`: 7/14/28/56/84/112/168/224 day sums, means, medians, std, skew, kurtosis.
- Exponential weighted moving averages with decay factors (0.9, 0.95, 0.98) for recency emphasis.
- Rolling quantiles (p05, p10, p20, p30, p50) to align with conservative objective.
- Rolling zero-streak length, days-since-last-delivery, and frequency (deliveries per 30/60/90 days).
- Trend estimators: simple slope via linear regression over last 30/60/90 days; second derivative for acceleration.

### 2. Lag & Momentum Bundle
- Direct lags of cumulative deliveries at 1, 3, 7, 14, 21, 30, 60, 90 days.
- Momentum ratios: `rolling_sum_14 / rolling_sum_60`, `lag_7 / (lag_7 + lag_30)`.
- Shock indicators: z-score of latest delivery against trailing window.

### 3. Calendar & Seasonality
- Encode `day_of_week`, `week_of_year`, `month`, `quarter` using sin/cos pairs.
- Holiday proximity flags (Norwegian + global trading holidays) ±3 days.
- Production cycle proxies: binary for last week of month, first week of quarter.

### 4. Purchase Order Intelligence
- Aggregate future PO weight by horizon buckets (0-7, 8-14, 15-30, 31-60, 61-90, 91+ days).
- Weighted PO reliability: historical ratio of delivered vs ordered within promised window per `supplier_id` and `rm_id`.
- Lead-time statistics: mean, median, std of `delivery_date - order_date`.
- Urgency signals: share of rush orders (promised window < 7 days).
- Derived PO-demand gap: difference between purchase order cumulative forecast and historical rolling supply.

### 5. Material Metadata & Transportation
- One-hot or target-encoded `material_family`, `material_group`, `source_site`, `destination_site`.
- Transportation mode frequency (ship, truck, rail) per material.
- Cost and transit time averages from `transportation.csv` if available.
- Supplier diversity index: Herfindahl score of suppliers per `rm_id`.

### 6. Interaction Layer
- Cross terms between `forecast_horizon` and key trends (e.g., slope × horizon).
- Multiplicative terms: `po_weight_bucket * recent_frequency` to capture PO impact conditioned on activity level.
- Thresholded features: indicator if rolling quantile exceeds PO-expected weight.

### 7. Risk & Conservatism Controls
- Under-supply probability proxy: share of historical days where cumulative < target.
- Conservative scaling factor suggestions: per material shrink factors learned on validation to cap overestimation.
- Exposure score: combination of volatility, PO pressure, and sparsity to rank risky predictions.

## ⚙️ Implementation Plan

1. **Build unified time grid** per `rm_id` using `receivals` expanded to daily data with forward fills.
2. **Generate feature blocks** in modular scripts/notebooks:
   - `features_temporal.py`
   - `features_purchase_orders.py`
   - `features_metadata.py`
   Each returns a `DataFrame` keyed by (`rm_id`, `date`).
3. **Join & prune**: merge blocks, handle multicollinearity (VIF / correlation > 0.95), cap feature count via variance and feature importance.
4. **Target alignment**: align engineered features with forecast horizons defined in `prediction_mapping.csv` (cumulative sums to horizon end).
5. **Validation strategy**: rolling-origin evaluation (e.g., train up to Oct 2024 → validate on Nov, repeat) to respect time order.
6. **Modeling**: start with CatBoost (quantile alpha=0.2) + LightGBM quantile, then blend with monotonic constraints on sensitive features.
7. **Post-processing**: enforce non-negativity, apply material-level conservative scaling from validation diagnostics.

## 🧪 Validation & Diagnostics
- Track under-prediction ratio per horizon bin; target ≈80%.
- Plot feature importances and SHAP for top materials to detect leakage or noisy signals.
- Maintain experiment log (e.g., `mlruns/` or simple CSV) with feature set version, validation QL, Kaggle score.

## 🛠️ Example Pipeline Snippet
```python
# pseudo-code
features = []
features.append(build_temporal_features(receivals_daily))
features.append(build_po_features(purchase_orders, calendar))
features.append(build_metadata_features(materials, transportation))

X = reduce(lambda left, right: left.join(right, how="left"), features)
X = clip_outliers(X, method="winsor", limits=(0.01, 0.99))
X = X.fillna(0)

train, valid = time_based_split(X, y, cutoff="2024-11-01")

model = CatBoostRegressor(
    loss_function="Quantile:alpha=0.2",
    task_type="GPU",
    depth=8,
    learning_rate=0.03,
    iterations=8000,
    subsample=0.7,
    random_strength=1.5,
    l2_leaf_reg=6
)
model.fit(train_X, train_y, eval_set=(valid_X, valid_y), early_stopping_rounds=300)
```

## 📈 Next Steps
1. Prototype temporal + PO feature blocks on a 50k-sample subset and evaluate delta on validation QL.
2. Graduate promising features to full dataset; monitor training time and GPU memory.
3. Tune CatBoost/LightGBM hyperparameters jointly with feature set (Optuna 300+ trials).
4. Export new submission (e.g., `submission_fe_v1.csv`) and compare Kaggle public LB vs current baseline.

**Target outcome:** Improved model (starting from `solution_v5_ultra_conservative.ipynb`) leveraging the enriched feature set, aiming to climb from rank 123/187 toward Top 80 on first iteration.
