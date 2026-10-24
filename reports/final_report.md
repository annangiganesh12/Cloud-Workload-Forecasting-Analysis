# Final Report: Cloud Workload Forecasting & Analysis

## 1. Executive Summary
This report details the development of a near-term aggregate demand forecasting pipeline for a curated subset of the Azure Public Dataset V2. The goal was to predict the fleet-level `active_vm_count` and `aggregate_cpu_utilization` over a 2-hour holdout horizon.

Our pipeline successfully outperformed the persistence baseline for CPU utilization, achieving a **7.04% improvement in MAE** (reducing error from 0.1105% to 0.1027%). The fleet size remained perfectly stable during the evaluation window, resulting in zero error for the VM count forecast.

## 2. Data Understanding & Preprocessing
The dataset consists of 10,000 VM traces with observations every five minutes over approximately 11.1 hours.

### Key Observations:
- **Timestamp Cadence:** Data is sampled every 300 seconds. A deep-dive audit confirmed a perfectly consistent cadence across the entire dataset with zero irregularities.
- **Data Completeness:** The dataset is exceptionally clean; 100% of VMs (10,000/10,000) have a complete set of 134 readings, with zero missing values or duplicates.
- **CPU Utilization:** Fleet-wide CPU readings are strictly bounded within [0, 100]. The global mean utilization is $\approx 9.39\%$, with a standard deviation of $14.27\%$.
- **Target Construction:** We aggregated VM-level `avg_cpu` into a fleet-level mean and counted unique `vm_id`s per timestamp to create the target series.
- **Preprocessing:** Because the data was found to be complete, no imputation was required, ensuring the "active fleet" definition remained perfectly consistent.

## 3. Workload Analysis
The analyzed workload is characterized by high stability and low volatility at the aggregate level.

### Aggregate Behavior:
- **Fleet Stability:** The `active_vm_count` is near-constant, indicating low churn during the observed window.
- **CPU Distribution:** The aggregate CPU utilization fluctuates within a very narrow band (between **8.98% and 9.91%**), suggesting a steady-state workload.
- **VM-Level Variance:** While the aggregate signal is stable, individual VMs exhibit significantly higher volatility (global $\sigma \approx 14.27\%$), with peaks reaching **99.49%**.

### Segment-Level Insights:
We analyzed the workload across different VM sizes (`virtual_core_bucket`):
- **Small VMs (2 cores):** The most dominant segment, with a mean CPU of **9.72%**.
- **Large VMs (>24 cores):** The most under-utilized and stable segment, with a lower mean CPU of **6.62%** and the lowest volatility ($\sigma \approx 8.63\%$).
- **Operational Insight:** The inverse relationship between VM size and utilization suggests that larger instances in this trace are over-provisioned relative to their current load.

### Operational Interpretation:
The high stability of the aggregate signal explains why the Persistence Baseline performs so well. However, the disparity between segment behaviors implies that a "one-size-fits-all" headroom policy may be inefficient; larger VM buckets could potentially sustain tighter buffers than smaller, more volatile ones.

## 4. Forecasting Approach
We implemented a supervised learning pipeline using autoregressive features.

### Feature Engineering:
To prevent data leakage and capture momentum, we used:
- **Lagged Targets:** Values from $t-1, t-2,$ and $t-3$ for both targets.
- **Rolling Statistics:** A 3-period rolling mean of CPU utilization to smooth short-term noise.
- **Temporal Index:** Observation sequence to capture the overall trend.

### Model Selection:
We employed a **RandomForestRegressor**. This model was chosen for its ability to handle non-linear relationships between lags without requiring strict stationarity assumptions, unlike ARIMA models.

## 5. Validation Methodology
To ensure the model is "defensible," we followed a strict temporal validation protocol:
- **Temporal Split:** The first 110 observations were used for training; the final 24 (2 hours) were held out as a blind test set.
- **No Leakage:** Features for any timestamp $T$ were derived exclusively from timestamps $< T$.
- **Baseline Comparison:** Results were compared against a **Persistence Baseline** (predicting $t+1 = t$), which is the industry standard for short-term, stable workloads.
- **Unit Testing:** A comprehensive `pytest` suite was implemented to verify the pipeline. All tests passed, confirming the correctness of:
    - **Aggregation Logic**: Correct fleet-level count and mean calculations.
    - **Lag Integrity**: Verified that features are shifted correctly without future leakage.
    - **Model Invariants**: Verified that the model output remains within the logical [0, 100] range and maintains deterministic behavior.

## 6. Results vs Baseline

| Target | Baseline MAE | ML Model MAE | Improvement |
| :--- | :--- | :--- | :--- |
| `active_vm_count` | 0.0000 | 0.0000 | 0.00% |
| `aggregate_cpu_utilization` | 0.1105% | 0.1027% | **7.04%** |

The ML model consistently outperformed the naive forecast for CPU utilization, proving that the recent trend and rolling average provide predictive value beyond the last observed value.

## 7. Capacity-Planning Recommendation
Based on the low forecast error (MAE $\approx 0.1\%$) and the stability of the workload, we recommend the following operational policy:

**Recommendation: Implement a 2% Safety Headroom Policy.**

- **The Policy:** Maintain physical capacity equal to $\text{Forecasted Demand} + 2\%$.
- **Trade-off Analysis:**
    - **Under-provisioning Risk:** With an RMSE of 0.1386%, a 2% buffer covers the vast majority of potential forecast misses, maintaining high availability.
    - **Over-provisioning Waste:** Because the workload is so stable, a larger buffer (e.g., 10%) would result in significant wasted capital and energy without providing meaningful additional safety.
- **Trigger:** If the `aggregate_cpu_utilization` forecast exceeds 80%, trigger an automated review of, the `p95_max_cpu` of the most active VMs to check for impending saturation.

## 8. Limitations & Future Work
- **Temporal Window:** The 11-hour window precludes the analysis of daily or weekly seasonality.
- **Model Complexity:** While RandomForest performed well, a simpler Ridge regression might offer similar accuracy with better interpretability.
- **Future Work:** Integrating `virtual_core_bucket` weights into the aggregate target would move the forecast from "Percentage" to "Absolute Core Demand", providing higher operational utility for physical host allocation.
