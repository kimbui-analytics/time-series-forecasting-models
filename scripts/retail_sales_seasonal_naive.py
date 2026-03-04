# ==========================================================
# RETAIL SALES: NAIVE vs SEASONAL NAIVE FORECAST
# ==========================================================

"""
GOAL OF THIS SCRIPT
-------------------
Demonstrate when a Seasonal Naive forecast is appropriate.

Dataset:
- U.S. Retail Sales (Not Seasonally Adjusted) from FRED: RSXFSN
- Monthly frequency
- Strong yearly seasonality due to recurring holiday shopping patterns

Models:
1) Naive:
   Forecast next month = last observed month

2) Seasonal Naive (12-month seasonality):
   Forecast next month = same month last year

Why this dataset is ideal for Seasonal Naive:
- Monthly data => a natural seasonal period of 12
- The seasonal pattern repeats every year (especially Nov/Dec spikes)
- Using "Not Seasonally Adjusted" preserves the seasonal pattern so the model can learn it
"""

# -----------------------
# Imports (what + why)
# -----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")


# -----------------------
# Load data
# -----------------------

"""
We load the dataset created by scripts/download_naive_datasets.py:
- data/retail_sales_rsxfsn_monthly.csv

Columns:
- date (monthly)
- retail_sales_nsa (sales level; units are millions of dollars in FRED’s series)
"""

df = pd.read_csv(
    "data/retail_sales_rsxfsn_monthly.csv",
    parse_dates=["date"],
    index_col="date"
)

# Our target time series
y = df["retail_sales_nsa"].asfreq("MS")  # Month Start frequency

# Safety: if anything missing, interpolate (should be complete already)
if y.isna().any():
    y = y.interpolate(method="time")


# -----------------------
# Train/Test split
# -----------------------

"""
We hold out the last 24 months (2 years) as a test set.

Why 24 months?
- Seasonal naive uses a 12-month lag
- Holding out 24 months includes two full seasonal cycles
- This makes it easier to judge seasonal forecasting performance
"""

test_size = 24
train = y.iloc[:-test_size]
test = y.iloc[-test_size:]


# -----------------------
# Model 1: Naive forecast
# -----------------------

"""
Naive forecast:
- Predict every month in the future as the last observed training value.
- This ignores seasonality completely.
"""

naive_forecast = pd.Series(train.iloc[-1], index=test.index)


# -----------------------
# Model 2: Seasonal Naive (12-month)
# -----------------------

"""
Seasonal Naive forecast (period = 12):
- Predict each month using the value from the same month last year.

Implementation detail:
- train.shift(12) aligns each point with the value 12 months earlier
- We take the last `test_size` points to match the test horizon
- Then force the forecast index to match test dates exactly
"""

seasonal_period = 12
seasonal_naive_forecast = train.shift(seasonal_period).iloc[-test_size:]
seasonal_naive_forecast.index = test.index


# -----------------------
# Evaluation metrics
# -----------------------

"""
We compute:
- MAPE: mean absolute percentage error (easy to interpret)
- MAE: average absolute error in sales units
- RMSE: penalizes larger misses more heavily
"""

def mape(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def mae(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    return np.mean(np.abs(actual - forecast))

def rmse(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    return np.sqrt(np.mean((actual - forecast) ** 2))


naive_mape = mape(test, naive_forecast)
seasonal_mape = mape(test, seasonal_naive_forecast)

naive_mae = mae(test, naive_forecast)
seasonal_mae = mae(test, seasonal_naive_forecast)

naive_rmse = rmse(test, naive_forecast)
seasonal_rmse = rmse(test, seasonal_naive_forecast)


# -----------------------
# Print results
# -----------------------

print("\n=== Retail Sales (RSXFSN) — Naive vs Seasonal Naive ===")
print("Holdout window:", test.index.min().date(), "to", test.index.max().date())

print("\nNaive Forecast:")
print(f"MAPE: {naive_mape:.2f}% | MAE: {naive_mae:.2f} | RMSE: {naive_rmse:.2f}")

print("\nSeasonal Naive (12-month) Forecast:")
print(f"MAPE: {seasonal_mape:.2f}% | MAE: {seasonal_mae:.2f} | RMSE: {seasonal_rmse:.2f}")

if seasonal_mape < naive_mape:
    print("\nInterpretation: Strong yearly seasonality exists (seasonal naive beats naive).")
else:
    print("\nInterpretation: Yearly seasonality may be weak/unstable (naive beats seasonal naive).")


# -----------------------
# Save metrics for README
# -----------------------

results = pd.DataFrame(
    [
        {"model": "Naive", "MAPE": naive_mape, "MAE": naive_mae, "RMSE": naive_rmse},
        {"model": "Seasonal Naive (12)", "MAPE": seasonal_mape, "MAE": seasonal_mae, "RMSE": seasonal_rmse},
    ]
).sort_values("MAPE")

results.to_csv("data/retail_sales_naive_metrics.csv", index=False)
print("\nSaved metrics to: data/retail_sales_naive_metrics.csv")


# -----------------------
# Visualization
# -----------------------

"""
Plot:
- Train history
- Actual test values
- Naive forecast (flat line)
- Seasonal naive forecast (tracks last year's seasonal shape)

This plot is usually the clearest way to see the value of seasonality.
"""

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, naive_forecast, label="Naive")
plt.plot(test.index, seasonal_naive_forecast, label="Seasonal Naive (12-month)")

plt.title("Retail Sales (RSXFSN) — Naive vs Seasonal Naive")
plt.xlabel("Date")
plt.ylabel("Retail Sales (Not Seasonally Adjusted)")
plt.legend()
plt.tight_layout()

plt.savefig("data/retail_sales_naive_vs_seasonal_naive_plot.png", dpi=300)
plt.close()

print("Saved plot to: data/retail_sales_naive_vs_seasonal_naive_plot.png")
