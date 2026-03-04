# ==========================================================
# S&P 500 NAIVE FORECAST
# ==========================================================

"""
GOAL OF THIS SCRIPT
-------------------

Demonstrate when a Naive forecast is appropriate.

We use:
- Daily S&P 500 closing prices

Why S&P 500?
- Financial asset prices are often modeled as a random walk.
- Tomorrow ≈ Today is historically difficult to beat.
- Weak stable seasonality.
- Strong short-term persistence.

The naive model serves as both:
1) A baseline benchmark
2) A theoretically justified model under Efficient Market Hypothesis
"""


# -----------------------
# Imports
# -----------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")


# -----------------------
# Load Data
# -----------------------

"""
We assume you already downloaded:
data/sp500_daily.csv

The dataset contains:
- Daily closing prices
- No missing calendar days (we forward-filled non-trading days)
"""

sp500 = pd.read_csv(
    "data/sp500_daily.csv",
    parse_dates=["date"],
    index_col="date"
)

# Rename index for clarity
sp500.index.name = "date"

# Target variable
y = sp500["sp500"]


# -----------------------
# Train/Test Split
# -----------------------

"""
We simulate real forecasting:

- Train on historical data
- Predict the next 90 days
- Compare against actual values

Time series MUST be split chronologically.
"""

test_size = 90
train = y.iloc[:-test_size]
test = y.iloc[-test_size:]


# -----------------------
# Naive Forecast
# -----------------------

"""
Naive assumption:
Tomorrow = Today

Implementation:
- Forecast all test days as the last observed training value
"""

naive_forecast = pd.Series(
    train.iloc[-1],
    index=test.index
)


# -----------------------
# Evaluation Metrics
# -----------------------

"""
We use:

MAPE  → Percentage error (scale-free)
MAE   → Absolute error in index points
RMSE  → Penalizes large misses more strongly
"""

def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))

def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))


naive_mape = mape(test, naive_forecast)
naive_mae = mae(test, naive_forecast)
naive_rmse = rmse(test, naive_forecast)

print("\n=== S&P 500 Naive Forecast Performance ===")
print(f"MAPE: {naive_mape:.2f}%")
print(f"MAE: {naive_mae:.2f}")
print(f"RMSE: {naive_rmse:.2f}")


# -----------------------
# Visualization
# -----------------------

"""
We visualize:
- Training history
- Actual test period
- Naive forecast (flat line)

This makes clear:
- Whether the market trends upward/downward during the test window
- How naive reacts to regime shifts
"""

plt.figure(figsize=(12, 6))

plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, naive_forecast, label="Naive Forecast")

plt.title("S&P 500 — Naive Forecast")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.legend()
plt.tight_layout()

plt.savefig("data/sp500_naive_plot.png", dpi=300)
plt.close()

print("Saved plot to: data/sp500_naive_plot.png")


# -----------------------
# Interpretation Notes
# -----------------------

"""
If naive performs well (low MAPE):

It supports the hypothesis that:
- S&P behaves close to a random walk
- Short-term forecasting is extremely difficult
- Additional model complexity may not yield meaningful gains

This is consistent with:
- Efficient Market Hypothesis
- Empirical finance literature
"""