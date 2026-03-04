# ==========================================================
# RETAIL SALES — HOLT-WINTERS FORECAST
# ==========================================================

"""
GOAL
----
Model retail sales using Holt-Winters (Triple Exponential Smoothing),
which accounts for:

- Level
- Trend
- Seasonality (12 months)

This model is appropriate when:
- Strong repeating yearly pattern exists
- There is a persistent upward trend
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

plt.style.use("ggplot")


# -----------------------
# Load data
# -----------------------

df = pd.read_csv(
    "data/retail_sales_rsxfsn_monthly.csv",
    parse_dates=["date"],
    index_col="date"
)

y = df["retail_sales_nsa"].asfreq("MS")

test_size = 24
train = y.iloc[:-test_size]
test = y.iloc[-test_size:]


# -----------------------
# Fit Holt-Winters
# -----------------------

"""
We use additive trend + additive seasonality.

Additive is appropriate when:
- Seasonal fluctuations are roughly constant in magnitude.

If seasonal amplitude grows with level,
multiplicative could be tested later.
"""

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)

fit = model.fit()

forecast = fit.forecast(test_size)
forecast.index = test.index


# -----------------------
# Metrics
# -----------------------

def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

hw_mape = mape(test, forecast)

print("\n=== Holt-Winters Forecast ===")
print(f"MAPE: {hw_mape:.2f}%")

# -----------------------
# Plot
# -----------------------

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Holt-Winters")

plt.title("Retail Sales — Holt-Winters Forecast")
plt.legend()
plt.tight_layout()
plt.savefig("data/retail_sales_holt_winters_plot.png", dpi=300)
plt.close()

print("Saved plot to: data/retail_sales_holt_winters_plot.png")