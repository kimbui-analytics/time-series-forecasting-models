# ==========================================================
# CPI INFLATION — ARIMA CASE STUDY
# ==========================================================

"""
GOAL
----
Demonstrate when ARIMA is appropriate.

Dataset:
- Month-over-month CPI inflation (%)

Why ARIMA?
- Inflation often shows autocorrelation.
- The series is closer to stationary than CPI levels.
- ARIMA models lag dependence directly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

plt.style.use("ggplot")


# -----------------------
# Load Data
# -----------------------

df = pd.read_csv(
    "data/cpi_inflation_mom.csv",
    parse_dates=["date"],
    index_col="date"
)

y = df["inflation_mom"].asfreq("MS")


# -----------------------
# Train/Test Split
# -----------------------

test_size = 24
train = y.iloc[:-test_size]
test = y.iloc[-test_size:]


# -----------------------
# Stationarity Test
# -----------------------

print("\n=== ADF Test ===")
adf_result = adfuller(train.dropna())

print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("Series appears stationary.")
else:
    print("Series may not be stationary.")


# -----------------------
# Naive Baseline
# -----------------------

naive_forecast = pd.Series(train.iloc[-1], index=test.index)


# -----------------------
# Fit ARIMA
# -----------------------

"""
Start simple:
ARIMA(1,0,1)

We assume:
- d=0 (inflation already close to stationary)
"""

model = ARIMA(train, order=(1, 0, 1))
fit = model.fit()

forecast = fit.forecast(steps=test_size)
forecast.index = test.index


# -----------------------
# Metrics
# -----------------------

def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))

def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

naive_mape = mape(test, naive_forecast)
arima_mape = mape(test, forecast)

naive_mae = mae(test, naive_forecast)
arima_mae = mae(test, forecast)

naive_rmse = rmse(test, naive_forecast)
arima_rmse = rmse(test, forecast)

print("\n=== Forecast Comparison ===")
print(f"Naive MAPE: {naive_mape:.2f}%")
print(f"ARIMA(1,0,1) MAPE: {arima_mape:.2f}%")
print("Since MAPE divides by the actual value avg(abs((actual-forecast/actual)) and inflation mom is frequently close to 0, MAPE explodes and is inappropriate for near-zero data")
print("For series with near-zero values, use MAE or RMSE instead of MAPE")
print(f"Naive MAE: {naive_mae:.4f}")
print(f"ARIMA(1,0,1) MAE: {arima_mae:.4f}")
print(f"Naive RMSE: {naive_rmse:.4f}")
print(f"ARIMA(1,0,1) RMSE: {arima_rmse:.4f}")

# -----------------------
# Plot
# -----------------------

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, naive_forecast, label="Naive")
plt.plot(test.index, forecast, label="ARIMA(1,0,1)")

plt.title("CPI Inflation — Naive vs ARIMA")
plt.legend()
plt.tight_layout()
plt.savefig("data/cpi_arima_plot.png", dpi=300)
plt.close()

print("Saved plot to: data/cpi_arima_plot.png")