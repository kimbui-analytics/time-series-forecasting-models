import os
import pandas as pd
import yfinance as yf

"""
Downloads two datasets for demonstrating naive vs seasonal naive forecasting:

1) S&P 500 daily close (naive is hard to beat)
2) Retail Sales (seasonal naive performs well)

Saves outputs to:
- data/sp500_daily.csv
- data/retail_sales_rsxfsn_monthly.csv
"""

# Ensure output folder exists (important for fresh clones)
os.makedirs("data", exist_ok=True)

# -----------------------
# 1) S&P 500 (daily)
# -----------------------
print("Downloading S&P 500 data...")

sp500_raw = yf.download("^GSPC", start="2010-01-01", progress=False)

# If MultiIndex columns exist (new yfinance behavior), flatten them
if isinstance(sp500_raw.columns, pd.MultiIndex):
    sp500_raw.columns = sp500_raw.columns.get_level_values(0)

# Use Close price
sp500 = sp500_raw[["Close"]].rename(columns={"Close": "sp500"}).sort_index()

# Fill missing calendar days (weekends/holidays) by forward-filling
sp500 = sp500.asfreq("D").ffill()

# Save clean CSV with explicit index label
sp500.to_csv("data/sp500_daily.csv", index_label="date")

print("Saved: data/sp500_daily.csv")
print(sp500.head())

# -----------------------
# 2) Retail Sales (monthly) — FRED RSXFSN (Not Seasonally Adjusted)
# -----------------------
print("\nDownloading Retail Sales (FRED RSXFSN) dataset...")

SERIES_ID = "RSXFSN"
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}"

retail = pd.read_csv(url)
retail.columns = ["date", "value"]
retail["date"] = pd.to_datetime(retail["date"])

# Coerce to numeric in case missing values appear as '.'
y = retail.set_index("date")["value"].asfreq("MS")  # month start
y = pd.to_numeric(y, errors="coerce").dropna()

retail_out = y.to_frame(name="retail_sales_nsa")
retail_out.to_csv("data/retail_sales_rsxfsn_monthly.csv", index_label="date")

print("Saved: data/retail_sales_rsxfsn_monthly.csv")
print(retail_out.head())

print("\nDownload complete.")

# ======================================================
# 3️⃣ CPI — MONTHLY INFLATION (ARIMA Case Study)
# ======================================================

"""
Why CPI?

- CPI level is non-stationary (strong upward trend).
- Month-over-month inflation is closer to stationary.
- ARIMA works well for autocorrelated stationary processes.
"""

print("\nDownloading CPI (FRED CPIAUCSL) dataset...")

SERIES_ID = "CPIAUCSL"
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}"

cpi = pd.read_csv(url)
cpi.columns = ["date", "value"]
cpi["date"] = pd.to_datetime(cpi["date"])

cpi_series = (
    cpi.set_index("date")["value"]
    .asfreq("MS")
)

cpi_series = pd.to_numeric(cpi_series, errors="coerce").dropna()

# Compute Month-over-Month Inflation (%)
inflation = 100 * (cpi_series.pct_change())
inflation = inflation.dropna()

inflation_df = inflation.to_frame(name="inflation_mom")

inflation_df.to_csv("data/cpi_inflation_mom.csv", index_label="date")

print("Saved: data/cpi_inflation_mom.csv")
print(inflation_df.tail())