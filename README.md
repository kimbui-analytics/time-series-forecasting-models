# What I Learned About Time Series Forecasting by Actually Testing different Models

When I first started exploring time series forecasting, I kept hearing the same model names over and over:

- ARIMA  
- Holt-Winters  
- Seasonal models  
- Naive baselines  

For this project: 

> I didn’t just want to *use* these models — I wanted to understand when they actually make sense.

Flipping the process from starting with a dataset and try to force models onto it.

**What kind of data structure makes each model appropriate?**

> **Context is considered first in this case** and the project became an experiment in matching models to structure.

Below is what I learned.

---

# Lesson 1: Sometimes Simple Is Already Optimal  
## Case Study: S&P 500 (Daily)

I started with daily S&P 500 closing prices.

The naive forecast assumes:
> Tomorrow equals today.

While this is overly simple, the results make the naive model forecast extremely difficult to beat.

### Results

| Metric | Value |
|--------|-------|
| MAPE | **0.83%** |
| MAE | 57.57 |
| RMSE | 68.67 |

![S&P 500 Naive Forecast](data/sp500_naive_plot.png)

### What This Taught Me

- Financial markets behave close to a random walk.
- Short-term price movements are very difficult to predict.
- Adding complexity does not automatically improve forecasts.

Learning Lesson:

> Complexity does not guarantee better performance.

Sometimes, the structure of the data simply doesn’t allow for strong predictive improvements.

---

# Lesson 2: Seasonality Alone Isn’t Enough  
## Case Study: Retail Sales (Monthly, Not Seasonally Adjusted)

Retail sales are clearly seasonal. Holiday spikes are visible every year.

So I assumed seasonal naive would dominate.

Seasonal naive predicts:

>Next December equals last December.

But when I tested it against naive:

| Model | MAPE |
|--------|-------|
| Naive | **8.30%** |
| Seasonal Naive | 10.00% |

Naive actually performed better.

### Why?

Retail sales don’t just repeat — they **grow over time**.

Using last year’s value underestimates the current level because of trend.

Learning Lesson

> Seasonality without accounting for growth can still fail.

---

# Lesson 3: When Structure Is Explicitly Modeled, Performance Improves Dramatically  
## Holt-Winters (Trend + Seasonality)

Retail sales contain:

- A long-term upward trend  
- A repeating yearly pattern  

Holt-Winters models both simultaneously.

When I tested it:

| Model | MAPE |
|--------|-------|
| Naive | 8.30% |
| Seasonal Naive | 10.00% |
| **Holt-Winters** | **1.80%** |

### What This Taught Me

When both trend and seasonality are present:

- You must model both explicitly.
- Ignoring either one costs accuracy.
- Structured models outperform simple baselines by a wide margin.

Learning Lesson:

> Model structure should reflect data structure.

![Retail Sales Holt-Winters Forecast](data/retail_sales_holt_winters_plot.png)

---

# Lesson 4: ARIMA Works When the Data Has Memory  
## Case Study: CPI Month-over-Month Inflation

Unlike retail sales, inflation doesn’t trend upward forever.  
It fluctuates around a relatively stable average.

However, it does show short-term patterns.

If inflation was high last month, it often remains elevated this month.

This is where ARIMA comes in.

### Results

| Model | MAE | RMSE |
|--------|--------|--------|
| Naive | 0.1151 | 0.1447 |
| **ARIMA(1,0,1)** | **0.1026** | **0.1318** |

ARIMA improved forecast accuracy by roughly 10–11%.

### What This Taught Me

Inflation contains short-term patterns that naive ignores.

ARIMA improves performance because it:

- Uses recent values intelligently
- Adjusts forecasts based on recent errors
- Captures short-term structure

The key insight:

> ARIMA works best when data fluctuates around a stable average but contains short-term patterns. We evaluate the model metrics based on MAE / RMSE because inflation values are close to zero, which MAPE is unsuitable for (since it divides by the actual value).

---

# My Final Conclusions

1. **There is no universal "best model"**: Depending on the dataset structure can help determine which model will be a good 'fit'.

| Dataset | Structure | Best Model | Why |
|----------|------------|------------|------|
| S&P 500 | Random walk | Naive | Little predictable structure |
| Retail Sales | Trend + Seasonality | Holt-Winters | Must model growth and repeating patterns |
| CPI Inflation | Stable average + short-term patterns | ARIMA | Uses recent behavior to improve forecasts |

2. **Start Simple**: Iterate continuously as you set a baseline to benchmark honestly, add complexity only when it is justified, and let the data guide the model.

3. **Forecasting is more than using a model**: It requires understanding the data structure/context, choosing the right tool, evaluating rigorously and communicating clearly.

---

If I extend this project further, I’d like to explore:

- Rolling cross-validation  
- Seasonal ARIMA (SARIMA)  
- Exogenous variable models (ARIMAX)  
- Business decision simulations  

But the foundation is now clear:

**Good forecasting starts with understanding the data, not the model.**

