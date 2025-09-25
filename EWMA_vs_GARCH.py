import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

# Step 1: Download SPY data (5 years)
data = yf.download("SPY", start="2018-01-01", end="2023-01-01")
prices = data['Adj Close']

# Step 2: Compute daily log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Step 3: EWMA volatility with lambda=0.94
lambda_ = 0.94
ewma_var = [returns.var()]  # start with sample variance
for r in returns:
    new_var = lambda_ * ewma_var[-1] + (1 - lambda_) * r**2
    ewma_var.append(new_var)
ewma_vol = np.sqrt(ewma_var[1:])  # drop initial placeholder

# Step 4: GARCH(1,1) model fitting
am = arch_model(returns * 100, p=1, q=1)  # scale returns (%) for stability
res = am.fit(disp="off")
garch_vol = res.conditional_volatility / 100.0  # scale back

# Step 5: Plot comparison
plt.figure(figsize=(12,6))
plt.plot(returns.index, ewma_vol, label="EWMA (λ=0.94)", alpha=0.8)
plt.plot(returns.index, garch_vol, label="GARCH(1,1)", alpha=0.8)
plt.title("Volatility Estimates: EWMA vs. GARCH(1,1) on SPY")
plt.ylabel("Daily Volatility")
plt.legend()
plt.show()

# Step 6: Print parameter estimates for GARCH
print("GARCH(1,1) parameter estimates:")
print(res.params)