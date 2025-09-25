#HW on volatility
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
#1: the volatility of an asset is 2% per day.
#what is the standard deviation of the percentage price change in three days?
# sigma dailiy= .02 
#we want the standard deviation of the 3 day return (percent price chang over 3 days)
#if returns are i.i.d. and uncorrelated, variance adds linearity with time so:
#Var[R3e]= 3*Var[R1d] so sigma3d= sqrt of 3 *sigmadaily
#=  sqrt(3)*.02 = 3.46%
print(round((0.02 * (3 ** 0.5)) * 100, 2), "%")
#def multi_day_volatility(daily_vol, days):
#    return daily_vol * (days ** 0.5) * 100  # convert to %
#
# Example: 2% daily volatility over 3 days
#print(round(multi_day_volatility(0.02, 3), 2), "%")

#2:
# Black-Scholes formula for a European call
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Implied volatility solver
def implied_volatility_call(S, K, T, r, market_price):
    # Define the function whose root we want to find
    objective = lambda sigma: bs_call_price(S, K, T, r, sigma) - market_price
    # Use a root-finding method (brentq works well for monotonic functions)
    return brentq(objective, 1e-6, 5.0)  # Search in [0.000001, 500%]

# Parameters
S = 120.0      # current stock price
K = 115.0      # strike price
T = 0.5        # time to expiration (in years)
r = 0.03       # risk-free rate
market_price = 8.75  # observed call price

iv = implied_volatility_call(S, K, T, r, market_price)
print(f"Implied Volatility: {iv:.4%}")

#3
#The most recent estimate of the daily volatility of an asset is 1.5% and the price
# of the asset at the close of trading yesterday was $30.00. the parameter lambda in the EWMA model is .94.
# Suppose that the price of the asset at the close of trading today is $30.50. 
# How will this cause the volatility to be updated by the EWMA model?

# Parameters
sigma_prev = 0.015     # yesterday's volatility (1.5%)
price_yesterday = 30.0
price_today = 30.5
lmbda = 0.94

# Step 1: Compute yesterday's variance
var_prev = sigma_prev**2

# Step 2: Compute today's return
r_t = (price_today - price_yesterday) / price_yesterday

# Step 3: Update variance using EWMA
var_new = lmbda * var_prev + (1 - lmbda) * r_t**2

# Step 4: Take square root to get new volatility
sigma_new = np.sqrt(var_new)

print(f"Updated daily volatility: {sigma_new:.4%}")