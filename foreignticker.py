import yfinance as yf
import pandas as pd

# Date range
start_date = "2024-01-01"
end_date = "2025-01-01"

# Example tickers from each exchange
tickers = {
    "Euronext Paris (LVMH)": "MC.PA",
    "Frankfurt (Germany, SAP)": "SAP.DE",
    "London (UK, HSBC)": "HSBA.L",
    "Shanghai (SSE, Bank of China)": "601988.SS"
}

data = {}

for name, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    data[name] = df["Close"]
    print(f"\n{name} ({ticker})")
    print(df.head(), "\n")

# Combine into one DataFrame
combined = pd.DataFrame(data)

# Save to CSV
combined.to_csv("global_stocks_2024.csv")

print("\nCombined Data:")
print(combined.head())
