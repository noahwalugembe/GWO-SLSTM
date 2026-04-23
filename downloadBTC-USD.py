import yfinance as yf
import pandas as pd

# ----------------------------
# CONFIG   4,039 days
# ----------------------------
ticker = "BTC-USD"
start_date = "2014-09-17"
end_date = "2025-10-09"
output_file = "btcusd_2014_2026.csv"

# ----------------------------
# DOWNLOAD DATA
# ----------------------------
print("Downloading BTC/USD data...")

data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False   # 🔥 IMPORTANT FIX
)

# ----------------------------
# CLEAN DATA
# ----------------------------
data.reset_index(inplace=True)

# Ensure correct order + names
data = data[[
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]]

# ----------------------------
# SAVE TO CSV
# ----------------------------
data.to_csv(output_file, index=False)

print(f"✅ Data saved to {output_file}")
print(data.head())