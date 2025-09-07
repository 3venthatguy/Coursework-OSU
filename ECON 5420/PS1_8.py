import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_top3_stock_data(days):

    # Top 3 by market cap (as of 2025)
    tickers = ["AAPL", "MSFT", "NVDA"]
    
    # Download data
    raw_data = yf.download(tickers, period=f"{days}d")
    if raw_data is not None and "Close" in raw_data:
        data = raw_data["Close"]
    else:
        data = pd.DataFrame()  # Return empty DataFrame if download fails

    return data

def plot_histogram(data):
    log_return = np.log(data / data.shift(1)).dropna().values.flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(log_return, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Log Returns")
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def estimate_time_trend(data):

data = get_top3_stock_data(100)
if not data.empty:
    plot_histogram(data)