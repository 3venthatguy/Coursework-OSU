import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

''' Part A '''
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

''' Part B '''
def plot_histogram(data):
    tickers = data.columns
    n_stocks = len(tickers)
    
    # Create subplots for closing prices and log returns
    fig, axes = plt.subplots(2, n_stocks, figsize=(15, 10))
    
    for i, ticker in enumerate(tickers):
        stock_data = data[ticker].dropna()
        
        # Calculate log returns
        log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
        
        # Plot histogram of closing prices
        axes[0, i].hist(stock_data, bins=30, edgecolor="black", alpha=0.7, color='skyblue')
        axes[0, i].set_title(f"{ticker} - Closing Prices")
        axes[0, i].set_xlabel("Price ($)")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].grid(axis='y', alpha=0.3)
        
        # Plot histogram of log returns
        axes[1, i].hist(log_returns, bins=30, edgecolor="black", alpha=0.7, color='lightcoral')
        axes[1, i].set_title(f"{ticker} - Log Returns")
        axes[1, i].set_xlabel("Log Return")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def estimate_time_trend(data):
    results = {}
    
    print("Time Trend Analysis Results:")
    print("=" * 60)
    
    for ticker in data.columns:
        stock_data = data[ticker].dropna()
        
        # Create time variable (trading days)
        time_var = np.arange(len(stock_data))
        
        # Perform linear regression: Price = alpha + beta * time + error
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_var, stock_data)
        
        # Calculate t-statistic for the slope coefficient
        t_stat = slope / std_err
        
        # Store results
        results[ticker] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            't_stat': t_stat,
            'significant_at_5pct': p_value < 0.05
        }
        
        # Print results
        print(f"\n{ticker}:")
        print(f"  Slope (β): {slope:.6f}")
        print(f"  Standard Error: {std_err:.6f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  Significant at 5% level: {'Yes' if p_value < 0.05 else 'No'}")
        
        if p_value < 0.05:
            trend_direction = "positive" if slope > 0 else "negative"
            print(f"  → Can reject H₀: significant {trend_direction} time trend detected")
        else:
            print(f"  → Cannot reject H₀: no significant time trend")
    
    return results

def plot_time_trends(data, results):
    """Plot the actual prices with fitted trend lines"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ticker in enumerate(data.columns):
        stock_data = data[ticker].dropna()
        time_var = np.arange(len(stock_data))
        
        # Plot actual prices
        axes[i].plot(time_var, stock_data, 'o', alpha=0.6, markersize=3, label='Actual Prices')
        
        # Plot fitted trend line
        trend_line = results[ticker]['intercept'] + results[ticker]['slope'] * time_var
        axes[i].plot(time_var, trend_line, 'r-', linewidth=2, label='Fitted Trend')
        
        axes[i].set_title(f"{ticker} - Price vs Time\np-value: {results[ticker]['p_value']:.4f}")
        axes[i].set_xlabel("Trading Days")
        axes[i].set_ylabel("Price ($)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Part A: Get the data
    print("Downloading stock data for top 3 companies by market cap...")
    data = get_top3_stock_data(100)
    
    if data.empty:
        print("Failed to download data. Please check your internet connection.")
        return
    
    print(f"Data downloaded successfully. Shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Part B: Plot histograms
    print("\nGenerating histograms...")
    plot_histograms(data)
    
    # Parts C & D: Time trend analysis
    print("\nPerforming time trend analysis...")
    results = estimate_time_trend(data)
    
    # Plot trends
    print("\nGenerating trend plots...")
    plot_time_trends(data, results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    significant_trends = [ticker for ticker, res in results.items() if res['significant_at_5pct']]
    
    if significant_trends:
        print(f"Stocks with significant time trends at 5% level: {', '.join(significant_trends)}")
        print("→ We CAN reject H₀ (no time trend) for these stocks.")
    else:
        print("No stocks show significant time trends at the 5% level.")
        print("→ We CANNOT reject H₀ (no time trend) for any of the stocks.")
    
    return data, results

# Run the analysis
if __name__ == "__main__":
    data, results = main()