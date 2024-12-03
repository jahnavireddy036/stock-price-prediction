import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_time_series_close(df, stock_ticker):
    """
    Plots the time series of the 'Close' price.

    Parameters:
        df (pd.DataFrame): Stock data.
        stock_ticker (str): Stock ticker for the plot title.
    """
    fig = px.line(df, x=df.index, y='Close', title=f'Time Series of Close Price for {stock_ticker}')
    fig.show()

def plot_stock_price_comparison(df, stock_ticker):
    """
    Plots time series of multiple stock price columns for comparison.

    Parameters:
        df (pd.DataFrame): Stock data.
        stock_ticker (str): Stock ticker for the plot title.
    """
    fig = px.line(df, x=df.index, y=['Open', 'High', 'Low', 'Close'], 
                  title=f'Time Series of Stock Prices for {stock_ticker}')
    fig.show()

def plot_log_returns_distribution(df):
    """
    Plots the distribution of log returns.

    Parameters:
        df (pd.DataFrame): Stock data.
    """
    df['Log_Returns'] = df['Close'].pct_change().apply(lambda x: np.log(1 + x))
    sns.histplot(df['Log_Returns'].dropna(), kde=True, bins=100)
    plt.title('Distribution of Log Returns')
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.show()

def plot_seasonal_decompose(df):
    """
    Performs seasonal decomposition of the 'Close' price.

    Parameters:
        df (pd.DataFrame): Stock data.
    """
    result = seasonal_decompose(df['Close'], model='multiplicative', period=365)

    plt.figure(figsize=(14, 10))
    plt.subplot(411)
    plt.plot(result.observed, label='Observed')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(result.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(result.resid, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_smoothing_techniques(df):
    """
    Applies and plots various smoothing techniques on the 'Close' price.

    Parameters:
        df (pd.DataFrame): Stock data.
    """
    # Simple Moving Average (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average (EMA)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Weighted Moving Average (WMA)
    weights = np.arange(1, 51)
    df['WMA_50'] = df['Close'].rolling(50).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    # Savitzky-Golay Filter
    df['SGF_50'] = savgol_filter(df['Close'], window_length=51, polyorder=2)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['WMA_50'], mode='lines', name='WMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SGF_50'], mode='lines', name='SGF 50'))

    fig.update_layout(title='Stock Price Smoothing Techniques',
                      xaxis_title='Date',
                      yaxis_title='Price')
    fig.show()

def print_stock_insights(df):
    """
    Prints insights about the stock data.

    Parameters:
        df (pd.DataFrame): Stock data.
    """
    print("Stock Data Insights:")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Total Data Points: {len(df)}")
    print(f"Mean Close Price: {df['Close'].mean():.2f}")
    print(f"Highest Close Price: {df['Close'].max():.2f} on {df['Close'].idxmax()}")
    print(f"Lowest Close Price: {df['Close'].min():.2f} on {df['Close'].idxmin()}")
    print(f"Volatility (Std Dev of Returns): {df['Close'].pct_change().std():.4f}")