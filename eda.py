import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

class StockAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the StockAnalyzer class with stock data.

        Parameters:
            df (pd.DataFrame): Stock data.
        """
        self.df = df

    def plot_time_series_close(self, stock_ticker: str):
        """
        Plots the time series of the 'Close' price.

        Parameters:
            stock_ticker (str): Stock ticker for the plot title.
        """
        fig = px.line(self.df, x=self.df.index, y='Close', title=f'Time Series of Close Price for {stock_ticker}')
        fig.show()

    def plot_stock_price_comparison(self, stock_ticker: str):
        """
        Plots time series of multiple stock price columns for comparison.

        Parameters:
            stock_ticker (str): Stock ticker for the plot title.
        """
        fig = px.line(self.df, x=self.df.index, y=['Open', 'High', 'Low', 'Close'], 
                      title=f'Time Series of Stock Prices for {stock_ticker}')
        fig.show()

    def plot_log_returns_distribution(self):
        """
        Plots the distribution of log returns.
        """
        self.df['Log_Returns'] = self.df['Close'].pct_change().apply(lambda x: np.log(1 + x))
        sns.histplot(self.df['Log_Returns'].dropna(), kde=True, bins=100)
        plt.title('Distribution of Log Returns')
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')
        plt.show()

    def plot_seasonal_decompose(self):
        """
        Performs seasonal decomposition of the 'Close' price.
        """
        result = seasonal_decompose(self.df['Close'], model='multiplicative', period=365)

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

    def plot_smoothing_techniques(self):
        """
        Applies and plots various smoothing techniques on the 'Close' price.
        """
        # Simple Moving Average (SMA)
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()

        # Exponential Moving Average (EMA)
        self.df['EMA_50'] = self.df['Close'].ewm(span=50, adjust=False).mean()

        # Weighted Moving Average (WMA)
        weights = np.arange(1, 51)
        self.df['WMA_50'] = self.df['Close'].rolling(50).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

        # Savitzky-Golay Filter
        self.df['SGF_50'] = savgol_filter(self.df['Close'], window_length=51, polyorder=2)

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['SMA_50'], mode='lines', name='SMA 50'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['EMA_50'], mode='lines', name='EMA 50'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['WMA_50'], mode='lines', name='WMA 50'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['SGF_50'], mode='lines', name='SGF 50'))

        fig.update_layout(title='Stock Price Smoothing Techniques',
                          xaxis_title='Date',
                          yaxis_title='Price')
        fig.show()

    def print_stock_insights(self):
        """
        Prints insights about the stock data.
        """
        print("Stock Data Insights:")
        print(f"Date Range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Total Data Points: {len(self.df)}")
        print(f"Mean Close Price: {self.df['Close'].mean():.2f}")
        print(f"Highest Close Price: {self.df['Close'].max():.2f} on {self.df['Close'].idxmax()}")
        print(f"Lowest Close Price: {self.df['Close'].min():.2f} on {self.df['Close'].idxmin()}")
        print(f"Volatility (Std Dev of Returns): {self.df['Close'].pct_change().std():.4f}")
