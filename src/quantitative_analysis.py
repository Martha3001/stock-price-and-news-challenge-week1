import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib
import pynance as pn
import numpy as np


class QuantitativeAnalysis:
    def __init__(self, df, stock_ticker):
        self.df = df
        self.stock_ticker = stock_ticker
        
    def format_datetime(self):
        # Convert 'date' column to datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='mixed', utc=True)

    def set_date_index(self):
        self.df.set_index('Date', inplace=True)
        self.df.sort_index(inplace=True)

    def calculate_technical_indicators(self):
        """
        Calculate various technical indicators using TA-Lib
        """
        # Moving Averages
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['EMA_20'] = talib.EMA(self.df['Close'], timeperiod=20)
        
        # RSI (Relative Strength Index)
        self.df['RSI_14'] = talib.RSI(self.df['Close'], timeperiod=14)
        
        # MACD
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = talib.MACD(self.df['Close'])
        
        return self.df
    
    def calculate_financial_metrics(self):
        """
        Calculate financial metrics using PyNance
        """
        # Daily returns
        self.df['daily_return'] = self.df['Close'].pct_change()
        
        # Calculate annualized volatility (20-day rolling std dev of returns)
        self.df['volatility_20'] = self.df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        
        return self.df

    def plot_price_with_indicators(self):
        """
        Create visualizations of price data with technical indicators
        """
        plt.figure(figsize=(15, 20))
        
        # Price and Moving Averages
        plt.subplot(4, 1, 1)
        plt.plot(self.df['Close'], label='Close Price', color='blue', alpha=0.5)
        plt.plot(self.df['SMA_20'], label='20-day SMA', color='red')
        plt.plot(self.df['EMA_20'], label='20-day EMA', color='green')
        plt.title(f'{self.stock_ticker} Price with Moving Averages')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # RSI
        plt.subplot(4, 1, 2)
        plt.plot(self.df['RSI_14'], label='14-day RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.axhline(30, linestyle='--', color='green', alpha=0.5)
        plt.title(f'{self.stock_ticker} Relative Strength Index (RSI)')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        # MACD
        plt.subplot(4, 1, 3)
        plt.plot(self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df['MACD_signal'], label='Signal Line', color='orange')
        plt.bar(self.df.index, self.df['MACD_hist'], label='MACD Histogram', color='gray')
        plt.title(f'{self.stock_ticker} Moving Average Convergence Divergence (MACD)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()



    def plot_financial_metrics(self):
        """
        Create visualizations of financial metrics
        """
        plt.figure(figsize=(15, 12))

        # Daily Returns
        plt.subplot(3, 1, 1)
        plt.plot(self.df['daily_return'], label='Daily Returns', color='blue', alpha=0.5)
        plt.title(f'{self.stock_ticker} Daily Returns')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        
        # Volatility
        plt.subplot(3, 1, 2)
        plt.plot(self.df['volatility_20'], label='20-day Volatility', color='red')
        plt.title(f'{self.stock_ticker} Annualized Volatility (20-day rolling)')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

