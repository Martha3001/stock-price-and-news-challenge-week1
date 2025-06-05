import pandas as pd
from textblob import TextBlob
import numpy as np
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
from datetime import datetime

class CorrelationAnalysis:
    def __init__(self, stock_list):
        """
        Initialize the analyzer with list of stocks and base data path
        
        Args:
            stock_list (list): List of stock tickers to analyze
            base_path (str): Base directory containing news and stock data
        """
        self.stock_list = stock_list
        self.base_path = '../data' 
        self.results = {}
        self.all_news = None
        
    def load_combined_news(self):
        """
        Load the combined news headlines file and filter for our target stocks
        
        Returns:
            DataFrame: News data filtered for our stocks
        """
        try:
            news_path = '../data/raw_analyst_ratings.csv'
            news_df = pd.read_csv(news_path)
            
            # Assuming the file has columns: 'date', 'headline', 'stock'
            # Filter for our target stocks
            news_df = news_df[news_df['stock'].isin(self.stock_list)]
            
            return news_df
            
        except FileNotFoundError as e:
            print(f"Error loading news data: {e}")
            return None
    
    def load_stock_data(self, stock):
        """
        Load stock data for a specific stock
        
        Args:
            stock (str): Stock ticker symbol
            
        Returns:
            DataFrame: Stock price data
        """
        try:
            if stock == 'FB':
                stock = 'META'
            
            if stock == 'MSF':
                stock = 'MSFT'

            stock_path = os.path.join('../data', f'{stock}_historical_data.csv')
            stock_df = pd.read_csv(stock_path)
            return stock_df
            
        except FileNotFoundError as e:
            print(f"Error loading stock data for {stock}: {e}")
            return None
    
    def preprocess_data(self, news_df, stock_df):
        """
        Preprocess and clean the news and stock data
        
        Args:
            news_df (DataFrame): News data
            stock_df (DataFrame): Stock price data
            
        Returns:
            tuple: Processed (news_df, stock_df)
        """
        # Convert dates to datetime and set as index
        if news_df.index.name != 'date':
            news_df['date'] = pd.to_datetime(news_df['date'], format='mixed', utc=True).dt.date
            news_df.set_index('date', inplace=True)

        # Check if 'Date' is not the index of stock_df
        if stock_df.index.name != 'Date':
            stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='mixed', utc=True).dt.date
            stock_df.set_index('Date', inplace=True)
      
        
        return news_df, stock_df
    
    def analyze_sentiment(self, text):
        """
        Perform sentiment analysis on text using TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment polarity score (-1 to 1)
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def calculate_daily_metrics(self, news_df, stock_df, stock):
        """
        Calculate daily sentiment scores and stock returns for a specific stock
        
        Args:
            news_df (DataFrame): Combined news data with headlines
            stock_df (DataFrame): Stock price data
            stock (str): Stock ticker symbol
            
        Returns:
            DataFrame: Merged DataFrame with daily metrics
        """
        # Filter news for this specific stock
        stock_news = news_df[news_df['stock'] == stock].copy()
        
        # Calculate sentiment scores
        stock_news['sentiment'] = stock_news['headline'].apply(self.analyze_sentiment)
        
        # Aggregate multiple headlines per day by taking mean sentiment
        daily_sentiment = stock_news.groupby('date')['sentiment'].mean().reset_index()
        
        # Calculate daily percentage returns
        stock_df['daily_return'] = stock_df['Close'].pct_change() * 100
        
        # Remove first row with NaN return
        stock_df = stock_df.iloc[1:].reset_index()
        
        stock_df.rename(columns={'Date': 'date'}, inplace=True)
        # Merge sentiment and stock data
        merged_df = pd.merge(daily_sentiment, stock_df, on='date', how='inner')
        
        # Ensure chronological order
        merged_df.sort_values('date', inplace=True)
        
        return merged_df
    
    def analyze_correlation(self, merged_df, stock):
        """
        Perform correlation analysis and store results
        
        Args:
            merged_df (DataFrame): DataFrame with daily metrics
            stock (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary containing correlation results
        """
        # Calculate Pearson correlation
        valid_data = merged_df.dropna(subset=['sentiment', 'daily_return'])

        if len(valid_data) < 2:
            print(f"Not enough valid data for {stock} to calculate correlation.")
        
            # Store results
            result = {
                'correlation': None,
                'p_value': None,
                'num_days': len(valid_data),
                'merged_data': merged_df
            }

        else:
            correlation, p_value = pearsonr(valid_data['sentiment'], valid_data['daily_return'])

            # Store results
            result = {
                'correlation': correlation,
                'p_value': p_value,
                'num_days': len(valid_data),
                'merged_data': merged_df
            }
        
        self.results[stock] = result
        return result
    
    def visualize_results(self, stock):
        """
        Generate visualization for a stock's analysis
        
        Args:
            stock (str): Stock ticker symbol
        """
        if stock not in self.results:
            print(f"No results available for {stock}")
            return
            
        result = self.results[stock]
        merged_df = result['merged_data']

        if result["correlation"] != None:
            plt.figure(figsize=(10, 6))
            plt.scatter(merged_df['sentiment'], merged_df['daily_return'])
            plt.title(f'{stock}: News Sentiment vs. Daily Returns\n'
                    f'Correlation: {result["correlation"]:.3f} (p={result["p_value"]:.4f})')
            plt.xlabel('Average Daily Sentiment Score')
            plt.ylabel('Daily Return (%)')
            plt.grid(True)

            plt.show()
        
    
    def analyze_stock(self, news_df, stock):
        """
        Complete analysis pipeline for a single stock
        
        Args:
            news_df (DataFrame): Combined news data
            stock (str): Stock ticker symbol
        """
        print(f"\nAnalyzing {stock}...")
        
        # Load stock data
        stock_df = self.load_stock_data(stock)
        if stock_df is None:
            return
            
        # Preprocess data
        news_df, stock_df = self.preprocess_data(news_df, stock_df)
        
        # Calculate daily metrics
        merged_df = self.calculate_daily_metrics(news_df, stock_df, stock)
        
        # Analyze correlation
        result = self.analyze_correlation(merged_df, stock)
        
        # Visualize results
        self.visualize_results(stock)

        if result['correlation'] is not None:
            print(f"Analysis complete for {stock}. Correlation: {result['correlation']:.3f}")
        else:
            print(f"Analysis complete for {stock}. Correlation could not be calculated due to insufficient data.")
    
    def analyze_all_stocks(self):
        """
        Run analysis for all stocks in the stock list
        """
        # Load combined news data once
        news_df = self.load_combined_news()
        if news_df is None:
            return
            
        # Store the loaded news data
        self.all_news = news_df
        
        for stock in self.stock_list:
            self.analyze_stock(news_df, stock)
    
    