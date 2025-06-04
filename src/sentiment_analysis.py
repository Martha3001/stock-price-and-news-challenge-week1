import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize
import re

print("Installing required NLTK resources...")
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
print("NLTK resources installed.")

class SentimentAnalysis:
    def __init__(self, df):
        self.df = df
        
        self.ensure_nltk_resources()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def ensure_nltk_resources(self):
        # Ensure necessary NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def format_datetime(self):
        # Convert 'date' column to datetime format and extract components
        self.df['date'] = pd.to_datetime(self.df['date'], format='mixed', utc=True)
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()

    def set_date_index(self):
        # Ensure 'date' column is in datetime format
        if self.df.index.name != 'date':
            self.df.set_index('date', inplace=True)
    
    def analyzing_headlines(self):
        #  Analyze headline lengths
        self.df['headline_length'] = self.df['headline'].apply(len)
        print("Headline Length Statistics:")
        print(self.df['headline_length'].describe())
        print(self.df.head())

    def plot_headline_length(self):
        # Plot histogram of headline lengths
        fig = plt.figure(figsize=(8,4))
        plt.hist(self.df['headline_length'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length (characters)')
        plt.ylabel('Number of Articles')
        plt.tight_layout()

        return fig
    
    def publisher_analysis(self):
        # Analyze the number of articles per publisher
        publisher_counts = self.df['publisher'].value_counts()
        print("\nTop 10 Publishers by Article Count:")
        print(publisher_counts.head(10))

        # Plotting the number of articles per publisher
        fig = plt.figure(figsize=(10,5))
        publisher_counts.head(10).plot(kind='bar', color='orange')
        plt.title('Top 10 Publishers by Number of Articles')
        plt.xlabel('Publisher')
        plt.ylabel('Article Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return fig     
        
    def analyze_publication_frequency(self):
        # Analyze publication frequency by day of the week
        fig = plt.figure(figsize=(10, 6))

        sns.countplot(data=self.df, x='day_of_week', order=[
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.title('Number of Articles Published by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Article Count')

        return fig
    
    # Preprocessing function
    def preprocess_text(self, text):
        '''      
        Preprocess the text by converting to lowercase, removing punctuation, and stopwords.
        Args:
            text (str): The text to preprocess.
        Returns:
            The preprocessed text.
        '''
        stopwords = nltk.corpus.stopwords.words('english')

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords]
        return ' '.join(tokens)
    
    # Function to display top N terms/phrases from a matrix
    def display_top_terms(vectorizer, feature_matrix, n=10):
        """
        Display top N terms/phrases from a given feature matrix.
        Args:
            vectorizer: The vectorizer used to transform the text data.
            feature_matrix: The matrix containing the transformed text data.
            n: Number of top terms/phrases to display.
        """
        # Display top N terms/phrases
        print(f"\nTop {n} common terms/phrases:")
        sum_tfidf = feature_matrix.sum(axis=0)
        sorted_indices = np.argsort(sum_tfidf.A1)[::-1]
        feature_names = vectorizer.get_feature_names_out()

        for i in range(n):
            term_index = sorted_indices[i]
            print(f"{feature_names[term_index]}")
    
    def text_analysis(self):  

        headlines = self.df['headline'].tolist()

        # Preprocess the headlines
        processed_headlines = [self.preprocess_text(headline) for headline in headlines]

        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_headlines)
        # Display top common terms
        self.display_top_terms(self.tfidf_vectorizer, self.tfidf_matrix)

        # To identify common phrases, N-grams can be used.
        self.ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3)) # Considering bigrams and trigrams
        self.ngram_matrix = self.ngram_vectorizer.fit_transform(processed_headlines)
        # Display top common phrases
        self.display_top_terms(self.ngram_vectorizer, self.ngram_matrix)



    def time_series_analysis_over_time(self):
        # Create time series data
        daily_counts = self.df.groupby(self.df.index.date).size()
        print("\nDaily Article Counts:")
        print(daily_counts.head())

        # Plot daily counts time series
        fig = plt.figure(figsize=(12,5))
        daily_counts.plot()
        plt.title('Daily Article Publication Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')

        return fig
    
    def time_series_analysis_over_hour(self):
        # Create time series data for hourly counts
        hourly_counts = self.df.groupby(self.df.index.hour).size()
        print("\nHourly Article Counts:")
        print(hourly_counts.head())

        # Plot hourly counts time series
        fig = plt.figure(figsize=(12,5))
        hourly_counts.plot(kind='bar')
        plt.title('Hourly Article Publication Frequency')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Articles')

        return fig
    
    def publisher_contribution(self):
        0# Publisher contribution
        publisher_share = self.df['publisher'].value_counts(normalize=True) * 100
        print("Publisher Share (% of total articles):")
        print(publisher_share)

        # Plotting the top 10 publisher contribution using pie chart  
        # Ensure the publisher_share is sorted
        publisher_share = publisher_share.sort_values(ascending=False).head(10)
        # Plotting the pie chart
        sns.set_palette("pastel") 
        fig = plt.figure(figsize=(10, 6))
        publisher_share.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Top 10 Publisher Contribution to Total Articles')
        plt.ylabel('')
        plt.tight_layout()
        plt.legend(title='Publisher', loc='upper left', bbox_to_anchor=(1, 1))

        return fig