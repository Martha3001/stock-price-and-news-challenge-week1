## Setup
1. Clone: `git clone https://github.com/Martha3001/stock-price-and-news-challenge-week1.git`
2. Create venv: `python3 -m venv .venv`
3. Activate: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

# Predicting Price Moves with News Sentiment

## 📌 Overview

This project analyzes the correlation between financial news sentiment and stock price movements for Nova Financial Solutions. It combines:
- Natural Language Processing (NLP) for sentiment analysis
- Technical indicators (RSI, MACD) for stock analysis
- Statistical correlation methods

**Key Stocks Analyzed**: AAPL, AMZN, NVDA, TSLA, FB, GOOG, MSFT

## 📂 Repository Structure

stock-price-and-news-challenge-week1/
├── data/
├── notebooks/
│ ├── correlation_eda.ipynb
│ ├── headline_eda.ipynb
│ └── quantitative_eda.ipynb
├── src/
│ ├── correlation.py
│ ├── data_loader.py 
│ ├── quantitative_analysis.py 
│ └── sentiment_analysis.py 
├── .github/
│ └── workflows/ # CI/CD pipelines
├── .gitignore
├── requirements.txt # Python dependencies
└── README.md # This file


## 🔍 Key Findings

1. **Strongest Correlations**:
   - FB: r = 0.295 (p=0.0107)
   - NVDA: r = 0.104 (p=0.0005)

2. **Publisher Insights**:
   - Paul contributors published 24.4% of all articles
   - Weekends has the lowest publication volume

## 🛠 Installation

1. Clone repository:
```bash
git clone git clone https://github.com/Martha3001/stock-price-and-news-challenge-week1.git
cd stock-price-and-news-challenge-week1
```

2. Create virtual environment:
```bash
python -m venv .venv
source venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3.Install dependencies:
```bash
pip install -r requirements.txt
```