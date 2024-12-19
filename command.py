import yfinance as yf
import pandas as pd
import numpy as np
import csv

# Function to load Indonesian stocks from a CSV file
def load_stock_symbols(csv_file):
    indonesian_stocks = []
    try:
        with open(csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            indonesian_stocks = [row[0] for row in csv_reader]
    except Exception as e:
        print(f"Error loading stock symbols from {csv_file}: {e}")
    return indonesian_stocks

# Fetch stock data
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        stock_data = yf.download(ticker, period=period, interval=interval)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Calculate Moving Averages
def calculate_moving_averages(data, short_window, long_window):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

# Moving Average Crossover Strategy
def moving_average_crossover_strategy(data):
    data['Signal'] = 0.0
    data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1.0, 0.0)
    data['Position'] = data['Signal'].diff()

    # Buy signal when short MA crosses above long MA
    buy_signals = data.loc[data['Position'] == 1.0]
    return not buy_signals.empty  # True if there's a buy signal

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Stock Screener to find buying opportunities
def stock_screener(stocks, short_window=20, long_window=50):
    buy_candidates = []
    
    for ticker in stocks:
        stock_data = fetch_data(ticker)
        if stock_data is not None:
            stock_data = calculate_moving_averages(stock_data, short_window, long_window)
            if moving_average_crossover_strategy(stock_data):
                buy_candidates.append(ticker)
                print(f"Buy signal detected for {ticker}")
            else:
                print(f"No buy signal for {ticker}")
    
    return buy_candidates

def main():
    # Load Indonesian stock symbols from CSV file
    indonesian_stocks = load_stock_symbols('stocklist.csv')
    
    # Run the screener on the list of Indonesian stocks
    buy_signals = stock_screener(indonesian_stocks)
    
    if buy_signals:
        print("Stocks with buy signals:", buy_signals)
    else:
        print("No stocks with buy signals at the moment.")

if __name__ == "__main__":
    main()