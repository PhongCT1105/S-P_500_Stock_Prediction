# arima_model.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def preprocess_data(stock_ticker, start_date='2010-01-01', end_date='2024-01-01'):
    # Download the stock data
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    
    # Use the 'Close' prices for analysis
    close_prices = stock_data['Close']
    
    # Split into training and testing datasets
    train_size = int(len(close_prices) * 0.8)
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    
    return train_data, test_data

def build_arima_model(train_data, order=(5, 1, 0)):
    # Build and train the ARIMA model
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def make_predictions(model_fit, test_data):
    # Make predictions
    start_index = len(model_fit.data.endog)
    end_index = start_index + len(test_data) - 1
    predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')
    return predictions

def plot_results(train_data, test_data, predictions):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Training Data')
    plt.plot(test_data, label='Actual Stock Price', color='blue')
    plt.plot(test_data.index, predictions, label='Predicted Stock Price', color='red')
    plt.title('ARIMA Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def evaluate_model(test_data, predictions):
    # Evaluate the model performance
    mse = mean_squared_error(test_data, predictions)
    print(f'Mean Squared Error: {mse:.2f}')

if __name__ == "__main__":
    # Preprocess data
    train_data, test_data = preprocess_data('MSFT')
    
    # Build and train ARIMA model
    arima_model = build_arima_model(train_data)
    
    # Make predictions
    predictions = make_predictions(arima_model, test_data)
    
    # Plot the results
    plot_results(train_data, test_data, predictions)
    
    # Evaluate model
    evaluate_model(test_data, predictions)
