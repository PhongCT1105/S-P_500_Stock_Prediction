
# 📈 Stock Price Prediction with ARIMA and LSTM Models

This project demonstrates the use of traditional time series models (ARIMA) and deep learning models (LSTM) for predicting stock prices. It integrates data preprocessing, model training, and deployment into a user-friendly Streamlit web application.

## ✨ Features

- **📊 Data Source**: Historical stock price data retrieved using the `yfinance` Python library.
- **🎯 Target Variable**: Closing prices of stocks in the S&P 500 index.
- **📂 Models**:
  - **ARIMA**: Traditional time series model for short-term forecasts.
  - **LSTM**: Deep learning model for capturing non-linear trends and long-term dependencies.
- **🌐 Web Interface**: Interactive Streamlit app for model visualization and comparison.

## 🔄 Project Workflow

### 1. 🛠️ Data Preprocessing
- Data normalized using `MinMaxScaler`.
- Sliding window of 60 days used to create input-output sequences for LSTM.
- Stationarity achieved using differencing for ARIMA.

### 2. 📘 Modeling Approaches
- **ARIMA**:
  - Hyperparameter selection via MINIC and ESACF methods.
  - Focused on short-term accuracy but struggled with non-linear patterns.
- **LSTM**:
  - Two LSTM layers with dense layers for output.
  - Used Adam optimizer and Mean Squared Error (MSE) loss.
  - Outperformed ARIMA in both short-term and long-term forecasting.

### 3. 🚀 Deployment
- Deployed on a Streamlit web app for interactive stock selection and prediction visualization.

## 🧩 Challenges and Solutions
- **ARIMA**:
  - Required data stationarity through extensive preprocessing.
  - Manual hyperparameter tuning using grid search and AIC.
- **LSTM**:
  - Overfitting resolved by adding dropout layers.
  - Computationally intensive, requiring significant hardware.

## 📊 Results
- **📈 Metrics**:
  - LSTM: MAE = 2.34, MSE = 8.92, R² = 0.87.
- LSTM outperformed ARIMA, excelling in capturing non-linear trends and volatility.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🌟 Future Directions
- Incorporate external factors like macroeconomic indicators and sentiment analysis.
- Explore advanced models (e.g., Transformers) for improved predictions.
- Combine ARIMA and LSTM in hybrid models.

## 🌐 Live Demo
Check out the deployed web app: [Stock Price Predictor](https://stockpredictsp500.streamlit.app/)


