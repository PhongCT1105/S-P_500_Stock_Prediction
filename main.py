import streamlit as st
from datetime import date
import yfinance as yf
import arima
import lstm  # Import the LSTM module (lstm.py)
import pandas as pd


# Constants for data range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


# Enhanced Title
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-family: Arial, sans-serif;'>
    📈 Stock Prediction App 🚀
    </h1>
    """,
    unsafe_allow_html=True
)


# Predefined dictionary of company names and ticker symbols
stock_mapping = {
    "3M": "MMM",
    "Abbott Laboratories": "ABT",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "American Express": "AXP",
    "Apple": "AAPL",
    "AT&T": "T",
    "Bank of America": "BAC",
    "Berkshire Hathaway": "BRK.B",
    "Boeing": "BA",
    "Caterpillar": "CAT",
    "Chevron": "CVX",
    "Cisco": "CSCO",
    "Citigroup": "C",
    "Coca-Cola": "KO",
    "Comcast": "CMCSA",
    "Disney": "DIS",
    "Exxon Mobil": "XOM",
    "Facebook (Meta)": "META",
    "Ford Motor": "F",
    "General Electric": "GE",
    "Goldman Sachs": "GS",
    "Google": "GOOGL",
    "Home Depot": "HD",
    "IBM": "IBM",
    "Intel": "INTC",
    "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ",
    "Lockheed Martin": "LMT",
    "Mastercard": "MA",
    "McDonald's": "MCD",
    "Merck": "MRK",
    "Microsoft": "MSFT",
    "Morgan Stanley": "MS",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "NVIDIA": "NVDA",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Procter & Gamble": "PG",
    "Qualcomm": "QCOM",
    "Salesforce": "CRM",
    "Starbucks": "SBUX",
    "Target": "TGT",
    "Tesla": "TSLA",
    "Texas Instruments": "TXN",
    "UnitedHealth Group": "UNH",
    "Verizon": "VZ",
    "Visa": "V",
    "Walmart": "WMT"
}


# Search bar with autocomplete suggestions
st.text("Start typing the company name and select from the suggestions:")


# Convert company names to list for autocomplete
company_names = list(stock_mapping.keys())


# Search with autocomplete
company_name = st.selectbox("Company Name", options=[""] + company_names)


if company_name:
    ticker = stock_mapping.get(company_name)


    if ticker:
        st.write(f"Selected company: {company_name} (Ticker: {ticker})")
        n_years = st.slider("Years of prediction:", 1, 4)
        period = n_years * 365


        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)  # Ensure 'Date' is a column, not an index
            return data


        data_load_state = st.text("Loading data...")
        data = load_data(ticker)
        data_load_state.text("Done!")


        # Display raw data
        st.subheader("Raw data")
        st.write(data.tail())


        # ARIMA Model Training and Prediction
        st.subheader("ARIMA Model Prediction")
        train_data, test_data, closing_prices = arima.preprocess_data(data, START, TODAY)


        with st.spinner("Building the model. Please wait..."):
            # Find the best ARIMA model order and train the model
            best_model = arima.build_arima_model(train_data, test_data)
        st.success("Model building complete!")


        # Make predictions
        arima_predictions = arima.make_predictions(best_model, test_data)
        arima_predictions = arima_predictions[:len(test_data)]
        arima_predictions = arima_predictions.reset_index(drop=True)
        #arima_residuals = arima.calculate_residuals(best_model)


        # Align predictions with dates
        arima_prediction_days = len(data['Close']) - len(arima_predictions)
        arima_predicted_dates = data['Date'].iloc[arima_prediction_days:].reset_index(drop=True)


        # Create a DataFrame for ARIMA predictions
        arima_predicted_df = pd.DataFrame({
            'Date': arima_predicted_dates,
            'ARIMA Predicted Close': arima_predictions
        })


        # Display plot for ARIMA Predicted Close Prices vs Time
        st.line_chart(arima_predicted_df.set_index('Date')['ARIMA Predicted Close'])
        #st.write(arima_residuals)
       
        # Preprocess the data for LSTM
        x_train, y_train, scaler = lstm.preprocess_data(data)


        # Build and train the LSTM model with a spinner
        with st.spinner("Building and training the model. Please wait..."):
            model = lstm.build_lstm_model(x_train.shape)
            model = lstm.train_lstm_model(model, x_train, y_train)
        st.success("Model training complete!")


        # Make predictions
        predictions = lstm.make_predictions(model, data, scaler)


        # Align predictions with dates
        prediction_days = len(data['Close']) - len(predictions)
        predicted_dates = data['Date'].iloc[prediction_days:].reset_index(drop=True)


        # Create a DataFrame for predictions
        predicted_df = pd.DataFrame({
            'Date': predicted_dates,
            'Predicted Close': predictions.flatten()
        })


        # Display plot for Predicted Close Prices vs Time
        st.subheader("Predicted Close Prices vs Time")
        st.line_chart(predicted_df.set_index('Date')['Predicted Close'])
    else:
        st.write("Company ticker not found.")
else:
    st.write("Please enter a company name to search.")