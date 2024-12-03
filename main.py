import streamlit as st
from datetime import date
import yfinance as yf
import lstm  # Import the LSTM module (lstm.py)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants for data range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App Styling
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Google Analytics script
# Google Analytics script
# GA_TRACKING_CODE = """
# <script async src="https://www.googletagmanager.com/gtag/js?id=G-PSQ3XQ4ZQR"></script>
# <script>
#   window.dataLayer = window.dataLayer || [];
#   function gtag(){dataLayer.push(arguments);}
#   gtag('js', new Date());
#   gtag('config', 'G-PSQ3XQ4ZQR');
# </script>
# """


# # Inject Google Analytics
# def inject_google_analytics():
#     st.markdown(
#         GA_TRACKING_CODE,
#         unsafe_allow_html=True
#     )

# # Call the function to inject GA
# inject_google_analytics()

# Title
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            color: #4CAF50;
            font-family: Arial, sans-serif;
        }
        .subheader {
            font-family: Arial, sans-serif;
            color: #333333;
        }
    </style>
    <h1 class='title'>📈 Stock Prediction App 🚀</h1>
    <p class='subheader'>Analyze and Predict Stock Trends with Insights</p>
    """,
    unsafe_allow_html=True
)

# Sidebar for Stock Selection
st.sidebar.header("Stock Selection")
stock_mapping = {
    "AbbVie": "ABBV",
    "Accenture": "ACN",
    "Adobe": "ADBE",
    "Advanced Micro Devices": "AMD",
    "Alphabet Class A": "GOOGL",
    "Alphabet Class C": "GOOG",
    "Amazon": "AMZN",
    "Apple": "AAPL",
    "Bank of America": "BAC",
    "Berkshire Hathaway": "BRK.B",
    "Broadcom": "AVGO",
    "Chevron": "CVX",
    "Cisco Systems": "CSCO",
    "Citigroup": "C",
    "Coca-Cola": "KO",
    "Comcast": "CMCSA",
    "Costco": "COST",
    "Exxon Mobil": "XOM",
    "Home Depot": "HD",
    "Intel": "INTC",
    "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ",
    "Mastercard": "MA",
    "McDonald's": "MCD",
    "Meta Platforms": "META",
    "Microsoft": "MSFT",
    "Nike": "NKE",
    "NVIDIA": "NVDA",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Procter & Gamble": "PG",
    "Salesforce": "CRM",
    "Tesla": "TSLA",
    "Texas Instruments": "TXN",
    "The Boeing Company": "BA",
    "The Charles Schwab Corporation": "SCHW",
    "The Goldman Sachs Group": "GS",
    "The Home Depot": "HD",
    "The Procter & Gamble Company": "PG",
    "The Walt Disney Company": "DIS",
    "Thermo Fisher Scientific": "TMO",
    "UnitedHealth Group": "UNH",
    "Verizon": "VZ",
    "Visa": "V",
    "Walt Disney": "DIS",
}


company_name = st.sidebar.selectbox(
    "Select a Company", options=[""] + list(stock_mapping.keys())
)

if company_name:
    ticker = stock_mapping[company_name]

    if ticker:
        st.sidebar.success(f"Selected: {company_name} ({ticker})")
        n_years = st.sidebar.slider(
            "Years of Prediction:", 1, 4, help="Choose the number of years to predict."
        )
        period = n_years * 365

        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

        st.sidebar.text("Loading stock data...")
        data = load_data(ticker)
        st.sidebar.success("Data loaded successfully!")

        # Data Analysis Section
        st.subheader(f"📊 Data Analysis and Insights for {company_name}")
        st.write("### Key Metrics")
        st.write(f"Date Range: {START} to {TODAY}")
        st.write(f"Total Data Points: {data.shape[0]}")

        # Calculate key statistics
        mean_price = data["Close"].mean()
        max_price = data["Close"].max()
        min_price = data["Close"].min()
        st.write(
            f"""
            - **Average Closing Price**: ${float(mean_price):.2f}
            - **Highest Closing Price**: ${float(max_price):.2f}
            - **Lowest Closing Price**: ${float(min_price):.2f}
            """
        )

        # Display historical trends
        st.write("### Historical Data Visualization")
        st.line_chart(data.set_index("Date")["Close"])

        # Volatility (Standard Deviation)
        volatility = data["Close"].std()
        st.write(f"- **Price Volatility (Standard Deviation)**: ${float(volatility):.2f}")

        # Performance Metrics Section (LSTM Predictions)
        st.subheader(f"📈 Predictions and Performance Metrics for {company_name}")
        x_train, y_train, scaler = lstm.preprocess_data(data)

        with st.spinner("Building and training the model..."):
            model = lstm.build_lstm_model(x_train.shape)
            model = lstm.train_lstm_model(model, x_train, y_train)
        st.success("Model training complete!")

        # Predictions
        predictions = lstm.make_predictions(model, data, scaler)
        prediction_days = len(data["Close"]) - len(predictions)
        predicted_dates = data["Date"].iloc[prediction_days:].reset_index(drop=True)

        # Calculate confidence intervals (±5% of predictions)
        confidence_margin = 0.05  # 5% margin
        upper_bound = predictions * (1 + confidence_margin)
        lower_bound = predictions * (1 - confidence_margin)

        # Update the prediction DataFrame
        predicted_df = pd.DataFrame({
            "Date": predicted_dates,
            "Predicted Close": predictions.flatten(),
            "Upper Bound": upper_bound.flatten(),
            "Lower Bound": lower_bound.flatten()
        })

        # Display Performance Metrics
        true_values = data["Close"].iloc[-len(predictions):].values
        mae = np.mean(np.abs(predictions.flatten() - true_values))
        mse = np.mean((predictions.flatten() - true_values) ** 2)
        rmse = np.sqrt(mse)

        st.write(
            f"""
            - **Mean Absolute Error (MAE)**: ${mae:.2f}
            - **Mean Squared Error (MSE)**: ${mse:.2f}
            - **Root Mean Squared Error (RMSE)**: ${rmse:.2f}
            """
        )

        # Plot predictions with confidence intervals
        st.subheader("Predicted Close Prices with Confidence Intervals")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot predicted close prices
        ax.plot(predicted_df["Date"], predicted_df["Predicted Close"], label="Predicted", color="blue")
        # Plot confidence intervals
        ax.fill_between(
            predicted_df["Date"],
            predicted_df["Lower Bound"],
            predicted_df["Upper Bound"],
            color="blue",
            alpha=0.2,
            label="Confidence Interval (±5%)"
        )

        # Customize plot
        ax.set_title(f"Predicted Close Prices for {company_name} with Confidence Intervals", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Close Price (USD)", fontsize=12)
        ax.legend(loc="upper left")
        plt.xticks(rotation=45)

        # Streamlit display
        st.pyplot(fig)
    else:
        st.error("Invalid company selection.")
else:
    st.info("Please select a company to analyze.")
