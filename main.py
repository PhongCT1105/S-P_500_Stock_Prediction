import streamlit as st
from datetime import date
import yfinance as yf

# Constants for data range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title("Stock Prediction App")

# Predefined dictionary of company names and ticker symbols
# Ideally, replace this with a more comprehensive list or data source
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
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text("Loading data...")
        data = load_data(ticker)
        data_load_state.text("Done!")

        # Display raw data
        st.subheader("Raw data")
        st.write(data.tail())
    else:
        st.write("Company ticker not found.")
else:
    st.write("Please enter a company name to search.")
