import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Signals — GrowLio", layout="wide")

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start = st.sidebar.date_input("Start", pd.to_datetime("2022-01-01"))
end = st.sidebar.date_input("End", pd.to_datetime("today"))

st.sidebar.header("Moving Averages")
short_window = st.sidebar.number_input("Short MA", 5, 100, 20)
long_window = st.sidebar.number_input("Long MA", 10, 300, 50)

st.sidebar.header("RSI Settings")
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, 14)

# ---------------------------
# Data Fetching Function
# ---------------------------
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end)
        if data.empty:
            return None
        return data
    except Exception:
        return None

data = load_data(ticker, start, end)

if data is None:
    st.error("❌ No data found. Please check ticker or date range.")
    st.stop()

# ---------------------------
# Technical Indicators
# ---------------------------
# Moving Averages
data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()

# RSI
delta = data["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(rsi_period).mean()
avg_loss = pd.Series(loss).rolling(rsi_period).mean()
rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# Signals
data["Signal"] = np.where(data["SMA_Short"] > data["SMA_Long"], 1, 0)
data["Position"] = data["Signal"].diff()

# ---------------------------
# Layout
# ---------------------------
st.title("📈 Stock Signals — GrowLio")

tab1, tab2, tab3 = st.tabs(["📊 Chart", "📑 Indicators", "💡 Strategy"])

with tab1:
    st.subheader(f"Price & Moving Averages for {ticker}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Close Price", alpha=0.7)
    ax.plot(data.index, data["SMA_Short"], label=f"SMA {short_window}", alpha=0.7)
    ax.plot(data.index, data["SMA_Long"], label=f"SMA {long_window}", alpha=0.7)

    # Buy/Sell markers
    ax.plot(data[data["Position"] == 1].index,
            data["SMA_Short"][data["Position"] == 1],
            "^", markersize=10, color="g", label="Buy Signal")
    ax.plot(data[data["Position"] == -1].index,
            data["SMA_Short"][data["Position"] == -1],
            "v", markersize=10, color="r", label="Sell Signal")

    ax.set_title(f"{ticker} Price Chart")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("RSI Indicator")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data["RSI"], label="RSI", color="purple")
    ax.axhline(70, linestyle="--", alpha=0.5, color="red")
    ax.axhline(30, linestyle="--", alpha=0.5, color="green")
    ax.set_title("Relative Strength Index (RSI)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Raw Data Preview")
    st.dataframe(data.tail(20))

with tab3:
    st.subheader("💡 Strategy Insights")
    st.write("""
    GrowLio combines **three core investing ideas** into one app:
    
    1. **Trend Following** — Using short-term vs long-term moving averages for buy/sell signals.  
    2. **Momentum** — Applying RSI to gauge overbought (70+) and oversold (30-) conditions.  
    3. **Historical Analysis** — Looking at stock price movements across user-selected date ranges.  

    ✅ Next steps could include adding **user login, database tracking of trades, portfolio optimization, and news sentiment analysis**.
    """)

