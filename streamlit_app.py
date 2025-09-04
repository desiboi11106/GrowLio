# app_fixed_stocks.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import requests
import plotly.graph_objects as go
import time

st.set_page_config(page_title="GrowLio - Stocks (Fixed loader)", layout="wide")
st.title("📊 GrowLio - Stock Viewer (robust loader)")

# ---------------------
# Helper: robust loader
# ---------------------
@st.cache_data(show_spinner=False)
def load_ticker_close_frames(ticker_list, start, end, auto_adjust=True, pause_between=0.1):
    """
    Returns dict: {ticker: df_close_or_ohlc}
    Each value is a DataFrame with columns including Close, Open, High, Low (if available).
    Will try yf.download first, then fallback to Ticker.history per ticker.
    """
    results = {}
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]

    # first try yf.download for all tickers (faster)
    try:
        raw = yf.download(ticker_list, start=start, end=end, group_by="ticker", auto_adjust=auto_adjust, threads=True, progress=False)
    except Exception:
        raw = pd.DataFrame()

    # Helper to extract per-ticker df from raw (download) result
    def extract_from_raw(t, raw_df):
        if raw_df.empty:
            return pd.DataFrame()
        # multiindex columns (ticker, colname)
        if isinstance(raw_df.columns, pd.MultiIndex):
            try:
                df = raw_df[t].copy()
            except Exception:
                df = pd.DataFrame()
        else:
            # single-ticker download returns flat columns
            if "Close" in raw_df.columns:
                df = raw_df[["Open","High","Low","Close","Volume"]].copy()
            else:
                df = pd.DataFrame()
        return df.dropna(how="all")

    # try to extract each ticker
    for t in ticker_list:
        df_t = extract_from_raw(t, raw)
        if df_t is None or df_t.empty:
            # fallback to single ticker history
            try:
                tk = yf.Ticker(t)
                df2 = tk.history(start=start, end=end, auto_adjust=auto_adjust, progress=False)
                # ensure standard columns
                if not df2.empty:
                    # sometimes history returns only 'Close' -> add other cols if missing
                    for col in ["Open","High","Low","Close","Volume"]:
                        if col not in df2.columns:
                            df2[col] = np.nan
                    df_t = df2[["Open","High","Low","Close","Volume"]].dropna(how="all")
            except Exception:
                df_t = pd.DataFrame()
        # small pause to avoid rate-limits on many tickers
        time.sleep(pause_between)
        results[t] = df_t.sort_index()
    return results

# ---------------------
# UI: Inputs
# ---------------------
st.sidebar.header("Stock Settings")
tickers_input = st.sidebar.text_input("Enter Stock tickers (comma-separated)", "AAPL, MSFT, TSLA")
start = st.sidebar.date_input("Start Date", dt.date(2023, 1, 1))
end = st.sidebar.date_input("End Date", dt.date.today())

# sanitize tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if len(tickers) == 0:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()

# basic validation: if end <= start or end in future
if end <= start:
    st.error("End date must be after start date.")
    st.stop()
if end > dt.date.today():
    st.warning("End date is in the future — using today's date instead.")
    end = dt.date.today()

# ---------------------
# Load data
# ---------------------
with st.spinner("Downloading data... (this may take a few seconds)"):
    frames = load_ticker_close_frames(tickers, start, end)

# report tickers with no data
no_data = [t for t, df in frames.items() if df is None or df.empty]
have_data = [t for t, df in frames.items() if not (df is None or df.empty)]

if len(have_data) == 0:
    st.error("No data found for any ticker. Check tickers and date range.")
    if no_data:
        st.write("Tried tickers:", ", ".join(no_data))
    st.stop()

if no_data:
    st.warning(f"No data found for: {', '.join(no_data)} — those tickers will be skipped.")

# ---------------------
# Metrics: last close, percent change since start
# ---------------------
st.subheader("📈 Stock Metrics")
cols = st.columns(len(have_data))
for i, t in enumerate(have_data):
    df = frames[t]
    # ensure Close exists
    if "Close" not in df.columns or df["Close"].dropna().empty:
        cols[i].warning(f"No close price for {t}")
        continue
    last_close = float(df["Close"].iloc[-1])
    first_close = float(df["Close"].iloc[0])
    change = (last_close - first_close)/first_close * 100.0 if first_close != 0 else 0.0
    cols[i].metric(t, f"${last_close:.2f}", f"{change:.2f}%")

# ---------------------
# Comparison chart (multi tickers)
# ---------------------
st.subheader("📉 Price Comparison")
fig = go.Figure()
for t in have_data:
    df = frames[t]
    if "Close" in df.columns and not df["Close"].dropna().empty:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=t))
fig.update_layout(title="Close Price Comparison", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ---------------------
# Detailed per-ticker analysis (MA, signals, volatility)
# ---------------------
st.subheader("🔍 Detailed Analysis")
for t in have_data:
    st.markdown(f"### {t}")
    df = frames[t].copy()
    if df.empty or "Close" not in df.columns or df["Close"].dropna().empty:
        st.write("No usable data for this ticker.")
        continue

    # compute MAs & signals
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Signal"] = 0
    df.loc[df["MA50"] > df["MA200"], "Signal"] = 1
    df.loc[df["MA50"] < df["MA200"], "Signal"] = -1
    buys = df[(df["Signal"]==1) & (df["Signal"].shift(1) != 1)]
    sells= df[(df["Signal"]==-1) & (df["Signal"].shift(1) != -1)]

    # candlestick using plotly if OHLC available
    if {"Open","High","Low","Close"}.issubset(df.columns) and not df[["Open","High","Low","Close"]].isnull().all().all():
        fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                                              open=df["Open"], high=df["High"],
                                              low=df["Low"], close=df["Close"])])
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

    fig2.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig2.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))
    if not buys.empty:
        fig2.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                  marker_symbol="triangle-up", marker_size=10, marker_color="green", name="Buy"))
    if not sells.empty:
        fig2.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                  marker_symbol="triangle-down", marker_size=10, marker_color="red", name="Sell"))

    fig2.update_layout(title=f"{t} Price & MA Signals", xaxis_title="Date")
    st.plotly_chart(fig2, use_container_width=True)

    # volatility plot
    df["Volatility"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df["Volatility"], mode="lines", name="Volatility"))
    fig_vol.update_layout(title=f"{t} 20-Day Annualized Volatility")
    st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------
# News: safe fallback
# ---------------------
st.subheader("📰 News (per ticker)")
def fetch_news_safe(ticker, limit=5):
    out = []
    try:
        n = yf.Ticker(ticker).news
        if n:
            for item in n[:limit]:
                out.append({"title": item.get("title",""), "link": item.get("link","#")})
    except Exception:
        pass
    if not out:
        # fallback to Yahoo search endpoint (may vary in reliability)
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
            j = requests.get(url, timeout=5).json()
            for it in j.get("news", [])[:limit]:
                out.append({"title": it.get("title",""), "link": it.get("link","#")})
        except Exception:
            pass
    return out

for t in have_data:
    st.markdown(f"#### {t} News")
    news = fetch_news_safe(t, limit=5)
    if not news:
        st.write("No news found.")
    else:
        for it in news:
            st.markdown(f"- [{it['title']}]({it['link']})")

# Footer
st.caption("Note: Data via yfinance/Yahoo. If a ticker returns no data, check ticker spelling, delisted symbols, or the selected date range (markets closed on weekends/holidays).")
