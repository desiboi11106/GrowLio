"""
GrowLio — Mega Investing Dashboard (single-file)
Combines: Portfolio Risk, Stock Signals, Trade/Anomaly Analysis + Market Overview, Screener, News, DCF, Backtest, Monte Carlo.
Robust yfinance loader included to avoid "No data found" issues.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import requests
import time

# --------------------------
# App config
# --------------------------
st.set_page_config(page_title="GrowLio 📈", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("GrowLio")
st.sidebar.write("Learning-first investing dashboard — demo / portfolio tool (no trading).")

# --------------------------
# Robust loader helpers
# --------------------------
@st.cache_data(show_spinner=False)
def load_tickers_frames(ticker_list, start, end, auto_adjust=True, pause_between=0.08):
    """
    Return dict {ticker: df} with Open/High/Low/Close/Volume where available.
    Tries yf.download for groups, falls back to Ticker.history per symbol.
    """
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]
    ticker_list = [t.strip().upper() for t in ticker_list if t and str(t).strip()]

    results = {}
    # Try batch download first (faster for many tickers)
    try:
        batch = yf.download(ticker_list, start=start, end=end, group_by="ticker", auto_adjust=auto_adjust,
                            threads=True, progress=False)
    except Exception:
        batch = pd.DataFrame()

    def extract_from_batch(sym, batch_df):
        if batch_df.empty:
            return pd.DataFrame()
        if isinstance(batch_df.columns, pd.MultiIndex):
            try:
                df = batch_df[sym].copy()
            except Exception:
                df = pd.DataFrame()
        else:
            # single-ticker batch returned flat columns
            if "Close" in batch_df.columns:
                df = batch_df[["Open","High","Low","Close","Volume"]].copy()
            else:
                df = pd.DataFrame()
        return df.dropna(how="all")

    for sym in ticker_list:
        df_sym = extract_from_batch(sym, batch)
        if df_sym is None or df_sym.empty:
            # fallback to ticker history
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(start=start, end=end, auto_adjust=auto_adjust, progress=False)
                if not hist.empty:
                    # Ensure columns exist
                    for c in ["Open","High","Low","Close","Volume"]:
                        if c not in hist.columns:
                            hist[c] = np.nan
                    df_sym = hist[["Open","High","Low","Close","Volume"]].dropna(how="all")
            except Exception:
                df_sym = pd.DataFrame()
        # small pause to be nice to APIs / avoid rate limits
        time.sleep(pause_between)
        results[sym] = df_sym.sort_index()
    return results

def safe_news(ticker, limit=6):
    out = []
    try:
        n = yf.Ticker(ticker).news
        if n:
            for it in n[:limit]:
                out.append({"title": it.get("title",""), "link": it.get("link","#")})
    except Exception:
        pass
    if not out:
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
            j = requests.get(url, timeout=5).json()
            for it in j.get("news", [])[:limit]:
                out.append({"title": it.get("title",""), "link": it.get("link","#")})
        except Exception:
            pass
    return out

# --------------------------
# Utility finance functions
# --------------------------
def annualize_returns(returns_series):
    r = returns_series.dropna()
    if r.empty:
        return 0.0, 0.0, 0.0
    ann = float(r.mean() * 252)
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(ann / vol) if vol != 0 else 0.0
    return ann, vol, sharpe

def ma_signals(df, short=20, long=50):
    df = df.copy()
    df["MA_short"] = df["Close"].rolling(short).mean()
    df["MA_long"]  = df["Close"].rolling(long).mean()
    df["Signal"] = 0
    df.loc[df["MA_short"] > df["MA_long"], "Signal"] = 1
    df.loc[df["MA_short"] < df["MA_long"], "Signal"] = -1
    return df

# --------------------------
# Navigation / top-level layout
# --------------------------
page = st.sidebar.radio("Pages", [
    "Portfolio Risk",
    "Stock Signals",
    "Trade & Anomalies",
    "Market Overview",
    "Screener",
    "News & Learning",
    "DCF (Toy)",
    "Backtest (MA)",
    "Monte Carlo"
])

# --------------------------
# PAGE: Portfolio Risk
# --------------------------
if page == "Portfolio Risk":
    st.title("🧺 Portfolio Risk — GrowLio")
    st.write("Sharpe, beta (vs benchmark), diversification & portfolio risk/return.")

    with st.form("portfolio_form"):
        tickers_in = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA")
        bench = st.text_input("Benchmark ticker (for beta)", "^GSPC")
        start = st.date_input("Start", dt.date(2022,1,1))
        end = st.date_input("End", dt.date.today())
        weights_str = st.text_input("Weights (comma-separated, optional)", "")
        submitted = st.form_submit_button("Analyze Portfolio")

    if submitted:
        tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
        if not tickers:
            st.error("Add at least one ticker.")
            st.stop()

        frames = load_tickers_frames(tickers, start, end)
        good = {t:df for t,df in frames.items() if df is not None and not df.empty}
        bad  = [t for t,df in frames.items() if df is None or df.empty]
        if bad:
            st.warning(f"No data for: {', '.join(bad)} — they will be skipped.")

        if not good:
            st.error("No valid data returned for your tickers.")
            st.stop()

        # build price panel
        prices = pd.concat([good[t]["Close"].rename(t) for t in good.keys()], axis=1)
        returns = prices.pct_change().dropna()
        bench_df = load_tickers_frames([bench], start, end).get(bench, pd.DataFrame())
        bench_ret = None
        if not bench_df.empty:
            bench_ret = bench_df["Close"].pct_change().dropna()

        # per-asset metrics
        rows = []
        for t in prices.columns:
            a,v,s = annualize_returns(returns[t])
            beta = np.nan
            if bench_ret is not None and not bench_ret.empty:
                joined = pd.concat([returns[t], bench_ret], axis=1).dropna()
                if joined.shape[0] > 1:
                    cov = np.cov(joined.iloc[:,0], joined.iloc[:,1])[0,1]
                    var = np.var(joined.iloc[:,1])
                    beta = cov/var if var!=0 else np.nan
            rows.append([t, a, v, s, beta])
        stats = pd.DataFrame(rows, columns=["Ticker","AnnReturn","AnnVol","Sharpe","Beta"]).set_index("Ticker")

        # portfolio weights
        if weights_str.strip():
            try:
                w = np.array([float(x) for x in weights_str.split(",")])
                if len(w) != len(prices.columns):
                    st.error("Weights length must match tickers (or leave blank).")
                    st.stop()
                w = w / float(w.sum())
            except Exception:
                st.error("Invalid weights format.")
                st.stop()
        else:
            w = np.repeat(1/len(prices.columns), len(prices.columns))

        port_daily = (returns * w).sum(axis=1)
        pA,pV,pS = annualize_returns(port_daily)

        # layout metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Portfolio Ann. Return", f"{pA*100:.2f}%")
        c2.metric("Portfolio Ann. Vol", f"{pV*100:.2f}%")
        c3.metric("Sharpe (rf≈0%)", f"{pS:.2f}")
        # simple diversification benefit (avg single vol - portfolio vol)
        div_benefit = stats["AnnVol"].mean() - pV
        c4.metric("Diversification Benefit", f"{div_benefit*100:.2f}%")

        st.subheader("Asset stats")
        st.dataframe(stats.style.format({"AnnReturn":"{:.2%}","AnnVol":"{:.2%}","Sharpe":"{:.2f}","Beta":"{:.2f}"}))

        st.subheader("Risk-Return plot")
        fig = px.scatter(stats.reset_index(), x="AnnVol", y="AnnReturn", size=np.abs(stats["Sharpe"])+0.1, text="Ticker")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# PAGE: Stock Signals (core)
# --------------------------
elif page == "Stock Signals":
    st.title("📈 Stock Signals — GrowLio")
    col_left, col_right = st.columns([2,1])
    with col_left:
        ticker = st.text_input("Ticker", "AAPL").upper().strip()
        start = st.date_input("Start", dt.date(2022,1,1))
        end = st.date_input("End", dt.date.today())
        short = st.number_input("Short MA", value=20, min_value=3, max_value=400)
        long  = st.number_input("Long MA", value=50, min_value=5, max_value=1000)

        frames = load_tickers_frames([ticker], start, end)
        df = frames.get(ticker, pd.DataFrame())
        if df.empty:
            st.warning("No data found for this ticker / date range.")
        else:
            df = df.copy()
            df["MA_short"] = df["Close"].rolling(short).mean()
            df["MA_long"]  = df["Close"].rolling(long).mean()
            df["Signal"]   = 0
            df.loc[df["MA_short"] > df["MA_long"], "Signal"] = 1
            df.loc[df["MA_short"] < df["MA_long"], "Signal"] = -1

            # Candlestick + MAs
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
            fig.add_trace(go.Scatter(x=df.index, y=df["MA_short"], mode="lines", name=f"MA{short}"))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA_long"],  mode="lines", name=f"MA{long}"))

            buys = df[(df["Signal"]==1) & (df["Signal"].shift(1) != 1)]
            sells= df[(df["Signal"]==-1) & (df["Signal"].shift(1) != -1)]
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", marker_symbol="triangle-up", marker_size=10, marker_color="green", name="Buy"))
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", marker_symbol="triangle-down", marker_size=10, marker_color="red", name="Sell"))

            fig.update_layout(title=f"{ticker} — Price & MA Signals", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)

            # Metrics area
            r = df["Close"].pct_change().dropna()
            a,v,s = annualize_returns(r)
            st.metric("Last Price", f"${df['Close'].iloc[-1]:.2f}")
            st.metric("Ann. Return", f"{a*100:.2f}%")
            st.metric("Ann. Vol", f"{v*100:.2f}%")
            st.metric("Sharpe", f"{s:.2f}")

    with col_right:
        st.markdown("**How to read**")
        st.write("- Golden cross = short MA crosses above long MA → potential buy signal.")
        st.write("- Death cross = short MA crosses below long MA → caution/sell.")
        st.write("- Use buy markers as *learning signals* not automated trade calls.")
        st.info("Educational only — not financial advice.")

# --------------------------
# PAGE: Trade & Anomalies (core)
# --------------------------
elif page == "Trade & Anomalies":
    st.title("🧪 Trade Data & Anomaly Analysis — GrowLio")
    mode = st.radio("Source", ["Generate synthetic", "Upload CSV"], horizontal=True)
    if mode == "Upload CSV":
        f = st.file_uploader("CSV with columns: trade_id,timestamp,price,volume", type=["csv"])
        if f:
            trades = pd.read_csv(f, parse_dates=["timestamp"])
        else:
            st.info("Upload a CSV or switch to Generate synthetic.")
            trades = pd.DataFrame()
    else:
        n = st.number_input("Number of synthetic trades", value=1000, min_value=100, max_value=20000, step=100)
        base = pd.Timestamp("2024-01-01 09:30")
        times = [base + pd.Timedelta(seconds=30*i) for i in range(int(n))]
        prices = np.cumsum(np.random.normal(0, 0.2, int(n))) + 100
        vols = np.abs(np.random.normal(200, 80, int(n))).astype(int)
        trades = pd.DataFrame({"trade_id": np.arange(1, int(n)+1), "timestamp": times, "price": prices, "volume": vols})

    if trades.empty:
        st.warning("No trades to analyze.")
    else:
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        st.dataframe(trades.head(10))
        # rolling z-scores
        window = st.slider("Rolling window for z-scores", min_value=20, max_value=500, value=100)
        trades["z_price"] = (trades["price"] - trades["price"].rolling(window).mean()) / trades["price"].rolling(window).std()
        trades["z_vol"]   = (trades["volume"] - trades["volume"].rolling(window).mean()) / trades["volume"].rolling(window).std()
        trades["anomaly"] = (trades["z_price"].abs() > 3) | (trades["z_vol"].abs() > 3)

        st.subheader("Price with anomalies highlighted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trades["timestamp"], y=trades["price"], mode="lines", name="Price"))
        an = trades[trades["anomaly"]]
        if not an.empty:
            fig.add_trace(go.Scatter(x=an["timestamp"], y=an["price"], mode="markers", name="Anomaly", marker_color="red", marker_size=8))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rolling average volume")
        trades["roll_vol"] = trades["volume"].rolling(50).mean()
        st.line_chart(trades.set_index("timestamp")["roll_vol"])

        st.write(f"Anomalies detected: {int(trades['anomaly'].sum())}")

# --------------------------
# PAGE: Market Overview
# --------------------------
elif page == "Market Overview":
    st.title("🌍 Market Overview")
    indices = ["^GSPC","^DJI","^IXIC","^RUT","^VIX"]
    frames_idx = load_tickers_frames(indices, dt.date(2022,1,1), dt.date.today())
    good_idx = {k:v for k,v in frames_idx.items() if v is not None and not v.empty}
    if not good_idx:
        st.warning("Index data not available.")
    else:
        px = pd.concat([good_idx[k]["Close"].rename(k) for k in good_idx.keys()], axis=1)
        latest = px.iloc[-1]
        colz = st.columns(len(latest))
        for i,k in enumerate(latest.index):
            prev = px.iloc[0][k]
            change = (latest[k]-prev)/prev*100 if prev!=0 else 0.0
            colz[i].metric(k, f"{latest[k]:.0f}", f"{change:.2f}%")
        st.line_chart(px)

# --------------------------
# PAGE: Screener
# --------------------------
elif page == "Screener":
    st.title("🧮 Screener — GrowLio")
    tickers_in = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA, AMZN")
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    if st.button("Run Screener"):
        rows = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                pe = info.get("trailingPE", np.nan)
                dy = info.get("dividendYield", np.nan)
                mcap = info.get("marketCap", np.nan)
                rows.append([t, pe, dy, mcap])
            except Exception:
                rows.append([t, np.nan, np.nan, np.nan])
        df = pd.DataFrame(rows, columns=["Ticker","P/E","DividendYield","MarketCap"]).set_index("Ticker")
        st.dataframe(df)

# --------------------------
# PAGE: News & Learning
# --------------------------
elif page == "News & Learning":
    st.title("📰 News & Learning")
    tick = st.text_input("Ticker for news", "AAPL").upper().strip()
    if st.button("Fetch News"):
        news = safe_news(tick, limit=8)
        if not news:
            st.write("No news found.")
        else:
            for it in news:
                st.markdown(f"- [{it['title']}]({it['link']})")
    st.markdown("---")
    st.subheader("Mini Learning Module")
    st.write("Example: Moving averages — a moving average smooths price and shows trends. A golden cross (short MA > long MA) often suggests an emerging uptrend.")

# --------------------------
# PAGE: DCF (toy)
# --------------------------
elif page == "DCF (Toy)":
    st.title("💵 Toy DCF (Educational)")
    ticker = st.text_input("Ticker (optional)", "AAPL")
    fcf = st.number_input("Starting FCF ($M)", value=10000.0)
    g = st.number_input("Growth rate % (yrs 1-5)", value=8.0)/100
    wacc = st.number_input("Discount rate (WACC %) ", value=10.0)/100
    tg = st.number_input("Terminal growth %", value=2.5)/100
    shares = st.number_input("Shares outstanding (millions)", value=16000.0)
    if st.button("Run DCF"):
        years = np.arange(1,6)
        fcf_proj = [fcf * ((1+g)**i) for i in years]
        pv = [fcf_proj[i]/((1+wacc)**(i+1)) for i in range(len(fcf_proj))]
        term = (fcf_proj[-1]*(1+tg))/(wacc - tg) if wacc>tg else np.nan
        pv_term = term/((1+wacc)**5) if not np.isnan(term) else 0
        ev = np.nansum(pv) + (pv_term if pd.notna(pv_term) else 0)
        per_share = ev / shares if shares>0 else np.nan
        st.metric("Equity value per share ($)", f"{per_share:,.2f}")

# --------------------------
# PAGE: Backtest (MA)
# --------------------------
elif page == "Backtest (MA)":
    st.title("🔁 Backtest: MA Crossover")
    ticker = st.text_input("Ticker", "AAPL")
    start = st.date_input("Start", dt.date(2022,1,1))
    end = st.date_input("End", dt.date.today())
    short = st.number_input("Short MA", value=20)
    long = st.number_input("Long MA", value=50)
    if st.button("Run Backtest"):
        frames = load_tickers_frames([ticker], start, end)
        df = frames.get(ticker, pd.DataFrame())
        if df.empty:
            st.error("No data for backtest.")
        else:
            df = df.copy()
            df["MA_short"] = df["Close"].rolling(short).mean()
            df["MA_long"]  = df["Close"].rolling(long).mean()
            df["pos"] = np.where(df["MA_short"] > df["MA_long"], 1, 0)
            df["strat_ret"] = df["pos"].shift(1) * df["Close"].pct_change()
            eq_strat = (1+df["strat_ret"].fillna(0)).cumprod()
            eq_bh = (1+df["Close"].pct_change().fillna(0)).cumprod()
            st.line_chart(pd.DataFrame({"Strategy":eq_strat, "Buy&Hold":eq_bh}))

# --------------------------
# PAGE: Monte Carlo
# --------------------------
elif page == "Monte Carlo":
    st.title("🎲 Monte Carlo Portfolio")
    tickers_in = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA")
    sims = st.number_input("Simulations", value=300, min_value=50)
    days = st.number_input("Days horizon", value=252, min_value=30)
    start = st.date_input("Hist start", dt.date(2022,1,1))
    end = st.date_input("Hist end", dt.date.today())
    if st.button("Simulate"):
        tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
        frames = load_tickers_frames(tickers, start, end)
        good = {t:df for t,df in frames.items() if not df.empty}
        if not good:
            st.error("No data for tickers.")
        else:
            px = pd.concat([good[t]["Close"].rename(t) for t in good.keys()], axis=1)
            ret = px.pct_change().dropna()
            mu = ret.mean().values
            cov = ret.cov().values
            w = np.repeat(1/len(px.columns), len(px.columns))
            paths = np.zeros((int(days)+1, sims))
            paths[0,:] = 1.0
            chol = np.linalg.cholesky(cov + 1e-12*np.eye(len(px.columns)))
            for t in range(1, int(days)+1):
                z = np.random.normal(size=(len(px.columns), sims))
                correlated = chol @ z
                step = (mu/252) + (correlated * (np.sqrt(1/252)))
                step_ret = (w @ step)
                paths[t,:] = paths[t-1,:] * (1+step_ret)
            df_paths = pd.DataFrame(paths)
            st.line_chart(df_paths.iloc[:,:min(sims,50)])
            st.metric("Median outcome (final)", f"{np.percentile(paths[-1,:],50):.2f}")

# Footer
st.markdown("---")
st.caption("© GrowLio — educational demo. Data via Yahoo Finance (yfinance). Not investment advice.")
 
