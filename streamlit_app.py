# GrowLio: Mega Investing Dashboard
# Core pillars preserved:
# 1) Portfolio Risk (Sharpe, beta, diversification, risk/return)
# 2) Trade Data Analysis (liquidity patterns, anomalies)
# 3) Stock Signals (MAs + buy/sell markers)
# + extras: Market Overview, Screener, News, DCF, Backtest, Monte Carlo

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import requests
import yfinance as yf

st.set_page_config(page_title="GrowLio 📈", layout="wide")

# ----------------------------
# Helpers & Cache
# ----------------------------
@st.cache_data(show_spinner=False)
def dl_prices(tickers, start, end, auto_adjust=True):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(
        tickers, start=start, end=end, group_by="ticker", auto_adjust=auto_adjust, threads=True
    )
    # Normalize to simple DataFrame with Close for single/multi
    frames = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[t]["Close"].dropna().to_frame(name=t)
            else:
                # single ticker returns flat columns
                close = data["Close"].dropna().to_frame(name=t)
            frames[t] = close
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames.values(), axis=1).sort_index()
    return combined

@st.cache_data(show_spinner=False)
def dl_ohlc(ticker, start, end, auto_adjust=True):
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust)
    return df

def annualize_returns(returns_series):
    daily = returns_series.dropna()
    if daily.empty:
        return 0.0, 0.0, 0.0
    ann_ret = float(daily.mean() * 252.0)
    ann_vol = float(daily.std() * np.sqrt(252.0))
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0.0
    return ann_ret, ann_vol, sharpe

def compute_beta(asset_ret, bench_ret):
    a = asset_ret.dropna()
    b = bench_ret.dropna()
    joined = pd.concat([a, b], axis=1).dropna()
    if joined.shape[0] < 2:
        return np.nan
    cov = np.cov(joined.iloc[:,0], joined.iloc[:,1])[0,1]
    var = np.var(joined.iloc[:,1])
    return cov / var if var != 0 else np.nan

def ma_signal(df, short=20, long=50):
    out = df.copy()
    out["MA_Short"] = out["Close"].rolling(short).mean()
    out["MA_Long"]  = out["Close"].rolling(long).mean()
    out["Signal"] = 0
    out.loc[out["MA_Short"] > out["MA_Long"], "Signal"] = 1
    out.loc[out["MA_Short"] < out["MA_Long"], "Signal"] = -1
    return out

def bollinger_bands(close, window=20, num_std=2):
    m = close.rolling(window).mean()
    s = close.rolling(window).std()
    upper = m + num_std*s
    lower = m - num_std*s
    return m, upper, lower

def yahoo_news(ticker, limit=6):
    # Try yfinance .news first; fall back to Yahoo search endpoint if missing
    items = []
    try:
        n = yf.Ticker(ticker).news
        if n:
            for k in n[:limit]:
                items.append({"title": k.get("title", ""), "link": k.get("link", "")})
    except Exception:
        pass
    if not items:
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
            js = requests.get(url, timeout=5).json()
            for n in js.get("news", [])[:limit]:
                items.append({"title": n.get("title", ""), "link": n.get("link", "#")})
        except Exception:
            pass
    return items

# ----------------------------
# Sidebar (global)
# ----------------------------
st.sidebar.title("GrowLio 📈")
st.sidebar.caption("Learning-first investing dashboard (free).")

page = st.sidebar.radio(
    "Navigate",
    [
        "🧺 Portfolio Risk",
        "📈 Stock Signals",
        "🧪 Trade / Anomaly Analysis",
        "🌍 Market Overview",
        "🧮 Screener",
        "📰 News",
        "💵 DCF (Toy)",
        "🔁 Backtest (MA Crossover)",
        "🎲 Monte Carlo (Portfolio)"
    ]
)

default_start = dt.date(2022, 1, 1)
default_end   = dt.date.today()

# ----------------------------
# 🧺 Portfolio Risk (core #1)
# ----------------------------
if page == "🧺 Portfolio Risk":
    st.title("🧺 Portfolio Risk Dashboard — GrowLio")
    st.write("Sharpe, beta, diversification, and risk–return tradeoffs.")

    tickers_in = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA, NVDA")
    bench = st.text_input("Benchmark (for beta)", "^GSPC")
    start = st.date_input("Start", default_start)
    end   = st.date_input("End", default_end)

    weights_str = st.text_input("Weights (comma-separated, same order; optional)", "")
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    w = None
    if weights_str.strip():
        try:
            w = np.array([float(x) for x in weights_str.split(",")])
            if len(w) != len(tickers):
                st.error("Weights length must match number of tickers.")
                st.stop()
            if w.sum() == 0:
                st.error("Weights must sum to > 0.")
                st.stop()
            w = w / w.sum()
        except Exception:
            st.error("Invalid weights. Use numbers separated by commas.")
            st.stop()

    prices = dl_prices(tickers, start, end)
    if prices.empty:
        st.warning("No price data found.")
        st.stop()

    ret = prices.pct_change().dropna()
    bench_px = dl_prices([bench], start, end)
    bench_ret = bench_px.pct_change().dropna().iloc[:,0] if not bench_px.empty else pd.Series(dtype=float)

    # Per-asset stats
    rows = []
    for t in prices.columns:
        a, v, s = annualize_returns(ret[t])
        b = compute_beta(ret[t], bench_ret) if not bench_ret.empty else np.nan
        rows.append([t, a, v, s, b])
    stats = pd.DataFrame(rows, columns=["Ticker","Ann.Return","Ann.Vol","Sharpe","Beta"]).set_index("Ticker")

    # Portfolio stats
    if w is None:
        w = np.repeat(1/len(tickers), len(tickers))
    port_ret_daily = (ret * w).sum(axis=1)
    pA, pV, pS = annualize_returns(port_ret_daily)
    div_benefit = float(stats["Ann.Vol"].mean() - pV)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Ann. Return", f"{pA*100:.2f}%")
    c2.metric("Portfolio Ann. Vol", f"{pV*100:.2f}%")
    c3.metric("Sharpe (rf≈0%)", f"{pS:.2f}")
    c4.metric("Diversification Benefit", f"{div_benefit*100:.2f}%")

    st.subheader("Asset Risk/Return")
    st.dataframe(stats.style.format({"Ann.Return":"{:.2%}","Ann.Vol":"{:.2%}","Sharpe":"{:.2f}","Beta":"{:.2f}"}))

    # Risk-Return Scatter
    fig = px.scatter(
        stats.reset_index(), x="Ann.Vol", y="Ann.Return", text="Ticker", size=np.abs(stats["Sharpe"])+0.1,
        labels={"Ann.Vol":"Ann.Volatility","Ann.Return":"Ann.Return"}, title="Risk–Return"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 📈 Stock Signals (core #3)
# ----------------------------
elif page == "📈 Stock Signals":
    st.title("📈 Stock Signals — GrowLio")
    st.write("Moving averages with buy/sell regimes, candlesticks, and Bollinger Bands.")

    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    start  = st.date_input("Start", default_start)
    end    = st.date_input("End", default_end)
    short  = st.number_input("Short MA", value=20, min_value=3, max_value=200)
    long   = st.number_input("Long MA", value=50, min_value=5, max_value=400)

    ohlc = dl_ohlc(ticker, start, end)
    if ohlc.empty:
        st.warning("No data found.")
        st.stop()

    sig = ma_signal(ohlc.copy(), short=short, long=long)
    mid, upper, lower = bollinger_bands(sig["Close"], window=20, num_std=2)

    # Candlestick + MAs + BBands
    fig = go.Figure(data=[go.Candlestick(
        x=sig.index, open=sig["Open"], high=sig["High"], low=sig["Low"], close=sig["Close"], name="OHLC"
    )])
    fig.add_trace(go.Scatter(x=sig.index, y=sig["MA_Short"], name=f"MA {short}", mode="lines"))
    fig.add_trace(go.Scatter(x=sig.index, y=sig["MA_Long"],  name=f"MA {long}", mode="lines"))
    fig.add_trace(go.Scatter(x=sig.index, y=upper, name="BBand Upper", mode="lines"))
    fig.add_trace(go.Scatter(x=sig.index, y=mid,   name="BBand Mid",   mode="lines"))
    fig.add_trace(go.Scatter(x=sig.index, y=lower, name="BBand Lower", mode="lines"))
    st.plotly_chart(fig, use_container_width=True)

    # Buy/Sell markers
    buys = sig.loc[(sig["Signal"] == 1) & (sig["Signal"].shift(1) != 1)]
    sells= sig.loc[(sig["Signal"] == -1) & (sig["Signal"].shift(1) != -1)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sig.index, y=sig["Close"], mode="lines", name="Close"))
    fig2.add_trace(go.Scatter(x=buys.index,  y=buys["Close"],  mode="markers", marker_symbol="triangle-up", marker_size=12, name="Buy"))
    fig2.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", marker_symbol="triangle-down", marker_size=12, name="Sell"))
    fig2.update_layout(title="Buy / Sell Regime Markers")
    st.plotly_chart(fig2, use_container_width=True)

    # Metrics & hint
    ret = sig["Close"].pct_change()
    a, v, s = annualize_returns(ret)
    last_regime = sig["Signal"].iloc[-1]
    if last_regime == 1:
        lab = "✅ Buy regime (short MA above long MA)"
        tip = "Balanced hint: look for pullbacks near short MA with a stop below long MA."
    elif last_regime == -1:
        lab = "❌ Avoid/Sell regime (short MA below long MA)"
        tip = "Balanced hint: avoid entries until trend improves."
    else:
        lab = "⏸️ Neutral"
        tip = "Wait for a clear MA crossover confirmation."

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price", f"${float(sig['Close'].iloc[-1]):.2f}")
    c2.metric("Ann. Return", f"{a*100:.2f}%")
    c3.metric("Ann. Vol", f"{v*100:.2f}%")
    c4.metric("Sharpe (rf≈0%)", f"{s:.2f}")
    st.success(f"Current Signal for {ticker}: {lab}")
    with st.expander("🧭 How to read this"):
        st.markdown(
            "- Golden cross → potential uptrend.\n"
            "- Death cross → potential downtrend.\n"
            "- Annualized metrics from daily returns (252 trading days/year).\n"
            "- **Disclaimer:** Educational only."
        )

# ----------------------------
# 🧪 Trade / Anomaly Analysis (core #2)
# ----------------------------
elif page == "🧪 Trade / Anomaly Analysis":
    st.title("🧪 Trade & Anomaly Analytics — GrowLio")
    st.write("Upload trade-level data or generate synthetic data, then detect outliers & liquidity patterns.")

    mode = st.radio("Data Source", ["Upload CSV", "Generate Sample"], horizontal=True)
    if mode == "Upload CSV":
        f = st.file_uploader("CSV with columns: trade_id, timestamp, price, volume", type=["csv"])
        if f is None:
            st.stop()
        trades = pd.read_csv(f, parse_dates=["timestamp"])
    else:
        # synthetic data
        np.random.seed(7)
        n=1000
        base_time = pd.Timestamp("2024-01-01 09:30")
        times = [base_time + pd.Timedelta(minutes=i) for i in range(n)]
        prices = np.cumsum(np.random.normal(0, 0.2, n)) + 100
        vols   = np.abs(np.random.normal(200, 100, n)).astype(int)
        trades = pd.DataFrame({
            "trade_id": np.arange(1, n+1),
            "timestamp": times,
            "price": prices,
            "volume": vols
        })

    trades = trades.sort_values("timestamp").reset_index(drop=True)
    st.dataframe(trades.head(20))

    # Liquidity / anomaly flags
    trades["ret"] = trades["price"].pct_change()
    z_price = (trades["price"] - trades["price"].rolling(100).mean()) / trades["price"].rolling(100).std()
    z_ret   = (trades["ret"]   - trades["ret"].rolling(100).mean())   / trades["ret"].rolling(100).std()
    z_vol   = (trades["volume"]- trades["volume"].rolling(100).mean())/ trades["volume"].rolling(100).std()
    trades["z_price"] = z_price
    trades["z_ret"]   = z_ret
    trades["z_vol"]   = z_vol
    trades["anomaly"] = (np.abs(z_ret) > 3) | (np.abs(z_vol) > 3)

    st.subheader("Price over time (anomalies highlighted)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades["timestamp"], y=trades["price"], mode="lines", name="price"))
    an = trades[trades["anomaly"]]
    fig.add_trace(go.Scatter(x=an["timestamp"], y=an["price"], mode="markers", name="anomaly"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Liquidity (rolling volume)")
    trades["roll_vol"] = trades["volume"].rolling(50).mean()
    st.line_chart(trades.set_index("timestamp")["roll_vol"])

    st.info("Heuristics: |z(ret)|>3 or |z(volume)|>3 → flagged. Tweak windows for sensitivity.")

# ----------------------------
# 🌍 Market Overview
# ----------------------------
elif page == "🌍 Market Overview":
    st.title("🌍 Market Overview — GrowLio")
    idx = ["^GSPC","^NDX","^DJI","^RUT","^VIX"]
    px = dl_prices(idx, default_start, default_end)
    if px.empty:
        st.warning("No index data.")
        st.stop()
    last = px.iloc[-1]
    first= px.iloc[0]
    cols = st.columns(len(idx))
    for i,t in enumerate(px.columns):
        chg = (last[t]-first[t])/first[t]*100
        cols[i].metric(t, f"{last[t]:.0f}", f"{chg:.2f}%")
    st.line_chart(px)

# ----------------------------
# 🧮 Screener (simple fundamentals via yfinance)
# ----------------------------
elif page == "🧮 Screener":
    st.title("🧮 Simple Screener — GrowLio")
    tickers_in = st.text_input("Tickers to screen (comma-separated)", "AAPL, MSFT, TSLA, AMZN, META")
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info  # note: may be slow / partial depending on yfinance version
            pe   = info.get("trailingPE", np.nan)
            dy   = info.get("dividendYield", np.nan)
            mcap = info.get("marketCap", np.nan)
            rows.append([t, pe, dy, mcap])
        except Exception:
            rows.append([t, np.nan, np.nan, np.nan])
    df = pd.DataFrame(rows, columns=["Ticker","P/E","DividendYield","MarketCap"]).set_index("Ticker")
    st.dataframe(df)

    # Filter UI
    pe_max = st.number_input("Max P/E", value=40.0)
    mcap_min = st.number_input("Min Market Cap ($)", value=10_000_000_000.0, step=1_000_000_000.0)
    mask = (df["P/E"].fillna(pe_max) <= pe_max) & (df["MarketCap"].fillna(0) >= mcap_min)
    st.subheader("Results")
    st.dataframe(df[mask])

# ----------------------------
# 📰 News
# ----------------------------
elif page == "📰 News":
    st.title("📰 News — GrowLio")
    tick = st.text_input("Ticker (for news)", "AAPL").upper().strip()
    items = yahoo_news(tick, limit=8)
    if not items:
        st.warning("No news found right now.")
    else:
        for it in items:
            st.markdown(f"- [{it['title']}]({it['link']})")

# ----------------------------
# 💵 DCF (toy)
# ----------------------------
elif page == "💵 DCF (Toy)":
    st.title("💵 Toy DCF — GrowLio (Educational)")
    st.caption("Very simplified: not investment advice.")
    ticker = st.text_input("Ticker (optional)", "AAPL")
    rev    = st.number_input("Starting FCF ($, millions)", value=10_000.0, step=100.0)
    g      = st.number_input("Growth rate (yrs 1–5, %)", value=8.0) / 100.0
    dr     = st.number_input("Discount rate (WACC, %)", value=10.0) / 100.0
    tg     = st.number_input("Terminal growth (Gordon, %)", value=2.5) / 100.0
    shares = st.number_input("Shares outstanding (millions)", value=16_000.0)

    years = np.arange(1,6)
    fcf = [rev * ((1+g)**i) for i in years]
    pv  = [fcf[i-1]/((1+dr)**i) for i in years]
    term = (fcf[-1]*(1+tg)) / (dr - tg) if dr>tg else np.nan
    pv_term = term / ((1+dr)**5) if pd.notna(term) else np.nan
    ev = np.nansum(pv) + (pv_term if pd.notna(pv_term) else 0)
    per_share = ev / shares if shares>0 else np.nan

    c1,c2,c3 = st.columns(3)
    c1.metric("PV of 5y FCF ($M)", f"{np.nansum(pv):,.0f}")
    c2.metric("PV Terminal ($M)", f"{(pv_term if pd.notna(pv_term) else 0):,.0f}")
    c3.metric("Equity Value / Share ($)", f"{per_share:,.2f}".replace(",",""))

# ----------------------------
# 🔁 Backtest: MA crossover
# ----------------------------
elif page == "🔁 Backtest (MA Crossover)":
    st.title("🔁 Backtest — MA Crossover")
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    start  = st.date_input("Start", default_start)
    end    = st.date_input("End", default_end)
    short  = st.number_input("Short MA", value=20, min_value=3)
    long   = st.number_input("Long MA", value=50, min_value=5)

    df = dl_ohlc(ticker, start, end)
    if df.empty:
        st.warning("No data.")
        st.stop()
    sig = ma_signal(df, short, long)
    sig["Position"] = sig["Signal"].shift(1).fillna(0)  # enter after signal day
    strat_ret = sig["Position"] * sig["Close"].pct_change()
    bench_ret = sig["Close"].pct_change()
    aS,vS,sS = annualize_returns(strat_ret)
    aB,vB,sB = annualize_returns(bench_ret)

    c1,c2,c3 = st.columns(3)
    c1.metric("Strategy Sharpe", f"{sS:.2f}")
    c2.metric("Bench Sharpe", f"{sB:.2f}")
    c3.metric("Excess Return (ann.)", f"{(aS-aB)*100:.2f}%")

    eq_strat = (1+strat_ret.fillna(0)).cumprod()
    eq_bench = (1+bench_ret.fillna(0)).cumprod()
    st.line_chart(pd.DataFrame({"Strategy":eq_strat, "Buy&Hold":eq_bench}))

# ----------------------------
# 🎲 Monte Carlo (Portfolio)
# ----------------------------
elif page == "🎲 Monte Carlo (Portfolio)":
    st.title("🎲 Monte Carlo — Portfolio")
    tickers_in = st.text_input("Tickers", "AAPL, MSFT, TSLA, NVDA")
    n_paths = st.number_input("Simulations", value=200, min_value=50, max_value=2000, step=50)
    horizon = st.number_input("Days", value=252, min_value=30, max_value=2520, step=21)
    start = st.date_input("Hist start", default_start)
    end   = st.date_input("Hist end", default_end)

    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    px = dl_prices(tickers, start, end)
    if px.empty:
        st.warning("No data.")
        st.stop()
    ret = px.pct_change().dropna()
    mu = ret.mean().values
    cov = ret.cov().values

    w = np.repeat(1/len(tickers), len(tickers))
    last_val = 1.0
    paths = np.zeros((int(horizon)+1, int(n_paths)))
    paths[0,:] = last_val

    chol = np.linalg.cholesky(cov + 1e-12*np.eye(len(tickers)))
    for t in range(1, int(horizon)+1):
        z = np.random.normal(size=(len(tickers), int(n_paths)))
        correlated = chol @ z
        step_ret = (mu @ w)/252 + (w @ correlated)*ret.std().mean()/np.sqrt(252)  # simple proxy scale
        paths[t,:] = paths[t-1,:]*(1+step_ret)

    df_paths = pd.DataFrame(paths)
    st.line_chart(df_paths.iloc[:, :min(50, int(n_paths))])  # show up to 50 paths
    pct_5 = np.percentile(paths[-1,:], 5)*100
    pct_50= np.percentile(paths[-1,:], 50)*100
    pct_95= np.percentile(paths[-1,:], 95)*100
    c1,c2,c3 = st.columns(3)
    c1.metric("5th pct outcome", f"{pct_5:.1f}%")
    c2.metric("Median outcome", f"{pct_50:.1f}%")
    c3.metric("95th pct outcome", f"{pct_95:.1f}%")

# Footer
st.caption("© GrowLio — learning app. Data via Yahoo Finance (yfinance). Not investment advice.")

