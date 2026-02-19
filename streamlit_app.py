import streamlit as st
from playwright.sync_api import sync_playwright
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime
import subprocess
import sys

# ─────────────────────────────────────────────
# Auto-install Playwright
# ─────────────────────────────────────────────
@st.cache_resource
def install_playwright():
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                       check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "playwright", "install-deps", "chromium"],
                       capture_output=True)
    except:
        pass

install_playwright()

warnings.filterwarnings('ignore', message='.*use_container_width.*')
warnings.filterwarnings("ignore", message=r".*YF.download\(\) has changed argument auto_adjust.*")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Listing Lift-Off · Darshan",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Futuristic Terminal Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root {
    --bg-base:       #050810;
    --bg-surface:    #090d1a;
    --bg-card:       #0d1225;
    --bg-hover:      #111830;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff88;
    --accent-yellow: #ffe500;
    --accent-red:    #ff3860;
    --accent-orange: #ff8c42;
    --border:        #1a2540;
    --border-bright: #2a3a5a;
    --text-primary:  #e8eeff;
    --text-secondary:#9aaaca;
    --text-dim:      #6a7a9a;
    --glow-cyan:     0 0 20px rgba(0,212,255,0.3);
    --glow-green:    0 0 20px rgba(0,255,136,0.3);
}

.stApp {
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,255,0.06) 0%, transparent 60%),
        linear-gradient(180deg, #050810 0%, #060a14 100%) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── More breathing room throughout ── */
.block-container { padding: 2rem 2.5rem 4rem 2.5rem !important; max-width: 1600px !important; }

/* ── Force sidebar always visible ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-bright) !important;
    min-width: 260px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    margin-left: 0 !important;
}
/* Hide the collapse arrow button */
[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }

[data-testid="stSidebar"] * { font-family: 'Rajdhani', sans-serif !important; }

/* Sidebar content padding */
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem !important; }

/* ── TABS — Large & Futuristic ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 4px !important;
    padding: 5px !important;
    gap: 5px !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 3px !important;
    padding: 0.75rem 2.4rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.05) !important;
    border-color: rgba(0,212,255,0.2) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.08) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.05em !important;
    color: var(--text-primary) !important;
}

/* ── Metrics — more padding, bigger label ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-top: 2px solid var(--accent-cyan) !important;
    border-radius: 4px !important;
    padding: 1.3rem 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-secondary) !important;
    border-radius: 3px !important;
    transition: all 0.2s ease !important;
    height: 3rem !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.05) !important;
    box-shadow: var(--glow-cyan) !important;
}
.stButton > button[kind="primary"] {
    background: rgba(0,212,255,0.1) !important;
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(0,212,255,0.2) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Inputs ── */
.stNumberInput > div > div > input {
    font-family: 'Space Mono', monospace !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 3px !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    padding: 0.5rem 0.8rem !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}
/* Input label */
.stNumberInput label, [data-testid="stNumberInput"] label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ── Checkboxes ── */
.stCheckbox > label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Dividers — more margin for air ── */
hr { border-color: var(--border-bright) !important; margin: 2rem 0 !important; }

/* ── Alerts — bigger text ── */
.stAlert {
    border-radius: 3px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.9rem 1.2rem !important;
}
[data-testid="stNotificationContentInfo"]    { background: rgba(0,212,255,0.09) !important;  border-left: 3px solid var(--accent-cyan) !important; }
[data-testid="stNotificationContentSuccess"] { background: rgba(0,255,136,0.09) !important;  border-left: 3px solid var(--accent-green) !important; }
[data-testid="stNotificationContentWarning"] { background: rgba(255,229,0,0.09) !important;  border-left: 3px solid var(--accent-yellow) !important; }
[data-testid="stNotificationContentError"]   { background: rgba(255,56,96,0.09) !important;  border-left: 3px solid var(--accent-red) !important; }

/* ── Dataframe — bigger readable font ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border-bright) !important; border-radius: 4px !important; }
[data-testid="stDataFrame"] * { font-family: 'Share Tech Mono', monospace !important; font-size: 0.85rem !important; }

/* ── Progress ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)) !important;
}

/* ── Caption — more visible ── */
.stCaption, small {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.05em !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 3px !important;
    padding: 0.8rem 1rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-top: none !important;
    padding: 1rem !important;
}

/* ── Custom section headers — bigger ── */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    border-bottom: 1px solid var(--border-bright);
    padding-bottom: 0.5rem;
    margin: 1.8rem 0 1rem 0;
}
.section-header-yellow { color: var(--accent-yellow); border-bottom-color: rgba(255,229,0,0.25); }
.section-header-green  { color: var(--accent-green);  border-bottom-color: rgba(0,255,136,0.25); }
.section-header-red    { color: var(--accent-red);    border-bottom-color: rgba(255,56,96,0.25); }

/* ── Badges — bigger & more visible ── */
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 2px;
    text-transform: uppercase;
}
.badge-cyan   { background: rgba(0,212,255,0.15);  color: var(--accent-cyan);   border: 1px solid rgba(0,212,255,0.4); }
.badge-green  { background: rgba(0,255,136,0.15);  color: var(--accent-green);  border: 1px solid rgba(0,255,136,0.4); }
.badge-yellow { background: rgba(255,229,0,0.15);  color: var(--accent-yellow); border: 1px solid rgba(255,229,0,0.4); }
.badge-red    { background: rgba(255,56,96,0.15);  color: var(--accent-red);    border: 1px solid rgba(255,56,96,0.4); }
.badge-dim    { background: rgba(58,74,106,0.25);  color: #9aaaca;              border: 1px solid rgba(154,170,202,0.3); }

/* Scanline texture */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.025) 2px, rgba(0,0,0,0.025) 4px);
    pointer-events: none;
    z-index: 9999;
}

/* ── Markdown — more readable ── */
.stMarkdown p   { font-family: 'Rajdhani', sans-serif !important; font-size: 1.05rem !important; color: var(--text-secondary) !important; line-height: 1.7 !important; }
.stMarkdown li  { font-family: 'Rajdhani', sans-serif !important; font-size: 1.05rem !important; color: var(--text-secondary) !important; line-height: 1.9 !important; }
.stMarkdown strong { color: var(--text-primary) !important; font-weight: 700 !important; }

/* Sidebar markdown slightly larger */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { font-size: 1rem !important; color: var(--text-secondary) !important; }
[data-testid="stSidebar"] .stMarkdown strong { color: var(--text-primary) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    height: 3rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ChartInk scraper
# ─────────────────────────────────────────────
def scrape_ipo_symbols():
    url = "https://chartink.com/screener/copy-ipo-base-scan-3950"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle")
            page.click("//div[contains(@class,'secondary-button') and .//span[normalize-space()='Copy']]")
            page.click("//span[span[normalize-space()='symbols']]")
            page.wait_for_timeout(1000)
            clipboard_text = page.evaluate("() => navigator.clipboard.readText()")
            symbols = [s.strip() for s in clipboard_text.split(",") if s.strip()]
            return len(symbols), symbols
        finally:
            browser.close()


# ─────────────────────────────────────────────
# TAB 1 — ATH Breakout
# ─────────────────────────────────────────────
def check_ipo_strategy(symbol, signal_days=5, volume_filter=False, rsi_filter=False):
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        if df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['EMA_20']        = df['Close'].ewm(span=20, adjust=False).mean()
        df['ATH']           = df['High'].cummax()
        df['Prior_ATH']     = df['ATH'].shift(1)
        df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
        if rsi_filter:
            delta = df['Close'].diff()
            gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs    = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        df['idx']        = np.arange(len(df))
        df['ATH_Break']  = df['Close'] > df['Prior_ATH']
        df['Buy_Signal'] = df['ATH_Break'].shift(1)
        if volume_filter:
            df['Buy_Signal'] = df['Buy_Signal'] & (df['Volume'].shift(1) > df['Avg_Volume_20'].shift(1) * 1.5)
        if rsi_filter:
            df['Buy_Signal'] = df['Buy_Signal'] & (df['RSI'].shift(1) > 50) & (df['RSI'].shift(1) < 80)
        recent_data = df.tail(signal_days)
        if not recent_data['Buy_Signal'].any():
            return None
        signal_rows   = recent_data[recent_data['Buy_Signal']]
        current_close = df['Close'].iloc[-1]
        signals = []
        for idx, row in signal_rows.iterrows():
            signal_idx = int(row['idx'])
            buy_price  = float(row['Open'])
            if signal_idx > 0:
                stop_loss     = float(df['Low'].iloc[signal_idx - 1])
                ath_break_date = df.index[signal_idx - 1]
            else:
                stop_loss     = float(row['Low'])
                ath_break_date = idx
            prior_ath        = float(df['Prior_ATH'].iloc[signal_idx]) if not np.isnan(df['Prior_ATH'].iloc[signal_idx]) else None
            days_since_signal = int(df['idx'].iloc[-1] - signal_idx)
            diff     = current_close - buy_price
            diff_pct = (diff / buy_price * 100) if buy_price != 0 else None
            sl_pct   = ((buy_price - stop_loss) / buy_price * 100) if stop_loss != 0 else None
            vol_ratio = None
            if volume_filter and signal_idx > 0:
                prev_vol     = df['Volume'].iloc[signal_idx - 1]
                prev_avg_vol = df['Avg_Volume_20'].iloc[signal_idx - 1]
                if not np.isnan(prev_avg_vol) and prev_avg_vol > 0:
                    vol_ratio = prev_vol / prev_avg_vol
            rsi_val = None
            if rsi_filter and signal_idx > 0 and 'RSI' in df.columns:
                v = df['RSI'].iloc[signal_idx - 1]
                rsi_val = float(v) if not np.isnan(v) else None
            signals.append({
                'ath_break_date':  pd.to_datetime(ath_break_date).strftime('%Y-%m-%d'),
                'signal_date':     pd.to_datetime(idx).strftime('%Y-%m-%d'),
                'buy_price':       round(buy_price, 4),
                'current_price':   round(float(current_close), 4),
                'diff':            round(diff, 4),
                'diff_pct':        round(diff_pct, 2) if diff_pct is not None else None,
                'days_since_signal': days_since_signal,
                'stop_loss':       round(stop_loss, 4),
                'sl_pct':          round(sl_pct, 2) if sl_pct is not None else None,
                'prior_ath':       round(prior_ath, 4) if prior_ath is not None else None,
                'volume_ratio':    round(vol_ratio, 2) if vol_ratio is not None else None,
                'rsi_at_break':    round(rsi_val, 2) if rsi_val is not None else None,
            })
        return [sorted(signals, key=lambda s: s['signal_date'], reverse=True)[0]]
    except Exception:
        return None


def scan_ipo_stocks(symbols, progress_bar, status_text, signal_days=5, volume_filter=False, rsi_filter=False):
    execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results = []
    total   = len(symbols)
    for i, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)
        signals = check_ipo_strategy(symbol, signal_days=signal_days,
                                     volume_filter=volume_filter, rsi_filter=rsi_filter)
        if signals:
            for s in signals:
                row = {
                    'Execution_Time':  execution_time,
                    'Symbol':          symbol,
                    'ATH Break Date':  s['ath_break_date'],
                    'Entry Date':      s['signal_date'],
                    'Buy Price':       round(s['buy_price'], 2),
                    'Current Price':   round(s['current_price'], 2),
                    'Gain/Loss':       round(s['diff'], 2),
                    'Gain/Loss %':     round(s['diff_pct'], 2) if s['diff_pct'] is not None else None,
                    'Stop Loss':       round(s['stop_loss'], 2),
                    'SL %':            round(s['sl_pct'], 2) if s['sl_pct'] is not None else None,
                    'Days Since Entry': s['days_since_signal'],
                    'Prior ATH':       round(s['prior_ath'], 2) if s['prior_ath'] is not None else None,
                }
                if volume_filter and s['volume_ratio'] is not None:
                    row['Volume Ratio'] = round(s['volume_ratio'], 2)
                if rsi_filter and s['rsi_at_break'] is not None:
                    row['RSI at Break'] = round(s['rsi_at_break'], 2)
                results.append(row)
    return pd.DataFrame(results) if results else None


# ─────────────────────────────────────────────
# TAB 2 — Breakout Watch
# ─────────────────────────────────────────────
PROXIMITY_PCT       = 2.0   # within 2% below key level = "On Watch"
ATH_RUN_FILTER_PCT  = 30.0  # exclude if ATH ran 30%+ above listing high AND price is back near listing high


def check_breakout_watch(symbol):
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        if df.empty or len(df) < 5:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        listing_date       = df.index[0].strftime('%Y-%m-%d')
        listing_day_high   = float(df['High'].iloc[0])
        ath                = float(df['High'].max())
        current_price      = float(df['Close'].iloc[-1])
        days_since_listing = len(df)

        # ── 30% filter: ATH already ran, stock is back near listing high — dead setup ──
        ath_run_pct          = ((ath - listing_day_high) / listing_day_high) * 100
        price_back_near_base = current_price <= listing_day_high * 1.05
        if ath_run_pct >= ATH_RUN_FILTER_PCT and price_back_near_base:
            return None

        pct_from_listing_high = ((current_price - listing_day_high) / listing_day_high) * 100
        pct_from_ath          = ((current_price - ath) / ath) * 100

        broke_listing_high = current_price > listing_day_high
        broke_ath          = current_price >= ath * 0.999

        near_listing_high = (not broke_listing_high) and (pct_from_listing_high >= -PROXIMITY_PCT)
        near_ath          = (not broke_ath)           and (pct_from_ath          >= -PROXIMITY_PCT)

        if broke_listing_high and not broke_ath:
            status = "🔴 Broke Listing High → Approaching ATH" if near_ath else "🟢 Above Listing High"
        elif not broke_listing_high and near_listing_high:
            status = "🟡 Approaching Listing High"
        elif near_ath and not near_listing_high:
            status = "🟡 Approaching ATH"
        else:
            return None

        return {
            'symbol':                     symbol,
            'listing_date':               listing_date,
            'days_since_listing':         days_since_listing,
            'listing_day_high':           round(listing_day_high, 2),
            'ath':                        round(ath, 2),
            'ath_run_%':                  round(ath_run_pct, 2),
            'current_price':              round(current_price, 2),
            'pct_from_listing_high':      round(pct_from_listing_high, 2),
            'pct_from_ath':               round(pct_from_ath, 2),
            'holding_above_listing_high': broke_listing_high,
            'holding_above_ath':          broke_ath,
            'near_listing_high':          near_listing_high,
            'near_ath':                   near_ath,
            'status':                     status,
        }
    except Exception:
        return None


def scan_breakout_watch(symbols, progress_bar, status_text):
    results = []
    total   = len(symbols)
    for i, symbol in enumerate(symbols):
        status_text.text(f"Scanning {symbol} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)
        r = check_breakout_watch(symbol)
        if r:
            results.append(r)
    if not results:
        return None
    df = pd.DataFrame(results)
    order = {
        "🔴 Broke Listing High → Approaching ATH": 0,
        "🟢 Above Listing High":                   1,
        "🟡 Approaching Listing High":             2,
        "🟡 Approaching ATH":                      3,
    }
    df['_sort'] = df['status'].map(order).fillna(99)
    return df.sort_values('_sort').drop(columns=['_sort'])


# ═══════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:0.22em;
                color:#00d4ff; text-transform:uppercase; margin-bottom:0.4rem;">⚡ System</div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:1.9rem; font-weight:700;
                color:#e8eeff; line-height:1.1; margin-bottom:0.4rem;">Listing Lift-Off</div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; font-weight:600;
                color:#9aaaca; letter-spacing:0.05em;">by Darshan Ramani</div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-header">Tab 01 — ATH Breakout</div>', unsafe_allow_html=True)
    st.markdown("""
    **Entry:** Day 1 close > prior ATH → Buy Day 2 open  
    **Stop Loss:** Low of Day 1 breakout candle  
    **Exit:** Close below EMA 20
    """)

    st.divider()

    st.markdown('<div class="section-header section-header-yellow">Tab 02 — Breakout Watch</div>', unsafe_allow_html=True)
    st.markdown("""
    **🔴** Broke Listing High, approaching ATH  
    **🟢** Holding above Listing Day High  
    **🟡** Within 2% of Listing High or ATH  

    **Dead Setup Filter:** If ATH already ran 30%+ above listing high and price has since fallen back — excluded automatically.
    """)

    st.divider()
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#9aaaca;
                letter-spacing:0.08em; line-height:2.2;">
    DATA · ChartInk + Yahoo Finance<br>
    UNIVERSE · NSE IPOs · Last 1 Year<br>
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────
col_title, col_controls = st.columns([3, 2])
with col_title:
    st.markdown("""
    <div style="margin-bottom:0.15rem;">
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:#00d4ff;
                     letter-spacing:0.2em; text-transform:uppercase;">◈ IPO Breakout Intelligence</span>
    </div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:2.3rem; font-weight:700;
                letter-spacing:0.05em; background:linear-gradient(90deg,#e8eeff 0%,#00d4ff 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;">
        LISTING LIFT-OFF
    </div>
    """, unsafe_allow_html=True)

with col_controls:
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⟳  Refresh Symbols", use_container_width=True):
            with st.spinner("Fetching..."):
                try:
                    n, syms = scrape_ipo_symbols()
                    st.session_state['ipo_symbols']     = syms
                    st.session_state['ipo_num_symbols'] = n
                    st.success(f"Loaded {n} symbols")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with c2:
        if st.button("✕  Clear Results", use_container_width=True):
            for k in ['ipo_results','watch_results','ipo_signal_days_used',
                      'ipo_volume_filter_used','ipo_rsi_filter_used','watch_run_time']:
                st.session_state.pop(k, None)
            st.success("Cleared")
            st.rerun()

st.divider()

# ── Load symbols ──────────────────────────────────────
if 'ipo_symbols' not in st.session_state:
    with st.spinner("Fetching IPO symbols from ChartInk..."):
        try:
            n, syms = scrape_ipo_symbols()
            st.session_state['ipo_symbols']     = syms
            st.session_state['ipo_num_symbols'] = n
        except Exception as e:
            st.error(f"Failed to fetch symbols: {str(e)}")
            st.stop()

n = st.session_state.get('ipo_num_symbols', 0)
st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem;">
    <span class="badge badge-cyan">⚡ {n} IPO Symbols Loaded</span>
    <span style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#9aaaca;">
        NSE · Listed Past 1 Year · Source: ChartInk
    </span>
</div>
""", unsafe_allow_html=True)
with st.expander("View symbol list"):
    st.write(st.session_state.get('ipo_symbols', []))

st.divider()

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["  📈  ATH BREAKOUT SCANNER  ", "  👀  BREAKOUT WATCH  "])


# ── TAB 1 ─────────────────────────────────────────────
with tab1:
    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1.5, 1.5])
    with col_cfg1:
        st.markdown('<div class="section-header">Scanner Configuration</div>', unsafe_allow_html=True)
        signal_days = st.number_input(
            "Look-back window (trading days)",
            min_value=1, max_value=30, value=5, step=1,
            help="Number of past trading days to check for buy signals"
        )
    with col_cfg2:
        st.markdown('<div class="section-header">Volume Filter</div>', unsafe_allow_html=True)
        volume_filter = st.checkbox("Require 1.5× avg volume on Day 1", value=False)
    with col_cfg3:
        st.markdown('<div class="section-header">RSI Filter</div>', unsafe_allow_html=True)
        rsi_filter = st.checkbox("RSI between 50 – 80 on Day 1", value=False)

    st.divider()

    col_r1, col_p1 = st.columns(2)
    with col_r1:
        run_t1 = st.button("⚡  RUN ATH BREAKOUT SCANNER", type="primary", use_container_width=True, key="run_tab1")
    with col_p1:
        show_prev_t1 = st.button("◎  View Previous Results", use_container_width=True, key="prev_tab1")

    if run_t1:
        with st.spinner("Scanning..."):
            try:
                syms = st.session_state['ipo_symbols']
                st.info(f"Analyzing **{len(syms)}** IPO stocks...")
                pb   = st.progress(0)
                stxt = st.empty()
                res  = scan_ipo_stocks(syms, pb, stxt,
                                       signal_days=signal_days,
                                       volume_filter=volume_filter,
                                       rsi_filter=rsi_filter)
                pb.empty(); stxt.empty()
                if res is not None and not res.empty:
                    st.session_state['ipo_results']            = res
                    st.session_state['ipo_signal_days_used']   = signal_days
                    st.session_state['ipo_volume_filter_used'] = volume_filter
                    st.session_state['ipo_rsi_filter_used']    = rsi_filter
                else:
                    st.warning(f"No ATH breakout signals in the last {signal_days} days.")
            except Exception as e:
                st.error(str(e))

    # Persistent results — always show if available
    if 'ipo_results' in st.session_state and st.session_state['ipo_results'] is not None:
        res       = st.session_state['ipo_results']
        days_used = st.session_state.get('ipo_signal_days_used', signal_days)
        st.success(f"✅  {len(res)} entry opportunities  ·  Look-back: {days_used} days")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("OPPORTUNITIES", len(res))
        with c2: st.metric("AVG PERFORMANCE", f"{res['Gain/Loss %'].mean():.2f}%")
        with c3: st.metric("POSITIVE", len(res[res['Gain/Loss %'] > 0]))
        with c4: st.metric("AVG STOP LOSS", f"{res['SL %'].mean():.2f}%")

        st.divider()
        st.markdown('<div class="section-header">Entry Opportunities</div>', unsafe_allow_html=True)
        st.dataframe(res, use_container_width=True, hide_index=True,
                     column_config={
                         "Symbol":       st.column_config.TextColumn("Symbol", width="small"),
                         "Gain/Loss %":  st.column_config.NumberColumn("Gain/Loss %", format="%.2f%%"),
                         "SL %":         st.column_config.NumberColumn("SL %", format="%.2f%%"),
                         "Volume Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2fx"),
                     })
        csv1 = res.to_csv(index=False)
        st.download_button("↓  Export CSV", data=csv1,
                           file_name=f"ath_breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)
    elif show_prev_t1:
        st.warning("No previous results. Run the scanner first.")


# ── TAB 2 ─────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="display:flex; gap:0.7rem; flex-wrap:wrap; margin-bottom:1.2rem; align-items:center;">
        <span class="badge badge-red">🔴 Broke Listing High → Near ATH</span>
        <span class="badge badge-green">🟢 Holding Above Listing High</span>
        <span class="badge badge-yellow">🟡 Within 2% of Key Level</span>
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#9aaaca;">
            · Dead setups filtered (ATH ran 30%+ then fell back)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col_r2, col_p2 = st.columns(2)
    with col_r2:
        run_t2 = st.button("⚡  RUN BREAKOUT WATCH SCANNER", type="primary", use_container_width=True, key="run_tab2")
    with col_p2:
        show_prev_t2 = st.button("◎  View Previous Results", use_container_width=True, key="prev_tab2")

    if run_t2:
        with st.spinner("Scanning for breakout setups..."):
            try:
                syms = st.session_state['ipo_symbols']
                st.info(f"Scanning **{len(syms)}** IPO stocks...")
                pb2   = st.progress(0)
                stxt2 = st.empty()
                wdf   = scan_breakout_watch(syms, pb2, stxt2)
                pb2.empty(); stxt2.empty()
                if wdf is not None and not wdf.empty:
                    st.session_state['watch_results']  = wdf
                    st.session_state['watch_run_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    st.warning("No stocks found near breakout levels right now.")
            except Exception as e:
                st.error(str(e))

    # Persistent results — always show if available
    if 'watch_results' in st.session_state and st.session_state['watch_results'] is not None:
        wdf      = st.session_state['watch_results']
        run_time = st.session_state.get('watch_run_time', '—')

        fresh    = wdf[wdf['status'].str.startswith("🔴") | wdf['status'].str.startswith("🟢")]
        on_watch = wdf[wdf['status'].str.startswith("🟡")]

        st.success(f"✅  Scan complete  ·  {len(fresh)} Active  ·  {len(on_watch)} On Watch  ·  Run: {run_time}")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("TOTAL SETUPS",      len(wdf))
        with c2: st.metric("🔴🟢 ACTIVE / ABOVE", len(fresh))
        with c3: st.metric("🟡 ON WATCH (≤2%)",  len(on_watch))

        st.divider()

        if not fresh.empty:
            st.markdown('<div class="section-header section-header-green">Active — Broke Listing High / Above Key Level</div>', unsafe_allow_html=True)
            st.caption("These stocks have already broken their listing day high. Watch for ATH breakout next.")
            df_fresh = fresh[[
                'symbol','status','listing_date','days_since_listing',
                'listing_day_high','ath','ath_run_%','current_price',
                'pct_from_listing_high','pct_from_ath',
                'holding_above_listing_high','holding_above_ath'
            ]].rename(columns={
                'symbol':'Symbol','status':'Status',
                'listing_date':'Listed On','days_since_listing':'Days Listed',
                'listing_day_high':'Day 1 High','ath':'ATH','ath_run_%':'ATH Run %',
                'current_price':'Current ₹',
                'pct_from_listing_high':'% vs Day1 High','pct_from_ath':'% vs ATH',
                'holding_above_listing_high':'Holding > Day1 High',
                'holding_above_ath':'Holding > ATH',
            })
            st.dataframe(df_fresh, use_container_width=True, hide_index=True,
                         column_config={
                             "% vs Day1 High": st.column_config.NumberColumn("% vs Day1 High", format="%.2f%%"),
                             "% vs ATH":       st.column_config.NumberColumn("% vs ATH",       format="%.2f%%"),
                             "ATH Run %":      st.column_config.NumberColumn("ATH Run %",       format="%.2f%%"),
                         })

        st.divider()

        if not on_watch.empty:
            st.markdown('<div class="section-header section-header-yellow">On Watch — Within 2% of Breakout Level</div>', unsafe_allow_html=True)
            st.caption("Very close to breaking a key level. Fresh opportunity may be imminent.")
            df_watch = on_watch[[
                'symbol','status','listing_date','days_since_listing',
                'listing_day_high','ath','ath_run_%','current_price',
                'pct_from_listing_high','pct_from_ath',
                'near_listing_high','near_ath'
            ]].rename(columns={
                'symbol':'Symbol','status':'Status',
                'listing_date':'Listed On','days_since_listing':'Days Listed',
                'listing_day_high':'Day 1 High','ath':'ATH','ath_run_%':'ATH Run %',
                'current_price':'Current ₹',
                'pct_from_listing_high':'% vs Day1 High','pct_from_ath':'% vs ATH',
                'near_listing_high':'Near Day1 High','near_ath':'Near ATH',
            })
            st.dataframe(df_watch, use_container_width=True, hide_index=True,
                         column_config={
                             "% vs Day1 High": st.column_config.NumberColumn("% vs Day1 High", format="%.2f%%"),
                             "% vs ATH":       st.column_config.NumberColumn("% vs ATH",       format="%.2f%%"),
                             "ATH Run %":      st.column_config.NumberColumn("ATH Run %",       format="%.2f%%"),
                         })

        st.divider()
        csv2 = wdf.to_csv(index=False)
        st.download_button("↓  Export Watch List CSV", data=csv2,
                           file_name=f"breakout_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)

    elif show_prev_t2:
        st.warning("No previous results. Run the scanner first.")


# ── FOOTER ────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#9aaaca; letter-spacing:0.1em;">
        LISTING LIFT-OFF · DARSHAN RAMANI · DATA: CHARTINK + YAHOO FINANCE
    </span>
    <span style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#9aaaca; letter-spacing:0.1em;">
        TAB 1: DAY 2 OPEN AFTER ATH BREAK · TAB 2: LISTING HIGH & ATH WATCH
    </span>
</div>
""", unsafe_allow_html=True)
