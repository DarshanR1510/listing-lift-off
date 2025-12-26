import streamlit as st
from playwright.sync_api import sync_playwright
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime
import subprocess
import sys

# Auto-install Playwright browsers on first run
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

# Page configuration
st.set_page_config(
    page_title="IPO Stock Scanner",
    page_icon="🎯",
    layout="wide"
)

def scrape_ipo_symbols():
    """Scrape IPO symbols from chartink"""
    url = "https://chartink.com/screener/copy-ipo-base-scan-3950"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            permissions=["clipboard-read", "clipboard-write"]
        )
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle")
            
            # Click Copy button
            page.click("//div[contains(@class,'secondary-button') and .//span[normalize-space()='Copy']]")
            
            # Click symbols option
            page.click("//span[span[normalize-space()='symbols']]")
            page.wait_for_timeout(1000)
            
            # Get clipboard content
            clipboard_text = page.evaluate("() => navigator.clipboard.readText()")
            symbols = [s.strip() for s in clipboard_text.split(",") if s.strip()]
            
            return len(symbols), symbols
            
        finally:
            browser.close()

def check_ipo_strategy(symbol, signal_days=5, volume_filter=False, rsi_filter=False):
    """
    IPO Strategy:
    - Day 1: ATH breakout (close > prior ATH)
    - Day 2: Buy at open (signal generated)
    - Exit: Close below EMA 20
    - Stop Loss: Low of Day 1 (ATH breakout candle)
    """
    try:
        ticker = f"{symbol}.NS"
        
        # Fetch data - IPOs are recent, so 2 years should be enough
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        
        if df.empty or len(df) < 50:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Calculate EMA 20 (for exit tracking only, not displayed)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Calculate All-Time High (ATH)
        df['ATH'] = df['High'].cummax()
        df['Prior_ATH'] = df['ATH'].shift(1)

        # Calculate average volume (20-day)
        df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
        
        # Calculate RSI if filter is enabled
        if rsi_filter:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        df['idx'] = np.arange(len(df))
        
        # Day 1: ATH Breakout condition
        df['ATH_Break'] = df['Close'] > df['Prior_ATH']
        
        # Day 2: Entry signal (unconditionally after ATH break)
        # We mark the day after ATH break as buy signal
        df['Buy_Signal'] = df['ATH_Break'].shift(1)
        
        # Add Volume Filter if enabled (on ATH break day - Day 1)
        if volume_filter:
            volume_condition = df['Volume'].shift(1) > (df['Avg_Volume_20'].shift(1) * 1.5)
            df['Buy_Signal'] = df['Buy_Signal'] & volume_condition
        
        # Add RSI Filter if enabled (on ATH break day - Day 1)
        if rsi_filter:
            rsi_healthy = (df['RSI'].shift(1) > 50) & (df['RSI'].shift(1) < 80)
            df['Buy_Signal'] = df['Buy_Signal'] & rsi_healthy
        
        # Check last N days for signals
        recent_data = df.tail(signal_days)
        
        if recent_data['Buy_Signal'].any():
            signal_rows = recent_data[recent_data['Buy_Signal']]
            current_close = df['Close'].iloc[-1]
            signals = []
            
            for idx, row in signal_rows.iterrows():
                signal_idx = int(row['idx'])
                
                # Buy price is Day 2 open (current row's open)
                buy_price = float(row['Open'])
                
                # Stop Loss: Low of Day 1 (ATH breakout candle - previous day)
                if signal_idx > 0:
                    sl_candle_idx = signal_idx - 1
                    stop_loss = float(df['Low'].iloc[sl_candle_idx])
                    ath_break_date = df.index[sl_candle_idx]
                else:
                    stop_loss = float(row['Low'])
                    ath_break_date = idx
                
                prior_ath = float(df['Prior_ATH'].iloc[signal_idx]) if signal_idx < len(df) and not np.isnan(df['Prior_ATH'].iloc[signal_idx]) else None
                
                days_since_signal = int(df['idx'].iloc[-1] - signal_idx)
                diff = current_close - buy_price
                diff_pct = (diff / buy_price * 100) if buy_price != 0 else None
                
                # Calculate Stop Loss %
                sl_pct = ((buy_price - stop_loss) / buy_price * 100) if stop_loss != 0 else None
                
                # Get volume info if filter was used
                vol_ratio = None
                if volume_filter and signal_idx > 0:
                    prev_vol = df['Volume'].iloc[signal_idx - 1]
                    prev_avg_vol = df['Avg_Volume_20'].iloc[signal_idx - 1]
                    if not np.isnan(prev_avg_vol) and prev_avg_vol > 0:
                        vol_ratio = prev_vol / prev_avg_vol
                
                # Get RSI if filter was used
                rsi_val = None
                if rsi_filter and signal_idx > 0 and 'RSI' in df.columns:
                    rsi_val = float(df['RSI'].iloc[signal_idx - 1]) if not np.isnan(df['RSI'].iloc[signal_idx - 1]) else None

                signals.append({
                    'ath_break_date': pd.to_datetime(ath_break_date).strftime('%Y-%m-%d'),
                    'signal_date': pd.to_datetime(idx).strftime('%Y-%m-%d'),
                    'buy_price': round(buy_price, 4),
                    'current_price': round(float(current_close), 4),
                    'diff': round(diff, 4),
                    'diff_pct': round(diff_pct, 2) if diff_pct is not None else None,
                    'days_since_signal': days_since_signal,
                    'stop_loss': round(stop_loss, 4),
                    'sl_pct': round(sl_pct, 2) if sl_pct is not None else None,
                    'prior_ath': round(prior_ath, 4) if prior_ath is not None else None,
                    'volume_ratio': round(vol_ratio, 2) if vol_ratio is not None else None,
                    'rsi_at_break': round(rsi_val, 2) if rsi_val is not None else None,
                })

            # Return most recent signal only
            signals_sorted = sorted(signals, key=lambda s: s['signal_date'], reverse=True)
            return [signals_sorted[0]]
            
        return None

    except Exception as e:
        return None

def scan_ipo_stocks(symbols, progress_bar, status_text, signal_days=5, volume_filter=False, rsi_filter=False):
    """Scan IPO stocks for buy signals"""
    execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    results = []
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)

        signals = check_ipo_strategy(symbol, signal_days=signal_days, 
                                     volume_filter=volume_filter, rsi_filter=rsi_filter)

        if signals:
            for s in signals:
                result_row = {
                    'Execution_Time': execution_time,
                    'Symbol': symbol,
                    'ATH Break Date': s['ath_break_date'],
                    'Entry Date': s['signal_date'],
                    'Buy Price': round(s['buy_price'], 2),
                    'Current Price': round(s['current_price'], 2),
                    'Gain/Loss': round(s['diff'], 2),
                    'Gain/Loss %': round(s['diff_pct'], 2) if s['diff_pct'] is not None else None,
                    'Stop Loss': round(s['stop_loss'], 2),
                    'SL %': round(s['sl_pct'], 2) if s['sl_pct'] is not None else None,
                    'Days Since Entry': s['days_since_signal'],
                    'Prior ATH': round(s['prior_ath'], 2) if s['prior_ath'] is not None else None,
                }
                
                # Add optional columns if filters were used
                if volume_filter and s['volume_ratio'] is not None:
                    result_row['Volume Ratio'] = round(s['volume_ratio'], 2)
                
                if rsi_filter and s['rsi_at_break'] is not None:
                    result_row['RSI at Break'] = round(s['rsi_at_break'], 2)
                
                results.append(result_row)

    if results:
        result_df = pd.DataFrame(results)
        return result_df
    else:
        return None


# Streamlit UI
st.title("🎯 IPO Stock Scanner by Darshan")
st.markdown("### ATH Breakout Strategy for Recent IPOs")

st.divider()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Strategy Info")
    st.success("""
    **IPO ATH Breakout Strategy:**
    
    📈 **Entry:**
    - **Day 1:** Stock breaks ATH (close > prior ATH)
    - **Day 2:** Buy at market open
    
    🛑 **Stop Loss:**
    - Low of Day 1 (ATH breakout candle)
    
    📤 **Exit:**
    - Close below EMA 20 (not shown, for tracking)
    
    🎯 **Target:**
    - IPOs listed in past 1 year
    """)

# Main content
st.markdown("### 🔍 Scanner Configuration")

col_input1, col_input2 = st.columns(2)

with col_input1:
    signal_days = st.number_input(
        "📅 Look back for signals (days)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Number of past trading days to check for buy signals"
    )
    st.caption(f"Scanner will check signals from last **{signal_days}** trading days")

with col_input2:
    st.markdown("**Data Source:**")
    st.info("IPO stocks from ChartInk (listed in past 1 year)")

# Filter checkboxes
st.markdown("### 🎯 Optional Filters")
col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    volume_filter = st.checkbox(
        "📊 Volume Confirmation",
        value=False,
        help="Require volume to be 1.5x above 20-day average on ATH break day"
    )
    if volume_filter:
        st.caption("✅ Volume must be >1.5x average on Day 1")

with col_filter2:
    rsi_filter = st.checkbox(
        "📈 RSI Filter (50-80)",
        value=False,
        help="Require RSI between 50-80 on ATH break day for healthy momentum"
    )
    if rsi_filter:
        st.caption("✅ RSI must be 50-80 on Day 1")

st.divider()

# Fetch or use cached symbols
if 'ipo_symbols' not in st.session_state:
    with st.spinner("Fetching IPO symbols from ChartInk..."):
        try:
            num_symbols, symbols = scrape_ipo_symbols()
            st.session_state['ipo_symbols'] = symbols
            st.session_state['ipo_num_symbols'] = num_symbols
            st.success(f"✅ Loaded {num_symbols} IPO symbols")
        except Exception as e:
            st.error(f"❌ Failed to fetch symbols: {str(e)}")
            st.stop()
else:
    st.info(f"ℹ️ Using {st.session_state['ipo_num_symbols']} IPO symbols from session cache")
    with st.expander("View IPO symbols"):
        st.write(st.session_state['ipo_symbols'])

if st.button("🚀 Run IPO Scanner", type="primary", use_container_width=True):
    with st.spinner("Starting IPO scanner..."):
        try:
            symbols = st.session_state['ipo_symbols']
            
            filters_text = []
            if volume_filter:
                filters_text.append("Volume Filter")
            if rsi_filter:
                filters_text.append("RSI Filter")
            filters_display = f" ({', '.join(filters_text)})" if filters_text else ""
            
            st.info(f"**Analyzing {len(symbols)} IPO stocks** for ATH breakout opportunities{filters_display}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result_df = scan_ipo_stocks(symbols, progress_bar, status_text, 
                                       signal_days=signal_days, 
                                       volume_filter=volume_filter,
                                       rsi_filter=rsi_filter)
            
            progress_bar.empty()
            status_text.empty()
            
            if result_df is not None and not result_df.empty:
                # Store results in session state
                st.session_state['ipo_results'] = result_df
                st.session_state['ipo_signal_days_used'] = signal_days
                st.session_state['ipo_volume_filter_used'] = volume_filter
                st.session_state['ipo_rsi_filter_used'] = rsi_filter
                
                st.success(f"🎉 Found {len(result_df)} entry opportunities!")
                
                # Display metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Total Opportunities", len(result_df))
                with col_b:
                    avg_gain = result_df['Gain/Loss %'].mean()
                    st.metric("Avg Performance", f"{avg_gain:.2f}%")
                with col_c:
                    positive_signals = len(result_df[result_df['Gain/Loss %'] > 0])
                    st.metric("Positive Performance", positive_signals)
                with col_d:
                    avg_sl = result_df['SL %'].mean()
                    st.metric("Avg SL %", f"{avg_sl:.2f}%")
                
                st.divider()
                
                # Display signals table
                st.markdown("### 📋 Entry Opportunities")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Gain/Loss %": st.column_config.NumberColumn("Gain/Loss %", format="%.2f%%"),
                        "SL %": st.column_config.NumberColumn("SL %", format="%.2f%%"),
                        "Volume Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2fx"),
                    }
                )
                
                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"ipo_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning(f"No ATH breakout opportunities found in the last {signal_days} days.")
                
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")

st.divider()

# View Previous Results
if st.button("📊 View Previous Results", use_container_width=True):
    if 'ipo_results' in st.session_state and st.session_state['ipo_results'] is not None:
        df = st.session_state['ipo_results']
        days_used = st.session_state.get('ipo_signal_days_used', 'N/A')
        vol_filter = st.session_state.get('ipo_volume_filter_used', False)
        rsi_filter = st.session_state.get('ipo_rsi_filter_used', False)
        
        filters_used = []
        if vol_filter:
            filters_used.append("Volume Filter")
        if rsi_filter:
            filters_used.append("RSI Filter")
        filters_text = f" | Filters: {', '.join(filters_used)}" if filters_used else " | No filters"
        
        st.success(f"Loaded {len(df)} opportunities from previous run (signal days: {days_used}{filters_text})")
        
        # Display metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Opportunities", len(df))
        with col_b:
            avg_gain = df['Gain/Loss %'].mean()
            st.metric("Avg Performance", f"{avg_gain:.2f}%")
        with col_c:
            positive_signals = len(df[df['Gain/Loss %'] > 0])
            st.metric("Positive Performance", positive_signals)
        
        st.divider()
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"previous_ipo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No previous results found. Run the scanner first!")

st.divider()

# Action buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🔄 Refresh IPO Symbols", use_container_width=True):
        with st.spinner("Fetching fresh IPO symbols..."):
            try:
                num_symbols, symbols = scrape_ipo_symbols()
                st.session_state['ipo_symbols'] = symbols
                st.session_state['ipo_num_symbols'] = num_symbols
                st.success(f"✅ Refreshed! Loaded {num_symbols} IPO symbols")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to refresh: {str(e)}")

with col_btn2:
    if st.button("🗑️ Clear Results", use_container_width=True):
        if 'ipo_results' in st.session_state:
            del st.session_state['ipo_results']
            if 'ipo_signal_days_used' in st.session_state:
                del st.session_state['ipo_signal_days_used']
            if 'ipo_volume_filter_used' in st.session_state:
                del st.session_state['ipo_volume_filter_used']
            if 'ipo_rsi_filter_used' in st.session_state:
                del st.session_state['ipo_rsi_filter_used']
            st.success("✅ Results cleared!")
            st.rerun()
        else:
            st.info("No results to clear.")

# Footer
st.divider()
st.caption("IPO Stock Scanner made by Darshan Ramani with ❤️ | Data: ChartInk & Yahoo Finance")
st.caption("⚡ Entry: Day 2 Open after ATH Break | Exit: Below EMA 20 | SL: Day 1 Low")