"""
alert_runner.py
───────────────
Runs at 3:00 PM IST via GitHub Actions.
Scans all recent IPO stocks and sends a Telegram alert ONLY when
genuinely high-quality breakout conditions are met.

Smart filters applied (no spam):
  1. Stock must be holding above breakout level at 3pm (not just touched it)
  2. Price must be at least 0.5% above the breakout level (not just barely)
  3. Today's volume must be above 20-day average (confirmed breakout)
  4. Breakout must be fresh — actual crossing point within last 3 trading days
  5. Dead setups filtered — ATH ran 30%+ then fell back to listing high
  6. Pullback recovery filtered — stock previously ran 15%+ above listing high,
     fell back below it, now crossing again. Different risk profile. Excluded.
  7. Deduplication — already-alerted symbols skipped for 2 days

Rate limit handling:
  - Uses Ticker().history() with browser User-Agent — avoids Yahoo IP blocks
  - Adaptive delay: no sleep until Yahoo pushes back, then 1.5s per request
  - First rate limit → 10s cooldown + retry, then throttled mode for rest of run

No alerts qualify → complete silence. No spam.
"""

import os
import json
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from requests import Session
from playwright.sync_api import sync_playwright

# ─────────────────────────────────────────────
# Config — pulled from GitHub Secrets / env vars
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─────────────────────────────────────────────
# Strategy constants (must match streamlit_app.py)
# ─────────────────────────────────────────────
PROXIMITY_PCT          = 2.0    # within 2% = "near" level
ATH_RUN_FILTER_PCT     = 30.0   # dead setup threshold
MIN_BREAKOUT_STRENGTH  = 0.5    # price must be 0.5%+ above level
FRESH_BREAKOUT_DAYS    = 3      # breakout must be within last N trading days
DEDUP_FILE             = "alerted_symbols.json"
DEDUP_COOLDOWN_DAYS    = 2      # don't re-alert same symbol for 2 days

# ─────────────────────────────────────────────
# IST timezone — FIX: datetime.now() returns UTC on GitHub runners
# ─────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    """Current time in IST regardless of server timezone (GitHub runs UTC)."""
    return datetime.now(timezone.utc).astimezone(IST)

# ─────────────────────────────────────────────
# Rate limit handling — Adaptive Delay
# ─────────────────────────────────────────────
THROTTLE_DELAY_SEC  = 1.5   # per-request delay once throttled
RATE_LIMIT_COOLDOWN = 10    # one-time pause when rate limit first hits

# FIX: Browser User-Agent — Yahoo aggressively blocks the default
# python-requests User-Agent on shared IPs like GitHub Actions.
# Mimicking a real Chrome browser gets through reliably.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Single shared session with browser headers — reused across all requests
_session = Session()
_session.headers.update(_HEADERS)

# Shared throttle state — flipped to True the moment Yahoo pushes back
_throttled = False


def is_rate_limit_error(e: Exception) -> bool:
    err = str(e)
    return any(k in err for k in ("YFRateLimitError", "Too Many Requests", "Rate limited", "429"))


def download_with_retry(ticker: str) -> pd.DataFrame:
    """
    Downloads 2y daily data using Ticker().history() with a browser User-Agent.
    On rate limit: pauses RATE_LIMIT_COOLDOWN seconds, switches to throttled
    mode globally, retries once. Gives up after 2 attempts.
    """
    global _throttled

    for attempt in range(2):
        try:
            t  = yf.Ticker(ticker, session=_session)
            df = t.history(period="2y", interval="1d", auto_adjust=False)

            if df.empty:
                return pd.DataFrame()

            # history() returns tz-aware index — strip for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            return df

        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == 0:
                    if not _throttled:
                        print(f"\n    ⚠️  Rate limit hit — switching to throttled mode ({THROTTLE_DELAY_SEC}s/request)")
                        _throttled = True
                    print(f"    cooling down {RATE_LIMIT_COOLDOWN}s before retry...")
                    time.sleep(RATE_LIMIT_COOLDOWN)
                else:
                    print(f"    gave up after retry (still rate limited)")
                    return pd.DataFrame()
            else:
                print(f"    download error: {e}")
                return pd.DataFrame()

    return pd.DataFrame()


# ─────────────────────────────────────────────
# Deduplication helpers
# ─────────────────────────────────────────────
def load_alerted_symbols() -> dict:
    """Load previously alerted symbols with their alert dates."""
    if os.path.exists(DEDUP_FILE):
        try:
            with open(DEDUP_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_alerted_symbols(data: dict):
    with open(DEDUP_FILE, "w") as f:
        json.dump(data, f, indent=2)


def was_recently_alerted(symbol: str, alerted: dict) -> bool:
    """Return True if this symbol was alerted within the cooldown window."""
    if symbol not in alerted:
        return False
    last_alerted = datetime.strptime(alerted[symbol], "%Y-%m-%d")
    return (datetime.now() - last_alerted).days < DEDUP_COOLDOWN_DAYS


# ─────────────────────────────────────────────
# ChartInk scraper
# ─────────────────────────────────────────────
def scrape_ipo_symbols() -> list:
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
            return symbols
        finally:
            browser.close()


# ─────────────────────────────────────────────
# Core analysis — per symbol
# ─────────────────────────────────────────────
def analyse_symbol(symbol: str) -> dict | None:
    """
    Returns an alert dict if the symbol qualifies, else None.

    Qualifies if ANY of these conditions hold (in priority order):
      A) Broke Listing High + approaching ATH (best setup)
      B) Just broke Listing High and holding (solid setup)
      C) Within 2% of Listing High, green candle, volume up (approaching)

    All conditions also require:
      - Volume above 20-day average
      - Price holding 0.5%+ above breakout level
      - Breakout is fresh — must find ACTUAL crossing point within last 3 days
        (price below on day N-1, above on day N). No fallback.
      - Not a dead setup (ATH ran 30%+ then price fell back to base)
      - Not a pullback recovery (previously ran 15%+ above listing high,
        fell back below it, now crossing again)
    """
    ticker = f"{symbol}.NS"
    df     = download_with_retry(ticker)

    if df.empty or len(df) < 20:
        return None

    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Key levels
        listing_day_high   = float(df['High'].iloc[0])
        listing_date       = df.index[0].strftime('%Y-%m-%d')
        ath                = float(df['High'].max())
        current_price      = float(df['Close'].iloc[-1])
        today_open         = float(df['Open'].iloc[-1])
        today_volume       = float(df['Volume'].iloc[-1])
        days_since_listing = len(df)

        # Volume check
        avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
        if np.isnan(avg_volume_20) or avg_volume_20 == 0:
            return None
        volume_ratio = today_volume / avg_volume_20

        # ── Dead setup filter ─────────────────────────────────────────────
        # ATH ran 30%+ above listing high AND price is back near listing high
        ath_run_pct          = ((ath - listing_day_high) / listing_day_high) * 100
        price_back_near_base = current_price <= listing_day_high * 1.05
        if ath_run_pct >= ATH_RUN_FILTER_PCT and price_back_near_base:
            return None

        # Distance calculations
        pct_from_listing_high = ((current_price - listing_day_high) / listing_day_high) * 100
        pct_from_ath          = ((current_price - ath) / ath) * 100

        broke_listing_high = current_price > listing_day_high
        near_ath           = pct_from_ath >= -PROXIMITY_PCT

        # ── Freshness check ───────────────────────────────────────────────
        # Require an ACTUAL crossing point within the last FRESH_BREAKOUT_DAYS.
        # Price must have been BELOW listing_day_high on day N-1 and ABOVE on day N.
        # No fallback — if we can't find the exact cross, it's not fresh.
        lookback          = df.tail(FRESH_BREAKOUT_DAYS + 1)
        breakout_is_fresh = False
        breakout_date     = None

        if broke_listing_high:
            closes = lookback['Close'].values
            for i in range(len(closes) - 1, 0, -1):
                if closes[i] > listing_day_high and closes[i - 1] <= listing_day_high:
                    breakout_is_fresh = True
                    breakout_date = lookback.index[i].strftime('%Y-%m-%d')
                    break

        # ── Pullback recovery filter ──────────────────────────────────────
        # If price previously ran 15%+ above listing high AND then dipped back
        # below it — this crossing is a second attempt, not a fresh breakout.
        # Catches stocks like ADVAIT (ran +22%, pulled back, now re-crossing).
        if breakout_is_fresh and broke_listing_high:
            prior_history = df.iloc[:(-(FRESH_BREAKOUT_DAYS + 1))]['Close']
            if len(prior_history) > 0:
                peak_before = float(prior_history.max())
                pct_peak    = ((peak_before - listing_day_high) / listing_day_high) * 100
                if pct_peak >= 15.0:
                    ever_below = any(float(p) <= listing_day_high for p in prior_history.values)
                    if ever_below:
                        return None  # pullback recovery — exclude

        # ── Case: approaching but not yet broken ─────────────────────────
        if not broke_listing_high:
            approaching = pct_from_listing_high >= -PROXIMITY_PCT
            green_today = current_price > today_open
            volume_ok   = volume_ratio >= 1.2
            if approaching and green_today and volume_ok:
                return {
                    'symbol':                symbol,
                    'alert_type':            'APPROACHING',
                    'listing_date':          listing_date,
                    'days_since_listing':    days_since_listing,
                    'listing_day_high':      round(listing_day_high, 2),
                    'ath':                   round(ath, 2),
                    'current_price':         round(current_price, 2),
                    'pct_from_listing_high': round(pct_from_listing_high, 2),
                    'pct_from_ath':          round(pct_from_ath, 2),
                    'volume_ratio':          round(volume_ratio, 2),
                    'breakout_date':         None,
                    'near_ath':              near_ath,
                    'priority':              3,
                }
            return None

        # ── Broke listing high — quality filters ──────────────────────────
        if pct_from_listing_high < MIN_BREAKOUT_STRENGTH:
            return None   # barely above, not convincing

        if volume_ratio < 1.0:
            return None   # breakout on weak volume

        if not breakout_is_fresh:
            return None   # old breakout, not a new signal

        # ── Classify and return ───────────────────────────────────────────
        if near_ath:
            alert_type = 'BROKE_LISTING_HIGH_NEAR_ATH'
            priority   = 1
        else:
            alert_type = 'BROKE_LISTING_HIGH'
            priority   = 2

        return {
            'symbol':                symbol,
            'alert_type':            alert_type,
            'listing_date':          listing_date,
            'days_since_listing':    days_since_listing,
            'listing_day_high':      round(listing_day_high, 2),
            'ath':                   round(ath, 2),
            'current_price':         round(current_price, 2),
            'pct_from_listing_high': round(pct_from_listing_high, 2),
            'pct_from_ath':          round(pct_from_ath, 2),
            'volume_ratio':          round(volume_ratio, 2),
            'breakout_date':         breakout_date,
            'near_ath':              near_ath,
            'priority':              priority,
        }

    except Exception as e:
        print(f"    parse error: {e}")
        return None


# ─────────────────────────────────────────────
# Telegram messenger
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram credentials not set. Message not sent.")
        print("─" * 50)
        print(message)
        return False
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            print("✅ Telegram message sent.")
            return True
        else:
            print(f"❌ Telegram error {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")
        return False


# ─────────────────────────────────────────────
# Message builder
# ─────────────────────────────────────────────
def build_message(alerts: list, run_time: str) -> str:
    lines = []
    lines.append("⚡ <b>IPO BREAKOUT ALERT — 3:00 PM IST</b>")
    lines.append(f"🗓 {run_time}")
    lines.append("─" * 32)

    for a in sorted(alerts, key=lambda x: x['priority']):
        sym   = a['symbol']
        price = a['current_price']
        lhigh = a['listing_day_high']
        ath   = a['ath']
        pct_l = a['pct_from_listing_high']
        pct_a = a['pct_from_ath']
        vol   = a['volume_ratio']
        days  = a['days_since_listing']
        bdate = a.get('breakout_date')

        if a['alert_type'] == 'BROKE_LISTING_HIGH_NEAR_ATH':
            lines.append(f"\n🔴 <b>{sym}</b> — Broke Listing High, Near ATH")
            lines.append(f"   Day 1 High : ₹{lhigh}  →  Now ₹{price} (<b>+{pct_l:.1f}%</b> above)")
            lines.append(f"   ATH        : ₹{ath}  |  {pct_a:.1f}% away")
            lines.append(f"   Volume     : {vol:.1f}× avg ✅")
            if bdate:
                lines.append(f"   Broke out  : {bdate}")
            lines.append(f"   Listed {days} trading days ago")
            lines.append(f"   <b>→ Watch for ATH breakout next</b>")

        elif a['alert_type'] == 'BROKE_LISTING_HIGH':
            lines.append(f"\n🟢 <b>{sym}</b> — Above Listing High, Holding")
            lines.append(f"   Day 1 High : ₹{lhigh}  →  Now ₹{price} (<b>+{pct_l:.1f}%</b>)")
            lines.append(f"   ATH        : ₹{ath}  |  {pct_a:.1f}% away")
            lines.append(f"   Volume     : {vol:.1f}× avg ✅")
            if bdate:
                lines.append(f"   Broke out  : {bdate}")
            lines.append(f"   Listed {days} trading days ago")

        elif a['alert_type'] == 'APPROACHING':
            lines.append(f"\n🟡 <b>{sym}</b> — Approaching Listing High")
            lines.append(f"   Day 1 High : ₹{lhigh}  |  Now ₹{price} ({pct_l:.1f}% away)")
            lines.append(f"   ATH        : ₹{ath}")
            lines.append(f"   Volume     : {vol:.1f}× avg ✅  |  Green candle today")
            lines.append(f"   <b>→ Watch for breakout above ₹{lhigh}</b>")

    lines.append(f"\n{'─' * 32}")
    count = len(alerts)
    lines.append(f"<i>{count} alert{'s' if count > 1 else ''} · IPO Scanner by Darshan</i>")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # FIX: Use IST explicitly — GitHub Actions runners are UTC
    run_time = now_ist().strftime("%Y-%m-%d %H:%M IST")

    print(f"\n{'='*50}")
    print(f"IPO Alert Runner — {run_time}")
    print(f"{'='*50}")

    # ── Step 1: Fetch symbols ──────────────────────────
    print("\n[1/4] Fetching IPO symbols from ChartInk...")
    try:
        symbols = scrape_ipo_symbols()
        print(f"  ✓ {len(symbols)} symbols loaded")
    except Exception as e:
        print(f"  ✗ Failed to fetch symbols: {e}")
        send_telegram(f"⚠️ IPO Alert Runner failed to fetch symbols.\nError: {e}")
        return

    # ── Step 2: Load dedup log ─────────────────────────
    print("\n[2/4] Loading deduplication log...")
    alerted = load_alerted_symbols()
    print(f"  ✓ {len(alerted)} symbols in cooldown log")

    # ── Step 3: Scan each symbol ───────────────────────
    print(f"\n[3/4] Scanning {len(symbols)} symbols...")
    alerts      = []
    skipped_dup = []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)

        if was_recently_alerted(symbol, alerted):
            print("skipped (cooldown)")
            skipped_dup.append(symbol)
            if _throttled:
                time.sleep(THROTTLE_DELAY_SEC)
            continue

        result = analyse_symbol(symbol)

        if result:
            alerts.append(result)
            print(f"✅ ALERT — {result['alert_type']}")
        else:
            print("–")

        # Only delay if Yahoo has pushed back — otherwise run free
        if _throttled:
            time.sleep(THROTTLE_DELAY_SEC)

    print(f"\n  → {len(alerts)} alerts | {len(skipped_dup)} on cooldown")

    # ── Step 4: Send or stay silent ───────────────────
    print("\n[4/4] Sending notifications...")
    if not alerts:
        print("  → No qualifying alerts. Staying silent. ✓")
    else:
        message = build_message(alerts, run_time)
        success = send_telegram(message)

        if success:
            today = now_ist().strftime("%Y-%m-%d")
            for a in alerts:
                alerted[a['symbol']] = today
            cutoff = (now_ist() - timedelta(days=DEDUP_COOLDOWN_DAYS + 1)).strftime("%Y-%m-%d")
            alerted = {k: v for k, v in alerted.items() if v >= cutoff}
            save_alerted_symbols(alerted)
            print(f"  ✓ Dedup log updated with {len(alerts)} new entries")

    print(f"\n{'='*50}")
    print("Done.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()