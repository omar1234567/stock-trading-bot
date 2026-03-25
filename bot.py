#!/usr/bin/env python3
"""
Daily Stock Trading Bot
Analyzes a watchlist using technical + fundamental analysis via Claude AI
Sends a daily email with trade signals every weekday morning
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import yfinance as yf
import pandas as pd
import ta
import anthropic
from tracker import log_signal, resolve_open_trades, generate_performance_summary, get_recent_closed_trades

# ============================================================
# YOUR WATCHLIST — Add or remove tickers here anytime
# ============================================================

WATCHLIST = {
    "US": [
        # ── Semiconductors & Hardware ──────────────────────────
        "NVDA", "AAPL", "MSFT", "AVGO", "AMD",
        "MU", "LRCX", "AMAT", "TXN", "KLAC",
        "MCHP", "SWKS", "MPWR", "ON", "TER",
        # ── Software & Cloud ──────────────────────────────────
        "GOOGL", "ORCL", "ADBE", "CRM", "NOW",
        "SNOW", "PANW", "CRWD", "ANSS", "CDNS",
        "SNPS", "ANET",
        # ── Healthcare & Biotech ──────────────────────────────
        "LLY", "JNJ", "ABBV", "MRK", "ABT",
        "TMO", "AMGN", "GILD", "REGN", "VRTX",
        "BSX", "SYK", "ISRG", "DHR", "EW",
        "DXCM", "GEHC", "RMD", "IDXX", "ALGN",
        # ── Energy ────────────────────────────────────────────
        "XOM", "CVX", "COP", "EOG", "OXY",
        # ── Consumer ──────────────────────────────────────────
        "TSLA", "HD", "PG", "PEP", "COST",
        "LOW", "NKE", "SBUX", "LIN",
        # ── Industrials & Transport ───────────────────────────
        "CAT", "DE", "HON", "GEV",
        "ODFL", "JBHT", "EXPD",
        # ── Healthcare & Medical Devices ──────────────────────
        "PODD", "HOLX", "TECH", "EXAS", "INSP",
        "ITGR", "NVCR", "PCVX", "RXRX",
        # ── Technology & Software ─────────────────────────────
        "AXON", "TRMB", "ENTG", "ACLS", "ONTO",
        "MKSI", "SITM", "SMTC", "DIOD",
        # ── Industrials & Logistics ───────────────────────────
        "SAIA", "XPO", "CHRW", "GNRC",
        # ── Consumer ──────────────────────────────────────────
        "DECK", "WSM", "LULU", "WING", "CELH", "ELF",
    ],
    "TSX": [
        # ── Technology ────────────────────────────────────────
        "CSU.TO", "SHOP.TO", "OTEX.TO", "GIB-A.TO",
        # ── Industrials & Transport ───────────────────────────
        "TIH.TO", "TFII.TO", "WCN.TO", "CNR.TO",
        "CP.TO", "STN.TO", "WSP.TO",
        # ── Energy ────────────────────────────────────────────
        "CNQ.TO", "SU.TO",
        # ── Mining & Materials ────────────────────────────────
        "NTR.TO", "WPM.TO", "FNV.TO",
        "AEM.TO", "IVN.TO",
        # ── Consumer & Retail ─────────────────────────────────
        "ATD.TO", "DOL.TO", "CTC-A.TO",
        # ── Automotive & Manufacturing ────────────────────────
        "MG.TO", "MRE.TO",
        # ── Technology & Software ─────────────────────────────
        "KXS.TO", "DSG.TO", "LSPD.TO",
        "MDA.TO", "ATA.TO",
        # ── Industrials & Engineering ─────────────────────────
        "BDT.TO", "CIGI.TO",
        # ── Mining & Resources ────────────────────────────────
        "LUN.TO", "ERO.TO", "HBM.TO",
        "AGI.TO", "K.TO",
        # ── Consumer & Food ───────────────────────────────────
        "PBH.TO", "MTY.TO", "DOO.TO",
    ]
}

# ============================================================
# SETTINGS — Loaded automatically from GitHub Secrets
# ============================================================

EMAIL_SENDER    = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD  = os.environ.get("EMAIL_PASSWORD")   # Gmail App Password
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# ============================================================
# CLAUDE SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a professional stock trading analyst bot. Every day you analyze stock data
and identify only the highest-conviction trade setups from a watchlist.

ANALYSIS FRAMEWORK

Primary — Technical Analysis (weighted heavily):
1. Support & Resistance — key price levels, bounces, breakouts
2. RSI / Momentum — overbought (>70) or oversold (<30), divergence
3. EMA/SMA Crossovers — bullish/bearish crossovers, price vs moving averages
4. Candlestick Patterns — reversal or continuation patterns at key levels

Secondary — Fundamental (supporting context only):
- Earnings trend & revenue growth (is the business growing?)
- Valuation: P/E, P/B vs peers (cheap, fair, or expensive?)
- Analyst consensus rating and mean price target

TRADE STYLE — Let the signal decide:
- Swing Trade (3–15 days): strong near-term momentum/pattern setup
- Position Trade (3–8 weeks): breakout or trend alignment on higher timeframes

STRICT QUALITY RULES:
- ONLY recommend trades where at least 3 of 4 technical indicators align clearly
- NEVER recommend a trade with Risk:Reward below 1:2
- Stop loss must come from chart structure (key support/resistance or pattern low/high), NOT a fixed %
- Calculate probability honestly — do not inflate scores
- If no quality setups exist today, say so clearly — do not force trades
- Flag currency clearly: USD for S&P 500 stocks, CAD for TSX (.TO) stocks

PHASE 2 FILTER RULES — apply these to every signal:
- MARKET CONTEXT: If the market context block shows SP500 or TSX as BEARISH, add a market warning label to any LONG signal. Do NOT skip it — just clearly warn: "⚠️ MARKET HEADWIND — broad market is bearish, trade with reduced size"
- EARNINGS: If the earnings warning shows "EARNINGS IN X DAY(S)", label the trade as "🔴 HIGH RISK — EARNINGS IMMINENT" and note that the trade carries elevated volatility risk
- VOLUME: If the volume signal shows WEAK or LOW VOLUME, note this in the Key Risks section as a lack of conviction
- SECTOR MOMENTUM: If sector momentum is BEARISH and the signal is LONG, add a note warning of sector headwind. If sector is BULLISH and signal is LONG, note this as a tailwind that improves the setup

OUTPUT FORMAT — Use this exact layout for each trade:

🟢 [TICKER] — [Swing / Position] — [LONG / SHORT]
Currency: [USD / CAD]

| Field | Detail |
|---|---|
| Entry Price | $XX.XX (limit or market — specify) |
| Entry Timing | [e.g. "at market open", "on pullback to $XX", "on breakout above $XX"] |
| Stop Loss | $XX.XX |
| Take Profit 1 | $XX.XX |
| Take Profit 2 | $XX.XX (if applicable) |
| Risk/Reward | 1 : X.X |
| Exit Timing | [e.g. "if daily close below stop", "trail stop after TP1 hit"] |
| Probability of Success | XX% |

📈 Technical Reasoning:
[2–4 sentences explaining the specific setup — reference exact indicators, levels, and patterns]

📋 Fundamental Support:
[2–3 sentences on earnings trend, valuation, analyst rating — note clearly if they conflict with the technical signal]

⚠️ Key Risks:
[1–2 sentences on what could invalidate this trade: upcoming earnings, macro events, weak volume, etc.]

☪️ Shariah Compliance (AAOIFI):
State the compliance status clearly: COMPLIANT, DOUBTFUL, or NON-COMPLIANT.
If NON-COMPLIANT or DOUBTFUL, list the specific violations or concerns.
Always include this section — even for compliant stocks, confirm it passed all screens.

---

If NO trades meet the criteria, respond with:
NO TRADES TODAY — No setups met the minimum criteria (3/4 indicator alignment + 1:2 R:R).
Watchlist reviewed: [list all tickers checked]
"""

# ============================================================
# AAOIFI SHARIAH COMPLIANCE SCREENING
# ============================================================

PROHIBITED_SECTORS = [
    "banks", "diversified banks", "regional banks", "thrifts & mortgage finance",
    "consumer finance", "capital markets", "insurance", "life insurance",
    "property & casualty insurance", "reinsurance", "multi-line insurance",
    "insurance brokers", "diversified financial services",
    "distillers & vintners", "brewers", "tobacco",
    "casinos & gaming", "adult entertainment", "aerospace & defense",
    "mortgage real estate investment trusts (reits)"
]

PROHIBITED_KEYWORDS = [
    "bank", "banking", "insurance", "alcohol", "brewery", "brewer",
    "distill", "tobacco", "casino", "gambling", "gaming", "defense",
    "weapon", "ammunition", "adult", "mortgage reit"
]


def shariah_screen(ticker):
    """
    Screen a stock for AAOIFI Shariah compliance.
    Checks: business activity, debt ratio, cash ratio, receivables ratio.
    """
    result = {
        "status": "UNKNOWN", "emoji": "UNKNOWN",
        "issues": [], "ratios": {},
        "sector": "N/A", "industry": "N/A", "note": ""
    }

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        sector   = (info.get("sector",   "") or "").strip()
        industry = (info.get("industry", "") or "").strip()
        result["sector"]   = sector   or "N/A"
        result["industry"] = industry or "N/A"

        industry_lower = industry.lower()
        sector_lower   = sector.lower()

        # 1. Business Activity Screen
        business_fail = False
        for prohibited in PROHIBITED_SECTORS:
            if prohibited in industry_lower or prohibited in sector_lower:
                business_fail = True
                result["issues"].append(f"Prohibited business: {industry or sector}")
                break
        if not business_fail:
            for keyword in PROHIBITED_KEYWORDS:
                if keyword in industry_lower or keyword in sector_lower:
                    business_fail = True
                    result["issues"].append(f"Prohibited activity detected: {industry}")
                    break

        # 2. Financial Ratio Screens
        try:
            balance_sheet = stock.balance_sheet

            def get_row(df, *names):
                for name in names:
                    if name in df.index:
                        val = df.loc[name].iloc[0]
                        if pd.notna(val):
                            return float(val)
                return None

            total_assets    = get_row(balance_sheet, "Total Assets", "TotalAssets")
            total_debt      = get_row(balance_sheet, "Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation")
            total_cash      = get_row(balance_sheet, "Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments")
            net_receivables = get_row(balance_sheet, "Net Receivables", "Receivables", "Accounts Receivable")

            if total_cash is None: total_cash = info.get("totalCash")
            if total_debt is None: total_debt = info.get("totalDebt")

            if total_assets and total_assets > 0:
                if total_debt is not None:
                    debt_ratio = round((total_debt / total_assets) * 100, 1)
                    result["ratios"]["Debt / Assets"] = f"{debt_ratio}% (limit: <30%)"
                    if debt_ratio >= 30:
                        result["issues"].append(f"Debt ratio too high: {debt_ratio}% >= 30%")

                if total_cash is not None:
                    cash_ratio = round((total_cash / total_assets) * 100, 1)
                    result["ratios"]["Cash / Assets"] = f"{cash_ratio}% (limit: <30%)"
                    if cash_ratio >= 30:
                        result["issues"].append(f"Cash/interest ratio too high: {cash_ratio}% >= 30%")

                if net_receivables is not None:
                    rec_ratio = round((net_receivables / total_assets) * 100, 1)
                    result["ratios"]["Receivables / Assets"] = f"{rec_ratio}% (limit: <70%)"
                    if rec_ratio >= 70:
                        result["issues"].append(f"Receivables ratio too high: {rec_ratio}% >= 70%")
            else:
                result["ratios"]["note"] = "Balance sheet unavailable — manual review required"

        except Exception as e:
            result["ratios"]["note"] = f"Could not retrieve balance sheet: {str(e)[:60]}"

        # 3. Final Status
        if business_fail or result["issues"]:
            result["status"] = "NON-COMPLIANT"
            result["emoji"]  = "[NON-COMPLIANT]"
        elif "unavailable" in result["ratios"].get("note", ""):
            result["status"] = "DOUBTFUL"
            result["emoji"]  = "[DOUBTFUL]"
            result["note"]   = "Financial data unavailable - manual Shariah review recommended"
        else:
            result["status"] = "COMPLIANT"
            result["emoji"]  = "[COMPLIANT]"

    except Exception as e:
        result["status"] = "UNKNOWN"
        result["emoji"]  = "[UNKNOWN]"
        result["note"]   = f"Screening error: {str(e)[:80]}"

    return result


def format_shariah_block(ticker, screen):
    """Format the Shariah screening result as a readable text block"""
    lines = [
        f"\nSHARIAH COMPLIANCE (AAOIFI Standard):",
        f"  Status:   {screen['emoji']} {screen['status']}",
        f"  Sector:   {screen['sector']}",
        f"  Industry: {screen['industry']}",
    ]
    if screen["ratios"]:
        lines.append("  Financial Ratios:")
        for k, v in screen["ratios"].items():
            lines.append(f"    - {k}: {v}")
    if screen["issues"]:
        lines.append("  Violations:")
        for issue in screen["issues"]:
            lines.append(f"    X {issue}")
    if screen["note"]:
        lines.append(f"  Note: {screen['note']}")
    return "\n".join(lines)


# ============================================================
# PHASE 2 — MARKET INTELLIGENCE
# ============================================================

# Sector ETF map — used to check if a stock's sector is trending up or down
SECTOR_ETF_MAP = {
    # US sectors
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Communication Services": "XLC",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Financial Services":     "XLF",
    # Canadian sectors (using TSX sector ETFs)
    "Financial":              "XFN.TO",
    "Energy (Canada)":        "XEG.TO",
    "Materials":              "XMA.TO",
    "Industrials (Canada)":   "XIN.TO",
    "Technology (Canada)":    "XIT.TO",
}


def get_market_context():
    """
    Fetch the current trend for S&P 500 (SPY) and TSX Composite (XIU.TO).
    Returns a dict with trend direction, EMA alignment, and a warning flag.
    Trend is determined by:
      - Price vs EMA50 (short-term trend)
      - EMA9 vs EMA21 (momentum)
      - Last 3 days consecutive direction
    """
    context = {}

    for label, ticker in [("SP500", "SPY"), ("TSX", "XIU.TO")]:
        try:
            df = yf.download(ticker, period="3mo", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                context[label] = {"trend": "UNKNOWN", "warning": False, "detail": "No data"}
                continue

            df["EMA9"]  = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
            df["EMA21"] = ta.trend.EMAIndicator(df["Close"], window=21).ema_indicator()
            df["EMA50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()

            latest  = df.iloc[-1]
            price   = float(latest["Close"])
            ema9    = float(latest["EMA9"])
            ema21   = float(latest["EMA21"])
            ema50   = float(latest["EMA50"])

            # Last 3 days consecutive closes
            last3   = df["Close"].iloc[-3:].tolist()
            consec_down = last3[0] > last3[1] > last3[2]
            consec_up   = last3[0] < last3[1] < last3[2]

            bullish_points = sum([
                price > ema50,
                ema9 > ema21,
                consec_up,
            ])
            bearish_points = sum([
                price < ema50,
                ema9 < ema21,
                consec_down,
            ])

            if bullish_points >= 2:
                trend   = "BULLISH"
                warning = False
            elif bearish_points >= 2:
                trend   = "BEARISH"
                warning = True
            else:
                trend   = "NEUTRAL"
                warning = False

            pct_vs_ema50 = round((price - ema50) / ema50 * 100, 2)
            detail = (
                f"Price {'+' if pct_vs_ema50 >= 0 else ''}{pct_vs_ema50}% vs EMA50 | "
                f"EMA9 {'>' if ema9 > ema21 else '<'} EMA21 | "
                f"Last 3 days: {'UP' if consec_up else 'DOWN' if consec_down else 'MIXED'}"
            )

            context[label] = {
                "trend":   trend,
                "warning": warning,
                "detail":  detail,
                "price":   round(price, 2),
            }

        except Exception as e:
            context[label] = {"trend": "UNKNOWN", "warning": False,
                              "detail": f"Error: {str(e)[:50]}"}

    return context


def get_sector_momentum(sector):
    """
    Check if the stock's sector ETF is trending up or down.
    Returns a short string summary.
    """
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return "Sector ETF not mapped — manual check recommended"

    try:
        df = yf.download(etf, period="1mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 10:
            return f"{etf}: No data"

        df["EMA9"]  = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
        df["EMA21"] = ta.trend.EMAIndicator(df["Close"], window=21).ema_indicator()

        latest  = df.iloc[-1]
        price   = float(latest["Close"])
        ema9    = float(latest["EMA9"])
        ema21   = float(latest["EMA21"])

        # 1-month return
        month_return = round((price - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0]) * 100, 2)
        momentum = "BULLISH" if ema9 > ema21 else "BEARISH"

        return (f"{etf}: {momentum} | "
                f"EMA9 {'>' if ema9 > ema21 else '<'} EMA21 | "
                f"1M return: {'+' if month_return >= 0 else ''}{month_return}%")

    except Exception as e:
        return f"{etf}: Error — {str(e)[:50]}"


def get_earnings_warning(ticker):
    """
    Check if the stock has earnings scheduled within the next 5 trading days.
    Returns (has_upcoming_earnings: bool, detail: str)
    """
    try:
        stock    = yf.Ticker(ticker)
        calendar = stock.calendar

        if calendar is None or calendar.empty:
            return False, "Earnings date: Not available"

        # calendar is a DataFrame with dates as columns
        if "Earnings Date" in calendar.index:
            earn_dates = calendar.loc["Earnings Date"]
        elif hasattr(calendar, "columns") and len(calendar.columns) > 0:
            earn_dates = calendar.iloc[0]
        else:
            return False, "Earnings date: Not available"

        today = pd.Timestamp.now().normalize()

        for val in earn_dates:
            try:
                earn_date = pd.Timestamp(val).normalize()
                days_away = (earn_date - today).days
                if 0 <= days_away <= 5:
                    return True, f"⚠️ EARNINGS IN {days_away} DAY(S): {earn_date.strftime('%b %d, %Y')}"
                elif 6 <= days_away <= 14:
                    return False, f"Earnings in {days_away} days: {earn_date.strftime('%b %d, %Y')}"
            except Exception:
                continue

        return False, "No earnings within 14 days"

    except Exception as e:
        return False, f"Earnings check unavailable: {str(e)[:50]}"


def get_volume_confirmation(df):
    """
    Check if today's volume confirms the move.
    Returns (confirmed: bool, detail: str)
    """
    try:
        avg_vol  = int(df["Volume"].iloc[-20:].mean())
        last_vol = int(df["Volume"].iloc[-1])
        ratio    = round(last_vol / avg_vol * 100) if avg_vol > 0 else 0

        if ratio >= 150:
            return True,  f"STRONG ({ratio}% of 20d avg — high conviction)"
        elif ratio >= 100:
            return True,  f"ABOVE AVERAGE ({ratio}% of 20d avg — confirmed)"
        elif ratio >= 75:
            return False, f"AVERAGE ({ratio}% of 20d avg — acceptable)"
        else:
            return False, f"WEAK ({ratio}% of 20d avg — low conviction, be cautious)"

    except Exception:
        return False, "Volume data unavailable"


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def find_support_resistance(df, window=15):
    """Identify key support and resistance levels from recent price history"""
    support_levels = []
    resistance_levels = []

    for i in range(window, len(df) - window):
        low  = df['Low'].iloc[i]
        high = df['High'].iloc[i]

        # Local low = support
        if low == df['Low'].iloc[i - window : i + window].min():
            support_levels.append(round(low, 2))

        # Local high = resistance
        if high == df['High'].iloc[i - window : i + window].max():
            resistance_levels.append(round(high, 2))

    # Return the 3 most relevant levels near current price
    current = df['Close'].iloc[-1]
    supports    = sorted(set(s for s in support_levels    if s < current))[-3:]
    resistances = sorted(set(r for r in resistance_levels if r > current))[:3]

    return supports, resistances


def describe_candlesticks(df):
    """Describe the last 3 daily candles and flag common patterns"""
    descriptions = []
    last3 = df.tail(3).reset_index()

    for i, row in last3.iterrows():
        body       = row['Close'] - row['Open']
        body_size  = abs(body)
        total_range = row['High'] - row['Low']
        upper_wick = row['High'] - max(row['Open'], row['Close'])
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        direction  = "Bullish" if body > 0 else "Bearish"

        note = ""
        if total_range > 0:
            if body_size / total_range < 0.1:
                note = " → DOJI"
            elif lower_wick > 2 * body_size and direction == "Bullish":
                note = " → HAMMER"
            elif upper_wick > 2 * body_size and direction == "Bearish":
                note = " → SHOOTING STAR"

        label = f"  Day-{2 - i}: {direction} | O:{round(row['Open'],2)} H:{round(row['High'],2)} L:{round(row['Low'],2)} C:{round(row['Close'],2)}{note}"
        descriptions.append(label)

    # Check for engulfing pattern on the most recent 2 candles
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if curr['Close'] > curr['Open'] and prev['Close'] < prev['Open']:
        if curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
            descriptions.append("  ⭐ BULLISH ENGULFING detected on latest candle")
    elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open']:
        if curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']:
            descriptions.append("  ⭐ BEARISH ENGULFING detected on latest candle")

    return descriptions


def analyze_ticker(ticker):
    """Download data and build a full technical + fundamental summary for one ticker"""
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period="1y", interval="1d")

        if df.empty or len(df) < 60:
            print(f"  ⚠️  {ticker}: Not enough data, skipping.")
            return None

        # --- Technical Indicators ---
        df['RSI']   = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['EMA9']  = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA200']= ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()

        latest = df.iloc[-1]
        price  = round(latest['Close'], 2)
        rsi    = round(latest['RSI'],   1) if not pd.isna(latest['RSI'])    else "N/A"
        ema9   = round(latest['EMA9'],  2) if not pd.isna(latest['EMA9'])   else "N/A"
        ema21  = round(latest['EMA21'], 2) if not pd.isna(latest['EMA21'])  else "N/A"
        ema50  = round(latest['EMA50'], 2) if not pd.isna(latest['EMA50'])  else "N/A"
        ema200 = round(latest['EMA200'],2) if not pd.isna(latest['EMA200']) else "N/A"

        # EMA alignment
        if all(isinstance(v, float) for v in [ema9, ema21, ema50]):
            if ema9 > ema21 > ema50:
                ema_align = "BULLISH (9 > 21 > 50)"
            elif ema9 < ema21 < ema50:
                ema_align = "BEARISH (9 < 21 < 50)"
            else:
                ema_align = "MIXED"
        else:
            ema_align = "N/A"

        price_vs_200 = (
            "ABOVE 200 EMA (long-term bullish)" if isinstance(ema200, float) and price > ema200
            else "BELOW 200 EMA (long-term bearish)"
        )

        rsi_note = (
            " [OVERSOLD ⬆ watch for reversal]"  if isinstance(rsi, float) and rsi < 30 else
            " [OVERBOUGHT ⬇ watch for pullback]" if isinstance(rsi, float) and rsi > 70 else
            " [NEUTRAL]"
        )

        supports, resistances = find_support_resistance(df)
        candles               = describe_candlesticks(df)

        week52_high = round(df['High'].max(), 2)
        week52_low  = round(df['Low'].min(),  2)

        avg_vol  = int(df['Volume'].tail(20).mean())
        last_vol = int(latest['Volume'])
        vol_pct  = round(last_vol / avg_vol * 100) if avg_vol > 0 else 0

        # --- Fundamentals ---
        info            = stock.info
        pe              = info.get('trailingPE',        'N/A')
        pb              = info.get('priceToBook',       'N/A')
        target          = info.get('targetMeanPrice',   'N/A')
        rating          = info.get('recommendationKey', 'N/A')
        earn_growth     = info.get('earningsGrowth',    'N/A')
        rev_growth      = info.get('revenueGrowth',     'N/A')

        if isinstance(pe,          float): pe          = round(pe, 1)
        if isinstance(pb,          float): pb          = round(pb, 2)
        if isinstance(target,      float): target      = f"${round(target, 2)}"
        if isinstance(earn_growth, float): earn_growth = f"{round(earn_growth * 100, 1)}%"
        if isinstance(rev_growth,  float): rev_growth  = f"{round(rev_growth  * 100, 1)}%"

        currency = "CAD" if ticker.endswith(".TO") else "USD"

        # --- Phase 2: Market Intelligence ---
        has_earnings, earnings_detail = get_earnings_warning(ticker)
        vol_confirmed, vol_detail     = get_volume_confirmation(df)
        sector_detail = get_sector_momentum(info.get("sector", ""))

        earnings_flag = "⚠️ HIGH RISK — EARNINGS IMMINENT" if has_earnings else ""
        volume_flag   = "" if vol_confirmed else "⚠️ LOW VOLUME — weak conviction"

        # --- Shariah Compliance ---
        print(f"     ☪  Running Shariah screen for {ticker}...")
        shariah = shariah_screen(ticker)
        shariah_block = format_shariah_block(ticker, shariah)

        summary = f"""
{'='*55}
TICKER: {ticker} | Currency: {currency}
Current Price: ${price}
52-Week Range: ${week52_low} — ${week52_high}

TECHNICAL INDICATORS:
  RSI (14):   {rsi}{rsi_note}
  EMA  9:     ${ema9}
  EMA 21:     ${ema21}
  EMA 50:     ${ema50}
  EMA 200:    ${ema200}
  EMA Align:  {ema_align}
  vs 200 EMA: {price_vs_200}

  Support levels (below price):    {supports}
  Resistance levels (above price): {resistances}

  Last 3 Daily Candles:
{chr(10).join(candles)}

  Volume (latest): {last_vol:,}
  Volume (20d avg): {avg_vol:,}
  Volume vs avg:   {vol_pct}%

FUNDAMENTAL DATA:
  P/E Ratio:        {pe}
  P/B Ratio:        {pb}
  Earnings Growth:  {earn_growth}
  Revenue Growth:   {rev_growth}
  Analyst Rating:   {rating}
  Analyst Target:   {target}

PHASE 2 — SIGNAL FILTERS:
  Earnings Warning: {earnings_detail} {earnings_flag}
  Volume Signal:    {vol_detail} {volume_flag}
  Sector Momentum:  {sector_detail}
{shariah_block}
"""
        return summary

    except Exception as e:
        print(f"  ❌ Error analyzing {ticker}: {e}")
        return None


# ============================================================
# CLAUDE ANALYSIS
# ============================================================

def get_trade_signals(ticker_summaries):
    """
    Send all ticker data to Claude and get back the daily trade report.
    Includes automatic retry with exponential backoff for server overload errors.
    """
    import time

    client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    combined_data = "\n".join(ticker_summaries)
    today_str     = datetime.now().strftime("%A, %B %d, %Y")

    max_retries = 5
    base_delay  = 30   # seconds — doubles each retry: 30, 60, 120, 240, 480

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  🧠 Claude API call — attempt {attempt}/{max_retries}...")
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"""Today is {today_str}.

Please analyze the following data for my complete watchlist and identify any high-conviction trade setups.
Apply your strict quality rules — only surface trades where at least 3 of 4 indicators align and R:R ≥ 1:2.
If nothing qualifies today, say so honestly.

{combined_data}

Provide your full daily trading report now."""
                }]
            )
            return message.content[0].text

        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"  ⏳ Anthropic servers overloaded (529). "
                      f"Waiting {wait}s before retry {attempt}/{max_retries}...")
                time.sleep(wait)
            else:
                raise   # Re-raise non-overload errors immediately

        except anthropic.APIConnectionError:
            wait = base_delay * (2 ** (attempt - 1))
            print(f"  ⏳ Connection error. Waiting {wait}s before retry {attempt}/{max_retries}...")
            time.sleep(wait)

    raise RuntimeError(
        f"Claude API unavailable after {max_retries} attempts. "
        f"Anthropic servers may be under heavy load — the bot will try again tomorrow."
    )


# ============================================================
# EMAIL
# ============================================================

def send_email(report, perf_summary, recent_trades, sp500, tsx, market_warning):
    """Send the daily report as a formatted HTML email"""
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    subject   = f"📈 Daily Trade Signals — {today_str}"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = EMAIL_SENDER
    msg['To']      = EMAIL_RECIPIENT

    plain = f"DAILY TRADE SIGNALS — {today_str}\n{'='*50}\n\n{report}\n\n{perf_summary}\n\n---\n⚠️ Not financial advice. Do your own research."

    html = f"""
<html>
<body style="font-family:Arial,sans-serif;max-width:750px;margin:0 auto;padding:20px;color:#222;">
  <h1 style="color:#154360;border-bottom:3px solid #154360;padding-bottom:10px;">
    📈 Daily Trade Signals
  </h1>
  <p style="color:#666;font-size:13px;margin-top:-10px;">{today_str} &nbsp;|&nbsp; Data: Yahoo Finance &nbsp;|&nbsp; Analysis: Claude AI</p>
  <div style="background:#f4f6f7;border-left:4px solid #154360;padding:12px 16px;border-radius:4px;margin-bottom:24px;">
    <strong>How to use this report:</strong> Review each signal before market open.
    Enter only trades that match your risk tolerance. Always respect your stop loss.
  </div>
  <div style="background:{'#fdf2f2' if market_warning else '#f0fdf4'};border-left:4px solid {'#e74c3c' if market_warning else '#2ecc71'};padding:12px 16px;border-radius:4px;margin-bottom:24px;font-size:13px;">
    <strong>🌍 Market Context:</strong><br>
    S&P 500: <strong>{sp500.get('trend','?')}</strong> — {sp500.get('detail','')}<br>
    TSX: <strong>{tsx.get('trend','?')}</strong> — {tsx.get('detail','')}
    {'<br><strong>⚠️ MARKET HEADWIND ACTIVE — exercise extra caution on LONG signals today</strong>' if market_warning else ''}
  </div>
  <div style="background:#fef9f0;border-left:4px solid #d4a017;padding:12px 16px;border-radius:4px;margin-bottom:24px;font-size:13px;">
    <strong>☪️ Shariah Compliance Legend (AAOIFI Standard):</strong><br>
    [COMPLIANT] — Passed all business activity and financial ratio screens<br>
    [DOUBTFUL] — Data unavailable or borderline — manual review recommended before trading<br>
    [NON-COMPLIANT] — Fails one or more AAOIFI screens — avoid if observing Shariah guidelines
  </div>
  <div style="white-space:pre-wrap;line-height:1.8;font-size:14px;">
{report}
  </div>
  <hr style="margin-top:30px;border:1px solid #ddd;">
  <h2 style="color:#154360;font-size:16px;">📊 Performance Tracker</h2>
  <div style="white-space:pre-wrap;line-height:1.8;font-size:13px;
              font-family:'Courier New',monospace;background:#f8f9fa;
              padding:16px;border-radius:6px;">
{perf_summary}

Recent Closed Trades:
{recent_trades}
  </div>
  <hr style="margin-top:40px;border:1px solid #ddd;">
  <p style="color:#aaa;font-size:11px;">
    ⚠️ This is an automated analysis and does not constitute financial advice.
    Past signal accuracy does not guarantee future results. Trade responsibly.
  </p>
</body>
</html>
"""

    msg.attach(MIMEText(plain, 'plain'))
    msg.attach(MIMEText(html,  'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

    print(f"✅ Email sent to {EMAIL_RECIPIENT}")


# ============================================================
# SIGNAL AUTO-LOGGER
# ============================================================

def _auto_log_signals(report):
    """
    Parse Claude's report text and log any trade signals found.
    Looks for the standard signal header format: TICKER — Style — LONG/SHORT
    """
    import re
    today = datetime.now().strftime("%Y-%m-%d")

    # Match lines like: 🟢 NVDA — Swing — LONG  or  NVDA — Position — SHORT
    pattern = re.compile(
        r'([A-Z]{1,5}(?:\.TO)?)\s*[—–-]+\s*(Swing|Position)\s*[—–-]+\s*(LONG|SHORT)',
        re.IGNORECASE
    )
    # Match entry, stop, TP1, TP2 from the table rows
    entry_pat  = re.compile(r'Entry Price.*?\$([0-9]+\.?[0-9]*)', re.IGNORECASE)
    stop_pat   = re.compile(r'Stop Loss.*?\$([0-9]+\.?[0-9]*)',   re.IGNORECASE)
    tp1_pat    = re.compile(r'Take Profit 1.*?\$([0-9]+\.?[0-9]*)', re.IGNORECASE)
    tp2_pat    = re.compile(r'Take Profit 2.*?\$([0-9]+\.?[0-9]*)', re.IGNORECASE)
    prob_pat   = re.compile(r'Probability.*?([0-9]+)%',            re.IGNORECASE)

    # Split report into per-trade blocks
    blocks = re.split(r'(?=(?:🟢|🔴)?\s*[A-Z]{1,5}(?:\.TO)?\s*[—–-]+\s*(?:Swing|Position))', report)

    for block in blocks:
        match = pattern.search(block)
        if not match:
            continue

        ticker    = match.group(1).upper()
        style     = match.group(2)
        direction = match.group(3).upper()

        try:
            entry = float(entry_pat.search(block).group(1))
            stop  = float(stop_pat.search(block).group(1))
            tp1   = float(tp1_pat.search(block).group(1))
            tp2_m = tp2_pat.search(block)
            tp2   = float(tp2_m.group(1)) if tp2_m else None
            prob_m = prob_pat.search(block)
            prob  = int(prob_m.group(1)) if prob_m else 0

            log_signal(ticker, direction, entry, stop, tp1, tp2,
                       style, prob, signal_date=today)
        except Exception:
            pass  # If parsing fails, skip silently


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"🤖 Stock Bot starting — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

    all_tickers = WATCHLIST["US"] + WATCHLIST["TSX"]
    print(f"📊 Watchlist: {len(all_tickers)} tickers\n")

    # --- Phase 2: Fetch market context once for the whole session ---
    print("🌍 Fetching market context (S&P 500 + TSX)...")
    market_ctx = get_market_context()
    sp500  = market_ctx.get("SP500", {})
    tsx    = market_ctx.get("TSX",   {})

    market_block = f"""
{'='*55}
MARKET CONTEXT (checked before individual stocks)
  S&P 500 : {sp500.get('trend', 'UNKNOWN')} — {sp500.get('detail', '')}
  TSX     : {tsx.get('trend',  'UNKNOWN')} — {tsx.get('detail',  '')}
{'⚠️  MARKET WARNING: Broad market is BEARISH — apply headwind labels to LONG signals' if sp500.get('warning') or tsx.get('warning') else '✅  Market conditions are neutral to bullish'}
{'='*55}
"""
    print(market_block)

    summaries = [market_block]
    for ticker in all_tickers:
        print(f"  → Fetching {ticker}...")
        result = analyze_ticker(ticker)
        if result:
            summaries.append(result)

    if not summaries:
        print("❌ Could not retrieve any data. Check internet connectivity.")
        return

    print(f"\n✅ Analyzed {len(summaries)}/{len(all_tickers)} tickers")
    print("🔄 Resolving open trades...")
    resolve_open_trades()

    print("🧠 Sending to Claude for analysis...\n")
    report = get_trade_signals(summaries)
    print(report[:500] + "...\n")

    # Parse signals from Claude's report and log them
    print("📝 Logging today's signals...")
    _auto_log_signals(report)

    # Build performance summary
    perf_summary  = generate_performance_summary()
    recent_trades = get_recent_closed_trades(10)

    market_warning = sp500.get("warning", False) or tsx.get("warning", False)

    print("📧 Sending email...")
    send_email(report, perf_summary, recent_trades, sp500, tsx, market_warning)

    print("✅ Done!")


if __name__ == "__main__":
    main()


