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

# ============================================================
# YOUR WATCHLIST — Add or remove tickers here anytime
# ============================================================

WATCHLIST = {
    "US": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
        "META", "TSLA", "SERV", "UBER", "LLY", "UPWK", "ONDS"
    ],
    "TSX": [
        "CSU.TO", "TD.TO", "BNS.TO", "ENB.TO", "CNQ.TO",
        "SU.TO", "BCE.TO", "ATRL.TO", "TRP.TO", "ABX.TO", "DOL.TO"
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
    """Send all ticker data to Claude and get back the daily trade report"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    combined_data = "\n".join(ticker_summaries)
    today_str     = datetime.now().strftime("%A, %B %d, %Y")

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


# ============================================================
# EMAIL
# ============================================================

def send_email(report):
    """Send the daily report as a formatted HTML email"""
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    subject   = f"📈 Daily Trade Signals — {today_str}"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = EMAIL_SENDER
    msg['To']      = EMAIL_RECIPIENT

    plain = f"DAILY TRADE SIGNALS — {today_str}\n{'='*50}\n\n{report}\n\n---\n⚠️ Not financial advice. Do your own research."

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
  <div style="background:#fef9f0;border-left:4px solid #d4a017;padding:12px 16px;border-radius:4px;margin-bottom:24px;font-size:13px;">
    <strong>☪️ Shariah Compliance Legend (AAOIFI Standard):</strong><br>
    [COMPLIANT] — Passed all business activity and financial ratio screens<br>
    [DOUBTFUL] — Data unavailable or borderline — manual review recommended before trading<br>
    [NON-COMPLIANT] — Fails one or more AAOIFI screens — avoid if observing Shariah guidelines
  </div>
  <div style="white-space:pre-wrap;line-height:1.8;font-size:14px;">
{report}
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
# MAIN
# ============================================================

def main():
    print(f"🤖 Stock Bot starting — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

    all_tickers = WATCHLIST["US"] + WATCHLIST["TSX"]
    print(f"📊 Watchlist: {len(all_tickers)} tickers\n")

    summaries = []
    for ticker in all_tickers:
        print(f"  → Fetching {ticker}...")
        result = analyze_ticker(ticker)
        if result:
            summaries.append(result)

    if not summaries:
        print("❌ Could not retrieve any data. Check internet connectivity.")
        return

    print(f"\n✅ Analyzed {len(summaries)}/{len(all_tickers)} tickers")
    print("🧠 Sending to Claude for analysis...\n")

    report = get_trade_signals(summaries)
    print(report[:500] + "...\n")   # Preview in logs

    print("📧 Sending email...")
    send_email(report)

    print("✅ Done!")


if __name__ == "__main__":
    main()

