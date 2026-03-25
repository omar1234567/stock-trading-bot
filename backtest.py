#!/usr/bin/env python3
"""
backtest.py — 2-Month Historical Backtest
Replays the bot's signal logic on the past 2 months of data.
Evaluates each trade outcome (TP hit or stopped out) and sends a full report email.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import ta

# ============================================================
# SETTINGS
# ============================================================

EMAIL_SENDER      = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD    = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT   = os.environ.get("EMAIL_RECIPIENT")

WATCHLIST = {
    "US": [
        "NVDA", "AAPL", "MSFT", "GOOGL", "AVGO",
        "AMD", "CSCO", "MU", "LRCX", "ORCL", "TXN",
        "LLY", "JNJ", "ABBV", "MRK", "ABT", "TMO",
        "TSLA", "HD", "PG", "PEP", "LIN", "XOM",
    ],
    "TSX": [
        "CSU.TO", "SHOP.TO",
        "TIH.TO", "TFII.TO", "WCN.TO", "CNR.TO", "CP.TO",
        "CNQ.TO", "SU.TO",
        "ATD.TO", "MG.TO",
    ]
}

BACKTEST_DAYS = 60       # How many calendar days back to test
FORWARD_DAYS  = 15       # Max trading days to check for TP/stop hit
MIN_RR        = 2.0      # Minimum R:R to take a trade


# ============================================================
# SIGNAL LOGIC
# ============================================================

def compute_indicators(df):
    """Add all technical indicators to the dataframe."""
    df = df.copy()
    df["RSI"]    = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["EMA9"]   = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
    df["EMA21"]  = ta.trend.EMAIndicator(df["Close"], window=21).ema_indicator()
    df["EMA50"]  = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["EMA200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    df["ATR"]    = ta.volatility.AverageTrueRange(
                       df["High"], df["Low"], df["Close"], window=14).average_true_range()
    return df


def find_support(df, window=10):
    """Find nearest support level below current price."""
    current = df["Close"].iloc[-1]
    lows = []
    for i in range(window, len(df) - 1):
        low = df["Low"].iloc[i]
        if low == df["Low"].iloc[i - window : i + window].min():
            lows.append(low)
    supports = sorted(set(s for s in lows if s < current))
    return supports[-1] if supports else current * 0.97   # fallback: 3% below


def find_resistance(df, window=10):
    """Find nearest resistance level above current price."""
    current = df["Close"].iloc[-1]
    highs = []
    for i in range(window, len(df) - 1):
        high = df["High"].iloc[i]
        if high == df["High"].iloc[i - window : i + window].max():
            highs.append(high)
    resistances = sorted(set(r for r in highs if r > current))
    return resistances[0] if resistances else current * 1.06   # fallback: 6% above


def detect_signal(df):
    """
    Rule-based signal detection mirroring the bot's criteria.
    Returns: ('LONG', score) or ('SHORT', score) or (None, 0)

    Scoring — 1 point per indicator aligned:
      1. EMA alignment (9 > 21 > 50 = bullish, reverse = bearish)
      2. Price vs EMA200 (above = bullish, below = bearish)
      3. RSI zone (40-65 = bullish momentum, 35-60 = bearish momentum)
      4. Candlestick (last candle body direction)

    Signal fires if score >= 3 (at least 3 of 4 aligned)
    """
    if len(df) < 50:
        return None, 0

    row    = df.iloc[-1]
    price  = row["Close"]
    rsi    = row["RSI"]
    ema9   = row["EMA9"]
    ema21  = row["EMA21"]
    ema50  = row["EMA50"]
    ema200 = row["EMA200"]

    if any(pd.isna(v) for v in [rsi, ema9, ema21, ema50, ema200]):
        return None, 0

    # Previous row for crossover check
    prev = df.iloc[-2]

    bull_score = 0
    bear_score = 0

    # 1. EMA alignment
    if ema9 > ema21 > ema50:
        bull_score += 1
    elif ema9 < ema21 < ema50:
        bear_score += 1

    # 2. Price vs 200 EMA
    if price > ema200:
        bull_score += 1
    else:
        bear_score += 1

    # 3. RSI momentum zone
    if 40 <= rsi <= 65:
        bull_score += 1
    elif 35 <= rsi <= 60 and rsi < 50:
        bear_score += 1

    # 4. Recent candle direction
    candle_body = row["Close"] - row["Open"]
    if candle_body > 0:
        bull_score += 1
    elif candle_body < 0:
        bear_score += 1

    if bull_score >= 3:
        return "LONG", bull_score
    elif bear_score >= 3:
        return "SHORT", bear_score
    return None, 0


# ============================================================
# BACKTEST ENGINE
# ============================================================

def simulate_trade(ticker, signal_date, direction, entry, stop, tp1, forward_df):
    """
    Given a signal, scan forward up to FORWARD_DAYS to see if TP or stop is hit first.
    Returns outcome dict.
    """
    future = forward_df[forward_df.index > signal_date].head(FORWARD_DAYS)

    for date_idx, row in future.iterrows():
        high  = float(row["High"])
        low   = float(row["Low"])

        if direction == "LONG":
            hit_tp   = high >= tp1
            hit_stop = low  <= stop
        else:
            hit_tp   = low  <= tp1
            hit_stop = high >= stop

        if hit_tp and hit_stop:
            hit_stop = True
            hit_tp   = False

        if hit_tp:
            pnl_pct = abs(tp1 - entry) / entry * 100
            rr      = abs(tp1 - entry) / abs(entry - stop)
            return {
                "outcome":    "WIN",
                "exit_price": round(tp1, 2),
                "exit_date":  str(date_idx.date()),
                "pnl_pct":    round(pnl_pct, 2),
                "pnl_rr":     round(rr, 2),
                "days_held":  (date_idx.date() - signal_date.date()).days,
            }
        elif hit_stop:
            pnl_pct = abs(stop - entry) / entry * 100
            return {
                "outcome":    "LOSS",
                "exit_price": round(stop, 2),
                "exit_date":  str(date_idx.date()),
                "pnl_pct":    -round(pnl_pct, 2),
                "pnl_rr":     -1.0,
                "days_held":  (date_idx.date() - signal_date.date()).days,
            }

    return {
        "outcome":    "OPEN",
        "exit_price": None,
        "exit_date":  None,
        "pnl_pct":    None,
        "pnl_rr":     None,
        "days_held":  None,
    }


def run_backtest():
    """Run the full 2-month backtest across all watchlist tickers."""
    print(f"🔬 Starting 2-month backtest — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n")

    all_tickers  = WATCHLIST["US"] + WATCHLIST["TSX"]
    end_date     = datetime.now()
    start_date   = end_date - timedelta(days=BACKTEST_DAYS + 250)  # extra for indicator warmup
    backtest_start = end_date - timedelta(days=BACKTEST_DAYS)

    results = []

    for ticker in all_tickers:
        print(f"  → Backtesting {ticker}...")
        try:
            df = yf.download(ticker,
                             start=start_date.strftime("%Y-%m-%d"),
                             end=end_date.strftime("%Y-%m-%d"),
                             interval="1d", progress=False, auto_adjust=True)

            if df.empty or len(df) < 100:
                print(f"    ⚠️  Not enough data, skipping.")
                continue

            df = compute_indicators(df)
            currency = "CAD" if ticker.endswith(".TO") else "USD"

            # Walk forward through each trading day in the backtest window
            backtest_days_index = df[df.index >= pd.Timestamp(backtest_start)].index

            for signal_date in backtest_days_index:
                # Use only data available up to and including this date (no lookahead)
                hist = df[df.index <= signal_date].copy()

                if len(hist) < 60:
                    continue

                direction, score = detect_signal(hist)
                if not direction:
                    continue

                entry  = float(hist["Close"].iloc[-1])
                atr    = float(hist["ATR"].iloc[-1])

                if direction == "LONG":
                    support  = find_support(hist)
                    stop     = min(support, entry - atr * 1.5)
                    stop     = round(stop, 2)
                    risk     = entry - stop
                    if risk <= 0 or risk / entry > 0.10:  # skip if stop > 10% away
                        continue
                    tp1 = round(entry + risk * MIN_RR, 2)
                    tp2 = round(entry + risk * 3,      2)
                else:
                    resistance = find_resistance(hist)
                    stop       = max(resistance, entry + atr * 1.5)
                    stop       = round(stop, 2)
                    risk       = stop - entry
                    if risk <= 0 or risk / entry > 0.10:
                        continue
                    tp1 = round(entry - risk * MIN_RR, 2)
                    tp2 = round(entry - risk * 3,      2)

                rr = abs(tp1 - entry) / abs(entry - stop)
                if rr < MIN_RR:
                    continue

                # Simulate the trade using forward data
                outcome = simulate_trade(ticker, signal_date, direction,
                                         entry, stop, tp1, df)

                results.append({
                    "ticker":       ticker,
                    "currency":     currency,
                    "signal_date":  str(signal_date.date()),
                    "direction":    direction,
                    "score":        score,
                    "entry":        round(entry, 2),
                    "stop_loss":    stop,
                    "tp1":          tp1,
                    "tp2":          tp2,
                    "risk_reward":  round(rr, 2),
                    **outcome,
                })

        except Exception as e:
            print(f"    ❌ Error on {ticker}: {e}")

    print(f"\n✅ Backtest complete — {len(results)} signals found\n")
    return results


# ============================================================
# REPORT GENERATION
# ============================================================

def build_report(results):
    """Build a comprehensive backtest report from results."""
    if not results:
        return "No signals were generated during the backtest period."

    closed   = [r for r in results if r["outcome"] in ("WIN", "LOSS")]
    wins     = [r for r in closed  if r["outcome"] == "WIN"]
    losses   = [r for r in closed  if r["outcome"] == "LOSS"]
    open_t   = [r for r in results if r["outcome"] == "OPEN"]

    total    = len(closed)
    win_rate = round(len(wins) / total * 100, 1) if total > 0 else 0
    total_rr = round(sum(r["pnl_rr"] for r in closed if r["pnl_rr"]), 2)
    avg_pnl  = round(sum(r["pnl_pct"] for r in closed if r["pnl_pct"] is not None) / total, 2) if total > 0 else 0
    avg_days = round(sum(r["days_held"] for r in closed if r["days_held"]) / total, 1) if total > 0 else 0

    best  = max(closed, key=lambda r: r["pnl_rr"]  or 0) if closed else None
    worst = min(closed, key=lambda r: r["pnl_rr"]  or 0) if closed else None

    # Per-ticker breakdown
    ticker_stats = {}
    for r in closed:
        tk = r["ticker"]
        if tk not in ticker_stats:
            ticker_stats[tk] = {"wins": 0, "losses": 0, "rr": 0.0, "signals": 0}
        ticker_stats[tk]["signals"] += 1
        ticker_stats[tk]["rr"]      += r["pnl_rr"] or 0
        if r["outcome"] == "WIN":
            ticker_stats[tk]["wins"] += 1
        else:
            ticker_stats[tk]["losses"] += 1

    # Sort tickers by total R:R
    sorted_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["rr"], reverse=True)

    # Build the full trade list
    trade_rows = []
    for r in sorted(results, key=lambda x: x["signal_date"]):
        outcome_icon = "✅" if r["outcome"] == "WIN" else ("❌" if r["outcome"] == "LOSS" else "⏳")
        pnl_str  = f"{r['pnl_pct']:+.2f}%"  if r["pnl_pct"]  is not None else "OPEN"
        rr_str   = f"{r['pnl_rr']:+.2f}R"   if r["pnl_rr"]   is not None else "OPEN"
        days_str = f"{r['days_held']}d"      if r["days_held"] is not None else "—"
        trade_rows.append(
            f"  {r['signal_date']} | {r['ticker']:<10} | {r['direction']:<5} | "
            f"${r['entry']:.2f} → ${r['exit_price'] or '?':<8} | "
            f"{rr_str:<8} | {pnl_str:<8} | {days_str:<5} | {outcome_icon}"
        )

    lines = [
        "=" * 65,
        "📊  2-MONTH BACKTEST REPORT",
        f"    Period: {(datetime.now() - timedelta(days=60)).strftime('%b %d, %Y')} → {datetime.now().strftime('%b %d, %Y')}",
        f"    Watchlist: {len(WATCHLIST['US'])} US + {len(WATCHLIST['TSX'])} TSX tickers",
        "=" * 65,
        "",
        "── SUMMARY ─────────────────────────────────────────────",
        f"  Total Signals Generated : {len(results)}",
        f"  Closed Trades           : {total}",
        f"  Still Open              : {len(open_t)}",
        "",
        f"  Win Rate                : {win_rate}%  ({len(wins)} wins / {len(losses)} losses)",
        f"  Total R:R Earned        : {total_rr:+.2f}R",
        f"  Avg P&L per Trade       : {avg_pnl:+.2f}%",
        f"  Avg Days Held           : {avg_days} days",
        "",
    ]

    if best:
        lines += [
            f"  🏆 Best Trade  : {best['ticker']} {best['direction']} on {best['signal_date']}",
            f"                   Entry ${best['entry']} → Exit ${best['exit_price']}",
            f"                   Result: {best['pnl_rr']:+.2f}R  ({best['pnl_pct']:+.2f}%)",
            "",
            f"  💔 Worst Trade : {worst['ticker']} {worst['direction']} on {worst['signal_date']}",
            f"                   Entry ${worst['entry']} → Exit ${worst['exit_price']}",
            f"                   Result: {worst['pnl_rr']:+.2f}R  ({worst['pnl_pct']:+.2f}%)",
            "",
        ]

    lines += [
        "── PER-TICKER BREAKDOWN ─────────────────────────────────",
        f"  {'Ticker':<12} {'Signals':>8} {'Wins':>6} {'Losses':>7} {'Total R:R':>10} {'Win Rate':>9}",
        "  " + "-" * 55,
    ]
    for tk, s in sorted_tickers:
        wr = round(s["wins"] / (s["wins"] + s["losses"]) * 100, 0) if (s["wins"] + s["losses"]) > 0 else 0
        lines.append(
            f"  {tk:<12} {s['signals']:>8} {s['wins']:>6} {s['losses']:>7} "
            f"  {s['rr']:>+8.2f}R  {wr:>7.0f}%"
        )

    lines += [
        "",
        "── ALL TRADES ───────────────────────────────────────────",
        f"  {'Date':<12} | {'Ticker':<10} | {'Dir':<5} | {'Entry → Exit':<20} | {'R:R':<8} | {'P&L%':<8} | {'Days':<5} | Result",
        "  " + "-" * 85,
    ]
    lines += trade_rows

    lines += [
        "",
        "=" * 65,
        "⚠️  DISCLAIMER",
        "  Backtest results are simulated using historical data.",
        "  Past performance does not guarantee future results.",
        "  Signal logic is rule-based and may differ from live Claude AI analysis.",
        "  Always do your own research before trading.",
        "=" * 65,
    ]

    return "\n".join(lines)


# ============================================================
# EMAIL
# ============================================================

def send_backtest_email(report):
    """Send the backtest report as a formatted email."""
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    subject   = f"📊 2-Month Backtest Report — {today_str}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT

    plain = f"2-MONTH BACKTEST REPORT\n{'='*50}\n\n{report}"

    html = f"""
<html>
<body style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px;color:#222;">
  <h1 style="color:#154360;border-bottom:3px solid #154360;padding-bottom:10px;">
    📊 2-Month Backtest Report
  </h1>
  <p style="color:#666;font-size:13px;margin-top:-10px;">
    {today_str} &nbsp;|&nbsp; Strategy: AAOIFI Shariah-Compliant Technical Signals
  </p>
  <div style="background:#fef9f0;border-left:4px solid #d4a017;padding:12px 16px;
              border-radius:4px;margin-bottom:24px;font-size:13px;">
    <strong>⚠️ About this report:</strong> Results are simulated on historical data using
    rule-based signals. Live signals are generated by Claude AI and may differ.
    Past performance does not guarantee future results.
  </div>
  <div style="white-space:pre-wrap;line-height:1.8;font-size:13px;
              font-family:'Courier New',monospace;background:#f8f9fa;
              padding:20px;border-radius:6px;">
{report}
  </div>
  <hr style="margin-top:40px;border:1px solid #ddd;">
  <p style="color:#aaa;font-size:11px;">
    This is an automated backtest report. Not financial advice.
  </p>
</body>
</html>
"""

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

    print(f"✅ Backtest email sent to {EMAIL_RECIPIENT}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = run_backtest()
    report  = build_report(results)
    print(report[:1000] + "\n...\n")
    print("📧 Sending backtest email...")
    send_backtest_email(report)
    print("✅ Done!")
