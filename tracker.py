"""
tracker.py — Trade Performance Tracker
Logs signals, resolves open trades, and generates performance summaries.
Data is stored in trades_log.json which is committed back to the GitHub repo daily.
"""

import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

TRADES_LOG_FILE = "trades_log.json"


# ============================================================
# LOAD / SAVE
# ============================================================

def load_trades():
    """Load the trades log from disk, return empty list if not found."""
    if os.path.exists(TRADES_LOG_FILE):
        with open(TRADES_LOG_FILE, "r") as f:
            return json.load(f)
    return []


def save_trades(trades):
    """Save the trades log to disk."""
    with open(TRADES_LOG_FILE, "w") as f:
        json.dump(trades, f, indent=2)


# ============================================================
# LOG A NEW SIGNAL
# ============================================================

def log_signal(ticker, direction, entry, stop_loss, tp1, tp2,
               trade_style, probability, signal_date=None):
    """
    Add a new trade signal to the log.
    Status starts as OPEN and gets resolved by resolve_open_trades().
    """
    trades = load_trades()

    date_str = signal_date or datetime.now().strftime("%Y-%m-%d")

    # Avoid duplicate entries for same ticker on same date
    for t in trades:
        if t["ticker"] == ticker and t["signal_date"] == date_str:
            print(f"  ℹ️  {ticker} already logged for {date_str}, skipping.")
            return

    risk = abs(entry - stop_loss)
    rr   = round(abs(tp1 - entry) / risk, 2) if risk > 0 else 0

    trade = {
        "ticker":       ticker,
        "signal_date":  date_str,
        "direction":    direction.upper(),   # LONG or SHORT
        "entry":        round(entry, 2),
        "stop_loss":    round(stop_loss, 2),
        "tp1":          round(tp1, 2),
        "tp2":          round(tp2, 2) if tp2 else None,
        "trade_style":  trade_style,
        "probability":  probability,
        "risk_reward":  rr,
        "status":       "OPEN",
        "outcome":      None,
        "exit_price":   None,
        "exit_date":    None,
        "pnl_pct":      None,
        "pnl_rr":       None,
    }

    trades.append(trade)
    save_trades(trades)
    print(f"  📝 Logged signal: {ticker} {direction} @ {entry}")


# ============================================================
# RESOLVE OPEN TRADES
# ============================================================

def resolve_open_trades():
    """
    Check all OPEN trades and resolve them if price has hit TP1 or stop loss.
    Uses Yahoo Finance data to check high/low on each day since signal date.
    Gives trades up to 30 calendar days to resolve before marking EXPIRED.
    """
    trades    = load_trades()
    today     = datetime.now().date()
    updated   = 0

    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        signal_date = datetime.strptime(trade["signal_date"], "%Y-%m-%d").date()
        days_open   = (today - signal_date).days

        # Expire trades older than 30 days that haven't resolved
        if days_open > 30:
            trade["status"]  = "EXPIRED"
            trade["outcome"] = "EXPIRED"
            trade["exit_date"] = str(today)
            updated += 1
            continue

        try:
            # Fetch daily data from signal date to today
            start = (signal_date + timedelta(days=1)).strftime("%Y-%m-%d")
            end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")
            df    = yf.download(trade["ticker"], start=start, end=end,
                                interval="1d", progress=False, auto_adjust=True)

            if df.empty:
                continue

            entry     = trade["entry"]
            stop      = trade["stop_loss"]
            tp1       = trade["tp1"]
            direction = trade["direction"]

            for date_idx, row in df.iterrows():
                day_high = float(row["High"])
                day_low  = float(row["Low"])
                day_close= float(row["Close"])
                day_str  = str(date_idx.date())

                if direction == "LONG":
                    hit_tp   = day_high >= tp1
                    hit_stop = day_low  <= stop
                else:  # SHORT
                    hit_tp   = day_low  <= tp1
                    hit_stop = day_high >= stop

                if hit_tp and hit_stop:
                    # Both hit same day — conservatively call it a stop (worst case)
                    hit_stop = True
                    hit_tp   = False

                if hit_tp:
                    pnl_pct = round(abs(tp1 - entry) / entry * 100, 2)
                    pnl_pct = pnl_pct if direction == "LONG" else -pnl_pct
                    trade.update({
                        "status":     "CLOSED",
                        "outcome":    "WIN",
                        "exit_price": tp1,
                        "exit_date":  day_str,
                        "pnl_pct":    pnl_pct,
                        "pnl_rr":     trade["risk_reward"],
                    })
                    updated += 1
                    break

                elif hit_stop:
                    pnl_pct = round(abs(stop - entry) / entry * 100, 2)
                    pnl_pct = -pnl_pct if direction == "LONG" else pnl_pct
                    risk    = abs(entry - stop)
                    trade.update({
                        "status":     "CLOSED",
                        "outcome":    "LOSS",
                        "exit_price": stop,
                        "exit_date":  day_str,
                        "pnl_pct":    -abs(pnl_pct),
                        "pnl_rr":     -1.0,
                    })
                    updated += 1
                    break

        except Exception as e:
            print(f"  ⚠️  Could not resolve {trade['ticker']}: {e}")

    if updated > 0:
        save_trades(trades)
        print(f"  ✅ Resolved {updated} trade(s)")


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def generate_performance_summary():
    """
    Generate a full performance breakdown from the trades log.
    Returns a formatted text block for use in emails.
    """
    trades  = load_trades()
    closed  = [t for t in trades if t["status"] == "CLOSED"]
    open_t  = [t for t in trades if t["status"] == "OPEN"]
    wins    = [t for t in closed  if t["outcome"] == "WIN"]
    losses  = [t for t in closed  if t["outcome"] == "LOSS"]

    if not closed:
        return """
PERFORMANCE TRACKER
===================
No closed trades yet. Signals are being tracked — check back soon.
Open trades being monitored: {}
""".format(len(open_t))

    total       = len(closed)
    win_rate    = round(len(wins) / total * 100, 1)
    avg_win_rr  = round(sum(t["pnl_rr"] for t in wins)   / len(wins),   2) if wins   else 0
    avg_loss_rr = round(sum(t["pnl_rr"] for t in losses) / len(losses), 2) if losses else 0
    total_rr    = round(sum(t["pnl_rr"] for t in closed if t["pnl_rr"] is not None), 2)
    avg_pnl_pct = round(sum(t["pnl_pct"] for t in closed if t["pnl_pct"] is not None) / total, 2)

    # Best and worst trades
    best  = max(closed, key=lambda t: t["pnl_rr"]  or 0)
    worst = min(closed, key=lambda t: t["pnl_rr"]  or 0)

    # Last 10 trades as a quick visual (W = win, L = loss)
    last10 = closed[-10:]
    streak = " ".join("✅" if t["outcome"] == "WIN" else "❌" for t in last10)

    # By ticker breakdown
    ticker_stats = {}
    for t in closed:
        tk = t["ticker"]
        if tk not in ticker_stats:
            ticker_stats[tk] = {"wins": 0, "losses": 0, "rr": 0}
        if t["outcome"] == "WIN":
            ticker_stats[tk]["wins"] += 1
        else:
            ticker_stats[tk]["losses"] += 1
        ticker_stats[tk]["rr"] += t["pnl_rr"] or 0

    top_ticker  = max(ticker_stats, key=lambda k: ticker_stats[k]["rr"])
    flop_ticker = min(ticker_stats, key=lambda k: ticker_stats[k]["rr"])

    summary = f"""
{'='*55}
📊 PERFORMANCE SCORECARD
{'='*55}

  Total Signals Tracked : {len(trades)}
  Closed Trades         : {total}
  Open Trades           : {len(open_t)}

  Win Rate              : {win_rate}%  ({len(wins)}W / {len(losses)}L)
  Total R:R Earned      : {total_rr:+.2f}R
  Avg P&L per Trade     : {avg_pnl_pct:+.2f}%
  Avg Win  R:R          : +{avg_win_rr}R
  Avg Loss R:R          : {avg_loss_rr}R

  Best Trade  : {best['ticker']} on {best['signal_date']} → {best['pnl_rr']:+.2f}R ({best['pnl_pct']:+.2f}%)
  Worst Trade : {worst['ticker']} on {worst['signal_date']} → {worst['pnl_rr']:+.2f}R ({worst['pnl_pct']:+.2f}%)

  Best Ticker  : {top_ticker}  (total {ticker_stats[top_ticker]['rr']:+.2f}R)
  Worst Ticker : {flop_ticker} (total {ticker_stats[flop_ticker]['rr']:+.2f}R)

  Last 10 Trades:
  {streak}

{'='*55}
"""
    return summary


def get_recent_closed_trades(n=10):
    """Return the last N closed trades as a formatted table."""
    trades = load_trades()
    closed = [t for t in trades if t["status"] == "CLOSED"][-n:]

    if not closed:
        return "  No closed trades yet.\n"

    lines = [
        f"  {'Date':<12} {'Ticker':<10} {'Dir':<6} {'Entry':>7} {'Exit':>7} {'R:R':>6} {'P&L%':>7} {'Result':<8}",
        "  " + "-" * 65,
    ]
    for t in reversed(closed):
        lines.append(
            f"  {t['signal_date']:<12} {t['ticker']:<10} {t['direction']:<6} "
            f"${t['entry']:>6.2f} ${t['exit_price']:>6.2f} "
            f"{t['pnl_rr']:>+6.2f}R {t['pnl_pct']:>+6.2f}% "
            f"{'✅ WIN' if t['outcome'] == 'WIN' else '❌ LOSS'}"
        )
    return "\n".join(lines)
