"""Snapshot of S&P 500 index (SPX, European, cash-settled) option quotes from Yahoo Finance via yfinance.
Writes data/spx_<UTC timestamp>.csv with bid/ask for every listed strike of the chosen expiries."""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TARGET_DAYS = [30, 60, 90, 180, 270, 365, 540, 730]


def main():
    tk = yf.Ticker("^SPX")
    now = dt.datetime.now(dt.timezone.utc)
    spot = float(tk.history(period="1d", interval="1m")["Close"].iloc[-1])
    expiries = pd.to_datetime(list(tk.options))
    days = (expiries - pd.Timestamp(now.date())).days
    chosen = sorted({expiries[abs(days - t).argmin()] for t in TARGET_DAYS})
    rows = []
    for e in chosen:
        ch = tk.option_chain(e.strftime("%Y-%m-%d"))
        for kind, df in (("C", ch.calls), ("P", ch.puts)):
            d = df[["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "lastTradeDate"]].copy()
            d["type"] = kind
            d["expiry"] = e.strftime("%Y-%m-%d")
            rows.append(d)
    out = pd.concat(rows)
    out["spot"] = spot
    out["snapshot_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    (ROOT / "data").mkdir(exist_ok=True)
    f = ROOT / "data" / f"spx_{now.strftime('%Y%m%dT%H%M')}.csv"
    out.to_csv(f, index=False)
    print(f, len(out), "quotes,", len(chosen), "expiries, spot", spot,
          "| with bid>0:", int((out.bid > 0).sum()), file=sys.stderr)


if __name__ == "__main__":
    main()
