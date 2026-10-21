"""Calibrate Heston to an SPX option snapshot and price a 1-year Barrier Reverse Convertible (BRC).

1. Per expiry, the discount factor DF and forward F come from put-call parity on near-the-money pairs
   (C - P = DF (F - K), least squares), so no rate or dividend assumption is needed.
2. Out-of-the-money mids (puts below F, calls above) are turned into implied volatilities. Prices are normalised
   by DF F and strikes by F, so every expiry is priced on a unit forward with r = q = 0.
3. Heston is calibrated to all expiries at once (C++ Levenberg-Marquardt on implied-vol errors).
4. BRC on SPX, 1 year: capital protected unless the index ever trades at or below 70 % of its initial level and ends
   below 100 %; fixed coupon. Investor = zero-coupon bond + coupon - down-and-in put / S0. The fair coupon (no issuer
   margin) is c = (1 - DF + DIP / S0) / DF. The DIP is priced with flat Black-Scholes at the ATM vol, flat
   Black-Scholes at the vol of the 70 % strike, and Heston (Monte Carlo with the Brownian bridge, and ADI PDE).
Usage: python scripts/calibrate_spx.py data/spx_<stamp>.csv
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import qlite  # noqa: E402


def parity(df_e):
    c = df_e[df_e.type == "C"].set_index("strike")["mid"]
    p = df_e[df_e.type == "P"].set_index("strike")["mid"]
    k = c.index.intersection(p.index)
    S = df_e.spot.iloc[0]
    k = k[(k > 0.95 * S) & (k < 1.05 * S)]
    y = (c[k] - p[k]).values
    A = np.column_stack([np.ones(len(k)), -k.values])
    (a, df), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([a, df])
    return float(df), float(a / df), float(np.sqrt(np.mean(resid ** 2))), len(k)


def main(path):
    raw = pd.read_csv(path)
    snap = raw.snapshot_utc.iloc[0]
    S0 = float(raw.spot.iloc[0])
    # quotes are the last close: measure time from the latest trade, to the AM settlement (09:30 New York = 13:30 UTC)
    t0 = pd.to_datetime(raw.lastTradeDate, utc=True).max()
    raw = raw[raw.contractSymbol.str[:-15] == "SPX"]          # monthly AM-settled series only (no SPXW weeklies)
    raw = raw[(raw.bid > 0) & (raw.ask > raw.bid)].copy()
    raw["mid"] = 0.5 * (raw.bid + raw.ask)
    raw["T"] = (pd.to_datetime(raw.expiry).dt.tz_localize("UTC") + pd.Timedelta(hours=13.5) - t0).dt.total_seconds() / (365.0 * 86400)
    out = dict(snapshot_utc=snap, quote_time_utc=str(t0), spot=S0, expiries=[])
    quotes, rows = [], []
    for e, d in raw.groupby("expiry"):
        T = float(d["T"].iloc[0])
        DF, F, perr, npar = parity(d)
        r_e = -np.log(DF) / T
        d = d.assign(m=d.strike / F)
        otm = d[((d.type == "P") & (d.m < 1.0)) | ((d.type == "C") & (d.m >= 1.0))]
        width = min(0.45, 2.5 * 0.2 * np.sqrt(T))          # about 2.5 standard deviations at 20 % vol
        otm = otm[(np.abs(np.log(otm.m)) <= width) & ((otm.ask - otm.bid) / otm.mid < 0.10)]
        # thin to about 15 strikes per expiry, evenly in log-moneyness
        grid = np.linspace(-width, width, 15)
        idx = sorted({int(np.abs(np.log(otm.m.values) - g).argmin()) for g in grid})
        otm = otm.iloc[idx]
        for o in otm.itertuples():
            call = o.type == "C"
            px = o.mid / (DF * F)
            iv = qlite.implied_vol(px, 1.0, o.m, T, 0.0, 0.0, call)
            ivb = qlite.implied_vol(o.bid / (DF * F), 1.0, o.m, T, 0.0, 0.0, call)
            iva = qlite.implied_vol(o.ask / (DF * F), 1.0, o.m, T, 0.0, 0.0, call)
            if np.isfinite(iv):
                quotes.append((float(o.m), T, float(iv), 1.0, bool(call)))
                rows.append(dict(expiry=e, T=T, m=o.m, K=o.strike, iv=iv, iv_bid=ivb, iv_ask=iva, call=call))
        out["expiries"].append(dict(expiry=e, T=T, DF=DF, F=F, r=r_e, parity_rmse=perr, n_pairs=npar, n_quotes=len(otm)))
    Q = pd.DataFrame(rows)

    h0 = qlite.Heston(0.02, 2.0, 0.04, 0.6, -0.7)
    h, rmse, its = qlite.calibrate(quotes, 1.0, 0.0, 0.0, h0)
    out["heston"] = dict(v0=h.v0, kappa=h.kappa, theta=h.theta, sigma=h.sigma, rho=h.rho, feller=h.feller(),
                         rmse_vol_pts=100 * rmse, iterations=its, n_quotes=len(quotes))
    Q["iv_heston"] = [qlite.implied_vol(qlite.heston_price(h, 1.0, m, T, 0.0, 0.0, c), 1.0, m, T, 0.0, 0.0, c)
                      for m, T, c in zip(Q.m, Q["T"], Q.call)]
    Q["err_pts"] = 100 * (Q.iv_heston - Q.iv)
    Q["inside_bidask"] = (Q.iv_heston >= Q.iv_bid) & (Q.iv_heston <= Q.iv_ask)
    out["fit_by_expiry"] = Q.groupby("expiry").apply(
        lambda g: dict(T=float(g["T"].iloc[0]), rmse_pts=float(np.sqrt(np.mean(g.err_pts ** 2))),
                       max_abs_pts=float(g.err_pts.abs().max()), inside_bidask=float(g.inside_bidask.mean())),
        include_groups=False).to_dict()

    # 1-year BRC: use the expiry closest to one year for DF and forward
    e1 = min(out["expiries"], key=lambda x: abs(x["T"] - 1.0))
    T, DF, F = e1["T"], e1["DF"], e1["F"]
    r = -np.log(DF) / T
    q = r - np.log(F / S0) / T
    K, H = S0, 0.7 * S0
    q1 = Q[Q.expiry == e1["expiry"]].sort_values("m")
    iv_atm = float(np.interp(1.0 * S0 / F, q1.m, q1.iv))
    iv_70 = float(np.interp(0.7 * S0 / F, q1.m, q1.iv))
    dip = {}
    dip["bs_atm_vol"] = qlite.bs_down_in_put(S0, K, H, T, r, q, iv_atm)
    dip["bs_barrier_vol"] = qlite.bs_down_in_put(S0, K, H, T, r, q, iv_70)
    mc, se = qlite.heston_mc_dip(h, S0, K, H, T, r, q, 250, 2_000_000, 2026, True)
    dip["heston_mc"], dip["heston_mc_se"] = mc, se
    van = qlite.heston_adi_put(h, S0, K, T, r, q, 0.0, 300, 150, 300)
    dop = qlite.heston_adi_put(h, S0, K, T, r, q, H, 300, 150, 300)
    dip["heston_adi"] = van - dop
    dip["heston_lewis_vanilla_put"] = qlite.heston_price(h, S0, K, T, r, q, False)
    dip["heston_adi_vanilla_put"] = van
    mcd, sed = qlite.heston_mc_dip(h, S0, K, H, T, r, q, int(round(252 * T)), 2_000_000, 2026, False)
    dip["heston_mc_daily_close_monitoring"] = mcd
    coupons = {k: 100 * (1 - DF + v / S0) / DF for k, v in dip.items()
               if k in ("bs_atm_vol", "bs_barrier_vol", "heston_mc", "heston_adi", "heston_mc_daily_close_monitoring")}
    out["brc"] = dict(T=T, DF=DF, F=F, r=r, q=q, barrier_pct=70, iv_atm=iv_atm, iv_70=iv_70, dip=dip,
                      fair_coupon_pct=coupons)
    # sensitivity of the fair coupon to the skew parameter rho (others fixed)
    sens = {}
    for rho in (-0.9, -0.7, -0.5, -0.3, 0.0):
        hh = qlite.Heston(h.v0, h.kappa, h.theta, h.sigma, rho)
        m_, _ = qlite.heston_mc_dip(hh, S0, K, H, T, r, q, 250, 400_000, 7, True)
        sens[f"{rho:+.1f}"] = 100 * (1 - DF + m_ / S0) / DF
    out["brc"]["coupon_vs_rho"] = sens

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    Q.to_csv(ROOT / "results" / "smile_fit.csv", index=False)
    (ROOT / "results" / "spx_heston_brc.json").write_text(json.dumps(out, indent=1, default=float))
    exps = sorted(Q.expiry.unique())
    n = len(exps)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(12, 5.6), sharey=False)
    for ax, e in zip(axes.ravel(), exps):
        g = Q[Q.expiry == e].sort_values("m")
        ax.fill_between(g.m, 100 * g.iv_bid, 100 * g.iv_ask, color="0.8", label="bid-ask")
        ax.plot(g.m, 100 * g.iv, "k.", ms=4, label="mid")
        ax.plot(g.m, 100 * g.iv_heston, "r-", lw=1.2, label="Heston")
        ax.set_title(f"{e} (T={g['T'].iloc[0]:.2f})", fontsize=9)
        ax.set_xlabel("K / F", fontsize=8)
    axes.ravel()[0].set_ylabel("implied vol (%)")
    axes.ravel()[0].legend(fontsize=7, frameon=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "spx_smile_fit.png", dpi=150)
    print(json.dumps({k: v for k, v in out.items() if k != "expiries"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
