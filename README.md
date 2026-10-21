# Heston Pricing Library in C++: Three Independent Pricers, Calibrated to SPX, Applied to a Barrier Reverse Convertible

A small C++17 library (`src/qlite`, header-only, Python bindings with pybind11) that prices European and
continuously monitored barrier options under Black-Scholes and Heston in three independent ways, checks each
against the others, calibrates to a real option surface and answers a desk question: **what fair coupon can a
1-year SPX Barrier Reverse Convertible (70 % barrier) pay, and how much does that depend on the model?**

## In short

| Question | Answer |
|---|---|
| Do the three Heston pricers agree? | Lewis (2001) Fourier integral vs Douglas ADI PDE: **2e-4 to 4e-4** on a 100 spot (Feller condition violated). QE Monte Carlo (Andersen 2008) within its standard error. On the real calibration, ADI and Lewis vanilla puts agree to **1.1e-4 relative**. |
| Barrier options? | Black-Scholes down-and-in put: PDE converges at **second order** (error ratio 4.1, 3.9 per grid doubling); Richardson extrapolation matches the Reiner-Rubinstein closed form to **6e-6**. Heston: Monte Carlo with the Brownian-bridge crossing correction matches the ADI in-out parity price within its standard error. |
| Does calibration work? | On synthetic quotes from known parameters it recovers all five to **1e-12** (RMSE 2e-14 vol). On **96 SPX quotes over 8 expiries** (close of 20 Oct 2026, 1 month to 2.2 years): **0.44 vol points** RMSE. |
| Where does Heston fail? | Short maturities: 0.69 vol points RMSE at 1 month, 1.3 points worst, and no short-dated fit inside the bid-ask spread; 45 % inside at 2 years. One parameter set cannot match both the steep short-dated skew and the long-dated one (calibrated vol-of-vol 1.53, Feller violated). |
| Fair coupon of the 1-year BRC (70 % barrier, strike 100 %) | **Heston 8.73 %** (Monte Carlo 8.74 %, ADI 8.73 %). Flat Black-Scholes at the ATM vol (17.8 %): **5.92 %**. Flat Black-Scholes at the vol of the 70 % strike (29.1 %): **11.30 %**. |
| What drives the gap? | The skew. The barrier put is worth 0.89 % of notional at ATM vol, 3.57 % under Heston and 6.01 % at the barrier-strike vol. Under Heston the coupon moves from 7.28 % (rho 0) to 9.01 % (rho -0.9). Daily-close monitoring instead of continuous lowers it by 0.13 points. |

**The product.** Investor's payoff at maturity on notional N: N(1 + c), unless SPX has touched 70 % of its initial
level and closes below 100 %, in which case N S_T / S_0 + N c. This is a zero-coupon bond plus the coupon minus a
down-and-in put, so the fair coupon with no issuer margin is c = (1 - DF + DIP/S_0) / DF. DF and the forward come
from put-call parity on the same snapshot (r = 4.94 %, q = 0.69 % at 1 year), so no rate or dividend is assumed.

## Library

| Header | Content |
|---|---|
| `bs.hpp` | Black-Scholes price, delta, gamma, vega; implied vol (safeguarded Newton); down-and-in put closed form |
| `heston.hpp` | Characteristic function ("little trap", Albrecher et al. 2007); Lewis single-integral pricing with 432-node Gauss-Legendre; Levenberg-Marquardt calibration to implied vols with an unconstrained parametrisation |
| `mc.hpp` | QE scheme with Andersen's martingale correction, antithetic paths, Brownian-bridge barrier crossing (conditional expectation, no indicator noise) |
| `pde.hpp` | 1-D Crank-Nicolson with Rannacher start; 2-D Heston Douglas ADI (in 't Hout & Foulon 2010), sinh-clustered grids, second-order one-sided stencil at v = 0 kept tridiagonal by exact elimination |
| `bindings.cpp` | pybind11 module used by the scripts |

## Verification (`build/test_qlite`, output in `results/verification.txt`)

| Check | Result |
|---|---|
| Lewis integral with the lognormal characteristic function vs Black-Scholes (3 strikes) | 1.3e-13 to 2.7e-13 |
| Delta, gamma, vega vs central differences; implied-vol round trip; put-call parity | 5e-11, 2e-8, 1e-5 (FD error); 2e-16; 7e-15 |
| Black-Scholes barrier: PDE order, Richardson vs closed form; vanilla PDE | order 1.97; 6.1e-6; 4.0e-6 |
| Heston put K = 80, 100: ADI vs Lewis | 4.1e-4, 2.0e-4 |
| Heston put: QE MC (400k paths) vs Lewis | 0.7 and 1.2 standard errors |
| Heston down-and-in put: MC + bridge vs ADI parity | 4.6e-3 (0.3 s.e.) |
| Heston with vol-of-vol to 0: MC down-and-in put vs Black-Scholes closed form | 1.7e-3 (0.1 s.e.) |
| Calibration recovery, 28 synthetic quotes | all parameters to 1e-12 |

## Limitations

* **Heston cannot fit the short end.** A model with jumps (Bates) or a local-stochastic-volatility model is the
  next step; for a 1-year barrier product the fitted 6-month to 2-year smiles matter most, and those fit to
  0.2-0.4 vol points.
* Closing quotes from Yahoo Finance (bid/ask at the close of 20 Oct 2026) are not an executable surface; one
  expiry (June 2028) shows stale quotes (parity residual 6 index points).
* American features, discrete dividends, the issuer's funding spread and hedging costs are ignored: the coupons
  are fair values, not offer terms.
* The ADI solver is serial and uses a uniform time step; there is no AAD; Greeks of exotics come from bumping.

## Run

    sh build.sh                                        # g++ >= 9, Python with pybind11
    ./build/test_qlite                                 # verification suite, about 25 s
    python scripts/fetch_quotes.py                     # new SPX snapshot (yfinance) into data/
    python scripts/calibrate_spx.py data/spx_20261021T1049.csv

Data: the snapshot used here (`data/spx_20261021T1049.csv`, Yahoo Finance close of 20 Oct 2026) is not redistributed; `fetch_quotes.py` takes a new one. Results: `results/spx_heston_brc.json`,
`results/smile_fit.csv`. Figure: `figures/spx_smile_fit.png`.
