// Verification suite: every pricing route is checked against an independent one.
// Build: g++ -O2 -std=c++17 -Isrc tests/test_qlite.cpp -o build/test_qlite && ./build/test_qlite
#include <cstdio>
#include <string>

#include "qlite/bs.hpp"
#include "qlite/heston.hpp"
#include "qlite/mc.hpp"
#include "qlite/pde.hpp"

using namespace qlite;

static int failures = 0;
static void check(const std::string& name, double got, double ref, double tol) {
    const bool ok = std::fabs(got - ref) <= tol;
    if (!ok) ++failures;
    std::printf("%-58s %s  got %.8f  ref %.8f  |diff| %.2e  tol %.1e\n", name.c_str(), ok ? "PASS" : "FAIL", got, ref,
                std::fabs(got - ref), tol);
}

int main() {
    const double S = 100, r = 0.02, q = 0.01, T = 1.0;
    LewisIntegrator L;

    // 1. Lewis integral reproduces Black-Scholes with the lognormal characteristic function
    for (double K : {70.0, 100.0, 140.0}) {
        const double vol = 0.25;
        auto cf = [&](cd u) { return std::exp(-0.5 * vol * vol * T * (u * u + cd(0, 1) * u)); };
        check("Lewis(BS cf) vs BS call K=" + std::to_string(int(K)), L.call(cf, S, K, T, r, q),
              bs_price(S, K, T, r, q, vol, true), 1e-10);
    }

    // 2. Greeks vs central finite differences; implied vol round trip; put-call parity
    {
        const double K = 95, vol = 0.3, h = 1e-3;
        check("BS delta vs FD", bs_delta(S, K, T, r, q, vol, true),
              (bs_price(S + h, K, T, r, q, vol, true) - bs_price(S - h, K, T, r, q, vol, true)) / (2 * h), 1e-8);
        check("BS gamma vs FD", bs_gamma(S, K, T, r, q, vol),
              (bs_price(S + h, K, T, r, q, vol, true) - 2 * bs_price(S, K, T, r, q, vol, true) +
               bs_price(S - h, K, T, r, q, vol, true)) / (h * h), 1e-5);
        check("BS vega vs FD", bs_vega(S, K, T, r, q, vol),
              (bs_price(S, K, T, r, q, vol + h, true) - bs_price(S, K, T, r, q, vol - h, true)) / (2 * h), 1e-4);
        check("implied vol round trip", implied_vol(bs_price(S, K, T, r, q, 0.37, false), S, K, T, r, q, false), 0.37, 1e-10);
        check("put-call parity", bs_price(S, K, T, r, q, vol, true) - bs_price(S, K, T, r, q, vol, false),
              S * std::exp(-q * T) - K * std::exp(-r * T), 1e-12);
    }

    // 3. Black-Scholes barrier: closed form vs PDE (in-out parity) and in + out = vanilla
    {
        const double K = 100, H = 70, vol = 0.25;
        const double cf = bs_down_in_put(S, K, H, T, r, q, vol);
        const double van = bs_price(S, K, T, r, q, vol, false);
        double d[3];
        int k = 0;
        for (int n : {200, 400, 800}) d[k++] = van - bs_pde(S, K, T, r, q, vol, PdeProduct::DownOutPut, H, n, n);
        const double e1 = std::fabs(d[0] - cf), e2 = std::fabs(d[1] - cf), e3 = std::fabs(d[2] - cf);
        std::printf("    PDE error 200 / 400 / 800 nodes: %.2e / %.2e / %.2e (ratios %.1f, %.1f)\n", e1, e2, e3, e1 / e2, e2 / e3);
        check("BS down-in put: observed PDE order (400 -> 800)", std::log2(e2 / e3), 2.0, 0.2);
        check("BS down-in put: Richardson(400, 800) vs closed form", (4 * d[2] - d[1]) / 3, cf, 2e-5);
        const double p4 = bs_pde(S, K, T, r, q, vol, PdeProduct::Put, H, 400, 400), p8 = bs_pde(S, K, T, r, q, vol, PdeProduct::Put, H, 800, 800);
        check("BS vanilla put: Richardson(400, 800) vs closed form", (4 * p8 - p4) / 3, van, 2e-5);
        // K < H branch: knocked-in surely before payoff matters -> equals vanilla
        check("BS down-in put K<H equals vanilla when S<=H", bs_down_in_put(60, 50, 65, T, r, q, vol),
              bs_price(60, 50, T, r, q, vol, false), 1e-12);
    }

    // 4. Heston: Lewis vs ADI PDE vs QE Monte Carlo (European put, ATM and OTM)
    Heston h{0.04, 1.5, 0.04, 0.5, -0.7};
    for (double K : {80.0, 100.0}) {
        const double ref = heston_price(h, S, K, T, r, q, false, L);
        const double pde = heston_adi(h, S, K, T, r, q, PdeProduct::Put, 0, 200, 100, 200);
        check("Heston put K=" + std::to_string(int(K)) + ": ADI (200x100x200) vs Lewis", pde, ref, 5e-3);
        const auto mc = heston_mc(h, S, K, T, r, q, Payoff::Put, {0, true}, 100, 400000, 7);
        check("Heston put K=" + std::to_string(int(K)) + ": QE MC (4e5 paths) vs Lewis, tol 3 s.e.", mc.price, ref,
              3 * mc.stderr_);
    }
    // Feller-violating case is the hard one for MC/PDE near v = 0
    std::printf("    Feller condition (2 kappa theta >= sigma^2): %s\n", h.feller() ? "satisfied" : "violated");

    // 5. Heston down-and-in put: MC (Brownian-bridge) vs ADI (vanilla - down-and-out)
    {
        const double K = 100, H = 70;
        const double van = heston_adi(h, S, K, T, r, q, PdeProduct::Put, 0, 200, 100, 200);
        const double dop = heston_adi(h, S, K, T, r, q, PdeProduct::DownOutPut, H, 200, 100, 200);
        const auto mc = heston_mc(h, S, K, T, r, q, Payoff::DownInPut, {H, true}, 200, 400000, 11);
        check("Heston down-in put: QE MC + bridge vs ADI parity", mc.price, van - dop, 3 * mc.stderr_ + 5e-3);
        const auto mcd = heston_mc(h, S, K, T, r, q, Payoff::DownInPut, {H, false}, 200, 400000, 11);
        std::printf("    discrete monitoring only (200 dates): %.4f vs continuous %.4f\n", mcd.price, mc.price);
        // BS limit: tiny vol-of-vol -> BS closed form
        Heston bsl{0.0625, 5.0, 0.0625, 1e-3, 0.0};
        const auto m2 = heston_mc(bsl, S, K, T, r, q, Payoff::DownInPut, {H, true}, 100, 400000, 13);
        check("Heston(sigma->0) DIP MC vs BS closed form, tol 3 s.e.", m2.price, bs_down_in_put(S, K, H, T, r, q, 0.25),
              3 * m2.stderr_ + 1e-3);
    }

    // 6. Calibration recovers known parameters from a synthetic surface
    {
        std::vector<Quote> qs;
        for (double t : {0.25, 0.5, 1.0, 2.0})
            for (double m : {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3}) {
                const double K = S * m;
                const bool call = K >= S;
                const double p = heston_price(h, S, K, t, r, q, call, L);
                qs.push_back({K, t, implied_vol(p, S, K, t, r, q, call), 1.0, call});
            }
        const auto res = calibrate(qs, S, r, q, Heston{0.06, 0.8, 0.06, 0.3, -0.3});
        check("calibration: rmse (vol)", res.rmse_vol, 0.0, 1e-6);
        check("calibration: v0", res.h.v0, h.v0, 1e-4);
        check("calibration: kappa", res.h.kappa, h.kappa, 2e-2);
        check("calibration: theta", res.h.theta, h.theta, 1e-3);
        check("calibration: sigma", res.h.sigma, h.sigma, 1e-2);
        check("calibration: rho", res.h.rho, h.rho, 1e-3);
    }

    std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
