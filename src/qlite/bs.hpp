// Black-Scholes prices, Greeks, implied volatility and continuously monitored barrier options.
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qlite {

constexpr double PI = 3.14159265358979323846;

inline double ncdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
inline double npdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI); }

struct Market {
    double S, r, q;  // spot, continuously compounded rate and dividend yield
};

inline double bs_price(double S, double K, double T, double r, double q, double vol, bool call) {
    if (T <= 0.0) return call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    const double sd = vol * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q) * T) / sd + 0.5 * sd;
    const double d2 = d1 - sd;
    if (call) return S * std::exp(-q * T) * ncdf(d1) - K * std::exp(-r * T) * ncdf(d2);
    return K * std::exp(-r * T) * ncdf(-d2) - S * std::exp(-q * T) * ncdf(-d1);
}

inline double bs_vega(double S, double K, double T, double r, double q, double vol) {
    const double sd = vol * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q) * T) / sd + 0.5 * sd;
    return S * std::exp(-q * T) * npdf(d1) * std::sqrt(T);
}

inline double bs_delta(double S, double K, double T, double r, double q, double vol, bool call) {
    const double sd = vol * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q) * T) / sd + 0.5 * sd;
    return call ? std::exp(-q * T) * ncdf(d1) : -std::exp(-q * T) * ncdf(-d1);
}

inline double bs_gamma(double S, double K, double T, double r, double q, double vol) {
    const double sd = vol * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q) * T) / sd + 0.5 * sd;
    return std::exp(-q * T) * npdf(d1) / (S * sd);
}

// Implied volatility by safeguarded Newton (falls back to bisection); NaN if the price is outside the no-arbitrage bounds.
inline double implied_vol(double price, double S, double K, double T, double r, double q, bool call) {
    const double df = std::exp(-r * T), dq = std::exp(-q * T);
    const double lo_b = call ? std::max(S * dq - K * df, 0.0) : std::max(K * df - S * dq, 0.0);
    const double hi_b = call ? S * dq : K * df;
    if (!(price > lo_b && price < hi_b)) return std::numeric_limits<double>::quiet_NaN();
    double lo = 1e-6, hi = 5.0;
    double v = std::sqrt(2.0 * PI / T) * price / S;  // Brenner-Subrahmanyam start
    if (!(v > lo && v < hi)) v = 0.3;
    for (int it = 0; it < 100; ++it) {
        const double f = bs_price(S, K, T, r, q, v, call) - price;
        if (std::fabs(f) < 1e-12 * std::max(1.0, price)) return v;
        if (f > 0) hi = v; else lo = v;
        const double vg = bs_vega(S, K, T, r, q, v);
        double nv = (vg > 1e-12) ? v - f / vg : 0.5 * (lo + hi);
        if (!(nv > lo && nv < hi)) nv = 0.5 * (lo + hi);
        if (std::fabs(nv - v) < 1e-14) return nv;
        v = nv;
    }
    return v;
}

// Down-and-in put, continuously monitored barrier H < S, no rebate (Reiner & Rubinstein 1991; Haug, The Complete
// Guide to Option Pricing Formulas, 2nd ed., sec. 4.17.1, with eta = 1, phi = -1).
inline double bs_down_in_put(double S, double K, double H, double T, double r, double q, double vol) {
    if (S <= H) return bs_price(S, K, T, r, q, vol, false);
    const double b = r - q, sd = vol * std::sqrt(T), mu = (b - 0.5 * vol * vol) / (vol * vol);
    const double phi = -1.0, eta = 1.0;
    const double dq = std::exp(-q * T), df = std::exp(-r * T);
    const double x = std::log(S / K) / sd + (1 + mu) * sd;
    const double x1 = std::log(S / H) / sd + (1 + mu) * sd;
    const double y = std::log(H * H / (S * K)) / sd + (1 + mu) * sd;
    const double y1 = std::log(H / S) / sd + (1 + mu) * sd;
    const double hs = H / S;
    const double A = phi * S * dq * ncdf(phi * x) - phi * K * df * ncdf(phi * x - phi * sd);
    const double B = phi * S * dq * ncdf(phi * x1) - phi * K * df * ncdf(phi * x1 - phi * sd);
    const double C = phi * S * dq * std::pow(hs, 2 * (mu + 1)) * ncdf(eta * y) -
                     phi * K * df * std::pow(hs, 2 * mu) * ncdf(eta * y - eta * sd);
    const double D = phi * S * dq * std::pow(hs, 2 * (mu + 1)) * ncdf(eta * y1) -
                     phi * K * df * std::pow(hs, 2 * mu) * ncdf(eta * y1 - eta * sd);
    return (K > H) ? B - C + D : A;
}

}  // namespace qlite
