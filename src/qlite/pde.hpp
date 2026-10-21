// Finite-difference solvers.
//  * bs_pde: Black-Scholes in S, Crank-Nicolson with Rannacher start (two implicit-Euler half steps), non-uniform grid
//    clustered at the strike; European puts/calls and down-and-out puts (Dirichlet zero at the barrier).
//  * heston_adi: Heston in (S, v), Douglas ADI scheme (theta = 1/2), mixed derivative explicit, non-uniform grids
//    as in in 't Hout & Foulon (2010, Int. J. Numer. Anal. Model. 7(2)); European puts and down-and-out puts.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "heston.hpp"

namespace qlite {

// sinh-stretched grid on [a, b] clustered around c (density parameter d, smaller = more clustering).
inline std::vector<double> sinh_grid(double a, double b, double c, double d, int n) {
    const double lo = std::asinh((a - c) / d), hi = std::asinh((b - c) / d);
    std::vector<double> x(n + 1);
    for (int i = 0; i <= n; ++i) x[i] = c + d * std::sinh(lo + (hi - lo) * i / n);
    x[0] = a;
    x[n] = b;
    return x;
}

// Thomas algorithm: sub a, diag b, super c, rhs d (all length n; a[0], c[n-1] unused).
inline void thomas(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, std::vector<double>& d) {
    const size_t n = d.size();
    for (size_t i = 1; i < n; ++i) {
        const double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    d[n - 1] /= b[n - 1];
    for (size_t i = n - 1; i-- > 0;) d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
}

// Central first and second derivative weights on a non-uniform grid at interior node i.
struct Stencil { double m, z, p; };
inline Stencil d1w(const std::vector<double>& x, int i) {
    const double hm = x[i] - x[i - 1], hp = x[i + 1] - x[i];
    return {-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))};
}
inline Stencil d2w(const std::vector<double>& x, int i) {
    const double hm = x[i] - x[i - 1], hp = x[i + 1] - x[i];
    return {2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp))};
}

enum class PdeProduct { Put, Call, DownOutPut };

// ---------------------------------------------------------------- Black-Scholes, 1-D
inline double bs_pde(double S0, double K, double T, double r, double q, double vol, PdeProduct prod, double H,
                     int ns, int nt) {
    const double smin = prod == PdeProduct::DownOutPut ? H : 0.0, smax = 8 * K;
    auto s = sinh_grid(smin, smax, K, 0.1 * K, ns);
    std::vector<double> V(ns + 1);
    for (int i = 0; i <= ns; ++i) V[i] = prod == PdeProduct::Call ? std::max(s[i] - K, 0.0) : std::max(K - s[i], 0.0);
    if (prod == PdeProduct::DownOutPut) V[0] = 0.0;
    const double dt = T / nt;
    // operator L V = 0.5 vol^2 S^2 V'' + (r - q) S V' - r V on interior nodes
    std::vector<Stencil> L(ns + 1);
    for (int i = 1; i < ns; ++i) {
        const auto a1 = d1w(s, i), a2 = d2w(s, i);
        const double dif = 0.5 * vol * vol * s[i] * s[i], adv = (r - q) * s[i];
        L[i] = {dif * a2.m + adv * a1.m, dif * a2.z + adv * a1.z - r, dif * a2.p + adv * a1.p};
    }
    auto bc = [&](double tau, double& lo, double& hi) {
        if (prod == PdeProduct::Call) { lo = 0.0; hi = smax * std::exp(-q * tau) - K * std::exp(-r * tau); }
        else if (prod == PdeProduct::Put) { lo = K * std::exp(-r * tau); hi = 0.0; }
        else { lo = 0.0; hi = 0.0; }
    };
    double tau = 0.0;
    auto step = [&](double h, double th) {  // theta-scheme step of length h
        std::vector<double> a(ns - 1), b(ns - 1), c(ns - 1), d(ns - 1);
        double lo, hi;
        bc(tau + h, lo, hi);
        for (int i = 1; i < ns; ++i) {
            const auto& l = L[i];
            const double ex = V[i] + (1 - th) * h * (l.m * V[i - 1] + l.z * V[i] + l.p * V[i + 1]);
            a[i - 1] = -th * h * l.m; b[i - 1] = 1 - th * h * l.z; c[i - 1] = -th * h * l.p; d[i - 1] = ex;
        }
        d[0] -= a[0] * lo;
        d[ns - 2] -= c[ns - 2] * hi;
        thomas(a, b, c, d);
        for (int i = 1; i < ns; ++i) V[i] = d[i - 1];
        V[0] = lo;
        V[ns] = hi;
        tau += h;
    };
    step(0.5 * dt, 1.0);
    step(0.5 * dt, 1.0);
    step(0.5 * dt, 1.0);
    step(0.5 * dt, 1.0);
    for (int n = 2; n < nt; ++n) step(dt, 0.5);
    // cubic Lagrange interpolation at S0
    int j = int(std::upper_bound(s.begin(), s.end(), S0) - s.begin()) - 2;
    j = std::max(0, std::min(j, ns - 3));
    double out = 0;
    for (int a = 0; a < 4; ++a) {
        double w = 1;
        for (int b = 0; b < 4; ++b) if (b != a) w *= (S0 - s[j + b]) / (s[j + a] - s[j + b]);
        out += w * V[j + a];
    }
    return out;
}

// ---------------------------------------------------------------- Heston, 2-D Douglas ADI
inline double heston_adi(const Heston& h, double S0, double K, double T, double r, double q, PdeProduct prod,
                         double H, int ns, int nv, int nt) {
    const double smin = prod == PdeProduct::DownOutPut ? H : 0.0, smax = 8 * K, vmax = 5.0;
    const auto s = sinh_grid(smin, smax, K, K / 5, ns);
    const auto v = sinh_grid(0.0, vmax, 0.0, vmax / 500, nv);
    const int NS = ns + 1, NV = nv + 1;
    auto id = [&](int i, int j) { return j * NS + i; };
    std::vector<double> U(NS * NV);
    auto payoff = [&](double S) { return prod == PdeProduct::Call ? std::max(S - K, 0.0) : std::max(K - S, 0.0); };
    for (int j = 0; j < NV; ++j)
        for (int i = 0; i < NS; ++i) U[id(i, j)] = payoff(s[i]);
    auto apply_bc = [&](std::vector<double>& W, double tau) {
        for (int j = 0; j < NV; ++j) {
            if (prod == PdeProduct::Call) { W[id(0, j)] = 0.0; W[id(ns, j)] = s[ns] * std::exp(-q * tau) - K * std::exp(-r * tau); }
            else if (prod == PdeProduct::Put) { W[id(0, j)] = K * std::exp(-r * tau); W[id(ns, j)] = 0.0; }
            else { W[id(0, j)] = 0.0; W[id(ns, j)] = 0.0; }
        }
        // v = vmax: dV/dv = 0 (Neumann, first order)
        for (int i = 1; i < ns; ++i) W[id(i, nv)] = W[id(i, nv - 1)];
    };
    // F0: mixed derivative, F1: S-direction, F2: v-direction (incl. the -rV term split evenly)
    auto F = [&](const std::vector<double>& W, std::vector<double>& f0, std::vector<double>& f1, std::vector<double>& f2) {
        std::fill(f0.begin(), f0.end(), 0.0);
        std::fill(f1.begin(), f1.end(), 0.0);
        std::fill(f2.begin(), f2.end(), 0.0);
        for (int j = 0; j < nv; ++j)
            for (int i = 1; i < ns; ++i) {
                const auto a1 = d1w(s, i), a2 = d2w(s, i);
                const double Vs = a1.m * W[id(i - 1, j)] + a1.z * W[id(i, j)] + a1.p * W[id(i + 1, j)];
                const double Vss = a2.m * W[id(i - 1, j)] + a2.z * W[id(i, j)] + a2.p * W[id(i + 1, j)];
                f1[id(i, j)] = 0.5 * s[i] * s[i] * v[j] * Vss + (r - q) * s[i] * Vs - 0.5 * r * W[id(i, j)];
                double Vv, Vvv = 0.0;
                if (j == 0) {  // v = 0: the PDE degenerates; one-sided first derivative (inflow from v > 0)
                    const double h1 = v[1] - v[0], h2 = v[2] - v[0];
                    Vv = (-(h1 + h2) / (h1 * h2)) * W[id(i, 0)] + (h2 / (h1 * (h2 - h1))) * W[id(i, 1)] -
                         (h1 / (h2 * (h2 - h1))) * W[id(i, 2)];
                } else {
                    const auto b1 = d1w(v, j), b2 = d2w(v, j);
                    Vv = b1.m * W[id(i, j - 1)] + b1.z * W[id(i, j)] + b1.p * W[id(i, j + 1)];
                    Vvv = b2.m * W[id(i, j - 1)] + b2.z * W[id(i, j)] + b2.p * W[id(i, j + 1)];
                    double Vsv = 0;
                    for (int a = -1; a <= 1; ++a)
                        for (int b = -1; b <= 1; ++b) {
                            const double wa = a == -1 ? a1.m : (a == 0 ? a1.z : a1.p);
                            const double wb = b == -1 ? b1.m : (b == 0 ? b1.z : b1.p);
                            Vsv += wa * wb * W[id(i + a, j + b)];
                        }
                    f0[id(i, j)] = h.rho * h.sigma * s[i] * v[j] * Vsv;
                }
                f2[id(i, j)] = 0.5 * h.sigma * h.sigma * v[j] * Vvv + h.kappa * (h.theta - v[j]) * Vv - 0.5 * r * W[id(i, j)];
            }
    };
    const double dt = T / nt, th = 0.5;
    std::vector<double> f0(NS * NV), f1(NS * NV), f2(NS * NV), Y0(NS * NV), Y1(NS * NV), Y2(NS * NV);
    std::vector<double> g0(NS * NV), g1(NS * NV), g2(NS * NV);
    double tau = 0;
    for (int n = 0; n < nt; ++n) {
        F(U, f0, f1, f2);
        for (size_t k = 0; k < U.size(); ++k) Y0[k] = U[k] + dt * (f0[k] + f1[k] + f2[k]);
        apply_bc(Y0, tau + dt);
        // S-direction implicit: (I - th dt A1) Y1 = Y0 - th dt A1 U
        Y1 = Y0;
        for (int j = 0; j < nv; ++j) {
            std::vector<double> a(ns - 1), b(ns - 1), c(ns - 1), d(ns - 1);
            for (int i = 1; i < ns; ++i) {
                const auto a1 = d1w(s, i), a2 = d2w(s, i);
                const double dif = 0.5 * s[i] * s[i] * v[j], adv = (r - q) * s[i];
                const double lm = dif * a2.m + adv * a1.m, lz = dif * a2.z + adv * a1.z - 0.5 * r,
                             lp = dif * a2.p + adv * a1.p;
                a[i - 1] = -th * dt * lm; b[i - 1] = 1 - th * dt * lz; c[i - 1] = -th * dt * lp;
                d[i - 1] = Y0[id(i, j)] - th * dt * f1[id(i, j)];
            }
            d[0] -= a[0] * Y0[id(0, j)];
            d[ns - 2] -= c[ns - 2] * Y0[id(ns, j)];
            thomas(a, b, c, d);
            for (int i = 1; i < ns; ++i) Y1[id(i, j)] = d[i - 1];
        }
        apply_bc(Y1, tau + dt);
        // v-direction implicit: (I - th dt A2) Y2 = Y1 - th dt A2 U, over j = 0..nv-1 with Neumann at vmax
        Y2 = Y1;
        for (int i = 1; i < ns; ++i) {
            std::vector<double> a(nv), b(nv), c(nv), d(nv);
            double e02 = 0.0;
            for (int j = 0; j < nv; ++j) {
                double lm = 0, lz, lp = 0, lpp = 0;
                if (j == 0) {
                    const double h1 = v[1] - v[0], h2 = v[2] - v[0], kt = h.kappa * h.theta;
                    lz = kt * (-(h1 + h2) / (h1 * h2)) - 0.5 * r;
                    lp = kt * (h2 / (h1 * (h2 - h1)));
                    lpp = -kt * (h1 / (h2 * (h2 - h1)));
                } else {
                    const auto b1 = d1w(v, j), b2 = d2w(v, j);
                    const double dif = 0.5 * h.sigma * h.sigma * v[j], adv = h.kappa * (h.theta - v[j]);
                    lm = dif * b2.m + adv * b1.m; lz = dif * b2.z + adv * b1.z - 0.5 * r; lp = dif * b2.p + adv * b1.p;
                }
                a[j] = -th * dt * lm; b[j] = 1 - th * dt * lz; c[j] = -th * dt * lp;
                d[j] = Y1[id(i, j)] - th * dt * f2[id(i, j)];
                if (j == 0) e02 = -th * dt * lpp;  // entry (0, 2) from the one-sided stencil
            }
            // remove the (0, 2) entry with row 1 so the system stays tridiagonal (exact elimination)
            const double m = e02 / c[1];
            b[0] -= m * a[1];
            c[0] -= m * b[1];
            d[0] -= m * d[1];
            // Neumann at j = nv: U(nv) = U(nv-1) -> fold the last super-diagonal into the diagonal of row nv-1
            b[nv - 1] += c[nv - 1];
            c[nv - 1] = 0.0;
            thomas(a, b, c, d);
            for (int j = 0; j < nv; ++j) Y2[id(i, j)] = d[j];
        }
        apply_bc(Y2, tau + dt);
        U.swap(Y2);
        tau += dt;
    }
    // bicubic-free: linear in v, cubic in S
    int jv = int(std::upper_bound(v.begin(), v.end(), h.v0) - v.begin()) - 1;
    jv = std::max(0, std::min(jv, nv - 1));
    auto interpS = [&](int j) {
        int k = int(std::upper_bound(s.begin(), s.end(), S0) - s.begin()) - 2;
        k = std::max(0, std::min(k, ns - 3));
        double out = 0;
        for (int a = 0; a < 4; ++a) {
            double w = 1;
            for (int b = 0; b < 4; ++b) if (b != a) w *= (S0 - s[k + b]) / (s[k + a] - s[k + b]);
            out += w * U[id(k + a, j)];
        }
        return out;
    };
    const double w = (h.v0 - v[jv]) / (v[jv + 1] - v[jv]);
    return (1 - w) * interpS(jv) + w * interpS(jv + 1);
}

}  // namespace qlite
