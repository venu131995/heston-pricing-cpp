// Heston (1993) model: characteristic function in the "little trap" form of Albrecher et al. (2007), European prices
// with the Lewis (2001) single-integral formula, and calibration to implied volatilities (Levenberg-Marquardt).
#pragma once
#include <array>
#include <complex>
#include <vector>

#include "bs.hpp"

namespace qlite {

using cd = std::complex<double>;

struct Heston {
    double v0, kappa, theta, sigma, rho;
    bool feller() const { return 2 * kappa * theta >= sigma * sigma; }
};

// E[exp(i u ln(S_T / F_T))] under the risk-neutral measure (F_T the forward).
inline cd heston_cf(const Heston& h, cd u, double T) {
    const cd i(0.0, 1.0);
    const double s2 = h.sigma * h.sigma;
    const cd b = h.kappa - h.rho * h.sigma * i * u;
    const cd d = std::sqrt(b * b + s2 * (i * u + u * u));
    const cd g = (b - d) / (b + d);
    const cd e = std::exp(-d * T);
    const cd C = h.kappa * h.theta / s2 * ((b - d) * T - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
    const cd D = (b - d) / s2 * (1.0 - e) / (1.0 - g * e);
    return std::exp(C + D * h.v0);
}

// Gauss-Legendre nodes on [-1, 1] (Golub-Welsch would be overkill; Newton on P_n).
inline void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    for (int k = 0; k < (n + 1) / 2; ++k) {
        double z = std::cos(PI * (k + 0.75) / (n + 0.5)), pp = 0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1, p2 = 0;
            for (int j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
            }
            pp = n * (z * p1 - p2) / (z * z - 1);
            const double dz = p1 / pp;
            z -= dz;
            if (std::fabs(dz) < 1e-15) break;
        }
        x[k] = -z;
        x[n - 1 - k] = z;
        w[k] = w[n - 1 - k] = 2.0 / ((1 - z * z) * pp * pp);
    }
}

struct LewisIntegrator {
    std::vector<double> nodes, weights;
    explicit LewisIntegrator(int per_panel = 48) {
        std::vector<double> x, w;
        gauss_legendre(per_panel, x, w);
        const double edges[] = {0, 2, 5, 10, 20, 40, 80, 160, 320, 640};
        for (int p = 0; p + 1 < 10; ++p) {
            const double a = edges[p], b = edges[p + 1];
            for (int k = 0; k < per_panel; ++k) {
                nodes.push_back(0.5 * (b - a) * x[k] + 0.5 * (a + b));
                weights.push_back(0.5 * (b - a) * w[k]);
            }
        }
    }
    // Call price: S e^{-qT} - sqrt(F K) e^{-rT} / pi * int_0^inf Re[e^{i u x} psi(u - i/2)] / (u^2 + 1/4) du, x = ln(F/K).
    template <class CF>
    double call(CF cf, double S, double K, double T, double r, double q) const {
        const double F = S * std::exp((r - q) * T), x = std::log(F / K);
        const cd i(0.0, 1.0);
        double acc = 0;
        for (size_t k = 0; k < nodes.size(); ++k) {
            const double u = nodes[k];
            acc += weights[k] * std::real(std::exp(i * u * x) * cf(cd(u, -0.5))) / (u * u + 0.25);
        }
        return S * std::exp(-q * T) - std::sqrt(F * K) * std::exp(-r * T) / PI * acc;
    }
};

inline double heston_price(const Heston& h, double S, double K, double T, double r, double q, bool call,
                           const LewisIntegrator& L) {
    const double c = L.call([&](cd u) { return heston_cf(h, u, T); }, S, K, T, r, q);
    return call ? c : c - S * std::exp(-q * T) + K * std::exp(-r * T);  // put-call parity
}

struct Quote {
    double K, T, iv, weight;  // implied vol of the market mid; weight in the objective
    bool call;
};

// Unconstrained parametrisation: v0, kappa, theta, sigma = exp(.), rho = tanh(.)
inline Heston from_x(const std::array<double, 5>& x) {
    return Heston{std::exp(x[0]), std::exp(x[1]), std::exp(x[2]), std::exp(x[3]), std::tanh(x[4])};
}
inline std::array<double, 5> to_x(const Heston& h) {
    return {std::log(h.v0), std::log(h.kappa), std::log(h.theta), std::log(h.sigma), std::atanh(h.rho)};
}

struct CalibResult {
    Heston h;
    double rmse_vol;  // root-mean-square implied-vol error over the quotes (absolute, e.g. 0.01 = 1 vol point)
    int iterations;
};

// Residual = (model implied vol - market vol) * sqrt(weight). Model vols from model prices via implied_vol().
inline std::vector<double> residuals(const Heston& h, const std::vector<Quote>& qs, double S, double r, double q,
                                     const LewisIntegrator& L) {
    std::vector<double> res(qs.size());
    for (size_t k = 0; k < qs.size(); ++k) {
        const auto& Q = qs[k];
        const double p = heston_price(h, S, Q.K, Q.T, r, q, Q.call, L);
        double iv = implied_vol(p, S, Q.K, Q.T, r, q, Q.call);
        if (!std::isfinite(iv)) iv = 0.0;  // price outside bounds: strongly penalised
        res[k] = (iv - Q.iv) * std::sqrt(Q.weight);
    }
    return res;
}

// Solve the small dense system A x = b (Gaussian elimination with partial pivoting).
inline std::vector<double> solve_dense(std::vector<std::vector<double>> A, std::vector<double> b) {
    const size_t n = b.size();
    for (size_t c = 0; c < n; ++c) {
        size_t p = c;
        for (size_t r = c + 1; r < n; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[p][c])) p = r;
        std::swap(A[c], A[p]);
        std::swap(b[c], b[p]);
        for (size_t r = c + 1; r < n; ++r) {
            const double f = A[r][c] / A[c][c];
            for (size_t k = c; k < n; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    std::vector<double> x(n);
    for (size_t c = n; c-- > 0;) {
        double s = b[c];
        for (size_t k = c + 1; k < n; ++k) s -= A[c][k] * x[k];
        x[c] = s / A[c][c];
    }
    return x;
}

inline CalibResult calibrate(const std::vector<Quote>& qs, double S, double r, double q, Heston start,
                             int max_iter = 1000) {
    LewisIntegrator L;
    auto x = to_x(start);
    auto res = residuals(from_x(x), qs, S, r, q, L);
    auto sse = [](const std::vector<double>& v) { double s = 0; for (double e : v) s += e * e; return s; };
    double f = sse(res), lambda = 1e-2;
    int it = 0;
    for (; it < max_iter; ++it) {
        // forward-difference Jacobian in x
        std::vector<std::vector<double>> J(res.size(), std::vector<double>(5));
        for (int j = 0; j < 5; ++j) {
            auto xp = x;
            const double h = 1e-5 * std::max(1.0, std::fabs(x[j]));
            xp[j] += h;
            const auto rp = residuals(from_x(xp), qs, S, r, q, L);
            for (size_t k = 0; k < res.size(); ++k) J[k][j] = (rp[k] - res[k]) / h;
        }
        std::vector<std::vector<double>> A(5, std::vector<double>(5, 0.0));
        std::vector<double> g(5, 0.0);
        for (size_t k = 0; k < res.size(); ++k)
            for (int a = 0; a < 5; ++a) {
                g[a] -= J[k][a] * res[k];
                for (int b = 0; b < 5; ++b) A[a][b] += J[k][a] * J[k][b];
            }
        bool improved = false;
        for (int tries = 0; tries < 20; ++tries) {
            auto Ad = A;
            for (int a = 0; a < 5; ++a) Ad[a][a] *= (1.0 + lambda);
            const auto dx = solve_dense(Ad, g);
            auto xn = x;
            for (int a = 0; a < 5; ++a) xn[a] += dx[a];
            xn[4] = std::max(-6.0, std::min(6.0, xn[4]));
            const auto rn = residuals(from_x(xn), qs, S, r, q, L);
            const double fn = sse(rn);
            if (fn < f) {
                const double rel = (f - fn) / f;
                x = xn; res = rn; f = fn; lambda = std::max(lambda / 3, 1e-9); improved = true;
                if (rel < 1e-10) it = max_iter;
                break;
            }
            lambda *= 4;
        }
        if (!improved) break;
    }
    double wsum = 0;
    for (auto& Q : qs) wsum += Q.weight;
    return CalibResult{from_x(x), std::sqrt(f / wsum), it};
}

}  // namespace qlite
