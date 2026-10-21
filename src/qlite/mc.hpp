// Monte Carlo under Heston with the Quadratic-Exponential (QE) scheme of Andersen (2008, J. Comput. Finance 11(3)),
// including his martingale correction, and continuously monitored down barriers handled with the Brownian-bridge
// crossing probability on each step (the knock-in indicator is replaced by its conditional expectation).
#pragma once
#include <random>
#include <vector>

#include "heston.hpp"

namespace qlite {

struct McResult {
    double price, stderr_;
};

struct BarrierSpec {
    double H;            // down barrier (absolute level)
    bool continuous;     // true: Brownian-bridge correction; false: monitored only at the time grid
};

// Product payoffs are written in terms of the terminal spot and the probability that the barrier was hit.
enum class Payoff { Call, Put, DownInPut, DownOutPut };

inline McResult heston_mc(const Heston& h, double S0, double K, double T, double r, double q, Payoff pay,
                          BarrierSpec bar, int steps, long paths, unsigned long seed, bool antithetic = true) {
    const double dt = T / steps, ek = std::exp(-h.kappa * dt), s2 = h.sigma * h.sigma;
    const double c1 = s2 * ek * (1 - ek) / h.kappa, c2 = h.theta * s2 * (1 - ek) * (1 - ek) / (2 * h.kappa);
    const double g1 = 0.5, g2 = 0.5;
    const double K1 = g1 * dt * (h.kappa * h.rho / h.sigma - 0.5) - h.rho / h.sigma;
    const double K2 = g2 * dt * (h.kappa * h.rho / h.sigma - 0.5) + h.rho / h.sigma;
    const double K3 = g1 * dt * (1 - h.rho * h.rho), K4 = g2 * dt * (1 - h.rho * h.rho);
    const double A = K2 + 0.5 * K4;
    const double psi_c = 1.5;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const long n_outer = antithetic ? paths / 2 : paths;
    double sum = 0, sum2 = 0;
    std::vector<double> z1(steps), z2(steps), uu(steps);
    const double df = std::exp(-r * T);
    for (long p = 0; p < n_outer; ++p) {
        for (int k = 0; k < steps; ++k) { z1[k] = N(rng); z2[k] = N(rng); uu[k] = U(rng); }
        double pay_pair = 0;
        for (int a = 0; a < (antithetic ? 2 : 1); ++a) {
            const double sgn = a == 0 ? 1.0 : -1.0;
            double v = h.v0, x = std::log(S0), surv = 1.0;
            bool hit = false;
            for (int k = 0; k < steps; ++k) {
                const double m = h.theta + (v - h.theta) * ek;
                const double s2v = v * c1 + c2;
                const double psi = s2v / (m * m);
                double vn, K0s;
                const double u = a == 0 ? uu[k] : 1.0 - uu[k];
                if (psi <= psi_c) {
                    const double b2 = 2 / psi - 1 + std::sqrt(2 / psi) * std::sqrt(2 / psi - 1);
                    const double aa = m / (1 + b2), b = std::sqrt(b2);
                    vn = aa * (b + sgn * z1[k]) * (b + sgn * z1[k]);
                    // martingale correction (Andersen 2008, eq. 33-34)
                    K0s = -A * b2 * aa / (1 - 2 * A * aa) + 0.5 * std::log(1 - 2 * A * aa) - (K1 + 0.5 * K3) * v;
                } else {
                    const double pp = (psi - 1) / (psi + 1), beta = (1 - pp) / m;
                    vn = (u <= pp) ? 0.0 : std::log((1 - pp) / (1 - u)) / beta;
                    K0s = -std::log(pp + beta * (1 - pp) / (beta - A)) - (K1 + 0.5 * K3) * v;
                }
                const double xn = x + (r - q) * dt + K0s + K1 * v + K2 * vn +
                                  std::sqrt(std::max(K3 * v + K4 * vn, 0.0)) * sgn * z2[k];
                if (pay == Payoff::DownInPut || pay == Payoff::DownOutPut) {
                    const double lh = std::log(bar.H);
                    if (xn <= lh || x <= lh) {
                        hit = true;
                    } else if (bar.continuous) {
                        const double var = std::max(0.5 * (v + vn), 1e-12) * dt;
                        surv *= 1.0 - std::exp(-2.0 * (x - lh) * (xn - lh) / var);
                    }
                }
                x = xn;
                v = vn;
            }
            const double ST = std::exp(x);
            const double put = std::max(K - ST, 0.0);
            double payoff = 0;
            switch (pay) {
                case Payoff::Call: payoff = std::max(ST - K, 0.0); break;
                case Payoff::Put: payoff = put; break;
                case Payoff::DownInPut: payoff = hit ? put : put * (1.0 - surv); break;
                case Payoff::DownOutPut: payoff = hit ? 0.0 : put * surv; break;
            }
            pay_pair += payoff;
        }
        if (antithetic) pay_pair *= 0.5;
        sum += pay_pair;
        sum2 += pay_pair * pay_pair;
    }
    const double mean = sum / n_outer, var = sum2 / n_outer - mean * mean;
    return McResult{df * mean, df * std::sqrt(std::max(var, 0.0) / n_outer)};
}

}  // namespace qlite
