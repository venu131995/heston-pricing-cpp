// Python bindings (pybind11) for the analysis scripts; all pricing logic stays in the C++ headers.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qlite/bs.hpp"
#include "qlite/heston.hpp"
#include "qlite/mc.hpp"
#include "qlite/pde.hpp"

namespace py = pybind11;
using namespace qlite;

static const LewisIntegrator& lewis() {
    static LewisIntegrator L;
    return L;
}

PYBIND11_MODULE(qlite, m) {
    m.doc() = "Heston / Black-Scholes pricing library (C++17)";
    py::class_<Heston>(m, "Heston")
        .def(py::init<double, double, double, double, double>(), py::arg("v0"), py::arg("kappa"), py::arg("theta"),
             py::arg("sigma"), py::arg("rho"))
        .def_readwrite("v0", &Heston::v0).def_readwrite("kappa", &Heston::kappa)
        .def_readwrite("theta", &Heston::theta).def_readwrite("sigma", &Heston::sigma)
        .def_readwrite("rho", &Heston::rho).def("feller", &Heston::feller);
    m.def("bs_price", &bs_price);
    m.def("bs_vega", &bs_vega);
    m.def("bs_delta", &bs_delta);
    m.def("implied_vol", &implied_vol);
    m.def("bs_down_in_put", &bs_down_in_put);
    m.def("heston_price", [](const Heston& h, double S, double K, double T, double r, double q, bool call) {
        return heston_price(h, S, K, T, r, q, call, lewis());
    });
    m.def("calibrate", [](const std::vector<std::tuple<double, double, double, double, bool>>& quotes, double S,
                          double r, double q, const Heston& start) {
        std::vector<Quote> qs;
        for (auto& t : quotes) qs.push_back({std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t)});
        const auto res = calibrate(qs, S, r, q, start);
        return py::make_tuple(res.h, res.rmse_vol, res.iterations);
    });
    m.def("heston_mc_dip", [](const Heston& h, double S, double K, double H, double T, double r, double q, int steps,
                              long paths, unsigned long seed, bool continuous) {
        const auto res = heston_mc(h, S, K, T, r, q, Payoff::DownInPut, {H, continuous}, steps, paths, seed);
        return py::make_tuple(res.price, res.stderr_);
    });
    m.def("heston_adi_put", [](const Heston& h, double S, double K, double T, double r, double q, double H, int ns,
                               int nv, int nt) {
        return heston_adi(h, S, K, T, r, q, H > 0 ? PdeProduct::DownOutPut : PdeProduct::Put, H, ns, nv, nt);
    });
}
