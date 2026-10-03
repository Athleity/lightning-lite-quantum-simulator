// src/DRAG.cpp
#include "Transmon.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace ll {

// ---------------------------------------------------------------------------
// Public API (declare these in bindings_pulse.cpp / any consumer)
//
//   Envelope gaussian_envelope(double theta, double duration, double sigma);
//   Envelope drag_envelope(double theta, double duration, double sigma, double beta,
//                          double anharmonicity_angular);
//   Envelope square_envelope(double theta, double duration);
//
// All envelopes are Omega(t) = Omega_I + i Omega_Q in rad/s, defined on [0, duration],
// normalised so that the in-phase area is the rotation angle:
//     integral_0^T Re Omega(t) dt = theta.
// ---------------------------------------------------------------------------
Envelope gaussian_envelope(double theta, double duration, double sigma);
Envelope drag_envelope(double theta, double duration, double sigma, double beta,
                       double anharmonicity_angular);
Envelope square_envelope(double theta, double duration);

namespace {

// duration / sigma below this makes the zero-offset normalisation ill-conditioned.
constexpr double MIN_DURATION_SIGMA_RATIO = 1.0;

[[noreturn]] void fail_arg(const std::string& who, const std::string& msg) {
    throw std::invalid_argument(who + ": " + msg);
}

bool finite_positive(double x) { return std::isfinite(x) && x > 0.0; }

// Truncated, zero-offset Gaussian
//     x(t) = A [ exp(-(t - t0)^2 / (2 sigma^2)) - e0 ],  t0 = T/2,  e0 = exp(-t0^2/(2 sigma^2)),
// which vanishes at t = 0 and t = T. With
//     int_0^T exp(-(t - t0)^2/(2 sigma^2)) dt = sigma sqrt(2 pi) erf(t0 / (sigma sqrt 2)),
// the area is A [ sigma sqrt(2 pi) erf(t0/(sigma sqrt 2)) - e0 T ] = theta.
struct GaussShape {
    double t0, sigma, e0, amp;
};

GaussShape make_shape(const char* who, double theta, double duration, double sigma) {
    if (!std::isfinite(theta)) fail_arg(who, "theta must be finite");
    if (!finite_positive(duration)) fail_arg(who, "duration must be finite and > 0");
    if (!finite_positive(sigma)) fail_arg(who, "sigma must be finite and > 0");
    if (duration / sigma < MIN_DURATION_SIGMA_RATIO)
        fail_arg(who, "duration / sigma must be >= " + std::to_string(MIN_DURATION_SIGMA_RATIO));
    GaussShape s;
    s.t0 = 0.5 * duration;
    s.sigma = sigma;
    s.e0 = std::exp(-s.t0 * s.t0 / (2.0 * sigma * sigma));
    const double area_g = sigma * std::sqrt(TWO_PI) * std::erf(s.t0 / (sigma * std::sqrt(2.0)));
    s.amp = theta / (area_g - s.e0 * duration);
    return s;
}

double shape_value(const GaussShape& s, double t) {
    const double d = t - s.t0;
    return s.amp * (std::exp(-d * d / (2.0 * s.sigma * s.sigma)) - s.e0);
}

double shape_derivative(const GaussShape& s, double t) {
    const double d = t - s.t0;
    return -s.amp * d / (s.sigma * s.sigma) * std::exp(-d * d / (2.0 * s.sigma * s.sigma));
}

}  // namespace

// Real Gaussian: Omega_Q = 0.
Envelope gaussian_envelope(double theta, double duration, double sigma) {
    const GaussShape s = make_shape("gaussian_envelope", theta, duration, sigma);
    return [s](double t) { return cplx(shape_value(s, t), 0.0); };
}

// First-order DRAG (Motzoi et al., PRL 103, 110501):
//     Omega(t) = x(t) - i beta x'(t) / alpha,    alpha = anharmonicity_angular < 0.
// The derivative quadrature cancels, to first order in 1/alpha, the spectral component of
// the drive at the 1-2 transition. Writing the |1> -> |2> amplitude as
//     A ~ (c12/2) int [ Omega_I + i Omega_Q ] c1(t) exp(i alpha t) dt,
// integration by parts leaves (c12/2) (i/alpha) int x' c1 exp(i alpha t) dt, and
// Omega_Q = -beta x'/alpha with beta = 1 removes it (c12 drops out). beta = 0 recovers the
// plain Gaussian; the sign of the quadrature is tied to the frame convention of
// Transmon::rotating_hamiltonian.
Envelope drag_envelope(double theta, double duration, double sigma, double beta,
                       double anharmonicity_angular) {
    const GaussShape s = make_shape("drag_envelope", theta, duration, sigma);
    if (!std::isfinite(beta)) fail_arg("drag_envelope", "beta must be finite");
    if (!std::isfinite(anharmonicity_angular) || anharmonicity_angular == 0.0)
        fail_arg("drag_envelope", "anharmonicity_angular must be finite and nonzero");
    const double k = -beta / anharmonicity_angular;
    return [s, k](double t) { return cplx(shape_value(s, t), k * shape_derivative(s, t)); };
}

// Constant real Omega = theta / T.
Envelope square_envelope(double theta, double duration) {
    if (!std::isfinite(theta)) fail_arg("square_envelope", "theta must be finite");
    if (!finite_positive(duration)) fail_arg("square_envelope", "duration must be finite and > 0");
    const double om = theta / duration;
    return [om](double) { return cplx(om, 0.0); };
}

}  // namespace ll

// ===========================================================================
// Validation
// ===========================================================================
#ifdef LL_TEST

#include <cstdio>
#include <limits>
#include <vector>

namespace {

struct Row {
    std::string name;
    double err;
    double tol;
};
std::vector<Row> g_rows;

void record(const std::string& name, double err, double tol) { g_rows.push_back({name, err, tol}); }

template <class E, class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Composite Simpson on [0, T], n even.
template <class F>
double simpson(F&& f, double T, int n) {
    const double h = T / n;
    double s = f(0.0) + f(T);
    for (int k = 1; k < n; ++k) s += f(k * h) * (k % 2 ? 4.0 : 2.0);
    return s * h / 3.0;
}

}  // namespace

int main() {
    using namespace ll;
    const double PI = TWO_PI / 2.0;
    const double EC = 300e6;
    try {
        // ---- 1. Area = rotation angle ---------------------------------------
        std::printf("== 1. Area: int Re Omega dt = theta (Gaussian and DRAG, T = 8 ns, sigma = 2 ns) ==\n");
        std::printf("%10s %14s %14s %10s\n", "theta/pi", "area/pi", "kind", "rel_err");
        const double T = 8e-9, sigma = 2e-9;
        const double alpha_demo = -TWO_PI * EC;
        double w1 = 0.0;
        for (double th : {0.5 * PI, PI, 2.0 * PI}) {
            const Envelope g = gaussian_envelope(th, T, sigma);
            const Envelope d = drag_envelope(th, T, sigma, 1.0, alpha_demo);
            const Envelope s = square_envelope(th, T);
            const double ag = simpson([&](double t) { return g(t).real(); }, T, 40000);
            const double ad = simpson([&](double t) { return d(t).real(); }, T, 40000);
            const double as = simpson([&](double t) { return s(t).real(); }, T, 40000);
            const double rg = std::abs(ag / th - 1.0), rd = std::abs(ad / th - 1.0),
                         rs = std::abs(as / th - 1.0);
            std::printf("%10.2f %14.10f %14s %10.2e\n", th / PI, ag / PI, "gaussian", rg);
            std::printf("%10.2f %14.10f %14s %10.2e\n", th / PI, ad / PI, "drag", rd);
            std::printf("%10.2f %14.10f %14s %10.2e\n", th / PI, as / PI, "square", rs);
            w1 = std::max({w1, rg, rd, rs});
        }
        record("1  area vs theta: max rel_err", w1, 1e-8);

        // ---- 2. Endpoints, symmetry, DRAG quadrature --------------------------
        std::printf("\n== 2. Endpoints, symmetry, Omega_Q = -beta Omega_I' / alpha ==\n");
        const double beta = 0.8;
        const Envelope d = drag_envelope(PI, T, sigma, beta, alpha_demo);
        const double peak = std::abs(d(0.5 * T));
        const double e_end = std::max(std::abs(d(0.0).real()), std::abs(d(T).real())) / peak;
        double e_sym = 0.0, e_der = 0.0;
        const double h = T * 1e-6;
        for (double f : {0.05, 0.2, 0.37, 0.5, 0.71, 0.9}) {
            const double t = f * T;
            e_sym = std::max(e_sym, std::abs(d(t).real() - d(T - t).real()) / peak);
            e_sym = std::max(e_sym, std::abs(d(t).imag() + d(T - t).imag()) / peak);
            const double dx = (d(t + h).real() - d(t - h).real()) / (2.0 * h);
            const double expect = -beta * dx / alpha_demo;
            const double scale = std::max(std::abs(expect), 1e-30);
            e_der = std::max(e_der, std::abs(d(t).imag() - expect) / scale);
        }
        std::printf("peak |Omega| = %.4e rad/s, edge/peak = %.2e, symmetry err = %.2e, "
                    "quadrature err = %.2e\n",
                    peak, e_end, e_sym, e_der);
        record("2a Re Omega vanishes at t = 0, T (rel. to peak)", e_end, 1e-12);
        record("2b Re even / Im odd about T/2", e_sym, 1e-12);
        record("2c Im Omega vs -beta Re Omega'/alpha (finite diff)", e_der, 1e-6);

        // ---- 3. DRAG suppresses leakage (RK4, RWA) ----------------------------------
        std::printf("\n== 3. Leakage vs beta: pi pulse, T = 8 ns, sigma = 2 ns, EJ/EC = 50, 4 levels ==\n");
        Transmon tr(50.0 * EC, EC, 4);
        const double alpha = tr.anharmonicity_angular();
        const double wd = tr.omega_01();
        auto pulse = [&](double b) {
            return DrivePulse{drag_envelope(PI, T, sigma, b, alpha), T, wd};
        };
        std::printf("alpha/2pi = %.3f MHz\n", alpha / TWO_PI / 1e6);
        std::printf("%8s %16s %16s\n", "beta", "leak(|1>)", "leak(|0>)");
        double L0 = 0.0, L1 = 0.0, Lm1 = 0.0;
        for (double b : {-1.0, -0.5, 0.0, 0.5, 1.0, 1.5}) {
            const double l1 = tr.leakage(pulse(b), 1);
            const double l0 = tr.leakage(pulse(b), 0);
            std::printf("%8.2f %16.6e %16.6e\n", b, l1, l0);
            if (b == 0.0) L0 = l1;
            if (b == 1.0) L1 = l1;
            if (b == -1.0) Lm1 = l1;
        }
        std::printf("leak(beta=1)/leak(beta=0) = %.4f\n", L1 / L0);
        record("3a leak(beta=1) / leak(beta=0)", L1 / L0, 0.5);
        record("3b wrong-sign DRAG worsens leakage (0 = yes)", Lm1 > L0 ? 0.0 : 1.0, 0.5);

        // ---- 4. Gate quality ----------------------------------------------------------
        std::printf("\n== 4. |0> -> |1> transfer with DRAG pi pulse ==\n");
        const CMatrix U = tr.propagator(pulse(1.0), DriveModel::RWA);
        const double p1 = std::norm(U(1, 0));
        const double e_unit = (U.adjoint() * U - CMatrix::Identity(4, 4)).cwiseAbs().maxCoeff();
        std::printf("P(1|0) = %.6f, 1 - P = %.3e, |U^+U - 1| = %.3e\n", p1, 1.0 - p1, e_unit);
        record("4a 1 - P(1|0) (uncorrected Stark phase included)", 1.0 - p1, 0.05);
        record("4b unitarity |U^+U - 1|", e_unit, 1e-8);

        // ---- 5. Input contract -----------------------------------------------------------
        std::printf("\n== 5. Input checks ==\n");
        const double nan = std::numeric_limits<double>::quiet_NaN();
        int bad = 0;
        bad += !throws<std::invalid_argument>([&] { gaussian_envelope(nan, T, sigma); });
        bad += !throws<std::invalid_argument>([&] { gaussian_envelope(PI, -T, sigma); });
        bad += !throws<std::invalid_argument>([&] { gaussian_envelope(PI, T, 0.0); });
        bad += !throws<std::invalid_argument>([&] { gaussian_envelope(PI, T, 2.0 * T); });
        bad += !throws<std::invalid_argument>([&] { drag_envelope(PI, T, sigma, 1.0, 0.0); });
        bad += !throws<std::invalid_argument>([&] { drag_envelope(PI, T, sigma, nan, alpha); });
        bad += !throws<std::invalid_argument>([&] { square_envelope(PI, 0.0); });
        std::printf("input-contract failures: %d / 7\n", bad);
        record("5  expected exceptions not thrown", bad, 0.5);
    } catch (const std::exception& e) {
        std::printf("\nUNEXPECTED EXCEPTION: %s\n", e.what());
        return 1;
    }

    // ---- Summary ------------------------------------------------------------------
    std::printf("\n== Summary ==\n%-52s %12s %12s  %s\n", "check", "max_err", "tol", "");
    bool ok = true;
    for (const Row& r : g_rows) {
        const bool pass = r.err <= r.tol;
        ok = ok && pass;
        std::printf("%-52s %12.3e %12.3e  %s\n", r.name.c_str(), r.err, r.tol, pass ? "PASS" : "FAIL");
    }
    std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}

#endif  // LL_TEST