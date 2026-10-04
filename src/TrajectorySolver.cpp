// Trajectory-solver demo: trajectories needed for a target standard error, per unraveling.
//
// TrajectorySolver.h is header-only and carries its own tests (-DLL_TEST). This translation unit
// is the executable that shows what the unravelings buy. For each system and time it runs a pilot
// ensemble per unraveling, estimates the per-trajectory variance sigma^2 = se^2 N, and reports
//   N(eps) = sigma^2 / eps^2   trajectories for a standard error eps.
//
//   Table 1  pure dephasing, L = Z, |+>, <X>. Pauli-like channel, PROJECTOR applies.
//            Closed form Var_std / Var_proj = 1 + exp(4 gamma t).
//   Table 2  amplitude damping, L = sigma_-, |1>, <P1>. Not Pauli-like, PROJECTOR falls back to
//            STANDARD (same plan, same seed, so the two columns agree exactly).
//
// The factor is a property of (unraveling, observable, initial state, gamma t), not a constant.
//
// Build:
//   g++ -O3 -std=c++17 -fopenmp -I/usr/include/eigen3 -Isrc src/TrajectorySolver.cpp \
//       -o build/trajectory_demo
// Run:
//   ./build/trajectory_demo [eps = 0.005] [pilot trajectories = 4000]
// Do not combine with -DLL_TEST: the header's test main would clash with main() below.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "TrajectorySolver.h"

using namespace ll;

namespace {

using C = std::complex<double>;

CMatrix mat2(C a, C b, C c, C d) {
    CMatrix m(2, 2);
    m << a, b, c, d;
    return m;
}

CVector vec2(C a, C b) {
    CVector v(2);
    v << a, b;
    return v;
}

double obs_x(const CVector& p) { return 2.0 * (std::conj(p(0)) * p(1)).real(); }
double obs_p1(const CVector& p) { return std::norm(p(1)); }

constexpr Unraveling KINDS[3] = {Unraveling::STANDARD, Unraveling::PROJECTOR, Unraveling::ANALOG};

struct Point {
    double var[3];
    double mean[3];
};

Point pilot(const TrajectorySolver& s, const CVector& psi0, double t, double dt, int n,
            double (*obs)(const CVector&), std::uint64_t seed) {
    Point p{};
    for (int k = 0; k < 3; ++k) {
        const auto e = s.estimate_observable(psi0, t, dt, n, KINDS[k], obs, seed);
        p.var[k] = e.std_error * e.std_error * n;
        p.mean[k] = e.mean;
    }
    return p;
}

std::string count(double var, double eps) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f", std::max(1.0, std::ceil(var / (eps * eps))));
    return buf;
}

std::string ratio(double a, double b) {
    char buf[32];
    if (b <= 0.0) return a <= 0.0 ? "1.00" : "inf";
    std::snprintf(buf, sizeof buf, "%.2f", a / b);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    const double eps = argc > 1 ? std::atof(argv[1]) : 0.005;
    const int pilot_n = argc > 2 ? std::atoi(argv[2]) : 4000;
    if (!(eps > 0.0) || pilot_n < 100) {
        std::fprintf(stderr, "usage: %s [eps > 0] [pilot trajectories >= 100]\n", argv[0]);
        return 1;
    }
    const CMatrix z = mat2(1.0, 0.0, 0.0, -1.0);
    const CMatrix sm = mat2(0.0, 1.0, 0.0, 0.0);  // |0><1|, |0> is the ground state
    const double r = 1.0 / std::sqrt(2.0);
    const double dt = 0.005;

    std::printf("Trajectories needed for standard error eps = %.4g (pilot %d, dt = %.3g, gamma = 1)\n", eps, pilot_n, dt);

    std::printf("\nTable 1: dephasing L = Z, |+>, <X>(t) = exp(-2 gamma t)\n");
    std::printf("  gamma t  Var STD    Var PROJ   Var ANALOG | N STD      N PROJ     N ANALOG   | STD/PROJ  analytic\n");
    {
        TrajectorySolver s(1);
        s.add_lindblad(z, 1.0);
        const CVector plus = vec2(r, r);
        for (double gt : {0.1, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0}) {
            const Point p = pilot(s, plus, gt, dt, pilot_n, obs_x, 11);
            std::printf("  %-8.2f %-9.5f %-10.5f %-10.5f | %-10s %-10s %-10s | %-9s %.2f\n", gt, p.var[0], p.var[1],
                        p.var[2], count(p.var[0], eps).c_str(), count(p.var[1], eps).c_str(),
                        count(p.var[2], eps).c_str(), ratio(p.var[0], p.var[1]).c_str(), 1.0 + std::exp(4.0 * gt));
        }
    }

    std::printf("\nTable 2: amplitude damping L = sigma_-, |1>, <P1>(t) = exp(-gamma t)\n");
    std::printf("  gamma t  Var STD    Var PROJ   Var ANALOG | N STD      N PROJ     N ANALOG   | STD/PROJ  STD/ANALOG\n");
    {
        TrajectorySolver s(1);
        s.add_lindblad(sm, 1.0);
        const CVector one = vec2(0.0, 1.0);
        for (double gt : {0.1, 0.25, 0.5, 1.0, 2.0}) {
            const Point p = pilot(s, one, gt, dt, pilot_n, obs_p1, 12);
            std::printf("  %-8.2f %-9.5f %-10.5f %-10.5f | %-10s %-10s %-10s | %-9s %s\n", gt, p.var[0], p.var[1], p.var[2],
                        count(p.var[0], eps).c_str(), count(p.var[1], eps).c_str(), count(p.var[2], eps).c_str(),
                        ratio(p.var[0], p.var[1]).c_str(), ratio(p.var[0], p.var[2]).c_str());
        }
    }

    std::printf("\nPROJECTOR rewrites only Pauli-like channels; sigma_- uses the standard unraveling.\n");
    return 0;
}