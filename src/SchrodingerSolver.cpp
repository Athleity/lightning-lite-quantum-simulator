// src/SchrodingerSolver.cpp
//
// Lindblad master-equation solver, complementing Transmon::propagator (unitary only).
//
//     d rho/dt = -i [H(t), rho] + sum_k gamma_k ( L_k rho L_k^+ - 1/2 { L_k^+ L_k, rho } )
//
// Units follow Transmon.h: H in rad/s, gamma_k in 1/s, L_k dimensionless, t in seconds.
// Plain declarations are repeated here (as in DRAG.cpp) until a header exists.
//
// Method: classical RK4 applied directly to the d x d matrix rho, not to the d^2-vector.
//   - A vectorised Liouvillian is d^2 x d^2, so each application costs O(d^4). The matrix
//     form costs O((2 + K) d^3) per stage for K Lindblad operators, which is what makes
//     d = 2^MAX_DM_QUBITS = 1024 feasible at all.
//   - The right-hand side is written as  drho = M + M^+ + sum_k gamma_k L_k rho L_k^+,
//     with M = K rho and K = -iH - Gamma, Gamma = (1/2) sum_k gamma_k L_k^+ L_k.
//     For Hermitian rho this is Hermitian by construction, and since RK4 is linear and
//     tr(drho) = 0 term by term, trace is preserved to roundoff.
//   - Euler is rejected: it is only first order, and its amplification factor |1 - i z|
//     exceeds 1 for every z, so it would need dt orders of magnitude smaller to hold the
//     same accuracy on oscillatory H.
//
// Step control reuses Transmon.h constants. rate_bound = 2 max_t ||H(t)||_inf
// + 2 sum_k gamma_k ||L_k||_1 ||L_k||_inf, with ||H|| sampled at SOLVER_RATE_SAMPLES
// points. Norm sampling does not detect time dependence faster than the grid, so for
// rapidly oscillating H(t) pass an explicit `steps`.
//
// Build (Transmon.cpp has its own LL_TEST main, so compile it WITHOUT -DLL_TEST):
//   g++ -O2 -std=c++17 -I/usr/include/eigen3 -c src/Transmon.cpp -o build/Transmon.o
//   g++ -O2 -std=c++17 -DLL_TEST -I/usr/include/eigen3 \
//       src/SchrodingerSolver.cpp build/Transmon.o -o build/solver_test

#include "Transmon.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace ll {

// Mirrors MAX_DM_QUBITS in Noise.h. Hilbert-space dimension may be any d <= 2^10.
constexpr int SOLVER_MAX_QUBITS = 10;
constexpr int SOLVER_RATE_SAMPLES = 1025;
constexpr double SOLVER_STATE_TOL = 1e-9;  // Hermiticity / trace tolerance on rho0 and H

using HamiltonianFn = std::function<CMatrix(double)>;  // H(t)/hbar in rad/s

struct LindbladOp {
    CMatrix L;    // jump operator, dim x dim
    double rate;  // gamma_k in 1/s, >= 0
};

class LindbladSolver {
public:
    // Throws std::invalid_argument for dim outside [1, 2^SOLVER_MAX_QUBITS], an empty H,
    // a mis-sized or non-finite L, or a negative / non-finite rate.
    LindbladSolver(int dim, HamiltonianFn h, std::vector<LindbladOp> ops = {});

    int dim() const { return dim_; }

    // rho at t_end, starting from rho0 at t_start (H is evaluated at absolute time).
    // steps == AUTO_STEPS picks dt with rate_bound * dt <= RK4_TARGET_PHASE_STEP; a
    // positive steps throws if rate_bound * dt > RK4_MAX_PHASE_STEP. Throws
    // std::invalid_argument for steps < 0, t_end < t_start, non-finite times, a rho0 that
    // is mis-sized, non-finite, non-Hermitian or not unit trace, an H(t) that is mis-sized,
    // non-finite or non-Hermitian, or an automatic step count above MAX_PULSE_STEPS.
    CMatrix evolve(const CMatrix& rho0, double t_start, double t_end,
                   int steps = AUTO_STEPS) const;

    double rate_bound(double t_start, double t_end) const;

private:
    CMatrix eval_h(double t, const char* who) const;
    CMatrix deriv(const CMatrix& h, const CMatrix& rho) const;

    int dim_;
    HamiltonianFn h_;
    std::vector<LindbladOp> ops_;
    CMatrix gamma_;       // (1/2) sum_k gamma_k L_k^+ L_k
    double diss_rate_;    // static bound on the dissipator's rate
};

namespace {

[[noreturn]] void fail_arg(const std::string& who, const std::string& msg) {
    throw std::invalid_argument(who + ": " + msg);
}

double norm_inf(const CMatrix& m) { return m.cwiseAbs().rowwise().sum().maxCoeff(); }
double norm_one(const CMatrix& m) { return m.cwiseAbs().colwise().sum().maxCoeff(); }

}  // namespace

LindbladSolver::LindbladSolver(int dim, HamiltonianFn h, std::vector<LindbladOp> ops)
    : dim_(dim), h_(std::move(h)), ops_(std::move(ops)), diss_rate_(0.0) {
    const char* who = "LindbladSolver";
    if (dim < 1 || dim > (1 << SOLVER_MAX_QUBITS))
        fail_arg(who, "dim must be in [1, " + std::to_string(1 << SOLVER_MAX_QUBITS) + "]");
    if (!h_) fail_arg(who, "Hamiltonian callable is empty");
    gamma_ = CMatrix::Zero(dim_, dim_);
    for (const LindbladOp& op : ops_) {
        if (op.L.rows() != dim_ || op.L.cols() != dim_) fail_arg(who, "L must be dim x dim");
        if (!op.L.allFinite()) fail_arg(who, "L contains a non-finite entry");
        if (!(std::isfinite(op.rate) && op.rate >= 0.0))
            fail_arg(who, "rate must be finite and >= 0");
        gamma_ += 0.5 * op.rate * (op.L.adjoint() * op.L);
        diss_rate_ += 2.0 * op.rate * norm_one(op.L) * norm_inf(op.L);
    }
}

CMatrix LindbladSolver::eval_h(double t, const char* who) const {
    CMatrix h = h_(t);
    if (h.rows() != dim_ || h.cols() != dim_)
        fail_arg(who, "H(t) must be dim x dim at t = " + std::to_string(t));
    if (!h.allFinite()) fail_arg(who, "H(t) non-finite at t = " + std::to_string(t));
    return h;
}

double LindbladSolver::rate_bound(double t_start, double t_end) const {
    const char* who = "LindbladSolver::rate_bound";
    double hmax = 0.0;
    for (int k = 0; k < SOLVER_RATE_SAMPLES; ++k) {
        const double t = t_start + (t_end - t_start) * k / (SOLVER_RATE_SAMPLES - 1);
        const CMatrix h = eval_h(t, who);
        const double n = norm_inf(h);
        const double herm = (h - h.adjoint()).cwiseAbs().maxCoeff();
        if (herm > SOLVER_STATE_TOL * std::max(1.0, n))
            fail_arg(who, "H(t) is not Hermitian at t = " + std::to_string(t));
        hmax = std::max(hmax, n);
    }
    return 2.0 * hmax + diss_rate_;
}

// drho = M + M^+ + sum_k gamma_k L_k rho L_k^+,  M = (-iH - Gamma) rho.
CMatrix LindbladSolver::deriv(const CMatrix& h, const CMatrix& rho) const {
    const CMatrix k = cplx(0.0, -1.0) * h - gamma_;
    const CMatrix m = k * rho;
    CMatrix d = m + m.adjoint();
    for (const LindbladOp& op : ops_) d += op.rate * (op.L * rho * op.L.adjoint());
    return d;
}

CMatrix LindbladSolver::evolve(const CMatrix& rho0, double t_start, double t_end,
                               int steps) const {
    const char* who = "LindbladSolver::evolve";
    if (rho0.rows() != dim_ || rho0.cols() != dim_) fail_arg(who, "rho0 must be dim x dim");
    if (!rho0.allFinite()) fail_arg(who, "rho0 contains a non-finite entry");
    if ((rho0 - rho0.adjoint()).cwiseAbs().maxCoeff() > SOLVER_STATE_TOL)
        fail_arg(who, "rho0 is not Hermitian");
    if (std::abs(rho0.trace() - cplx(1.0, 0.0)) > SOLVER_STATE_TOL)
        fail_arg(who, "rho0 must have unit trace");
    if (!std::isfinite(t_start) || !std::isfinite(t_end)) fail_arg(who, "times must be finite");
    if (t_end < t_start) fail_arg(who, "t_end must be >= t_start");
    if (steps < 0) fail_arg(who, "steps must be >= 0");
    if (t_end == t_start) return rho0;

    const double span = t_end - t_start;
    const double rb = rate_bound(t_start, t_end);

    int n_steps;
    if (steps == AUTO_STEPS) {
        const double need = std::ceil(span * rb / RK4_TARGET_PHASE_STEP);
        if (!(need <= MAX_PULSE_STEPS)) fail_arg(who, "automatic step count exceeds MAX_PULSE_STEPS");
        n_steps = std::max(1, static_cast<int>(need));
    } else {
        n_steps = steps;
        if (rb * span / n_steps > RK4_MAX_PHASE_STEP)
            fail_arg(who, "rate_bound * dt exceeds RK4_MAX_PHASE_STEP; increase steps");
    }

    const double dt = span / n_steps;
    CMatrix rho = rho0;
    CMatrix h0 = eval_h(t_start, who);
    for (int s = 0; s < n_steps; ++s) {
        const double t0 = t_start + s * dt;
        const double t1 = (s + 1 == n_steps) ? t_end : t_start + (s + 1) * dt;
        const CMatrix hm = eval_h(t0 + 0.5 * dt, who);
        const CMatrix h1 = eval_h(t1, who);
        const CMatrix k1 = deriv(h0, rho);
        const CMatrix k2 = deriv(hm, rho + (0.5 * dt) * k1);
        const CMatrix k3 = deriv(hm, rho + (0.5 * dt) * k2);
        const CMatrix k4 = deriv(h1, rho + dt * k3);
        rho += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        h0 = h1;
    }
    return rho;
}

}  // namespace ll

// ===========================================================================
// Validation
// ===========================================================================
#ifdef LL_TEST

#include <cstdio>
#include <limits>

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

}  // namespace

int main() {
    using namespace ll;
    const double PI = TWO_PI / 2.0;
    try {
        // ---- 1. No Lindblad operators -> unitary evolution -------------------------
        std::printf("== 1. No jump operators vs Transmon::propagator and exact exp(-iHT) "
                    "(EJ/EC = 50, 3 levels, RWA) ==\n");
        const double EC = 300e6;
        Transmon tr(50.0 * EC, EC, 3);
        const double alpha = std::abs(tr.anharmonicity_angular());
        const double wd = tr.omega_01();

        const double Om0 = 0.1 * alpha, Tc = PI / Om0;
        const DrivePulse p_const{[Om0](double) { return cplx(Om0, 0.0); }, Tc, wd};
        const double Td = 20e-9, A = TWO_PI / Td;
        const DrivePulse p_td{[A, Td, PI](double t) {
                                  const double s = std::sin(PI * t / Td);
                                  return cplx(A * s * s, 0.3 * A * std::sin(TWO_PI * t / Td));
                              },
                              Td, wd};

        CVector psi(3);
        psi << cplx(1, 0), cplx(0, 1), cplx(-1, 0);
        psi /= std::sqrt(3.0);
        CMatrix rho_sup = psi * psi.adjoint();
        CMatrix rho_one = CMatrix::Zero(3, 3);
        rho_one(1, 1) = 1.0;

        std::printf("%-10s %-12s %18s %18s\n", "pulse", "state", "|rho-U rho U^+|", "|rho-exact|");
        double w1a = 0.0, w1b = 0.0;
        for (int ip = 0; ip < 2; ++ip) {
            const DrivePulse& p = ip == 0 ? p_const : p_td;
            LindbladSolver s(3, [&](double t) { return tr.rotating_hamiltonian(t, p, DriveModel::RWA); });
            const CMatrix U = tr.propagator(p, DriveModel::RWA);
            for (int is = 0; is < 2; ++is) {
                const CMatrix& r0 = is == 0 ? rho_one : rho_sup;
                const CMatrix r = s.evolve(r0, 0.0, p.duration);
                const double e_prop = (r - U * r0 * U.adjoint()).cwiseAbs().maxCoeff();
                double e_ex = std::numeric_limits<double>::quiet_NaN();
                if (ip == 0) {
                    Eigen::SelfAdjointEigenSolver<CMatrix> es(
                        tr.rotating_hamiltonian(0.0, p, DriveModel::RWA));
                    const CVector ph =
                        (es.eigenvalues().cast<cplx>().array() * cplx(0.0, -p.duration)).exp().matrix();
                    const CMatrix Uex = es.eigenvectors() * ph.asDiagonal() * es.eigenvectors().adjoint();
                    e_ex = (r - Uex * r0 * Uex.adjoint()).cwiseAbs().maxCoeff();
                    w1b = std::max(w1b, e_ex);
                }
                w1a = std::max(w1a, e_prop);
                std::printf("%-10s %-12s %18.3e %18.3e\n", ip == 0 ? "square" : "sin2+quad",
                            is == 0 ? "|1><1|" : "superpos", e_prop, e_ex);
            }
        }
        record("1a no-jump vs Transmon::propagator", w1a, 1e-6);
        record("1b no-jump vs exact exp(-iHT) (constant RWA)", w1b, 1e-7);

        // ---- 2. Pure dephasing -------------------------------------------------------
        // The solver takes L and rate separately: L = sigma_z with rate gamma/2 gives
        // d rho01/dt = -gamma rho01, i.e. the requested exp(-gamma t). H = (w/2) sigma_z
        // adds the free precession rho01 ~ exp(-i w t).
        const double gam = 1e6, w = TWO_PI * 5e6;
        CMatrix sz = CMatrix::Zero(2, 2);
        sz(0, 0) = 1.0;
        sz(1, 1) = -1.0;
        CMatrix sm = CMatrix::Zero(2, 2);  // sigma_- = |0><1|, index 1 is the excited state
        sm(0, 1) = 1.0;
        const HamiltonianFn hz = [&](double) { return CMatrix(0.5 * w * sz); };
        CMatrix plus(2, 2);
        plus << 0.5, 0.5, 0.5, 0.5;
        CMatrix exc = CMatrix::Zero(2, 2);
        exc(1, 1) = 1.0;
        const double times[] = {0.5e-6, 1e-6, 2e-6, 3e-6};

        std::printf("\n== 2. Pure dephasing: L = sigma_z, rate = gamma/2, gamma = 1e6 /s, "
                    "H = (w/2) sigma_z ==\n");
        std::printf("%10s %14s %14s %12s %12s\n", "t_us", "|rho01|_num", "|rho01|_exact", "err_rho01",
                    "err_pops");
        double w2 = 0.0;
        {
            LindbladSolver s(2, hz, {{sz, 0.5 * gam}});
            for (double t : times) {
                const CMatrix r = s.evolve(plus, 0.0, t);
                const cplx ex = 0.5 * std::exp(cplx(-gam * t, -w * t));
                const double e01 = std::abs(r(0, 1) - ex);
                const double ep = std::max(std::abs(r(0, 0) - 0.5), std::abs(r(1, 1) - 0.5));
                std::printf("%10.2f %14.8f %14.8f %12.3e %12.3e\n", t * 1e6, std::abs(r(0, 1)),
                            std::abs(ex), e01, ep);
                w2 = std::max({w2, e01, ep});
            }
        }
        record("2  dephasing: rho01 = exp(-gamma t) exp(-i w t)/2", w2, 5e-8);

        // ---- 3. Amplitude damping -------------------------------------------------------
        std::printf("\n== 3. Amplitude damping: L = sigma_-, rate = gamma = 1e6 /s ==\n");
        std::printf("%10s %14s %14s %12s %12s\n", "t_us", "P1(|1>)_num", "exp(-gamma t)", "err_P1",
                    "err_rho01(+)");
        double w3 = 0.0;
        {
            LindbladSolver s(2, hz, {{sm, gam}});
            for (double t : times) {
                const CMatrix r1 = s.evolve(exc, 0.0, t);
                const CMatrix r2 = s.evolve(plus, 0.0, t);
                const double p1 = std::exp(-gam * t);
                const double e_p = std::abs(r1(1, 1) - p1);
                const cplx ex01 = 0.5 * std::exp(cplx(-0.5 * gam * t, -w * t));
                const double e_c = std::max(std::abs(r2(0, 1) - ex01),
                                            std::abs(r2(1, 1) - 0.5 * p1));
                std::printf("%10.2f %14.8f %14.8f %12.3e %12.3e\n", t * 1e6, r1(1, 1).real(), p1,
                            e_p, e_c);
                w3 = std::max({w3, e_p, e_c});
            }
        }
        record("3  amplitude damping: P1, rho01 vs analytic", w3, 5e-8);

        // ---- 4-6. Trace, Hermiticity, positivity across a driven open-system run -----------
        // Driven 3-level transmon (RWA, sin^2 + quadrature pulse), pure superposition start,
        // exaggerated rates so the dissipators matter: relaxation 1->0 (1/50 ns), 2->1 at
        // twice that, and dephasing diag(0,1,2). Evolved in 40 chained segments.
        std::printf("\n== 4-6. Invariants along a driven open-system run (40 segments, 20 ns) ==\n");
        const double g1 = 2e7, gphi = 1e7;
        CMatrix L01 = CMatrix::Zero(3, 3), L12 = CMatrix::Zero(3, 3), Ld = CMatrix::Zero(3, 3);
        L01(0, 1) = 1.0;
        L12(1, 2) = 1.0;
        Ld(1, 1) = 1.0;
        Ld(2, 2) = 2.0;
        LindbladSolver so(3, [&](double t) { return tr.rotating_hamiltonian(t, p_td, DriveModel::RWA); },
                          {{L01, g1}, {L12, 2.0 * g1}, {Ld, gphi}});
        const int NSEG = 40;
        CMatrix rho = rho_sup;
        double w_tr = 0.0, w_herm = 0.0, lmin = 1e9;
        auto probe = [&](const CMatrix& r) {
            w_tr = std::max(w_tr, std::abs(r.trace() - cplx(1.0, 0.0)));
            w_herm = std::max(w_herm, (r - r.adjoint()).cwiseAbs().maxCoeff());
            Eigen::SelfAdjointEigenSolver<CMatrix> es(r, Eigen::EigenvaluesOnly);
            lmin = std::min(lmin, es.eigenvalues()(0));
        };
        probe(rho);
        std::printf("%10s %14s %14s %16s %10s\n", "t_ns", "|Tr-1|", "|rho-rho^+|", "min eig", "P(|2>)");
        for (int k = 0; k < NSEG; ++k) {
            rho = so.evolve(rho, k * Td / NSEG, (k + 1) * Td / NSEG);
            probe(rho);
            if ((k + 1) % 5 == 0) {
                Eigen::SelfAdjointEigenSolver<CMatrix> es(rho, Eigen::EigenvaluesOnly);
                std::printf("%10.2f %14.3e %14.3e %16.6e %10.6f\n", (k + 1) * Td / NSEG * 1e9,
                            std::abs(rho.trace() - cplx(1.0, 0.0)),
                            (rho - rho.adjoint()).cwiseAbs().maxCoeff(), es.eigenvalues()(0),
                            rho(2, 2).real());
            }
        }
        std::printf("max |Tr - 1| = %.3e, max |rho - rho^+| = %.3e, min eigenvalue = %.3e\n", w_tr,
                    w_herm, lmin);
        record("4  trace preservation max |Tr rho - 1|", w_tr, 1e-10);
        record("5  Hermiticity max |rho - rho^+|", w_herm, 1e-12);
        record("6  positivity: max(0, -min eigenvalue)", std::max(0.0, -lmin), 1e-8);

        // ---- 7. Input contract ----------------------------------------------------------------
        std::printf("\n== 7. Input checks ==\n");
        const double nan = std::numeric_limits<double>::quiet_NaN();
        LindbladSolver s2(2, hz, {{sm, gam}});
        CMatrix bad_herm = plus;
        bad_herm(0, 1) = cplx(0.5, 0.2);
        CMatrix bad_tr = 2.0 * plus;
        int bad = 0;
        bad += !throws<std::invalid_argument>([&] { LindbladSolver(0, hz); });
        bad += !throws<std::invalid_argument>([&] { LindbladSolver((1 << SOLVER_MAX_QUBITS) + 1, hz); });
        bad += !throws<std::invalid_argument>([&] { LindbladSolver(2, HamiltonianFn{}); });
        bad += !throws<std::invalid_argument>([&] { LindbladSolver(2, hz, {{sm, -1.0}}); });
        bad += !throws<std::invalid_argument>([&] { LindbladSolver(3, hz, {{sm, gam}}); });
        bad += !throws<std::invalid_argument>([&] { s2.evolve(CMatrix::Identity(3, 3) / 3.0, 0.0, 1e-6); });
        bad += !throws<std::invalid_argument>([&] { s2.evolve(plus, 1e-6, 0.0); });
        bad += !throws<std::invalid_argument>([&] { s2.evolve(plus, 0.0, 1e-6, -1); });
        bad += !throws<std::invalid_argument>([&] { s2.evolve(bad_herm, 0.0, 1e-6); });
        bad += !throws<std::invalid_argument>([&] { s2.evolve(bad_tr, 0.0, 1e-6); });
        bad += !throws<std::invalid_argument>([&] { s2.evolve(plus, 0.0, 1e-6, 10); });
        bad += !throws<std::invalid_argument>([&] {
            LindbladSolver s3(2, [nan](double) { return CMatrix::Constant(2, 2, cplx(nan, 0.0)); });
            s3.evolve(plus, 0.0, 1e-6);
        });
        std::printf("input-contract failures: %d / 12\n", bad);
        record("7  expected exceptions not thrown", bad, 0.5);
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