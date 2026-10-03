// src/Transmon.cpp
#include "Transmon.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace ll {

namespace {

// Smallest |<k|n|k+1>| accepted when fixing the sign gauge.
constexpr double N_ELEMENT_TOL = 1e-12;

[[noreturn]] void fail_arg(const std::string& who, const std::string& msg) {
    throw std::invalid_argument(who + ": " + msg);
}

bool finite_positive(double x) { return std::isfinite(x) && x > 0.0; }

// Delta_j = omega_j - j * omega_d, with omega_j = 2 pi (E_j - E_0)/h.
Eigen::VectorXd detunings(const Transmon& tr, double wd) {
    Eigen::VectorXd d(tr.n_levels());
    for (int j = 0; j < tr.n_levels(); ++j) d(j) = tr.omega(0, j) - j * wd;
    return d;
}

cplx eval_envelope(const DrivePulse& p, double t, const char* who) {
    const cplx v = p.envelope(t);
    if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
        fail_arg(who, "envelope returned a non-finite value at t = " + std::to_string(t));
    return v;
}

// H_rot(t) from precomputed coupling matrix c and detunings (shared with the RK4 loop).
CMatrix assemble_rotating(const Eigen::MatrixXd& c, const Eigen::VectorXd& delta,
                          cplx env, double t, double wd, DriveModel model) {
    const int n = static_cast<int>(delta.size());
    CMatrix h = CMatrix::Zero(n, n);
    for (int j = 0; j < n; ++j) h(j, j) = delta(j);

    if (model == DriveModel::RWA) {
        // (1/2) c_{j,j+1} [ Omega* |j><j+1| + Omega |j+1><j| ]
        for (int j = 0; j + 1 < n; ++j) {
            const cplx x = 0.5 * c(j, j + 1) * env;
            h(j + 1, j) += x;
            h(j, j + 1) += std::conj(x);
        }
    } else {
        // f(t) sum_ij c_ij exp(i (i-j) omega_d t) |i><j|, f = Re[Omega exp(-i omega_d t)]
        const double f = std::real(env * std::exp(cplx(0.0, -wd * t)));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                h(i, j) += f * c(i, j) * std::exp(cplx(0.0, (i - j) * wd * t));
    }
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Transmon::Transmon(double ej, double ec, int n_levels, int charge_cutoff, double ng)
    : ej_(ej), ec_(ec), ng_(ng), n_levels_(n_levels), cutoff_(charge_cutoff) {
    if (!finite_positive(ej)) fail_arg("Transmon", "ej must be finite and > 0");
    if (!finite_positive(ec)) fail_arg("Transmon", "ec must be finite and > 0");
    if (n_levels < MIN_LEVELS || n_levels > MAX_LEVELS)
        fail_arg("Transmon", "n_levels must be in [" + std::to_string(MIN_LEVELS) + ", " +
                                 std::to_string(MAX_LEVELS) + "]");
    if (charge_cutoff < MIN_CHARGE_CUTOFF)
        fail_arg("Transmon", "charge_cutoff must be >= " + std::to_string(MIN_CHARGE_CUTOFF));
    if (2 * charge_cutoff + 1 <= n_levels)
        fail_arg("Transmon", "charge basis dimension must exceed n_levels");
    if (!std::isfinite(ng)) fail_arg("Transmon", "ng must be finite");
    diagonalize();
}

void Transmon::diagonalize() {
    const Eigen::MatrixXd h = hamiltonian_charge_basis();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(h);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("Transmon: eigensolver did not converge");

    energies_ = es.eigenvalues().head(n_levels_);
    states_ = es.eigenvectors().leftCols(n_levels_);

    for (int k = 0; k < n_levels_; ++k)
        if (edge_weight(k) > CHARGE_EDGE_TOL)
            fail_arg("Transmon", "level " + std::to_string(k) + " has edge weight " +
                                     std::to_string(edge_weight(k)) +
                                     " > CHARGE_EDGE_TOL; increase charge_cutoff");

    const int dim = charge_dim();
    Eigen::VectorXd nvec(dim);
    for (int i = 0; i < dim; ++i) nvec(i) = static_cast<double>(i - cutoff_);

    // Sign gauge <k|n|k+1> > 0, fixed sequentially from level 0.
    for (int k = 0; k + 1 < n_levels_; ++k) {
        const double e = states_.col(k).dot(nvec.cwiseProduct(states_.col(k + 1)));
        if (!(std::abs(e) > N_ELEMENT_TOL))
            throw std::runtime_error("Transmon: n_{" + std::to_string(k) + "," +
                                     std::to_string(k + 1) + "} vanishes");
        if (e < 0.0) states_.col(k + 1) *= -1.0;
    }

    n_matrix_ = states_.transpose() * nvec.asDiagonal() * states_;
    n_matrix_ = (0.5 * (n_matrix_ + n_matrix_.transpose())).eval();
}

void Transmon::check_level(int level, const char* who) const {
    if (level < 0 || level >= n_levels_)
        throw std::out_of_range(std::string(who) + ": level " + std::to_string(level) +
                                " not in [0, " + std::to_string(n_levels_) + ")");
}

void Transmon::check_pulse(const DrivePulse& pulse, const char* who) const {
    if (!pulse.envelope) fail_arg(who, "pulse envelope is empty");
    if (!finite_positive(pulse.duration)) fail_arg(who, "duration must be finite and > 0");
    if (!finite_positive(pulse.drive_frequency))
        fail_arg(who, "drive_frequency must be finite and > 0");
}

// ---------------------------------------------------------------------------
// Spectrum
// ---------------------------------------------------------------------------

double Transmon::energy(int level) const {
    check_level(level, "Transmon::energy");
    return energies_(level);
}

double Transmon::frequency(int i, int j) const {
    check_level(i, "Transmon::frequency");
    check_level(j, "Transmon::frequency");
    return energies_(j) - energies_(i);
}

double Transmon::omega(int i, int j) const { return TWO_PI * frequency(i, j); }

double Transmon::anharmonicity() const { return energies_(2) - 2.0 * energies_(1) + energies_(0); }

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

Eigen::MatrixXd Transmon::hamiltonian_charge_basis() const {
    const int dim = charge_dim();
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) {
        const double n = static_cast<double>(i - cutoff_) - ng_;
        h(i, i) = 4.0 * ec_ * n * n;
        if (i + 1 < dim) h(i, i + 1) = h(i + 1, i) = -0.5 * ej_;
    }
    return h;
}

Eigen::VectorXd Transmon::eigenstate(int level) const {
    check_level(level, "Transmon::eigenstate");
    return states_.col(level);
}

double Transmon::edge_weight(int level) const {
    check_level(level, "Transmon::edge_weight");
    const Eigen::Index last = states_.rows() - 1;
    return states_(0, level) * states_(0, level) + states_(last, level) * states_(last, level);
}

double Transmon::n_element(int i, int j) const {
    check_level(i, "Transmon::n_element");
    check_level(j, "Transmon::n_element");
    return n_matrix_(i, j);
}

Eigen::MatrixXd Transmon::coupling_matrix() const { return n_matrix_ / n_matrix_(0, 1); }

// ---------------------------------------------------------------------------
// Transmon-limit formulas
// ---------------------------------------------------------------------------

double Transmon::plasma_frequency() const { return std::sqrt(8.0 * ej_ * ec_); }

double Transmon::frequency_01_asymptotic() const {
    return plasma_frequency() - ec_ - ec_ * std::sqrt(ec_ / ej_) / (2.0 * std::sqrt(2.0));
}

double Transmon::anharmonicity_leading() const { return -ec_; }

double Transmon::anharmonicity_asymptotic() const {
    return -ec_ * (1.0 + 9.0 / (8.0 * std::sqrt(2.0)) * std::sqrt(ec_ / ej_));
}

double Transmon::n01_asymptotic() const { return 0.5 * std::pow(ej_ / (2.0 * ec_), 0.25); }

// ---------------------------------------------------------------------------
// Driven dynamics
// ---------------------------------------------------------------------------

CMatrix Transmon::rotating_hamiltonian(double t, const DrivePulse& pulse, DriveModel model) const {
    check_pulse(pulse, "Transmon::rotating_hamiltonian");
    const cplx env = eval_envelope(pulse, t, "Transmon::rotating_hamiltonian");
    return assemble_rotating(coupling_matrix(), detunings(*this, pulse.drive_frequency), env, t,
                             pulse.drive_frequency, model);
}

double Transmon::rate_bound(const DrivePulse& pulse, double peak_omega, DriveModel model) const {
    const Eigen::VectorXd delta = detunings(*this, pulse.drive_frequency);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(coupling_matrix(), Eigen::EigenvaluesOnly);
    const double cnorm = es.eigenvalues().cwiseAbs().maxCoeff();  // C is real symmetric
    double r = delta.cwiseAbs().maxCoeff() + peak_omega * cnorm;
    if (model == DriveModel::Full) r += n_levels_ * pulse.drive_frequency;
    return r;
}

CMatrix Transmon::propagator(const DrivePulse& pulse, DriveModel model, int steps) const {
    const char* who = "Transmon::propagator";
    check_pulse(pulse, who);
    if (steps < 0) fail_arg(who, "steps must be >= 0");

    double peak = 0.0;
    for (int k = 0; k < ENVELOPE_SAMPLES; ++k) {
        const double t = pulse.duration * k / (ENVELOPE_SAMPLES - 1);
        peak = std::max(peak, std::abs(eval_envelope(pulse, t, who)));
    }
    const double rb = rate_bound(pulse, peak, model);

    int n_steps;
    if (steps == AUTO_STEPS) {
        const double need = std::ceil(pulse.duration * rb / RK4_TARGET_PHASE_STEP);
        if (!(need <= MAX_PULSE_STEPS))
            fail_arg(who, "automatic step count exceeds MAX_PULSE_STEPS");
        n_steps = std::max(1, static_cast<int>(need));
    } else {
        n_steps = steps;
        if (rb * pulse.duration / n_steps > RK4_MAX_PHASE_STEP)
            fail_arg(who, "rate_bound * dt exceeds RK4_MAX_PHASE_STEP; increase steps");
    }

    const int n = n_levels_;
    const double wd = pulse.drive_frequency;
    const double dt = pulse.duration / n_steps;
    const Eigen::MatrixXd c = coupling_matrix();
    const Eigen::VectorXd delta = detunings(*this, wd);
    const cplx mi(0.0, -1.0);

    auto H = [&](double t) {
        return assemble_rotating(c, delta, eval_envelope(pulse, t, who), t, wd, model);
    };

    CMatrix U = CMatrix::Identity(n, n);
    CMatrix h0 = H(0.0);
    CMatrix k1, k2, k3, k4;
    for (int s = 0; s < n_steps; ++s) {
        const double t0 = s * dt;
        const double t1 = (s + 1 == n_steps) ? pulse.duration : (s + 1) * dt;
        const CMatrix hm = H(t0 + 0.5 * dt);
        const CMatrix h1 = H(t1);
        k1 = mi * (h0 * U);
        k2 = mi * (hm * (U + (0.5 * dt) * k1));
        k3 = mi * (hm * (U + (0.5 * dt) * k2));
        k4 = mi * (h1 * (U + dt * k3));
        U += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        h0 = h1;
    }
    return U;
}

double Transmon::leakage(const DrivePulse& pulse, int initial_level, DriveModel model,
                         int steps) const {
    if (initial_level != 0 && initial_level != 1)
        fail_arg("Transmon::leakage", "initial_level must be 0 or 1");
    const CMatrix U = propagator(pulse, model, steps);
    double p = 0.0;
    for (int k = 2; k < n_levels_; ++k) p += std::norm(U(k, initial_level));
    return p;
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

ll::DrivePulse square(double omega, double T, double wd) {
    return ll::DrivePulse{[omega](double) { return ll::cplx(omega, 0.0); }, T, wd};
}

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
    const double EC = 300e6;
    const double ratios[] = {20.0, 50.0, 100.0, 500.0};
    try {
                // ---- 1. Anharmonicity -> -EC ----------------------------------------
        std::printf("== 1. Anharmonicity -> -EC (EC = 300 MHz) ==\n");
        std::printf("%8s %12s %12s %15s %10s\n", "EJ/EC", "alpha_MHz", "-EC_MHz", "asymptotic_MHz",
                    "rel_err");
        double rel_prev = 1e9, rel_last = 0.0, dev_prev = 1e9, dev500 = 0.0;
        bool rel_mono = true, mono = true;
        for (double r : ratios) {
            Transmon t(r * EC, EC);
            const double a = t.anharmonicity(), asy = t.anharmonicity_asymptotic();
            const double rel = std::abs(a - asy) / std::abs(asy);
            std::printf("%8.0f %12.4f %12.4f %15.4f %10.2e\n", r, a / 1e6, -EC / 1e6, asy / 1e6, rel);
            if (rel >= rel_prev) rel_mono = false;
            rel_prev = rel;
            rel_last = rel;  // ratios are ascending, so this ends as the EJ/EC = 500 value
            const double dev = std::abs(a + EC) / EC;
            if (dev >= dev_prev) mono = false;
            dev_prev = dev;
            dev500 = dev;
        }
        record("1a alpha vs asymptotic: rel_err at EJ/EC = 500", rel_last, 5e-3);
        record("1b alpha vs asymptotic: rel_err decreasing (0 = yes)", rel_mono ? 0.0 : 1.0, 0.5);
        record("1c |alpha + EC|/EC at EJ/EC = 500", dev500, 0.06);
        record("1d |alpha + EC|/EC decreasing in EJ/EC (0 = yes)", mono ? 0.0 : 1.0, 0.5);

        // ---- 2. Frequency -----------------------------------------------------
        std::printf("\n== 2. f01 vs transmon-limit expansion ==\n");
        std::printf("%8s %14s %14s %10s\n", "EJ/EC", "f01_GHz", "asym_GHz", "rel_err");
        double w2 = 0.0;
        for (double r : ratios) {
            Transmon t(r * EC, EC);
            const double f = t.frequency_01(), fa = t.frequency_01_asymptotic();
            const double rel = std::abs(f - fa) / f;
            std::printf("%8.0f %14.6f %14.6f %10.2e\n", r, f / 1e9, fa / 1e9, rel);
            w2 = std::max(w2, rel);
        }
        record("2  f01 vs asymptotic: max rel_err", w2, 0.01);

        // ---- 3. Parity --------------------------------------------------------
        std::printf("\n== 3. Parity: n_ij = 0 for i + j even (ng = 0, EJ/EC = 50, 6 levels) ==\n");
        double w3 = 0.0;
        {
            Transmon t(50.0 * EC, EC, 6, 40, 0.0);
            const Eigen::MatrixXd& nm = t.n_matrix();
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    if ((i + j) % 2 == 0) w3 = std::max(w3, std::abs(nm(i, j)));
            std::printf("max |n_ij| (i+j even) = %.3e\n", w3);
        }
        record("3  parity: max |n_ij|, i+j even", w3, 1e-10);

        // ---- 4. Matrix elements ------------------------------------------------
        std::printf("\n== 4. n_12 / n_01 -> sqrt(2) ==\n");
        std::printf("%8s %10s %10s %10s %12s %12s\n", "EJ/EC", "n01", "n01_asym", "n12", "n12/n01",
                    "dev_vs_sqrt2");
        double dev100 = 0.0, dev500b = 0.0, n01err = 0.0;
        for (double r : {100.0, 500.0}) {
            Transmon t(r * EC, EC);
            const double n01 = t.n_element(0, 1), n12 = t.n_element(1, 2);
            const double dev = std::abs(n12 / n01 / std::sqrt(2.0) - 1.0);
            std::printf("%8.0f %10.5f %10.5f %10.5f %12.6f %12.3e\n", r, n01, t.n01_asymptotic(), n12,
                        n12 / n01, dev);
            n01err = std::max(n01err, std::abs(n01 - t.n01_asymptotic()) / t.n01_asymptotic());
            (r == 100.0 ? dev100 : dev500b) = dev;
        }
        record("4a |n12/n01/sqrt2 - 1| at EJ/EC = 100", dev100, 0.15);
        record("4b n01 vs asymptotic: max rel_err", n01err, 0.05);
        record("4c deviation shrinks 100 -> 500 (0 = yes)", dev500b < dev100 ? 0.0 : 1.0, 0.5);

        // ---- 5. Weak-drive leakage scaling ---------------------------------------
        // Resonant square pi pulse (T = pi/Omega), RWA, start in |1>. To leading order
        // P_2 = (c12 Omega / 2 alpha)^2, so leakage / (Omega/alpha)^2 -> c12^2 / 4.
        std::printf("\n== 5. Weak-drive leakage, square pi pulse, RWA (EJ/EC = 50) ==\n");
        Transmon t5(50.0 * EC, EC, 3);
        const double alpha = std::abs(t5.anharmonicity_angular());
        const double wd = t5.omega_01();
        const double c12 = t5.coupling_matrix()(1, 2);
        const double pred = 0.25 * c12 * c12;
        std::printf("c12 = %.6f, predicted leakage/(Omega/alpha)^2 = %.6f\n", c12, pred);
        std::printf("%10s %14s %16s %12s\n", "Om/|alpha|", "leakage", "leak/(Om/al)^2", "/predicted");
        double q0 = 0.0, spread = 0.0;
        for (double r : {0.01, 0.02, 0.05, 0.1, 0.2}) {
            const double Om = r * alpha;
            const double L = t5.leakage(square(Om, M_PI / Om, wd), 1, DriveModel::RWA);
            const double q = L / (r * r);
            if (r == 0.01) q0 = q;
            spread = std::max(spread, std::abs(q / q0 - 1.0));
            std::printf("%10.3f %14.6e %16.6f %12.5f\n", r, L, q, q / pred);
        }
        record("5a leak/(Om/al)^2 at 0.01 vs c12^2/4: rel_err", std::abs(q0 / pred - 1.0), 0.03);
        record("5b leak/(Om/al)^2 spread over 0.01..0.2", spread, 0.15);

        // ---- 6. n_levels convergence --------------------------------------------
        std::printf("\n== 6. Leakage vs n_levels (Omega/|alpha| = 0.2, pi pulse, EJ/EC = 50) ==\n");
        std::printf("%8s %16s %14s\n", "n_levels", "leakage", "change");
        double L[7] = {0};
        for (int nl = 3; nl <= 6; ++nl) {
            Transmon t(50.0 * EC, EC, nl);
            const double Om = 0.2 * std::abs(t.anharmonicity_angular());
            L[nl] = t.leakage(square(Om, M_PI / Om, t.omega_01()), 1, DriveModel::RWA);
            if (nl == 3)
                std::printf("%8d %16.8e %14s\n", nl, L[nl], "-");
            else
                std::printf("%8d %16.8e %14.3e\n", nl, L[nl], L[nl] - L[nl - 1]);
        }
        record("6  saturation: max(|L5-L4|,|L6-L4|)/L6",
               std::max(std::abs(L[5] - L[4]), std::abs(L[6] - L[4])) / L[6], 2e-3);

        // ---- 7. Propagator, Hamiltonian, input contract -----------------------------
        std::printf("\n== 7. RK4 vs exact exp(-iHT), Hermiticity, input checks ==\n");
        {
            const double Om = 0.1 * alpha, T = M_PI / Om;
            const DrivePulse p = square(Om, T, wd);
            const CMatrix U = t5.propagator(p, DriveModel::RWA);
            Eigen::SelfAdjointEigenSolver<CMatrix> es(t5.rotating_hamiltonian(0.0, p, DriveModel::RWA));
            const CVector ph = (es.eigenvalues().cast<cplx>().array() * cplx(0.0, -T)).exp().matrix();
            const CMatrix Uex = es.eigenvectors() * ph.asDiagonal() * es.eigenvectors().adjoint();
            const double e_exact = (U - Uex).cwiseAbs().maxCoeff();
            const double e_unit =
                (U.adjoint() * U - CMatrix::Identity(3, 3)).cwiseAbs().maxCoeff();
            double e_herm = 0.0;
            for (DriveModel m : {DriveModel::RWA, DriveModel::Full}) {
                const CMatrix H = t5.rotating_hamiltonian(3.7e-9, p, m);
                e_herm = std::max(e_herm, (H - H.adjoint()).cwiseAbs().maxCoeff());
            }
            std::printf("max|U_rk4 - U_exact| = %.3e, |U^+U - 1| = %.3e, |H - H^+| = %.3e\n", e_exact,
                        e_unit, e_herm);
            record("7a RK4 vs exact (constant RWA pulse)", e_exact, 1e-7);
            record("7b unitarity |U^+U - 1|", e_unit, 1e-8);
            record("7c Hermiticity of H_rot (RWA, Full)", e_herm, 1e-6);

            const double nan = std::numeric_limits<double>::quiet_NaN();
            int bad = 0;
            bad += !throws<std::invalid_argument>([&] { Transmon t(-1.0, EC); (void)t; });
            bad += !throws<std::invalid_argument>([&] { Transmon t(50 * EC, EC, 2); (void)t; });
            bad += !throws<std::invalid_argument>([&] { Transmon t(500 * EC, EC, 3, 8); (void)t; });
            bad += !throws<std::out_of_range>([&] { t5.energy(3); });
            bad += !throws<std::invalid_argument>([&] { t5.leakage(p, 2); });
            bad += !throws<std::invalid_argument>([&] { t5.propagator(p, DriveModel::RWA, -1); });
            bad += !throws<std::invalid_argument>([&] { t5.propagator(p, DriveModel::RWA, 10); });
            bad += !throws<std::invalid_argument>([&] {
                t5.propagator(DrivePulse{[nan](double) { return cplx(nan, 0.0); }, 1e-8, wd});
            });
            std::printf("input-contract failures: %d / 8\n", bad);
            record("7d expected exceptions not thrown", bad, 0.5);
        }
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