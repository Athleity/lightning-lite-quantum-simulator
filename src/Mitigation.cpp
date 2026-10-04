// src/Mitigation.cpp
//
// Exact (density-matrix) counterpart of NoisyBackend in Mitigation.h, built on the Kraus channels
// of Noise.h. Provides a variance-free reference for ZNE and the cross-check for the sampled
// unraveling.
//
// ---------------------------------------------------------------------------------------
// DensityNoiseBackend
//
// Same noise model as NoisyBackend: after every gate application (folds included) the touched
// qubits get ll::DepolarizingChannel noise of total Pauli probability p,
//
//     m = 1 gate: 1-qubit channel,   m = 2 gate (cnot, 2-qubit mcz): 2-qubit channel,
//     m >= 3 (wide mcz): independent 1-qubit channels.
//
// The state is rho (2^n x 2^n). A gate U is the single-Kraus channel rho -> U rho U^+, applied
// with KrausChannel::apply, so no 2^n x 2^n operator is built. E(lambda) is therefore the exact
// expectation Tr(O rho_noisy), the limit of the trajectory average as N -> infinity, and
//
//     zne_exact(...) = sum_k c_k E(lambda_k)
//
// has no statistical error: it isolates the pure extrapolation bias. Backend::state() is not
// defined for a mixed state and throws std::logic_error; probabilities() returns diag(rho), so
// every diagonal observable written for Backend works unchanged.
//
// Closed forms used in validation (depolarizing, Pauli error probability p):
//     r1 = 1 - 4p/3,  r2 = 1 - 16p/15
//     1 qubit, g gates:  <Z>(lambda) = z0 r1^(g lambda)
//     Bell circuit H(0) CNOT(0,1):  <ZZ>(lambda) = r2^lambda
//
// Build (this file has its own main under LL_TEST; Noise.cpp uses OpenMP):
//   g++ -O2 -std=c++17 -fopenmp -DLL_TEST -I/usr/include/eigen3 \
//       src/Mitigation.cpp src/Noise.cpp -o build/mitigation_cpp_test
// ---------------------------------------------------------------------------------------

// Mitigation.h carries its own LL_TEST main. Hide the flag while including it.
#ifdef LL_TEST
#undef LL_TEST
#define LL_MITIGATION_CPP_TEST
#endif

#include "Mitigation.h"
#include "Noise.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ll {

class DensityNoiseBackend final : public Backend {
public:
    // p in [0, 1] (NaN rejected), n_qubits in [1, MAX_DM_QUBITS].
    explicit DensityNoiseBackend(double p, int n_qubits = 1) : p_(p), folds_(0), logical_(0), physical_(0), n_(0) {
        if (!(p >= 0.0 && p <= 1.0)) zne_detail::fail_arg("DensityNoiseBackend", "p must be in [0, 1]");
        reset(n_qubits);
    }

    double probability() const { return p_; }
    const CMatrix& density() const { return rho_; }

    // Odd integer scale in [1, MAX_ZNE_SCALE], else std::invalid_argument.
    void set_scale(int scale) {
        if (scale < 1 || scale > MAX_ZNE_SCALE || scale % 2 == 0)
            zne_detail::fail_arg("DensityNoiseBackend::set_scale",
                                 "scale must be an odd integer in [1, " + std::to_string(MAX_ZNE_SCALE) + "]");
        folds_ = (scale - 1) / 2;
    }
    int scale() const { return 2 * folds_ + 1; }
    long long logical_gates() const { return logical_; }
    long long physical_gates() const { return physical_; }

    void reset(int n_qubits) override {
        if (n_qubits < 1 || n_qubits > MAX_DM_QUBITS)
            zne_detail::fail_arg("DensityNoiseBackend::reset",
                                 "n_qubits must be in [1, " + std::to_string(MAX_DM_QUBITS) + "]");
        n_ = n_qubits;
        rho_ = ground_state_density(n_);
        logical_ = 0;
        physical_ = 0;
    }
    int n_qubits() const override { return n_; }

    void h(int q) override {
        const double r = 1.0 / std::sqrt(2.0);
        const CMatrix u = mat2(r, r, r, -r);
        fold1(q, u, u);
    }
    void x(int q) override {
        const CMatrix u = mat2(0.0, 1.0, 1.0, 0.0);
        fold1(q, u, u);
    }
    void rx(int q, double t) override {
        check_angle(t, "rx");
        fold1(q, rxm(t), rxm(-t));
    }
    void ry(int q, double t) override {
        check_angle(t, "ry");
        fold1(q, rym(t), rym(-t));
    }
    void rz(int q, double t) override {
        check_angle(t, "rz");
        fold1(q, rzm(t), rzm(-t));
    }
    void cnot(int control, int target) override {
        check_qubit(control);
        check_qubit(target);
        if (control == target) zne_detail::fail_arg("DensityNoiseBackend::cnot", "control == target");
        // Operator index bit 0 = control, bit 1 = target: |c=1,t=0> (1) <-> |c=1,t=1> (3).
        CMatrix u = CMatrix::Zero(4, 4);
        u(0, 0) = u(2, 2) = u(3, 1) = u(1, 3) = 1.0;
        const std::vector<int> qs = {control, target};
        fold(u, u, qs);
    }
    void mcz(const std::vector<int>& qubits) override {
        if (qubits.empty()) zne_detail::fail_arg("DensityNoiseBackend::mcz", "empty qubit list");
        for (std::size_t i = 0; i < qubits.size(); ++i) {
            check_qubit(qubits[i]);
            for (std::size_t j = 0; j < i; ++j)
                if (qubits[i] == qubits[j]) zne_detail::fail_arg("DensityNoiseBackend::mcz", "duplicate qubit");
        }
        const int dim = 1 << qubits.size();
        CMatrix u = CMatrix::Identity(dim, dim);
        u(dim - 1, dim - 1) = -1.0;
        fold(u, u, qubits);
    }

    std::vector<double> probabilities() const override {
        std::vector<double> p(static_cast<std::size_t>(rho_.rows()));
        for (Eigen::Index i = 0; i < rho_.rows(); ++i) p[static_cast<std::size_t>(i)] = rho_(i, i).real();
        return p;
    }
    std::vector<std::complex<double>> state() const override {
        throw std::logic_error("DensityNoiseBackend::state: the state is mixed, use density()");
    }

private:
    static CMatrix mat2(cplx a, cplx b, cplx c, cplx d) {
        CMatrix m(2, 2);
        m << a, b, c, d;
        return m;
    }
    static CMatrix rxm(double t) {
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        return mat2(c, cplx(0.0, -s), cplx(0.0, -s), c);
    }
    static CMatrix rym(double t) {
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        return mat2(c, -s, s, c);
    }
    static CMatrix rzm(double t) { return mat2(std::polar(1.0, -0.5 * t), 0.0, 0.0, std::polar(1.0, 0.5 * t)); }

    void check_qubit(int q) const {
        if (q < 0 || q >= n_)
            zne_detail::fail_arg("DensityNoiseBackend", "qubit " + std::to_string(q) + " outside [0, " +
                                                             std::to_string(n_) + ")");
    }
    static void check_angle(double t, const char* who) {
        if (!std::isfinite(t)) zne_detail::fail_arg(std::string("DensityNoiseBackend::") + who, "non-finite angle");
    }

    void unitary(const CMatrix& u, const std::vector<int>& qs) {
        KrausChannel({u}, qs).apply(rho_, n_);
    }

    void noise(const std::vector<int>& qs) {
        if (p_ == 0.0) return;
        if (qs.size() == 1)
            DepolarizingChannel(p_, 1).to_kraus(qs).apply(rho_, n_);
        else if (qs.size() == 2)
            DepolarizingChannel(p_, 2).to_kraus(qs).apply(rho_, n_);
        else
            for (int q : qs) DepolarizingChannel(p_, 1).to_kraus({q}).apply(rho_, n_);
    }

    void fold1(int q, const CMatrix& fwd, const CMatrix& inv) {
        check_qubit(q);
        fold(fwd, inv, std::vector<int>{q});
    }

    // U (U^+ U)^k with a noise site after every application. The first unitary validates the
    // targets before anything is counted.
    void fold(const CMatrix& fwd, const CMatrix& inv, const std::vector<int>& qs) {
        unitary(fwd, qs);
        noise(qs);
        for (int f = 0; f < folds_; ++f) {
            unitary(inv, qs);
            noise(qs);
            unitary(fwd, qs);
            noise(qs);
        }
        ++logical_;
        physical_ += 1 + 2LL * folds_;
    }

    double p_;
    int folds_;
    long long logical_;
    long long physical_;
    int n_;
    CMatrix rho_;
};

// Exact ZNE: E_k = Tr(O rho_noisy(lambda_k)) from the density matrix, then Richardson. The
// backend is reset to |0...0> at its current size before each scale, and its scale in force on
// entry is restored on return (also on exceptions). std_errors are 0, trajectories = 1.
// Throws like ZNE::extrapolate: std::invalid_argument for empty callables, invalid scales or a
// non-finite observable; std::runtime_error if circuit_fn applied no gates or the folded count
// is inconsistent.
ZNEResult zne_exact(DensityNoiseBackend& b, const CircuitFn& circuit_fn, const ObservableFn& observable_fn,
                    const std::vector<double>& scales = ZNE::default_scales()) {
    const char* who = "zne_exact";
    if (!circuit_fn) zne_detail::fail_arg(who, "circuit_fn is empty");
    if (!observable_fn) zne_detail::fail_arg(who, "observable_fn is empty");
    ZNE::validate_scales(scales);

    struct Guard {
        DensityNoiseBackend& b;
        int saved;
        explicit Guard(DensityNoiseBackend& x) : b(x), saved(x.scale()) {}
        ~Guard() { b.set_scale(saved); }
    } guard(b);

    ZNEResult out;
    out.scales = scales;
    out.weights = ZNE::richardson_weights(scales);
    out.trajectories = 1;
    for (double s : scales) {
        const int scale = static_cast<int>(s);
        b.set_scale(scale);
        b.reset(b.n_qubits());
        circuit_fn(b);
        if (b.logical_gates() == 0)
            throw std::runtime_error(std::string(who) + ": circuit_fn applied no gates to the backend");
        if (b.physical_gates() != static_cast<long long>(scale) * b.logical_gates())
            throw std::runtime_error(std::string(who) + ": folded gate count is inconsistent");
        const double v = observable_fn(b);
        if (!std::isfinite(v)) zne_detail::fail_arg(who, "observable returned a non-finite value");
        out.values.push_back(v);
        out.std_errors.push_back(0.0);
    }
    for (std::size_t k = 0; k < scales.size(); ++k) out.value += out.weights[k] * out.values[k];
    return out;
}

}  // namespace ll

// ===========================================================================
// Validation
// ===========================================================================
#ifdef LL_MITIGATION_CPP_TEST

#include <cstdio>

namespace {

struct Row {
    std::string name;
    double err;
    double tol;
};
std::vector<Row> g_rows;

void record(const std::string& name, double err, double tol) { g_rows.push_back({name, err, tol}); }

double sigma_dev(double est, double se, double expected) {
    return std::abs(est - expected) / std::max(se, 1e-12);
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
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double NSIG = 5.0;

    auto obs3 = [](const Backend& b) {
        const std::vector<double> p = b.probabilities();
        double e = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double s0 = 1.0 - 2.0 * static_cast<double>(i & 1);
            const double s1 = 1.0 - 2.0 * static_cast<double>((i >> 1) & 1);
            const double s2 = 1.0 - 2.0 * static_cast<double>((i >> 2) & 1);
            e += p[i] * (s0 * s1 + 0.5 * s2);
        }
        return e;
    };
    auto obsZ = [](const Backend& b) {
        const std::vector<double> p = b.probabilities();
        return p[0] - p[1];
    };
    auto obsZZ = [](const Backend& b) {
        const std::vector<double> p = b.probabilities();
        return p[0] + p[3] - p[1] - p[2];
    };
    auto circ3 = [](Backend& b) {
        b.h(0); b.ry(1, 0.7); b.cnot(0, 2); b.rz(2, 0.3);
        b.rx(0, 0.9); b.ry(2, -0.4); b.mcz({0, 1}); b.mcz({0, 1, 2});
        b.x(1); b.h(2); b.rx(1, 1.3); b.cnot(1, 0); b.rz(0, 0.8); b.mcz({2}); b.ry(0, 0.5);
    };
    auto circ1 = [](Backend& b) {
        b.ry(0, 1.1);
        b.rz(0, 0.7);
        b.rx(0, 0.4);
    };

    try {
        // ---- 1. Noiseless: density backend equals the reference state vector ---------------
        std::printf("== 1. Noiseless (p = 0): DensityNoiseBackend vs ReferenceBackend, 3 qubits ==\n");
        ReferenceBackend id3(3);
        circ3(id3);
        const double e_ideal = obs3(id3);
        {
            DensityNoiseBackend db(0.0, 3);
            circ3(db);
            const std::vector<double> pa = db.probabilities(), pb = id3.probabilities();
            double w_p = 0.0;
            for (std::size_t i = 0; i < pa.size(); ++i) w_p = std::max(w_p, std::abs(pa[i] - pb[i]));
            // Full rho against |psi><psi|.
            const std::vector<std::complex<double>> st = id3.state();
            CVector psi(8);
            for (int i = 0; i < 8; ++i) psi(i) = st[static_cast<std::size_t>(i)];
            const double w_rho = (db.density() - density_from_statevector(psi)).cwiseAbs().maxCoeff();
            const ZNEResult r = zne_exact(db, circ3, obs3, {1.0, 3.0, 5.0});
            std::printf("ideal <O> = %.14f, zne_exact = %.14f, max|p diff| = %.2e, max|rho diff| = %.2e\n",
                        e_ideal, r.value, w_p, w_rho);
            record("1a probabilities vs ReferenceBackend", w_p, 1e-12);
            record("1b rho vs |psi><psi|", w_rho, 1e-12);
            record("1c |zne_exact - ideal| (p = 0, scales 1,3,5)", std::abs(r.value - e_ideal), 1e-12);
        }

        // ---- 2. 1 qubit: E(lambda) = z0 r1^(3 lambda), and Richardson bias -------------------
        std::printf("\n== 2. 1 qubit, p = 0.02: <Z>(lambda) = z0 r1^(3 lambda), exact ==\n");
        const double p1 = 0.02, r1 = 1.0 - 4.0 * p1 / 3.0;
        ReferenceBackend id1(1);
        circ1(id1);
        const double z0 = obsZ(id1);
        {
            DensityNoiseBackend db(p1, 1);
            const ZNEResult r = zne_exact(db, circ1, obsZ, {1.0, 3.0, 5.0, 7.0});
            std::printf("z0 = %.10f, r1 = %.10f\n%8s %18s %18s %10s\n", z0, r1, "scale", "E_exact", "z0 r1^(3 lam)", "diff");
            double w = 0.0;
            for (std::size_t k = 0; k < r.scales.size(); ++k) {
                const double ex = z0 * std::pow(r1, 3.0 * r.scales[k]);
                w = std::max(w, std::abs(r.values[k] - ex));
                std::printf("%8.0f %18.12f %18.12f %10.1e\n", r.scales[k], r.values[k], ex, std::abs(r.values[k] - ex));
            }
            record("2a E(lambda) vs z0 r1^(3 lambda)", w, 1e-12);

            // Bias of n-point Richardson on E = z0 exp(-a lambda), a = -3 ln r1:
            // n = 1 (unmitigated), 2, 3, 4 points must shrink monotonically.
            std::printf("\n%8s %14s %14s\n", "points", "E(0) est", "|bias|");
            double prev = 1e9;
            bool mono = true;
            for (int n = 1; n <= 4; ++n) {
                double est;
                if (n == 1) {
                    est = r.values[0];
                } else {
                    const std::vector<double> s(r.scales.begin(), r.scales.begin() + n);
                    const std::vector<double> v(r.values.begin(), r.values.begin() + n);
                    est = ZNE::richardson(s, v);
                }
                const double bias = std::abs(est - z0);
                std::printf("%8d %14.10f %14.3e\n", n, est, bias);
                if (bias >= prev) mono = false;
                prev = bias;
            }
            record("2b bias decreases with number of scales (0 = yes)", mono ? 0.0 : 1.0, 0.5);

            // Lagrange remainder: E(0) - sum c E = z0 a^n exp(-a xi) prod(lambda)/n!, xi in (0, lambda_max).
            const double a = -3.0 * std::log(r1);
            const std::vector<double> s3 = {1.0, 3.0, 5.0};
            const std::vector<double> v3(r.values.begin(), r.values.begin() + 3);
            const double bias3 = z0 - ZNE::richardson(s3, v3);
            const double lo = z0 * std::pow(a, 3) * std::exp(-a * 5.0) * 15.0 / 6.0;
            const double hi = z0 * std::pow(a, 3) * 15.0 / 6.0;
            std::printf("3-point bias %.4e, Lagrange remainder bracket [%.4e, %.4e]\n", bias3, std::min(lo, hi), std::max(lo, hi));
            record("2c bias inside Lagrange remainder bracket (0 = yes)",
                   (bias3 >= std::min(lo, hi) - 1e-14 && bias3 <= std::max(lo, hi) + 1e-14) ? 0.0 : 1.0, 0.5);
        }

        // ---- 3. Bell circuit: <ZZ>(lambda) = r2^lambda -----------------------------------------
        std::printf("\n== 3. Two-qubit depolarizing: H(0) CNOT(0,1), <ZZ> = r2^lambda ==\n");
        {
            const double p2 = 0.03, r2 = 1.0 - 16.0 * p2 / 15.0;
            DensityNoiseBackend db(p2, 2);
            auto bell = [](Backend& b) {
                b.h(0);
                b.cnot(0, 1);
            };
            const ZNEResult r = zne_exact(db, bell, obsZZ, {1.0, 3.0, 5.0});
            double w = 0.0;
            std::printf("r2 = %.10f\n%8s %18s %18s\n", r2, "scale", "<ZZ>_exact", "r2^lambda");
            for (std::size_t k = 0; k < r.scales.size(); ++k) {
                const double ex = std::pow(r2, r.scales[k]);
                w = std::max(w, std::abs(r.values[k] - ex));
                std::printf("%8.0f %18.12f %18.12f\n", r.scales[k], r.values[k], ex);
            }
            record("3  <ZZ>(lambda) vs r2^lambda", w, 1e-12);
        }

        // ---- 4. Density-matrix invariants after a noisy folded circuit ----------------------------
        std::printf("\n== 4. Invariants: circ3, p = 0.05, scale 5 ==\n");
        {
            DensityNoiseBackend db(0.05, 3);
            db.set_scale(5);
            circ3(db);
            const CMatrix& rho = db.density();
            const double w_tr = std::abs(rho.trace() - cplx(1.0, 0.0));
            const double w_h = (rho - rho.adjoint()).cwiseAbs().maxCoeff();
            Eigen::SelfAdjointEigenSolver<CMatrix> es(rho, Eigen::EigenvaluesOnly);
            const double lmin = es.eigenvalues()(0);
            const double purity = (rho * rho).trace().real();
            std::printf("|Tr - 1| = %.2e, |rho - rho^+| = %.2e, min eig = %.2e, purity = %.6f, valid = %s\n",
                        w_tr, w_h, lmin, purity, is_valid_density_matrix(rho) ? "yes" : "NO");
            record("4a trace", w_tr, 1e-12);
            record("4b Hermiticity", w_h, 1e-12);
            record("4c positivity: max(0, -min eig)", std::max(0.0, -lmin), 1e-12);
            record("4d is_valid_density_matrix (0 = yes)", is_valid_density_matrix(rho) ? 0.0 : 1.0, 0.5);
            record("4e noise reduces purity below 1 (0 = yes)", purity < 1.0 ? 0.0 : 1.0, 0.5);
        }

        // ---- 5. Sampled unraveling vs exact density matrix -----------------------------------------
        std::printf("\n== 5. NoisyBackend (sampled) vs DensityNoiseBackend (exact), circ3, p = 0.02 ==\n");
        {
            const double p = 0.02;
            DensityNoiseBackend db(p, 3);
            const ZNEResult ex = zne_exact(db, circ3, obs3, {1.0, 3.0, 5.0});
            ReferenceBackend in3(3);
            NoisyBackend nb(in3, p, 2025);
            const ZNE zne(20000);
            const ZNEResult mc = zne.extrapolate(nb, circ3, obs3, {1.0, 3.0, 5.0});
            std::printf("ideal = %.6f\n%8s %12s %10s %12s %8s\n", e_ideal, "scale", "E_mc", "se", "E_exact", "dev/se");
            double w = 0.0;
            for (std::size_t k = 0; k < mc.scales.size(); ++k) {
                const double d = sigma_dev(mc.values[k], mc.std_errors[k], ex.values[k]);
                w = std::max(w, d);
                std::printf("%8.0f %12.6f %10.2e %12.6f %8.2f\n", mc.scales[k], mc.values[k], mc.std_errors[k],
                            ex.values[k], d);
            }
            const double dz = sigma_dev(mc.value, mc.std_error, ex.value);
            std::printf("ZNE: sampled %.6f +- %.1e, exact %.6f (dev %.2f sigma); unmitigated err %.4f, exact-ZNE err %.4f\n",
                        mc.value, mc.std_error, ex.value, dz, std::abs(ex.values[0] - e_ideal),
                        std::abs(ex.value - e_ideal));
            record("5a E_k sampled vs exact, max sigma", w, NSIG);
            record("5b ZNE value sampled vs exact, sigma", dz, NSIG);
            record("5c exact ZNE error below unmitigated (0 = yes)",
                   std::max(0.0, std::abs(ex.value - e_ideal) - std::abs(ex.values[0] - e_ideal)), 0.0);
        }

        // ---- 6. Input contract ------------------------------------------------------------------------
        std::printf("\n== 6. Input checks ==\n");
        {
            DensityNoiseBackend db(p1, 2);
            int n_exc = 0, bad = 0;
            auto chk = [&](bool ok) {
                ++n_exc;
                bad += !ok;
            };
            chk(throws<std::invalid_argument>([&] { DensityNoiseBackend b(-0.1); }));
            chk(throws<std::invalid_argument>([&] { DensityNoiseBackend b(1.1); }));
            chk(throws<std::invalid_argument>([&] { DensityNoiseBackend b(nan); }));
            chk(throws<std::invalid_argument>([&] { DensityNoiseBackend b(0.1, 0); }));
            chk(throws<std::invalid_argument>([&] { DensityNoiseBackend b(0.1, MAX_DM_QUBITS + 1); }));
            chk(throws<std::invalid_argument>([&] { db.h(2); }));
            chk(throws<std::invalid_argument>([&] { db.x(-1); }));
            chk(throws<std::invalid_argument>([&] { db.rx(0, nan); }));
            chk(throws<std::invalid_argument>([&] { db.cnot(0, 0); }));
            chk(throws<std::invalid_argument>([&] { db.mcz({}); }));
            chk(throws<std::invalid_argument>([&] { db.mcz({0, 0}); }));
            chk(throws<std::invalid_argument>([&] { db.set_scale(2); }));
            chk(throws<std::logic_error>([&] { db.state(); }));
            chk(throws<std::invalid_argument>([&] { zne_exact(db, circ1, obsZ, {1.0}); }));
            chk(throws<std::invalid_argument>([&] { zne_exact(db, circ1, obsZ, {1.0, 4.0}); }));
            chk(throws<std::invalid_argument>([&] { zne_exact(db, circ1, obsZ, {1.0, 3.0, 3.0}); }));
            chk(throws<std::invalid_argument>([&] { zne_exact(db, CircuitFn{}, obsZ); }));
            chk(throws<std::invalid_argument>([&] { zne_exact(db, circ1, ObservableFn{}); }));
            chk(throws<std::invalid_argument>([&] {
                zne_exact(db, circ1, [nan](const Backend&) { return nan; });
            }));
            chk(throws<std::runtime_error>([&] { zne_exact(db, [](Backend&) {}, obsZ); }));
            std::printf("input-contract failures: %d / %d\n", bad, n_exc);
            record("6  expected exceptions not thrown", bad, 0.5);
        }

        // ---- 7. Scale restoration ------------------------------------------------------------------------
        std::printf("\n== 7. Scale restoration ==\n");
        {
            DensityNoiseBackend db(p1, 1);
            db.set_scale(3);
            zne_exact(db, circ1, obsZ);
            const bool kept = db.scale() == 3;
            try {
                zne_exact(db, circ1, [nan](const Backend&) { return nan; });
            } catch (const std::invalid_argument&) {
            }
            const bool kept_exc = db.scale() == 3;
            std::printf("scale restored after run: %s, after exception: %s\n", kept ? "yes" : "NO",
                        kept_exc ? "yes" : "NO");
            record("7  scale restored (0 = yes)", (kept && kept_exc) ? 0.0 : 1.0, 0.5);
        }
    } catch (const std::exception& e) {
        std::printf("\nUNEXPECTED EXCEPTION: %s\n", e.what());
        return 1;
    }

    // ---- Summary ------------------------------------------------------------------
    std::printf("\n== Summary ==\n%-60s %12s %12s  %s\n", "check", "max_err", "tol", "");
    bool ok = true;
    for (const Row& r : g_rows) {
        const bool pass = r.err <= r.tol;
        ok = ok && pass;
        std::printf("%-60s %12.3e %12.3e  %s\n", r.name.c_str(), r.err, r.tol, pass ? "PASS" : "FAIL");
    }
    std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}

#endif  // LL_MITIGATION_CPP_TEST