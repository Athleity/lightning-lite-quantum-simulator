// src/Mitigation.h
//
// Zero-noise extrapolation (ZNE) on the ll::Backend interface. Header only; it needs
// Backend.h and nothing else (no Eigen, no link against Noise.cpp).
//
// ---------------------------------------------------------------------------------------
// Noise model: NoisyBackend
//
// After every gate NoisyBackend applies depolarizing noise of total Pauli-error probability p
// to the qubits that gate touched, with the convention of ll::DepolarizingChannel (Noise.h):
//
//     rho -> (1 - p) rho + p / (4^m - 1) sum_{P != I} P rho P,    m = number of qubits.
//
// Every non-identity Pauli string has an identity-shifted sign pattern with 4^m / 2 commuting and
// 4^m / 2 anticommuting members, so every non-identity Pauli expectation shrinks by
//
//     m = 1:  r1 = 1 - 4p/3,        m = 2:  r2 = 1 - 16p/15.
//
// Gate arity: 1 qubit -> m = 1, 2 qubits (cnot, 2-qubit mcz) -> m = 2, 3 or more qubits (wide
// mcz) -> independent m = 1 channels on each qubit.
//
// Unraveling. The backend holds one pure state vector, so the channel is sampled: at each
// noise site draw u in [0, 1); if u >= p do nothing, otherwise apply one of the 4^m - 1
// non-identity Paulis uniformly at random. The Pauli is applied through the wrapped backend's
// own gates (X = x, Z = mcz({q}), Y = Z then X up to a global phase), so no 2^n x 2^n operator
// and no access to the amplitudes is needed and any Backend implementation works. The average of
// an observable over trajectories equals Tr(O rho_noisy) exactly, and its statistical error is
// sigma / sqrt(N) for N trajectories.
//
// ---------------------------------------------------------------------------------------
// Gate folding
//
// Every logical gate U is replaced by U (U^+ U)^k, with k >= 0 folds. Noiselessly this is U.
// Each of the 2k + 1 applications gets its own noise site, so the noise is amplified by the
// scale factor
//
//     lambda = 2k + 1     (1, 3, 5, ...).
//
// Inverses: H, X, CNOT and MCZ are self-inverse, R_a(theta)^+ = R_a(-theta). Only odd integer
// scales are accepted; partial folding (non-integer lambda) is not implemented.
//
// ---------------------------------------------------------------------------------------
// Richardson extrapolation
//
// Measure E_k = E(lambda_k) at n distinct scales and evaluate at lambda = 0 the unique
// polynomial of degree n - 1 through the points (Lagrange form):
//
//     E(0) ~ sum_k c_k E_k,      c_k = prod_{j != k} lambda_j / (lambda_j - lambda_k).
//
// Properties: sum_k c_k = 1 and sum_k c_k lambda_k^m = 0 for m = 1..n-1, so every polynomial of
// degree <= n - 1 is extrapolated exactly. Scales {1, 3} give c = (3/2, -1/2); {1, 3, 5} give
// c = (15/8, -5/4, 3/8).
//
// Bias. If each gate fails independently with small probability, E(lambda) is a power series
// in lambda p, and n points cancel it through order (lambda p)^{n-1}. For single-qubit
// depolarizing noise with g gates the dependence is exactly exponential,
//
//     E(lambda) = E0 r^{g lambda} = E0 exp(-a lambda),      a = -g ln r,
//
// and the Lagrange remainder gives the n-point bias
//
//     E(0) - sum_k c_k E_k = E0 a^n exp(-a xi) (prod_k lambda_k) / n!,     0 < xi < lambda_max.
//
// Variance. The estimator variance is sum_k c_k^2 Var(E_k), so Richardson amplifies the
// statistical error by sqrt(sum c_k^2): 1.58 for {1, 3}, 2.29 for {1, 3, 5}. More scales cut
// the bias and raise the variance.
//
// Limitations. ZNE removes noise only to the extent the noise scales with the folded gate
// count. This header folds gates in a simulated depolarizing model, which is the idealised
// setting; on hardware the noise of U^+ need not equal that of U. Observables are exact
// expectations of the sampled state, so there is no shot noise beyond the trajectory noise.
//
// Test build (the LL_TEST block defines main, so do not define LL_TEST when including this
// header from another translation unit):
//   g++ -O2 -std=c++17 -DLL_TEST -x c++ src/Mitigation.h -o build/mitigation_test
// ---------------------------------------------------------------------------------------
#ifndef LL_MITIGATION_H
#define LL_MITIGATION_H

#include "Backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ll {

constexpr int MAX_ZNE_SCALES = 8;              // Richardson order cap (conditioning)
constexpr int MAX_ZNE_SCALE = 99;              // largest accepted fold scale 2k + 1
constexpr int DEFAULT_ZNE_TRAJECTORIES = 1000;

namespace zne_detail {
[[noreturn]] inline void fail_arg(const std::string& who, const std::string& msg) {
    throw std::invalid_argument(who + ": " + msg);
}
}  // namespace zne_detail

// ---------------------------------------------------------------------------
// NoisyBackend
// ---------------------------------------------------------------------------

// Decorator over any Backend: forwards each gate, folds it, and injects sampled depolarizing
// noise after every application. `inner` must outlive this object.
class NoisyBackend final : public Backend {
public:
    // p in [0, 1] is the total non-identity Pauli probability per noise site. Throws
    // std::invalid_argument otherwise (NaN included).
    NoisyBackend(Backend& inner, double p, std::uint64_t seed = 1)
        : inner_(inner), p_(p), folds_(0), rng_(seed), scratch_(1, 0), logical_(0), physical_(0) {
        if (!(p >= 0.0 && p <= 1.0)) zne_detail::fail_arg("NoisyBackend", "p must be in [0, 1]");
    }

    double probability() const { return p_; }
    Backend& inner() { return inner_; }
    void reseed(std::uint64_t seed) { rng_.seed(seed); }

    // Noise scale lambda = 2k + 1. Throws std::invalid_argument unless scale is odd and in
    // [1, MAX_ZNE_SCALE].
    void set_scale(int scale) {
        if (scale < 1 || scale > MAX_ZNE_SCALE || scale % 2 == 0)
            zne_detail::fail_arg("NoisyBackend::set_scale",
                                 "scale must be an odd integer in [1, " +
                                     std::to_string(MAX_ZNE_SCALE) + "]");
        folds_ = (scale - 1) / 2;
    }
    int scale() const { return 2 * folds_ + 1; }

    // Gates requested by the circuit, and gate applications actually executed (folds included),
    // since the last reset(). physical_gates() == scale() * logical_gates().
    long long logical_gates() const { return logical_; }
    long long physical_gates() const { return physical_; }

    // --- Backend interface ---
    void reset(int n_qubits) override {
        inner_.reset(n_qubits);
        logical_ = 0;
        physical_ = 0;
    }
    int n_qubits() const override { return inner_.n_qubits(); }

    void h(int q) override {
        const int qs[1] = {q};
        fold([&] { inner_.h(q); }, [&] { inner_.h(q); }, qs, 1);
    }
    void x(int q) override {
        const int qs[1] = {q};
        fold([&] { inner_.x(q); }, [&] { inner_.x(q); }, qs, 1);
    }
    void rx(int q, double t) override {
        const int qs[1] = {q};
        fold([&] { inner_.rx(q, t); }, [&] { inner_.rx(q, -t); }, qs, 1);
    }
    void ry(int q, double t) override {
        const int qs[1] = {q};
        fold([&] { inner_.ry(q, t); }, [&] { inner_.ry(q, -t); }, qs, 1);
    }
    void rz(int q, double t) override {
        const int qs[1] = {q};
        fold([&] { inner_.rz(q, t); }, [&] { inner_.rz(q, -t); }, qs, 1);
    }
    void cnot(int control, int target) override {
        const int qs[2] = {control, target};
        fold([&] { inner_.cnot(control, target); }, [&] { inner_.cnot(control, target); }, qs, 2);
    }
    void mcz(const std::vector<int>& qubits) override {
        fold([&] { inner_.mcz(qubits); }, [&] { inner_.mcz(qubits); }, qubits.data(),
             static_cast<int>(qubits.size()));
    }

    std::vector<double> probabilities() const override { return inner_.probabilities(); }
    std::vector<std::complex<double>> state() const override { return inner_.state(); }

private:
    // U (U^+ U)^k with a noise site after every application. fwd() runs first, so an invalid
    // gate throws from the wrapped backend before any noise is drawn or counted.
    template <class F, class G>
    void fold(F&& fwd, G&& inv, const int* qs, int nq) {
        fwd();
        noise(qs, nq);
        for (int f = 0; f < folds_; ++f) {
            inv();
            noise(qs, nq);
            fwd();
            noise(qs, nq);
        }
        ++logical_;
        physical_ += 1 + 2LL * folds_;
    }

    void noise(const int* qs, int nq) {
        if (p_ == 0.0) return;
        if (nq == 1)
            depolarize1(qs[0]);
        else if (nq == 2)
            depolarize2(qs[0], qs[1]);
        else
            for (int i = 0; i < nq; ++i) depolarize1(qs[i]);
    }

    // Uniform double in [0, 1) from 53 random bits (portable, unlike std::uniform_real_distribution).
    double u01() { return static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0); }

    void depolarize1(int q) {
        if (u01() >= p_) return;
        pauli(q, 1 + static_cast<int>(u01() * 3.0));
    }

    // One of the 15 non-identity two-qubit Paulis: idx = 1..15, low two bits act on a, high on b.
    void depolarize2(int a, int b) {
        if (u01() >= p_) return;
        const int idx = 1 + static_cast<int>(u01() * 15.0);
        pauli(a, idx & 3);
        pauli(b, idx >> 2);
    }

    // 0 = I, 1 = X, 2 = Y, 3 = Z. Y = i X Z, the global phase is irrelevant.
    void pauli(int q, int which) {
        switch (which) {
            case 1:
                inner_.x(q);
                break;
            case 2:
                scratch_[0] = q;
                inner_.mcz(scratch_);
                inner_.x(q);
                break;
            case 3:
                scratch_[0] = q;
                inner_.mcz(scratch_);
                break;
            default:
                break;
        }
    }

    Backend& inner_;
    double p_;
    int folds_;
    std::mt19937_64 rng_;
    std::vector<int> scratch_;
    long long logical_;
    long long physical_;
};

// ---------------------------------------------------------------------------
// ZNE
// ---------------------------------------------------------------------------

using CircuitFn = std::function<void(Backend&)>;
using ObservableFn = std::function<double(const Backend&)>;

struct ZNEResult {
    double value = 0.0;               // E(0) = sum_k c_k E_k
    double std_error = 0.0;           // sqrt(sum_k c_k^2 se_k^2); NaN if not estimable (see below)
    std::vector<double> scales;       // lambda_k
    std::vector<double> values;       // E_k, trajectory mean at each scale
    std::vector<double> std_errors;   // se_k = sample std / sqrt(N)
    std::vector<double> weights;      // c_k
    int trajectories = 0;             // trajectories actually run per scale
};

class ZNE {
public:
    // trajectories >= 1 per scale. With p = 0 the run is deterministic and one trajectory is used.
    // Throws std::invalid_argument otherwise.
    explicit ZNE(int trajectories = DEFAULT_ZNE_TRAJECTORIES) : trajectories_(trajectories) {
        if (trajectories < 1) zne_detail::fail_arg("ZNE", "trajectories must be >= 1");
    }

    int trajectories() const { return trajectories_; }

    static std::vector<double> default_scales() { return {1.0, 3.0, 5.0}; }

    // Scales accepted by extrapolate: 2..MAX_ZNE_SCALES distinct odd integers in [1, MAX_ZNE_SCALE].
    // Throws std::invalid_argument otherwise.
    static void validate_scales(const std::vector<double>& scales) {
        const char* who = "ZNE::validate_scales";
        if (scales.size() < 2 || scales.size() > static_cast<std::size_t>(MAX_ZNE_SCALES))
            zne_detail::fail_arg(who, "need between 2 and " + std::to_string(MAX_ZNE_SCALES) +
                                          " noise scales");
        for (double s : scales)
            if (!std::isfinite(s) || s < 1.0 || s > MAX_ZNE_SCALE || s != std::floor(s) ||
                std::fmod(s, 2.0) != 1.0)
                zne_detail::fail_arg(who, "each scale must be an odd integer in [1, " +
                                              std::to_string(MAX_ZNE_SCALE) + "]");
        for (std::size_t i = 0; i < scales.size(); ++i)
            for (std::size_t j = i + 1; j < scales.size(); ++j)
                if (scales[i] == scales[j]) zne_detail::fail_arg(who, "scales must be distinct");
    }

    // c_k = prod_{j != k} lambda_j / (lambda_j - lambda_k). Any finite, distinct, nonzero
    // scales are accepted (not only odd integers). Throws std::invalid_argument otherwise.
    static std::vector<double> richardson_weights(const std::vector<double>& scales) {
        const char* who = "ZNE::richardson_weights";
        if (scales.size() < 2) zne_detail::fail_arg(who, "need at least 2 scales");
        for (double s : scales)
            if (!std::isfinite(s) || s == 0.0) zne_detail::fail_arg(who, "scales must be finite and nonzero");
        const std::size_t n = scales.size();
        std::vector<double> c(n, 1.0);
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t j = 0; j < n; ++j) {
                if (j == k) continue;
                if (scales[j] == scales[k]) zne_detail::fail_arg(who, "scales must be distinct");
                c[k] *= scales[j] / (scales[j] - scales[k]);
            }
        return c;
    }

    // sum_k c_k values_k. Throws std::invalid_argument on a size mismatch or non-finite value.
    static double richardson(const std::vector<double>& scales, const std::vector<double>& values) {
        if (values.size() != scales.size())
            zne_detail::fail_arg("ZNE::richardson", "scales and values differ in length");
        for (double v : values)
            if (!std::isfinite(v)) zne_detail::fail_arg("ZNE::richardson", "non-finite value");
        const std::vector<double> c = richardson_weights(scales);
        double e = 0.0;
        for (std::size_t k = 0; k < c.size(); ++k) e += c[k] * values[k];
        return e;
    }

    // Runs circuit_fn on `noisy` at each fold scale, `trajectories` times per scale, averages
    // observable_fn, and extrapolates to zero noise with Richardson weights.
    //
    // Before every trajectory the backend is reset to |0...0> at its current size (size it with
    // reset(n) first). circuit_fn must apply its gates to the Backend& it receives; the number
    // of executed gate applications is checked to equal scale * logical gates. The scale in
    // force on entry is restored on return, also when an exception is thrown.
    //
    // std_error is NaN when p > 0 and trajectories == 1 (variance not estimable), 0 when p == 0.
    //
    // Throws std::invalid_argument for an empty callable, invalid scales, or a non-finite
    // observable; std::runtime_error if circuit_fn applied no gates or the folded gate count is
    // inconsistent. Errors from the wrapped backend (bad qubit, non-finite angle) propagate.
    ZNEResult extrapolate(NoisyBackend& noisy, const CircuitFn& circuit_fn,
                          const ObservableFn& observable_fn,
                          const std::vector<double>& scales = default_scales()) const {
        const char* who = "ZNE::extrapolate";
        if (!circuit_fn) zne_detail::fail_arg(who, "circuit_fn is empty");
        if (!observable_fn) zne_detail::fail_arg(who, "observable_fn is empty");
        validate_scales(scales);

        ScaleGuard guard(noisy);
        ZNEResult out;
        out.scales = scales;
        out.weights = richardson_weights(scales);
        const bool noisy_run = noisy.probability() > 0.0;
        const int n = noisy_run ? trajectories_ : 1;
        out.trajectories = n;

        for (double s : scales) {
            const int scale = static_cast<int>(s);
            noisy.set_scale(scale);
            double mean = 0.0, m2 = 0.0;
            for (int t = 0; t < n; ++t) {
                noisy.reset(noisy.n_qubits());
                circuit_fn(noisy);
                if (noisy.logical_gates() == 0)
                    throw std::runtime_error(std::string(who) + ": circuit_fn applied no gates to the backend");
                if (noisy.physical_gates() != static_cast<long long>(scale) * noisy.logical_gates())
                    throw std::runtime_error(std::string(who) + ": folded gate count is inconsistent");
                const double v = observable_fn(noisy);
                if (!std::isfinite(v)) zne_detail::fail_arg(who, "observable returned a non-finite value");
                const double d = v - mean;  // Welford
                mean += d / (t + 1);
                m2 += d * (v - mean);
            }
            double se = 0.0;
            if (n > 1)
                se = std::sqrt(std::max(0.0, m2 / (n - 1)) / n);
            else if (noisy_run)
                se = std::numeric_limits<double>::quiet_NaN();
            out.values.push_back(mean);
            out.std_errors.push_back(se);
        }

        double var = 0.0;
        for (std::size_t k = 0; k < scales.size(); ++k) {
            out.value += out.weights[k] * out.values[k];
            var += out.weights[k] * out.weights[k] * out.std_errors[k] * out.std_errors[k];
        }
        out.std_error = std::sqrt(var);
        return out;
    }

private:
    struct ScaleGuard {
        NoisyBackend& b;
        int saved;
        explicit ScaleGuard(NoisyBackend& nb) : b(nb), saved(nb.scale()) {}
        ~ScaleGuard() { b.set_scale(saved); }
        ScaleGuard(const ScaleGuard&) = delete;
        ScaleGuard& operator=(const ScaleGuard&) = delete;
    };

    int trajectories_;
};

}  // namespace ll

// ===========================================================================
// Validation
// ===========================================================================
#ifdef LL_TEST

#include <cstdio>

namespace {

struct Row {
    std::string name;
    double err;
    double tol;
};
std::vector<Row> g_rows;

void record(const std::string& name, double err, double tol) { g_rows.push_back({name, err, tol}); }

// Deviation in units of the standard error, tolerance 5 sigma. A NaN standard error fails.
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

    // Diagonal observables from the probability vector.
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

    // 3 qubits, 15 gates, every gate type including 1-, 2- and 3-qubit mcz.
    auto circ3 = [](Backend& b) {
        b.h(0); b.ry(1, 0.7); b.cnot(0, 2); b.rz(2, 0.3);
        b.rx(0, 0.9); b.ry(2, -0.4); b.mcz({0, 1}); b.mcz({0, 1, 2});
        b.x(1); b.h(2); b.rx(1, 1.3); b.cnot(1, 0); b.rz(0, 0.8); b.mcz({2}); b.ry(0, 0.5);
    };
    // 1 qubit, 3 gates: depolarizing noise gives exactly E(lambda) = z0 * r1^(3 lambda).
    auto circ1 = [](Backend& b) {
        b.ry(0, 1.1);
        b.rz(0, 0.7);
        b.rx(0, 0.4);
    };

    try {
        // ---- 1. Noiseless: folding is the identity, ZNE returns the exact value ----------------
        std::printf("== 1. Noiseless (p = 0): 3 qubits, 15 gates, scales {1, 3, 5} ==\n");
        {
            ReferenceBackend id3(3);
            circ3(id3);
            const double e_ideal = obs3(id3);

            ReferenceBackend in3(3);
            NoisyBackend nb(in3, 0.0);
            const ZNE zne(5);
            const ZNEResult r = zne.extrapolate(nb, circ3, obs3, {1.0, 3.0, 5.0});
            std::printf("ideal = %.14f, ZNE = %.14f, trajectories used = %d\n", e_ideal, r.value,
                        r.trajectories);
            std::printf("%8s %18s %12s\n", "scale", "E(scale)", "weight");
            double w_vals = 0.0;
            for (std::size_t k = 0; k < r.scales.size(); ++k) {
                std::printf("%8.0f %18.14f %12.6f\n", r.scales[k], r.values[k], r.weights[k]);
                w_vals = std::max(w_vals, std::abs(r.values[k] - e_ideal));
            }

            nb.set_scale(5);
            nb.reset(3);
            circ3(nb);
            const std::vector<std::complex<double>> a = nb.state(), b = id3.state();
            double w_amp = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) w_amp = std::max(w_amp, std::abs(a[i] - b[i]));
            const double w_cnt = std::abs(static_cast<double>(nb.physical_gates()) - 5.0 * 15.0) +
                                 std::abs(static_cast<double>(nb.logical_gates()) - 15.0);
            std::printf("scale-5 state vs ideal: max |amp diff| = %.3e, gates logical/physical = %lld/%lld\n",
                        w_amp, nb.logical_gates(), nb.physical_gates());

            record("1a |ZNE - ideal|", std::abs(r.value - e_ideal), 1e-10);
            record("1b max_k |E(lambda_k) - ideal|", w_vals, 1e-12);
            record("1c folded state vs ideal (scale 5), max |amp diff|", w_amp, 1e-12);
            record("1d gate counts: |physical - 5 logical| + |logical - 15|", w_cnt, 0.5);
        }

        // ---- 2. Two-point (linear) extrapolation vs analytic ------------------------------------
        std::printf("\n== 2. Two-point linear, 1 qubit, p = 0.02, scales {1, 3}, E(lam) = z0 r^(3 lam) ==\n");
        const double p1 = 0.02, r1 = 1.0 - 4.0 * p1 / 3.0;
        ReferenceBackend id1(1);
        circ1(id1);
        const double z0 = obsZ(id1);
        {
            ReferenceBackend in1(1);
            NoisyBackend nb1(in1, p1, 2024);
            const ZNE zne(60000);
            const ZNEResult r = zne.extrapolate(nb1, circ1, obsZ, {1.0, 3.0});
            std::printf("z0 = %.8f, r1 = %.8f, trajectories = %d\n", z0, r1, r.trajectories);
            std::printf("%8s %12s %10s %12s %8s\n", "scale", "E_mc", "se", "E_analytic", "dev/se");
            double w_pts = 0.0;
            std::vector<double> ana;
            for (std::size_t k = 0; k < r.scales.size(); ++k) {
                const double ek = z0 * std::pow(r1, 3.0 * r.scales[k]);
                ana.push_back(ek);
                const double d = sigma_dev(r.values[k], r.std_errors[k], ek);
                w_pts = std::max(w_pts, d);
                std::printf("%8.0f %12.6f %10.2e %12.6f %8.2f\n", r.scales[k], r.values[k],
                            r.std_errors[k], ek, d);
            }
            const double e0_ana = (3.0 * ana[0] - 1.0 * ana[1]) / (3.0 - 1.0);
            const double dev = sigma_dev(r.value, r.std_error, e0_ana);
            const double err_raw = std::abs(r.values[0] - z0), err_zne = std::abs(r.value - z0);
            std::printf("ideal %.6f | unmitigated %.6f (err %.4f) | ZNE %.6f +- %.1e (err %.4f) | "
                        "analytic Richardson %.6f\n",
                        z0, r.values[0], err_raw, r.value, r.std_error, err_zne, e0_ana);
            record("2a E(lambda) vs z0 r^(3 lambda), max sigma", w_pts, NSIG);
            record("2b two-point value vs analytic Richardson, sigma", dev, NSIG);
            record("2c mitigated error below unmitigated (0 = yes)", std::max(0.0, err_zne - err_raw), 0.0);
        }

        // ---- 3. Three-point (quadratic) extrapolation ----------------------------------------------
        std::printf("\n== 3. Three-point quadratic, scales {1, 3, 5} ==\n");
        {
            ReferenceBackend in1(1);
            NoisyBackend nb1(in1, p1, 77);
            const ZNE zne(60000);
            const ZNEResult r = zne.extrapolate(nb1, circ1, obsZ, {1.0, 3.0, 5.0});
            const double c[3] = {15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0};
            double e0_ana = 0.0, w_pts = 0.0;
            std::printf("%8s %12s %10s %12s %8s\n", "scale", "E_mc", "se", "E_analytic", "dev/se");
            for (std::size_t k = 0; k < 3; ++k) {
                const double ek = z0 * std::pow(r1, 3.0 * r.scales[k]);
                e0_ana += c[k] * ek;
                const double d = sigma_dev(r.values[k], r.std_errors[k], ek);
                w_pts = std::max(w_pts, d);
                std::printf("%8.0f %12.6f %10.2e %12.6f %8.2f\n", r.scales[k], r.values[k],
                            r.std_errors[k], ek, d);
            }
            const double dev = sigma_dev(r.value, r.std_error, e0_ana);
            const double err_raw = std::abs(r.values[0] - z0), err_zne = std::abs(r.value - z0);
            std::printf("ideal %.6f | unmitigated err %.4f | ZNE %.6f +- %.1e (err %.4f) | "
                        "analytic Richardson %.6f (bias %.1e)\n",
                        z0, err_raw, r.value, r.std_error, err_zne, e0_ana, std::abs(e0_ana - z0));

            // Exactness of the Richardson algebra, no sampling involved.
            const std::vector<double> s3 = {1.0, 3.0, 5.0}, s4 = {1.0, 3.0, 5.0, 7.0}, s2 = {1.0, 3.0};
            const double a = 0.3, b = -0.07, q = 0.004;
            std::vector<double> yq, yl;
            for (double s : s3) yq.push_back(a + b * s + q * s * s);
            for (double s : s2) yl.push_back(a + b * s);
            const double e_quad = std::abs(ZNE::richardson(s3, yq) - a);
            const double e_lin = std::abs(ZNE::richardson(s2, yl) - a);
            const std::vector<double> w3 = ZNE::richardson_weights(s3), w4 = ZNE::richardson_weights(s4);
            const double hand3[3] = {15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0};
            const double hand4[4] = {2.1875, -2.1875, 1.3125, -0.3125};
            double e_w = 0.0, e_mom = std::abs([&] { double s = 0; for (double x : w4) s += x; return s; }() - 1.0);
            for (int k = 0; k < 3; ++k) e_w = std::max(e_w, std::abs(w3[k] - hand3[k]));
            for (int k = 0; k < 4; ++k) e_w = std::max(e_w, std::abs(w4[k] - hand4[k]));
            for (int m = 1; m <= 3; ++m) {
                double mom = 0.0;
                for (int k = 0; k < 4; ++k) mom += w4[k] * std::pow(s4[k], m);
                e_mom = std::max(e_mom, std::abs(mom));
            }
            std::printf("synthetic a + b x + c x^2 at {1,3,5}: |E(0) - a| = %.2e; linear at {1,3}: %.2e\n",
                        e_quad, e_lin);
            std::printf("weights {1,3,5} = (%.6f, %.6f, %.6f); max |w - hand| = %.2e; "
                        "max |sum c_k lam^m - delta_m0|, m = 0..3 (4 points) = %.2e\n",
                        w3[0], w3[1], w3[2], e_w, e_mom);

            record("3a E(lambda) vs z0 r^(3 lambda), max sigma", w_pts, NSIG);
            record("3b three-point value vs analytic Richardson, sigma", dev, NSIG);
            record("3c mitigated error below unmitigated (0 = yes)", std::max(0.0, err_zne - err_raw), 0.0);
            record("3d exact on quadratic data (3 points) / linear (2 points)", std::max(e_quad, e_lin), 1e-12);
            record("3e weights vs hand values, 3 and 4 points", e_w, 1e-13);
            record("3f moment conditions: sum c = 1, sum c lam^m = 0", e_mom, 1e-10);
        }

        // ---- 4. Input validation ---------------------------------------------------------------------
        std::printf("\n== 4. Input checks ==\n");
        {
            ReferenceBackend in1(1);
            NoisyBackend nb1(in1, p1, 1);
            const ZNE zne(100);
            int n_exc = 0, bad = 0;
            auto chk = [&](bool ok) {
                ++n_exc;
                bad += !ok;
            };
            auto run = [&](const std::vector<double>& s) { zne.extrapolate(nb1, circ1, obsZ, s); };
            chk(throws<std::invalid_argument>([&] { run({1.0}); }));
            chk(throws<std::invalid_argument>([&] { run({}); }));
            chk(throws<std::invalid_argument>([&] { run({1.0, 3.0, 3.0}); }));
            chk(throws<std::invalid_argument>([&] { run({1.0, 4.0}); }));
            chk(throws<std::invalid_argument>([&] { run({1.0, 2.5}); }));
            chk(throws<std::invalid_argument>([&] { run({-1.0, 1.0}); }));
            chk(throws<std::invalid_argument>([&] { run({1.0, nan}); }));
            chk(throws<std::invalid_argument>([&] { run({1.0, 101.0}); }));
            chk(throws<std::invalid_argument>([&] { run({1, 3, 5, 7, 9, 11, 13, 15, 17}); }));
            chk(throws<std::invalid_argument>([&] {
                zne.extrapolate(nb1, circ1, [nan](const Backend&) { return nan; });
            }));
            chk(throws<std::invalid_argument>([&] { zne.extrapolate(nb1, CircuitFn{}, obsZ); }));
            chk(throws<std::invalid_argument>([&] { zne.extrapolate(nb1, circ1, ObservableFn{}); }));
            chk(throws<std::runtime_error>([&] { zne.extrapolate(nb1, [](Backend&) {}, obsZ); }));
            chk(throws<std::invalid_argument>([&] { ZNE bad_zne(0); }));
            chk(throws<std::invalid_argument>([&] { NoisyBackend b(in1, -0.1); }));
            chk(throws<std::invalid_argument>([&] { NoisyBackend b(in1, 1.2); }));
            chk(throws<std::invalid_argument>([&] { NoisyBackend b(in1, nan); }));
            chk(throws<std::invalid_argument>([&] { nb1.set_scale(2); }));
            chk(throws<std::invalid_argument>([&] { nb1.set_scale(0); }));
            chk(throws<std::invalid_argument>([&] { nb1.h(5); }));
            chk(throws<std::invalid_argument>([&] { nb1.rx(0, nan); }));
            chk(throws<std::invalid_argument>([&] { ZNE::richardson({1.0, 3.0}, {1.0}); }));
            chk(throws<std::invalid_argument>([&] { ZNE::richardson({1.0, 3.0}, {1.0, nan}); }));
            chk(throws<std::invalid_argument>([&] { ZNE::richardson_weights({2.0, 2.0}); }));
            std::printf("input-contract failures: %d / %d\n", bad, n_exc);
            record("4  expected exceptions not thrown", bad, 0.5);
        }

        // ---- 5. Two-qubit convention: Bell circuit, E(lambda) = r2^lambda ---------------------------------
        std::printf("\n== 5. Two-qubit depolarizing: H(0) CNOT(0,1), <ZZ> = r2^lambda, r2 = 1 - 16p/15 ==\n");
        {
            const double p2 = 0.03, r2 = 1.0 - 16.0 * p2 / 15.0;
            ReferenceBackend in2(2);
            NoisyBackend nb2(in2, p2, 99);
            const ZNE zne(40000);
            auto bell = [](Backend& b) {
                b.h(0);
                b.cnot(0, 1);
            };
            const ZNEResult r = zne.extrapolate(nb2, bell, obsZZ, {1.0, 3.0});
            std::printf("r2 = %.6f\n%8s %12s %10s %12s %8s\n", r2, "scale", "<ZZ>_mc", "se", "r2^lam", "dev/se");
            double w = 0.0;
            for (std::size_t k = 0; k < 2; ++k) {
                const double ex = std::pow(r2, r.scales[k]);
                const double d = sigma_dev(r.values[k], r.std_errors[k], ex);
                w = std::max(w, d);
                std::printf("%8.0f %12.6f %10.2e %12.6f %8.2f\n", r.scales[k], r.values[k], r.std_errors[k], ex, d);
            }
            record("5  <ZZ>(lambda) vs (1 - 16p/15)^lambda, max sigma", w, NSIG);
        }

        // ---- 6. Reproducibility and state restoration -------------------------------------------------------
        std::printf("\n== 6. Reproducibility and scale restoration ==\n");
        {
            ReferenceBackend in1(1);
            NoisyBackend nb1(in1, p1, 5);
            const ZNE zne(2000);
            nb1.set_scale(3);
            nb1.reseed(31);
            const ZNEResult a = zne.extrapolate(nb1, circ1, obsZ);
            const bool kept = nb1.scale() == 3;
            nb1.reseed(31);
            const ZNEResult b = zne.extrapolate(nb1, circ1, obsZ);
            const double d_rep = std::abs(a.value - b.value);
            try {
                zne.extrapolate(nb1, circ1, [nan](const Backend&) { return nan; });
            } catch (const std::invalid_argument&) {
            }
            const bool kept_exc = nb1.scale() == 3;
            std::printf("same seed: |dE| = %.1e; scale restored after run: %s, after exception: %s\n", d_rep,
                        kept ? "yes" : "NO", kept_exc ? "yes" : "NO");
            record("6a same seed gives identical result", d_rep, 0.0);
            record("6b scale restored (0 = yes)", (kept && kept_exc) ? 0.0 : 1.0, 0.5);
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

#endif  // LL_TEST

#endif  // LL_MITIGATION_H