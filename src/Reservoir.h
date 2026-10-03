// src/Reservoir.h
//
// Quantum reservoir computing (QRC): a fixed N-qubit quantum system plus a trained linear readout.
// Header only. Units are dimensionless: H in 1/time, tau in time, so only H*tau matters.
//
// ---------------------------------------------------------------------------------------
// Reservoir (Fujii-Nakajima / Martinez-Pena, time-multiplexed)
//
//     H = sum_{i<j} J_ij X_i X_j + sum_i h_i Z_i,
//     J_ij = coupling * u,  h_i = field * (1 + field_disorder * u'),  u, u' ~ U[-1, 1], fixed by seed.
//
// For each input s_t in [0, 1]:
//   1. Injection (CPTP map). Qubit 0 is replaced by the pure state
//          |psi_s> = sqrt(1-s) |0> + sqrt(s) |1>,
//      so rho -> |psi_s><psi_s| (x) Tr_0(rho). Information about earlier inputs survives only
//      in qubits 1..N-1 and their correlations, which is where the fading memory comes from.
//   2. Evolution for time tau split into V sub-steps, rho -> U rho U^+ with
//          U = exp(-i H tau / V)   (eigendecomposition of the real symmetric H).
//   3. After each sub-step ("virtual node" v = 1..V) record the expectation values
//          Z_i, X_i, Z_i Z_j, X_i X_j   (any subset, see Observable).
//      Expectations are exact (no shot noise). They are the features of step t:
//          phi_t in R^m,  m = V * (observables per measurement).
//
// Basis convention: |b> = |q_{N-1} ... q_0>, qubit i is bit i of b, Z|0> = +|0>.
// Dense rho costs O(d^3) per sub-step, d = 2^N, so MAX_RESERVOIR_QUBITS is 8.
//
// ---------------------------------------------------------------------------------------
// Readout (ridge regression)
//
// With Phi in R^{T x m} the feature matrix and y the target, the fit is
//     w* = (Phi_c^T Phi_c + lambda I)^{-1} Phi_c^T y_c,    b* = mean(y) - mean(Phi) . w*,
// where _c is column-centred (the bias is not penalised) and lambda = ridge * tr(Phi_c^T Phi_c) / m.
// For ridge -> 0 and full column rank this is w* = (Phi^T Phi)^{-1} Phi^T y. The ridge term
// keeps collinear features (e.g. virtual nodes that measure a conserved quantity) well posed.
//
// ---------------------------------------------------------------------------------------
// Memory capacity (Jaeger 2002), i.i.d. input s_t ~ U[0, 1]:
//     MC_k = squared Pearson correlation(y_hat_k, s_{t-k}) on held-out data,  MC = sum_k MC_k,
// where y_hat_k is the readout trained on target s_{t-k}. Linear features obey MC <= m.
//
// Single-qubit reference (N = 1): replace-injection erases the state each step, so MC_k = 0
// for k >= 1. If Z is measured and H commutes with Z (H = 0 or H = h Z), <Z> = 1 - 2 s_t at
// every virtual node, so MC_0 = 1 exactly. Hence MC >= 1 for any reservoir that includes
// this readout, which is the lower bound validated below.
//
// Mackey-Glass: dx/dt = beta x(t-tau) / (1 + x(t-tau)^n) - gamma x, standard chaotic
// regime tau = 17, beta = 0.2, gamma = 0.1, n = 10. Fixed point x* = (beta/gamma - 1)^{1/n} = 1.
//
// Test build (the LL_TEST block defines main, so do not define LL_TEST when including this
// header from another translation unit such as Reservoir.cpp):
//   g++ -O2 -std=c++17 -DLL_TEST -I/usr/include/eigen3 -x c++ src/Reservoir.h -o build/reservoir_test
// ---------------------------------------------------------------------------------------
#ifndef LL_RESERVOIR_H
#define LL_RESERVOIR_H

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ll {

// Same aliases as Noise.h / Transmon.h.
using cplx = std::complex<double>;
using CMatrix = Eigen::MatrixXcd;
using CVector = Eigen::VectorXcd;

constexpr int MAX_RESERVOIR_QUBITS = 8;
constexpr double DEFAULT_RIDGE = 1e-8;  // relative to mean diagonal of the centred Gram matrix

namespace detail {
[[noreturn]] inline void fail_arg(const std::string& who, const std::string& msg) {
    throw std::invalid_argument(who + ": " + msg);
}
inline bool finite_positive(double x) { return std::isfinite(x) && x > 0.0; }
}  // namespace detail

// ---------------------------------------------------------------------------
// Reservoir
// ---------------------------------------------------------------------------

enum class Observable { Z, X, ZZ, XX };  // Z_i, X_i (N each); Z_iZ_j, X_iX_j (N(N-1)/2 each, i < j)

inline int observable_count(Observable o, int n) {
    return (o == Observable::Z || o == Observable::X) ? n : n * (n - 1) / 2;
}

struct ReservoirConfig {
    int n_qubits = 4;               // [1, MAX_RESERVOIR_QUBITS]
    double coupling = 1.0;          // J scale, >= 0
    double field = 1.0;             // h scale, >= 0
    double field_disorder = 0.5;    // relative spread of h_i, in [0, 1]
    double tau = 1.0;               // evolution time per input, > 0
    int virtual_nodes = 4;          // V >= 1 measurements per input
    std::vector<Observable> observables{Observable::Z};
    std::uint64_t seed = 1;         // fixes J_ij and h_i
};

class QuantumReservoir {
public:
    // Throws std::invalid_argument for any out-of-range parameter, an empty observable list,
    // or a list that yields zero features (e.g. only ZZ with one qubit).
    // Throws std::runtime_error if the eigensolver does not converge.
    explicit QuantumReservoir(const ReservoirConfig& cfg) : cfg_(cfg), n_obs_(0) {
        validate();
        for (Observable o : cfg_.observables) n_obs_ += observable_count(o, cfg_.n_qubits);
        if (n_obs_ == 0) detail::fail_arg("QuantumReservoir", "observables yield zero features");
        build_hamiltonian();
        build_unitary();
        reset();
    }

    const ReservoirConfig& config() const { return cfg_; }
    int n_qubits() const { return cfg_.n_qubits; }
    int dim() const { return 1 << cfg_.n_qubits; }
    int n_observables() const { return n_obs_; }                    // per measurement
    int n_features() const { return n_obs_ * cfg_.virtual_nodes; }  // per input
    const Eigen::MatrixXd& hamiltonian() const { return h_; }
    const Eigen::MatrixXd& couplings() const { return j_; }         // symmetric, zero diagonal
    const Eigen::VectorXd& fields() const { return fields_; }
    const CMatrix& step_unitary() const { return u_; }              // exp(-i H tau / V)
    const CMatrix& state() const { return rho_; }

    // rho = |0...0><0...0|.
    void reset() {
        rho_ = CMatrix::Zero(dim(), dim());
        rho_(0, 0) = 1.0;
    }

    // Expectation values of cfg.observables in the current state, in list order
    // (Z_0..Z_{N-1}, then X_0.., then pairs (0,1),(0,2),...,(N-2,N-1)).
    Eigen::VectorXd measure() const {
        const int n = cfg_.n_qubits, d = dim();
        Eigen::VectorXd p(d);
        for (int b = 0; b < d; ++b) p(b) = rho_(b, b).real();
        auto sgn = [](int b, int i) { return 1.0 - 2.0 * ((b >> i) & 1); };
        // Tr(rho X_mask) = sum_b <b| rho |b ^ mask>
        auto xexp = [&](int mask) {
            double acc = 0.0;
            for (int b = 0; b < d; ++b) acc += rho_(b, b ^ mask).real();
            return acc;
        };
        Eigen::VectorXd out(n_obs_);
        int k = 0;
        for (Observable o : cfg_.observables) {
            switch (o) {
                case Observable::Z:
                    for (int i = 0; i < n; ++i) {
                        double a = 0.0;
                        for (int b = 0; b < d; ++b) a += p(b) * sgn(b, i);
                        out(k++) = a;
                    }
                    break;
                case Observable::X:
                    for (int i = 0; i < n; ++i) out(k++) = xexp(1 << i);
                    break;
                case Observable::ZZ:
                    for (int i = 0; i < n; ++i)
                        for (int j = i + 1; j < n; ++j) {
                            double a = 0.0;
                            for (int b = 0; b < d; ++b) a += p(b) * sgn(b, i) * sgn(b, j);
                            out(k++) = a;
                        }
                    break;
                case Observable::XX:
                    for (int i = 0; i < n; ++i)
                        for (int j = i + 1; j < n; ++j) out(k++) = xexp((1 << i) | (1 << j));
                    break;
            }
        }
        return out;
    }

    // Inject s in [0, 1], evolve, return the n_features() features of this input
    // (virtual node v occupies entries [v * n_observables(), (v+1) * n_observables())).
    // Throws std::invalid_argument if s is outside [0, 1] or NaN.
    Eigen::VectorXd step(double s) {
        if (!(s >= 0.0 && s <= 1.0))
            detail::fail_arg("QuantumReservoir::step", "input must be in [0, 1]");
        inject(s);
        Eigen::VectorXd f(n_features());
        for (int v = 0; v < cfg_.virtual_nodes; ++v) {
            rho_ = u_ * rho_ * u_.adjoint();
            f.segment(v * n_obs_, n_obs_) = measure();
        }
        return f;
    }

    // reset(), then one step per input. Returns T x n_features(), row t = features of input t.
    Eigen::MatrixXd run(const Eigen::VectorXd& inputs) {
        reset();
        Eigen::MatrixXd f(inputs.size(), n_features());
        for (Eigen::Index k = 0; k < inputs.size(); ++k) f.row(k) = step(inputs(k)).transpose();
        return f;
    }

private:
    void validate() const {
        const char* who = "QuantumReservoir";
        if (cfg_.n_qubits < 1 || cfg_.n_qubits > MAX_RESERVOIR_QUBITS)
            detail::fail_arg(who, "n_qubits must be in [1, " + std::to_string(MAX_RESERVOIR_QUBITS) + "]");
        if (!(std::isfinite(cfg_.coupling) && cfg_.coupling >= 0.0))
            detail::fail_arg(who, "coupling must be finite and >= 0");
        if (!(std::isfinite(cfg_.field) && cfg_.field >= 0.0))
            detail::fail_arg(who, "field must be finite and >= 0");
        if (!(cfg_.field_disorder >= 0.0 && cfg_.field_disorder <= 1.0))
            detail::fail_arg(who, "field_disorder must be in [0, 1]");
        if (!detail::finite_positive(cfg_.tau)) detail::fail_arg(who, "tau must be finite and > 0");
        if (cfg_.virtual_nodes < 1) detail::fail_arg(who, "virtual_nodes must be >= 1");
        if (cfg_.observables.empty()) detail::fail_arg(who, "observables is empty");
    }

    void build_hamiltonian() {
        const int n = cfg_.n_qubits, d = dim();
        std::mt19937_64 rng(cfg_.seed);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        fields_.resize(n);
        for (int i = 0; i < n; ++i) fields_(i) = cfg_.field * (1.0 + cfg_.field_disorder * uni(rng));
        j_ = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < n; ++i)
            for (int k = i + 1; k < n; ++k) j_(i, k) = j_(k, i) = cfg_.coupling * uni(rng);

        h_ = Eigen::MatrixXd::Zero(d, d);
        for (int b = 0; b < d; ++b) {
            double diag = 0.0;
            for (int i = 0; i < n; ++i) diag += fields_(i) * (1.0 - 2.0 * ((b >> i) & 1));
            h_(b, b) = diag;
        }
        // X_i X_j |b> = |b ^ mask>; iterating over all b fills both triangles.
        for (int i = 0; i < n; ++i)
            for (int k = i + 1; k < n; ++k) {
                const int mask = (1 << i) | (1 << k);
                for (int b = 0; b < d; ++b) h_(b ^ mask, b) += j_(i, k);
            }
    }

    void build_unitary() {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(h_);
        if (es.info() != Eigen::Success)
            throw std::runtime_error("QuantumReservoir: eigensolver did not converge");
        const double dt = cfg_.tau / cfg_.virtual_nodes;
        CVector ph(dim());
        for (int k = 0; k < dim(); ++k) ph(k) = std::exp(cplx(0.0, -es.eigenvalues()(k) * dt));
        const CMatrix v = es.eigenvectors().cast<cplx>();
        u_ = v * ph.asDiagonal() * v.adjoint();
    }

    // rho -> |psi_s><psi_s| (x) Tr_0(rho). With qubit 0 as the lowest bit, index = 2a + p.
    void inject(double s) {
        const int r = dim() / 2;
        CMatrix rest(r, r);
        for (int a = 0; a < r; ++a)
            for (int b = 0; b < r; ++b) rest(a, b) = rho_(2 * a, 2 * b) + rho_(2 * a + 1, 2 * b + 1);
        const double c = std::sqrt(1.0 - s), sn = std::sqrt(s);
        const double sg[2][2] = {{c * c, c * sn}, {c * sn, sn * sn}};
        for (int a = 0; a < r; ++a)
            for (int b = 0; b < r; ++b)
                for (int p = 0; p < 2; ++p)
                    for (int q = 0; q < 2; ++q) rho_(2 * a + p, 2 * b + q) = rest(a, b) * sg[p][q];
    }

    ReservoirConfig cfg_;
    int n_obs_;
    Eigen::VectorXd fields_;
    Eigen::MatrixXd j_;
    Eigen::MatrixXd h_;
    CMatrix u_;
    CMatrix rho_;
};

// ---------------------------------------------------------------------------
// Linear readout
// ---------------------------------------------------------------------------

class LinearReadout {
public:
    // ridge >= 0 and finite, relative to the mean diagonal of the centred Gram matrix.
    explicit LinearReadout(double ridge = DEFAULT_RIDGE) : ridge_(ridge), b_(0.0), fitted_(false) {
        if (!(std::isfinite(ridge) && ridge >= 0.0))
            detail::fail_arg("LinearReadout", "ridge must be finite and >= 0");
    }

    // f: T x m features, y: T targets. Throws std::invalid_argument on size mismatch,
    // T < 2, m < 1 or non-finite data, and std::runtime_error if the Gram matrix is singular.
    void fit(const Eigen::MatrixXd& f, const Eigen::VectorXd& y) {
        const char* who = "LinearReadout::fit";
        if (f.rows() != y.size()) detail::fail_arg(who, "rows of features and length of target differ");
        if (f.rows() < 2 || f.cols() < 1) detail::fail_arg(who, "need at least 2 samples and 1 feature");
        if (!f.allFinite() || !y.allFinite()) detail::fail_arg(who, "non-finite data");

        const Eigen::VectorXd mu = f.colwise().mean().transpose();
        const double my = y.mean();
        const Eigen::MatrixXd fc = f.rowwise() - mu.transpose();
        Eigen::VectorXd yc = y;
        yc.array() -= my;

        Eigen::MatrixXd g = fc.transpose() * fc;
        g.diagonal().array() += ridge_ * g.trace() / static_cast<double>(g.cols());
        Eigen::LDLT<Eigen::MatrixXd> ldlt(g);
        const Eigen::VectorXd dv = ldlt.vectorD();
        if (ldlt.info() != Eigen::Success || !(dv.minCoeff() > 1e-14 * dv.maxCoeff()))
            throw std::runtime_error(std::string(who) + ": singular Gram matrix; increase ridge");
        w_ = ldlt.solve(fc.transpose() * yc);
        b_ = my - mu.dot(w_);
        fitted_ = true;
    }

    // y_hat = Phi w + b. Throws std::logic_error before fit, std::invalid_argument on width mismatch.
    Eigen::VectorXd predict(const Eigen::MatrixXd& f) const {
        if (!fitted_) throw std::logic_error("LinearReadout::predict: fit() has not been called");
        if (f.cols() != w_.size()) detail::fail_arg("LinearReadout::predict", "feature width mismatch");
        Eigen::VectorXd out = f * w_;
        out.array() += b_;
        return out;
    }

    const Eigen::VectorXd& weights() const { return w_; }
    double bias() const { return b_; }
    double ridge() const { return ridge_; }

private:
    double ridge_;
    Eigen::VectorXd w_;
    double b_;
    bool fitted_;
};

// sum (pred - target)^2 / sum (target - mean(target))^2.
inline double nmse(const Eigen::VectorXd& pred, const Eigen::VectorXd& target) {
    if (pred.size() != target.size() || target.size() < 2)
        detail::fail_arg("nmse", "need equal sizes >= 2");
    const double den = (target.array() - target.mean()).square().sum();
    if (!(den > 0.0)) detail::fail_arg("nmse", "target is constant");
    return (pred - target).squaredNorm() / den;
}

// Squared Pearson correlation; 0 if either input is constant (no information).
inline double squared_correlation(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    if (a.size() != b.size() || a.size() < 2) detail::fail_arg("squared_correlation", "need equal sizes >= 2");
    const Eigen::ArrayXd ca = a.array() - a.mean(), cb = b.array() - b.mean();
    const double saa = ca.square().sum(), sbb = cb.square().sum(), sab = (ca * cb).sum();
    if (!(saa > 0.0 && sbb > 0.0)) return 0.0;
    return sab * sab / (saa * sbb);
}

// Affine map of x onto [0, 1] using its own min and max. Throws if x is constant or non-finite.
inline Eigen::VectorXd scale_to_unit(const Eigen::VectorXd& x) {
    if (x.size() < 2 || !x.allFinite()) detail::fail_arg("scale_to_unit", "need >= 2 finite samples");
    const double lo = x.minCoeff(), hi = x.maxCoeff();
    if (!(hi > lo)) detail::fail_arg("scale_to_unit", "series is constant");
    Eigen::VectorXd out = x;
    out.array() = (out.array() - lo) / (hi - lo);
    return out;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

struct MemoryCapacityResult {
    Eigen::VectorXd mc;  // MC_k, k = 0..max_delay
    double total = 0.0;  // sum_k MC_k
    int n_features = 0;  // upper bound on the true total, up to finite-sample bias
};

// Short-term memory task. Drives `res` (reset first) with i.i.d. s_t ~ U[0, 1], trains one
// readout per delay k on n_train rows after the washout, scores MC_k on the next n_test rows.
// Throws std::invalid_argument if max_delay < 0, washout < 0, n_test < 2 or n_train <= m + 1.
inline MemoryCapacityResult memory_capacity(QuantumReservoir& res, int max_delay, int n_train,
                                            int n_test, int washout = 100,
                                            std::uint64_t seed = 12345, double ridge = DEFAULT_RIDGE) {
    const char* who = "memory_capacity";
    const int m = res.n_features();
    if (max_delay < 0) detail::fail_arg(who, "max_delay must be >= 0");
    if (washout < 0) detail::fail_arg(who, "washout must be >= 0");
    if (n_test < 2) detail::fail_arg(who, "n_test must be >= 2");
    if (n_train <= m + 1) detail::fail_arg(who, "n_train must exceed n_features + 1");

    const int start = std::max(washout, max_delay);
    const int len = start + n_train + n_test;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Eigen::VectorXd s(len);
    for (int t = 0; t < len; ++t) s(t) = u(rng);

    const Eigen::MatrixXd f = res.run(s);
    const Eigen::MatrixXd ftr = f.middleRows(start, n_train);
    const Eigen::MatrixXd fte = f.middleRows(start + n_train, n_test);

    MemoryCapacityResult out;
    out.mc.resize(max_delay + 1);
    out.n_features = m;
    LinearReadout ro(ridge);
    for (int k = 0; k <= max_delay; ++k) {
        Eigen::VectorXd ytr(n_train), yte(n_test);
        for (int i = 0; i < n_train; ++i) ytr(i) = s(start + i - k);
        for (int i = 0; i < n_test; ++i) yte(i) = s(start + n_train + i - k);
        ro.fit(ftr, ytr);
        out.mc(k) = squared_correlation(ro.predict(fte), yte);
        out.total += out.mc(k);
    }
    return out;
}

// Mackey-Glass series, RK4 with step h = sample_dt / substeps and a constant history x0.
// The delay is evaluated on the grid, with the half-step value linearly interpolated, so
// tau / h must be an integer. `transient` samples are discarded, then n are returned.
// Throws std::invalid_argument for non-positive sizes or parameters or a non-integer tau / h.
inline Eigen::VectorXd mackey_glass(int n, double tau = 17.0, double beta = 0.2, double gamma = 0.1,
                                    double exponent = 10.0, double x0 = 1.2, int transient = 200,
                                    double sample_dt = 1.0, int substeps = 10) {
    const char* who = "mackey_glass";
    if (n < 1 || transient < 0 || substeps < 1) detail::fail_arg(who, "n >= 1, transient >= 0, substeps >= 1");
    if (!detail::finite_positive(tau) || !detail::finite_positive(sample_dt) ||
        !detail::finite_positive(exponent) || !detail::finite_positive(x0) ||
        !(std::isfinite(beta) && beta >= 0.0) || !(std::isfinite(gamma) && gamma >= 0.0))
        detail::fail_arg(who, "invalid parameter");
    const double h = sample_dt / substeps;
    const int delay = static_cast<int>(std::lround(tau / h));
    if (delay < 1 || std::abs(tau / h - delay) > 1e-9)
        detail::fail_arg(who, "tau must be an integer multiple of sample_dt / substeps");

    const long long total = static_cast<long long>(transient + n) * substeps;
    // x[i] is the value at t = (i - delay) h, so x[delay + k] = x(t_k) and x[k] = x(t_k - tau).
    std::vector<double> x(static_cast<std::size_t>(delay + 1 + total), x0);
    auto f = [&](double xc, double xd) { return beta * xd / (1.0 + std::pow(xd, exponent)) - gamma * xc; };
    for (long long k = 0; k < total; ++k) {
        const double xc = x[delay + k];
        const double xd0 = x[k], xd1 = x[k + 1], xdm = 0.5 * (xd0 + xd1);
        const double k1 = f(xc, xd0);
        const double k2 = f(xc + 0.5 * h * k1, xdm);
        const double k3 = f(xc + 0.5 * h * k2, xdm);
        const double k4 = f(xc + h * k3, xd1);
        x[delay + k + 1] = xc + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    Eigen::VectorXd out(n);
    for (int j = 0; j < n; ++j) out(j) = x[delay + static_cast<long long>(transient + j + 1) * substeps];
    return out;
}

struct PredictionResult {
    double nmse_train = 0.0;
    double nmse_test = 0.0;
    double nmse_persistence = 0.0;  // test NMSE of the baseline x_{t+h} = x_t
    Eigen::VectorXd target_test;
    Eigen::VectorXd prediction_test;
};

// h-step-ahead prediction of a scalar series. The series is min-max scaled to [0, 1] (whole-series
// range, no target information beyond its extent) and fed to the reservoir. Row t of the features
// (input x_t) is regressed onto x_{t+horizon}. Rows [washout, washout + n_train) train, the rest test.
// Throws std::invalid_argument if horizon or washout < 0, n_train <= m + 1, or fewer than 2 test rows remain.
inline PredictionResult predict_series(QuantumReservoir& res, const Eigen::VectorXd& series, int horizon,
                                       int washout, int n_train, double ridge = DEFAULT_RIDGE) {
    const char* who = "predict_series";
    if (horizon < 0 || washout < 0) detail::fail_arg(who, "horizon and washout must be >= 0");
    if (n_train <= res.n_features() + 1) detail::fail_arg(who, "n_train must exceed n_features + 1");
    const Eigen::Index rows = series.size() - horizon;
    const Eigen::Index n_test = rows - washout - n_train;
    if (n_test < 2) detail::fail_arg(who, "series too short for washout + n_train + 2 test rows");

    const Eigen::VectorXd s = scale_to_unit(series);
    const Eigen::VectorXd in = s.head(rows);
    const Eigen::MatrixXd f = res.run(in);
    const Eigen::MatrixXd ftr = f.middleRows(washout, n_train);
    const Eigen::MatrixXd fte = f.middleRows(washout + n_train, n_test);
    const Eigen::VectorXd ytr = series.segment(washout + horizon, n_train);
    const Eigen::VectorXd yte = series.segment(washout + n_train + horizon, n_test);

    LinearReadout ro(ridge);
    ro.fit(ftr, ytr);
    PredictionResult out;
    out.prediction_test = ro.predict(fte);
    out.target_test = yte;
    out.nmse_train = nmse(ro.predict(ftr), ytr);
    out.nmse_test = nmse(out.prediction_test, yte);
    out.nmse_persistence = nmse(series.segment(washout + n_train, n_test), yte);
    return out;
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
    const double nan = std::numeric_limits<double>::quiet_NaN();
    try {
        // ---- 1. Single-qubit memory capacity ----------------------------------------
        std::printf("== 1. Single qubit, Z readout: MC_0 = 1, MC_k = 0 for k >= 1 ==\n");
        struct Case {
            const char* name;
            double field;
            int v;
        };
        const Case cases[] = {{"H = 0,      V = 1", 0.0, 1}, {"H = 1.3 Z,  V = 3", 1.3, 3}};
        double w1a = 0.0, w1b = 0.0;
        std::printf("%-18s %8s %8s %8s %8s %8s %8s %8s\n", "case", "MC_0", "MC_1", "MC_2", "MC_3", "MC_4",
                    "MC_5", "total");
        for (const Case& c : cases) {
            ReservoirConfig cfg;
            cfg.n_qubits = 1;
            cfg.coupling = 0.0;
            cfg.field = c.field;
            cfg.field_disorder = 0.0;
            cfg.tau = 1.0;
            cfg.virtual_nodes = c.v;
            QuantumReservoir res(cfg);
            const MemoryCapacityResult mc = memory_capacity(res, 5, 2000, 2000, 100, 7);
            std::printf("%-18s", c.name);
            for (int k = 0; k <= 5; ++k) std::printf(" %8.5f", mc.mc(k));
            std::printf(" %8.5f\n", mc.total);
            w1a = std::max(w1a, std::abs(mc.mc(0) - 1.0));
            for (int k = 1; k <= 5; ++k) w1b = std::max(w1b, mc.mc(k));
        }
        record("1a |MC_0 - 1| (N = 1, H commutes with Z)", w1a, 1e-6);
        record("1b max MC_k, k >= 1 (N = 1, finite-sample noise)", w1b, 0.02);

        // ---- 2. Closed-form dynamics ---------------------------------------------------
        std::printf("\n== 2. Closed-form dynamics (injection, H, virtual-node timing) ==\n");
        double w2a = 0.0, w2b = 0.0, w2c = 0.0;
        {
            // N = 1, H = h Z: <X>(t) = 2 sqrt(s(1-s)) cos(2 h t)
            ReservoirConfig cfg;
            cfg.n_qubits = 1;
            cfg.coupling = 0.0;
            cfg.field = 0.9;
            cfg.field_disorder = 0.0;
            cfg.tau = 1.5;
            cfg.virtual_nodes = 4;
            cfg.observables = {Observable::X};
            QuantumReservoir r(cfg);
            const double s = 0.3;
            const Eigen::VectorXd f = r.step(s);
            std::printf("%3s %14s %14s\n", "v", "<X> numeric", "<X> exact");
            for (int v = 0; v < 4; ++v) {
                const double ex = 2.0 * std::sqrt(s * (1.0 - s)) * std::cos(2.0 * 0.9 * (1.5 / 4) * (v + 1));
                std::printf("%3d %14.10f %14.10f\n", v + 1, f(v), ex);
                w2a = std::max(w2a, std::abs(f(v) - ex));
            }
        }
        {
            // N = 2, H = J X0 X1, input on qubit 0, qubit 1 in |0>:
            //   <Z1>(t) = cos(2 J t),  <Z0>(t) = (1 - 2s) cos(2 J t)
            ReservoirConfig cfg;
            cfg.n_qubits = 2;
            cfg.coupling = 1.3;
            cfg.field = 0.0;
            cfg.field_disorder = 0.0;
            cfg.tau = 2.1;
            cfg.virtual_nodes = 3;
            cfg.observables = {Observable::Z};
            QuantumReservoir r(cfg);
            const double J = r.couplings()(0, 1), s = 0.37;
            const Eigen::VectorXd f = r.step(s);
            std::printf("\nJ_01 = %.6f\n%3s %12s %12s %12s %12s\n", J, "v", "Z0 num", "Z0 exact", "Z1 num",
                        "Z1 exact");
            for (int v = 0; v < 3; ++v) {
                const double c = std::cos(2.0 * J * (v + 1) * cfg.tau / 3.0);
                std::printf("%3d %12.8f %12.8f %12.8f %12.8f\n", v + 1, f(2 * v), (1.0 - 2.0 * s) * c,
                            f(2 * v + 1), c);
                w2b = std::max({w2b, std::abs(f(2 * v) - (1.0 - 2.0 * s) * c), std::abs(f(2 * v + 1) - c)});
            }
        }
        {
            // N = 2, H = 0: features [Z0, Z1, X0, X1] = [1-2s, 1, 2 sqrt(s(1-s)), 0]
            ReservoirConfig cfg;
            cfg.n_qubits = 2;
            cfg.coupling = 0.0;
            cfg.field = 0.0;
            cfg.field_disorder = 0.0;
            cfg.virtual_nodes = 1;
            cfg.observables = {Observable::Z, Observable::X};
            std::printf("\n%8s %10s %10s %10s %10s\n", "s", "Z0", "Z1", "X0", "X1");
            for (double s : {0.0, 0.25, 0.8, 1.0}) {
                QuantumReservoir r(cfg);
                const Eigen::VectorXd f = r.step(s);
                std::printf("%8.2f %10.6f %10.6f %10.6f %10.6f\n", s, f(0), f(1), f(2), f(3));
                w2c = std::max({w2c, std::abs(f(0) - (1.0 - 2.0 * s)), std::abs(f(1) - 1.0),
                                std::abs(f(2) - 2.0 * std::sqrt(s * (1.0 - s))), std::abs(f(3))});
            }
        }
        record("2a N=1 H=hZ: <X>(t) vs closed form", w2a, 1e-12);
        record("2b N=2 H=J XX: <Z0>, <Z1> vs closed form", w2b, 1e-12);
        record("2c injection + bit ordering (H = 0)", w2c, 1e-12);

        // ---- 3. Linear regression on a known target -----------------------------------------
        std::printf("\n== 3. Ridge readout on y = Phi w + b (T = 300, m = 6) ==\n");
        double w3a = 0.0, w3b = 0.0, w3c = 0.0;
        {
            std::mt19937_64 rng(42);
            std::normal_distribution<double> g(0.0, 1.0);
            const int T = 300, m = 6;
            Eigen::VectorXd wt(m);
            wt << 1.0, -2.0, 0.5, 3.0, -1.0, 0.25;
            const double bt = 0.7;
            Eigen::MatrixXd f(T, m), ft(500, m);
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < m; ++j) f(i, j) = g(rng);
            for (int i = 0; i < 500; ++i)
                for (int j = 0; j < m; ++j) ft(i, j) = g(rng);
            Eigen::VectorXd y = f * wt;
            y.array() += bt;
            Eigen::VectorXd yt = ft * wt;
            yt.array() += bt;

            LinearReadout exact(1e-14);
            exact.fit(f, y);
            w3a = std::max((exact.weights() - wt).cwiseAbs().maxCoeff(), std::abs(exact.bias() - bt));
            LinearReadout def;  // default ridge
            def.fit(f, y);
            w3b = (def.predict(ft) - yt).cwiseAbs().maxCoeff();
            std::printf("ridge 1e-14: max |w - w_true| = %.3e, |b - b_true| = %.3e\n",
                        (exact.weights() - wt).cwiseAbs().maxCoeff(), std::abs(exact.bias() - bt));
            std::printf("ridge %.0e: max held-out prediction error = %.3e\n", DEFAULT_RIDGE, w3b);

            // Collinear features: ridge keeps the fit well posed and exact on the span.
            Eigen::MatrixXd fc(T, 3);
            for (int i = 0; i < T; ++i) fc.row(i).setConstant(f(i, 0));
            Eigen::VectorXd yc = f.col(0);
            yc.array() *= 2.0;
            LinearReadout rc;
            rc.fit(fc, yc);
            w3c = (rc.predict(fc) - yc).cwiseAbs().maxCoeff();
            std::printf("collinear columns: max fit error = %.3e\n", w3c);
        }
        {
            // End-to-end: trivial reservoir (N = 1, H = 0, Z) reproduces its own input, horizon 0.
            ReservoirConfig cfg;
            cfg.n_qubits = 1;
            cfg.coupling = 0.0;
            cfg.field = 0.0;
            cfg.field_disorder = 0.0;
            cfg.virtual_nodes = 1;
            QuantumReservoir r(cfg);
            Eigen::VectorXd ser(1000);
            for (int t = 0; t < 1000; ++t) ser(t) = std::sin(0.1 * t) + 0.3 * std::cos(0.37 * t);
            const PredictionResult pr = predict_series(r, ser, 0, 10, 300);
            std::printf("predict_series, identity reservoir: NMSE train %.3e, test %.3e\n", pr.nmse_train,
                        pr.nmse_test);
            record("3d predict_series identity NMSE (test)", pr.nmse_test, 1e-10);
        }
        record("3a weights/bias vs truth (ridge 1e-14)", w3a, 1e-9);
        record("3b held-out error (default ridge)", w3b, 1e-6);
        record("3c collinear features: fit error", w3c, 1e-7);

        // ---- 4. Invariants of a driven run ----------------------------------------------------------
        std::printf("\n== 4. Invariants: N = 3, Z + XX readout, V = 4, 300 inputs ==\n");
        {
            ReservoirConfig cfg;
            cfg.n_qubits = 3;
            cfg.coupling = 2.0;
            cfg.field = 1.0;
            cfg.tau = 2.0;
            cfg.virtual_nodes = 4;
            cfg.observables = {Observable::Z, Observable::XX};
            cfg.seed = 5;
            QuantumReservoir res(cfg);
            std::mt19937_64 rng(3);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            double w_tr = 0.0, w_h = 0.0, lmin = 1e9, fmax = 0.0;
            for (int t = 0; t < 300; ++t) {
                const Eigen::VectorXd f = res.step(u(rng));
                fmax = std::max(fmax, f.cwiseAbs().maxCoeff());
                const CMatrix& r = res.state();
                w_tr = std::max(w_tr, std::abs(r.trace() - cplx(1.0, 0.0)));
                w_h = std::max(w_h, (r - r.adjoint()).cwiseAbs().maxCoeff());
                Eigen::SelfAdjointEigenSolver<CMatrix> es(r, Eigen::EigenvaluesOnly);
                lmin = std::min(lmin, es.eigenvalues()(0));
            }
            const double e_unit =
                (res.step_unitary().adjoint() * res.step_unitary() - CMatrix::Identity(8, 8))
                    .cwiseAbs().maxCoeff();
            const double e_sym = (res.hamiltonian() - res.hamiltonian().transpose()).cwiseAbs().maxCoeff();
            std::printf("n_features = %d, max |Tr - 1| = %.3e, max |rho - rho^+| = %.3e\n", res.n_features(),
                        w_tr, w_h);
            std::printf("min eigenvalue = %.3e, max |feature| = %.6f, |U^+U - 1| = %.3e, |H - H^T| = %.1e\n",
                        lmin, fmax, e_unit, e_sym);
            record("4a trace preservation max |Tr rho - 1|", w_tr, 1e-12);
            record("4b Hermiticity max |rho - rho^+|", w_h, 1e-12);
            record("4c positivity: max(0, -min eigenvalue)", std::max(0.0, -lmin), 1e-10);
            record("4d features bounded: max(0, max|f| - 1)", std::max(0.0, fmax - 1.0), 1e-9);
            record("4e unitarity |U^+U - 1|", e_unit, 1e-12);
            record("4f H symmetric", e_sym, 1e-14);
        }

        // ---- 5. Memory beyond one qubit -----------------------------------------------------------------
        std::printf("\n== 5. Memory capacity, N = 4, V = 4, Z readout (m = 16) ==\n");
        {
            ReservoirConfig cfg;
            cfg.n_qubits = 4;
            cfg.coupling = 2.0;
            cfg.field = 1.0;
            cfg.tau = 2.0;
            cfg.virtual_nodes = 4;
            cfg.seed = 11;
            QuantumReservoir res(cfg);
            const MemoryCapacityResult mc = memory_capacity(res, 10, 2000, 2000, 100, 21);
            std::printf("%4s %10s\n", "k", "MC_k");
            for (int k = 0; k <= 10; ++k) std::printf("%4d %10.5f\n", k, mc.mc(k));
            std::printf("total = %.4f (upper bound m = %d)\n", mc.total, mc.n_features);
            record("5a MC_total exceeds single-qubit value 1.1 (0 = yes)", std::max(0.0, 1.1 - mc.total), 0.0);
            record("5b MC_total <= n_features (+ finite-sample slack)", std::max(0.0, mc.total - mc.n_features),
                   0.05);
        }

        // ---- 6. Mackey-Glass generator ------------------------------------------------------------------------
        std::printf("\n== 6. Mackey-Glass: fixed point and attractor range ==\n");
        {
            const Eigen::VectorXd fp = mackey_glass(100, 17.0, 0.2, 0.1, 10.0, 1.0, 50);
            const double e_fp = (fp.array() - 1.0).abs().maxCoeff();
            const Eigen::VectorXd mg = mackey_glass(2000);
            const double lo = mg.minCoeff(), hi = mg.maxCoeff();
            std::printf("x0 = x* = 1: max |x - 1| = %.3e\n", e_fp);
            std::printf("x0 = 1.2: range [%.4f, %.4f], mean %.4f\n", lo, hi, mg.mean());
            record("6a fixed point x* = 1 preserved", e_fp, 1e-12);
            record("6b attractor inside [0.2, 1.8]", std::max({0.0, 0.2 - lo, hi - 1.8}), 0.0);

            // Informational: 1-step-ahead prediction (no pass/fail).
            ReservoirConfig cfg;
            cfg.n_qubits = 4;
            cfg.coupling = 2.0;
            cfg.field = 1.0;
            cfg.tau = 2.0;
            cfg.virtual_nodes = 5;
            cfg.observables = {Observable::Z, Observable::XX};
            cfg.seed = 11;
            QuantumReservoir res(cfg);
            const PredictionResult pr = predict_series(res, mackey_glass(1600), 1, 100, 1000);
            std::printf("1-step prediction (m = %d): NMSE train %.3e, test %.3e, persistence %.3e\n",
                        res.n_features(), pr.nmse_train, pr.nmse_test, pr.nmse_persistence);
        }

        // ---- 7. Input contract ---------------------------------------------------------------------------------------
        std::printf("\n== 7. Input checks ==\n");
        int n_exc = 0, bad = 0;
        auto chk = [&](bool ok) {
            ++n_exc;
            bad += !ok;
        };
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.n_qubits = 0; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] {
            ReservoirConfig c; c.n_qubits = MAX_RESERVOIR_QUBITS + 1; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.tau = 0.0; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.virtual_nodes = 0; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.observables.clear(); QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] {
            ReservoirConfig c; c.n_qubits = 1; c.observables = {Observable::ZZ}; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.coupling = -1.0; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.field = nan; QuantumReservoir r(c); }));
        chk(throws<std::invalid_argument>([&] { ReservoirConfig c; c.field_disorder = 2.0; QuantumReservoir r(c); }));
        {
            ReservoirConfig c;
            c.n_qubits = 2;
            QuantumReservoir r(c);
            chk(throws<std::invalid_argument>([&] { r.step(-0.1); }));
            chk(throws<std::invalid_argument>([&] { r.step(1.1); }));
            chk(throws<std::invalid_argument>([&] { r.step(nan); }));
            chk(throws<std::invalid_argument>([&] { memory_capacity(r, 3, 5, 100); }));
            chk(throws<std::invalid_argument>([&] { predict_series(r, Eigen::VectorXd::LinSpaced(500, 0, 1), -1, 10, 100); }));
        }
        chk(throws<std::invalid_argument>([&] { LinearReadout r(-1.0); }));
        chk(throws<std::invalid_argument>([&] {
            LinearReadout r; r.fit(Eigen::MatrixXd::Random(10, 2), Eigen::VectorXd::Zero(9)); }));
        chk(throws<std::logic_error>([&] { LinearReadout r; r.predict(Eigen::MatrixXd::Zero(3, 2)); }));
        chk(throws<std::runtime_error>([&] {
            LinearReadout r; r.fit(Eigen::MatrixXd::Ones(10, 2), Eigen::VectorXd::LinSpaced(10, 0, 1)); }));
        chk(throws<std::invalid_argument>([&] { scale_to_unit(Eigen::VectorXd::Ones(5)); }));
        chk(throws<std::invalid_argument>([&] { mackey_glass(10, 17.05); }));
        chk(throws<std::invalid_argument>([&] { mackey_glass(0); }));
        std::printf("input-contract failures: %d / %d\n", bad, n_exc);
        record("7  expected exceptions not thrown", bad, 0.5);
    } catch (const std::exception& e) {
        std::printf("\nUNEXPECTED EXCEPTION: %s\n", e.what());
        return 1;
    }

    // ---- Summary ------------------------------------------------------------------
    std::printf("\n== Summary ==\n%-56s %12s %12s  %s\n", "check", "max_err", "tol", "");
    bool ok = true;
    for (const Row& r : g_rows) {
        const bool pass = r.err <= r.tol;
        ok = ok && pass;
        std::printf("%-56s %12.3e %12.3e  %s\n", r.name.c_str(), r.err, r.tol, pass ? "PASS" : "FAIL");
    }
    std::printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}

#endif  // LL_TEST

#endif  // LL_RESERVOIR_H