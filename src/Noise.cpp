#include "Noise.h"

#include <algorithm>
#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ll {

namespace {

// Below this many 2^m x 2^m blocks the OpenMP fork/join costs more than the work.
constexpr std::ptrdiff_t PARALLEL_MIN_BLOCKS = 4096;

constexpr int MAX_DEPOLARIZING_QUBITS = 2;
constexpr int NUM_PAULIS = 4;  // I, X, Y, Z

[[noreturn]] void fail(const std::string& msg) {
    throw std::invalid_argument(msg);
}

bool is_power_of_two(std::size_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

int log2_exact(std::size_t x) {
    int n = 0;
    while ((std::size_t{1} << n) < x) ++n;
    return n;
}

void check_qubit_index(int q, const char* who) {
    if (q < 0) fail(std::string(who) + ": qubit index must be non-negative");
}

void check_probability(double x, const char* who, const char* name) {
    // Written as a negated conjunction so NaN is rejected too.
    if (!(x >= 0.0 && x <= 1.0)) {
        fail(std::string(who) + ": " + name + " must lie in [0, 1], got " + std::to_string(x));
    }
}

void check_duration(double t, const char* who) {
    if (!(t >= 0.0) || !std::isfinite(t)) {
        fail(std::string(who) + ": duration must be finite and >= 0, got " + std::to_string(t));
    }
}

void check_distinct_targets(const std::vector<int>& targets, const char* who) {
    for (std::size_t i = 0; i < targets.size(); ++i) {
        check_qubit_index(targets[i], who);
        for (std::size_t j = i + 1; j < targets.size(); ++j) {
            if (targets[i] == targets[j]) fail(std::string(who) + ": repeated target qubit");
        }
    }
}

CMatrix kron(const CMatrix& A, const CMatrix& B) {
    CMatrix C(A.rows() * B.rows(), A.cols() * B.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }
    return C;
}

// 0 = I, 1 = X, 2 = Y, 3 = Z
CMatrix pauli(int k) {
    CMatrix P = CMatrix::Zero(2, 2);
    switch (k) {
        case 0: P(0, 0) = 1.0; P(1, 1) = 1.0; break;
        case 1: P(0, 1) = 1.0; P(1, 0) = 1.0; break;
        case 2: P(0, 1) = cplx(0.0, -1.0); P(1, 0) = cplx(0.0, 1.0); break;
        case 3: P(0, 0) = 1.0; P(1, 1) = -1.0; break;
        default: fail("pauli: index out of range");
    }
    return P;
}

// Pauli string whose digit j (base 4) acts on operator bit j, i.e. on
// targets()[j]. Bit 0 is the least significant, so the most significant
// digit is the outermost Kronecker factor.
CMatrix pauli_string(int code, int m) {
    CMatrix M = CMatrix::Identity(1, 1);
    for (int j = m - 1; j >= 0; --j) {
        const int digit = (code >> (2 * j)) & 3;
        M = kron(M, pauli(digit));
    }
    return M;
}

}  // namespace

// ---------------------------------------------------------------------------
// Density-matrix helpers
// ---------------------------------------------------------------------------

CMatrix density_from_statevector(const CVector& psi) {
    const std::size_t len = static_cast<std::size_t>(psi.size());
    if (!is_power_of_two(len)) {
        fail("density_from_statevector: length must be a power of two, got " + std::to_string(len));
    }
    if (log2_exact(len) > MAX_DM_QUBITS) {
        fail("density_from_statevector: more than MAX_DM_QUBITS qubits");
    }
    if (!psi.allFinite()) fail("density_from_statevector: non-finite amplitude");
    if (std::abs(psi.squaredNorm() - 1.0) > DEFAULT_TOL) {
        fail("density_from_statevector: state is not normalized");
    }
    return psi * psi.adjoint();
}

CMatrix ground_state_density(int n_qubits) {
    if (n_qubits < 0 || n_qubits > MAX_DM_QUBITS) {
        fail("ground_state_density: n_qubits must be in [0, " + std::to_string(MAX_DM_QUBITS) + "]");
    }
    const Eigen::Index dim = Eigen::Index{1} << n_qubits;
    CMatrix rho = CMatrix::Zero(dim, dim);
    rho(0, 0) = 1.0;
    return rho;
}

bool is_valid_density_matrix(const CMatrix& rho, double tol) {
    if (rho.rows() == 0 || rho.rows() != rho.cols()) return false;
    if (!rho.allFinite()) return false;
    if ((rho - rho.adjoint()).cwiseAbs().maxCoeff() > tol) return false;
    if (std::abs(rho.trace() - cplx(1.0, 0.0)) > tol) return false;

    const CMatrix herm = 0.5 * (rho + rho.adjoint());
    Eigen::SelfAdjointEigenSolver<CMatrix> solver(herm, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success) return false;
    return solver.eigenvalues().minCoeff() >= -tol;
}

// ---------------------------------------------------------------------------
// KrausChannel
// ---------------------------------------------------------------------------

KrausChannel::KrausChannel(std::vector<CMatrix> ops, std::vector<int> targets)
    : ops_(std::move(ops)), targets_(std::move(targets)) {
    constexpr const char* who = "KrausChannel";
    if (ops_.empty()) fail(std::string(who) + ": no Kraus operators");
    if (targets_.empty()) fail(std::string(who) + ": no target qubits");
    if (static_cast<int>(targets_.size()) > MAX_DM_QUBITS) {
        fail(std::string(who) + ": more targets than MAX_DM_QUBITS");
    }
    check_distinct_targets(targets_, who);

    const Eigen::Index d = Eigen::Index{1} << targets_.size();
    for (const CMatrix& K : ops_) {
        if (K.rows() != d || K.cols() != d) {
            fail(std::string(who) + ": operator is not 2^m x 2^m for m = " +
                 std::to_string(targets_.size()));
        }
        if (!K.allFinite()) fail(std::string(who) + ": non-finite operator entry");
    }
    if (!is_trace_preserving(DEFAULT_TOL)) {
        fail(std::string(who) + ": operators are not trace preserving");
    }
}

bool KrausChannel::is_trace_preserving(double tol) const {
    if (ops_.empty()) return false;
    const Eigen::Index d = ops_.front().rows();
    CMatrix S = CMatrix::Zero(d, d);
    for (const CMatrix& K : ops_) S += K.adjoint() * K;
    return (S - CMatrix::Identity(d, d)).cwiseAbs().maxCoeff() <= tol;
}

void KrausChannel::apply(CMatrix& rho, int n_qubits) const {
    constexpr const char* who = "KrausChannel::apply";
    if (ops_.empty()) fail(std::string(who) + ": channel is empty");
    if (n_qubits < 0 || n_qubits > MAX_DM_QUBITS) {
        fail(std::string(who) + ": n_qubits must be in [0, " + std::to_string(MAX_DM_QUBITS) + "]");
    }
    const Eigen::Index dim = Eigen::Index{1} << n_qubits;
    if (rho.rows() != dim || rho.cols() != dim) {
        fail(std::string(who) + ": rho is not 2^n x 2^n");
    }
    for (int t : targets_) {
        if (t >= n_qubits) fail(std::string(who) + ": target qubit outside the register");
    }

    const int m = num_targets();
    const Eigen::Index blk = Eigen::Index{1} << m;
    const std::size_t n_rest = std::size_t{1} << (n_qubits - m);

    // Positions of the bits that are not targets, in ascending order.
    std::vector<int> rest_pos;
    for (int q = 0; q < n_qubits; ++q) {
        if (std::find(targets_.begin(), targets_.end(), q) == targets_.end()) rest_pos.push_back(q);
    }

    // rest_base[r]: register index with the bits of r deposited at the
    // non-target positions and zeros at the target positions.
    std::vector<std::size_t> rest_base(n_rest, 0);
    for (std::size_t r = 0; r < n_rest; ++r) {
        std::size_t base = 0;
        for (std::size_t b = 0; b < rest_pos.size(); ++b) {
            if ((r >> b) & 1) base |= std::size_t{1} << rest_pos[b];
        }
        rest_base[r] = base;
    }

    // tgt_off[t]: register index with the m bits of t deposited at the target
    // positions, bit j of t going to targets_[j].
    std::vector<std::size_t> tgt_off(static_cast<std::size_t>(blk), 0);
    for (std::size_t t = 0; t < tgt_off.size(); ++t) {
        std::size_t off = 0;
        for (int j = 0; j < m; ++j) {
            if ((t >> j) & 1) off |= std::size_t{1} << targets_[j];
        }
        tgt_off[t] = off;
    }

    // The 4^(n-m) blocks (r1, r2) partition the entries of rho, so every entry
    // of out is written exactly once and threads never touch the same element.
    CMatrix out(dim, dim);
    const std::ptrdiff_t n_blocks = static_cast<std::ptrdiff_t>(n_rest * n_rest);

#pragma omp parallel if (n_blocks >= PARALLEL_MIN_BLOCKS)
    {
        CMatrix B(blk, blk), acc(blk, blk), tmp(blk, blk);
#pragma omp for schedule(static)
        for (std::ptrdiff_t idx = 0; idx < n_blocks; ++idx) {
            const std::size_t r1 = static_cast<std::size_t>(idx) / n_rest;
            const std::size_t r2 = static_cast<std::size_t>(idx) % n_rest;
            const std::size_t row0 = rest_base[r1];
            const std::size_t col0 = rest_base[r2];

            for (Eigen::Index a = 0; a < blk; ++a) {
                for (Eigen::Index b = 0; b < blk; ++b) {
                    B(a, b) = rho(static_cast<Eigen::Index>(row0 | tgt_off[a]),
                                  static_cast<Eigen::Index>(col0 | tgt_off[b]));
                }
            }

            // B -> sum_k K_k B K_k^dagger
            acc.setZero();
            for (const CMatrix& K : ops_) {
                tmp.noalias() = K * B;
                acc.noalias() += tmp * K.adjoint();
            }

            for (Eigen::Index a = 0; a < blk; ++a) {
                for (Eigen::Index b = 0; b < blk; ++b) {
                    out(static_cast<Eigen::Index>(row0 | tgt_off[a]),
                        static_cast<Eigen::Index>(col0 | tgt_off[b])) = acc(a, b);
                }
            }
        }
    }
    rho.swap(out);
}

KrausChannel KrausChannel::compose(const KrausChannel& other) const {
    if (ops_.empty() || other.ops_.empty()) fail("KrausChannel::compose: empty channel");
    if (targets_ != other.targets_) {
        fail("KrausChannel::compose: compose requires identical target sets");
    }
    // other(this(rho)) = sum_j sum_i (L_j K_i) rho (L_j K_i)^dagger
    std::vector<CMatrix> ops;
    ops.reserve(ops_.size() * other.ops_.size());
    for (const CMatrix& K : ops_) {
        for (const CMatrix& L : other.ops_) ops.emplace_back(L * K);
    }
    return KrausChannel(std::move(ops), targets_);
}

KrausChannel KrausChannel::amplitude_damping(double gamma, int qubit) {
    constexpr const char* who = "KrausChannel::amplitude_damping";
    check_probability(gamma, who, "gamma");
    check_qubit_index(qubit, who);

    CMatrix K0 = CMatrix::Zero(2, 2);
    CMatrix K1 = CMatrix::Zero(2, 2);
    K0(0, 0) = 1.0;
    K0(1, 1) = std::sqrt(std::max(0.0, 1.0 - gamma));
    K1(0, 1) = std::sqrt(gamma);
    return KrausChannel({K0, K1}, {qubit});
}

KrausChannel KrausChannel::phase_damping(double lambda, int qubit) {
    constexpr const char* who = "KrausChannel::phase_damping";
    check_probability(lambda, who, "lambda");
    check_qubit_index(qubit, who);

    CMatrix K0 = CMatrix::Zero(2, 2);
    CMatrix K1 = CMatrix::Zero(2, 2);
    K0(0, 0) = 1.0;
    K0(1, 1) = std::sqrt(std::max(0.0, 1.0 - lambda));
    K1(1, 1) = std::sqrt(lambda);
    return KrausChannel({K0, K1}, {qubit});
}

KrausChannel KrausChannel::identity(int qubit) {
    check_qubit_index(qubit, "KrausChannel::identity");
    return KrausChannel({CMatrix::Identity(2, 2)}, {qubit});
}

// ---------------------------------------------------------------------------
// DepolarizingChannel
// ---------------------------------------------------------------------------

DepolarizingChannel::DepolarizingChannel(double p, int num_qubits) : p_(p), m_(num_qubits) {
    check_probability(p, "DepolarizingChannel", "p");
    if (num_qubits < 1 || num_qubits > MAX_DEPOLARIZING_QUBITS) {
        fail("DepolarizingChannel: num_qubits must be 1 or 2");
    }
}

KrausChannel DepolarizingChannel::to_kraus(const std::vector<int>& targets) const {
    if (static_cast<int>(targets.size()) != m_) {
        fail("DepolarizingChannel::to_kraus: expected " + std::to_string(m_) + " target(s)");
    }
    check_distinct_targets(targets, "DepolarizingChannel::to_kraus");

    const int n_strings = 1 << (2 * m_);  // 4^m
    const double w_identity = std::sqrt(1.0 - p_);
    const double w_error = std::sqrt(p_ / static_cast<double>(n_strings - 1));

    std::vector<CMatrix> ops;
    ops.reserve(n_strings);
    for (int code = 0; code < n_strings; ++code) {
        const double w = (code == 0) ? w_identity : w_error;
        if (w > 0.0) ops.emplace_back(w * pauli_string(code, m_));
    }
    return KrausChannel(std::move(ops), targets);
}

// ---------------------------------------------------------------------------
// T1T2Model
// ---------------------------------------------------------------------------

T1T2Model::T1T2Model(double t1, double t2) : t1_(t1), t2_(t2) {
    if (!(t1 > 0.0)) fail("T1T2Model: T1 must be positive");
    if (!(t2 > 0.0)) fail("T1T2Model: T2 must be positive");
    if (t2 > 2.0 * t1 * (1.0 + T2_BOUND_RTOL)) {
        fail("T1T2Model: T2 > 2 T1 is unphysical (negative pure dephasing rate)");
    }
}

double T1T2Model::t_phi() const {
    const double inv_t2 = 1.0 / t2_;
    const double inv_tphi = inv_t2 - 0.5 / t1_;
    // At T2 = 2 T1 the difference is zero up to rounding.
    if (inv_tphi <= T2_BOUND_RTOL * inv_t2) return std::numeric_limits<double>::infinity();
    return 1.0 / inv_tphi;
}

double T1T2Model::gamma(double duration) const {
    check_duration(duration, "T1T2Model::gamma");
    return -std::expm1(-duration / t1_);
}

double T1T2Model::lambda(double duration) const {
    check_duration(duration, "T1T2Model::lambda");
    // sqrt(1 - lambda) = exp(-t / Tphi)  =>  lambda = 1 - exp(-2 t / Tphi)
    return -std::expm1(-2.0 * duration / t_phi());
}

KrausChannel T1T2Model::channel(double duration, int qubit) const {
    check_duration(duration, "T1T2Model::channel");
    check_qubit_index(qubit, "T1T2Model::channel");
    const KrausChannel relax = KrausChannel::amplitude_damping(gamma(duration), qubit);
    const KrausChannel dephase = KrausChannel::phase_damping(lambda(duration), qubit);
    return relax.compose(dephase);
}

double T1T2Model::excited_population(double duration) const {
    check_duration(duration, "T1T2Model::excited_population");
    return std::exp(-duration / t1_);
}

double T1T2Model::coherence_magnitude(double duration) const {
    check_duration(duration, "T1T2Model::coherence_magnitude");
    return std::exp(-duration / t2_);
}

// ---------------------------------------------------------------------------
// PurcellModel
// ---------------------------------------------------------------------------

PurcellModel::PurcellModel(double g, double delta, double kappa, double filter_suppression)
    : g_(g), delta_(delta), kappa_(kappa), filter_(filter_suppression) {
    constexpr const char* who = "PurcellModel";
    if (!(g >= 0.0) || !std::isfinite(g)) fail(std::string(who) + ": g must be finite and >= 0");
    if (!(std::abs(delta) > 0.0) || !std::isfinite(delta)) {
        fail(std::string(who) + ": delta must be finite and non-zero");
    }
    if (!(kappa >= 0.0) || !std::isfinite(kappa)) {
        fail(std::string(who) + ": kappa must be finite and >= 0");
    }
    if (!(filter_suppression > 0.0 && filter_suppression <= 1.0)) {
        fail(std::string(who) + ": filter suppression must lie in (0, 1]");
    }
    if (dispersive_ratio() > MAX_DISPERSIVE_RATIO) {
        fail(std::string(who) + ": g/|delta| too large for the dispersive approximation");
    }
}

double PurcellModel::dispersive_ratio() const {
    return g_ / std::abs(delta_);
}

double PurcellModel::rate() const {
    const double r = dispersive_ratio();
    return filter_ * kappa_ * r * r;
}

double PurcellModel::t1_limit() const {
    const double gp = rate();
    return gp > 0.0 ? 1.0 / gp : std::numeric_limits<double>::infinity();
}

PurcellModel PurcellModel::with_filter(double suppression) const {
    if (!(suppression > 0.0 && suppression <= 1.0)) {
        fail("PurcellModel::with_filter: suppression must lie in (0, 1]");
    }
    return PurcellModel(g_, delta_, kappa_, suppression);
}

PurcellModel PurcellModel::with_filter_db(double attenuation_db) const {
    if (!(attenuation_db >= 0.0) || !std::isfinite(attenuation_db)) {
        fail("PurcellModel::with_filter_db: attenuation must be finite and >= 0 dB");
    }
    return with_filter(std::pow(10.0, -attenuation_db / DB_POWER_DIVISOR));
}

KrausChannel PurcellModel::channel(double duration, int qubit) const {
    check_duration(duration, "PurcellModel::channel");
    check_qubit_index(qubit, "PurcellModel::channel");
    return KrausChannel::amplitude_damping(-std::expm1(-rate() * duration), qubit);
}

T1T2Model PurcellModel::combined_with(const T1T2Model& intrinsic) const {
    const double tphi = intrinsic.t_phi();
    const double inv_tphi = std::isinf(tphi) ? 0.0 : 1.0 / tphi;

    const double inv_t1_eff = 1.0 / intrinsic.t1() + rate();
    const double inv_t2_eff = 0.5 * inv_t1_eff + inv_tphi;

    // 1/0 evaluates to +inf for doubles, which T1T2Model accepts for T1 = T2 = inf.
    return T1T2Model(1.0 / inv_t1_eff, 1.0 / inv_t2_eff);
}

}  // namespace ll

// ---------------------------------------------------------------------------
// Validation: g++ -O2 -std=c++17 -fopenmp -DLL_TEST -I/usr/include/eigen3 \
//                 src/Noise.cpp -o noise_test && ./noise_test
// ---------------------------------------------------------------------------
#ifdef LL_TEST

#include <cstdio>
#include <random>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        const double va_ = (a), vb_ = (b);                                             \
        if (!(std::abs(va_ - vb_) <= (tol))) {                                         \
            ++g_failures;                                                              \
            std::fprintf(stderr, "FAIL %s:%d  %s = %.15g, %s = %.15g\n", __FILE__,     \
                         __LINE__, #a, va_, #b, vb_);                                  \
        }                                                                              \
    } while (0)

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

ll::CMatrix random_density(int n, std::mt19937& gen) {
    const int dim = 1 << n;
    std::normal_distribution<double> nd(0.0, 1.0);
    ll::CMatrix rho = ll::CMatrix::Zero(dim, dim);
    double wsum = 0.0;
    for (int k = 0; k < 3; ++k) {
        ll::CVector psi(dim);
        for (int i = 0; i < dim; ++i) psi(i) = ll::cplx(nd(gen), nd(gen));
        psi.normalize();
        const double w = 1.0 + k;
        rho += w * ll::density_from_statevector(psi);
        wsum += w;
    }
    return rho / wsum;
}

// Independent reference: embed each K into a full 2^n x 2^n matrix and
// multiply. Only used here, to check the bit-block algorithm.
ll::CMatrix reference_apply(const ll::KrausChannel& ch, const ll::CMatrix& rho, int n) {
    const int dim = 1 << n;
    const int m = ch.num_targets();
    int tmask = 0;
    for (int t : ch.targets()) tmask |= 1 << t;
    auto tbits = [&](int idx) {
        int v = 0;
        for (int j = 0; j < m; ++j) v |= ((idx >> ch.targets()[j]) & 1) << j;
        return v;
    };
    ll::CMatrix out = ll::CMatrix::Zero(dim, dim);
    for (const ll::CMatrix& K : ch.operators()) {
        ll::CMatrix F = ll::CMatrix::Zero(dim, dim);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                if ((i & ~tmask) == (j & ~tmask)) F(i, j) = K(tbits(i), tbits(j));
            }
        }
        out += F * rho * F.adjoint();
    }
    return out;
}

void check_against_reference(const ll::KrausChannel& ch, int n, std::mt19937& gen) {
    ll::CMatrix rho = random_density(n, gen);
    const ll::CMatrix want = reference_apply(ch, rho, n);
    ch.apply(rho, n);
    CHECK((rho - want).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(ll::is_valid_density_matrix(rho));
}

void test_index_manipulation() {
    std::mt19937 gen(12345);
    constexpr int n = 4;

    check_against_reference(ll::KrausChannel::amplitude_damping(0.3, 1), n, gen);
    check_against_reference(ll::KrausChannel::phase_damping(0.4, 3), n, gen);
    check_against_reference(ll::DepolarizingChannel(0.2, 1).to_kraus({0}), n, gen);
    // Non-adjacent targets in reversed order exercise the bit deposit tables.
    check_against_reference(ll::DepolarizingChannel(0.35, 2).to_kraus({3, 0}), n, gen);
    check_against_reference(ll::T1T2Model(50e-6, 70e-6).channel(30e-6, 2), n, gen);

    const ll::PurcellModel pm(6.283185307179586 * 100e6, 6.283185307179586 * 1.5e9,
                              6.283185307179586 * 5e6);
    check_against_reference(pm.channel(5e-6, 1), n, gen);
}

void test_t1_t2_analytic() {
    const double T1 = 50e-6, T2 = 70e-6, t = 30e-6;
    const ll::T1T2Model model(T1, T2);

    CHECK_NEAR(model.t_phi(), 1.0 / (1.0 / T2 - 0.5 / T1), 1e-18);

    ll::CMatrix one = ll::CMatrix::Zero(2, 2);
    one(1, 1) = 1.0;
    model.channel(t, 0).apply(one, 1);
    CHECK_NEAR(one(1, 1).real(), std::exp(-t / T1), 1e-12);
    CHECK_NEAR(one(0, 0).real(), 1.0 - std::exp(-t / T1), 1e-12);
    CHECK_NEAR(one(1, 1).real(), model.excited_population(t), 1e-12);

    ll::CMatrix plus = ll::CMatrix::Constant(2, 2, 0.5);
    model.channel(t, 0).apply(plus, 1);
    CHECK_NEAR(std::abs(plus(0, 1)), 0.5 * std::exp(-t / T2), 1e-12);
    CHECK_NEAR(std::abs(plus(0, 1)), 0.5 * model.coherence_magnitude(t), 1e-12);
    CHECK_NEAR(plus(0, 0).real(), 0.5 + 0.5 * model.gamma(t), 1e-12);

    // Semigroup: ten steps of t/10 equal one step of t.
    ll::CMatrix a = ll::CMatrix::Constant(2, 2, 0.5);
    ll::CMatrix b = a;
    const ll::KrausChannel step = model.channel(t / 10.0, 0);
    for (int i = 0; i < 10; ++i) step.apply(a, 1);
    model.channel(t, 0).apply(b, 1);
    CHECK((a - b).cwiseAbs().maxCoeff() < 1e-12);

    // T2 = 2 T1: no pure dephasing.
    const ll::T1T2Model limit(40e-6, 80e-6);
    CHECK(std::isinf(limit.t_phi()));
    CHECK_NEAR(limit.lambda(t), 0.0, 0.0);

    // No decoherence at all.
    const double inf = std::numeric_limits<double>::infinity();
    const ll::T1T2Model ideal(inf, inf);
    CHECK_NEAR(ideal.gamma(t), 0.0, 0.0);
    CHECK_NEAR(ideal.coherence_magnitude(t), 1.0, 0.0);

    CHECK(throws([] { ll::T1T2Model(10e-6, 25e-6); }));
    CHECK(throws([] { ll::T1T2Model(-1.0, 1.0); }));
    CHECK(throws([&] { model.gamma(-1.0); }));
}

void test_depolarizing() {
    // One qubit: Bloch vector shrinks by 1 - 4p/3.
    const double p = 0.2;
    ll::CMatrix rho = ll::ground_state_density(1);
    ll::DepolarizingChannel(p, 1).to_kraus({0}).apply(rho, 1);
    CHECK_NEAR((rho(0, 0) - rho(1, 1)).real(), 1.0 - 4.0 * p / 3.0, 1e-12);

    // Two qubits at p = 15/16 gives the maximally mixed state.
    ll::CMatrix r2 = ll::ground_state_density(2);
    ll::DepolarizingChannel(15.0 / 16.0, 2).to_kraus({0, 1}).apply(r2, 2);
    CHECK((r2 - ll::CMatrix::Identity(4, 4) * 0.25).cwiseAbs().maxCoeff() < 1e-12);

    // p = 0 is the identity, p = 1 drops the identity term and stays CPTP.
    CHECK(ll::DepolarizingChannel(0.0, 1).to_kraus({0}).operators().size() == 1);
    CHECK(ll::DepolarizingChannel(1.0, 2).to_kraus({0, 1}).is_trace_preserving());

    CHECK(throws([] { ll::DepolarizingChannel(-0.1, 1); }));
    CHECK(throws([] { ll::DepolarizingChannel(1.1, 1); }));
    CHECK(throws([] { ll::DepolarizingChannel(std::nan(""), 1); }));
    CHECK(throws([] { ll::DepolarizingChannel(0.1, 3); }));
    CHECK(throws([] { ll::DepolarizingChannel(0.1, 2).to_kraus({0}); }));
    CHECK(throws([] { ll::DepolarizingChannel(0.1, 2).to_kraus({1, 1}); }));
}

void test_purcell() {
    constexpr double two_pi = 6.283185307179586;
    const double g = two_pi * 100e6, delta = two_pi * 1.5e9, kappa = two_pi * 5e6;
    const ll::PurcellModel pm(g, delta, kappa);

    const double expect = kappa * (g / delta) * (g / delta);
    CHECK_NEAR(pm.rate() / expect, 1.0, 1e-14);
    CHECK_NEAR(pm.t1_limit() * pm.rate(), 1.0, 1e-14);

    // Rate scales like 1/delta^2, independent of the sign of delta.
    const ll::PurcellModel far(g, -2.0 * delta, kappa);
    CHECK_NEAR(far.rate() / pm.rate(), 0.25, 1e-14);

    // Filters: 10 dB is a factor 10 in power, 20 dB a factor 100.
    CHECK_NEAR(pm.with_filter_db(10.0).rate() / pm.rate(), 0.1, 1e-14);
    CHECK_NEAR(pm.with_filter_db(20.0).rate() / pm.rate(), 0.01, 1e-14);
    CHECK_NEAR(pm.with_filter(0.5).filter_suppression(), 0.5, 0.0);

    // Channel: |1> decays at gamma_P, |0> is untouched.
    const double t = 4e-6;
    ll::CMatrix one = ll::CMatrix::Zero(2, 2);
    one(1, 1) = 1.0;
    pm.channel(t, 0).apply(one, 1);
    CHECK_NEAR(one(1, 1).real(), std::exp(-pm.rate() * t), 1e-12);
    ll::CMatrix zero = ll::ground_state_density(1);
    pm.channel(t, 0).apply(zero, 1);
    CHECK_NEAR(zero(0, 0).real(), 1.0, 1e-14);

    // combined_with: Tphi preserved and T2_eff <= 2 T1_eff, in a case where
    // holding T2 fixed would break the bound.
    const ll::PurcellModel strong(two_pi * 100e6, two_pi * 0.5e9, two_pi * 20e6);
    const ll::T1T2Model intrinsic(100e-6, 150e-6);
    const ll::T1T2Model eff = strong.combined_with(intrinsic);
    CHECK_NEAR(1.0 / eff.t1(), 1.0 / intrinsic.t1() + strong.rate(), 1e-6);
    CHECK_NEAR(eff.t_phi() / intrinsic.t_phi(), 1.0, 1e-9);
    CHECK(eff.t2() <= 2.0 * eff.t1() * (1.0 + T2_BOUND_RTOL));
    CHECK(2.0 * eff.t1() < intrinsic.t2());  // fixed-T2 would have been invalid

    // Intrinsic T2 = 2 T1 stays on the bound after adding Purcell decay.
    const ll::T1T2Model eff2 = strong.combined_with(ll::T1T2Model(60e-6, 120e-6));
    CHECK_NEAR(eff2.t2() / (2.0 * eff2.t1()), 1.0, 1e-9);

    CHECK(throws([] { ll::PurcellModel(1.0, 0.0, 1.0); }));
    CHECK(throws([] { ll::PurcellModel(1.0, 1.0, 1.0); }));  // g/delta above the limit
    CHECK(throws([] { ll::PurcellModel(0.1, 1.0, -1.0); }));
    CHECK(throws([] { ll::PurcellModel(0.1, 1.0, 1.0, 0.0); }));
    CHECK(throws([] { ll::PurcellModel(0.1, 1.0, 1.0, 1.5); }));
    CHECK(throws([&] { pm.with_filter_db(-3.0); }));
}

void test_edge_cases() {
    // n = 0
    const ll::CMatrix r0 = ll::ground_state_density(0);
    CHECK(r0.rows() == 1 && r0.cols() == 1);
    CHECK_NEAR(r0(0, 0).real(), 1.0, 0.0);
    ll::CMatrix r0_copy = r0;
    CHECK(throws([&] { ll::KrausChannel::identity(0).apply(r0_copy, 0); }));
    CHECK(ll::density_from_statevector(ll::CVector::Ones(1)).rows() == 1);

    CHECK(throws([] { ll::ground_state_density(ll::MAX_DM_QUBITS + 1); }));
    CHECK(throws([] { ll::ground_state_density(-1); }));
    CHECK(throws([] { ll::density_from_statevector(ll::CVector::Ones(3)); }));
    CHECK(throws([] { ll::density_from_statevector(ll::CVector::Ones(2)); }));  // not normalized

    // Register / target mismatches
    ll::CMatrix rho = ll::ground_state_density(2);
    CHECK(throws([&] { ll::KrausChannel::identity(2).apply(rho, 2); }));
    CHECK(throws([&] { ll::KrausChannel::identity(0).apply(rho, 3); }));
    CHECK(throws([&] { ll::KrausChannel().apply(rho, 2); }));

    // Bad channels
    CHECK(throws([] { ll::KrausChannel::amplitude_damping(1.2, 0); }));
    CHECK(throws([] { ll::KrausChannel::phase_damping(-0.1, 0); }));
    CHECK(throws([] { ll::KrausChannel::identity(-1); }));
    CHECK(throws([] {
        ll::KrausChannel({ll::CMatrix::Identity(2, 2) * 2.0}, {0});  // not trace preserving
    }));
    CHECK(throws([] { ll::KrausChannel({ll::CMatrix::Identity(3, 3)}, {0}); }));
    CHECK(throws([] { ll::KrausChannel({ll::CMatrix::Identity(4, 4)}, {1, 1}); }));
    CHECK(throws([] {
        ll::KrausChannel::identity(0).compose(ll::KrausChannel::identity(1));
    }));

    // Compose agrees with applying in sequence.
    std::mt19937 gen(7);
    const ll::KrausChannel a = ll::KrausChannel::amplitude_damping(0.25, 1);
    const ll::KrausChannel b = ll::KrausChannel::phase_damping(0.5, 1);
    ll::CMatrix s1 = random_density(3, gen);
    ll::CMatrix s2 = s1;
    a.apply(s1, 3);
    b.apply(s1, 3);
    a.compose(b).apply(s2, 3);
    CHECK((s1 - s2).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(a.compose(b).is_trace_preserving());

    // MAX_DM_QUBITS register runs and stays physical.
    ll::CMatrix big = ll::ground_state_density(ll::MAX_DM_QUBITS);
    ll::DepolarizingChannel(0.05, 2).to_kraus({3, 7}).apply(big, ll::MAX_DM_QUBITS);
    CHECK_NEAR(big.trace().real(), 1.0, 1e-12);
}

}  // namespace

int main() {
    test_index_manipulation();
    test_t1_t2_analytic();
    test_depolarizing();
    test_purcell();
    test_edge_cases();

    if (g_failures == 0) {
        std::printf("Noise: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "Noise: %d check(s) failed\n", g_failures);
    return 1;
}

#endif  // LL_TEST