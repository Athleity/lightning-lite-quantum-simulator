#pragma once

#include <Eigen/Dense>

#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ll {

using cplx = std::complex<double>;
using CMatrix = Eigen::MatrixXcd;
using CVector = Eigen::VectorXcd;

// Largest register the density-matrix code accepts. rho is 2^n x 2^n complex,
// so n = 10 is 1024 x 1024 x 16 B = 16 MiB per matrix.
constexpr int MAX_DM_QUBITS = 10;

// Default absolute tolerance for trace-preservation and density-matrix checks.
constexpr double DEFAULT_TOL = 1e-10;

// Relative slack allowed when checking T2 <= 2*T1, so values that sit on the
// bound up to rounding (T2 = 2*T1 written as a float) are accepted.
constexpr double T2_BOUND_RTOL = 1e-9;

// Largest g/|delta| accepted by PurcellModel. Past this the dispersive
// expansion gamma_P = kappa (g/delta)^2 is no longer a controlled
// approximation and the constructor throws instead of returning a wrong rate.
constexpr double MAX_DISPERSIVE_RATIO = 0.5;

// Power attenuation: suppression = 10^(-dB / DB_POWER_DIVISOR).
constexpr double DB_POWER_DIVISOR = 10.0;

// ---------------------------------------------------------------------------
// Density-matrix helpers
// ---------------------------------------------------------------------------

// |psi><psi|. Throws std::invalid_argument if the length of psi is not a power
// of two, if it exceeds 2^MAX_DM_QUBITS, or if psi is not normalized to within
// DEFAULT_TOL.
CMatrix density_from_statevector(const CVector& psi);

// |0...0><0...0| on n qubits. Throws if n is outside [0, MAX_DM_QUBITS].
// n = 0 returns the 1x1 matrix [1].
CMatrix ground_state_density(int n_qubits);

// Hermitian, unit trace, positive semidefinite, all to within tol. The PSD
// check uses the smallest eigenvalue of the Hermitian part, so it costs one
// eigendecomposition. Intended for tests and debug assertions, not inner loops.
bool is_valid_density_matrix(const CMatrix& rho, double tol = DEFAULT_TOL);

// ---------------------------------------------------------------------------
// KrausChannel
// ---------------------------------------------------------------------------

// A completely positive trace-preserving map on m target qubits,
//
//     rho' = sum_k K_k rho K_k^dagger,      sum_k K_k^dagger K_k = I.
//
// Operator convention: each K_k is 2^m x 2^m. Bit j of an operator index
// refers to targets()[j], so targets()[0] is the least significant bit of the
// operator's row and column index. Register convention: qubit q is bit q of
// the basis-state index, qubit 0 least significant.
//
// Application algorithm (no 2^n x 2^n operator is ever built). Write a basis
// index of the register as (rest, t), where t is the m target bits packed in
// operator order and rest is everything else. Then
//
//     rho'[(r1,t1),(r2,t2)] = sum_k sum_{a,b} K_k[t1,a] rho[(r1,a),(r2,b)]
//                                             conj(K_k[t2,b])
//
// so for every fixed pair (r1, r2) the entries rho[(r1,.),(r2,.)] form a
// 2^m x 2^m block B and the update is B -> K B K^dagger. The code loops over
// all 4^(n-m) pairs (r1, r2), gathers B using precomputed bit-deposit offsets,
// does two small Eigen products per Kraus operator, accumulates, and scatters
// the result back. Cost is O(4^n * 2^m * K) time and one block of scratch per
// thread. Blocks are independent, so the loop is parallelized with OpenMP.
//
// Validate against: amplitude_damping(gamma) must map |1><1| to
// gamma |0><0| + (1-gamma) |1><1| and scale off-diagonals by sqrt(1-gamma).
// phase_damping(lambda) must leave populations alone and scale off-diagonals
// by sqrt(1-lambda).
class KrausChannel {
public:
    KrausChannel() = default;

    // Throws std::invalid_argument if ops or targets is empty, targets has a
    // negative or repeated entry, any operator is not 2^m x 2^m, or the
    // operators are not trace preserving to within DEFAULT_TOL.
    KrausChannel(std::vector<CMatrix> ops, std::vector<int> targets);

    // In-place update of rho on n_qubits qubits. Throws std::invalid_argument
    // if n_qubits is outside [0, MAX_DM_QUBITS], rho is not 2^n x 2^n, or any
    // target is >= n_qubits. A default-constructed channel throws too, and so
    // does any channel with targets when n_qubits == 0.
    void apply(CMatrix& rho, int n_qubits) const;

    // max_{ij} |(sum_k K_k^dagger K_k - I)_{ij}| <= tol
    bool is_trace_preserving(double tol = DEFAULT_TOL) const;

    // The channel that applies *this first and then other.
    // compose requires identical target sets: same qubits in the same order.
    // Throws std::invalid_argument otherwise. The result has
    // operators().size() * other.operators().size() Kraus operators, ordered
    // with the *this index as the slow one.
    KrausChannel compose(const KrausChannel& other) const;

    const std::vector<CMatrix>& operators() const { return ops_; }
    const std::vector<int>& targets() const { return targets_; }
    int num_targets() const { return static_cast<int>(targets_.size()); }

    // K0 = [[1, 0], [0, sqrt(1-gamma)]]
    // K1 = [[0, sqrt(gamma)], [0, 0]]
    // gamma in [0, 1], else std::invalid_argument.
    static KrausChannel amplitude_damping(double gamma, int qubit);

    // K0 = [[1, 0], [0, sqrt(1-lambda)]]
    // K1 = [[0, 0], [0, sqrt(lambda)]]
    // lambda in [0, 1], else std::invalid_argument.
    static KrausChannel phase_damping(double lambda, int qubit);

    static KrausChannel identity(int qubit);

private:
    std::vector<CMatrix> ops_;
    std::vector<int> targets_;
};

// ---------------------------------------------------------------------------
// DepolarizingChannel
// ---------------------------------------------------------------------------

// Symmetric Pauli noise on m = 1 or 2 qubits,
//
//     rho' = (1 - p) rho + p / (4^m - 1) * sum_{P != I} P rho P.
//
// p is the total probability of a non-identity Pauli error, p in [0, 1].
// This is the convention of a Pauli error rate, not the "shrink toward I/2^m"
// parameter. The map sends rho to (1 - lam) rho + lam I/2^m with
// lam = p * 4^m / (4^m - 1), so the fully depolarizing point is
// p = (4^m - 1) / 4^m (3/4 for one qubit, 15/16 for two). Larger p is still a
// valid CPTP map but is "past" fully depolarizing.
//
// Validate against: on one qubit the Bloch vector shrinks by 1 - 4p/3, and
// the average gate fidelity of the channel is 1 - 2p/3 (one qubit) or
// 1 - 4p/5 (two qubits).
class DepolarizingChannel {
public:
    // Throws std::invalid_argument if p is outside [0, 1] (or NaN) or
    // num_qubits is not 1 or 2.
    DepolarizingChannel(double p, int num_qubits);

    double probability() const { return p_; }
    int num_qubits() const { return m_; }

    // targets.size() must equal num_qubits() with distinct non-negative
    // entries, else std::invalid_argument. Kraus operators are
    // sqrt(1-p) I and sqrt(p/(4^m-1)) P for each non-identity Pauli string P.
    KrausChannel to_kraus(const std::vector<int>& targets) const;

private:
    double p_;
    int m_;
};

// ---------------------------------------------------------------------------
// T1T2Model
// ---------------------------------------------------------------------------

// Single-qubit relaxation and dephasing over a window of length t (an idle
// period or a gate duration). With T2 the Ramsey coherence time and
// 1/T2 = 1/(2 T1) + 1/Tphi, the Bloch-vector components evolve as
//
//     P1(t)    = P1(0) exp(-t / T1)                  (longitudinal)
//     |rho01|  = |rho01(0)| exp(-t / T2)             (transverse)
//
// T1 = infinity is allowed (no relaxation). It is stored as +inf and then
// Tphi = T2. T2 = infinity is allowed only together with T1 = infinity.
//
// The channel is amplitude damping followed by pure dephasing (the two
// commute). Amplitude damping alone multiplies rho01 by sqrt(1-gamma) =
// exp(-t/(2 T1)). Phase damping multiplies it by sqrt(1-lambda), so matching
// exp(-1/Tphi * t) requires
//
//     gamma  = 1 - exp(-t / T1)
//     lambda = 1 - exp(-2 t / Tphi).
//
// Note the factor 2 in lambda: lambda is a probability, while Tphi is the
// decay time of the amplitude (coherence) itself.
//
// Validate against: excited_population(t) and coherence_magnitude(t) are the
// closed forms above. Applying channel(t) to |1><1| and to |+><+| must
// reproduce them to machine precision, and N applications of channel(t/N)
// must equal one application of channel(t) (semigroup property).
class T1T2Model {
public:
    // Throws std::invalid_argument if t1 <= 0, t2 <= 0, either is NaN, or
    // t2 > 2 * t1 * (1 + T2_BOUND_RTOL) (unphysical: negative Tphi rate).
    T1T2Model(double t1, double t2);

    double t1() const { return t1_; }
    double t2() const { return t2_; }

    // Pure dephasing time, 1 / (1/T2 - 1/(2 T1)). Returns +infinity when
    // T2 = 2 T1 (no pure dephasing).
    double t_phi() const;

    // gamma and lambda as defined above. duration >= 0, else
    // std::invalid_argument.
    double gamma(double duration) const;
    double lambda(double duration) const;

    // duration >= 0 and qubit >= 0, else std::invalid_argument.
    KrausChannel channel(double duration, int qubit) const;

    // exp(-t / T1): excited-state population starting from |1>.
    double excited_population(double duration) const;

    // exp(-t / T2): |rho01| / |rho01(0)|.
    double coherence_magnitude(double duration) const;

private:
    double t1_;
    double t2_;
};

// ---------------------------------------------------------------------------
// PurcellModel
// ---------------------------------------------------------------------------

// Qubit decay through its readout resonator into the output line (Purcell
// effect). In the dispersive limit, with qubit-resonator coupling g,
// detuning delta = omega_q - omega_r and resonator linewidth kappa (all in
// rad/s),
//
//     gamma_P = kappa * (g / delta)^2.
//
// This is a dispersive-limit scalar approximation. Real pi-filters reshape
// the frequency-dependent impedance Z(w) seen by the resonator; here we model
// that as a single linear suppression factor on kappa. Not a full Z(w)
// treatment.
//
// filter_suppression s in (0, 1] multiplies kappa, so gamma_P = s * kappa *
// (g/delta)^2 and s = 1 means no filter. For an attenuation of A dB in power
// at the qubit frequency, s = 10^(-A/10).
//
// Constructor limits: g >= 0, delta != 0, kappa >= 0, s in (0, 1], and
// g/|delta| <= MAX_DISPERSIVE_RATIO. Violations throw std::invalid_argument.
//
// Validate against: the closed form above for a few (g, delta, kappa)
// triples, a rate that scales as 1/delta^2 and linearly in s, and, for the
// channel, population decay exp(-gamma_P t) on |1><1| with no effect on |0>.
class PurcellModel {
public:
    PurcellModel(double g, double delta, double kappa, double filter_suppression = 1.0);

    // Dispersive parameter g / |delta|.
    double dispersive_ratio() const;

    // gamma_P in 1/s.
    double rate() const;

    // 1 / gamma_P in seconds, +infinity if the rate is zero.
    double t1_limit() const;

    double filter_suppression() const { return filter_; }

    // Copies of this model with the filter factor replaced (not multiplied)
    // by the given value. with_filter needs a value in (0, 1];
    // with_filter_db needs attenuation_db >= 0. Otherwise they throw
    // std::invalid_argument.
    PurcellModel with_filter(double suppression) const;
    PurcellModel with_filter_db(double attenuation_db) const;

    // Pure amplitude damping with gamma = 1 - exp(-gamma_P t). duration >= 0
    // and qubit >= 0, else std::invalid_argument.
    KrausChannel channel(double duration, int qubit) const;

    // Intrinsic T1/T2 plus the Purcell channel, with the intrinsic pure
    // dephasing time held fixed:
    //
    //     Tphi      = 1 / (1/T2 - 1/(2 T1))      (intrinsic, unchanged)
    //     1/T1_eff  = 1/T1 + gamma_P
    //     1/T2_eff  = 1/(2 T1_eff) + 1/Tphi
    //
    // Scaling T2 along with T1 would be wrong, and so would keeping T2 fixed
    // while T1 drops: the latter can break T2 <= 2 T1_eff. Rebuilding T2_eff
    // from the same Tphi satisfies the bound by construction, since
    // 1/T2_eff >= 1/(2 T1_eff). Purcell decay is a pure energy-loss process
    // and does not add dephasing beyond what the relaxation itself implies.
    T1T2Model combined_with(const T1T2Model& intrinsic) const;

private:
    double g_;
    double delta_;
    double kappa_;
    double filter_;
};

}  // namespace ll