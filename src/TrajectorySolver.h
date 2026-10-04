// Quantum trajectories with variance-reduced unravelings (Tier 3).
//
// Lindblad master equation with jump operators C_k = sqrt(gamma_k) L_k:
//   d rho/dt = -i [H(t), rho] + sum_k ( C_k rho C_k^+ - 1/2 {C_k^+ C_k, rho} )
//
// An unraveling is a stochastic pure-state process |psi(t)> whose ensemble average
// rho = E[ |psi><psi| ] obeys this equation. Many unravelings give the same rho; they differ in
// the variance of per-trajectory estimators, and the standard error after N trajectories is
// sigma / sqrt(N), so trajectories needed for a target error scale with sigma^2.
//
// STANDARD  Monte-Carlo wave function (quantum jumps). Between jumps the state follows the
//           non-Hermitian H_eff = H - (i/2) sum_k C_k^+ C_k. Each step evolves with H_eff to
//           |psi~>; the jump probability is the norm loss 1 - <psi~|psi~>. On a jump, channel k
//           is chosen with weight ||C_k psi~||^2 and |psi> <- C_k psi~ / ||C_k psi~||; otherwise
//           |psi> <- psi~ / ||psi~||. For a Pauli channel (L = P, P^2 = 1) a jump applies P at
//           rate gamma ("Pauli-flip" unraveling).
//
// PROJECTOR For a Hermitian involution L = sqrt(c) P (P^2 = 1) the channel is
//             g (P rho P - rho),  g = gamma c.
//           It has the equivalent jump operator C = sqrt(g) (1 - P), because
//             C rho C^+ - 1/2 {C^+C, rho} = g (P rho P - rho)   exactly (C^+C = 2g (1 - P)).
//           The jump projects onto the P = -1 eigenspace at rate 2g (1 - <P>) and the no-jump
//           evolution drifts toward P = +1. Channels that are not Pauli-like (e.g. sigma_-) have
//           no such form and fall back to STANDARD (see projector_applies()).
//
// ANALOG    Diffusive unraveling, no jumps (homodyne / quantum-state-diffusion form), real Wiener
//           increments dW_k, x_k = <C_k + C_k^+>:
//             d psi = [ -iH - 1/2 sum_k (C_k^+C_k - x_k C_k + x_k^2/4) ] psi dt
//                     + sum_k (C_k - x_k/2) psi dW_k
//           Integrated by Euler-Maruyama with renormalisation (weak order 1; keep gamma*dt and
//           ||H||*dt small). With constant H the Hamiltonian part is applied exactly (Lie split).
//           Terminology: the tensor-jump / GPU trajectory papers call a Kraus-operator-sampling
//           scheme "analog"; here ANALOG follows the diffusive definition in the interface spec.
//
// Variance reduction is not a universal factor. It depends on the unraveling, the observable and
// the initial state. Closed form for pure dephasing L = Z, rate gamma, |psi0> = |+>, observable
// <X>, m = exp(-2 gamma t), so E[<X>] = m:
//   STANDARD   <X> = +-1 per trajectory:                      Var = 1 - m^2
//   PROJECTOR  <X> = 2m/(1+m^2) with prob (1+m^2)/2, else 0:  Var = m^2 (1 - m^2)/(1 + m^2)
//   Var_std / Var_proj = 1 + exp(4 gamma t),  = 21.1 at gamma t = 0.75.
// The same estimator would be worse under PROJECTOR for <Z> on |+>. The variance-reduction
// claim of ~21x (arXiv:2607.17678, projector vs Pauli-flip, strong-noise regime) is reproduced
// here by that closed form and measured in test 4.
//
// Thread safety and reproducibility. Trajectory i draws from its own generator seeded by
// splitmix64(seed ^ splitmix64(i + 1)), so results do not depend on the thread count; run_one
// with seed s equals trajectory 0 of an ensemble with seed s. The Hamiltonian function and the
// observable are called concurrently (when built with -fopenmp) and must be thread-safe.
//
// Time-dependent H(t) is sampled once per call at the step endpoints and midpoints, shared by all
// trajectories, and integrated with RK4 (jump unravelings). Memory for the samples is checked.
//
// Test build:
//   g++ -O2 -std=c++17 -DLL_TEST -fopenmp -I/usr/include/eigen3 -x c++ src/TrajectorySolver.h \
//       -o build/trajectory_test

#ifndef LL_TRAJECTORY_SOLVER_H
#define LL_TRAJECTORY_SOLVER_H

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ll {

using CMatrix = Eigen::MatrixXcd;
using CVector = Eigen::VectorXcd;
using HamiltonianFn = std::function<CMatrix(double)>;  // t -> H(t), dim x dim, Hermitian

enum class Unraveling { STANDARD, PROJECTOR, ANALOG };

namespace traj_detail {

[[noreturn]] inline void fail(const char* who, const std::string& msg) {
    throw std::invalid_argument(std::string(who) + ": " + msg);
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// exp(A): scaling and squaring, 18-term Taylor series on A / 2^s with ||A / 2^s||_inf <= 1/2.
inline CMatrix expm(const CMatrix& a) {
    const double nrm = a.cwiseAbs().rowwise().sum().maxCoeff();
    int s = 0;
    if (nrm > 0.5) s = std::max(0, static_cast<int>(std::ceil(std::log2(nrm / 0.5))));
    const CMatrix b = a / std::pow(2.0, s);
    CMatrix term = CMatrix::Identity(a.rows(), a.cols());
    CMatrix result = term;
    for (int k = 1; k <= 18; ++k) {
        term = (term * b) / static_cast<double>(k);
        result += term;
    }
    for (int i = 0; i < s; ++i) result = result * result;
    return result;
}

}  // namespace traj_detail

class TrajectorySolver {
public:
    static constexpr int MAX_QUBITS = 10;  // dense dim x dim matrices

    struct Stats {
        double mean = 0.0;
        double std_error = 0.0;  // sample std / sqrt(n); 0 for n = 1
        int n = 0;
    };

    // Throws std::invalid_argument if n_qubits is outside [1, MAX_QUBITS].
    explicit TrajectorySolver(int n_qubits) : n_(n_qubits) {
        if (n_qubits < 1 || n_qubits > MAX_QUBITS)
            traj_detail::fail("TrajectorySolver", "n_qubits must be in [1, " + std::to_string(MAX_QUBITS) + "]");
        dim_ = 1 << n_;
        h_const_ = CMatrix::Zero(dim_, dim_);
        reset();
    }

    int n_qubits() const { return n_; }
    int dim() const { return dim_; }
    std::size_t n_channels() const { return channels_.size(); }

    // H = 0 by default. Throws if the matrix has the wrong size, is non-finite or not Hermitian.
    void set_hamiltonian(const CMatrix& h) {
        check_hamiltonian(h, "set_hamiltonian");
        h_const_ = h;
        h_fn_ = nullptr;
    }

    // Time-dependent H(t); H(0) is validated here, later calls are validated when used.
    void set_hamiltonian(HamiltonianFn h) {
        if (!h) traj_detail::fail("set_hamiltonian", "empty function");
        check_hamiltonian(h(0.0), "set_hamiltonian");
        h_fn_ = std::move(h);
    }

    // Adds the term gamma (L rho L^+ - 1/2 {L^+L, rho}). gamma = 0 adds nothing.
    // Throws if L has the wrong size or is non-finite, or gamma is negative or non-finite.
    void add_lindblad(const CMatrix& l, double gamma) {
        const char* who = "add_lindblad";
        if (l.rows() != dim_ || l.cols() != dim_) traj_detail::fail(who, "operator must be dim x dim");
        if (!l.allFinite()) traj_detail::fail(who, "non-finite operator");
        if (!(std::isfinite(gamma) && gamma >= 0.0)) traj_detail::fail(who, "gamma must be finite and >= 0");
        if (gamma == 0.0) return;
        Channel ch;
        ch.l = l;
        ch.gamma = gamma;
        ch.pauli = pauli_like(l, ch.p, ch.g);
        if (ch.pauli) ch.g *= gamma;  // g = gamma c
        channels_.push_back(std::move(ch));
    }

    void clear_lindblad() { channels_.clear(); }

    // True if PROJECTOR rewrites channel i (L Hermitian with L^2 = c 1, c > 0).
    bool projector_applies(std::size_t i) const { return channels_.at(i).pauli; }

    // Convenience default state |0...0>; the evolution routines take psi0 explicitly.
    void reset() {
        psi_ = CVector::Zero(dim_);
        psi_(0) = 1.0;
    }
    const CVector& state() const { return psi_; }

    // One trajectory over `duration` with step ~dt (the step is duration / ceil(duration / dt)).
    // psi0 is normalised internally. Returns the normalised final state.
    // Throws std::invalid_argument for a wrong-size or zero psi0, duration < 0 or dt <= 0.
    CVector run_one(const CVector& psi0, double duration, double dt, std::uint64_t seed,
                    Unraveling kind = Unraveling::STANDARD) const {
        const char* who = "run_one";
        CVector psi = checked_state(psi0, who);
        check_time(duration, dt, who);
        if (duration == 0.0) return psi;
        const Plan plan = make_plan(kind, duration, dt);
        std::mt19937_64 rng = make_rng(seed, 0);
        trajectory(psi, plan, rng);
        return psi;
    }

    // rho = (1/N) sum_i |psi_i><psi_i|. Same exceptions as run_one, and n_trajectories >= 1.
    CMatrix evolve_ensemble(const CVector& psi0, double duration, double dt, int n_trajectories,
                            Unraveling kind, std::uint64_t seed) const {
        const char* who = "evolve_ensemble";
        const CVector start = checked_state(psi0, who);
        check_time(duration, dt, who);
        if (n_trajectories < 1) traj_detail::fail(who, "n_trajectories must be >= 1");
        if (duration == 0.0) return start * start.adjoint();
        const Plan plan = make_plan(kind, duration, dt);

        // Fixed chunking, summed in order: the result does not depend on the thread count.
        const long long dd = static_cast<long long>(dim_) * dim_ * 16;
        const long long max_by_mem = std::max<long long>(1, (256LL << 20) / dd);
        const int n_chunks = static_cast<int>(std::min<long long>({64LL, n_trajectories, max_by_mem}));
        std::vector<CMatrix> part(static_cast<std::size_t>(n_chunks), CMatrix::Zero(dim_, dim_));

        std::exception_ptr err;
#pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < n_chunks; ++c) {
            try {
                const long long lo = static_cast<long long>(n_trajectories) * c / n_chunks;
                const long long hi = static_cast<long long>(n_trajectories) * (c + 1) / n_chunks;
                CVector psi;
                for (long long i = lo; i < hi; ++i) {
                    psi = start;
                    std::mt19937_64 rng = make_rng(seed, static_cast<std::uint64_t>(i));
                    trajectory(psi, plan, rng);
                    part[static_cast<std::size_t>(c)].noalias() += psi * psi.adjoint();
                }
            } catch (...) {
#pragma omp critical(ll_traj_error)
                {
                    if (!err) err = std::current_exception();
                }
            }
        }
        if (err) std::rethrow_exception(err);

        CMatrix rho = CMatrix::Zero(dim_, dim_);
        for (const CMatrix& p : part) rho += p;
        rho /= static_cast<double>(n_trajectories);
        return rho;
    }

    // Mean and standard error of obs(psi_T) over n_trajectories (n >= 1). obs must be thread-safe.
    Stats estimate_observable(const CVector& psi0, double duration, double dt, int n_trajectories,
                              Unraveling kind, const std::function<double(const CVector&)>& obs,
                              std::uint64_t seed) const {
        const char* who = "estimate_observable";
        const CVector start = checked_state(psi0, who);
        check_time(duration, dt, who);
        if (n_trajectories < 1) traj_detail::fail(who, "n_trajectories must be >= 1");
        if (!obs) traj_detail::fail(who, "empty observable");
        const Plan plan = (duration > 0.0) ? make_plan(kind, duration, dt) : Plan();

        std::vector<double> v(static_cast<std::size_t>(n_trajectories));
        std::exception_ptr err;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_trajectories; ++i) {
            try {
                CVector psi = start;
                if (duration > 0.0) {
                    std::mt19937_64 rng = make_rng(seed, static_cast<std::uint64_t>(i));
                    trajectory(psi, plan, rng);
                }
                v[static_cast<std::size_t>(i)] = obs(psi);
            } catch (...) {
#pragma omp critical(ll_traj_error)
                {
                    if (!err) err = std::current_exception();
                }
            }
        }
        if (err) std::rethrow_exception(err);

        Stats st;
        st.n = n_trajectories;
        double sum = 0.0;
        for (double x : v) sum += x;
        st.mean = sum / n_trajectories;
        if (n_trajectories > 1) {
            double ss = 0.0;
            for (double x : v) ss += (x - st.mean) * (x - st.mean);
            st.std_error = std::sqrt(ss / (n_trajectories - 1) / n_trajectories);
        }
        return st;
    }

private:
    struct Channel {
        CMatrix l;
        double gamma = 0.0;
        bool pauli = false;
        CMatrix p;       // L / sqrt(c), P^2 = 1
        double g = 0.0;  // c at detection, gamma c after add_lindblad
    };

    struct Plan {
        Unraveling kind = Unraveling::STANDARD;
        long long steps = 0;
        double h = 0.0;
        std::vector<CMatrix> op;      // jump operators (STANDARD, PROJECTOR) or C_k (ANALOG)
        std::vector<CMatrix> op_dop;  // op^+ op
        bool constant = true;
        CMatrix prop;                 // constant H: exp(-i H_eff h)
        std::vector<CMatrix> nodes;   // time-dependent H: jump 2*steps+1 nodes, ANALOG steps nodes
    };

    int n_ = 0;
    int dim_ = 0;
    CMatrix h_const_;
    HamiltonianFn h_fn_;
    std::vector<Channel> channels_;
    CVector psi_;

    static std::mt19937_64 make_rng(std::uint64_t seed, std::uint64_t index) {
        return std::mt19937_64(traj_detail::splitmix64(seed ^ traj_detail::splitmix64(index + 1)));
    }

    // L Hermitian with L^2 = c 1, c > 0. On success p = L / sqrt(c) and c is returned in `c`.
    static bool pauli_like(const CMatrix& l, CMatrix& p, double& c) {
        if ((l - l.adjoint()).norm() > 1e-9 * std::max(1.0, l.norm())) return false;
        const CMatrix l2 = l * l;
        c = l2(0, 0).real();
        if (!(c > 1e-12)) return false;
        if ((l2 - c * CMatrix::Identity(l.rows(), l.cols())).norm() > 1e-9 * std::max(1.0, l2.norm()))
            return false;
        p = l / std::sqrt(c);
        return true;
    }

    void check_hamiltonian(const CMatrix& h, const char* who) const {
        if (h.rows() != dim_ || h.cols() != dim_) traj_detail::fail(who, "H must be dim x dim");
        if (!h.allFinite()) traj_detail::fail(who, "non-finite H");
        if ((h - h.adjoint()).norm() > 1e-9 * std::max(1.0, h.norm())) traj_detail::fail(who, "H is not Hermitian");
    }

    CVector checked_state(const CVector& psi0, const char* who) const {
        if (psi0.size() != dim_) traj_detail::fail(who, "psi0 must have length 2^n_qubits");
        if (!psi0.allFinite()) traj_detail::fail(who, "non-finite psi0");
        const double nrm = psi0.norm();
        if (!(nrm > 1e-12)) traj_detail::fail(who, "psi0 has zero norm");
        return psi0 / nrm;
    }

    static void check_time(double duration, double dt, const char* who) {
        if (!(std::isfinite(duration) && duration >= 0.0)) traj_detail::fail(who, "duration must be finite and >= 0");
        if (!(std::isfinite(dt) && dt > 0.0)) traj_detail::fail(who, "dt must be finite and > 0");
        if (duration / dt > 2e9) traj_detail::fail(who, "too many time steps");
    }

    CMatrix eval_h(double t) const {
        CMatrix h = h_fn_(t);
        if (h.rows() != dim_ || h.cols() != dim_) traj_detail::fail("H(t)", "returned a matrix of the wrong size");
        if (!h.allFinite()) traj_detail::fail("H(t)", "returned non-finite values");
        return h;
    }

    Plan make_plan(Unraveling kind, double duration, double dt) const {
        const std::complex<double> I(0.0, 1.0);
        Plan p;
        if (channels_.empty()) kind = Unraveling::STANDARD;  // unitary evolution: exact propagator
        p.kind = kind;
        p.steps = std::max<long long>(1, static_cast<long long>(std::ceil(duration / dt - 1e-9)));
        p.h = duration / static_cast<double>(p.steps);

        const CMatrix id = CMatrix::Identity(dim_, dim_);
        for (const Channel& ch : channels_) {
            CMatrix j = (kind == Unraveling::PROJECTOR && ch.pauli) ? CMatrix(std::sqrt(ch.g) * (id - ch.p))
                                                                    : CMatrix(std::sqrt(ch.gamma) * ch.l);
            p.op_dop.push_back(j.adjoint() * j);
            p.op.push_back(std::move(j));
        }

        const bool jump = (kind != Unraveling::ANALOG);
        CMatrix hnh = CMatrix::Zero(dim_, dim_);  // H_eff = H + hnh
        if (jump)
            for (const CMatrix& m : p.op_dop) hnh -= std::complex<double>(0.0, 0.5) * m;

        p.constant = !h_fn_;
        if (p.constant) {
            p.prop = traj_detail::expm((-I) * (h_const_ + hnh) * p.h);
            return p;
        }

        const long long n_nodes = jump ? 2 * p.steps + 1 : p.steps;
        const double bytes = static_cast<double>(n_nodes) * dim_ * dim_ * 16.0;
        if (bytes > 1.0e9)
            traj_detail::fail("TrajectorySolver", "time-dependent H(t) samples exceed 1 GB; increase dt");
        p.nodes.reserve(static_cast<std::size_t>(n_nodes));
        for (long long j = 0; j < n_nodes; ++j) {
            const double t = jump ? 0.5 * p.h * static_cast<double>(j) : p.h * static_cast<double>(j);
            p.nodes.push_back((-I) * (eval_h(t) + hnh));
        }
        return p;
    }

    void trajectory(CVector& psi, const Plan& p, std::mt19937_64& rng) const {
        if (p.kind == Unraveling::ANALOG) diffusive(psi, p, rng);
        else jumps(psi, p, rng);
    }

    // Norm-loss quantum-jump algorithm (STANDARD and PROJECTOR).
    void jumps(CVector& psi, const Plan& p, std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        const std::size_t nch = p.op.size();
        std::vector<double> weight(nch);
        CVector tmp(dim_), w(dim_), k1(dim_), k2(dim_), k3(dim_), k4(dim_);
        const double h = p.h;

        for (long long s = 0; s < p.steps; ++s) {
            if (p.constant) {
                tmp.noalias() = p.prop * psi;
            } else {
                const CMatrix& a0 = p.nodes[static_cast<std::size_t>(2 * s)];
                const CMatrix& a1 = p.nodes[static_cast<std::size_t>(2 * s + 1)];
                const CMatrix& a2 = p.nodes[static_cast<std::size_t>(2 * s + 2)];
                k1.noalias() = a0 * psi;
                w = psi + (0.5 * h) * k1;
                k2.noalias() = a1 * w;
                w = psi + (0.5 * h) * k2;
                k3.noalias() = a1 * w;
                w = psi + h * k3;
                k4.noalias() = a2 * w;
                tmp = psi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            }

            const double survive = tmp.squaredNorm();
            if (!(survive > 0.0) || !std::isfinite(survive))
                throw std::runtime_error("TrajectorySolver: state norm vanished; reduce dt");

            if (nch > 0 && uni(rng) >= survive) {
                double total = 0.0;
                for (std::size_t k = 0; k < nch; ++k) {
                    w.noalias() = p.op[k] * tmp;
                    weight[k] = w.squaredNorm();
                    total += weight[k];
                }
                if (total > 0.0) {
                    const double r = uni(rng) * total;
                    std::size_t pick = nch;
                    double acc = 0.0;
                    for (std::size_t k = 0; k < nch; ++k) {
                        acc += weight[k];
                        if (r < acc && weight[k] > 0.0) {
                            pick = k;
                            break;
                        }
                    }
                    if (pick == nch) {  // rounding: take the last channel with positive weight
                        for (std::size_t k = nch; k-- > 0;)
                            if (weight[k] > 0.0) {
                                pick = k;
                                break;
                            }
                    }
                    psi.noalias() = p.op[pick] * tmp;
                    psi /= std::sqrt(weight[pick]);
                    continue;
                }
            }
            psi = tmp / std::sqrt(survive);
        }
    }

    // Homodyne diffusion, Euler-Maruyama with renormalisation (ANALOG).
    void diffusive(CVector& psi, const Plan& p, std::mt19937_64& rng) const {
        std::normal_distribution<double> gauss(0.0, std::sqrt(p.h));
        const std::size_t nch = p.op.size();
        CVector drift(dim_), noise(dim_), cp(dim_), t2(dim_), tmp(dim_);

        for (long long s = 0; s < p.steps; ++s) {
            if (p.constant) {
                tmp.noalias() = p.prop * psi;  // exact Hamiltonian step
                psi = tmp;
                drift.setZero();
            } else {
                drift.noalias() = p.nodes[static_cast<std::size_t>(s)] * psi;  // -iH(t) psi
            }
            noise.setZero();
            for (std::size_t k = 0; k < nch; ++k) {
                cp.noalias() = p.op[k] * psi;
                const double x = 2.0 * psi.dot(cp).real();  // <C + C^+>
                t2.noalias() = p.op_dop[k] * psi;
                drift += -0.5 * (t2 - x * cp + 0.25 * x * x * psi);
                noise += (cp - 0.5 * x * psi) * gauss(rng);
            }
            psi += p.h * drift + noise;
            const double nrm = psi.norm();
            if (!(nrm > 0.0) || !std::isfinite(nrm))
                throw std::runtime_error("TrajectorySolver: diffusion state diverged; reduce dt");
            psi /= nrm;
        }
    }
};

}  // namespace ll

// ===========================================================================
// Tests
// ===========================================================================
#ifdef LL_TEST

#include <cstdio>

namespace {

using namespace ll;
using C = std::complex<double>;

int g_total = 0, g_failed = 0;

void chk(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_total;
    if (!ok) ++g_failed;
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  ", detail.c_str());
}

std::string fmt(const char* f, double a, double b = 0.0, double c = 0.0) {
    char buf[200];
    std::snprintf(buf, sizeof buf, f, a, b, c);
    return buf;
}

const char* kind_name(Unraveling k) {
    return k == Unraveling::STANDARD ? "STANDARD " : (k == Unraveling::PROJECTOR ? "PROJECTOR" : "ANALOG   ");
}

CMatrix mat2(C a, C b, C c, C d) {
    CMatrix m(2, 2);
    m << a, b, c, d;
    return m;
}
CMatrix sx() { return mat2(0.0, 1.0, 1.0, 0.0); }
CMatrix sz() { return mat2(1.0, 0.0, 0.0, -1.0); }
CMatrix sm() { return mat2(0.0, 1.0, 0.0, 0.0); }  // |0><1|, |0> is the ground state
CVector vec2(C a, C b) {
    CVector v(2);
    v << a, b;
    return v;
}
CMatrix kron(const CMatrix& a, const CMatrix& b) {
    CMatrix k(a.rows() * b.rows(), a.cols() * b.cols());
    for (Eigen::Index i = 0; i < a.rows(); ++i)
        for (Eigen::Index j = 0; j < a.cols(); ++j) k.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) = a(i, j) * b;
    return k;
}
double obs_p1(const CVector& p) { return std::norm(p(1)); }
double obs_x(const CVector& p) { return 2.0 * (std::conj(p(0)) * p(1)).real(); }

// Exact reference: RK4 on the density-matrix Lindblad equation.
struct Chan {
    CMatrix l;
    double g;
};

CMatrix lindblad_rhs(const CMatrix& h, const std::vector<Chan>& ch, const CMatrix& rho) {
    const C I(0.0, 1.0);
    CMatrix out = -I * (h * rho - rho * h);
    for (const Chan& c : ch) {
        const CMatrix ldl = c.l.adjoint() * c.l;
        out += c.g * (c.l * rho * c.l.adjoint() - 0.5 * (ldl * rho + rho * ldl));
    }
    return out;
}

CMatrix exact_rho(const std::function<CMatrix(double)>& hf, const std::vector<Chan>& ch, CMatrix rho,
                  double T, int steps) {
    const double h = T / steps;
    for (int i = 0; i < steps; ++i) {
        const double t = i * h;
        const CMatrix h0 = hf(t), hm = hf(t + 0.5 * h), h1 = hf(t + h);
        const CMatrix k1 = lindblad_rhs(h0, ch, rho);
        const CMatrix k2 = lindblad_rhs(hm, ch, CMatrix(rho + 0.5 * h * k1));
        const CMatrix k3 = lindblad_rhs(hm, ch, CMatrix(rho + 0.5 * h * k2));
        const CMatrix k4 = lindblad_rhs(h1, ch, CMatrix(rho + h * k3));
        rho += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    return rho;
}

template <class F>
bool throws_invalid(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// 1. T1 decay
void test_t1() {
    std::printf("\n[1] Single-qubit T1: <P1>(t) = exp(-t/T1), start |1>, T1 = 2\n");
    const double t1 = 2.0;
    TrajectorySolver s(1);
    s.add_lindblad(sm(), 1.0 / t1);
    const CVector one = vec2(0.0, 1.0);
    for (Unraveling kind : {Unraveling::STANDARD, Unraveling::PROJECTOR, Unraveling::ANALOG}) {
        const bool an = (kind == Unraveling::ANALOG);
        bool ok = true;
        double worst = 0.0;
        for (double f : {0.5, 1.0, 2.0}) {
            const double t = f * t1, exact = std::exp(-t / t1);
            const auto e = s.estimate_observable(one, t, an ? 0.005 : 0.02, an ? 8000 : 20000, kind, obs_p1, 101);
            const double tol = 4.0 * e.std_error + (an ? 0.01 : 0.0);
            worst = std::max(worst, std::abs(e.mean - exact) / e.std_error);
            std::printf("    %s t = %.1f  mean %.4f  exact %.4f  se %.4f\n", kind_name(kind), t, e.mean, exact, e.std_error);
            ok = ok && std::abs(e.mean - exact) <= tol;
        }
        chk(std::string(kind_name(kind)) + ": mean within 4 se of exp(-t/T1) at t = T1/2, T1, 2 T1", ok,
            fmt("worst %.2f se", worst));
    }
    chk("sigma_- is not Pauli-like: PROJECTOR falls back to STANDARD", !s.projector_applies(0));
}

// 2. T2 decay (T1 + pure dephasing)
void test_t2() {
    std::printf("\n[2] Single-qubit T2: <X>(t) = exp(-t/T2), 1/T2 = 1/(2 T1) + 1/Tphi, start |+>\n");
    const double t1 = 2.0, tphi = 1.5, inv_t2 = 0.5 / t1 + 1.0 / tphi, t2 = 1.0 / inv_t2;
    TrajectorySolver s(1);
    s.add_lindblad(sm(), 1.0 / t1);
    s.add_lindblad(sz(), 0.5 / tphi);  // D[sqrt(g) Z]: coherence decays at 2 g = 1/Tphi
    const double r = 1.0 / std::sqrt(2.0);
    const CVector plus = vec2(r, r);
    chk("Z channel is Pauli-like, sigma_- is not", s.projector_applies(1) && !s.projector_applies(0));
    for (Unraveling kind : {Unraveling::STANDARD, Unraveling::PROJECTOR, Unraveling::ANALOG}) {
        const bool an = (kind == Unraveling::ANALOG);
        bool ok = true;
        double worst = 0.0;
        for (double f : {0.5, 1.0, 2.0}) {
            const double t = f * t2, exact = std::exp(-t / t2);
            const auto e = s.estimate_observable(plus, t, an ? 0.005 : 0.02, an ? 8000 : 20000, kind, obs_x, 202);
            const double tol = 4.0 * e.std_error + (an ? 0.01 : 0.002);
            worst = std::max(worst, std::abs(e.mean - exact) / e.std_error);
            std::printf("    %s t = %.2f  mean %.4f  exact %.4f  se %.4f\n", kind_name(kind), t, e.mean, exact, e.std_error);
            ok = ok && std::abs(e.mean - exact) <= tol;
        }
        chk(std::string(kind_name(kind)) + ": mean within 4 se of exp(-t/T2) at t = T2/2, T2, 2 T2", ok,
            fmt("worst %.2f se", worst));
    }
}

// 3. 1/sqrt(N) scaling
void test_scaling() {
    std::printf("\n[3] Standard error scales as 1/sqrt(N) (pure dephasing, gamma t = 0.75, STANDARD)\n");
    TrajectorySolver s(1);
    s.add_lindblad(sz(), 1.0);
    const double r = 1.0 / std::sqrt(2.0);
    const CVector plus = vec2(r, r);
    std::vector<double> lx, ly;
    for (int n : {500, 2000, 8000, 32000}) {
        const auto e = s.estimate_observable(plus, 0.75, 0.01, n, Unraveling::STANDARD, obs_x, 303);
        std::printf("    N = %-6d se = %.5f   se * sqrt(N) = %.4f\n", n, e.std_error, e.std_error * std::sqrt(double(n)));
        lx.push_back(std::log(double(n)));
        ly.push_back(std::log(e.std_error));
    }
    double mx = 0, my = 0;
    for (std::size_t i = 0; i < lx.size(); ++i) { mx += lx[i]; my += ly[i]; }
    mx /= lx.size();
    my /= ly.size();
    double sxy = 0, sxx = 0;
    for (std::size_t i = 0; i < lx.size(); ++i) { sxy += (lx[i] - mx) * (ly[i] - my); sxx += (lx[i] - mx) * (lx[i] - mx); }
    const double slope = sxy / sxx;
    chk("log-log slope of se vs N is -1/2 (+-0.06)", std::abs(slope + 0.5) < 0.06, fmt("slope %.4f", slope));
}

// 4. Variance reduction
void test_variance_reduction() {
    std::printf("\n[4] Variance reduction: dephasing L = Z, |+>, observable <X>; Var ratio = 1 + exp(4 gamma t)\n");
    TrajectorySolver s(1);
    s.add_lindblad(sz(), 1.0);
    const double r = 1.0 / std::sqrt(2.0);
    const CVector plus = vec2(r, r);
    const int n = 20000;
    const double eps = 0.005;
    std::printf("    gamma t   Var(STD)   Var(PROJ)  Var(ANALOG)  ratio    analytic   N_std(eps=%.3f)  N_proj\n", eps);
    double ratio_main = 0.0, analytic_main = 0.0, vproj_main = 0.0, bias_main = 1.0;
    for (double gt : {0.35, 0.75, 1.5}) {
        const double dt = 0.005;
        const auto es = s.estimate_observable(plus, gt, dt, n, Unraveling::STANDARD, obs_x, 404);
        const auto ep = s.estimate_observable(plus, gt, dt, n, Unraveling::PROJECTOR, obs_x, 405);
        const auto ea = s.estimate_observable(plus, gt, dt, n, Unraveling::ANALOG, obs_x, 406);
        const double vs = es.std_error * es.std_error * n, vp = ep.std_error * ep.std_error * n,
                     va = ea.std_error * ea.std_error * n;
        const double ratio = vs / vp, analytic = 1.0 + std::exp(4.0 * gt);
        std::printf("    %-8.2f  %-9.5f  %-9.5f  %-11.5f  %-7.2f  %-9.2f  %-16.0f %.0f\n", gt, vs, vp, va, ratio, analytic,
                    vs / (eps * eps), vp / (eps * eps));
        if (std::abs(gt - 0.75) < 1e-12) {
            ratio_main = ratio;
            analytic_main = analytic;
            vproj_main = vp;
            bias_main = std::abs(ep.mean - std::exp(-2.0 * gt));
        }
    }
    chk("PROJECTOR unbiased at gamma t = 0.75 (4 se)", bias_main < 4.0 * std::sqrt(vproj_main / n));
    chk("variance ratio matches 1 + exp(4 gamma t) within 10% at gamma t = 0.75",
        std::abs(ratio_main / analytic_main - 1.0) < 0.10, fmt("measured %.2f, analytic %.2f", ratio_main, analytic_main));
    chk("PROJECTOR needs at least 15x fewer trajectories than STANDARD here", ratio_main > 15.0,
        fmt("%.1fx", ratio_main));
}

// 5. Ensemble rho vs exact Lindblad
void check_rho(const char* label, const CMatrix& rho, const CMatrix& exact, double tol) {
    const double err = (rho - exact).cwiseAbs().maxCoeff();
    const double tr = std::abs(rho.trace() - C(1.0, 0.0));
    const double herm = (rho - rho.adjoint()).norm();
    Eigen::SelfAdjointEigenSolver<CMatrix> es(0.5 * (rho + rho.adjoint()), Eigen::EigenvaluesOnly);
    const double lmin = es.eigenvalues().minCoeff();
    chk(std::string(label) + ": max |rho - rho_exact| < " + fmt("%.3f", tol), err < tol, fmt("%.4f", err));
    chk(std::string(label) + ": trace = 1, Hermitian, positive", tr < 1e-12 && herm < 1e-12 && lmin > -1e-12,
        fmt("|tr-1| %.1e, min eig %.1e", tr, lmin));
}

void test_vs_master_equation() {
    std::printf("\n[5] Ensemble rho vs exact Lindblad integration (RK4 on the density matrix)\n");
    std::printf("    Reference is local to this test; compare with LindbladSolver once its interface is wired in.\n");
    const CMatrix id2 = CMatrix::Identity(2, 2);
    const CMatrix h = 0.3 * kron(sz(), id2) + 0.2 * kron(id2, sz()) + 0.5 * kron(sx(), sx());
    const CMatrix l1 = kron(sm(), id2), l2 = kron(id2, sm()), l3 = kron(sz(), id2), l4 = kron(sx(), sx());
    const double g1 = 0.3, g2 = 0.2, g3 = 0.25, g4 = 0.1;

    TrajectorySolver s(2);
    s.set_hamiltonian(h);
    s.add_lindblad(l1, g1);
    s.add_lindblad(l2, g2);
    s.add_lindblad(l3, g3);
    s.add_lindblad(l4, g4);
    chk("detection: Z x 1 and X x X are Pauli-like, sigma_- channels are not",
        !s.projector_applies(0) && !s.projector_applies(1) && s.projector_applies(2) && s.projector_applies(3));

    CVector psi0(4);
    psi0 << C(1.0, 0.0), C(0.5, 0.0), C(0.0, -0.3), C(0.2, 0.1);
    psi0 /= psi0.norm();
    const double T = 1.5;
    const std::vector<Chan> ch{{l1, g1}, {l2, g2}, {l3, g3}, {l4, g4}};
    const CMatrix exact = exact_rho([&](double) { return h; }, ch, CMatrix(psi0 * psi0.adjoint()), T, 3000);

    check_rho("STANDARD ", s.evolve_ensemble(psi0, T, 0.01, 20000, Unraveling::STANDARD, 501), exact, 0.02);
    check_rho("PROJECTOR", s.evolve_ensemble(psi0, T, 0.01, 20000, Unraveling::PROJECTOR, 502), exact, 0.02);
    check_rho("ANALOG   ", s.evolve_ensemble(psi0, T, 0.004, 10000, Unraveling::ANALOG, 503), exact, 0.03);

    // Time-dependent H(t): RK4 branch.
    const double om = 1.3, rabi = 2.0;
    auto hf = [&](double t) { return CMatrix(0.5 * rabi * std::sin(om * t) * sx() + 0.25 * sz()); };
    TrajectorySolver s1(1);
    s1.set_hamiltonian(HamiltonianFn(hf));
    s1.add_lindblad(sm(), 0.2);
    s1.add_lindblad(sz(), 0.1);
    const CVector g0 = vec2(1.0, 0.0);
    const std::vector<Chan> ch1{{sm(), 0.2}, {sz(), 0.1}};
    const CMatrix exact1 = exact_rho(hf, ch1, CMatrix(g0 * g0.adjoint()), 2.0, 3000);
    check_rho("STANDARD, H(t) driven", s1.evolve_ensemble(g0, 2.0, 0.01, 20000, Unraveling::STANDARD, 504), exact1, 0.02);
    check_rho("ANALOG, H(t) driven  ", s1.evolve_ensemble(g0, 2.0, 0.004, 10000, Unraveling::ANALOG, 505), exact1, 0.03);
}

// 6. Reproducibility and input validation
void test_misc() {
    std::printf("\n[6] Reproducibility and exceptions\n");
    TrajectorySolver s(2);
    s.add_lindblad(kron(sm(), CMatrix::Identity(2, 2)), 0.5);
    CVector psi0(4);
    psi0 << 0.0, 1.0, 1.0, 0.0;
    const CMatrix a = s.evolve_ensemble(psi0, 1.0, 0.01, 300, Unraveling::STANDARD, 7);
    const CMatrix b = s.evolve_ensemble(psi0, 1.0, 0.01, 300, Unraveling::STANDARD, 7);
    chk("same seed gives an identical ensemble", (a - b).norm() == 0.0);
    const CVector p1 = s.run_one(psi0, 1.0, 0.01, 7);
    const CVector p2 = s.run_one(psi0, 1.0, 0.01, 7);
    chk("run_one is deterministic and normalised", (p1 - p2).norm() == 0.0 && std::abs(p1.norm() - 1.0) < 1e-12);
    const CMatrix single = s.evolve_ensemble(psi0, 1.0, 0.01, 1, Unraveling::STANDARD, 7);
    chk("run_one equals trajectory 0 of an ensemble", (single - p1 * p1.adjoint()).norm() < 1e-14);

    chk("n_qubits = 0 throws", throws_invalid([] { TrajectorySolver t(0); }));
    chk("wrong-size Lindblad operator throws", throws_invalid([&] { s.add_lindblad(sz(), 1.0); }));
    chk("negative gamma throws", throws_invalid([&] { s.add_lindblad(CMatrix::Identity(4, 4), -1.0); }));
    chk("non-Hermitian H throws", throws_invalid([&] { s.set_hamiltonian(CMatrix(sm().replicate(2, 2))); }));
    chk("dt <= 0 throws", throws_invalid([&] { s.run_one(psi0, 1.0, 0.0, 1); }));
    chk("n_trajectories = 0 throws",
        throws_invalid([&] { s.evolve_ensemble(psi0, 1.0, 0.01, 0, Unraveling::STANDARD, 1); }));
    chk("wrong-size psi0 throws", throws_invalid([&] { s.run_one(vec2(1.0, 0.0), 1.0, 0.01, 1); }));
}

}  // namespace

int main() {
    std::printf("TrajectorySolver tests\n");
    test_t1();
    test_t2();
    test_scaling();
    test_variance_reduction();
    test_vs_master_equation();
    test_misc();
    std::printf("\n%d/%d checks pass\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}

#endif  // LL_TEST
#endif  // LL_TRAJECTORY_SOLVER_H