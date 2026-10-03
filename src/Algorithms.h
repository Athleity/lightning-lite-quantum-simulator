// Quantum algorithms (Tier 2): Grover, QAOA, VQE.
//
// Backend-agnostic: every routine issues gate calls through ll::Backend (Backend.h) and never
// touches amplitudes, so the Eigen reference backend and the StateVector fast path are
// interchangeable. Qubit q is bit q of the basis-state index (little endian), so
// probabilities()[z] is the probability of the bitstring with z_q = (z >> q) & 1.
//
// Header-only. Invalid input throws std::invalid_argument.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "Backend.h"

namespace ll {

namespace alg_detail {

constexpr double PI = 3.14159265358979323846;

[[noreturn]] inline void fail(const char* who, const char* msg) {
    throw std::invalid_argument(std::string(who) + ": " + msg);
}

}  // namespace alg_detail

// ===========================================================================
// Grover search
// ===========================================================================
//
//   |s> = H^{⊗n}|0>^n = N^{-1/2} Σ_z |z>,   N = 2^n
//   Oracle    O = I - 2 Σ_{m ∈ marked} |m><m|          (phase flip on marked states)
//   Diffusion D = H^{⊗n} X^{⊗n} (multi-CZ) X^{⊗n} H^{⊗n} = -(2|s><s| - I)  (global phase only)
//   G = D O rotates the state by 2θ per iteration in the plane spanned by the marked and
//   unmarked uniform superpositions, with sin θ = sqrt(M/N). After k iterations
//   P(marked) = sin²((2k+1)θ), maximal at k ≈ π/(4θ) - 1/2 ≈ (π/4) sqrt(N/M).
//
// The phase flip on |m> is X on every qubit where m has a 0 bit, then a multi-controlled Z
// on all n qubits (which acts only on |1...1>), then the same X layer again.

struct GroverResult {
    std::vector<double> probabilities;  // size 2^n, after the final iteration
    int marked_index = -1;              // most probable basis state
    int iterations = 0;                 // Grover iterations applied
};

class Grover {
public:
    // iterations = -1: nearest integer to π/(4θ) - 1/2 with θ = asin sqrt(M/N), i.e. (π/4) sqrt(N/M).
    // Throws if n_qubits is outside [1, 30], marked_states is empty, contains an index outside
    // [0, 2^n) or covers every state, or iterations < -1. Duplicate marked states are merged.
    static GroverResult run(Backend& backend, int n_qubits, const std::vector<int>& marked_states,
                            int iterations = -1) {
        const char* who = "Grover::run";
        if (n_qubits < 1 || n_qubits > 30) alg_detail::fail(who, "n_qubits must be in [1, 30]");
        if (iterations < -1) alg_detail::fail(who, "iterations must be >= 0, or -1 for automatic");
        const long long n_states = 1LL << n_qubits;

        std::vector<int> marked = marked_states;
        for (int m : marked)
            if (m < 0 || m >= n_states) alg_detail::fail(who, "marked state outside [0, 2^n)");
        std::sort(marked.begin(), marked.end());
        marked.erase(std::unique(marked.begin(), marked.end()), marked.end());
        if (marked.empty()) alg_detail::fail(who, "no marked states");
        if (static_cast<long long>(marked.size()) >= n_states)
            alg_detail::fail(who, "marked states must be a proper subset of the search space");

        int k = iterations;
        if (k < 0) {
            const double theta = std::asin(std::sqrt(static_cast<double>(marked.size()) /
                                                     static_cast<double>(n_states)));
            k = std::max(0, static_cast<int>(std::lround(alg_detail::PI / (4.0 * theta) - 0.5)));
        }

        backend.reset(n_qubits);
        for (int q = 0; q < n_qubits; ++q) backend.h(q);
        for (int it = 0; it < k; ++it) {
            for (int m : marked) phase_flip(backend, n_qubits, m);
            diffusion(backend, n_qubits);
        }

        GroverResult out;
        out.probabilities = backend.probabilities();
        if (static_cast<long long>(out.probabilities.size()) != n_states)
            throw std::runtime_error("Grover::run: backend returned wrong number of probabilities");
        out.marked_index = static_cast<int>(
            std::max_element(out.probabilities.begin(), out.probabilities.end()) -
            out.probabilities.begin());
        out.iterations = k;
        return out;
    }

private:
    static std::vector<int> all_qubits(int n) {
        std::vector<int> q(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) q[static_cast<std::size_t>(i)] = i;
        return q;
    }

    // |z> -> -|z> for z == target.
    static void phase_flip(Backend& b, int n, int target) {
        for (int q = 0; q < n; ++q)
            if (!((target >> q) & 1)) b.x(q);
        b.mcz(all_qubits(n));
        for (int q = 0; q < n; ++q)
            if (!((target >> q) & 1)) b.x(q);
    }

    // Inversion about the mean, up to a global phase of -1.
    static void diffusion(Backend& b, int n) {
        for (int q = 0; q < n; ++q) { b.h(q); b.x(q); }
        b.mcz(all_qubits(n));
        for (int q = 0; q < n; ++q) { b.x(q); b.h(q); }
    }
};

// ===========================================================================
// QAOA for MaxCut
// ===========================================================================
//
//   |ψ(γ, β)> = Π_{l=p..1} U_B(β_l) U_C(γ_l) |+>^n
//   H_C = Σ_{(i,j)} w_ij (1 - Z_i Z_j)/2          (eigenvalue = cut weight of the bitstring)
//   H_B = Σ_i X_i
//   U_C(γ) = exp(-iγ H_C) = (global phase) Π_{(i,j)} exp(+i γ w_ij Z_i Z_j / 2)
//          each factor = CNOT(i,j) RZ_j(-γ w_ij) CNOT(i,j),   RZ(θ) = exp(-iθZ/2)
//   U_B(β) = exp(-iβ H_B) = Π_i RX_i(2β),                     RX(θ) = exp(-iθX/2)
//
// The ansatz maximises <H_C>. best_cut below is the cut weight of the most probable bitstring;
// bitstrings z and ~z have equal cut and equal probability, so either one may be reported.

struct Graph {
    int n_nodes = 0;
    std::vector<std::tuple<int, int, double>> edges;  // (i, j, w_ij), i != j
};

struct QAOAResult {
    std::vector<double> probabilities;  // size 2^n_nodes
    double best_cut = 0.0;              // cut weight of the most probable bitstring
};

class QAOA {
public:
    // p = gammas.size() = betas.size() >= 1 layers.
    // Throws if n_nodes is outside [1, 30], an edge has an endpoint outside [0, n_nodes) or
    // i == j, a weight is not finite, or the angle lists differ in length or are empty.
    static QAOAResult run(Backend& backend, const Graph& g, const std::vector<double>& gammas,
                          const std::vector<double>& betas) {
        const char* who = "QAOA::run";
        if (g.n_nodes < 1 || g.n_nodes > 30) alg_detail::fail(who, "n_nodes must be in [1, 30]");
        if (gammas.empty() || gammas.size() != betas.size())
            alg_detail::fail(who, "gammas and betas must be non-empty and of equal length");
        for (const auto& e : g.edges) {
            const int i = std::get<0>(e), j = std::get<1>(e);
            if (i < 0 || j < 0 || i >= g.n_nodes || j >= g.n_nodes)
                alg_detail::fail(who, "edge endpoint outside [0, n_nodes)");
            if (i == j) alg_detail::fail(who, "self-loop edge");
            if (!std::isfinite(std::get<2>(e))) alg_detail::fail(who, "non-finite edge weight");
        }
        for (double a : gammas) if (!std::isfinite(a)) alg_detail::fail(who, "non-finite gamma");
        for (double a : betas) if (!std::isfinite(a)) alg_detail::fail(who, "non-finite beta");

        backend.reset(g.n_nodes);
        for (int q = 0; q < g.n_nodes; ++q) backend.h(q);
        for (std::size_t l = 0; l < gammas.size(); ++l) {
            for (const auto& e : g.edges) {
                const int i = std::get<0>(e), j = std::get<1>(e);
                backend.cnot(i, j);
                backend.rz(j, -gammas[l] * std::get<2>(e));
                backend.cnot(i, j);
            }
            for (int q = 0; q < g.n_nodes; ++q) backend.rx(q, 2.0 * betas[l]);
        }

        QAOAResult out;
        out.probabilities = backend.probabilities();
        if (static_cast<long long>(out.probabilities.size()) != (1LL << g.n_nodes))
            throw std::runtime_error("QAOA::run: backend returned wrong number of probabilities");
        const auto best = std::max_element(out.probabilities.begin(), out.probabilities.end());
        out.best_cut = cut_value(g, static_cast<std::uint64_t>(best - out.probabilities.begin()));
        return out;
    }

    // Cut weight Σ w_ij [z_i != z_j] of the bitstring with index z.
    static double cut_value(const Graph& g, std::uint64_t z) {
        double c = 0.0;
        for (const auto& e : g.edges)
            if (((z >> std::get<0>(e)) & 1u) != ((z >> std::get<1>(e)) & 1u)) c += std::get<2>(e);
        return c;
    }
};

// ===========================================================================
// VQE
// ===========================================================================
//
//   E(θ) = <ψ(θ)|H|ψ(θ)>,   |ψ(θ)> = Π_{l=1..L} [ CNOT ring · Π_q RY_q(θ_{l,q}) ] |0>^n
//   Parameters are ordered layer by layer: θ[l * n + q]. The CNOT ring is CNOT(q, q+1 mod n)
//   for n >= 3, a single CNOT(0, 1) for n = 2 and absent for n = 1.
//
//   Gradient by the parameter-shift rule. RY(θ) = exp(-iθY/2) has generator eigenvalues ±1/2,
//   so the rule is exact:
//       ∂E/∂θ_k = [ E(θ + (π/2) e_k) - E(θ - (π/2) e_k) ] / 2
//   Optimiser: plain gradient descent, θ ← θ - lr ∇E, stopping when |∇E| < 1e-6 or after
//   max_iter steps. The lowest energy seen is returned.
//
// energy_fn maps parameters to <H>. It owns the measurement: typically it calls prepare() on the
// same backend and then evaluates the Hamiltonian from backend.probabilities() (or the state).
// optimize() leaves the best state prepared on `backend`.

struct VQEResult {
    double energy = 0.0;         // lowest energy seen
    std::vector<double> params;  // parameters that gave it
    int iterations = 0;          // gradient steps taken
};

class VQE {
public:
    // Throws if n_qubits is outside [1, 30] or n_layers < 1. seed fixes the initial parameters,
    // drawn uniformly from [-0.1, 0.1] (all-zero start is a stationary point for many H).
    VQE(int n_qubits, int n_layers, std::uint64_t seed = 1) : n_(n_qubits), layers_(n_layers), seed_(seed) {
        if (n_qubits < 1 || n_qubits > 30) alg_detail::fail("VQE", "n_qubits must be in [1, 30]");
        if (n_layers < 1) alg_detail::fail("VQE", "n_layers must be >= 1");
    }

    int n_qubits() const { return n_; }
    int n_layers() const { return layers_; }
    int n_params() const { return n_ * layers_; }

    // Reset the backend to |0...0> and apply the ansatz with the given parameters.
    void prepare(Backend& backend, const std::vector<double>& params) const {
        if (static_cast<int>(params.size()) != n_params())
            alg_detail::fail("VQE::prepare", "wrong number of parameters");
        backend.reset(n_);
        for (int l = 0; l < layers_; ++l) {
            for (int q = 0; q < n_; ++q) backend.ry(q, params[static_cast<std::size_t>(l * n_ + q)]);
            if (n_ == 2) {
                backend.cnot(0, 1);
            } else if (n_ > 2) {
                for (int q = 0; q < n_; ++q) backend.cnot(q, (q + 1) % n_);
            }
        }
    }

    // Throws if max_iter < 0, lr is not finite and positive, or energy_fn returns a non-finite value.
    VQEResult optimize(Backend& backend, const std::function<double(const std::vector<double>&)>& energy_fn,
                       int max_iter = 200, double lr = 0.05) const {
        const char* who = "VQE::optimize";
        if (max_iter < 0) alg_detail::fail(who, "max_iter must be >= 0");
        if (!(std::isfinite(lr) && lr > 0.0)) alg_detail::fail(who, "lr must be finite and > 0");
        if (!energy_fn) alg_detail::fail(who, "energy_fn is empty");

        auto eval = [&](const std::vector<double>& th) {
            const double e = energy_fn(th);
            if (!std::isfinite(e)) throw std::runtime_error(std::string(who) + ": non-finite energy");
            return e;
        };

        std::mt19937_64 rng(seed_);
        std::uniform_real_distribution<double> uni(-0.1, 0.1);
        std::vector<double> theta(static_cast<std::size_t>(n_params()));
        for (double& t : theta) t = uni(rng);

        VQEResult out;
        out.energy = eval(theta);
        out.params = theta;

        const double shift = alg_detail::PI / 2.0;
        constexpr double grad_tol = 1e-6;
        std::vector<double> grad(theta.size());

        for (int it = 0; it < max_iter; ++it) {
            double g2 = 0.0;
            for (std::size_t k = 0; k < theta.size(); ++k) {
                const double t0 = theta[k];
                theta[k] = t0 + shift;
                const double ep = eval(theta);
                theta[k] = t0 - shift;
                const double em = eval(theta);
                theta[k] = t0;
                grad[k] = 0.5 * (ep - em);
                g2 += grad[k] * grad[k];
            }
            if (std::sqrt(g2) < grad_tol) break;

            for (std::size_t k = 0; k < theta.size(); ++k) theta[k] -= lr * grad[k];
            ++out.iterations;

            const double e = eval(theta);
            if (e < out.energy) {
                out.energy = e;
                out.params = theta;
            }
        }

        prepare(backend, out.params);
        return out;
    }

private:
    int n_;
    int layers_;
    std::uint64_t seed_;
};

}  // namespace ll