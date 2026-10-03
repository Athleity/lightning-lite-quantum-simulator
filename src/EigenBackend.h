// Eigen-vectorised state-vector backend (Tier 2, middle speed tier).
//
// Same conventions as Backend.h: qubit q is bit q of the basis-state index, RX/RY/RZ(t) =
// exp(-i t P/2), mcz negates every amplitude whose bits are all 1 on the listed qubits.
//
// A gate on qubit q views psi as a (2^q) x (2^(n-q)) column-major matrix. Even columns hold the
// q = 0 amplitudes and odd columns the q = 1 amplitudes, so both halves are strided Eigen maps and
// the 2x2 update becomes coefficient-wise expressions that Eigen vectorises (runs of 2^q
// contiguous elements; the gain is smallest for q = 0). No OpenMP. A gate allocates one temporary
// of half the state size.

#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Backend.h"

namespace ll {

class EigenBackend final : public Backend {
public:
    static constexpr int MAX_QUBITS = 24;  // 2^24 complex<double> = 256 MiB

    explicit EigenBackend(int n_qubits = 1) { reset(n_qubits); }

    void reset(int n_qubits) override {
        if (n_qubits < 1 || n_qubits > MAX_QUBITS)
            throw std::invalid_argument("EigenBackend::reset: n_qubits must be in [1, " +
                                        std::to_string(MAX_QUBITS) + "]");
        n_ = n_qubits;
        psi_.setZero(Eigen::Index(1) << n_);
        psi_(0) = 1.0;
    }

    int n_qubits() const override { return n_; }

    void h(int q) override {
        const double r = 1.0 / std::sqrt(2.0);
        apply1(q, r, r, r, -r);
    }

    void x(int q) override {
        check_qubit(q);
        const Eigen::Index bit = Eigen::Index(1) << q;
        const Eigen::Index cols = psi_.size() / (2 * bit);
        Strided s0(psi_.data(), bit, cols, Eigen::OuterStride<>(2 * bit));
        Strided s1(psi_.data() + bit, bit, cols, Eigen::OuterStride<>(2 * bit));
        s0.swap(s1);
    }

    void rx(int q, double t) override {
        check_angle(t, "rx");
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        apply1(q, c, Complex(0.0, -s), Complex(0.0, -s), c);
    }

    void ry(int q, double t) override {
        check_angle(t, "ry");
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        apply1(q, c, -s, s, c);
    }

    void rz(int q, double t) override {
        check_angle(t, "rz");
        check_qubit(q);
        const Eigen::Index bit = Eigen::Index(1) << q;
        const Eigen::Index cols = psi_.size() / (2 * bit);
        Strided s0(psi_.data(), bit, cols, Eigen::OuterStride<>(2 * bit));
        Strided s1(psi_.data() + bit, bit, cols, Eigen::OuterStride<>(2 * bit));
        s0 *= std::polar(1.0, -0.5 * t);
        s1 *= std::polar(1.0, 0.5 * t);
    }

    // Swap the control = 1 amplitudes between target = 0 and target = 1, in contiguous runs of
    // 2^min(control, target) elements.
    void cnot(int control, int target) override {
        check_qubit(control);
        check_qubit(target);
        if (control == target) throw std::invalid_argument("EigenBackend::cnot: control == target");

        const int lo = std::min(control, target), hi = std::max(control, target);
        const Eigen::Index run = Eigen::Index(1) << lo;
        const Eigen::Index n_outer = psi_.size() >> (hi + 1);
        const Eigen::Index n_mid = Eigen::Index(1) << (hi - lo - 1);
        const Eigen::Index cbit = Eigen::Index(1) << control, tbit = Eigen::Index(1) << target;

        for (Eigen::Index o = 0; o < n_outer; ++o) {
            for (Eigen::Index m = 0; m < n_mid; ++m) {
                const Eigen::Index base = (o << (hi + 1)) | (m << (lo + 1));
                const Eigen::Index i0 = base | cbit;  // control = 1, target = 0
                psi_.segment(i0, run).swap(psi_.segment(i0 | tbit, run));
            }
        }
    }

    void mcz(const std::vector<int>& qubits) override {
        if (qubits.empty()) throw std::invalid_argument("EigenBackend::mcz: empty qubit list");
        Eigen::Index mask = 0;
        for (int q : qubits) {
            check_qubit(q);
            const Eigen::Index b = Eigen::Index(1) << q;
            if (mask & b) throw std::invalid_argument("EigenBackend::mcz: duplicate qubit");
            mask |= b;
        }
        for (Eigen::Index i = 0; i < psi_.size(); ++i)
            if ((i & mask) == mask) psi_(i) = -psi_(i);
    }

    std::vector<double> probabilities() const override {
        const Eigen::VectorXd p = psi_.cwiseAbs2();
        return std::vector<double>(p.data(), p.data() + p.size());
    }

    std::vector<std::complex<double>> state() const override {
        return std::vector<std::complex<double>>(psi_.data(), psi_.data() + psi_.size());
    }

private:
    using Complex = std::complex<double>;
    using Strided = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::OuterStride<>>;

    int n_ = 0;
    Eigen::VectorXcd psi_;

    void check_qubit(int q) const {
        if (q < 0 || q >= n_)
            throw std::invalid_argument("EigenBackend: qubit " + std::to_string(q) +
                                        " outside [0, " + std::to_string(n_) + ")");
    }

    static void check_angle(double t, const char* who) {
        if (!std::isfinite(t))
            throw std::invalid_argument(std::string("EigenBackend::") + who + ": non-finite angle");
    }

    // [a b; c d] acting on qubit q. The new q = 1 half reads the old q = 0 half, so the new q = 0
    // half goes into a temporary first.
    void apply1(int q, Complex a, Complex b, Complex c, Complex d) {
        check_qubit(q);
        const Eigen::Index bit = Eigen::Index(1) << q;
        const Eigen::Index cols = psi_.size() / (2 * bit);
        Strided s0(psi_.data(), bit, cols, Eigen::OuterStride<>(2 * bit));
        Strided s1(psi_.data() + bit, bit, cols, Eigen::OuterStride<>(2 * bit));
        const Eigen::MatrixXcd t = a * s0 + b * s1;
        s1 = c * s0 + d * s1;
        s0 = t;
    }
};

}  // namespace ll