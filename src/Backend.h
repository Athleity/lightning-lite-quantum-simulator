// Gate-level backend interface (Tier 2) and a dense reference implementation.
//
// Algorithms.h talks only to ll::Backend. Conventions shared by every implementation:
//   - qubit q is bit q of the basis-state index (little endian)
//   - RX(t) = exp(-i t X/2), RY(t) = exp(-i t Y/2), RZ(t) = exp(-i t Z/2)
//   - mcz(qubits) negates every amplitude whose bits are all 1 on `qubits`
//     (Z for one qubit, CZ for two, CCZ for three, ...)
//   - invalid qubit indices throw std::invalid_argument
//
// To plug in another simulator (Eigen reference, StateVector fast path), derive from Backend
// and forward each call to it.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ll {

class Backend {
public:
    virtual ~Backend() = default;

    // Allocate n_qubits and set the state to |0...0>.
    virtual void reset(int n_qubits) = 0;
    virtual int n_qubits() const = 0;

    virtual void h(int q) = 0;
    virtual void x(int q) = 0;
    virtual void rx(int q, double theta) = 0;
    virtual void ry(int q, double theta) = 0;
    virtual void rz(int q, double theta) = 0;
    virtual void cnot(int control, int target) = 0;
    virtual void mcz(const std::vector<int>& qubits) = 0;

    // |amplitude|^2 for every basis state, size 2^n.
    virtual std::vector<double> probabilities() const = 0;
    // Full amplitudes, size 2^n.
    virtual std::vector<std::complex<double>> state() const = 0;
};

// Dense state vector, no optimisation beyond in-place pair updates. Serves as the correctness
// reference for the other backends.
class ReferenceBackend final : public Backend {
public:
    static constexpr int MAX_QUBITS = 26;  // 2^26 complex<double> = 1 GiB

    explicit ReferenceBackend(int n_qubits = 1) { reset(n_qubits); }

    void reset(int n_qubits) override {
        if (n_qubits < 1 || n_qubits > MAX_QUBITS)
            throw std::invalid_argument("ReferenceBackend::reset: n_qubits must be in [1, " +
                                        std::to_string(MAX_QUBITS) + "]");
        n_ = n_qubits;
        psi_.assign(std::size_t{1} << n_, Complex(0.0, 0.0));
        psi_[0] = 1.0;
    }

    int n_qubits() const override { return n_; }

    void h(int q) override {
        const double r = 1.0 / std::sqrt(2.0);
        apply1(q, r, r, r, -r);
    }

    void x(int q) override { apply1(q, 0.0, 1.0, 1.0, 0.0); }

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
        apply1(q, std::polar(1.0, -0.5 * t), 0.0, 0.0, std::polar(1.0, 0.5 * t));
    }

    void cnot(int control, int target) override {
        check_qubit(control);
        check_qubit(target);
        if (control == target) throw std::invalid_argument("ReferenceBackend::cnot: control == target");
        const std::size_t cb = std::size_t{1} << control, tb = std::size_t{1} << target;
        for (std::size_t i = 0; i < psi_.size(); ++i)
            if ((i & cb) && !(i & tb)) std::swap(psi_[i], psi_[i | tb]);
    }

    void mcz(const std::vector<int>& qubits) override {
        if (qubits.empty()) throw std::invalid_argument("ReferenceBackend::mcz: empty qubit list");
        std::size_t mask = 0;
        for (int q : qubits) {
            check_qubit(q);
            const std::size_t b = std::size_t{1} << q;
            if (mask & b) throw std::invalid_argument("ReferenceBackend::mcz: duplicate qubit");
            mask |= b;
        }
        for (std::size_t i = 0; i < psi_.size(); ++i)
            if ((i & mask) == mask) psi_[i] = -psi_[i];
    }

    std::vector<double> probabilities() const override {
        std::vector<double> p(psi_.size());
        for (std::size_t i = 0; i < psi_.size(); ++i) p[i] = std::norm(psi_[i]);
        return p;
    }

    std::vector<std::complex<double>> state() const override { return psi_; }

private:
    using Complex = std::complex<double>;

    int n_ = 0;
    std::vector<Complex> psi_;

    void check_qubit(int q) const {
        if (q < 0 || q >= n_)
            throw std::invalid_argument("ReferenceBackend: qubit " + std::to_string(q) +
                                        " outside [0, " + std::to_string(n_) + ")");
    }

    static void check_angle(double t, const char* who) {
        if (!std::isfinite(t))
            throw std::invalid_argument(std::string("ReferenceBackend::") + who + ": non-finite angle");
    }

    // [a b; c d] acting on qubit q.
    void apply1(int q, Complex a, Complex b, Complex c, Complex d) {
        check_qubit(q);
        const std::size_t bit = std::size_t{1} << q;
        for (std::size_t i = 0; i < psi_.size(); ++i) {
            if (i & bit) continue;
            const Complex lo = psi_[i], hi = psi_[i | bit];
            psi_[i] = a * lo + b * hi;
            psi_[i | bit] = c * lo + d * hi;
        }
    }
};

}  // namespace ll