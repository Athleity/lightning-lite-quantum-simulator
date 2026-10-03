// OpenMP state-vector backend (Tier 2, fast tier).
//
// Same conventions as Backend.h: qubit q is bit q of the basis-state index, RX/RY/RZ(t) =
// exp(-i t P/2), mcz negates every amplitude whose bits are all 1 on the listed qubits.
//
// Every gate walks the state in blocks. A gate on qubit q pairs index i (bit q = 0) with
// i + 2^q, so the state splits into N / 2^(q+1) blocks of 2^(q+1) amplitudes whose lower and
// upper halves are contiguous runs of 2^q. The outer block loop is parallel; when q is so high
// that fewer than 32 blocks remain, the inner run is parallel instead. CNOT uses the same
// scheme over (outer, middle) groups of contiguous runs of 2^min(control, target).
// States below 2^14 amplitudes run serially, where thread start-up costs more than the gate.
// No 2^n x 2^n matrix is ever built.
//
// Complex products are written out in real arithmetic, because std::complex operator* calls the
// NaN-checking __muldc3 without -ffast-math.
//
// Build (used directly, or via #include "StateVectorBackend.cpp" from a test; all members are
// defined in the class body, so including it in several translation units is ODR-safe):
//   g++ -O3 -march=native -fopenmp -std=c++17 -Isrc ...

#ifndef LL_STATEVECTOR_BACKEND_CPP
#define LL_STATEVECTOR_BACKEND_CPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Backend.h"

namespace ll {

class StateVectorBackend final : public Backend {
public:
    static constexpr int MAX_QUBITS = 25;  // 2^25 complex<double> = 512 MiB

    explicit StateVectorBackend(int n_qubits = 1) { reset(n_qubits); }

    void reset(int n_qubits) override {
        if (n_qubits < 1 || n_qubits > MAX_QUBITS)
            throw std::invalid_argument("StateVectorBackend::reset: n_qubits must be in [1, " +
                                        std::to_string(MAX_QUBITS) + "]");
        n_ = n_qubits;
        size_ = 1LL << n_;
        psi_.assign(static_cast<std::size_t>(size_), Complex(0.0, 0.0));
        psi_[0] = 1.0;
    }

    int n_qubits() const override { return n_; }

    void h(int q) override {
        const double r = 1.0 / std::sqrt(2.0);
        for_pairs(q, [r](Complex& a0, Complex& a1) {
            const double x0 = a0.real(), y0 = a0.imag(), x1 = a1.real(), y1 = a1.imag();
            a0 = Complex(r * (x0 + x1), r * (y0 + y1));
            a1 = Complex(r * (x0 - x1), r * (y0 - y1));
        });
    }

    void x(int q) override {
        for_pairs(q, [](Complex& a0, Complex& a1) { std::swap(a0, a1); });
    }

    // [c, -is; -is, c]
    void rx(int q, double t) override {
        check_angle(t, "rx");
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        for_pairs(q, [c, s](Complex& a0, Complex& a1) {
            const double x0 = a0.real(), y0 = a0.imag(), x1 = a1.real(), y1 = a1.imag();
            a0 = Complex(c * x0 + s * y1, c * y0 - s * x1);
            a1 = Complex(s * y0 + c * x1, -s * x0 + c * y1);
        });
    }

    // [c, -s; s, c]
    void ry(int q, double t) override {
        check_angle(t, "ry");
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        for_pairs(q, [c, s](Complex& a0, Complex& a1) {
            const double x0 = a0.real(), y0 = a0.imag(), x1 = a1.real(), y1 = a1.imag();
            a0 = Complex(c * x0 - s * x1, c * y0 - s * y1);
            a1 = Complex(s * x0 + c * x1, s * y0 + c * y1);
        });
    }

    // diag(e^{-it/2}, e^{+it/2})
    void rz(int q, double t) override {
        check_angle(t, "rz");
        const double c = std::cos(0.5 * t), s = std::sin(0.5 * t);
        for_pairs(q, [c, s](Complex& a0, Complex& a1) {
            a0 = Complex(c * a0.real() + s * a0.imag(), c * a0.imag() - s * a0.real());
            a1 = Complex(c * a1.real() - s * a1.imag(), c * a1.imag() + s * a1.real());
        });
    }

    // Swap the control = 1 amplitudes between target = 0 and target = 1. With lo = min and
    // hi = max of the two qubits, a group g selects the high bits (above hi) and middle bits
    // (between lo and hi); the 2^lo low bits form the contiguous run.
    void cnot(int control, int target) override {
        check_qubit(control);
        check_qubit(target);
        if (control == target) throw std::invalid_argument("StateVectorBackend::cnot: control == target");

        const int lo = std::min(control, target), hi = std::max(control, target);
        const int mid_bits = hi - lo - 1;
        const ll n_mid = 1LL << mid_bits;
        const ll groups = (size_ >> (hi + 1)) * n_mid;
        const ll run = 1LL << lo;
        const ll cbit = 1LL << control, tbit = 1LL << target;
        Complex* psi = psi_.data();
        const bool par = size_ >= PAR_MIN;

        auto base_of = [=](ll g) { return ((g >> mid_bits) << (hi + 1)) | ((g & (n_mid - 1)) << (lo + 1)); };

        if (groups >= 32) {
#pragma omp parallel for schedule(static) if (par)
            for (ll g = 0; g < groups; ++g) {
                Complex* p0 = psi + (base_of(g) | cbit);
                Complex* p1 = p0 + tbit;
                for (ll j = 0; j < run; ++j) std::swap(p0[j], p1[j]);
            }
        } else {
            for (ll g = 0; g < groups; ++g) {
                Complex* p0 = psi + (base_of(g) | cbit);
                Complex* p1 = p0 + tbit;
#pragma omp parallel for schedule(static) if (par)
                for (ll j = 0; j < run; ++j) std::swap(p0[j], p1[j]);
            }
        }
    }

    void mcz(const std::vector<int>& qubits) override {
        if (qubits.empty()) throw std::invalid_argument("StateVectorBackend::mcz: empty qubit list");
        ll mask = 0;
        for (int q : qubits) {
            check_qubit(q);
            const ll b = 1LL << q;
            if (mask & b) throw std::invalid_argument("StateVectorBackend::mcz: duplicate qubit");
            mask |= b;
        }
        Complex* psi = psi_.data();
        const bool par = size_ >= PAR_MIN;
#pragma omp parallel for schedule(static) if (par)
        for (ll i = 0; i < size_; ++i)
            if ((i & mask) == mask) psi[i] = -psi[i];
    }

    std::vector<double> probabilities() const override {
        std::vector<double> p(static_cast<std::size_t>(size_));
        const Complex* psi = psi_.data();
        double* out = p.data();
        const bool par = size_ >= PAR_MIN;
#pragma omp parallel for schedule(static) if (par)
        for (ll i = 0; i < size_; ++i) out[i] = psi[i].real() * psi[i].real() + psi[i].imag() * psi[i].imag();
        return p;
    }

    std::vector<std::complex<double>> state() const override { return psi_; }

private:
    using Complex = std::complex<double>;
    using ll = long long;

    static constexpr ll PAR_MIN = 1LL << 14;

    int n_ = 0;
    ll size_ = 0;
    std::vector<Complex> psi_;

    void check_qubit(int q) const {
        if (q < 0 || q >= n_)
            throw std::invalid_argument("StateVectorBackend: qubit " + std::to_string(q) +
                                        " outside [0, " + std::to_string(n_) + ")");
    }

    static void check_angle(double t, const char* who) {
        if (!std::isfinite(t))
            throw std::invalid_argument(std::string("StateVectorBackend::") + who + ": non-finite angle");
    }

    // kernel(a0, a1) acts on every amplitude pair (bit q = 0, bit q = 1).
    template <class K>
    void for_pairs(int q, K kernel) {
        check_qubit(q);
        const ll stride = 1LL << q;
        const ll n_blocks = size_ >> (q + 1);
        Complex* psi = psi_.data();
        const bool par = size_ >= PAR_MIN;

        if (n_blocks >= 32) {
#pragma omp parallel for schedule(static) if (par)
            for (ll b = 0; b < n_blocks; ++b) {
                Complex* p0 = psi + b * 2 * stride;
                Complex* p1 = p0 + stride;
                for (ll j = 0; j < stride; ++j) kernel(p0[j], p1[j]);
            }
        } else {
            for (ll b = 0; b < n_blocks; ++b) {
                Complex* p0 = psi + b * 2 * stride;
                Complex* p1 = p0 + stride;
#pragma omp parallel for schedule(static) if (par)
                for (ll j = 0; j < stride; ++j) kernel(p0[j], p1[j]);
            }
        }
    }
};

}  // namespace ll

#endif  // LL_STATEVECTOR_BACKEND_CPP