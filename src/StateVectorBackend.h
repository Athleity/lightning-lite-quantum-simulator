// OpenMP state-vector backend (Tier 2, fast tier).
//
// Same conventions as Backend.h: qubit q is bit q of the basis-state index, RX/RY/RZ(t) =
// exp(-i t P/2), mcz negates every amplitude whose bits are all 1 on the listed qubits.
//
// Every gate is one OpenMP parallel region. A gate on qubit q pairs index i (bit q = 0) with
// i + 2^q, giving N/2 independent pairs. Pair p sits at
//   i0 = ((p >> q) << (q + 1)) | (p & (2^q - 1)),   partner i1 = i0 + 2^q.
// Each thread takes a contiguous range of pair indices and walks it in runs of contiguous pairs
// (a run ends where p >> q changes), so the inner loop is unit stride for every q and the work
// splits evenly even when q is large and only a few blocks exist. CNOT uses the same scheme over
// N/4 pairs, with runs of 2^min(control, target).
// States below PAR_MIN = 2^15 amplitudes run in a single thread, where thread start-up costs more
// than the gate. No 2^n x 2^n matrix is ever built.
//
// Complex products are written out in real arithmetic, because std::complex operator* calls the
// NaN-checking __muldc3 without -ffast-math.
//
// Build (all members are defined in the class body, so including it in several translation
// units is ODR-safe):
//   g++ -O3 -march=native -fopenmp -std=c++17 -Isrc ...

#ifndef LL_STATEVECTOR_BACKEND_H
#define LL_STATEVECTOR_BACKEND_H

#ifdef _OPENMP
#include <omp.h>
#endif

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
    // hi = max of the two qubits, group g = p >> lo selects the bits between lo and hi (low
    // mid_bits of g) and above hi (the rest); the 2^lo low bits form the contiguous run.
    void cnot(int control, int target) override {
        check_qubit(control);
        check_qubit(target);
        if (control == target) throw std::invalid_argument("StateVectorBackend::cnot: control == target");

        const int lo = std::min(control, target), hi = std::max(control, target);
        const int mid_bits = hi - lo - 1;
        const ll n_mid = 1LL << mid_bits;
        const ll n_pairs = size_ >> 2;
        const ll cbit = 1LL << control, tbit = 1LL << target;
        Complex* psi = psi_.data();
        const bool par = size_ >= PAR_MIN;
#pragma omp parallel if (par)
        {
            ll p, end;
            thread_range(n_pairs, p, end);
            while (p < end) {
                const ll g = p >> lo;
                const ll run_end = std::min(end, (g + 1) << lo);
                const ll base = ((g >> mid_bits) << (hi + 1)) | ((g & (n_mid - 1)) << (lo + 1));
                Complex* p0 = psi + (base | cbit | (p & ((1LL << lo) - 1)));
                Complex* p1 = p0 + tbit;
                for (ll j = 0, len = run_end - p; j < len; ++j) std::swap(p0[j], p1[j]);
                p = run_end;
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

    // Measured, not derived: below this a single thread beats the cost of a parallel region.
    static constexpr ll PAR_MIN = 1LL << 15;

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

    // Contiguous share [lo, hi) of n work items for the calling thread; call inside a parallel region.
    static void thread_range(ll n, ll& lo, ll& hi) {
#ifdef _OPENMP
        const ll t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const ll t = 0, nt = 1;
#endif
        lo = n * t / nt;
        hi = n * (t + 1) / nt;
    }

    // kernel(a0, a1) acts on every amplitude pair (bit q = 0, bit q = 1).
    template <class K>
    void for_pairs(int q, K kernel) {
        check_qubit(q);
        const ll stride = 1LL << q;
        const ll n_pairs = size_ >> 1;
        Complex* psi = psi_.data();
        const bool par = size_ >= PAR_MIN;
#pragma omp parallel if (par)
        {
            ll p, end;
            thread_range(n_pairs, p, end);
            while (p < end) {
                const ll run_end = std::min(end, ((p >> q) + 1) << q);
                Complex* p0 = psi + (((p >> q) << (q + 1)) | (p & (stride - 1)));
                Complex* p1 = p0 + stride;
                for (ll j = 0, len = run_end - p; j < len; ++j) kernel(p0[j], p1[j]);
                p = run_end;
            }
        }
    }
};

}  // namespace ll

#endif  // LL_STATEVECTOR_BACKEND_H