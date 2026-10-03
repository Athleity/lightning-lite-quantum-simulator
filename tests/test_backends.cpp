// Cross-backend test: ReferenceBackend (Backend.h) vs EigenBackend (EigenBackend.h).
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/include/eigen3 -Isrc tests/test_backends.cpp -o build/test_backends
//   ./build/test_backends
//
// Checks that both backends give the same physics through the whole Backend interface:
// the fixed 4-qubit circuit, closed-form states, randomised circuits at 1..6 qubits (every gate
// type on every qubit and qubit pair), and the documented exceptions.
// Exit code 0 if every check passes, 1 otherwise.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "Backend.h"
#include "EigenBackend.h"

using namespace ll;
using Complex = std::complex<double>;

static int g_total = 0, g_failed = 0;

static void check(const std::string& name, bool ok, double value = -1.0) {
    ++g_total;
    if (!ok) ++g_failed;
    if (value >= 0.0)
        std::printf("  [%s] %-58s %.2e\n", ok ? "PASS" : "FAIL", name.c_str(), value);
    else
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
}

static double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static double max_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) return 1e300;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static bool is_distribution(const std::vector<double>& p, double tol, double* sum_err) {
    double s = 0.0;
    bool nonneg = true;
    for (double v : p) {
        s += v;
        nonneg = nonneg && v >= 0.0;
    }
    *sum_err = std::abs(s - 1.0);
    return nonneg && *sum_err < tol;
}

// ---------------------------------------------------------------------------
// Circuits
// ---------------------------------------------------------------------------

static void spec_circuit(Backend& b) {
    b.reset(4);
    for (int q = 0; q < 4; ++q) b.h(q);
    b.cnot(0, 1);
    b.cnot(2, 3);
    b.rz(0, 0.3);
    b.rx(1, 0.5);
    b.ry(2, -0.2);
    b.mcz({0, 1, 2, 3});
}

struct Op {
    int kind;  // 0 h, 1 x, 2 rx, 3 ry, 4 rz, 5 cnot, 6 mcz
    int a, b;
    double t;
    std::vector<int> qs;
};

static std::vector<Op> random_circuit(int n, int depth, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> qd(0, n - 1);
    std::uniform_int_distribution<int> kd(0, n >= 2 ? 6 : 4);
    std::uniform_real_distribution<double> ang(-3.2, 3.2);
    std::bernoulli_distribution coin(0.5);
    std::vector<Op> ops;
    for (int d = 0; d < depth; ++d) {
        Op op{kd(rng), qd(rng), 0, ang(rng), {}};
        if (op.kind == 5) {
            do op.b = qd(rng); while (op.b == op.a);
        } else if (op.kind == 6) {
            for (int q = 0; q < n; ++q)
                if (coin(rng)) op.qs.push_back(q);
            if (op.qs.empty()) op.qs.push_back(op.a);
        }
        ops.push_back(op);
    }
    return ops;
}

static void run_ops(Backend& be, const std::vector<Op>& ops) {
    for (const Op& op : ops) {
        switch (op.kind) {
            case 0: be.h(op.a); break;
            case 1: be.x(op.a); break;
            case 2: be.rx(op.a, op.t); break;
            case 3: be.ry(op.a, op.t); break;
            case 4: be.rz(op.a, op.t); break;
            case 5: be.cnot(op.a, op.b); break;
            default: be.mcz(op.qs); break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_spec_circuit() {
    std::printf("\n[1] Fixed 4-qubit circuit (H x4, CNOT(0,1), CNOT(2,3), RZ, RX, RY, MCZ)\n");
    ReferenceBackend ref(4);
    EigenBackend eig(4);
    spec_circuit(ref);
    spec_circuit(eig);

    check("probabilities: max |P_ref - P_eigen| < 1e-12",
          max_diff(ref.probabilities(), eig.probabilities()) < 1e-12,
          max_diff(ref.probabilities(), eig.probabilities()));
    check("state: max |psi_ref - psi_eigen| < 1e-12",
          max_diff(ref.state(), eig.state()) < 1e-12, max_diff(ref.state(), eig.state()));

    double e_ref = 0.0, e_eig = 0.0;
    check("Reference probabilities: sum = 1, all >= 0",
          is_distribution(ref.probabilities(), 1e-12, &e_ref), e_ref);
    check("Eigen probabilities: sum = 1, all >= 0",
          is_distribution(eig.probabilities(), 1e-12, &e_eig), e_eig);

    double norm = 0.0;
    for (const Complex& a : eig.state()) norm += std::norm(a);
    check("Eigen state norm = 1", std::abs(norm - 1.0) < 1e-12, std::abs(norm - 1.0));
}

template <class B>
static void closed_form(const char* name) {
    std::printf("\n[2] Closed-form states: %s\n", name);
    const double r = 1.0 / std::sqrt(2.0);
    B b;

    b.reset(4);
    b.x(2);
    check("X on qubit 2 -> basis index 4 (qubit q is bit q)", b.probabilities()[4] > 1.0 - 1e-14);

    b.reset(2);
    b.h(0);
    b.cnot(0, 1);
    const auto p = b.probabilities();
    check("Bell state: P(00) = P(11) = 1/2",
          std::abs(p[0] - 0.5) < 1e-14 && std::abs(p[3] - 0.5) < 1e-14 && p[1] < 1e-14 && p[2] < 1e-14);

    b.reset(3);
    b.x(2);
    b.cnot(2, 0);
    check("CNOT(2, 0) on |100> -> |101> (control above target)", b.probabilities()[5] > 1.0 - 1e-14);

    b.reset(3);
    b.x(0);
    b.cnot(0, 2);
    check("CNOT(0, 2) on |001> -> |101> (control below target)", b.probabilities()[5] > 1.0 - 1e-14);

    b.reset(1);
    b.ry(0, 3.14159265358979323846);
    check("RY(pi)|0> = |1>", b.probabilities()[1] > 1.0 - 1e-14);

    b.reset(1);
    b.rx(0, 3.14159265358979323846);
    check("RX(pi)|0> = -i|1>", std::abs(b.state()[1] - Complex(0.0, -1.0)) < 1e-14);

    b.reset(1);
    b.h(0);
    b.rz(0, 0.7);
    const auto s = b.state();
    check("H then RZ(t): amplitudes e^{-it/2}/sqrt2 and e^{+it/2}/sqrt2",
          std::abs(s[0] - r * std::polar(1.0, -0.35)) < 1e-14 &&
              std::abs(s[1] - r * std::polar(1.0, 0.35)) < 1e-14);

    b.reset(3);
    for (int q = 0; q < 3; ++q) b.h(q);
    b.mcz({0, 1, 2});
    const auto m = b.state();
    bool ok = std::abs(m[7] + r * r * r) < 1e-14;
    for (int i = 0; i < 7; ++i) ok = ok && std::abs(m[i] - r * r * r) < 1e-14;
    check("MCZ on |+++>: only |111> flips sign", ok);

    b.reset(2);
    b.h(0);
    b.h(1);
    b.mcz({1});
    check("MCZ on one qubit equals Z", std::abs(b.state()[2] + 0.5) < 1e-14 && std::abs(b.state()[0] - 0.5) < 1e-14);
}

static void test_random_circuits() {
    std::printf("\n[3] Random circuits, Reference vs Eigen (200 gates each)\n");
    std::mt19937_64 rng(2024);
    for (int n = 1; n <= 6; ++n) {
        double worst = 0.0;
        for (int trial = 0; trial < 5; ++trial) {
            const auto ops = random_circuit(n, 200, rng);
            ReferenceBackend ref(n);
            EigenBackend eig(n);
            ref.reset(n);
            eig.reset(n);
            run_ops(ref, ops);
            run_ops(eig, ops);
            worst = std::max(worst, max_diff(ref.state(), eig.state()));
        }
        check("n = " + std::to_string(n) + ": max |psi_ref - psi_eigen| < 1e-12", worst < 1e-12, worst);
    }

    double worst_norm = 0.0;
    {
        const auto ops = random_circuit(6, 500, rng);
        EigenBackend eig(6);
        run_ops(eig, ops);
        double norm = 0.0;
        for (const Complex& a : eig.state()) norm += std::norm(a);
        worst_norm = std::abs(norm - 1.0);
    }
    check("Eigen norm preserved after 500 random gates (n = 6)", worst_norm < 1e-12, worst_norm);

    // Every (control, target) pair and every single-qubit gate on a generic 5-qubit state.
    double worst_pair = 0.0;
    for (int c = 0; c < 5; ++c) {
        for (int t = 0; t < 5; ++t) {
            if (c == t) continue;
            ReferenceBackend ref(5);
            EigenBackend eig(5);
            for (Backend* b : {static_cast<Backend*>(&ref), static_cast<Backend*>(&eig)}) {
                for (int q = 0; q < 5; ++q) {
                    b->ry(q, 0.3 + 0.4 * q);
                    b->rz(q, 0.1 * (q + 1));
                }
                b->cnot(c, t);
            }
            worst_pair = std::max(worst_pair, max_diff(ref.state(), eig.state()));
        }
    }
    check("all 20 (control, target) pairs at n = 5", worst_pair < 1e-12, worst_pair);

    double worst_gate = 0.0;
    for (int q = 0; q < 5; ++q) {
        ReferenceBackend ref(5);
        EigenBackend eig(5);
        for (Backend* b : {static_cast<Backend*>(&ref), static_cast<Backend*>(&eig)}) {
            for (int k = 0; k < 5; ++k) b->ry(k, 0.5 + 0.3 * k);
            b->h(q);
            b->x(q);
            b->rx(q, 0.9);
            b->ry(q, -1.1);
            b->rz(q, 2.3);
        }
        worst_gate = std::max(worst_gate, max_diff(ref.state(), eig.state()));
    }
    check("every single-qubit gate on every qubit at n = 5", worst_gate < 1e-12, worst_gate);
}

static bool throws_invalid(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

template <class B>
static void exceptions(const char* name, int max_qubits) {
    std::printf("\n[4] Exceptions: %s\n", name);
    const double nan = std::nan("");
    B b;
    b.reset(3);

    check("reset(0) throws", throws_invalid([&] { b.reset(0); }));
    check("reset(MAX_QUBITS + 1) throws", throws_invalid([&] { b.reset(max_qubits + 1); }));
    b.reset(3);
    check("h(-1) throws", throws_invalid([&] { b.h(-1); }));
    check("x(3) throws (n = 3)", throws_invalid([&] { b.x(3); }));
    check("rx(0, NaN) throws", throws_invalid([&] { b.rx(0, nan); }));
    check("ry(0, inf) throws", throws_invalid([&] { b.ry(0, INFINITY); }));
    check("rz(5, 0.1) throws", throws_invalid([&] { b.rz(5, 0.1); }));
    check("cnot(1, 1) throws", throws_invalid([&] { b.cnot(1, 1); }));
    check("cnot(0, 3) throws", throws_invalid([&] { b.cnot(0, 3); }));
    check("mcz({}) throws", throws_invalid([&] { b.mcz({}); }));
    check("mcz({0, 0}) throws (duplicate)", throws_invalid([&] { b.mcz({0, 0}); }));
    check("mcz({0, 4}) throws", throws_invalid([&] { b.mcz({0, 4}); }));
    check("n_qubits() == 3 after failed calls", b.n_qubits() == 3);
}

int main() {
    std::printf("Backend cross-check: ReferenceBackend vs EigenBackend\n");

    test_spec_circuit();
    closed_form<ReferenceBackend>("ReferenceBackend");
    closed_form<EigenBackend>("EigenBackend");
    test_random_circuits();
    exceptions<ReferenceBackend>("ReferenceBackend", ReferenceBackend::MAX_QUBITS);
    exceptions<EigenBackend>("EigenBackend", EigenBackend::MAX_QUBITS);

    std::printf("\n%d/%d checks pass\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}