"""Grover vs Qiskit Aer (Tier 2): industry-standard baseline for the gate-level backends.

Runs the identical Grover circuit on three simulators and times only the simulation:

  Qiskit Aer      AerSimulator(method="statevector"), circuit built and transpiled once
  EigenBackend    ll::EigenBackend through python/algorithms.py
  StateVector     ll::StateVectorBackend (OpenMP) through python/algorithms.py

  docs/images/algorithms_vs_qiskit.png   time vs state size 2^n, log-log, three curves
  docs/algorithms_vs_qiskit.json         raw timings, speedups, versions, gate counts

Circuit (same gate sequence as src/Algorithms.h): H on every qubit, then ITERS times
  oracle      X on the qubits where the marked state has a 0 bit, multi-controlled Z, same X layer
  diffusion   H X on every qubit, multi-controlled Z, X H on every qubit
with a single marked state. Multi-controlled Z is H(target) MCX(other qubits -> target) H(target),
which is exactly Z on |1...1>. Qubit q is bit q of the index in both Qiskit and this simulator.

What is timed. Qiskit: sim.run(transpiled_circuit, shots=1).result() with a save_probabilities
instruction, so the output is the same 2^n probabilities our backends return. Construction and
transpile are excluded. Ours: Grover.run through the Python binding, which includes the copy of
the probabilities into numpy. Each point is the best of REPEATS runs after a warm-up.

Reading the comparison.
  - Aer fuses gates by default and uses all cores (OMP_NUM_THREADS applies to both Aer and
    ours); both are Aer's normal configuration and are left unchanged.
  - Transpile uses optimization_level=0, so the circuit stays gate-for-gate; the script checks
    that Aer did not decompose the multi-controlled gates.
  - The v1.0 claim of 3.5x was measured on a different circuit. This benchmark does not
    reproduce it by construction, so the result is reported as is.

Needs: pip install qiskit qiskit-aer

Run: python3 benchmarks/algorithms_vs_qiskit.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import qiskit
    import qiskit_aer
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except ImportError as exc:
    print(f"Qiskit Aer is not available ({exc}).")
    print("Install it with:  pip install qiskit qiskit-aer")
    sys.exit(1)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import algorithms as alg  # noqa: E402

IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)
FIG_PATH = IMAGES / "algorithms_vs_qiskit.png"
JSON_PATH = ROOT / "docs" / "algorithms_vs_qiskit.json"

N_VALUES = [10, 12, 14, 16]
ITERS = 10
REPEATS = 5
V1_CLAIM = 3.5  # speedup over Qiskit Aer claimed for v1.0

BLUE, RED, GREEN = "#3b6ea5", "#d1495b", "#4c9a6a"

_checks = []


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def warn(name, ok, detail=""):
    """Reported but not counted in the pass/fail total."""
    print(f"  [{'PASS' if ok else 'WARN'}] {name}" + (f"  ({detail})" if detail else ""))


def marked_state(n):
    return (1 << n) // 3


def p_theory(n, k):
    return math.sin((2 * k + 1) * math.asin(2.0 ** (-n / 2))) ** 2


# ---------------------------------------------------------------------------
# Qiskit circuit
# ---------------------------------------------------------------------------

def append_mcz(qc, n):
    """Z on |1...1>: H(target) MCX(controls -> target) H(target)."""
    if n == 1:
        qc.z(0)
        return
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)


def build_grover(n, marked, iters):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    zeros = [q for q in range(n) if not (marked >> q) & 1]
    for _ in range(iters):
        if zeros:
            qc.x(zeros)
        append_mcz(qc, n)
        if zeros:
            qc.x(zeros)
        qc.h(range(n))
        qc.x(range(n))
        append_mcz(qc, n)
        qc.x(range(n))
        qc.h(range(n))
    qc.save_probabilities()
    return qc


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_qiskit(sim, tc):
    best, result = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = sim.run(tc, shots=1).result()
        best = min(best, time.perf_counter() - t0)
    if not result.success:
        raise RuntimeError(f"Aer run failed: {result.status}")
    return best, np.asarray(result.data(0)["probabilities"], dtype=float)


def time_ours(kind, n):
    marked = marked_state(n)
    be = alg.make_backend(kind, n)
    best, probs = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = alg.Grover.run(be, n, [marked], ITERS)
        best = min(best, time.perf_counter() - t0)
        probs = np.asarray(r.probabilities)
    return best, probs


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep():
    sim = AerSimulator(method="statevector")
    print(f"\n[1/3] Grover, {ITERS} iterations, single marked state, best of {REPEATS}")
    print(f"  qiskit {qiskit.__version__}, qiskit-aer {qiskit_aer.__version__}")
    print(f"  cores: {os.cpu_count()}   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")

    # Warm-up: Aer start-up and thread pools, not recorded.
    warm = transpile(build_grover(8, 5, 2), sim, optimization_level=0)
    sim.run(warm, shots=1).result()
    time_ours(alg.Backend.STATEVECTOR, 12)

    rows = []
    worst_agree, worst_theory = 0.0, 0.0
    for n in N_VALUES:
        marked = marked_state(n)
        qc = build_grover(n, marked, ITERS)
        tc = transpile(qc, sim, optimization_level=0)

        t_q, p_q = time_qiskit(sim, tc)
        t_e, p_e = time_ours(alg.Backend.EIGEN, n)
        t_s, p_s = time_ours(alg.Backend.STATEVECTOR, n)

        agree = max(float(np.max(np.abs(p_q - p_e))), float(np.max(np.abs(p_q - p_s))))
        theory_err = max(abs(float(p[marked]) - p_theory(n, ITERS)) for p in (p_q, p_e, p_s))
        worst_agree, worst_theory = max(worst_agree, agree), max(worst_theory, theory_err)

        rows.append({
            "n": n, "marked": marked,
            "qiskit_ms": 1e3 * t_q, "eigen_ms": 1e3 * t_e, "statevector_ms": 1e3 * t_s,
            "sv_vs_qiskit": t_q / t_s, "sv_vs_eigen": t_e / t_s, "eigen_vs_qiskit": t_q / t_e,
            "peak_qiskit": float(p_q[marked]), "peak_eigen": float(p_e[marked]),
            "peak_statevector": float(p_s[marked]), "peak_theory": p_theory(n, ITERS),
            "circuit_gates": int(qc.size()), "transpiled_gates": int(tc.size()),
        })
        print(f"  n={n} done", flush=True)
    return rows, worst_agree, worst_theory


def print_table(rows):
    print("\n  speed columns: how many times faster StateVector is (>1 means StateVector wins)")
    print("\n  n    2^n      qiskit[ms]  eigen[ms]   statevec[ms]  sv vs qiskit  sv vs eigen  "
          "peak P")
    for r in rows:
        print(f"  {r['n']:<4d} {2 ** r['n']:<8d} {r['qiskit_ms']:<11.1f} {r['eigen_ms']:<11.1f} "
              f"{r['statevector_ms']:<13.1f} {r['sv_vs_qiskit']:<13.2f} {r['sv_vs_eigen']:<12.2f} "
              f"{r['peak_statevector']:.6f}")
    print("\n  n    circuit gates  transpiled gates")
    for r in rows:
        print(f"  {r['n']:<4d} {r['circuit_gates']:<14d} {r['transpiled_gates']}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(rows):
    data = {
        "n_values": N_VALUES,
        "grover_iterations": ITERS,
        "repeats": REPEATS,
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "v1_claim_speedup_vs_aer": V1_CLAIM,
        "rows": rows,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"  saved {JSON_PATH.relative_to(ROOT)}")


def plot(rows):
    x = [2 ** r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, [r["qiskit_ms"] for r in rows], "D-", color=GREEN, lw=1.6, label="Qiskit Aer (statevector)")
    ax.plot(x, [r["eigen_ms"] for r in rows], "o-", color=BLUE, lw=1.6, label="Eigen backend")
    ax.plot(x, [r["statevector_ms"] for r in rows], "^-", color=RED, lw=1.6, label="StateVector backend (OpenMP)")

    last = rows[-1]
    ax.annotate(f"{last['sv_vs_qiskit']:.2f}x vs Aer", (x[-1], last["statevector_ms"]),
                xytext=(-95, -30), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->"))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$2^{{{r['n']}}}$" for r in rows])
    ax.set_xlabel("state size (amplitudes)")
    ax.set_ylabel(f"time for Grover, {ITERS} iterations [ms]")
    ax.set_title("Same Grover circuit: Qiskit Aer vs this simulator")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved {FIG_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Grover: Qiskit Aer vs Lightning-Lite backends")
    print("=" * 64)

    rows, worst_agree, worst_theory = sweep()

    print("\n[2/3] Timings")
    print_table(rows)

    big = rows[-1]
    print(f"\n  n={big['n']}: StateVector is {big['sv_vs_qiskit']:.2f}x Qiskit Aer, "
          f"{big['sv_vs_eigen']:.2f}x Eigen; Eigen is {big['eigen_vs_qiskit']:.2f}x Qiskit Aer")

    print("\n[3/3] Output and checks")
    save_json(rows)
    plot(rows)

    check("all three simulators agree on the full distribution: max |dP| < 1e-10",
          worst_agree < 1e-10, f"{worst_agree:.2e}")
    check("marked-state probability = sin^2((2k+1) theta) on all three, < 1e-9",
          worst_theory < 1e-9, f"{worst_theory:.2e}")
    check("figure and JSON written", FIG_PATH.exists() and JSON_PATH.exists())
    warn("Aer ran the circuit gate-for-gate (no decomposition of multi-controlled gates)",
         all(r["transpiled_gates"] == r["circuit_gates"] for r in rows),
         f"{rows[-1]['circuit_gates']} -> {rows[-1]['transpiled_gates']} gates at n={rows[-1]['n']}")
    warn(f"StateVector at least as fast as Qiskit Aer at every n",
         all(r["sv_vs_qiskit"] >= 1.0 for r in rows),
         f"min {min(r['sv_vs_qiskit'] for r in rows):.2f}x")
    warn(f"v1.0 claim of {V1_CLAIM}x over Aer reproduced on this circuit at n={big['n']}",
         big["sv_vs_qiskit"] >= V1_CLAIM, f"{big['sv_vs_qiskit']:.2f}x")

    print(f"\n{sum(_checks)}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())