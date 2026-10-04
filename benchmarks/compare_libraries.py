"""Grover across simulators (Tier 2): this simulator against Qiskit Aer, Cirq, Qulacs, PennyLane.

Runs the identical Grover circuit on five state-vector simulators and times only the simulation:

  ours        ll::StateVectorBackend (OpenMP) through python/algorithms.py
  Qiskit Aer  AerSimulator(method="statevector"), circuit built and transpiled once
  Cirq        cirq.Simulator(dtype=complex128), circuit built once
  Qulacs      QuantumCircuit.update_quantum_state on a preallocated QuantumState
  PennyLane   default.qubit executing a prebuilt tape (QNode tracing excluded)

  docs/images/library_comparison.png   time vs state size 2^n, log-log, one curve per library
  docs/library_comparison.json         raw timings, memory, success probabilities, versions

Circuit (same gate sequence as src/Algorithms.h and benchmarks/algorithms_vs_qiskit.py): H on every
qubit, then ITERS times
  oracle      X on the qubits where the marked state has a 0 bit, multi-controlled Z, same X layer
  diffusion   H X on every qubit, multi-controlled Z, X H on every qubit
with a single marked state, (2^n) // 3. Multi-controlled Z is Z on |1...1>. Qubit q is bit q of the
basis-state index everywhere: Cirq and PennyLane are big-endian natively, so their qubit order is
reversed to match.

Process isolation. Every (library, n) point runs in its own subprocess, so a library's import cost,
earlier allocations and thread pools do not leak into another point, and a crash, timeout or
memory blow-up only ends that library's sweep. The subprocess does a warm-up, prints READY, then
times REPEATS runs and keeps the best.

What is timed. One call that returns the full 2^n probability vector as a numpy array, for every
library. Circuit construction (and Aer's transpile) is excluded.

Peak memory. The parent process polls the subprocess's resident set size with psutil every 2 ms.
Reported: the absolute peak (MB) and the peak minus the RSS at READY, i.e. what the timed runs added
on top of imports, circuit and warm-up. The state itself is 16 * 2^n bytes (4 MiB at n = 18), so at
these sizes the delta is small and allocator behaviour matters as much as the library. Treat memory
as indicative.

Success probability. P(marked state) after ITERS iterations. Every library must agree with this
simulator, and with sin^2((2k+1) theta), to 1e-9.

Reference figures from the brief, not verified and not tested here: CAST reports an 8.03x speedup
over Qiskit on a 32-qubit CPU; Qiskit Aer fails at 30 qubits. This benchmark stops at n = 18 by
default (StateVectorBackend.MAX_QUBITS is 25), so it cannot reproduce either; they are printed and
saved as context only.

Missing libraries are skipped with a warning. A library that errors, times out or passes the memory
limit at some n is not run at larger n.

Needs: pip install qiskit qiskit-aer cirq qulacs pennylane psutil matplotlib numpy

Run: python3 benchmarks/compare_libraries.py
     python3 benchmarks/compare_libraries.py --n-values 10 12 14   (shorter sweep)
"""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

IMAGES = ROOT / "docs" / "images"
FIG_PATH = IMAGES / "library_comparison.png"
JSON_PATH = ROOT / "docs" / "library_comparison.json"

N_VALUES = [10, 12, 14, 16, 18]
ITERS = 10
REPEATS = 3
TIMEOUT_S = 900          # per (library, n) point
POLL_S = 0.002
AGREE_TOL = 1e-9

# key: (display name, import module, pip distribution name, colour, marker)
LIBRARIES = {
    "ours": ("This simulator (StateVector)", "algorithms", None, "#d1495b", "^"),
    "aer": ("Qiskit Aer", "qiskit_aer", "qiskit-aer", "#4c9a6a", "D"),
    "cirq": ("Cirq", "cirq", "cirq-core", "#3b6ea5", "o"),
    "qulacs": ("Qulacs", "qulacs", "qulacs", "#8a6bb5", "s"),
    "pennylane": ("PennyLane", "pennylane", "pennylane", "#c58b2a", "v"),
}

REFERENCE_CLAIMS = [
    {"claim": "CAST reports an 8.03x speedup over Qiskit on a 32-qubit CPU",
     "source": "project brief (citation marker 9), not verified"},
    {"claim": "Qiskit Aer fails at 30 qubits",
     "source": "project brief (citation marker 3), not verified"},
]

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
# Per-library circuit construction. Each setup_* returns run() -> probabilities, little endian.
# ---------------------------------------------------------------------------

def setup_ours(n):
    import algorithms as alg
    marked = marked_state(n)
    be = alg.make_backend(alg.Backend.STATEVECTOR, n)

    def run():
        return np.asarray(alg.Grover.run(be, n, [marked], ITERS).probabilities, dtype=float)
    return run


def setup_aer(n):
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    marked = marked_state(n)
    zeros = [q for q in range(n) if not (marked >> q) & 1]

    def mcz(qc):
        if n == 1:
            qc.z(0)
            return
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)

    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(ITERS):
        if zeros:
            qc.x(zeros)
        mcz(qc)
        if zeros:
            qc.x(zeros)
        qc.h(range(n))
        qc.x(range(n))
        mcz(qc)
        qc.x(range(n))
        qc.h(range(n))
    qc.save_probabilities()
    sim = AerSimulator(method="statevector")
    tc = transpile(qc, sim, optimization_level=0)

    def run():
        result = sim.run(tc, shots=1).result()
        if not result.success:
            raise RuntimeError(f"Aer run failed: {result.status}")
        return np.asarray(result.data(0)["probabilities"], dtype=float)
    return run


def setup_cirq(n):
    import cirq

    marked = marked_state(n)
    qs = cirq.LineQubit.range(n)
    zeros = [qs[q] for q in range(n) if not (marked >> q) & 1]

    def mcz():
        return cirq.Z(qs[0]) if n == 1 else cirq.Z(qs[-1]).controlled_by(*qs[:-1])

    ops = [cirq.H(q) for q in qs]
    for _ in range(ITERS):
        ops += [cirq.X(q) for q in zeros]
        ops.append(mcz())
        ops += [cirq.X(q) for q in zeros]
        ops += [cirq.H(q) for q in qs]
        ops += [cirq.X(q) for q in qs]
        ops.append(mcz())
        ops += [cirq.X(q) for q in qs]
        ops += [cirq.H(q) for q in qs]
    circuit = cirq.Circuit(ops)
    sim = cirq.Simulator(dtype=np.complex128)
    order = qs[::-1]  # Cirq lists the most significant qubit first; reverse for little endian

    def run():
        state = sim.simulate(circuit, qubit_order=order).final_state_vector
        return np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    return run


def setup_qulacs(n):
    from qulacs import QuantumCircuit, QuantumState
    from qulacs.gate import DenseMatrix

    marked = marked_state(n)
    zeros = [q for q in range(n) if not (marked >> q) & 1]
    zmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    def add_mcz(circ):
        if n == 1:
            circ.add_Z_gate(0)
            return
        gate = DenseMatrix([n - 1], zmat)
        for c in range(n - 1):
            gate.add_control_qubit(c, 1)
        circ.add_gate(gate)

    circ = QuantumCircuit(n)
    for q in range(n):
        circ.add_H_gate(q)
    for _ in range(ITERS):
        for q in zeros:
            circ.add_X_gate(q)
        add_mcz(circ)
        for q in zeros:
            circ.add_X_gate(q)
        for q in range(n):
            circ.add_H_gate(q)
            circ.add_X_gate(q)
        add_mcz(circ)
        for q in range(n):
            circ.add_X_gate(q)
            circ.add_H_gate(q)
    state = QuantumState(n)

    def run():
        state.set_zero_state()
        circ.update_quantum_state(state)
        return np.abs(np.asarray(state.get_vector())) ** 2
    return run


def setup_pennylane(n):
    import pennylane as qml

    marked = marked_state(n)

    def w(q):  # wire of qubit q: default.qubit is big endian, so wire n-1-q is bit q
        return n - 1 - q

    zeros = [w(q) for q in range(n) if not (marked >> q) & 1]

    def mcz():
        if n == 1:
            return qml.PauliZ(wires=0)
        return qml.ctrl(qml.PauliZ(wires=w(n - 1)), control=[w(q) for q in range(n - 1)])

    allw = list(range(n))
    ops = [qml.Hadamard(wires=i) for i in allw]
    for _ in range(ITERS):
        ops += [qml.PauliX(wires=i) for i in zeros]
        ops.append(mcz())
        ops += [qml.PauliX(wires=i) for i in zeros]
        ops += [qml.Hadamard(wires=i) for i in allw]
        ops += [qml.PauliX(wires=i) for i in allw]
        ops.append(mcz())
        ops += [qml.PauliX(wires=i) for i in allw]
        ops += [qml.Hadamard(wires=i) for i in allw]
    tape = qml.tape.QuantumScript(ops, [qml.probs(wires=allw)])
    dev = qml.device("default.qubit", wires=n)

    def run():
        res = dev.execute(tape)
        if isinstance(res, (list, tuple)):
            res = res[0]
        return np.asarray(res, dtype=float)
    return run


SETUP = {"ours": setup_ours, "aer": setup_aer, "cirq": setup_cirq,
         "qulacs": setup_qulacs, "pennylane": setup_pennylane}


# ---------------------------------------------------------------------------
# Worker: one (library, n) point in its own process
# ---------------------------------------------------------------------------

def worker(lib, n):
    SETUP[lib](8)()                      # warm-up on a small circuit: thread pools, lazy imports
    run = SETUP[lib](n)
    print("@@READY", flush=True)

    best, probs = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        probs = run()
        best = min(best, time.perf_counter() - t0)

    marked = marked_state(n)
    print("@@RESULT " + json.dumps({
        "time_ms": 1e3 * best,
        "p_success": float(probs[marked]),
        "p_total": float(np.sum(probs)),
        "argmax": int(np.argmax(probs)),
        "size": int(probs.size),
    }), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Parent: spawn a worker, watch its memory, enforce the limits
# ---------------------------------------------------------------------------

def run_point(lib, n, timeout_s, mem_limit_mb):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", lib, str(n)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    state = {"last": 0.0, "base": None, "peak": 0.0, "reason": None}
    stop = threading.Event()

    def poll():
        if psutil is None:
            return
        try:
            ps = psutil.Process(proc.pid)
        except psutil.Error:
            return
        while not stop.is_set() and proc.poll() is None:
            try:
                rss = ps.memory_info().rss / 2 ** 20
            except psutil.Error:
                break
            state["last"] = rss
            state["peak"] = max(state["peak"], rss)
            if mem_limit_mb and rss > mem_limit_mb:
                state["reason"] = "memory_limit"
                proc.kill()
                break
            stop.wait(POLL_S)

    def on_timeout():
        state["reason"] = "timeout"
        proc.kill()

    poller = threading.Thread(target=poll, daemon=True)
    timer = threading.Timer(timeout_s, on_timeout)
    poller.start()
    timer.start()

    result, tail = None, deque(maxlen=4)
    for line in proc.stdout:
        if line.startswith("@@READY"):
            state["base"] = state["last"]
            state["peak"] = state["last"]
        elif line.startswith("@@RESULT "):
            result = json.loads(line[len("@@RESULT "):])
        elif line.strip():
            tail.append(line.strip())
    proc.wait()
    stop.set()
    timer.cancel()
    poller.join(timeout=1.0)

    out = {"n": n, "status": "ok"}
    if result is None or proc.returncode != 0:
        out["status"] = state["reason"] or "error: " + (tail[-1] if tail else f"exit {proc.returncode}")
        return out
    out.update(result)
    if psutil is not None and state["base"] is not None:
        out["peak_rss_mb"] = state["peak"]
        out["delta_mb"] = max(0.0, state["peak"] - state["base"])
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def dist_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def available_libraries():
    found, missing = [], []
    for key, (display, module, dist, _, _) in LIBRARIES.items():
        if importlib.util.find_spec(module) is None:
            missing.append(key)
            print(f"  WARNING: {display} is not installed, skipping"
                  + (f"  (pip install {dist})" if dist else ""))
        else:
            found.append(key)
    return found, missing


def sweep(libs, n_values, timeout_s, mem_limit_mb):
    results = {lib: [] for lib in libs}
    print(f"\n[1/3] Grover, {ITERS} iterations, single marked state, best of {REPEATS}, "
          f"one subprocess per point")
    print(f"  cores: {os.cpu_count()}   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"  timeout {timeout_s:.0f} s per point, memory limit "
          f"{'none' if not mem_limit_mb else f'{mem_limit_mb:.0f} MB'}")
    for lib in libs:
        for n in n_values:
            r = run_point(lib, n, timeout_s, mem_limit_mb)
            results[lib].append(r)
            if r["status"] != "ok":
                print(f"  {LIBRARIES[lib][0]} n={n}: {r['status']}; no larger n for this library",
                      flush=True)
                break
            print(f"  {LIBRARIES[lib][0]} n={n} done ({r['time_ms']:.1f} ms)", flush=True)
    return results


def by_n(results, lib):
    return {r["n"]: r for r in results.get(lib, []) if r["status"] == "ok"}


def print_tables(results, libs, n_values):
    names = {lib: LIBRARIES[lib][0] for lib in libs}
    w = 15

    def table(title, fmt, key):
        print(f"\n  {title}")
        print("  n    2^n      " + "".join(f"{names[l][:w - 1]:<{w}}" for l in libs))
        for n in n_values:
            row = f"  {n:<4d} {2 ** n:<8d} "
            for lib in libs:
                r = by_n(results, lib).get(n)
                row += f"{(fmt(r[key]) if r and r.get(key) is not None else '-'):<{w}}"
            print(row)

    table("wall time [ms], best of %d" % REPEATS, lambda v: f"{v:.1f}", "time_ms")
    table("peak RSS [MB] (absolute) / added by the timed runs", lambda v: f"{v:.0f}", "peak_rss_mb")
    table("added by timed runs [MB]", lambda v: f"{v:.1f}", "delta_mb")
    table("success probability P(marked)", lambda v: f"{v:.9f}", "p_success")

    others = [l for l in libs if l != "ours"]
    if "ours" in libs and others:
        print("\n  speedup of this simulator (other time / our time, >1 means ours is faster)")
        print("  n    " + "".join(f"{names[l][:w - 1]:<{w}}" for l in others))
        ours = by_n(results, "ours")
        for n in n_values:
            row = f"  {n:<4d} "
            for lib in others:
                r = by_n(results, lib).get(n)
                row += f"{(f'{r['time_ms'] / ours[n]['time_ms']:.2f}x' if r and n in ours else '-'):<{w}}"
            print(row)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(results, libs, missing, n_values):
    data = {
        "n_values": n_values,
        "grover_iterations": ITERS,
        "repeats": REPEATS,
        "timeout_s": TIMEOUT_S,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "psutil_available": psutil is not None,
        "versions": {lib: (dist_version(LIBRARIES[lib][2]) if LIBRARIES[lib][2] else None)
                     for lib in libs},
        "libraries_skipped": missing,
        "reference_claims": REFERENCE_CLAIMS,
        "rows": {lib: results[lib] for lib in libs},
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"  saved {JSON_PATH.relative_to(ROOT)}")


def plot(results, libs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ticks = set()
    for lib in libs:
        display, _, _, color, marker = LIBRARIES[lib]
        pts = sorted(by_n(results, lib).items())
        if not pts:
            continue
        x = [2 ** n for n, _ in pts]
        y = [r["time_ms"] for _, r in pts]
        ticks.update(pts_n for pts_n, _ in pts)
        if lib == "ours":
            ax.plot(x, y, marker + "-", color=color, lw=2.8, ms=8, zorder=5, label=display)
            ax.annotate(f"{y[-1]:.0f} ms", (x[-1], y[-1]), xytext=(-8, -18),
                        textcoords="offset points", fontsize=9, color=color)
        else:
            ax.plot(x, y, marker + "-", color=color, lw=1.3, ms=5, alpha=0.8, zorder=3, label=display)

    ns = sorted(ticks)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2 ** n for n in ns])
    ax.set_xticklabels([f"$2^{{{n}}}$" for n in ns])
    ax.set_xlabel("state size (amplitudes)")
    ax.set_ylabel(f"time for Grover, {ITERS} iterations [ms]")
    ax.set_title("Same Grover circuit across state-vector simulators")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved {FIG_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("LIB", "N"), help=argparse.SUPPRESS)
    ap.add_argument("--n-values", type=int, nargs="+", default=N_VALUES)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per (library, n) point")
    ap.add_argument("--mem-limit-mb", type=float, default=None,
                    help="kill a point above this RSS; default 80%% of physical memory")
    args = ap.parse_args()

    if args.worker:
        return worker(args.worker[0], int(args.worker[1]))

    print("=" * 64)
    print("Grover: this simulator vs Qiskit Aer, Cirq, Qulacs, PennyLane")
    print("=" * 64)

    n_values = sorted(set(args.n_values))
    mem_limit = args.mem_limit_mb
    if psutil is None:
        print("  WARNING: psutil is not installed, memory columns will be empty "
              "(pip install psutil)")
    elif mem_limit is None:
        mem_limit = 0.8 * psutil.virtual_memory().total / 2 ** 20

    libs, missing = available_libraries()
    if "ours" not in libs:
        print("\nThis simulator's python/algorithms.py could not be found; nothing to compare.")
        return 1

    results = sweep(libs, n_values, args.timeout, mem_limit)

    print("\n[2/3] Results")
    print_tables(results, libs, n_values)
    print("\n  Reference figures from the brief, not reproduced here (this sweep stops at "
          f"n = {n_values[-1]}):")
    for c in REFERENCE_CLAIMS:
        print(f"    - {c['claim']}  [{c['source']}]")

    print("\n[3/3] Output and checks")
    save_json(results, libs, missing, n_values)
    plot(results, libs)

    ours = by_n(results, "ours")
    check("this simulator completed every requested n", all(n in ours for n in n_values),
          f"{len(ours)}/{len(n_values)} points")
    err_theory = max((abs(r["p_success"] - p_theory(n, ITERS)) for n, r in ours.items()), default=1.0)
    check(f"marked-state probability = sin^2((2k+1) theta) for this simulator, < {AGREE_TOL:g}",
          err_theory < AGREE_TOL, f"{err_theory:.2e}")

    worst = 0.0
    for lib in libs:
        if lib == "ours":
            continue
        for n, r in by_n(results, lib).items():
            if n in ours:
                worst = max(worst, abs(r["p_success"] - ours[n]["p_success"]))
    compared = [lib for lib in libs if lib != "ours" and by_n(results, lib)]
    check(f"success probability matches this simulator on every library that ran, < {AGREE_TOL:g}",
          bool(compared) and worst < AGREE_TOL, f"{worst:.2e}, {len(compared)} libraries compared")

    off = max((abs(r["p_total"] - 1.0) for lib in libs for r in by_n(results, lib).values()), default=1.0)
    check("every probability vector sums to 1, < 1e-9", off < 1e-9, f"{off:.2e}")
    bad_argmax = [(lib, n) for lib in libs for n, r in by_n(results, lib).items()
                  if r["argmax"] != marked_state(n)]
    check("most probable state is the marked state everywhere", not bad_argmax,
          "none off" if not bad_argmax else str(bad_argmax[:3]))
    check("figure and JSON written", FIG_PATH.exists() and JSON_PATH.exists())

    for lib in libs:
        if lib == "ours":
            continue
        rows = by_n(results, lib)
        common = [n for n in rows if n in ours]
        if common:
            slowest = min(rows[n]["time_ms"] / ours[n]["time_ms"] for n in common)
            warn(f"this simulator at least as fast as {LIBRARIES[lib][0]} at every n",
                 all(rows[n]["time_ms"] >= ours[n]["time_ms"] for n in common),
                 f"min speedup {slowest:.2f}x")
        failed = [r for r in results[lib] if r["status"] != "ok"]
        if failed:
            warn(f"{LIBRARIES[lib][0]} completed every requested n", False,
                 f"stopped at n={failed[0]['n']}: {failed[0]['status']}")

    print(f"\n{sum(_checks)}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())