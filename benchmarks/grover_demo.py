"""Grover walkthrough (Tier 2, interview slide).

Four parts, all on the gate-level backends through python/algorithms.py:

  1. 4-qubit search: probability of every basis state after k = 0..3 Grover iterations, and the
     marked-state probability climbing from 1/16 to its maximum at k = 3
  2. 8-qubit search for a hidden state, with the automatically chosen iteration count
  3. Backend comparison at n = 10, 12, 14, 16 (same circuit, fixed iteration count)
  4. docs/images/grover_demo.png (left: P(z) at k = 0..3, right: P(marked) vs k up to 4)
     and docs/grover_demo.json (all numbers above)

Theory: with M marked of N states, theta = asin sqrt(M/N) and
P(marked after k iterations) = sin^2((2k + 1) theta), maximal at k = round(pi/(4 theta) - 1/2).
Every simulated probability is checked against this formula.

Run: python3 benchmarks/grover_demo.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import algorithms as alg  # noqa: E402

IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)
FIG_PATH = IMAGES / "grover_demo.png"
JSON_PATH = ROOT / "docs" / "grover_demo.json"

N4, MARKED4 = 4, 5
K_MAX4 = 4  # k = 4 overshoots; it is simulated for the right-hand plot
N8 = 8
BACKEND_N = [10, 12, 14, 16]
BACKEND_ITERS = 10
REPEATS = 3

BLUE, RED, GREY = "#3b6ea5", "#d1495b", "#999999"
K_COLOURS = ["#c9d3df", "#9db7d5", "#6b95c4", RED]

_checks = []


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def theta_of(n_qubits, n_marked=1):
    return math.asin(math.sqrt(n_marked / 2 ** n_qubits))


def p_theory(n_qubits, k, n_marked=1):
    return math.sin((2 * k + 1) * theta_of(n_qubits, n_marked)) ** 2


def k_optimal(n_qubits, n_marked=1):
    return max(0, int(math.floor(math.pi / (4 * theta_of(n_qubits, n_marked)) - 0.5 + 0.5)))


# ---------------------------------------------------------------------------
# 1. Four-qubit walkthrough
# ---------------------------------------------------------------------------

def four_qubit():
    print(f"\n[1/4] 4-qubit Grover, marked state {MARKED4} = |{MARKED4:04b}>")
    dists, p_marked = [], []
    for k in range(K_MAX4 + 1):
        r = alg.run_grover(N4, [MARKED4], iterations=k)
        dists.append(np.asarray(r.probabilities))
        p_marked.append(float(dists[-1][MARKED4]))

    print("\n  state   " + "".join(f"k={k:<8d}" for k in range(4)))
    for z in range(2 ** N4):
        tag = " <" if z == MARKED4 else "  "
        print(f"  {z:04b}{tag} " + "".join(f"{dists[k][z]:<10.4f}" for k in range(4)))

    print("\n  k    P(marked)   theory      P(each unmarked)")
    for k in range(K_MAX4 + 1):
        others = (1.0 - p_marked[k]) / (2 ** N4 - 1)
        print(f"  {k:<4d} {p_marked[k]:<11.6f} {p_theory(N4, k):<11.6f} {others:.6f}")

    auto = alg.run_grover(N4, [MARKED4])
    print(f"\n  automatic iteration count: {auto.iterations}, found {auto.marked_index}, "
          f"P = {auto.probabilities[auto.marked_index]:.4f}")
    return dists, p_marked, auto


# ---------------------------------------------------------------------------
# 2. Eight-qubit hidden state
# ---------------------------------------------------------------------------

def eight_qubit():
    rng = np.random.default_rng(2026)
    hidden = int(rng.integers(0, 2 ** N8))
    print(f"\n[2/4] 8-qubit Grover, hidden state {hidden} = |{hidden:08b}>")
    r = alg.run_grover(N8, [hidden])
    p = np.asarray(r.probabilities)
    peak = float(p[hidden])
    print(f"  iterations      {r.iterations}   (theory: {k_optimal(N8)})")
    print(f"  found           {r.marked_index}")
    print(f"  P(hidden)       {peak:.6f}   (theory: {p_theory(N8, r.iterations):.6f})")
    print(f"  P(any other)    {(1.0 - peak) / (2 ** N8 - 1):.2e} each")
    print(f"  oracle calls    {r.iterations} vs {2 ** N8 // 2} expected for classical search "
          f"({2 ** N8 // 2 / r.iterations:.1f}x fewer)")
    return hidden, r, peak


# ---------------------------------------------------------------------------
# 3. Backend comparison
# ---------------------------------------------------------------------------

def backends():
    print(f"\n[3/4] Backend comparison, {BACKEND_ITERS} Grover iterations, best of {REPEATS}")
    print(f"  cores: {os.cpu_count()}   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    alg.run_grover(12, [1], backend=alg.Backend.STATEVECTOR, iterations=BACKEND_ITERS)  # warm-up

    ms = alg.benchmark_backends(
        lambda n, b: alg.run_grover(n, [1], backend=b, iterations=BACKEND_ITERS),
        BACKEND_N, repeats=REPEATS, reference_max_n=max(BACKEND_N))

    print("\n  n    2^n      reference[ms]  eigen[ms]   statevec[ms]  eigen/sv   ref/sv")
    for i, n in enumerate(BACKEND_N):
        ref, eig, sv = ms["reference"][i], ms["eigen"][i], ms["statevector"][i]
        print(f"  {n:<4d} {2 ** n:<8d} {ref:<14.1f} {eig:<11.1f} {sv:<13.1f} "
              f"{eig / sv:<10.2f} {ref / sv:.1f}x")

    # Same circuit on all three backends must give the same distribution.
    n = BACKEND_N[0]
    ps = [np.asarray(alg.run_grover(n, [1], backend=b, iterations=BACKEND_ITERS).probabilities)
          for b in alg.Backend]
    spread = max(float(np.max(np.abs(ps[0] - p))) for p in ps[1:])
    return ms, spread


# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------

def plot(dists, p_marked):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.8, 1]})

    x = np.arange(2 ** N4)
    w = 0.2
    for k in range(4):
        a0.bar(x + (k - 1.5) * w, dists[k], width=w, color=K_COLOURS[k], label=f"k = {k}")
    a0.set_xticks(x)
    a0.set_xticklabels([f"{z:04b}" for z in x], rotation=90, fontsize=8)
    for lab in a0.get_xticklabels():
        if lab.get_text() == f"{MARKED4:04b}":
            lab.set_color(RED)
            lab.set_fontweight("bold")
    a0.set_xlabel("basis state (marked state in red)")
    a0.set_ylabel("probability")
    a0.set_title("4-qubit Grover: amplitude moves onto the marked state")
    a0.legend(frameon=False)
    a0.grid(axis="y", alpha=0.3)

    ks = np.linspace(0, K_MAX4, 200)
    a1.plot(ks, [p_theory(N4, k) for k in ks], color=GREY, ls=":", lw=1.3, label=r"$\sin^2((2k+1)\theta)$")
    a1.plot(range(K_MAX4 + 1), p_marked, "o-", color=BLUE, lw=1.6, label="simulated")
    k_opt = k_optimal(N4)
    a1.plot([k_opt], [p_marked[k_opt]], "*", color=RED, markersize=14, zorder=5, label=f"optimum k = {k_opt}")
    for k, p in enumerate(p_marked):
        a1.annotate(f"{p:.3f}", (k, p), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    a1.set_ylim(0, 1.12)
    a1.set_xticks(range(K_MAX4 + 1))
    a1.set_xlabel("Grover iterations k")
    a1.set_ylabel("P(marked)")
    a1.set_title("Probability rises, then overshoots")
    a1.legend(frameon=False, loc="lower center", fontsize=8)
    a1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved {FIG_PATH.relative_to(ROOT)}")


def save_json(dists, p_marked, auto4, hidden, r8, peak8, ms):
    data = {
        "four_qubit": {
            "n_qubits": N4,
            "marked": MARKED4,
            "theta": theta_of(N4),
            "k": list(range(K_MAX4 + 1)),
            "p_marked": p_marked,
            "p_marked_theory": [p_theory(N4, k) for k in range(K_MAX4 + 1)],
            "distributions": [d.tolist() for d in dists],
            "auto_iterations": auto4.iterations,
        },
        "eight_qubit": {
            "n_qubits": N8,
            "hidden": hidden,
            "iterations": r8.iterations,
            "found": r8.marked_index,
            "p_hidden": peak8,
            "p_hidden_theory": p_theory(N8, r8.iterations),
            "classical_expected_queries": 2 ** N8 // 2,
        },
        "backends": {
            "n_values": BACKEND_N,
            "grover_iterations": BACKEND_ITERS,
            "repeats": REPEATS,
            "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "milliseconds": ms,
        },
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"  saved {JSON_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Grover walkthrough")
    print("=" * 64)

    dists, p_marked, auto4 = four_qubit()
    hidden, r8, peak8 = eight_qubit()
    ms, spread = backends()

    print("\n[4/4] Output and checks")
    plot(dists, p_marked)
    save_json(dists, p_marked, auto4, hidden, r8, peak8, ms)

    err4 = max(abs(p_marked[k] - p_theory(N4, k)) for k in range(K_MAX4 + 1))
    check("4q: P(marked) = sin^2((2k+1) theta) for k = 0..4, < 1e-10", err4 < 1e-10, f"{err4:.2e}")
    check("4q: P(marked) at k = 0 is 1/16", abs(p_marked[0] - 1 / 16) < 1e-12)
    check("4q: P(marked) increases for k = 0..3", all(p_marked[k + 1] > p_marked[k] for k in range(3)))
    check("4q: k = 3 is optimal (k = 4 overshoots)", p_marked[3] > p_marked[4] and p_marked[3] > p_marked[2],
          f"{p_marked[3]:.4f}")
    check("4q: automatic iteration count is 3", auto4.iterations == 3)
    check("4q: unmarked states are uniform at every k",
          all(np.ptp(np.delete(d, MARKED4)) < 1e-12 for d in dists))

    r2 = alg.run_grover(2, [3])
    check("2q: single iteration finds the marked state with certainty",
          abs(r2.probabilities[3] - 1.0) < 1e-12, f"{r2.probabilities[3]:.12f}")

    check("8q: hidden state is the most probable", r8.marked_index == hidden)
    check("8q: iteration count matches theory", r8.iterations == k_optimal(N8), str(r8.iterations))
    check("8q: peak probability > 0.999", peak8 > 0.999, f"{peak8:.6f}")
    check("8q: matches sin^2((2k+1) theta), < 1e-10",
          abs(peak8 - p_theory(N8, r8.iterations)) < 1e-10)

    check("backends agree on the output distribution, < 1e-10", spread < 1e-10, f"{spread:.2e}")
    check("StateVector faster than Reference at every n tested",
          all(s < r for s, r in zip(ms["statevector"], ms["reference"])))
    check("figure and JSON written", FIG_PATH.exists() and JSON_PATH.exists())

    print(f"\n{sum(_checks)}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())