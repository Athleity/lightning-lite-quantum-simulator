"""Algorithm benchmark (Tier 2): Grover, QAOA and VQE on the gate-level backends.

  1. Grover, n = 8, 10, 12, 14: automatic iteration count, peak probability vs
     sin^2((2k + 1) theta), StateVector time
  2. QAOA MaxCut on a random weighted 8-node graph, p = 1..4. Multi-start compass search:
     p = 1 starts from a grid-search optimum plus random restarts; p >= 2 starts from the INTERP
     extension of the depth p - 1 optimum (Zhou et al.) plus jittered copies of it and of the
     zero-padded point. The unperturbed zero-padded point is a saddle (new layer gamma = beta = 0
     reproduces depth p - 1 exactly, and single-coordinate moves cannot leave it), so it is used
     only as a floor: depth p is never reported worse than depth p - 1.
  3. VQE for H2 (2 qubits, R = 0.735 A, STO-3G, parity mapping), RY + CNOT ansatz, 2 layers,
     gradient descent with parameter-shift gradients, 50 iterations, best of several seeds
  4. Eigen vs StateVector wall time for all three algorithms
  5. Figures, JSON and checks

  docs/images/grover_scaling.png
  docs/images/qaoa_maxcut.png
  docs/images/vqe_h2_convergence.png
  docs/images/algorithms_backend_comparison.png
  docs/algorithms_benchmark.json

H2 Hamiltonian (electronic part, Hartree), basis index z = bit0 + 2 bit1:
  H = g0 + g1 Z0 + g2 Z1 + g3 Z0 Z1 + g4 X0 X1
The X0 X1 expectation is evaluated from the state vector as 2 Re(conj(psi[0]) psi[3] +
conj(psi[1]) psi[2]); the diagonal part from the probabilities. Exact diagonalisation gives the
FCI energy and is checked against the published value.

QAOA quality is <H_C> / optimum, the usual approximation ratio. best_cut is the cut of the single
most probable bitstring, reported alongside.

Run: python3 benchmarks/algorithms_benchmark.py
"""

import json
import math
import os
import sys
import time
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
JSON_PATH = ROOT / "docs" / "algorithms_benchmark.json"

REPEATS = 3
SV = alg.Backend.STATEVECTOR

GROVER_N = [8, 10, 12, 14]

QAOA_NODES = 8
QAOA_SEED = 7
QAOA_P = [1, 2, 3, 4]
QAOA_RESTARTS = 12
QAOA_JITTER = 0.2

H2_G0, H2_G1, H2_G2, H2_G3, H2_G4 = (-1.052373245772859, 0.39793742484318045,
                                     -0.39793742484318045, -0.01128010425623538,
                                     0.18093119978423156)
H2_E_ELEC_PUBLISHED = -1.857275030202382
H2_E_NUCLEAR = 0.7199689944489797
CHEM_ACC = 1.6e-3
VQE_LAYERS = 2
VQE_ITERS = 50
VQE_LR = 0.5
VQE_SEEDS = (1, 2, 3)
VQE_LONG_ITERS = 500

BACKEND_ITERS = 10  # Grover iterations in the backend comparison (fixed, same circuit everywhere)

BLUE, RED, GREY, GREEN = "#3b6ea5", "#d1495b", "#999999", "#4c9a6a"

_checks = []


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def warn(name, ok, detail=""):
    """Reported but not counted in the pass/fail total."""
    print(f"  [{'PASS' if ok else 'WARN'}] {name}" + (f"  ({detail})" if detail else ""))


def timed(fn, repeats=REPEATS):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def theta_of(n_qubits, n_marked=1):
    return math.asin(math.sqrt(n_marked / 2 ** n_qubits))


def p_theory(n_qubits, k, n_marked=1):
    return math.sin((2 * k + 1) * theta_of(n_qubits, n_marked)) ** 2


def save_fig(fig, name):
    path = IMAGES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# 1. Grover
# ---------------------------------------------------------------------------

def grover_section():
    print("\n[1/5] Grover, single marked state, automatic iteration count, StateVector backend")
    alg.run_grover(12, [1])  # thread-pool warm-up
    rows = []
    print("  n    2^n     iterations  peak P      theory P    |diff|     time[ms]")
    for n in GROVER_N:
        marked = (1 << n) // 3
        t, r = timed(lambda: alg.run_grover(n, [marked], backend=SV))
        peak = float(r.probabilities[marked])
        theory = p_theory(n, r.iterations)
        rows.append({"n": n, "marked": marked, "iterations": r.iterations, "found": r.marked_index,
                     "peak": peak, "theory": theory, "ms": 1e3 * t})
        print(f"  {n:<4d} {2 ** n:<7d} {r.iterations:<11d} {peak:<11.6f} {theory:<11.6f} "
              f"{abs(peak - theory):<10.1e} {1e3 * t:.2f}")
    return rows


def plot_grover(rows):
    ns = [r["n"] for r in rows]
    peak = [r["peak"] for r in rows]
    theory = [r["theory"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ns, theory, "x--", color=GREY, lw=1.2, markersize=9, label=r"$\sin^2((2k+1)\theta)$")
    ax.plot(ns, peak, "o", color=BLUE, markersize=7, label="simulated")
    for r in rows:
        ax.annotate(f"k = {r['iterations']}", (r["n"], r["peak"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)
    lo = min(peak + theory)
    ax.set_ylim(lo - 6e-4, 1.0 + 6e-4)
    ax.set_xticks(ns)
    ax.set_xlabel("qubits n  (search space 2^n)")
    ax.set_ylabel("peak probability of the marked state")
    ax.set_title("Grover: peak probability at the optimal iteration count")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    return save_fig(fig, "grover_scaling.png")


# ---------------------------------------------------------------------------
# 2. QAOA MaxCut
# ---------------------------------------------------------------------------

def random_graph(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        edges = [(i, j, round(float(rng.uniform(0.5, 2.0)), 2))
                 for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
        if len(edges) >= n:
            return edges


def interp(a):
    """INTERP (Zhou et al. 2020): optimal angles at depth p -> initial angles at depth p + 1."""
    p = len(a)
    pad = np.concatenate(([0.0], a, [0.0]))
    return np.array([(i - 1) / p * pad[i - 1] + (p - i + 1) / p * pad[i] for i in range(1, p + 2)])


def refine(f, x0, step=0.15, tol=5e-4, max_eval=10000):
    """Compass search maximising f. Returns (x, f(x), number of evaluations)."""
    x = np.array(x0, dtype=float)
    fx = f(x)
    n_eval = 1
    while step > tol and n_eval < max_eval:
        improved = False
        for i in range(len(x)):
            for s in (step, -step):
                y = x.copy()
                y[i] += s
                fy = f(y)
                n_eval += 1
                if fy > fx + 1e-12:
                    x, fx, improved = y, fy, True
                    break
        if not improved:
            step *= 0.5
    return x, fx, n_eval


def qaoa_section():
    n = QAOA_NODES
    edges = random_graph(n, QAOA_SEED)
    cut = alg.cut_values(edges, n)
    opt = float(cut.max())
    total_w = sum(w for _, _, w in edges)
    baseline = total_w / 2.0
    print(f"\n[2/5] QAOA MaxCut, {n} nodes, {len(edges)} weighted edges (seed {QAOA_SEED})")
    print(f"  total weight {total_w:.2f}, optimum cut {opt:.2f}, random-guess <cut> {baseline:.2f} "
          f"(ratio {baseline / opt:.3f})")

    graph = alg.Graph(n, edges)
    be = alg.make_backend(SV, n)

    def run(x):
        p = len(x) // 2
        return alg.QAOA.run(be, graph, [float(v) for v in x[:p]], [float(v) for v in x[p:]])

    def expected(x):
        return float(np.dot(run(x).probabilities, cut))

    rng = np.random.default_rng(QAOA_SEED)
    results, params = [], {}
    dists = {}
    for p in QAOA_P:
        if p == 1:
            gs, bs = np.linspace(0.05, math.pi, 40), np.linspace(0.05, math.pi / 2, 20)
            _, g0, b0 = max((expected(np.array([g, b])), g, b) for g in gs for b in bs)
            starts = [np.array([g0, b0])]
            starts += [np.array([rng.uniform(0.0, math.pi), rng.uniform(0.0, math.pi / 2)])
                       for _ in range(QAOA_RESTARTS - 1)]
        else:
            prev = params[p - 1]
            gp, bp = prev[:p - 1], prev[p - 1:]
            padded = np.concatenate((gp, [0.0], bp, [0.0]))
            interp_start = np.concatenate((interp(gp), interp(bp)))
            # The unperturbed padded point is a saddle, so only jittered copies of it are used.
            starts = [interp_start]
            starts += [s + QAOA_JITTER * rng.standard_normal(2 * p)
                       for s in (interp_start, padded) for _ in range(QAOA_RESTARTS // 2)]

        best_x, best_f, n_eval = None, -np.inf, 0
        for x0 in starts:
            xi, fi, ne = refine(expected, x0)
            n_eval += ne
            if fi > best_f:
                best_x, best_f = xi, fi
        if p > 1 and expected(padded) > best_f:  # depth p can always reproduce depth p - 1
            best_x, best_f = padded, expected(padded)
        x, fx = best_x, best_f
        params[p] = x
        r = run(x)
        probs = np.asarray(r.probabilities)
        dists[p] = probs
        results.append({
            "p": p, "expected_cut": fx, "best_cut": float(r.best_cut), "optimum": opt,
            "ratio_expected": fx / opt, "ratio_best": float(r.best_cut) / opt,
            "p_optimum": float(probs[cut >= opt - 1e-9].sum()), "evals": n_eval,
            "gammas": [float(v) for v in x[:p]], "betas": [float(v) for v in x[p:]],
        })

    print("\n  p   <cut>     best cut  optimum   <cut>/opt  P(optimum)  evals")
    for r in results:
        print(f"  {r['p']:<3d} {r['expected_cut']:<9.4f} {r['best_cut']:<9.2f} {r['optimum']:<9.2f} "
              f"{r['ratio_expected']:<10.4f} {r['p_optimum']:<11.4f} {r['evals']}")
    return {"edges": edges, "cut": cut, "optimum": opt, "baseline": baseline, "results": results,
            "dists": dists}


def plot_qaoa(q):
    res, cut, opt = q["results"], q["cut"], q["optimum"]
    ps = [r["p"] for r in res]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})

    a0.plot(ps, [r["ratio_expected"] for r in res], "o-", color=BLUE, lw=1.6, label=r"$\langle C\rangle$ / optimum")
    a0.plot(ps, [r["ratio_best"] for r in res], "s", color=RED, markersize=7, label="most probable bitstring")
    a0.axhline(1.0, color="black", ls="--", lw=0.9)
    a0.axhline(q["baseline"] / opt, color=GREY, ls=":", lw=1.2, label="random guess")
    a0.set_xticks(ps)
    a0.set_ylim(q["baseline"] / opt - 0.1, 1.05)
    a0.set_xlabel("QAOA layers p")
    a0.set_ylabel("cut / optimum cut")
    a0.set_title(f"MaxCut, {QAOA_NODES} nodes, {len(q['edges'])} edges")
    a0.legend(frameon=False, loc="lower right", fontsize=8)
    a0.grid(alpha=0.3)

    p_hi = max(QAOA_P)
    for p, colour in ((1, GREY), (p_hi, RED)):
        a1.scatter(cut, q["dists"][p], s=14, color=colour, alpha=0.65, label=f"p = {p}")
    a1.axvline(opt, color="black", ls="--", lw=0.9)
    a1.set_yscale("log")
    a1.set_xlabel("cut weight of the bitstring")
    a1.set_ylabel("probability")
    a1.set_title("Probability concentrates on large cuts")
    a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=0.3, which="both")
    return save_fig(fig, "qaoa_maxcut.png")


# ---------------------------------------------------------------------------
# 3. VQE for H2
# ---------------------------------------------------------------------------

def h2_matrix():
    diag = H2_G0 + alg.ising_diagonal(2, zz=[(0, 1, H2_G3)], z=[(0, H2_G1), (1, H2_G2)])
    m = np.diag(diag)
    for z in range(4):
        m[z, z ^ 3] += H2_G4
    return m, diag


def vqe_section():
    m, diag = h2_matrix()
    e_fci = float(np.linalg.eigvalsh(m)[0])
    print(f"\n[3/5] VQE for H2 (R = 0.735 A, STO-3G), {VQE_LAYERS} layers, {VQE_ITERS} iterations, "
          f"lr {VQE_LR}, seeds {VQE_SEEDS}")
    print(f"  FCI electronic energy {e_fci:.9f} Ha  (published {H2_E_ELEC_PUBLISHED:.9f})")
    print(f"  FCI total energy      {e_fci + H2_E_NUCLEAR:.9f} Ha  (with nuclear repulsion)")

    def build(seed):
        vqe = alg.VQE(2, VQE_LAYERS, seed)
        be = alg.make_backend(SV, 2)

        def energy(params):
            vqe.prepare(be, params)
            psi = np.asarray(be.state())
            p = psi.real ** 2 + psi.imag ** 2
            xx = 2.0 * float(np.real(np.conj(psi[0]) * psi[3] + np.conj(psi[1]) * psi[2]))
            return float(np.dot(p, diag)) + H2_G4 * xx

        return vqe, be, energy

    # optimize() returns only the final result, so the curve is the best energy seen after
    # k iterations for k = 0..VQE_ITERS. The run is deterministic, so these are prefixes.
    hist = {}
    for seed in VQE_SEEDS:
        vqe, be, energy = build(seed)
        hist[seed] = [vqe.optimize(be, energy, k, VQE_LR).energy for k in range(VQE_ITERS + 1)]

    best_seed = min(VQE_SEEDS, key=lambda s: hist[s][-1])
    h = hist[best_seed]

    print(f"\n  best seed: {best_seed}")
    print("  iteration  energy [Ha]      error vs FCI [Ha]")
    for k in (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, VQE_ITERS):
        print(f"  {k:<10d} {h[k]:<16.9f} {h[k] - e_fci:.3e}")

    vqe, be, energy = build(best_seed)
    long_run = vqe.optimize(be, energy, VQE_LONG_ITERS, VQE_LR)
    err50, err_long = h[-1] - e_fci, long_run.energy - e_fci
    print(f"\n  after {VQE_ITERS} iterations: error {err50:.3e} Ha ({1e3 * err50:.2f} mHa), chemical "
          f"accuracy (1.6 mHa) {'reached' if err50 < CHEM_ACC else 'not reached'}")
    print(f"  after {VQE_LONG_ITERS} iterations: error {err_long:.3e} Ha ({1e3 * err_long:.3f} mHa), "
          f"chemical accuracy {'reached' if err_long < CHEM_ACC else 'not reached'}")
    return {"e_fci": e_fci, "hist": hist, "best_seed": best_seed, "err50": err50,
            "err_long": err_long, "long_energy": long_run.energy}


def plot_vqe(v):
    e_fci, hist, best = v["e_fci"], v["hist"], v["best_seed"]
    ks = np.arange(VQE_ITERS + 1)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for seed, h in hist.items():
        a0.plot(ks, h, color=GREY, lw=1.0, alpha=0.6)
    a0.plot(ks, hist[best], color=BLUE, lw=2.0, label=f"VQE (seed {best})")
    a0.axhline(e_fci, color="black", ls="--", lw=1.0, label=f"FCI {e_fci:.6f} Ha")
    a0.axhspan(e_fci, e_fci + CHEM_ACC, color=GREEN, alpha=0.25, label="chemical accuracy")
    a0.set_xlabel("iteration")
    a0.set_ylabel("electronic energy [Ha]")
    a0.set_title("H2 VQE: best energy found vs iteration")
    a0.legend(frameon=False, fontsize=8)
    a0.grid(alpha=0.3)

    for seed, h in hist.items():
        a1.semilogy(ks, np.maximum(np.array(h) - e_fci, 1e-12), color=GREY, lw=1.0, alpha=0.6)
    a1.semilogy(ks, np.maximum(np.array(hist[best]) - e_fci, 1e-12), color=BLUE, lw=2.0, label=f"seed {best}")
    a1.axhline(CHEM_ACC, color=GREEN, ls="--", lw=1.0, label="1.6 mHa")
    a1.set_xlabel("iteration")
    a1.set_ylabel("energy above FCI [Ha]")
    a1.set_title("Error vs FCI")
    a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=0.3, which="both")
    return save_fig(fig, "vqe_h2_convergence.png")


# ---------------------------------------------------------------------------
# 4. Backend comparison
# ---------------------------------------------------------------------------

def random_edges(n, m, seed):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < m:
        i, j = (int(v) for v in rng.integers(0, n, 2))
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return [(i, j, 1.0) for i, j in sorted(pairs)]


def backend_section():
    print(f"\n[4/5] Eigen vs StateVector (best of {REPEATS}); "
          f"cores {os.cpu_count()}, OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    alg.run_grover(12, [1], backend=SV, iterations=BACKEND_ITERS)  # warm-up

    cases = []
    for n in (16, 18):
        cases.append(("Grover", n, f"{BACKEND_ITERS} iterations",
                      lambda b, n=n: alg.run_grover(n, [1], backend=b, iterations=BACKEND_ITERS)))
    for n in (16, 18):
        edges = random_edges(n, 2 * n, n)
        cases.append(("QAOA", n, f"p = 3, {2 * n} edges",
                      lambda b, n=n, e=edges: alg.run_maxcut(e, [0.5] * 3, [0.3] * 3, backend=b, n_nodes=n)))
    for n in (12, 14):
        d = alg.ising_diagonal(n, zz=[(i, (i + 1) % n, 1.0) for i in range(n)], z=[(0, 0.3)])
        cases.append(("VQE", n, "2 layers, 3 steps",
                      lambda b, n=n, d=d: alg.run_vqe(n, hamiltonian_diag=d, n_layers=VQE_LAYERS,
                                                      backend=b, max_iter=3)))

    rows = []
    print("\n  algorithm  qubits  workload             Eigen[ms]   StateVec[ms]  speedup")
    for name, n, note, fn in cases:
        t_e, _ = timed(lambda: fn(alg.Backend.EIGEN))
        t_s, _ = timed(lambda: fn(SV))
        rows.append({"algorithm": name, "n": n, "workload": note, "eigen_ms": 1e3 * t_e,
                     "statevector_ms": 1e3 * t_s, "speedup": t_e / t_s})
        print(f"  {name:<10s} {n:<7d} {note:<20s} {1e3 * t_e:<11.1f} {1e3 * t_s:<13.1f} {t_e / t_s:.2f}x")

    # Same circuits on every backend must agree.
    edges = random_edges(10, 20, 3)
    qs = [np.asarray(alg.run_maxcut(edges, [0.4, 0.7], [0.3, 0.2], backend=b, n_nodes=10).probabilities)
          for b in alg.Backend]
    q_spread = max(float(np.max(np.abs(qs[0] - p))) for p in qs[1:])
    d8 = alg.ising_diagonal(8, zz=[(i, i + 1, 1.0) for i in range(7)], z=[(0, 0.3)])
    es = [alg.run_vqe(8, hamiltonian_diag=d8, n_layers=2, backend=b, max_iter=5).energy for b in alg.Backend]
    v_spread = max(es) - min(es)
    return rows, q_spread, v_spread


def plot_backends(rows):
    labels = [f"{r['algorithm']}\nn = {r['n']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w / 2, [r["eigen_ms"] for r in rows], w, color=BLUE, label="Eigen")
    ax.bar(x + w / 2, [r["statevector_ms"] for r in rows], w, color=RED, label="StateVector (OpenMP)")
    for i, r in enumerate(rows):
        ax.text(i + w / 2, r["statevector_ms"], f"{r['speedup']:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("wall time [ms]")
    ax.set_title("Eigen vs StateVector backend (label: speedup)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, which="both")
    return save_fig(fig, "algorithms_backend_comparison.png")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(g, q, v, b):
    data = {
        "grover": g,
        "qaoa": {
            "n_nodes": QAOA_NODES, "seed": QAOA_SEED, "edges": q["edges"], "optimum": q["optimum"],
            "random_guess": q["baseline"], "layers": q["results"],
            "best_distribution_p": max(QAOA_P),
            "best_distribution": q["dists"][max(QAOA_P)].tolist(),
        },
        "vqe_h2": {
            "bond_length_angstrom": 0.735, "layers": VQE_LAYERS, "iterations": VQE_ITERS, "lr": VQE_LR,
            "e_fci_electronic": v["e_fci"], "e_nuclear": H2_E_NUCLEAR, "best_seed": v["best_seed"],
            "energy_history": {str(s): h for s, h in v["hist"].items()},
            "error_after_iterations": v["err50"],
            "error_after_long_run": v["err_long"], "long_run_iterations": VQE_LONG_ITERS,
        },
        "backends": {"cpu_count": os.cpu_count(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                     "repeats": REPEATS, "rows": b},
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"  saved {JSON_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Algorithm benchmark: Grover, QAOA, VQE")
    print("=" * 64)

    g = grover_section()
    q = qaoa_section()
    v = vqe_section()
    b, q_spread, v_spread = backend_section()

    print("\n[5/5] Output and checks")
    paths = [plot_grover(g), plot_qaoa(q), plot_vqe(v), plot_backends(b)]
    save_json(g, q, v, b)

    res = q["results"]
    exp = [r["expected_cut"] for r in res]

    check("Grover: peak = sin^2((2k+1) theta) at every n, < 1e-10",
          max(abs(r["peak"] - r["theory"]) for r in g) < 1e-10)
    check("Grover: most probable state is the marked one at every n", all(r["found"] == r["marked"] for r in g))
    check("Grover: peak probability > 0.999 at every n", min(r["peak"] for r in g) > 0.999,
          f"{min(r['peak'] for r in g):.6f}")

    check("QAOA: distributions normalised", all(abs(p.sum() - 1.0) < 1e-12 for p in q["dists"].values()))
    check("QAOA: <cut> never exceeds the optimum", all(e <= q["optimum"] + 1e-9 for e in exp))
    check("QAOA: best bitstring cut never exceeds the optimum", all(r["best_cut"] <= q["optimum"] + 1e-9 for r in res))
    check("QAOA: <cut> non-decreasing in p", all(exp[i + 1] >= exp[i] - 1e-9 for i in range(len(exp) - 1)))
    check("QAOA: p = 1 beats random guessing", exp[0] > q["baseline"] + 1e-6,
          f"{exp[0]:.3f} > {q['baseline']:.3f}")
    warn("QAOA: deepest p improves on p = 1", exp[-1] > exp[0], f"{exp[-1]:.3f} vs {exp[0]:.3f}")

    check("H2: Hamiltonian reproduces the published FCI energy, < 1e-6 Ha",
          abs(v["e_fci"] - H2_E_ELEC_PUBLISHED) < 1e-6, f"{abs(v['e_fci'] - H2_E_ELEC_PUBLISHED):.2e}")
    check("VQE: variational bound, no energy below FCI",
          min(min(h) for h in v["hist"].values()) >= v["e_fci"] - 1e-9)
    hb = v["hist"][v["best_seed"]]
    check("VQE: best-so-far energy non-increasing", all(hb[i + 1] <= hb[i] + 1e-12 for i in range(len(hb) - 1)))
    check("VQE: error halved within the iteration budget", (hb[-1] - v["e_fci"]) < 0.5 * (hb[0] - v["e_fci"]),
          f"{hb[0] - v['e_fci']:.3f} -> {hb[-1] - v['e_fci']:.3f} Ha")
    check("VQE: long-run energy not below FCI", v["long_energy"] >= v["e_fci"] - 1e-9)

    check("backends agree on a QAOA distribution, < 1e-10", q_spread < 1e-10, f"{q_spread:.2e}")
    check("backends agree on a VQE energy, < 1e-9", v_spread < 1e-9, f"{v_spread:.2e}")
    check("four figures and JSON written", all(p.exists() for p in paths) and JSON_PATH.exists())

    print(f"\n{sum(_checks)}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())