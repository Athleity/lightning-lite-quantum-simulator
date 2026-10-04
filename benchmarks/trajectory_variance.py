"""Trajectory solver benchmark (Tier 3).

  1. Dephasing L = Z, |+>, <X>: trajectories needed for standard error EPS under STANDARD,
     PROJECTOR and ANALOG over gamma t, against the closed forms
       Var_STANDARD = 1 - m^2,  Var_PROJECTOR = m^2 (1 - m^2)/(1 + m^2),  m = exp(-2 gamma t)
  2. Amplitude damping L = sigma_-, |1>, <P1>: not Pauli-like, so PROJECTOR falls back to
     STANDARD and the two must agree exactly
  3. Multi-qubit chain (H = J sum X_q X_{q+1} + (h/2) sum Z_q, T1 and dephasing on every qubit,
     |+>^n, observable mean X): cost per trajectory vs n, trajectories needed vs n, time to reach
     EPS, and agreement with a numpy density-matrix RK4 reference
  4. Figures, JSON and checks

  docs/images/trajectory_variance_reduction.png   STANDARD / PROJECTOR ratio vs gamma t
  docs/images/trajectory_n_required.png           trajectories for EPS vs gamma t
  docs/images/trajectory_scaling.png              cost, trajectories and time-to-EPS vs n
  docs/trajectory_benchmark.json

N(EPS) = sigma^2 / EPS^2 with sigma^2 the per-trajectory variance of the estimator, measured from
a pilot ensemble (relative error about sqrt(2 / N_pilot)). Per-trajectory times are wall time
over all OpenMP threads (set OMP_NUM_THREADS to control). The density-matrix reference is plain
numpy, so its time shows the 4^n growth but is not a like-for-like comparison with the C++ solver.

Set TRAJ_QUICK=1 for n = 1..3 and fewer trajectories.

Run: python3 benchmarks/trajectory_benchmark.py
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

import trajectory as tr  # noqa: E402

QUICK = os.environ.get("TRAJ_QUICK", "0") == "1"

IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)
JSON_PATH = ROOT / "docs" / "trajectory_benchmark.json"

EPS = 0.005
GAMMA_T = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
T1_TIMES = [0.25, 0.5, 1.0, 2.0]
N_PILOT_1Q = 4000 if QUICK else 10000

N_RANGE = [1, 2, 3] if QUICK else [1, 2, 3, 4, 5]
N_PILOT = 400 if QUICK else 1000
JUMP_STEPS, ANALOG_STEPS = 200, 400
T_FINAL = 1.5
T1, TPHI = 3.0, 2.0
J_COUPLING, H_FIELD = 0.5, 0.4
EXACT_STEPS = 400

KINDS = tr.ALL_KINDS
NAMES = [k.name for k in KINDS]
COLOURS = {"STANDARD": "#3b6ea5", "PROJECTOR": "#d1495b", "ANALOG": "#4c9a6a"}
MARKS = {"STANDARD": "o", "PROJECTOR": "s", "ANALOG": "^"}
GREY = "#999999"

_checks = []


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def warn(name, ok, detail=""):
    """Reported but not counted in the pass/fail total."""
    print(f"  [{'PASS' if ok else 'WARN'}] {name}" + (f"  ({detail})" if detail else ""))


def save_fig(fig, name):
    path = IMAGES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# 1. Dephasing
# ---------------------------------------------------------------------------

def dephasing_section():
    print(f"\n[1/4] Dephasing L = Z (gamma = 1), |+>, <X>: trajectories for standard error {EPS}")
    s = tr.make_dephasing(1.0)
    psi0 = tr.plus()
    rows = []
    print("  gamma t  N STD      N PROJ     N ANALOG   | STD/PROJ  analytic   STD/ANALOG")
    for gt in GAMMA_T:
        vr = tr.variance_reduction(s, psi0, gt, "X", eps=EPS, n_pilot=N_PILOT_1Q, seed=11)
        m2 = math.exp(-4.0 * gt)
        row = {
            "gamma_t": gt,
            "n_required": {k: vr[k]["n_required"] for k in NAMES},
            "variance": {k: vr[k]["variance"] for k in NAMES},
            "factor": {k: vr[k]["factor_vs_standard"] for k in NAMES},
            "analytic_ratio": tr.analytic_dephasing_ratio(gt),
            "analytic_var_standard": 1.0 - m2,
            "analytic_var_projector": m2 * (1.0 - m2) / (1.0 + m2),
        }
        rows.append(row)
        print(f"  {gt:<8.2f} {row['n_required']['STANDARD']:<10d} {row['n_required']['PROJECTOR']:<10d} "
              f"{row['n_required']['ANALOG']:<10d} | {row['factor']['PROJECTOR']:<9.2f} "
              f"{row['analytic_ratio']:<10.2f} {row['factor']['ANALOG']:.2f}")
    return rows


def plot_n_required(rows):
    gt = np.array([r["gamma_t"] for r in rows])
    grid = np.linspace(gt.min(), gt.max(), 200)
    m2 = np.exp(-4.0 * grid)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(grid, (1 - m2) / EPS ** 2, color=COLOURS["STANDARD"], lw=1.0, ls=":")
    ax.semilogy(grid, m2 * (1 - m2) / (1 + m2) / EPS ** 2, color=COLOURS["PROJECTOR"], lw=1.0, ls=":",
                label="closed forms")
    for k in NAMES:
        ax.semilogy(gt, [r["n_required"][k] for r in rows], MARKS[k], color=COLOURS[k], markersize=6, label=k)
    ax.set_xlabel(r"$\gamma t$")
    ax.set_ylabel(f"trajectories for standard error {EPS}")
    ax.set_title("Dephasing, |+>, <X>: trajectories needed")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    return save_fig(fig, "trajectory_n_required.png")


# ---------------------------------------------------------------------------
# 2. Amplitude damping (PROJECTOR falls back)
# ---------------------------------------------------------------------------

def damping_section():
    print("\n[2/4] Amplitude damping L = sigma_- (gamma = 1), |1>, <P1>: PROJECTOR is not applicable")
    s = tr.make_qubit(t1=1.0)
    rows = []
    print("  gamma t  Var STD     Var PROJ    Var ANALOG  | N STD      N ANALOG   | STD/ANALOG")
    for t in T1_TIMES:
        vr = tr.variance_reduction(s, tr.excited(), t, "P1", eps=EPS, n_pilot=N_PILOT_1Q, seed=12)
        rows.append({"gamma_t": t, "variance": {k: vr[k]["variance"] for k in NAMES},
                     "n_required": {k: vr[k]["n_required"] for k in NAMES},
                     "factor_analog": vr["ANALOG"]["factor_vs_standard"]})
        print(f"  {t:<8.2f} {vr['STANDARD']['variance']:<11.5f} {vr['PROJECTOR']['variance']:<11.5f} "
              f"{vr['ANALOG']['variance']:<11.5f} | {vr['STANDARD']['n_required']:<10d} "
              f"{vr['ANALOG']['n_required']:<10d} | {vr['ANALOG']['factor_vs_standard']:.2f}")
    return rows


# ---------------------------------------------------------------------------
# 3. Multi-qubit scaling
# ---------------------------------------------------------------------------

def embed(n, q, op):
    """2x2 operator on qubit q of n (qubit q is bit q of the basis index)."""
    return np.kron(np.kron(np.eye(2 ** (n - 1 - q), dtype=complex), op), np.eye(2 ** q, dtype=complex))


def build_chain(n):
    h = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n - 1):
        h += J_COUPLING * embed(n, q, tr.SIGMA_X) @ embed(n, q + 1, tr.SIGMA_X)
    for q in range(n):
        h += 0.5 * H_FIELD * embed(n, q, tr.SIGMA_Z)
    channels = []
    for q in range(n):
        channels.append((embed(n, q, tr.SIGMA_MINUS), 1.0 / T1))
        channels.append((embed(n, q, tr.SIGMA_Z), 0.5 / TPHI))
    solver = tr.TrajectorySolver(n)
    solver.set_hamiltonian(h)
    for l, g in channels:
        solver.add_lindblad(l, g)
    psi0 = tr.plus()
    for _ in range(n - 1):
        psi0 = np.kron(psi0, tr.plus())
    op = sum(embed(n, q, tr.SIGMA_X) for q in range(n)) / n
    return solver, h, channels, psi0, op


def exact_expectation(h, channels, psi0, op, duration, steps=EXACT_STEPS):
    """RK4 on the density-matrix Lindblad equation; returns Tr(op rho(T))."""
    rho = np.outer(psi0, psi0.conj())
    ldl = [l.conj().T @ l for l, _ in channels]

    def rhs(r):
        out = -1j * (h @ r - r @ h)
        for (l, g), m in zip(channels, ldl):
            out = out + g * (l @ r @ l.conj().T - 0.5 * (m @ r + r @ m))
        return out

    dt = duration / steps
    for _ in range(steps):
        k1 = rhs(rho)
        k2 = rhs(rho + 0.5 * dt * k1)
        k3 = rhs(rho + 0.5 * dt * k2)
        k4 = rhs(rho + dt * k3)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(np.real(np.trace(op @ rho)))


def steps_for(kind):
    return ANALOG_STEPS if kind == tr.Unraveling.ANALOG else JUMP_STEPS


def scaling_section():
    print(f"\n[3/4] Chain of n qubits, T = {T_FINAL}, T1 = {T1}, Tphi = {TPHI}, J = {J_COUPLING}, "
          f"h = {H_FIELD}, observable mean X, {N_PILOT} trajectories per point")
    print(f"  cores: {os.cpu_count()}   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    rows = []
    for n in N_RANGE:
        solver, h, channels, psi0, op = build_chain(n)
        t0 = time.perf_counter()
        exact = exact_expectation(h, channels, psi0, op, T_FINAL)
        t_exact = time.perf_counter() - t0

        row = {"n": n, "dim": 2 ** n, "exact": exact, "t_exact_s": t_exact, "kinds": {}}
        for kind in KINDS:
            t0 = time.perf_counter()
            e = tr.estimate(solver, psi0, T_FINAL, op, kind, N_PILOT, steps_for(kind), 100 + n)
            wall = time.perf_counter() - t0
            var = e.std_error ** 2 * N_PILOT
            n_req = tr.trajectories_needed(var, EPS)
            ms = 1e3 * wall / N_PILOT
            row["kinds"][kind.name] = {
                "mean": e.mean, "std_error": e.std_error, "variance": var, "ms_per_trajectory": ms,
                "n_required": n_req, "time_to_eps_s": n_req * ms * 1e-3,
                "z": (e.mean - exact) / e.std_error if e.std_error > 0 else 0.0,
            }
        rows.append(row)
        print(f"  n={n} done", flush=True)

    print("\n  n  dim   exact <M>   | ms per trajectory                | N(eps)                        | PROJ factor")
    print("                         | STD        PROJ       ANALOG     | STD        PROJ       ANALOG |")
    for r in rows:
        k = r["kinds"]
        print(f"  {r['n']:<2d} {r['dim']:<5d} {r['exact']:<11.5f} | {k['STANDARD']['ms_per_trajectory']:<10.3f} "
              f"{k['PROJECTOR']['ms_per_trajectory']:<10.3f} {k['ANALOG']['ms_per_trajectory']:<10.3f} | "
              f"{k['STANDARD']['n_required']:<10d} {k['PROJECTOR']['n_required']:<10d} "
              f"{k['ANALOG']['n_required']:<10d} | "
              f"{k['STANDARD']['variance'] / max(k['PROJECTOR']['variance'], 1e-300):.2f}")
    print("\n  n  time to eps [s]: STD        PROJ       ANALOG     | numpy rho RK4 [s] | z-scores STD / PROJ / ANALOG")
    for r in rows:
        k = r["kinds"]
        print(f"  {r['n']:<2d} {k['STANDARD']['time_to_eps_s']:<24.3f}{k['PROJECTOR']['time_to_eps_s']:<11.3f}"
              f"{k['ANALOG']['time_to_eps_s']:<11.3f}| {r['t_exact_s']:<17.3f} | "
              f"{k['STANDARD']['z']:+.2f} / {k['PROJECTOR']['z']:+.2f} / {k['ANALOG']['z']:+.2f}")
    return rows


def plot_scaling(rows):
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    for k in NAMES:
        axes[0].semilogy(ns, [r["kinds"][k]["ms_per_trajectory"] for r in rows], MARKS[k] + "-",
                         color=COLOURS[k], lw=1.4, label=k)
    axes[0].set_xlabel("qubits n")
    axes[0].set_ylabel("ms per trajectory (wall)")
    axes[0].set_title(f"Cost per trajectory ({JUMP_STEPS} jump / {ANALOG_STEPS} diffusion steps)")
    axes[0].legend(frameon=False)

    for k in NAMES:
        axes[1].semilogy(ns, [r["kinds"][k]["n_required"] for r in rows], MARKS[k] + "-",
                         color=COLOURS[k], lw=1.4, label=k)
    axes[1].set_xlabel("qubits n")
    axes[1].set_ylabel(f"trajectories for standard error {EPS}")
    axes[1].set_title("Trajectories needed (mean X)")
    axes[1].legend(frameon=False)

    for k in NAMES:
        axes[2].semilogy(ns, [r["kinds"][k]["time_to_eps_s"] for r in rows], MARKS[k] + "-",
                         color=COLOURS[k], lw=1.4, label=k)
    axes[2].semilogy(ns, [r["t_exact_s"] for r in rows], "x--", color=GREY, lw=1.2,
                     label="numpy density matrix (400 RK4 steps)")
    axes[2].set_xlabel("qubits n")
    axes[2].set_ylabel("seconds")
    axes[2].set_title("Time to standard error vs master equation")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(ns)
        ax.grid(alpha=0.3, which="both")
    return save_fig(fig, "trajectory_scaling.png")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(deph, damp, scal):
    data = {
        "eps": EPS,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "quick": QUICK,
        "dephasing": {"gamma": 1.0, "n_pilot": N_PILOT_1Q, "rows": deph},
        "amplitude_damping": {"gamma": 1.0, "n_pilot": N_PILOT_1Q, "rows": damp},
        "chain": {
            "duration": T_FINAL, "t1": T1, "tphi": TPHI, "j": J_COUPLING, "h": H_FIELD,
            "n_pilot": N_PILOT, "jump_steps": JUMP_STEPS, "analog_steps": ANALOG_STEPS,
            "exact_steps": EXACT_STEPS, "rows": scal,
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
    print("Trajectory solver benchmark" + ("  [QUICK]" if QUICK else ""))
    print("=" * 64)

    deph = dephasing_section()
    damp = damping_section()
    scal = scaling_section()

    print("\n[4/4] Output and checks")
    figs = [
        tr.plot_variance_reduction([{"gamma_t": r["gamma_t"], "ratio": r["factor"]["PROJECTOR"]} for r in deph],
                                   IMAGES / "trajectory_variance_reduction.png"),
        plot_n_required(deph),
        plot_scaling(scal),
    ]
    save_json(deph, damp, scal)

    at = next(r for r in deph if abs(r["gamma_t"] - 0.75) < 1e-12)
    ratio, analytic = at["factor"]["PROJECTOR"], at["analytic_ratio"]
    check("dephasing: STANDARD/PROJECTOR variance ratio within 10% of 1 + exp(4 gamma t) at gamma t = 0.75",
          abs(ratio / analytic - 1.0) < 0.10, f"{ratio:.2f} vs {analytic:.2f}")
    check("dephasing: PROJECTOR needs at least 15x fewer trajectories than STANDARD at gamma t = 0.75",
          ratio > 15.0, f"{ratio:.1f}x")
    f = [r["factor"]["PROJECTOR"] for r in deph]
    check("dephasing: variance ratio non-decreasing in gamma t", all(f[i + 1] >= 0.97 * f[i] for i in range(len(f) - 1)))
    check("dephasing: measured variances match the closed forms within 10% (STANDARD, PROJECTOR)",
          all(abs(r["variance"]["STANDARD"] / r["analytic_var_standard"] - 1.0) < 0.10 and
              abs(r["variance"]["PROJECTOR"] / r["analytic_var_projector"] - 1.0) < 0.10 for r in deph))

    e = tr.estimate(tr.make_dephasing(1.0), tr.plus(), 0.75, "X", tr.Unraveling.PROJECTOR, N_PILOT_1Q, 200, 99)
    exact_x = math.exp(-1.5)
    check("dephasing: PROJECTOR unbiased at gamma t = 0.75 (4 se)", abs(e.mean - exact_x) < 4.0 * e.std_error,
          f"{e.mean:.4f} vs {exact_x:.4f}, se {e.std_error:.4f}")

    check("amplitude damping: PROJECTOR falls back and equals STANDARD exactly",
          all(abs(r["variance"]["PROJECTOR"] - r["variance"]["STANDARD"]) < 1e-12 for r in damp))

    bad = [(r["n"], k) for r in scal for k in NAMES
           if abs(r["kinds"][k]["mean"] - r["exact"]) > 4.5 * r["kinds"][k]["std_error"] +
           (0.01 if k == "ANALOG" else 0.0)]
    check("chain: every unraveling matches the density-matrix reference (4.5 se)", not bad,
          f"outliers {bad}" if bad else f"{len(scal) * len(NAMES)} estimates")
    cost = [r["kinds"]["STANDARD"]["ms_per_trajectory"] for r in scal]
    check("chain: cost per trajectory grows with n", cost[-1] > cost[0], f"{cost[0]:.3f} -> {cost[-1]:.3f} ms")
    check("figures and JSON written", all(p.exists() for p in figs) and JSON_PATH.exists())

    reduce_all = all(r["kinds"]["PROJECTOR"]["n_required"] <= r["kinds"]["STANDARD"]["n_required"] for r in scal)
    warn("chain: PROJECTOR needs no more trajectories than STANDARD at every n", reduce_all,
         ", ".join(f"n={r['n']}: {r['kinds']['STANDARD']['variance'] / max(r['kinds']['PROJECTOR']['variance'], 1e-300):.2f}x"
                   for r in scal))
    warn("chain: ANALOG needs fewer trajectories than STANDARD at every n",
         all(r["kinds"]["ANALOG"]["n_required"] <= r["kinds"]["STANDARD"]["n_required"] for r in scal))

    print(f"\n{sum(_checks)}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())