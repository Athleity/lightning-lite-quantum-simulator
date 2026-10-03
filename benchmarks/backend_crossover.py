"""Backend crossover benchmark (Tier 2).

Times the same Grover circuit on the three backends and finds where the OpenMP StateVector
backend sustainably overtakes Eigen:

  docs/images/backend_crossover[_tN].png   time vs state size 2^n, log-log, crossover marked
  docs/backend_crossover[_tN].json         raw timings and run parameters

The circuit is Grover search for one marked state with a fixed number of iterations (ITERS),
so every backend executes the identical gate sequence and the ratios compare gate throughput.
(Absolute times are not Grover time-to-solution: passes per iteration grow with n.) Each point
is the best of REPEATS runs. Reference is skipped above REF_MAX_N because it takes too long.
Every backend must give the same probabilities, and the marked-state probability must equal
sin^2((2k + 1) theta), theta = asin(2^(-n/2)).

The crossover is sustained: the smallest n from which StateVector is ahead at every larger n
tested, so a single noisy point cannot create or hide it.

Thread control. With OMP_NUM_THREADS set, outputs get a _tN suffix so runs do not overwrite each
other. Run both to separate kernel gain from threading:

  OMP_NUM_THREADS=1 python3 benchmarks/backend_crossover.py   StateVector vs Eigen: kernel only
  OMP_NUM_THREADS=2 python3 benchmarks/backend_crossover.py   adds OpenMP scaling

Imports quantum_sim_algorithms from build/ directly.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

try:
    import quantum_sim_algorithms as qa
except ImportError as exc:
    raise ImportError(
        f"quantum_sim_algorithms not found in {ROOT / 'build'}. Build it with the command "
        "at the top of src/bindings_algorithms.cpp."
    ) from exc

TAG = os.environ.get("OMP_NUM_THREADS")
SUFFIX = f"_t{TAG}" if TAG else ""

IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)
JSON_PATH = ROOT / "docs" / f"backend_crossover{SUFFIX}.json"

N_VALUES = [10, 12, 13, 14, 15, 16, 17, 18]
REF_MAX_N = 16
ITERS = 10
REPEATS = 7
OMP_THRESHOLD_N = 15  # StateVectorBackend runs in one thread below 2^15 amplitudes (PAR_MIN)

FACTORIES = {
    "reference": qa.make_reference_backend,
    "eigen": qa.make_eigen_backend,
    "statevector": qa.make_statevector_backend,
}
STYLE = {
    "reference": ("#999999", "s", "Reference (scalar loop)"),
    "eigen": ("#3b6ea5", "o", "Eigen (vectorised)"),
    "statevector": ("#d1495b", "^", "StateVector (OpenMP)"),
}

_checks = []


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def marked_state(n):
    return (1 << n) // 3


def time_backend(name, n):
    """Best-of-REPEATS wall time of one Grover run; also returns the final probabilities."""
    backend = FACTORIES[name](n)
    marked = marked_state(n)
    best, probs = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = qa.Grover.run(backend, n, [marked], ITERS)
        best = min(best, time.perf_counter() - t0)
        probs = result.probabilities
    return best, probs


def grover_theory(n, k):
    theta = np.arcsin(2.0 ** (-n / 2))
    return float(np.sin((2 * k + 1) * theta) ** 2)


def find_crossover(ns, t_eigen, t_sv):
    """Sustained crossover: smallest n from which StateVector beats Eigen at every larger n
    tested, interpolated in log(ratio) from the last size where Eigen was ahead.
    Returns (n_x, status) with status in {"inside", "below", "none"}."""
    ratio = np.log(np.array(t_eigen) / np.array(t_sv))  # > 0 means StateVector faster
    if ratio[-1] <= 0:
        return None, "none"
    behind = [i for i in range(len(ns)) if ratio[i] <= 0]
    if not behind:
        return None, "below"
    i = behind[-1]
    frac = -ratio[i] / (ratio[i + 1] - ratio[i])
    return ns[i] + frac * (ns[i + 1] - ns[i]), "inside"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep():
    print(f"\n[1/3] Grover timing, {ITERS} iterations, best of {REPEATS}")
    print(f"  cores: {os.cpu_count()}   OMP_NUM_THREADS: {TAG if TAG else 'unset'}")

    time_backend("statevector", 12)  # thread-pool warm-up, not recorded

    times = {name: {} for name in FACTORIES}
    worst_agree, worst_theory = 0.0, 0.0
    for n in N_VALUES:
        probs = {}
        for name in FACTORIES:
            if name == "reference" and n > REF_MAX_N:
                continue
            times[name][n], probs[name] = time_backend(name, n)
        base = probs["eigen"]
        for name, p in probs.items():
            worst_agree = max(worst_agree, float(np.max(np.abs(p - base))))
        worst_theory = max(worst_theory,
                           abs(base[marked_state(n)] - grover_theory(n, ITERS)))
        print(f"  n={n} done", flush=True)
    return times, worst_agree, worst_theory


def print_table(times):
    print("\n  n    2^n       reference[s]  eigen[s]   statevec[s]  eigen/sv   ref/sv")
    for n in N_VALUES:
        ref = times["reference"].get(n)
        e, s = times["eigen"][n], times["statevector"][n]
        ref_txt = f"{ref:<13.4f}" if ref is not None else f"{'-':<13}"
        ref_ratio = f"{ref / s:.1f}x" if ref is not None else "-"
        print(f"  {n:<4d} {2 ** n:<9d} {ref_txt} {e:<10.4f} {s:<12.4f} {e / s:<10.2f} {ref_ratio}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(times, n_x, status):
    data = {
        "n_values": N_VALUES,
        "grover_iterations": ITERS,
        "repeats": REPEATS,
        "reference_max_n": REF_MAX_N,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": TAG,
        "seconds": {name: {str(n): t for n, t in d.items()} for name, d in times.items()},
        "crossover": {"statevector_sustainably_beats_eigen_at_n": n_x, "status": status},
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"  saved {JSON_PATH.relative_to(ROOT)}")


def plot(times, n_x, status):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, (colour, mark, label) in STYLE.items():
        ns = sorted(times[name])
        ax.plot([2 ** n for n in ns], [times[name][n] for n in ns], marker=mark, color=colour,
                lw=1.6, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2 ** n for n in N_VALUES])
    ax.set_xticklabels([f"$2^{{{n}}}$" for n in N_VALUES])
    ax.axvline(2 ** OMP_THRESHOLD_N, color="black", ls=":", lw=0.9)
    ax.text(2 ** OMP_THRESHOLD_N, ax.get_ylim()[0], " OpenMP on", fontsize=8, va="bottom")

    if status == "inside":
        ns = sorted(times["eigen"])
        y = float(np.exp(np.interp(n_x, ns, np.log([times["eigen"][n] for n in ns]))))
        ax.plot([2 ** n_x], [y], "k*", markersize=13, zorder=5)
        ax.annotate(f"sustained crossover  n = {n_x:.1f}", (2 ** n_x, y), xytext=(-130, 28),
                    textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="->"))
    else:
        msg = ("StateVector faster at every n tested" if status == "below"
               else "StateVector not ahead at the largest n tested")
        ax.text(0.03, 0.95, msg, transform=ax.transAxes, fontsize=9, va="top")

    ax.set_xlabel("state size (amplitudes)")
    ax.set_ylabel(f"time for Grover, {ITERS} iterations [s]")
    ax.set_title("Backend scaling" + (f" ({TAG} thread{'s' if TAG != '1' else ''})" if TAG else ""))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = IMAGES / f"backend_crossover{SUFFIX}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Backend crossover benchmark")
    print("=" * 64)

    times, worst_agree, worst_theory = sweep()

    print("\n[2/3] Timings")
    print_table(times)

    ns = N_VALUES
    n_x, status = find_crossover(ns, [times["eigen"][n] for n in ns],
                                 [times["statevector"][n] for n in ns])
    if status == "inside":
        print(f"\n  StateVector sustainably overtakes Eigen at n = {n_x:.1f} "
              f"(2^n = {2 ** n_x:.0f} amplitudes)")
    elif status == "below":
        print(f"\n  StateVector is ahead at every n tested (from n = {ns[0]}); "
              "the crossover lies below the tested range")
    else:
        print("\n  StateVector is not ahead at the largest n tested; no sustained crossover")
    big = ns[-1]
    print(f"  n={big}: StateVector is {times['eigen'][big] / times['statevector'][big]:.2f}x Eigen")
    n_ref = max(times["reference"])
    print(f"  n={n_ref}: StateVector is {times['reference'][n_ref] / times['statevector'][n_ref]:.1f}x Reference")

    print("\n[3/3] Output and checks")
    save_json(times, n_x, status)
    fig_path = plot(times, n_x, status)

    check("all backends agree: max |P_a - P_eigen| < 1e-10", worst_agree < 1e-10, f"{worst_agree:.2e}")
    check("marked-state probability = sin^2((2k+1) theta), < 1e-9", worst_theory < 1e-9,
          f"{worst_theory:.2e}")
    check("figure and JSON written", fig_path.exists() and JSON_PATH.exists())

    print(f"\n{sum(_checks)}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())