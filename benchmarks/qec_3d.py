"""3D toric code threshold sweep (sector z, SD6 noise, MWPM) and comparison to the 2D surface code.

Runs the periodic d x d x d toric code memory from src/ThreeDSurfaceCode.py for d = 3, 5, 7 over a
log grid of physical error rates p, scoring all three logical qubits (a shot fails if any of the
three observables is mispredicted). Reported:

  * per-round logical error eps_d(p) with standard errors, and the suppression
    ratios eps_3/eps_5 and eps_5/eps_7 at every p
  * the d = 3/5 and d = 5/7 threshold crossings, with a resampling interval from the
    standard errors
  * Lambda(5 -> 7) at fixed p = 0.3 % (and Lambda(3 -> 5) for reference)
  * a comparison with the 2D surface code (1.30 % correlated MWPM, 1.11 % plain MWPM, both
    circuit level, from benchmarks/qec_correlated.py) and with the literature code-capacity
    thresholds for the 3D toric code (point-like ~23 %, loop-like ~3.3 %)

Only sector z is swept: sector x has loop-like syndromes and no matching decoder in this repo.
The per-round eps assumes one logical bit, so with three logicals scored it is approximate;
the crossings are unaffected in the sense that every distance is converted the same way, but
treat absolute eps values with care. The literature numbers are code-capacity thresholds under
optimal decoding and are not directly comparable with a circuit-level MWPM threshold, so the
comparison is context, not a test. Finite-size effects at d <= 7 move the crossings away from
the asymptotic threshold, and the crossing is linear interpolation in log-log, so it is an
estimate.

The "eps rises with p" check is a whole-range test, not an adjacent-pair test: adjacent grid
points at low p hold few failures and can fluctuate. It keeps points with failures and
p_logical < 0.5 (beyond that the one-bit per-round conversion saturates), requires a positive
log-log slope, and flags only pairs of points at least 1.5x apart in p whose eps drops by more
than 2 standard errors.

Run:   python benchmarks/qec_3d.py            (long: d = 7 with SD6 is slow at low p)
       python benchmarks/qec_3d.py --quick    (small shot budget, noisier)
Exit status is 0 when every check passes. PNG goes to docs/images/, JSON to docs/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("src", "python"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import SurfaceCode as sc
    import ThreeDSurfaceCode as t3
except ImportError as exc:
    sys.exit(f"Could not import src/SurfaceCode.py or src/ThreeDSurfaceCode.py: {exc}\n"
             "Install with: pip install stim pymatching")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from qec_threshold import find_crossing  # same folder, same crossing rule as the 2D sweeps

OUT_DIR = ROOT / "docs" / "images"
JSON_PATH = ROOT / "docs" / "qec_3d_benchmark.json"

DISTANCES = (3, 5, 7)
NOISE = "sd6"
SECTOR = "z"
LOGICALS = 3

LAMBDA_P = 0.003
P_GRID = np.unique(np.append(np.geomspace(0.001, 0.02, 12), LAMBDA_P))

FULL = dict(max_shots=200_000, max_errors=200)
QUICK = dict(max_shots=6_000, max_errors=40)
LAMBDA_BOOST = 2          # the Lambda point gets this many times the shot budget
CHUNK = 5_000

BOOTSTRAP_DRAWS = 2000
BELOW_FRACTION = 0.7      # "below threshold" means p <= this fraction of the d=5/7 crossing
RISE_SIGMAS = 2.0         # drop in eps tolerated, in standard errors, by the rises-with-p check
RISE_MIN_RATIO = 1.5      # only pairs of points at least this far apart in p are compared
P_LOGICAL_MAX = 0.5       # beyond this the one-bit per-round conversion is not meaningful

C_BLUE, C_TEAL, C_RED, C_GRAY = "#1f4e79", "#2a9d8f", "#b5473a", "#6b6b6b"
C_2D, C_2D_MWPM = "#7a5195", "#d08c2e"
DIST_COLORS = {3: C_BLUE, 5: C_TEAL, 7: C_RED}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sample_point(d: int, p: float, budget: dict, seed: int) -> sc.LogicalErrorRate:
    """Sample in chunks until max_errors failures or max_shots shots, whichever comes first."""
    code = t3.ThreeDSurfaceCode(d, float(p), noise=NOISE, sector=SECTOR, logicals=LOGICALS)
    shots = errors = k = 0
    seconds = 0.0
    while shots < budget["max_shots"] and errors < budget["max_errors"]:
        n = min(CHUNK, budget["max_shots"] - shots)
        r = code.sample(n, seed=seed + k)
        shots += r.shots
        errors += r.errors
        seconds += r.seconds
        k += 1
    return sc._logical_rate(d, code.rounds, float(p), shots, errors, seconds)


def run_sweep(budget: dict, seed: int = 31) -> dict[int, list[sc.LogicalErrorRate]]:
    data: dict[int, list[sc.LogicalErrorRate]] = {d: [] for d in DISTANCES}
    for i, p in enumerate(P_GRID):
        b = dict(budget)
        if math.isclose(p, LAMBDA_P):
            b = {key: LAMBDA_BOOST * v for key, v in budget.items()}
        for j, d in enumerate(DISTANCES):
            data[d].append(sample_point(d, float(p), b, seed + 1000 * (i * len(DISTANCES) + j)))
        print(f"  p = {p:.4f} done", flush=True)
    return data


def eps_arrays(rows):
    return np.array([r.eps for r in rows]), np.array([r.eps_se for r in rows])


def rises_with_p(rows, sigmas=RISE_SIGMAS, min_ratio=RISE_MIN_RATIO):
    """Whole-range monotonicity of eps(p), on points where eps is meaningful.

    Keeps points with failures and p_logical < P_LOGICAL_MAX (the per-round conversion breaks
    down beyond that). Requires a positive log-log slope, and no pair of points at least
    min_ratio apart in p where eps drops by more than `sigmas` standard errors.
    Returns (ok, list of violating (p_low, p_high) pairs).
    """
    pts = [(float(P_GRID[k]), r) for k, r in enumerate(rows)
           if r.errors > 0 and r.p_logical < P_LOGICAL_MAX]
    bad = []
    for i, (pi, ri) in enumerate(pts):
        for pj, rj in pts[i + 1:]:
            if pj >= min_ratio * pi and rj.eps + sigmas * math.hypot(ri.eps_se, rj.eps_se) < ri.eps:
                bad.append((pi, pj))
    slope = (np.polyfit(np.log([p for p, _ in pts]), np.log([r.eps for _, r in pts]), 1)[0]
             if len(pts) >= 3 else float("nan"))
    return (len(pts) >= 3 and slope > 0 and not bad), bad


def crossing_interval(small_rows, large_rows, rng):
    """Crossing of two eps(p) curves with a 16-84 % interval from resampling.

    Each eps is redrawn from a normal with its standard error, the crossing is recomputed,
    and draws without a crossing inside the grid are counted and dropped.
    Returns (point, low, high, fraction_of_draws_with_a_crossing).
    """
    e_s, s_s = eps_arrays(small_rows)
    e_l, s_l = eps_arrays(large_rows)
    point = find_crossing(P_GRID, e_s, e_l)
    found = []
    for _ in range(BOOTSTRAP_DRAWS):
        x = find_crossing(P_GRID, np.maximum(rng.normal(e_s, s_s), 1e-300),
                          np.maximum(rng.normal(e_l, s_l), 1e-300))
        if x is not None:
            found.append(x)
    if not found:
        return point, None, None, 0.0
    lo, hi = np.percentile(found, [16, 84])
    return point, float(lo), float(hi), len(found) / BOOTSTRAP_DRAWS


def lambda_at(data, d_small, p):
    """Lambda(d_small -> d_small + 2) at the grid point nearest p, or None without failures."""
    k = int(np.argmin(np.abs(P_GRID - p)))
    a, b = data[d_small][k], data[d_small + 2][k]
    if not (a.errors and b.errors):
        return None
    lam = sc.lambda_pairwise(a, b)
    return float(lam.value), float(lam.se)


def ratio(a, b):
    return a.eps / b.eps if a.errors and b.errors else float("nan")


# ---------------------------------------------------------------------------
# Printed tables
# ---------------------------------------------------------------------------

def heading(text):
    print()
    print(text)
    print("-" * len(text))


def fmt_interval(point, lo, hi, frac):
    if point is None and lo is None:
        return "no crossing"
    s = "n/a" if point is None else f"{point * 100:.2f} %"
    if lo is not None:
        s += f"  [{lo * 100:.2f}, {hi * 100:.2f}]"
    if frac < 0.95:
        s += f"  ({frac * 100:.0f}% of draws cross)"
    return s


def print_eps_table(data):
    heading(f"Per-round logical error eps_d(p): 3D toric code, sector {SECTOR}, {NOISE}, "
            f"{LOGICALS} logicals, MWPM")
    print(f"{'p':>8}{'eps_d3':>12}{'eps_d5':>12}{'eps_d7':>12}{'ratio 3->5':>12}{'ratio 5->7':>12}")
    for k, p in enumerate(P_GRID):
        r3, r5, r7 = data[3][k], data[5][k], data[7][k]
        print(f"{p:>8.4f}{r3.eps:>12.3e}{r5.eps:>12.3e}{r7.eps:>12.3e}"
              f"{ratio(r3, r5):>12.2f}{ratio(r5, r7):>12.2f}")
    print("  ratio = eps_d / eps_(d+2); n/a (nan) where either distance saw no failures")


def print_shot_table(data):
    heading("Shots and logical failures")
    print(f"{'p':>8}" + "".join(f"{'shots d=' + str(d):>14}{'fails':>8}{'p_logical':>11}" for d in DISTANCES))
    for k, p in enumerate(P_GRID):
        row = f"{p:>8.4f}"
        for d in DISTANCES:
            r = data[d][k]
            row += f"{r.shots:>14d}{r.errors:>8d}{r.p_logical:>11.3f}"
        print(row)


def print_threshold_table(data, rng):
    heading("Threshold estimates (curve crossings, 16-84 % interval from resampling)")
    out = {}
    for small, large in ((3, 5), (5, 7)):
        res = crossing_interval(data[small], data[large], rng)
        out[(small, large)] = res
        print(f"  d={small}/{large}: {fmt_interval(*res)}")
    return out


def print_lambda(data):
    heading(f"Lambda at fixed p = {LAMBDA_P * 100:g} % (Willow 2D: {sc.WILLOW_LAMBDA} +/- {sc.WILLOW_LAMBDA_SE})")
    out = {}
    for d in (3, 5):
        lam = lambda_at(data, d, LAMBDA_P)
        out[(d, d + 2)] = lam
        print(f"  Lambda({d} -> {d + 2}) = " + ("n/a (no failures at one distance)" if lam is None
                                                else f"{lam[0]:.2f} +/- {lam[1]:.2f}"))
    return out


def print_comparison(thresholds):
    heading("Comparison with the 2D surface code")
    p57 = thresholds[(5, 7)][0]
    p35 = thresholds[(3, 5)][0]
    print(f"  2D surface code, circuit level (qec_correlated): {t3.TWO_D_THRESHOLD * 100:.2f} % correlated "
          f"MWPM, {t3.TWO_D_THRESHOLD_MWPM * 100:.2f} % plain MWPM")
    print("  3D toric code, circuit level, sector z MWPM (this scan): "
          f"d=3/5 {'none' if p35 is None else f'{p35 * 100:.2f} %'}, "
          f"d=5/7 {'none' if p57 is None else f'{p57 * 100:.2f} %'}")
    if p57 is not None:
        print(f"  3D / 2D threshold ratio (d=5/7 vs 2D): {p57 / t3.TWO_D_THRESHOLD:.2f} (correlated), "
              f"{p57 / t3.TWO_D_THRESHOLD_MWPM:.2f} (plain MWPM)")
    print(f"  literature, code capacity, optimal decoding (as recalled, check before quoting): "
          f"point-like ~{t3.LIT_POINT_CODE_CAPACITY * 100:.0f} %, loop-like ~{t3.LIT_LOOP_CODE_CAPACITY * 100:.1f} %")
    print("  the literature numbers are code-capacity, this scan is circuit level with a matching "
          "decoder: different quantities, do not compare directly")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "regular", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#dcdcdc",
        "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
        "legend.fontsize": 9, "mathtext.default": "regular",
    })


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def fig_threshold(data, thresholds):
    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    for d in DISTANCES:
        eps, se = eps_arrays(data[d])
        keep = eps > 0
        ax.errorbar(P_GRID[keep], eps[keep], yerr=se[keep], color=DIST_COLORS[d], marker="o",
                    markersize=4.5, capsize=2, label=f"d = {d}")

    p57 = thresholds[(5, 7)][0]
    if p57 is not None:
        ax.axvline(p57, color=C_GRAY, ls="--", lw=1.0)
        ax.annotate(f"3D d=5/7 crossing {p57 * 100:.2f} %", xy=(p57, 0.06),
                    xycoords=("data", "axes fraction"), xytext=(-6, 0),
                    textcoords="offset points", ha="right", fontsize=8.5, color="#555555")
    ax.axvline(t3.TWO_D_THRESHOLD_MWPM, color=C_2D_MWPM, ls=":", lw=1.3)
    ax.annotate(f"2D MWPM {t3.TWO_D_THRESHOLD_MWPM * 100:.2f} %", xy=(t3.TWO_D_THRESHOLD_MWPM, 0.20),
                xycoords=("data", "axes fraction"), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=C_2D_MWPM)
    ax.axvline(t3.TWO_D_THRESHOLD, color=C_2D, ls=":", lw=1.3)
    ax.annotate(f"2D correlated {t3.TWO_D_THRESHOLD * 100:.2f} %", xy=(t3.TWO_D_THRESHOLD, 0.32),
                xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points",
                ha="left", fontsize=8.5, color=C_2D)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 100:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Physical error rate p (%)")
    ax.set_ylabel("Logical error per round")
    ax.set_title("3D toric code, SD6, MWPM: d = 3, 5, 7 against the 2D thresholds", loc="left")
    ax.legend(loc="lower right")
    return save(fig, "qec_3d_threshold.png")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def num(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def write_json(data, thresholds, lambdas, budget, checks):
    out = {
        "config": {"distances": list(DISTANCES), "noise": NOISE, "sector": SECTOR,
                   "logicals": LOGICALS, "p_grid": [float(p) for p in P_GRID],
                   "lambda_p": LAMBDA_P, "budget": budget, "lambda_boost": LAMBDA_BOOST,
                   "rise_check": {"sigmas": RISE_SIGMAS, "min_ratio": RISE_MIN_RATIO,
                                  "p_logical_max": P_LOGICAL_MAX}},
        "results": {str(d): [{"p": float(P_GRID[k]), "rounds": r.rounds, "shots": int(r.shots),
                              "errors": int(r.errors), "p_logical": num(r.p_logical),
                              "eps": num(r.eps), "eps_se": num(r.eps_se)}
                             for k, r in enumerate(data[d])] for d in DISTANCES},
        "thresholds": {f"{a}/{b}": {"point": num(v[0]), "low": num(v[1]), "high": num(v[2]),
                                    "fraction_crossing": num(v[3])}
                       for (a, b), v in thresholds.items()},
        "lambda": {f"{a}->{b}": (None if v is None else {"value": num(v[0]), "se": num(v[1])})
                   for (a, b), v in lambdas.items()},
        "comparison": {"two_d_correlated_mwpm": t3.TWO_D_THRESHOLD,
                       "two_d_mwpm": t3.TWO_D_THRESHOLD_MWPM,
                       "literature_point_like_code_capacity": t3.LIT_POINT_CODE_CAPACITY,
                       "literature_loop_like_code_capacity": t3.LIT_LOOP_CODE_CAPACITY,
                       "note": "literature numbers are code-capacity under optimal decoding, "
                               "not comparable with a circuit-level matching threshold"},
        "checks": [{"name": n, "passed": bool(ok)} for n, ok in checks],
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))
    return JSON_PATH


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok):
        self.rows.append((name, bool(ok)))

    def report(self):
        heading("Checks")
        for name, ok in self.rows:
            print(f"{name:<72}{'PASS' if ok else 'FAIL'}")
        print()
        print("ALL CHECKS PASSED" if all(ok for _, ok in self.rows) else "SOME CHECKS FAILED")
        return all(ok for _, ok in self.rows)


def run_checks(data, thresholds, lambdas):
    c = Checks()
    for d in DISTANCES:
        ok_d, bad = rises_with_p(data[d])
        c.add(f"d={d}: eps rises with p (whole range, 2 sigma, p_logical < 0.5)", ok_d)
        for pi, pj in bad:
            print(f"  d={d}: eps({pj:.4f}) below eps({pi:.4f}) beyond 2 sigma")

    p57 = thresholds[(5, 7)][0]
    below = [k for k, p in enumerate(P_GRID)
             if p57 is not None and p <= BELOW_FRACTION * p57 and all(data[d][k].errors > 0 for d in DISTANCES)]
    c.add(f"Below threshold (p <= {BELOW_FRACTION:g} x d=5/7 crossing): eps_3 > eps_5 > eps_7",
          len(below) > 0 and all(data[3][k].eps > data[5][k].eps > data[7][k].eps for k in below))

    c.add("Crossings d=3/5 and d=5/7 both found inside the scan range",
          thresholds[(3, 5)][0] is not None and thresholds[(5, 7)][0] is not None)

    lam = lambdas[(5, 7)]
    c.add(f"Lambda(5 -> 7) > 1 at p = {LAMBDA_P * 100:g} %", lam is not None and lam[0] > 1.0)
    return c.report(), c.rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small shot budget, noisier")
    args = ap.parse_args()
    budget = QUICK if args.quick else FULL

    set_style()
    print(f"Sweeping p = {P_GRID[0]:.4f} .. {P_GRID[-1]:.4f} ({len(P_GRID)} points), d = {DISTANCES}, "
          f"sector {SECTOR}, noise {NOISE}, up to {budget['max_shots']} shots or "
          f"{budget['max_errors']} failures per point ({LAMBDA_BOOST}x at p = {LAMBDA_P:g})")
    data = run_sweep(budget)

    print_eps_table(data)
    print_shot_table(data)
    thresholds = print_threshold_table(data, np.random.default_rng(0))
    lambdas = print_lambda(data)
    print_comparison(thresholds)

    fig_path = fig_threshold(data, thresholds)
    ok, rows = run_checks(data, thresholds, lambdas)
    json_path = write_json(data, thresholds, lambdas, budget, rows)
    print()
    print(f"saved {fig_path.relative_to(ROOT)}")
    print(f"saved {json_path.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())