"""Surface code threshold and error suppression sweep.

Runs the rotated surface code memory from src/SurfaceCode.py for d = 3, 5, 7 over a
range of physical error rates p (SD6 circuit noise, MWPM decoding) and reports:

  * per-round logical error eps_d(p), with standard errors
  * the threshold p_th, taken as the crossing of the d = 3 and d = 7 curves
    (below p_th larger codes win, above it they lose)
  * Lambda(p) = eps_{d-2} / eps_d for d = 3 -> 5 and 5 -> 7, against Willow's 2.14
  * eps against d at fixed p, with the exponential fit eps ~ C / Lambda^((d+1)/2)
  * logical lifetime against a physical qubit at the lowest p

The crossing is linear interpolation of ln eps_3 - ln eps_7 in ln p, so it is a few
tens of percent accurate at this shot budget, and finite-size effects at d <= 7
move it from the asymptotic threshold. Treat it as an estimate.

Run:   python benchmarks/qec_threshold.py            (about 10 min on one core)
       python benchmarks/qec_threshold.py --quick    (about a minute, noisier)
Exit status is 0 when every check passes. PNGs go to docs/images/.
"""

from __future__ import annotations

import argparse
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
except ImportError as exc:
    sys.exit(f"Could not import src/SurfaceCode.py: {exc}\nInstall with: pip install stim pymatching")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

OUT_DIR = ROOT / "docs" / "images"

DISTANCES = (3, 5, 7)
P_GRID = np.geomspace(0.002, 0.018, 10)
REFERENCE_P = float(P_GRID[0])

FULL = dict(max_shots=1_000_000, max_errors=250)
QUICK = dict(max_shots=60_000, max_errors=60)

# Plausible circuit-level threshold window for this noise model, used as a sanity
# check on the estimate, not as a target.
THRESHOLD_WINDOW = (0.004, 0.017)

C_BLUE, C_TEAL, C_RED, C_GRAY = "#1f4e79", "#2a9d8f", "#b5473a", "#6b6b6b"
DIST_COLORS = {3: C_BLUE, 5: C_TEAL, 7: C_RED}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(budget: dict, seed: int = 11) -> dict[int, list[sc.LogicalErrorRate]]:
    data: dict[int, list[sc.LogicalErrorRate]] = {d: [] for d in DISTANCES}
    for i, p in enumerate(P_GRID):
        for d in DISTANCES:
            r = sc.SurfaceCodeMemory(d, float(p)).sample(seed=seed + 100 * i + d, **budget)
            data[d].append(r)
        print(f"  p = {p:.4f} done", flush=True)
    return data


def eps_arrays(rows):
    eps = np.array([r.eps for r in rows])
    se = np.array([r.eps_se for r in rows])
    return eps, se


def find_crossing(p, eps_small, eps_large):
    """p where eps_small(p) = eps_large(p), by linear interpolation in log-log.

    Returns None if the curves do not cross inside the grid. Points with no observed
    errors are skipped.
    """
    ok = (eps_small > 0) & (eps_large > 0)
    lp = np.log(np.asarray(p)[ok])
    diff = np.log(eps_small[ok]) - np.log(eps_large[ok])  # > 0 below threshold
    for k in range(len(diff) - 1):
        if diff[k] > 0 >= diff[k + 1]:
            t = diff[k] / (diff[k] - diff[k + 1])
            return float(math.exp(lp[k] + t * (lp[k + 1] - lp[k])))
    return None


def lambda_curve(data, d_small):
    out_p, out_l, out_se = [], [], []
    for k, p in enumerate(P_GRID):
        a, b = data[d_small][k], data[d_small + 2][k]
        if a.errors and b.errors:
            lam = sc.lambda_pairwise(a, b)
            out_p.append(p)
            out_l.append(lam.value)
            out_se.append(lam.se)
    return np.array(out_p), np.array(out_l), np.array(out_se)


def crossing_of_lambda(p, lam, target):
    """p at which Lambda(p) falls to `target`, interpolated in log-log."""
    for k in range(len(p) - 1):
        if lam[k] >= target > lam[k + 1]:
            t = (math.log(lam[k]) - math.log(target)) / (math.log(lam[k]) - math.log(lam[k + 1]))
            return float(math.exp(math.log(p[k]) + t * (math.log(p[k + 1]) - math.log(p[k]))))
    return None


# ---------------------------------------------------------------------------
# Printed tables
# ---------------------------------------------------------------------------

def heading(text):
    print()
    print(text)
    print("-" * len(text))


def print_eps_table(data):
    heading("Per-round logical error eps_d(p)")
    print(f"{'p':>8}" + "".join(f"{'d=' + str(d):>12}{'+/-':>10}" for d in DISTANCES) + f"{'d=3/d=7':>10}")
    for k, p in enumerate(P_GRID):
        row = f"{p:>8.4f}"
        for d in DISTANCES:
            r = data[d][k]
            row += f"{r.eps:>12.3e}{r.eps_se:>10.1e}"
        e3, e7 = data[3][k].eps, data[7][k].eps
        row += f"{(e3 / e7 if e7 > 0 else float('nan')):>10.2f}"
        print(row)


def print_shot_table(data):
    heading("Shots and logical failures")
    print(f"{'p':>8}" + "".join(f"{'shots d=' + str(d):>14}{'fails':>8}" for d in DISTANCES))
    for k, p in enumerate(P_GRID):
        row = f"{p:>8.4f}"
        for d in DISTANCES:
            r = data[d][k]
            row += f"{r.shots:>14d}{r.errors:>8d}"
        print(row)


def print_lambda_table(curves):
    heading(f"Lambda(p) against Willow {sc.WILLOW_LAMBDA} +/- {sc.WILLOW_LAMBDA_SE}")
    print(f"{'p':>8}{'3 -> 5':>10}{'+/-':>8}{'5 -> 7':>10}{'+/-':>8}")
    (p1, l1, s1), (p2, l2, s2) = curves
    for p in P_GRID:
        row = f"{p:>8.4f}"
        for pp, ll, ss in ((p1, l1, s1), (p2, l2, s2)):
            hit = np.nonzero(np.isclose(pp, p))[0]
            row += (f"{ll[hit[0]]:>10.2f}{ss[hit[0]]:>8.2f}" if len(hit) else f"{'n/a':>10}{'':>8}")
        print(row)


# ---------------------------------------------------------------------------
# Plots
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


def log_xticks(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 100:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def fig_threshold(data, curves, p_th):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained")

    for d in DISTANCES:
        eps, se = eps_arrays(data[d])
        keep = eps > 0
        ax1.errorbar(P_GRID[keep], eps[keep], yerr=se[keep], color=DIST_COLORS[d], marker="o",
                     markersize=4.5, capsize=2, label=f"d = {d}")
    if p_th is not None:
        ax1.axvline(p_th, color=C_GRAY, ls="--", lw=1.0)
        ax1.annotate(f"crossing {p_th * 100:.2f} %", xy=(p_th, 0.6), xycoords=("data", "axes fraction"),
                     xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8.5,
                     color="#555555")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    log_xticks(ax1, [0.002, 0.003, 0.005, 0.008, 0.012, 0.018])
    ax1.set_xlabel("Physical error rate p (%)")
    ax1.set_ylabel("Logical error per round")
    ax1.set_title("Threshold: d = 3, 5, 7 cross where larger codes stop helping", loc="left")
    ax1.legend(loc="lower right")

    for (pp, ll, ss), (a, b), color in zip(curves, ((3, 5), (5, 7)), (C_BLUE, C_RED)):
        ax2.errorbar(pp, ll, yerr=ss, color=color, marker="o", markersize=4.5, capsize=2,
                     label=f"d = {a} to {b}")
    ax2.axhline(sc.WILLOW_LAMBDA, color=C_GRAY, ls="--", lw=1.0)
    ax2.annotate(f"Willow, {sc.WILLOW_LAMBDA}", xy=(0.99, sc.WILLOW_LAMBDA), xycoords=("axes fraction", "data"),
                 xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8.5, color="#555555")
    ax2.axhline(1.0, color="#999999", ls=":", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    log_xticks(ax2, [0.002, 0.003, 0.005, 0.008, 0.012, 0.018])
    ax2.set_yticks([1, 2, 4, 8])
    ax2.set_yticklabels(["1", "2", "4", "8"])
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel("Physical error rate p (%)")
    ax2.set_ylabel(r"Suppression factor $\Lambda$")
    ax2.set_title(r"$\Lambda = \epsilon_{d-2}\,/\,\epsilon_d$ against p", loc="left")
    ax2.legend(loc="upper right")
    return save(fig, "qec_threshold.png")


def fig_suppression(data):
    ps = [REFERENCE_P, float(P_GRID[2]), float(P_GRID[4])]
    idx = [int(np.argmin(np.abs(P_GRID - p))) for p in ps]
    palette = (C_BLUE, C_TEAL, C_RED)
    fig, ax = plt.subplots(figsize=(5.8, 4.4), layout="constrained")
    for k, color in zip(idx, palette):
        rows = [data[d][k] for d in DISTANCES]
        d_arr = np.array(DISTANCES)
        eps = np.array([r.eps for r in rows])
        se = np.array([r.eps_se for r in rows])
        ax.errorbar(d_arr, eps, yerr=se, color=color, marker="o", markersize=5, capsize=2, ls="none")
        label = f"p = {P_GRID[k] * 100:.2f} %"
        try:
            fit = sc.lambda_fit(rows)
            x = np.linspace(2.7, 7.3, 50)
            c = math.exp(np.mean(np.log(eps) + (d_arr + 1) / 2 * math.log(fit.value)))
            ax.plot(x, c / fit.value ** ((x + 1) / 2), color=color, lw=1.4,
                    label=rf"{label}, $\Lambda$ = {fit.value:.2f}")
        except ValueError:
            ax.plot([], [], color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(DISTANCES)
    ax.set_xlabel("Code distance d")
    ax.set_ylabel("Logical error per round")
    ax.set_title(r"Exponential suppression, $\epsilon_d \propto \Lambda^{-(d+1)/2}$", loc="left")
    ax.legend(loc="upper right")
    return save(fig, "qec_suppression.png")


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
            print(f"{name:<66}{'PASS' if ok else 'FAIL'}")
        print()
        print("ALL CHECKS PASSED" if all(ok for _, ok in self.rows) else "SOME CHECKS FAILED")
        return all(ok for _, ok in self.rows)


def run_checks(data, p_th, curves):
    c = Checks()
    for d in DISTANCES:
        eps, se = eps_arrays(data[d])
        slack = 2.0 * np.hypot(se[1:], se[:-1])
        c.add(f"d={d}: eps rises with p (within 2 sigma)", np.all(eps[1:] + slack >= eps[:-1]))
    e = [data[d][0].eps for d in DISTANCES]
    c.add(f"Below threshold (p={P_GRID[0]:.4f}): eps_3 > eps_5 > eps_7", e[0] > e[1] > e[2])
    e = [data[d][-1].eps for d in DISTANCES]
    c.add(f"Above threshold (p={P_GRID[-1]:.4f}): ordering reversed", e[0] < e[2])
    c.add("d=3 and d=7 curves cross inside the grid", p_th is not None)
    c.add(f"Crossing inside {THRESHOLD_WINDOW[0] * 100:g}% to {THRESHOLD_WINDOW[1] * 100:g}%",
          p_th is not None and THRESHOLD_WINDOW[0] <= p_th <= THRESHOLD_WINDOW[1])
    (_, l1, _), (_, l2, _) = curves
    c.add("Lambda > 1 at the lowest p, both distance pairs",
          len(l1) > 0 and len(l2) > 0 and l1[0] > 1.0 and l2[0] > 1.0)
    c.add("Lambda falls as p rises (3 -> 5)", len(l1) > 1 and l1[0] > l1[-1])
    return c.report()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small shot budget, noisier")
    args = ap.parse_args()
    budget = QUICK if args.quick else FULL

    set_style()
    print(f"Sweeping p = {P_GRID[0]:.4f} .. {P_GRID[-1]:.4f}, d = {DISTANCES}, "
          f"up to {budget['max_shots']} shots or {budget['max_errors']} failures per point")
    data = run_sweep(budget)

    curves = [lambda_curve(data, 3), lambda_curve(data, 5)]
    e3, _ = eps_arrays(data[3])
    e7, _ = eps_arrays(data[7])
    p_th = find_crossing(P_GRID, e3, e7)

    print_eps_table(data)
    print_shot_table(data)
    print_lambda_table(curves)

    heading("Summary")
    print(f"  threshold estimate (d=3 / d=7 crossing): "
          f"{'not found' if p_th is None else f'{p_th * 100:.2f} %'}")
    for (pp, ll, _), name in zip(curves, ("3 -> 5", "5 -> 7")):
        p_w = crossing_of_lambda(pp, ll, sc.WILLOW_LAMBDA) if len(pp) > 1 else None
        print(f"  Lambda({name}) reaches {sc.WILLOW_LAMBDA} at p = "
              f"{'outside grid' if p_w is None else f'{p_w * 100:.2f} %'}")
    top = data[7][0] if data[7][0].errors else None
    if top is not None:
        print(f"  lowest p, d = 7: {sc.breakeven(top)}")
        print(f"    against Willow's best physical qubit: "
              f"{sc.breakeven(top, physical_lifetime=sc.WILLOW_BEST_PHYSICAL_S)}")
        print("  (the simulated p is not tied to a T1, so these ratios depend on the chosen cycle time)")

    paths = [fig_threshold(data, curves, p_th), fig_suppression(data)]
    ok = run_checks(data, p_th, curves)
    print()
    for p in paths:
        print(f"saved {p.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())