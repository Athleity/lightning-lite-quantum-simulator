"""Correlated matching against plain MWPM, decoded on identical shots.

Both decoders see the same sampled syndromes at every (d, p), so the comparison is
paired: the difference in failures comes from the shots on which the decoders
disagree, not from independent sampling noise. Reported:

  * per-round logical error eps_d(p) for each decoder
  * the fractional drop in logical failures from correlated matching, with a
    standard error and a McNemar z-score (negative means correlated is worse)
  * Lambda(p) for each decoder. Lambda(5 -> 7) is the clean number: correlated
    matching changes d = 3 differently from d = 5 and 7, which distorts 3 -> 5
  * the d = 5 / d = 7 and d = 3 / d = 7 threshold crossings with a resampling
    interval from the standard errors
  * decoding time per shot
  * a negative control: with reset and measurement flips only there are no Y-type
    error mechanisms, so both decoders must agree on every shot

Run:   python benchmarks/qec_correlated.py            (about a minute)
       python benchmarks/qec_correlated.py --quick    (smaller budget, noisier)
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

from qec_threshold import find_crossing  # same folder, same crossing rule as the baseline sweep

OUT_DIR = ROOT / "docs" / "images"

DISTANCES = (3, 5, 7)
DECODERS = ("mwpm", "correlated")
P_GRID = np.geomspace(0.002, 0.018, 9)

FULL = dict(max_shots=1_000_000, max_errors=250)
QUICK = dict(max_shots=60_000, max_errors=60)

BOOTSTRAP_DRAWS = 2000
LOW_P_MAX = 0.006        # "below threshold" region used by the checks
Z_SIGNIFICANT = 3.0      # McNemar z needed to call a difference real

CONTROL_P = 0.01
CONTROL_SHOTS = 50_000

C_BLUE, C_TEAL, C_RED, C_GRAY = "#1f4e79", "#2a9d8f", "#b5473a", "#6b6b6b"
DIST_COLORS = {3: C_BLUE, 5: C_TEAL, 7: C_RED}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(budget: dict, seed: int = 21) -> dict[int, list[sc.PairedComparison]]:
    data: dict[int, list[sc.PairedComparison]] = {d: [] for d in DISTANCES}
    for i, p in enumerate(P_GRID):
        for d in DISTANCES:
            mem = sc.SurfaceCodeMemory(d, float(p))
            data[d].append(mem.compare(DECODERS, seed=seed + 100 * i + d, **budget))
        print(f"  p = {p:.4f} done", flush=True)
    return data


def rates(data, d, name):
    return [c.rates[name] for c in data[d]]


def eps_arrays(rows):
    return np.array([r.eps for r in rows]), np.array([r.eps_se for r in rows])


def lambda_series(data, d_small, name):
    """Lambda(p) for one decoder, NaN where either distance saw no failures."""
    out = np.full(len(P_GRID), np.nan)
    for k in range(len(P_GRID)):
        a, b = data[d_small][k].rates[name], data[d_small + 2][k].rates[name]
        if a.errors and b.errors:
            out[k] = sc.lambda_pairwise(a, b).value
    return out


def crossing_interval(small_rows, large_rows, rng):
    """Crossing of two eps(p) curves with a 16-84 % interval from resampling.

    Each eps is redrawn from a normal with its standard error, the crossing is
    recomputed, and draws without a crossing inside the grid are counted and dropped.
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


def run_control(seed: int = 5):
    """Reset/measurement flips only: no Y-type hyperedges, decoders must agree."""
    return sc.SurfaceCodeMemory(5, CONTROL_P, noise="readout").compare(
        DECODERS, max_shots=CONTROL_SHOTS, max_errors=10**9, seed=seed)


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
    heading("Per-round logical error eps_d(p), plain MWPM | correlated")
    head = f"{'p':>8}"
    for d in DISTANCES:
        head += f"{'d=' + str(d) + ' mwpm':>13}{'corr':>11}"
    print(head)
    for k, p in enumerate(P_GRID):
        row = f"{p:>8.4f}"
        for d in DISTANCES:
            row += f"{data[d][k].rates['mwpm'].eps:>13.3e}{data[d][k].rates['correlated'].eps:>11.3e}"
        print(row)


def print_reduction_table(data):
    heading("Drop in logical failures from correlated matching (paired, identical shots)")
    print("  value = (failures_mwpm - failures_corr) / failures_mwpm, +/- standard error, [McNemar z]")
    print("  negative: correlated is worse")
    print(f"{'p':>8}" + "".join(f"{'d=' + str(d):>26}" for d in DISTANCES))
    for k, p in enumerate(P_GRID):
        row = f"{p:>8.4f}"
        for d in DISTANCES:
            red, se, z = data[d][k].reduction("correlated")
            row += f"{red * 100:>+11.1f} % +/- {se * 100:4.1f} [{z:>+6.1f}]"
        print(row)


def print_discordant_table(data):
    heading("Shots on which the decoders disagree (only one of them fails)")
    print(f"{'p':>8}" + "".join(f"{'d=' + str(d) + ' mwpm only':>16}{'corr only':>11}" for d in DISTANCES))
    for k, p in enumerate(P_GRID):
        row = f"{p:>8.4f}"
        for d in DISTANCES:
            c = data[d][k]
            row += f"{c.only_reference['correlated']:>16d}{c.only_other['correlated']:>11d}"
        print(row)


def print_lambda_table(data):
    heading(f"Lambda(p), plain MWPM | correlated (Willow {sc.WILLOW_LAMBDA} +/- {sc.WILLOW_LAMBDA_SE})")
    print("  Same shots feed both decoders, so the ratio has no simple standard error.")
    print(f"{'p':>8}{'3->5 mwpm':>12}{'corr':>8}{'5->7 mwpm':>12}{'corr':>8}{'ratio 5->7':>12}")
    l35 = {n: lambda_series(data, 3, n) for n in DECODERS}
    l57 = {n: lambda_series(data, 5, n) for n in DECODERS}
    for k, p in enumerate(P_GRID):
        ratio = l57["correlated"][k] / l57["mwpm"][k]
        print(f"{p:>8.4f}{l35['mwpm'][k]:>12.2f}{l35['correlated'][k]:>8.2f}"
              f"{l57['mwpm'][k]:>12.2f}{l57['correlated'][k]:>8.2f}{ratio:>12.2f}")
    return l35, l57


def print_threshold_table(data, rng):
    heading("Threshold estimates (curve crossings, 16-84 % interval from resampling)")
    out = {}
    print(f"{'pair':>10}{'plain MWPM':>40}{'correlated':>40}")
    for small, large in ((5, 7), (3, 7)):
        cells = []
        for name in DECODERS:
            res = crossing_interval(rates(data, small, name), rates(data, large, name), rng)
            out[(small, large, name)] = res
            cells.append(fmt_interval(*res))
        print(f"{'d=' + str(small) + '/' + str(large):>10}{cells[0]:>40}{cells[1]:>40}")
    return out


def print_timing_table(data):
    heading("Decoding time per 1000 shots (seconds, summed over the sweep)")
    print(f"{'':>6}{'plain MWPM':>14}{'correlated':>14}{'slowdown':>12}")
    for d in DISTANCES:
        t0 = sum(c.rates["mwpm"].seconds for c in data[d])
        t1 = sum(c.rates["correlated"].seconds for c in data[d])
        n = sum(c.shots for c in data[d])
        print(f"{'d=' + str(d):>6}{1000 * t0 / n:>14.3f}{1000 * t1 / n:>14.3f}{t1 / t0:>11.1f}x")


def print_control(control):
    heading(f"Negative control: reset/measurement flips only (d=5, p={CONTROL_P:g}, {control.shots} shots)")
    print(f"  failures: plain MWPM {control.rates['mwpm'].errors}, "
          f"correlated {control.rates['correlated'].errors}")
    print(f"  shots where only one decoder fails: "
          f"{control.only_reference['correlated'] + control.only_other['correlated']}")


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
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 100:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


XT = [0.002, 0.003, 0.005, 0.008, 0.012, 0.018]


def fig_gain(data, l57):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained")

    for d in DISTANCES:
        red = np.array([data[d][k].reduction("correlated")[0] for k in range(len(P_GRID))]) * 100
        se = np.array([data[d][k].reduction("correlated")[1] for k in range(len(P_GRID))]) * 100
        ax1.errorbar(P_GRID, red, yerr=se, color=DIST_COLORS[d], marker="o", markersize=4.5,
                     capsize=2, label=f"d = {d}")
    ax1.axhline(0.0, color="#444444", lw=1.0)
    log_xticks(ax1, XT)
    ax1.set_xlabel("Physical error rate p (%)")
    ax1.set_ylabel("Drop in logical failures (%)")
    ax1.set_title("Correlated vs plain matching, same shots", loc="left")
    ax1.legend(loc="best")

    ax2.plot(P_GRID, l57["mwpm"], color=C_BLUE, marker="o", markersize=4.5, ls="--", label="plain MWPM")
    ax2.plot(P_GRID, l57["correlated"], color=C_RED, marker="o", markersize=4.5, label="correlated")
    ax2.axhline(sc.WILLOW_LAMBDA, color=C_GRAY, ls="--", lw=1.0)
    ax2.annotate(f"Willow, {sc.WILLOW_LAMBDA}", xy=(0.99, sc.WILLOW_LAMBDA),
                 xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                 ha="right", fontsize=8.5, color="#555555")
    ax2.axhline(1.0, color="#999999", ls=":", lw=1.0)
    log_xticks(ax2, XT)
    ax2.set_yscale("log")
    ax2.set_yticks([1, 2, 4, 8])
    ax2.set_yticklabels(["1", "2", "4", "8"])
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel("Physical error rate p (%)")
    ax2.set_ylabel(r"Suppression factor $\Lambda$ (d = 5 to 7)")
    ax2.set_title(r"$\Lambda$ with and without correlations", loc="left")
    ax2.legend(loc="upper right")
    return save(fig, "qec_correlated_gain.png")


def fig_crossings(data, thresholds):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained", sharey=True)
    for ax, name, title in zip(axes, DECODERS, ("Plain MWPM", "Correlated matching")):
        for d in DISTANCES:
            eps, se = eps_arrays(rates(data, d, name))
            keep = eps > 0
            ax.errorbar(P_GRID[keep], eps[keep], yerr=se[keep], color=DIST_COLORS[d], marker="o",
                        markersize=4.5, capsize=2, label=f"d = {d}")
        point = thresholds[(5, 7, name)][0]
        if point is not None:
            ax.axvline(point, color=C_GRAY, ls="--", lw=1.0)
            ax.annotate(f"d=5/7 crossing {point * 100:.2f} %", xy=(point, 0.07),
                        xycoords=("data", "axes fraction"), xytext=(-6, 0),
                        textcoords="offset points", ha="right", fontsize=8.5, color="#555555")
        log_xticks(ax, XT)
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate p (%)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Logical error per round")
    axes[0].legend(loc="lower right")
    return save(fig, "qec_correlated_threshold.png")


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
            print(f"{name:<74}{'PASS' if ok else 'FAIL'}")
        print()
        print("ALL CHECKS PASSED" if all(ok for _, ok in self.rows) else "SOME CHECKS FAILED")
        return all(ok for _, ok in self.rows)


def run_checks(data, control, l57, thresholds):
    c = Checks()
    c.add("Control (no Y-type mechanisms): decoders agree on every shot",
          control.only_reference["correlated"] == 0 and control.only_other["correlated"] == 0)

    low = [k for k, p in enumerate(P_GRID) if p <= LOW_P_MAX]
    for d in (5, 7):
        # A single low-p point can hold too few failures to be significant on its
        # own, so test the sign at every point and the significance of the pooled
        # discordant shots over the region.
        b = sum(data[d][k].only_reference["correlated"] for k in low)
        n_only = sum(data[d][k].only_other["correlated"] for k in low)
        z_pool = (b - n_only) / math.sqrt(b + n_only) if b + n_only else 0.0
        signs = all(data[d][k].reduction("correlated")[0] > 0 for k in low)
        c.add(f"d={d}, p <= {LOW_P_MAX * 100:g} %: correlated better at every p, pooled z > {Z_SIGNIFICANT:g}",
              signs and z_pool > Z_SIGNIFICANT)

    lam_ok = [l57["correlated"][k] > l57["mwpm"][k] for k in low
              if not (np.isnan(l57["correlated"][k]) or np.isnan(l57["mwpm"][k]))]
    c.add(f"Lambda(5->7) higher with correlations at every p <= {LOW_P_MAX * 100:g} %",
          len(lam_ok) > 0 and all(lam_ok))

    pm, pc = thresholds[(5, 7, "mwpm")][0], thresholds[(5, 7, "correlated")][0]
    c.add("d=5/7 crossing found for both decoders", pm is not None and pc is not None)
    c.add("d=5/7 crossing: correlated >= plain MWPM", pm is not None and pc is not None and pc >= pm)

    last = len(P_GRID) - 1
    c.add("Above threshold: correlated and plain both still lose to larger d (d=3 beats d=7)",
          all(data[3][last].rates[n].eps < data[7][last].rates[n].eps for n in DECODERS))
    return c.report()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small shot budget, noisier")
    args = ap.parse_args()
    budget = QUICK if args.quick else FULL

    set_style()
    print(f"Sweeping p = {P_GRID[0]:.4f} .. {P_GRID[-1]:.4f}, d = {DISTANCES}, decoders = {DECODERS}, "
          f"up to {budget['max_shots']} shots or {budget['max_errors']} plain-MWPM failures per point")
    data = run_sweep(budget)
    control = run_control()

    print_eps_table(data)
    print_reduction_table(data)
    print_discordant_table(data)
    _, l57 = print_lambda_table(data)
    thresholds = print_threshold_table(data, np.random.default_rng(0))
    print_timing_table(data)
    print_control(control)

    paths = [fig_gain(data, l57), fig_crossings(data, thresholds)]
    ok = run_checks(data, control, l57, thresholds)
    print()
    for p in paths:
        print(f"saved {p.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())