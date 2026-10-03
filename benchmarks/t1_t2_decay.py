"""Validate python/noise_models.py (and the C++ core behind it) against closed forms.

Analytic ground truth, all computed independently of the module under test:

    P1(t)       = exp(-t / T1)                      excited population from |1>
    C(t)        = |rho01(t)| / |rho01(0)| = exp(-t / T2)
    gamma_P     = s * kappa * (g / delta)^2         s = 10^(-dB / 10)
    1/T1_eff    = 1/T1 + gamma_P
    1/Tphi      = 1/T2 - 1/(2 T1)                   intrinsic, held fixed
    1/T2_eff    = 1/(2 T1_eff) + 1/Tphi

g, delta and kappa are angular frequencies (2 pi times the cyclic values in Hz).

Run:  python benchmarks/t1_t2_decay.py
Exit status is 0 when every check passes, 1 otherwise. PNGs go to docs/images/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("python", "build"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import quantum_sim_noise  # noqa: F401
except ImportError:
    sys.exit(
        "quantum_sim_noise is not built.\n"
        "From the repo root, build it with the command at the top of src/bindings_noise.cpp:\n"
        "  g++ -O3 -shared -std=c++17 -fopenmp -fPIC -I/usr/include/eigen3 \\\n"
        "      $(python3 -m pybind11 --includes) src/Noise.cpp src/bindings_noise.cpp \\\n"
        "      -o build/quantum_sim_noise$(python3-config --extension-suffix)\n"
        f"Looked in {ROOT / 'build'} and on PYTHONPATH."
    )

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

import noise_models as nm

OUT_DIR = ROOT / "docs" / "images"

US = 1e-6
GHZ = 1e9

T1 = 100 * US
T2 = 80 * US
G_HZ = 100e6
DELTA_HZ = 1.0 * GHZ
KAPPA_HZ = 10e6

TOL_ABS = 1e-12         # populations and coherences, absolute
TOL_REL = 1e-12         # rates and times that are one formula away from the input
TOL_REL_DERIVED = 1e-9  # quantities rebuilt through a subtraction (Tphi, T2_eff)
TOL_SIM = 1e-11         # accumulated over repeated channel applications
ERR_FLOOR = 1e-17       # exact zeros are drawn here on the log error axes

C_BLUE, C_RED, C_TEAL, C_GOLD, C_GRAY = "#1f4e79", "#b5473a", "#2a9d8f", "#c98b16", "#6b6b6b"
FILTER_PALETTE = (C_BLUE, C_TEAL, C_GOLD, C_RED)

INTRINSIC = nm.T1T2Model(T1, T2)
TPHI = 1.0 / (1.0 / T2 - 0.5 / T1)


# ---------------------------------------------------------------------------
# Closed forms (independent of the module under test)
# ---------------------------------------------------------------------------

def exact_rate(g_hz, delta_hz, kappa_hz, atten_db=0.0):
    s = 10.0 ** (-np.asarray(atten_db, dtype=float) / 10.0)
    return s * nm.TWO_PI * kappa_hz * (np.asarray(g_hz) / np.asarray(delta_hz)) ** 2


def exact_combined(gamma_p):
    inv_t1 = 1.0 / T1 + gamma_p
    inv_tphi = 1.0 / T2 - 0.5 / T1
    return 1.0 / inv_t1, 1.0 / (0.5 * inv_t1 + inv_tphi)


def module_purcell(g_hz, delta_hz, atten_db=0.0):
    return nm.purcell_from_frequencies(g_hz, delta_hz, KAPPA_HZ, atten_db)


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, err, tol):
        err = float(err)
        self.rows.append((name, err, tol, bool(err <= tol)))  # NaN fails

    def all_passed(self):
        return all(r[3] for r in self.rows)

    def print_table(self):
        heading("Summary of checks")
        print(f"{'check':<52}{'max error':>12}{'tolerance':>12}  status")
        for name, err, tol, ok in self.rows:
            print(f"{name:<52}{err:>12.2e}{tol:>12.0e}  {'PASS' if ok else 'FAIL'}")
        print()
        print("ALL CHECKS PASSED" if self.all_passed() else "SOME CHECKS FAILED")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dcdcdc",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "mathtext.default": "regular",
    })


def sim_style(color):
    return dict(linestyle="none", marker="o", markersize=5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.3, zorder=3)


def legend_with_key(ax, handles, **kwargs):
    key = [
        Line2D([], [], color=C_GRAY, lw=1.8, label="analytic"),
        Line2D([], [], color=C_GRAY, lw=0, marker="o", markersize=5, markerfacecolor="white",
               markeredgewidth=1.3, label="quantum_sim_noise"),
    ]
    ax.legend(handles=list(handles) + key, **kwargs)


def log_ticks(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1: intrinsic T1 and T2 decay
# ---------------------------------------------------------------------------

def fig_intrinsic(checks):
    t = np.linspace(0.0, 3.0 * T1, 121)
    tf = np.linspace(0.0, 3.0 * T1, 400)
    sim = (nm.t1_experiment(INTRINSIC, t), nm.ramsey_experiment(INTRINSIC, t))
    ref = (np.exp(-t / T1), np.exp(-t / T2))
    err = [np.abs(s - r) for s, r in zip(sim, ref)]
    checks.add("T1 decay: P1(t) vs exp(-t/T1)", err[0].max(), TOL_ABS)
    checks.add("T2 coherence: |rho01| vs exp(-t/T2)", err[1].max(), TOL_ABS)

    panels = (
        (T1, C_BLUE, r"$P_1(t)$", r"$T_1$ relaxation from $|1\rangle$, $T_1$ = %g µs" % (T1 / US),
         r"$e^{-t/T_1}$", lambda x: np.exp(-x / T1), "T$_1$"),
        (T2, C_RED, r"$|\rho_{01}(t)|\,/\,|\rho_{01}(0)|$",
         r"Ramsey coherence from $|+\rangle$, $T_2$ = %g µs" % (T2 / US),
         r"$e^{-t/T_2}$", lambda x: np.exp(-x / T2), "T$_2$"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.15]}, layout="constrained")
    for col, (tau, color, ylabel, title, ref_label, f, tau_name) in enumerate(panels):
        top, bot = axes[0, col], axes[1, col]
        top.plot(tf / US, f(tf), color=color, label=ref_label)
        top.plot(t[::6] / US, sim[col][::6], label="quantum_sim_noise", **sim_style(color))
        top.plot([tau / US], [1 / np.e], marker="o", color=C_GRAY, markersize=3.5, zorder=4)
        top.annotate(f"t = {tau_name}, 1/e", xy=(tau / US, 1 / np.e),
                     xytext=(tau / US + 14, 1 / np.e + 0.2), fontsize=9, color="#555555",
                     arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
        top.set_title(title, loc="left")
        top.set_ylabel(ylabel)
        top.set_ylim(-0.02, 1.04)
        top.legend(loc="upper right")

        bot.semilogy(t / US, np.maximum(err[col], ERR_FLOOR), color=color, lw=1.2,
                     marker=".", markersize=4)
        bot.axhline(TOL_ABS, color="#777777", ls="--", lw=1.0)
        bot.set_ylim(ERR_FLOOR / 3, 1e-10)
        bot.text(0.99, 0.76, "pass threshold 1e-12", transform=bot.transAxes, ha="right",
                 fontsize=8, color="#666666")
        bot.text(0.99, 0.40, "exact zeros drawn at 1e-17", transform=bot.transAxes, ha="right",
                 fontsize=8, color="#666666")
        bot.set_ylabel("|sim - analytic|")
        bot.set_xlabel("Delay t (µs)")
        bot.set_xlim(0, 3 * T1 / US)
    return save(fig, "t1_t2_decay.png")


# ---------------------------------------------------------------------------
# Figure 2: filter attenuation sweep and detuning sweep
# ---------------------------------------------------------------------------

def fig_filter_detuning(checks):
    db = np.arange(0.0, 40.0 + 1e-9, 0.5)
    models = [module_purcell(G_HZ, DELTA_HZ, d) for d in db]
    sim_t1p = np.array([m.t1_limit() for m in models])
    sim_t1e = np.array([m.combined_with(INTRINSIC).t1() for m in models])
    gp = exact_rate(G_HZ, DELTA_HZ, KAPPA_HZ, db)
    ref_t1p, ref_t1e = 1.0 / gp, exact_combined(gp)[0]
    s_exact = 10.0 ** (-db / 10.0)
    checks.add("Filter factor s = 10^(-dB/10), 0 to 40 dB",
               rel([m.filter_suppression() for m in models], s_exact), TOL_REL)
    checks.add("Purcell T1 limit vs filter attenuation", rel(sim_t1p, ref_t1p), TOL_REL)
    checks.add("Combined T1_eff vs filter attenuation", rel(sim_t1e, ref_t1e), TOL_REL)

    deltas = np.geomspace(0.25 * GHZ, 3.0 * GHZ, 41)
    curves = []
    worst_rate_err, worst_slope_err = 0.0, 0.0
    for atten, color in ((0.0, C_BLUE), (20.0, C_RED)):
        sim = np.array([module_purcell(G_HZ, d, atten).rate() for d in deltas])
        ref = exact_rate(G_HZ, deltas, KAPPA_HZ, atten)
        worst_rate_err = max(worst_rate_err, rel(sim, ref))
        slope = np.polyfit(np.log(deltas), np.log(sim), 1)[0]
        worst_slope_err = max(worst_slope_err, abs(slope + 2.0))
        curves.append((atten, color, sim, ref))
    checks.add("Purcell rate vs detuning (0 and 20 dB)", worst_rate_err, TOL_REL)
    checks.add("Rate scaling exponent in delta (expect -2)", worst_slope_err, TOL_REL_DERIVED)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5), layout="constrained")

    ax1.semilogy(db, ref_t1p / US, color=C_BLUE)
    ax1.semilogy(db[::4], sim_t1p[::4] / US, **sim_style(C_BLUE))
    ax1.semilogy(db, ref_t1e / US, color=C_RED)
    ax1.semilogy(db[::4], sim_t1e[::4] / US, **sim_style(C_RED))
    ax1.axhline(T1 / US, color=C_GRAY, ls="--", lw=1.2)
    ax1.set_xlim(0, 40)
    ax1.set_xticks(range(0, 41, 5))
    ax1.set_xlabel("Filter attenuation at the qubit frequency (dB)")
    ax1.set_ylabel("Time (µs)")
    ax1.set_title(r"Purcell limit and effective $T_1$ vs filter", loc="left")
    legend_with_key(ax1, [
        Line2D([], [], color=C_BLUE, lw=1.8, label=r"Purcell limit $1/\gamma_P$"),
        Line2D([], [], color=C_RED, lw=1.8, label=r"$T_{1,\mathrm{eff}}$"),
        Line2D([], [], color=C_GRAY, lw=1.2, ls="--", label=r"intrinsic $T_1$ = %g µs" % (T1 / US)),
    ], loc="upper left")

    for atten, color, sim, ref in curves:
        ax2.loglog(deltas / GHZ, ref, color=color)
        ax2.loglog(deltas[::3] / GHZ, sim[::3], **sim_style(color))
    log_ticks(ax2, [0.25, 0.5, 1, 2, 3])
    ax2.set_xlabel(r"Qubit-resonator detuning $\Delta/2\pi$ (GHz)")
    ax2.set_ylabel(r"Purcell rate $\gamma_P$ (s$^{-1}$)")
    ax2.set_title(r"$\gamma_P \propto \Delta^{-2}$, $g/2\pi$ = %g MHz, $\kappa/2\pi$ = %g MHz"
                  % (G_HZ / 1e6, KAPPA_HZ / 1e6), loc="left")
    legend_with_key(ax2, [Line2D([], [], color=c, lw=1.8, label=f"{a:g} dB filter")
                          for a, c, _, _ in curves], loc="lower left")
    return save(fig, "purcell_filter_detuning.png")


# ---------------------------------------------------------------------------
# Figure 3: combined T1_eff and T2_eff against g/delta
# ---------------------------------------------------------------------------

def fig_combined(checks):
    ratios = np.geomspace(0.01, 0.40, 40)
    filters = (0.0, 10.0, 20.0, 30.0)
    err_t1 = err_t2 = err_tphi = err_bound = 0.0
    data = []
    for atten in filters:
        eff = [module_purcell(r * DELTA_HZ, DELTA_HZ, atten).combined_with(INTRINSIC) for r in ratios]
        sim_t1 = np.array([e.t1() for e in eff])
        sim_t2 = np.array([e.t2() for e in eff])
        sim_tphi = np.array([e.t_phi() for e in eff])
        gp = exact_rate(ratios * DELTA_HZ, DELTA_HZ, KAPPA_HZ, atten)
        ref_t1, ref_t2 = exact_combined(gp)
        err_t1 = max(err_t1, rel(sim_t1, ref_t1))
        err_t2 = max(err_t2, rel(sim_t2, ref_t2, ) if False else rel(sim_t2, ref_t2))
        err_tphi = max(err_tphi, float(np.max(np.abs(sim_tphi / TPHI - 1.0))))
        err_bound = max(err_bound, float(np.max(sim_t2 / (2.0 * sim_t1) - 1.0)), 0.0)
        data.append((atten, sim_t1, sim_t2, ref_t1, ref_t2))
    checks.add("Combined T1_eff vs g/delta (4 filters)", err_t1, TOL_REL)
    checks.add("Combined T2_eff vs g/delta (4 filters)", err_t2, TOL_REL_DERIVED)
    checks.add("Intrinsic Tphi preserved by combined_with", err_tphi, TOL_REL_DERIVED)
    checks.add("T2_eff <= 2 T1_eff", err_bound, TOL_REL_DERIVED)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5), layout="constrained")
    for (atten, sim_t1, sim_t2, ref_t1, ref_t2), color in zip(data, FILTER_PALETTE):
        for ax, sim, ref in ((ax1, sim_t1, ref_t1), (ax2, sim_t2, ref_t2)):
            ax.loglog(ratios, ref / US, color=color)
            ax.loglog(ratios[::3], sim[::3] / US, **sim_style(color))
    ax2.loglog(ratios, 2.0 * data[0][3] / US, color="#9a9a9a", ls=":", lw=1.4)

    for ax in (ax1, ax2):
        log_ticks(ax, [0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
        ax.set_xlabel(r"Coupling ratio $g/\Delta$")
        ax.set_ylabel("Time (µs)")
    ax1.axhline(T1 / US, color=C_GRAY, ls="--", lw=1.2)
    ax2.axhline(T2 / US, color=C_GRAY, ls="--", lw=1.2)
    ax1.set_title(r"Effective $T_1$ with Purcell decay", loc="left")
    ax2.set_title(r"Effective $T_2$, intrinsic $T_\phi$ held fixed", loc="left")

    filt = [Line2D([], [], color=c, lw=1.8, label=f"{a:g} dB filter")
            for a, c in zip(filters, FILTER_PALETTE)]
    legend_with_key(ax1, filt + [Line2D([], [], color=C_GRAY, lw=1.2, ls="--",
                                        label=r"intrinsic $T_1$")], loc="lower left")
    legend_with_key(ax2, filt + [
        Line2D([], [], color=C_GRAY, lw=1.2, ls="--", label=r"intrinsic $T_2$"),
        Line2D([], [], color="#9a9a9a", lw=1.4, ls=":", label=r"$2\,T_{1,\mathrm{eff}}$ bound, 0 dB"),
    ], loc="lower left")
    return save(fig, "combined_t1_t2_eff.png")


# ---------------------------------------------------------------------------
# Tables and simulator-level checks
# ---------------------------------------------------------------------------

def heading(text):
    print()
    print(text)
    print("-" * len(text))


def print_parameters():
    ratio = G_HZ / DELTA_HZ
    gp = float(exact_rate(G_HZ, DELTA_HZ, KAPPA_HZ))
    t1e, t2e = exact_combined(gp)
    heading("Parameters")
    print(f"  T1 = {T1 / US:g} us, T2 = {T2 / US:g} us, Tphi = {TPHI / US:.4f} us")
    print(f"  g/2pi = {G_HZ / 1e6:g} MHz, delta/2pi = {DELTA_HZ / GHZ:g} GHz, "
          f"kappa/2pi = {KAPPA_HZ / 1e6:g} MHz, g/delta = {ratio:g}")
    print(f"  gamma_P = {gp:.6e} 1/s  (Purcell limit {1e6 / gp:.4f} us)")
    print(f"  T1_eff = {t1e / US:.4f} us, T2_eff = {t2e / US:.4f} us")


def print_intrinsic_table():
    heading("Intrinsic decay: simulation vs analytic")
    print(f"{'t [us]':>8}{'P1 sim':>18}{'P1 exact':>18}{'|err|':>11}"
          f"{'C sim':>18}{'C exact':>18}{'|err|':>11}")
    for t_us in (0, 25, 50, 100, 200):
        t = t_us * US
        p_s = nm.t1_experiment(INTRINSIC, [t])[0]
        c_s = nm.ramsey_experiment(INTRINSIC, [t])[0]
        p_e, c_e = np.exp(-t / T1), np.exp(-t / T2)
        print(f"{t_us:>8.0f}{p_s:>18.12f}{p_e:>18.12f}{abs(p_s - p_e):>11.2e}"
              f"{c_s:>18.12f}{c_e:>18.12f}{abs(c_s - c_e):>11.2e}")


def print_filter_table():
    heading("Purcell rate vs filter attenuation (g/delta = 0.1)")
    print(f"{'dB':>5}{'s':>11}{'gamma_P sim [1/s]':>20}{'gamma_P exact [1/s]':>21}"
          f"{'rel err':>11}{'T1 limit [us]':>16}{'T1_eff [us]':>14}")
    for atten in (0, 10, 20, 30, 40):
        m = module_purcell(G_HZ, DELTA_HZ, atten)
        ref = float(exact_rate(G_HZ, DELTA_HZ, KAPPA_HZ, atten))
        t1e = m.combined_with(INTRINSIC).t1()
        print(f"{atten:>5}{m.filter_suppression():>11.5f}{m.rate():>20.6e}{ref:>21.6e}"
              f"{abs(m.rate() - ref) / ref:>11.2e}{m.t1_limit() / US:>16.4f}{t1e / US:>14.4f}")


def print_combined_table():
    heading("Combined T1/T2 vs g/delta (no filter)")
    print(f"{'g/delta':>8}{'gamma_P [1/s]':>15}{'T1_eff sim':>13}{'exact':>11}"
          f"{'T2_eff sim':>13}{'exact':>11}{'T2/(2 T1_eff)':>15}{'Tphi_eff/Tphi':>15}")
    for r in (0.02, 0.05, 0.10, 0.20, 0.40):
        eff = module_purcell(r * DELTA_HZ, DELTA_HZ).combined_with(INTRINSIC)
        gp = float(exact_rate(r * DELTA_HZ, DELTA_HZ, KAPPA_HZ))
        t1e, t2e = exact_combined(gp)
        print(f"{r:>8.2f}{gp:>15.4e}{eff.t1() / US:>13.5f}{t1e / US:>11.5f}"
              f"{eff.t2() / US:>13.5f}{t2e / US:>11.5f}"
              f"{eff.t2() / (2 * eff.t1()):>15.6f}{eff.t_phi() / TPHI:>15.9f}")
    print("  (times in us)")


def run_simulator_checks(checks, n_steps=16):
    heading("End-to-end: DensityMatrixSimulator with NoiseModel + Purcell")
    print(f"{'dB':>5}{'t final [us]':>14}{'P1 sim':>16}{'P1 exact':>16}"
          f"{'C sim':>16}{'C exact':>16}{'max |err|':>12}")
    worst = 0.0
    for atten in (0.0, 20.0):
        pm = module_purcell(G_HZ, DELTA_HZ, atten)
        noise = nm.NoiseModel.uniform(1, T1, T2, purcell=pm)
        t1e, t2e = exact_combined(float(exact_rate(G_HZ, DELTA_HZ, KAPPA_HZ, atten)))
        dt = 2.0 * t1e / n_steps
        pop = nm.DensityMatrixSimulator(1, noise, initial=[0.0, 1.0])
        ram = nm.DensityMatrixSimulator(1, noise, initial=np.array([1.0, 1.0]) / np.sqrt(2.0))
        err = 0.0
        for k in range(1, n_steps + 1):
            pop.idle(dt)
            ram.idle(dt)
            p_s = pop.populations()[1]
            c_s = 2.0 * abs(ram.rho[0, 1])
            err = max(err, abs(p_s - np.exp(-k * dt / t1e)), abs(c_s - np.exp(-k * dt / t2e)))
        worst = max(worst, err)
        print(f"{atten:>5.0f}{n_steps * dt / US:>14.4f}{p_s:>16.12f}{np.exp(-n_steps * dt / t1e):>16.12f}"
              f"{c_s:>16.12f}{np.exp(-n_steps * dt / t2e):>16.12f}{err:>12.2e}")
    checks.add("Simulator idle decay vs T1_eff, T2_eff", worst, TOL_SIM)

    tp_ok = all(
        module_purcell(G_HZ, DELTA_HZ, a).combined_with(INTRINSIC).channel(t * US, 0).is_trace_preserving()
        for a in (0.0, 20.0) for t in (0.0, 0.1, 5.0, 300.0)
    ) and all(INTRINSIC.channel(t * US, 0).is_trace_preserving() for t in (0.0, 1.0, 100.0, 1000.0))
    checks.add("Kraus trace preservation (T1/T2, Purcell)", 0.0 if tp_ok else 1.0, 0.0)


def main():
    set_style()
    checks = Checks()
    print_parameters()
    paths = [fig_intrinsic(checks), fig_filter_detuning(checks), fig_combined(checks)]
    print_intrinsic_table()
    print_filter_table()
    print_combined_table()
    run_simulator_checks(checks)
    checks.print_table()
    print()
    for p in paths:
        print(f"saved {p.relative_to(ROOT)}")
    return 0 if checks.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main()) 