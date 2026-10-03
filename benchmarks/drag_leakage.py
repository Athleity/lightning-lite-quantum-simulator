# benchmarks/drag_leakage.py
"""
DRAG pulse benchmarks: leakage, envelopes, convergence and coherent (Stark) error.

Transmon: EC = 300 MHz, EJ/EC = 50 (alpha ~ -345 MHz, f01 ~ 5.68 GHz).
Plots are written to docs/images/. Run from anywhere:
    python3 benchmarks/drag_leakage.py

Rotation convention: every pulse is a theta = pi rotation on resonance with omega_01
(no Stark-shift correction). Omega/|alpha| below means (theta/T) / |alpha_angular|, the
mean Rabi rate over the pulse, so square, Gaussian and DRAG are compared at equal area
and equal duration. Gaussian and DRAG use sigma = T/4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "python", ROOT / "build"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pulse as P  # noqa: E402

IMG = ROOT / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)

PI = np.pi
EC = 300 * P.MHZ
EJ = 50 * EC
T0 = 8 * P.NS          # reference gate time
SIGMA0 = 2 * P.NS      # reference Gaussian width
N_LEVELS = 4

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
C_SQ, C_GA, C_DR, C_0, C_1 = "tab:gray", "tab:blue", "tab:red", "tab:blue", "tab:red"


def banner(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


def save(fig, name):
    path = IMG / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def loglog_slope(x, y):
    m = np.isfinite(y) & (y > 0)
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]


# ---------------------------------------------------------------------------------------
tr = P.Transmon(EJ, EC, N_LEVELS)
alpha = tr.anharmonicity_angular()
A = abs(alpha)
wd = tr.omega_01()
banner("Transmon")
print(f"EJ/EC = {tr.ej_ec_ratio:.1f}, n_levels = {tr.n_levels}")
print(f"f01 = {tr.f01_ghz:.4f} GHz, alpha = {tr.alpha_mhz:.2f} MHz")
print(f"gate: pi pulse, T = {T0 / P.NS:.1f} ns, sigma = {SIGMA0 / P.NS:.1f} ns")

# ---- 1. beta sweep ------------------------------------------------------------------------
banner("1. Leakage vs DRAG beta")
betas = np.linspace(-1.0, 3.0, 81)
L1 = P.beta_sweep(tr, PI, T0, SIGMA0, betas, initial_level=1)
L0 = P.beta_sweep(tr, PI, T0, SIGMA0, betas, initial_level=0)
i0, i1 = int(np.argmin(L0)), int(np.argmin(L1))
ib0, ib1 = int(np.argmin(abs(betas))), int(np.argmin(abs(betas - 1.0)))
print(f"{'beta':>7} {'leak(|0>)':>14} {'leak(|1>)':>14}")
for k in range(0, betas.size, 5):
    print(f"{betas[k]:7.2f} {L0[k]:14.4e} {L1[k]:14.4e}")
print(f"min leak(|1>) = {L1[i1]:.3e} at beta = {betas[i1]:.2f}")
print(f"min leak(|0>) = {L0[i0]:.3e} at beta = {betas[i0]:.2f}")
print(f"suppression leak(beta=0)/leak(beta=1): |1> {L1[ib0] / L1[ib1]:.1f}x, "
      f"|0> {L0[ib0] / L0[ib1]:.1f}x")
if not (0.5 <= betas[i1] <= 1.5):
    print("WARNING: leak(|1>) minimum is not near beta = 1")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(betas, L0, color=C_0, label=r"start in $|0\rangle$")
ax.semilogy(betas, L1, color=C_1, label=r"start in $|1\rangle$")
ax.axvline(1.0, color="k", ls=":", lw=1)
ax.set_xlabel(r"DRAG coefficient $\beta$")
ax.set_ylabel(r"leakage out of $\{|0\rangle,|1\rangle\}$")
ax.set_title(rf"DRAG $\pi$ pulse, $T$ = {T0 / P.NS:.0f} ns, $\sigma$ = {SIGMA0 / P.NS:.0f} ns, "
             rf"$\alpha/2\pi$ = {tr.alpha_mhz:.0f} MHz", fontsize=9)
ax.legend()
save(fig, "drag_beta_sweep.png")

# ---- 2. envelopes ----------------------------------------------------------------------------
banner("2. Time-domain envelopes")
t = np.linspace(0.0, T0, 801)
shapes = {
    "square": P.pulse_from_square(PI, T0, wd),
    "Gaussian": P.pulse_from_gaussian(PI, T0, SIGMA0, wd),
    "DRAG ($\\beta$=1)": P.pulse_from_drag(PI, T0, SIGMA0, 1.0, wd, tr),
}
samples = {k: np.asarray(p.envelope.sample(t)) for k, p in shapes.items()}
unit = 1.0 / (P.TWO_PI * P.MHZ)  # rad/s -> MHz
print(f"{'shape':<18} {'peak Re (MHz)':>14} {'peak Im (MHz)':>14} {'area Re / pi':>14}")
for k, s in samples.items():
    area = np.trapezoid(s.real, t) / PI
    print(f"{k:<18} {np.max(np.abs(s.real)) * unit:14.3f} {np.max(np.abs(s.imag)) * unit:14.3f} "
          f"{area:14.6f}")

fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
for (k, s), c in zip(samples.items(), (C_SQ, C_GA, C_DR)):
    ax_i.plot(t / P.NS, s.real * unit, color=c, label=k, lw=1.8)
    ax_q.plot(t / P.NS, s.imag * unit, color=c, label=k, lw=1.8)
ax_i.set_ylabel(r"$\Omega_I/2\pi$ (MHz)")
ax_q.set_ylabel(r"$\Omega_Q/2\pi$ (MHz)")
ax_q.set_xlabel("t (ns)")
ax_i.legend()
ax_i.set_title(r"$\pi$ pulse envelopes, equal area", fontsize=10)
save(fig, "drag_vs_gaussian.png")

# ---- 3. leakage vs Omega/|alpha| -----------------------------------------------------------------
banner("3. Leakage vs Omega/|alpha| (Omega = pi/T)")
ratios = np.logspace(np.log10(0.03), np.log10(0.5), 16)
leak = {k: np.empty(ratios.size) for k in ("square", "Gaussian", "DRAG")}
print(f"{'Om/|a|':>8} {'T (ns)':>9} {'square':>13} {'Gaussian':>13} {'DRAG':>13}")
for k, r in enumerate(ratios):
    T = PI / (r * A)
    sg = T / 4
    leak["square"][k] = tr.leakage(P.pulse_from_square(PI, T, wd), 1)
    leak["Gaussian"][k] = tr.leakage(P.pulse_from_gaussian(PI, T, sg, wd), 1)
    leak["DRAG"][k] = tr.leakage(P.pulse_from_drag(PI, T, sg, 1.0, wd, tr), 1)
    print(f"{r:8.4f} {T / P.NS:9.2f} {leak['square'][k]:13.4e} {leak['Gaussian'][k]:13.4e} "
          f"{leak['DRAG'][k]:13.4e}")
weak = ratios <= 0.15
print("fitted log-log slope for Om/|alpha| <= 0.15 (expect square ~2, DRAG ~4 if "
      "perturbative):")
for k in leak:
    print(f"  {k:<9} slope = {loglog_slope(ratios[weak], leak[k][weak]):6.2f}")

fig, ax = plt.subplots(figsize=(6, 4.2))
for (k, y), c, mk in zip(leak.items(), (C_SQ, C_GA, C_DR), ("s", "o", "^")):
    ax.loglog(ratios, y, marker=mk, ms=4, color=c, label=k)
x0 = ratios[0]
for n, ls in ((2, "--"), (4, ":")):
    ax.loglog(ratios, leak["square" if n == 2 else "DRAG"][0] * (ratios / x0) ** n,
              color="k", ls=ls, lw=0.9, label=rf"$\propto(\Omega/\alpha)^{n}$")
ax.set_xlabel(r"$\Omega/|\alpha|$,  $\Omega=\pi/T$")
ax.set_ylabel(r"leakage from $|1\rangle$")
ax.legend(fontsize=8)
save(fig, "leakage_vs_omega.png")

# ---- 4. n_levels convergence -----------------------------------------------------------------------
banner("4. Leakage vs n_levels")
nl = np.arange(3, 7)
Ldrag = P.leak_vs_nlevels(EJ, EC, PI, T0, SIGMA0, 1.0, nl)
Lgaus = P.leak_vs_nlevels(EJ, EC, PI, T0, SIGMA0, 0.0, nl)
print(f"{'n_levels':>8} {'Gaussian':>14} {'DRAG':>14} {'dG/G':>10} {'dD/D':>10}")
for k, n in enumerate(nl):
    dg = abs(Lgaus[k] / Lgaus[k - 1] - 1) if k else float("nan")
    dd = abs(Ldrag[k] / Ldrag[k - 1] - 1) if k else float("nan")
    print(f"{n:8d} {Lgaus[k]:14.5e} {Ldrag[k]:14.5e} {dg:10.2e} {dd:10.2e}")

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.semilogy(nl, Lgaus, "o-", color=C_GA, label=r"Gaussian ($\beta$=0)")
ax.semilogy(nl, Ldrag, "^-", color=C_DR, label=r"DRAG ($\beta$=1)")
ax.set_xticks(nl)
ax.set_xlabel("retained transmon levels")
ax.set_ylabel(r"leakage out of $\{|0\rangle,|1\rangle\}$")
ax.legend()
save(fig, "nlevels_convergence.png")

# ---- 5. coherent error vs duration --------------------------------------------------------------------
banner("5. Coherent error (Stark) vs duration, DRAG pi pulse, drive at omega_01")
Ts = np.logspace(np.log10(3.0), np.log10(60.0), 18) * P.NS
one_minus_p = np.empty(Ts.size)
leak0 = np.empty(Ts.size)
p00 = np.empty(Ts.size)
print(f"{'T (ns)':>8} {'1-P(1|0)':>13} {'leak(|0>)':>13} {'P(0|0)':>13}")
for k, T in enumerate(Ts):
    U = np.asarray(tr.propagator(P.pulse_from_drag(PI, T, T / 4, 1.0, wd, tr)))
    pop = np.abs(U[:, 0]) ** 2
    one_minus_p[k] = 1.0 - pop[1]
    leak0[k] = pop[2:].sum()
    p00[k] = pop[0]
    print(f"{T / P.NS:8.2f} {one_minus_p[k]:13.4e} {leak0[k]:13.4e} {p00[k]:13.4e}")
kmin = int(np.argmin(one_minus_p))
print(f"minimum 1-P(1|0) = {one_minus_p[kmin]:.3e} at T = {Ts[kmin] / P.NS:.1f} ns")

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(Ts / P.NS, one_minus_p, "o-", color="k", ms=4, label=r"$1-P(1|0)$")
ax.loglog(Ts / P.NS, leak0, "^-", color=C_DR, ms=4, label=r"leakage to $\geq|2\rangle$")
ax.loglog(Ts / P.NS, p00, "s-", color=C_GA, ms=4,
          label=r"$P(0|0)$ (incomplete rotation, Stark-shift detuning)")
ax.set_xlabel("pulse duration (ns)")
ax.set_ylabel("error")
ax.set_title(r"DRAG $\beta$=1, $\sigma=T/4$, no Stark correction", fontsize=9)
ax.legend(fontsize=8)
save(fig, "stark_shift.png")

# ---- done -------------------------------------------------------------------------------------------------
checks = [L0, L1, *leak.values(), Ldrag, Lgaus, one_minus_p, leak0, p00]
if not all(np.all(np.isfinite(a)) for a in checks):
    print("\nFAILED: non-finite values encountered")
    sys.exit(1)
banner("All plots written to docs/images/")
sys.exit(0)