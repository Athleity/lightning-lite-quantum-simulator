"""Quantum reservoir computing benchmarks (Tier 4).

Produces every QRC figure used in the interview material:

  docs/images/qrc_mc_vs_qubits.png   MC_total vs N for V = 2 and V = 4
  docs/images/qrc_mc_heatmap.png     MC_total over the (N, V) grid
  docs/images/qrc_mg_prediction.png  Mackey-Glass 1-step prediction + NMSE bars
  docs/images/qrc_mg_horizon.png     NMSE vs prediction horizon (reservoir, delay-ridge, persistence)
  docs/images/qrc_features.png       MC_k for the best (N, V) configuration

All sweeps use <Z_i> readout, i.i.d. U[0, 1] input for memory capacity and
Mackey-Glass (tau = 17) for prediction. The Mackey-Glass section also compares against a
classical delay-embedded ridge regression (numpy only) on the identical train/test split.
Observed result (N=4, V=4, Z readout, K=10, ridge=1e-8): delay-ridge wins at short horizon,
the reservoir overtakes it near h = 7 and wins at h = 8..10. This is a post-hoc observation
for one seed and untuned baselines, not a general claim.

Run:
  python3 benchmarks/qrc_mackey_glass.py              full sweep, N = 1..6 (15-20 min)
  QRC_QUICK=1 python3 benchmarks/qrc_mackey_glass.py  trimmed sweep, N = 1..4
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import qrc  # noqa: E402

QUICK = os.environ.get("QRC_QUICK", "0") == "1"

IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

SEEDS = (1,)
N_RANGE = [1, 2, 3, 4] if QUICK else list(range(1, 7))
V_RANGE = [1, 2, 4] if QUICK else [1, 2, 3, 4, 6, 8]
MAX_DELAY = 20
N_TRAIN, N_TEST = 1000, 500

MG_SAMPLES, MG_WASHOUT, MG_TRAIN = 3000, 200, 1500
MG_FEATURES = 4 * 4   # reservoir readout width: N=4 qubits x V=4 virtual nodes, Z only
HORIZONS = list(range(1, 11))
LONG_H_MIN = 8        # "long horizon" window for the delay-ridge check: h = 8..10

BLUE, RED, GREY = "#3b6ea5", "#d1495b", "#999999"

_checks = []


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def mc_total(n, v, seed):
    return qrc.run_memory_capacity(n_qubits=n, virtual_nodes=v, max_delay=MAX_DELAY,
                                   n_train=N_TRAIN, n_test=N_TEST, seed=seed).total


def mc_stats(n, v):
    vals = np.array([mc_total(n, v, s) for s in SEEDS])
    return vals.mean(), vals.std()


def save(fig, name):
    path = IMAGES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def delay_ridge_nmse(x, horizon, K, pr=None, ridge=1e-8):
    """Delay-embedded ridge baseline, test NMSE on the reservoir's split.

    Row s: [x_s, x_{s-1}, ..., x_{s-K+1}] -> x_{s+h}; for h=1 this is [x_{t-1}..x_{t-K}] -> x_t.
    Same rows, train/test split, centred bias and relative ridge as LinearReadout.
    """
    if MG_WASHOUT < K - 1:
        raise ValueError("MG_WASHOUT must be >= K - 1")
    n_test = len(x) - horizon - MG_WASHOUT - MG_TRAIN
    s = np.arange(MG_WASHOUT, MG_WASHOUT + MG_TRAIN + n_test)
    F = np.stack([x[s - j] for j in range(K)], axis=1)
    y = x[s + horizon]
    Ftr, Fte, ytr, yte = F[:MG_TRAIN], F[MG_TRAIN:], y[:MG_TRAIN], y[MG_TRAIN:]
    if pr is not None:
        assert np.allclose(yte, np.asarray(pr.target_test)), "split differs from reservoir"
    mu, my = Ftr.mean(0), ytr.mean()
    Fc = Ftr - mu
    G = Fc.T @ Fc
    G += np.eye(K) * ridge * np.trace(G) / K
    w = np.linalg.solve(G, Fc.T @ (ytr - my))
    b = my - mu @ w
    return float(np.sum((Fte @ w + b - yte) ** 2) / np.sum((yte - yte.mean()) ** 2))


# ---------------------------------------------------------------------------
# 1 + 2. Memory capacity sweeps
# ---------------------------------------------------------------------------

def sweep_memory():
    print(f"\n[1/5] MC_total over the (N, V) grid, seeds {SEEDS}"
          f"{'  [QUICK]' if QUICK else ''}")
    mean = np.zeros((len(N_RANGE), len(V_RANGE)))
    std = np.zeros_like(mean)
    for i, n in enumerate(N_RANGE):
        for j, v in enumerate(V_RANGE):
            mean[i, j], std[i, j] = mc_stats(n, v)
        print(f"  N={n} done", flush=True)

    print("\n  MC_total (mean over seeds)")
    print("  N \\ V " + "".join(f"{v:>8d}" for v in V_RANGE))
    for i, n in enumerate(N_RANGE):
        print(f"  {n:>5d} " + "".join(f"{mean[i, j]:>8.2f}" for j in range(len(V_RANGE))))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for v, colour, mark in ((2, GREY, "s"), (4, BLUE, "o")):
        j = V_RANGE.index(v)
        ax.errorbar(N_RANGE, mean[:, j], yerr=std[:, j], color=colour, marker=mark,
                    capsize=3, lw=1.6, label=f"V = {v}")
        ax.plot(N_RANGE, [n * v for n in N_RANGE], color=colour, ls=":", lw=1.0)
    ax.set_xlabel("qubits N")
    ax.set_ylabel(r"$MC_{total}$")
    ax.set_title("Memory capacity vs reservoir size (dotted: feature bound N·V)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save(fig, "qrc_mc_vs_qubits.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    im = ax.imshow(mean, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(V_RANGE)), V_RANGE)
    ax.set_yticks(range(len(N_RANGE)), N_RANGE)
    ax.set_xlabel("virtual nodes V")
    ax.set_ylabel("qubits N")
    ax.set_title(r"$MC_{total}$ over (N, V)")
    for i in range(len(N_RANGE)):
        for j in range(len(V_RANGE)):
            ax.text(j, i, f"{mean[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if mean[i, j] < 0.6 * mean.max() else "black")
    fig.colorbar(im, ax=ax, label=r"$MC_{total}$")
    save(fig, "qrc_mc_heatmap.png")

    return mean


# ---------------------------------------------------------------------------
# 3. Mackey-Glass 1-step prediction
# ---------------------------------------------------------------------------

def mackey_glass_prediction():
    print("\n[2/5] Mackey-Glass 1-step prediction (N=4, V=4)")
    pr = qrc.run_mackey_glass(n_qubits=4, n_samples=MG_SAMPLES, horizon=1,
                              washout=MG_WASHOUT, n_train=MG_TRAIN, virtual_nodes=4)
    gain = pr.nmse_persistence / pr.nmse_test
    print(f"  NMSE train        {pr.nmse_train:.3e}")
    print(f"  NMSE test         {pr.nmse_test:.3e}")
    print(f"  NMSE persistence  {pr.nmse_persistence:.3e}")
    print(f"  improvement       {gain:.1f}x  (vs persistence only; see delay-ridge below)")
    x = np.asarray(qrc.mackey_glass(MG_SAMPLES), dtype=float)
    dr10 = delay_ridge_nmse(x, 1, 10, pr)
    drK = delay_ridge_nmse(x, 1, MG_FEATURES, pr)
    print(f"\n  {'method':<24}{'NMSE test':>12}")
    print(f"  {'persistence':<24}{pr.nmse_persistence:>12.3e}")
    print(f"  {'delay-ridge (K=10)':<24}{dr10:>12.3e}")
    print(f"  {f'delay-ridge (K={MG_FEATURES})':<24}{drK:>12.3e}   (K = reservoir feature count)")
    print(f"  {'reservoir':<24}{pr.nmse_test:>12.3e}")
    qrc.plot_mackey_glass_prediction(pr, IMAGES / "qrc_mg_prediction.png")
    print("  saved docs/images/qrc_mg_prediction.png")
    return pr, gain, dr10, drK


# ---------------------------------------------------------------------------
# 4. NMSE vs horizon
# ---------------------------------------------------------------------------

def horizon_sweep():
    print("\n[3/5] NMSE vs horizon (N=4, V=4)")
    res_nmse, base_nmse, dr_nmse = [], [], []
    x = np.asarray(qrc.mackey_glass(MG_SAMPLES), dtype=float)
    print("  h    reservoir     delay-ridge   persistence   ridge/res")
    for h in HORIZONS:
        pr = qrc.run_mackey_glass(n_qubits=4, n_samples=MG_SAMPLES, horizon=h,
                                  washout=MG_WASHOUT, n_train=MG_TRAIN, virtual_nodes=4)
        res_nmse.append(pr.nmse_test)
        base_nmse.append(pr.nmse_persistence)
        dr = delay_ridge_nmse(x, h, 10, pr)
        dr_nmse.append(dr)
        print(f"  {h:<4d} {pr.nmse_test:<13.3e} {dr:<13.3e} {pr.nmse_persistence:<13.3e} "
              f"{dr / pr.nmse_test:.2f}x")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(HORIZONS, res_nmse, "o-", color=BLUE, lw=1.6, label="reservoir (N=4, V=4)")
    ax.semilogy(HORIZONS, base_nmse, "s--", color=GREY, lw=1.4, label="persistence")
    ax.semilogy(HORIZONS, dr_nmse, "^-.", color=RED, lw=1.4, label="delay-ridge (K=10)")
    ax.set_xlabel("prediction horizon h")
    ax.set_ylabel("test NMSE")
    ax.set_title("Mackey-Glass: error growth with horizon")
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, "qrc_mg_horizon.png")
    return np.array(res_nmse), np.array(base_nmse), np.array(dr_nmse)


# ---------------------------------------------------------------------------
# 5. MC_k for the best configuration
# ---------------------------------------------------------------------------

def best_config_profile(mean):
    i, j = np.unravel_index(np.argmax(mean), mean.shape)
    n, v = N_RANGE[i], V_RANGE[j]
    print(f"\n[4/5] Best configuration: N={n}, V={v}, MC_total={mean[i, j]:.2f}")
    r = qrc.run_memory_capacity(n_qubits=n, virtual_nodes=v, max_delay=MAX_DELAY,
                                n_train=N_TRAIN, n_test=N_TEST, seed=SEEDS[0])
    mc = np.asarray(r.mc)
    print("  k    MC_k")
    for k in range(0, MAX_DELAY + 1, 2):
        print(f"  {k:<4d} {mc[k]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(mc.size), mc, color=BLUE, width=0.8)
    ax.set_xlabel("delay k")
    ax.set_ylabel(r"$MC_k$")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Best config N={n}, V={v}: total {r.total:.2f} (bound {r.n_features})")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "qrc_features.png")
    return n, v, r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Quantum reservoir computing benchmarks" + ("  [QUICK]" if QUICK else ""))
    print("=" * 64)

    mean = sweep_memory()
    pr, gain, dr10, drK = mackey_glass_prediction()
    res_nmse, base_nmse, dr_nmse = horizon_sweep()
    n_best, v_best, best = best_config_profile(mean)

    print("\n[5/5] Checks")
    check("MC_0 ~ 1 for best config", best.mc[0] > 0.99, f"{best.mc[0]:.4f}")
    check("MC_total <= n_features (+0.5 finite-sample slack)",
          best.total <= best.n_features + 0.5, f"{best.total:.2f} <= {best.n_features}")
    check("MC_total grows from N=1 to N=4 (V=4)",
          mean[N_RANGE.index(4), V_RANGE.index(4)] > mean[N_RANGE.index(1), V_RANGE.index(4)])
    check("MC_total grows from V=1 to V=4 (N=4)",
          mean[N_RANGE.index(4), V_RANGE.index(4)] > mean[N_RANGE.index(4), V_RANGE.index(1)])
    check("1-step reservoir NMSE beats persistence by > 10x", gain > 10, f"{gain:.1f}x")
    check("reservoir beats persistence at every horizon", np.all(res_nmse < base_nmse))

    wins = res_nmse < dr_nmse
    h_cross = next((h for h, w in zip(HORIZONS, wins) if w), None)
    print(f"  [INFO] delay-ridge beats reservoir at 1-step by {pr.nmse_test / dr10:.0f}x "
          f"(K=10), {pr.nmse_test / drK:.0f}x (K={MG_FEATURES})")
    check(f"reservoir beats delay-ridge at long horizon (h = {LONG_H_MIN}..{HORIZONS[-1]}, K=10)",
          bool(np.all(wins[HORIZONS.index(LONG_H_MIN):])), f"first win at h={h_cross}")

    check("no overfitting: test NMSE < 3x train NMSE", pr.nmse_test < 3 * pr.nmse_train)
    check("all five figures written",
          all((IMAGES / f).exists() for f in (
              "qrc_mc_vs_qubits.png", "qrc_mc_heatmap.png", "qrc_mg_prediction.png",
              "qrc_mg_horizon.png", "qrc_features.png")))

    passed = sum(_checks)
    print(f"\n{passed}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())