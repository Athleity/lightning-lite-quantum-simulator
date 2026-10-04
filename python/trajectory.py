"""High-level Python interface to the quantum trajectory solver (Tier 3).

Wraps build/quantum_sim_trajectory (src/bindings_trajectory.cpp, src/TrajectorySolver.h).

    from trajectory import make_qubit, plus, decay_curve, variance_reduction, Unraveling
    s = make_qubit(t1=2.0, tphi=1.5)                    # amplitude damping + pure dephasing
    c = decay_curve(s, plus(), "X", times, rate=1/s_t2) # <X>(t) vs exp(-t/T2)
    v = variance_reduction(make_dephasing(1.0), plus(), 0.75, "X", eps=0.005)

Unravelings of the same Lindblad equation (all give the same ensemble average):
  STANDARD   quantum jumps; a Pauli channel applies P at rate gamma
  PROJECTOR  Pauli-like channels rewritten with jump operator sqrt(g)(1 - P); other channels
             fall back to STANDARD
  ANALOG     homodyne diffusion, no jumps
The variance reduction depends on the observable and state, not just the solver; for dephasing,
|+> and <X> it is 1 + exp(4 gamma t) (analytic_dephasing_ratio).

Basis: |0> is the ground state, sigma_- = |0><1|, so decay takes |1> to |0>.
Run as a script to regenerate docs/images/trajectory_*.png.
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
IMAGES_DIR = ROOT / "docs" / "images"

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import quantum_sim_trajectory as _traj
except ImportError as exc:
    raise ImportError(
        f"quantum_sim_trajectory not found in {BUILD_DIR}. Build it with the command "
        "at the top of src/bindings_trajectory.cpp."
    ) from exc

Unraveling = _traj.Unraveling
TrajectorySolver = _traj.TrajectorySolver
Stats = _traj.Stats

ALL_KINDS = (Unraveling.STANDARD, Unraveling.PROJECTOR, Unraveling.ANALOG)

__all__ = [
    "Unraveling", "TrajectorySolver", "Stats", "ALL_KINDS",
    "SIGMA_X", "SIGMA_Y", "SIGMA_Z", "SIGMA_MINUS", "OBSERVABLES", "pauli_string",
    "ground", "excited", "plus",
    "make_qubit", "make_dephasing",
    "estimate", "decay_curve",
    "trajectories_needed", "pilot_variance", "variance_reduction",
    "analytic_dephasing_ratio", "dephasing_variance_ratio",
    "plot_decay", "plot_variance_reduction",
]


# ---------------------------------------------------------------------------
# Operators and states
# ---------------------------------------------------------------------------

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|

OBSERVABLES = {
    "X": SIGMA_X,
    "Y": SIGMA_Y,
    "Z": SIGMA_Z,
    "P0": np.array([[1, 0], [0, 0]], dtype=complex),
    "P1": np.array([[0, 0], [0, 1]], dtype=complex),
}


def pauli_string(label):
    """Tensor product of single-qubit Paulis, e.g. 'XZ' = X (x) Z, qubit 0 on the right."""
    paulis = {"I": np.eye(2, dtype=complex), "X": SIGMA_X, "Y": SIGMA_Y, "Z": SIGMA_Z}
    out = np.eye(1, dtype=complex)
    for ch in label.upper():
        if ch not in paulis:
            raise ValueError(f"unknown Pauli '{ch}' in '{label}'")
        out = np.kron(out, paulis[ch])
    return out


def ground():
    return np.array([1, 0], dtype=complex)


def excited():
    return np.array([0, 1], dtype=complex)


def plus():
    return np.array([1, 1], dtype=complex) / math.sqrt(2.0)


def _kind(k):
    if isinstance(k, str):
        try:
            return Unraveling.__members__[k.upper()]
        except KeyError:
            raise ValueError(f"unknown unraveling '{k}', expected STANDARD, PROJECTOR or ANALOG")
    return k


def _observable(obs):
    if isinstance(obs, str):
        try:
            return OBSERVABLES[obs.upper()]
        except KeyError:
            raise ValueError(f"unknown observable '{obs}', expected one of {list(OBSERVABLES)}")
    if callable(obs):
        return obs
    return np.asarray(obs, dtype=complex)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def make_qubit(t1=None, t2=None, tphi=None, omega=0.0):
    """Single qubit with amplitude damping (T1) and pure dephasing.

    Coherences decay as exp(-t/T2) with 1/T2 = 1/(2 T1) + 1/Tphi. Give tphi or t2, not both;
    t2 alone means pure dephasing (no T1), and t2 > 2 T1 is rejected. omega != 0 adds
    H = -(omega/2) Z, so |0> stays the ground state. Channels are added as
    sigma_- with rate 1/T1 and Z with rate 1/(2 Tphi).
    """
    for name, val in (("t1", t1), ("t2", t2), ("tphi", tphi)):
        if val is not None and not (np.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be positive and finite")
    if t2 is not None and tphi is not None:
        raise ValueError("give tphi or t2, not both")

    inv_tphi = 0.0
    if tphi is not None:
        inv_tphi = 1.0 / tphi
    elif t2 is not None:
        inv_tphi = 1.0 / t2 - (0.5 / t1 if t1 is not None else 0.0)
        if inv_tphi < -1e-12:
            raise ValueError("t2 cannot exceed 2 * t1")
        inv_tphi = max(inv_tphi, 0.0)

    s = TrajectorySolver(1)
    if omega != 0.0:
        s.set_hamiltonian(-0.5 * omega * SIGMA_Z)
    if t1 is not None:
        s.add_lindblad(SIGMA_MINUS, 1.0 / t1)
    if inv_tphi > 0.0:
        s.add_lindblad(SIGMA_Z, 0.5 * inv_tphi)
    return s


def make_dephasing(gamma):
    """Pure dephasing D[sqrt(gamma) Z]: <X> from |+> decays as exp(-2 gamma t)."""
    s = TrajectorySolver(1)
    s.add_lindblad(SIGMA_Z, float(gamma))
    return s


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate(solver, psi0, duration, observable, kind=Unraveling.STANDARD, n_trajectories=4000,
             steps=None, seed=1):
    """Mean and standard error of an observable at time `duration`. Returns Stats.

    observable: 'X', 'Y', 'Z', 'P0', 'P1', a Hermitian matrix (no Python callbacks, fully
    parallel), or a callable psi -> float. steps defaults to 200 (800 for ANALOG, which needs
    smaller steps).
    """
    kind = _kind(kind)
    if steps is None:
        steps = 800 if kind == Unraveling.ANALOG else 200
    dt = duration / steps if duration > 0 else 1.0
    psi0 = np.asarray(psi0, dtype=complex)
    obs = _observable(observable)
    if callable(obs):
        return solver.estimate_observable(psi0, duration, dt, n_trajectories=int(n_trajectories),
                                          obs=obs, kind=kind, seed=int(seed))
    return solver.estimate_expectation(psi0, duration, dt, n_trajectories=int(n_trajectories),
                                       op=obs, kind=kind, seed=int(seed))


def decay_curve(solver, psi0, observable, times, rate=None, kind=Unraveling.STANDARD,
                n_trajectories=4000, steps=None, seed=1):
    """Observable vs time. Returns dict of arrays: times, mean, std_error, and, if `rate` is
    given, exact = exp(-rate t) and zscore = (mean - exact) / std_error."""
    times = np.asarray(times, dtype=float)
    mean = np.empty(times.size)
    se = np.empty(times.size)
    for i, t in enumerate(times):
        e = estimate(solver, psi0, float(t), observable, kind, n_trajectories, steps, seed + i)
        mean[i], se[i] = e.mean, e.std_error
    out = {"times": times, "mean": mean, "std_error": se}
    if rate is not None:
        exact = np.exp(-rate * times)
        out["exact"] = exact
        with np.errstate(divide="ignore", invalid="ignore"):
            out["zscore"] = np.where(se > 0, (mean - exact) / se, 0.0)
    return out


# ---------------------------------------------------------------------------
# Variance reduction
# ---------------------------------------------------------------------------

def trajectories_needed(variance, eps):
    """Trajectories for standard error eps: ceil(variance / eps^2), at least 1."""
    if not eps > 0:
        raise ValueError("eps must be positive")
    return max(1, int(math.ceil(variance / (eps * eps))))


def pilot_variance(solver, psi0, duration, observable, kinds=ALL_KINDS, n_pilot=4000, steps=None, seed=1):
    """Per-trajectory variance of the observable estimator for each unraveling: {name: var}."""
    out = {}
    for k in kinds:
        k = _kind(k)
        e = estimate(solver, psi0, duration, observable, k, n_pilot, steps, seed)
        out[k.name] = e.std_error ** 2 * n_pilot
    return out


def variance_reduction(solver, psi0, duration, observable, eps=0.005, kinds=ALL_KINDS,
                       n_pilot=4000, steps=None, seed=1):
    """Trajectories needed for standard error eps under each unraveling.

    Returns {name: {"variance", "n_required", "factor_vs_standard"}}; the factor is
    N_standard / N_this (above 1 means fewer trajectories than STANDARD). The pilot variance
    carries its own statistical error, roughly sqrt(2 / n_pilot) relative.
    """
    var = pilot_variance(solver, psi0, duration, observable, (Unraveling.STANDARD,) + tuple(
        k for k in map(_kind, kinds) if k != Unraveling.STANDARD), n_pilot, steps, seed)
    base = var["STANDARD"]
    out = {}
    for name, v in var.items():
        out[name] = {
            "variance": v,
            "n_required": trajectories_needed(v, eps),
            "factor_vs_standard": (base / v) if v > 0 else float("inf"),
        }
    return out


def analytic_dephasing_ratio(gamma_t):
    """Var_STANDARD / Var_PROJECTOR for L = Z, rate gamma, |+>, <X>: 1 + exp(4 gamma t)."""
    return 1.0 + math.exp(4.0 * gamma_t)


def dephasing_variance_ratio(gamma_t, gamma=1.0, n_pilot=20000, steps=None, seed=1):
    """Measured STANDARD/PROJECTOR variance ratio for pure dephasing, with the analytic value."""
    s = make_dephasing(gamma)
    var = pilot_variance(s, plus(), gamma_t / gamma, "X", (Unraveling.STANDARD, Unraveling.PROJECTOR),
                         n_pilot, steps, seed)
    return {
        "gamma_t": gamma_t,
        "var_standard": var["STANDARD"],
        "var_projector": var["PROJECTOR"],
        "ratio": var["STANDARD"] / var["PROJECTOR"] if var["PROJECTOR"] > 0 else float("inf"),
        "analytic": analytic_dephasing_ratio(gamma_t),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _prepare(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return plt, path


def plot_decay(curves, rate, path=IMAGES_DIR / "trajectory_decay.png", title="Trajectory decay",
               ylabel="observable"):
    """curves: {unraveling name: decay_curve result}. Draws means with error bars and exp(-rate t)."""
    plt, path = _prepare(path)
    colours = {"STANDARD": "#3b6ea5", "PROJECTOR": "#d1495b", "ANALOG": "#4c9a6a"}
    marks = {"STANDARD": "o", "PROJECTOR": "s", "ANALOG": "^"}
    first = next(iter(curves.values()))
    tt = np.linspace(0.0, float(first["times"].max()), 200)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(tt, np.exp(-rate * tt), color="black", lw=1.2, label=f"exp(-{rate:.3g} t)")
    for i, (name, c) in enumerate(curves.items()):
        shift = 0.012 * (i - 1) * float(first["times"].max())
        ax.errorbar(c["times"] + shift, c["mean"], yerr=c["std_error"], fmt=marks.get(name, "o"),
                    color=colours.get(name), capsize=3, markersize=5, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_variance_reduction(rows, path=IMAGES_DIR / "trajectory_variance_reduction.png"):
    """rows: list of dephasing_variance_ratio results. Measured and analytic ratio vs gamma t."""
    plt, path = _prepare(path)
    gt = np.array([r["gamma_t"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    grid = np.linspace(gt.min(), gt.max(), 200)
    ax.semilogy(grid, [analytic_dephasing_ratio(g) for g in grid], color="black", lw=1.2,
                label=r"$1+e^{4\gamma t}$")
    ax.semilogy(gt, [r["ratio"] for r in rows], "o", color="#d1495b", markersize=6, label="measured")
    ax.axhline(21.0, color="#999999", ls="--", lw=1.0, label="21x")
    ax.set_xlabel(r"$\gamma t$")
    ax.set_ylabel("trajectories: STANDARD / PROJECTOR")
    ax.set_title("Dephasing, |+>, <X>: variance reduction of the projector unraveling")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t1, tphi = 2.0, 1.5
    t2 = 1.0 / (0.5 / t1 + 1.0 / tphi)

    s1 = make_qubit(t1=t1)
    times1 = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]) * t1
    curves1 = {k.name: decay_curve(s1, excited(), "P1", times1, rate=1.0 / t1, kind=k, n_trajectories=4000)
               for k in ALL_KINDS}
    print(f"T1 = {t1}: worst |z| STANDARD {np.abs(curves1['STANDARD']['zscore']).max():.2f}, "
          f"PROJECTOR {np.abs(curves1['PROJECTOR']['zscore']).max():.2f}, "
          f"ANALOG {np.abs(curves1['ANALOG']['zscore']).max():.2f}")
    print("saved", plot_decay(curves1, 1.0 / t1, IMAGES_DIR / "trajectory_t1_decay.png",
                              f"T1 decay, T1 = {t1}", r"$\langle P_1\rangle$"))

    s2 = make_qubit(t1=t1, tphi=tphi)
    times2 = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]) * t2
    curves2 = {k.name: decay_curve(s2, plus(), "X", times2, rate=1.0 / t2, kind=k, n_trajectories=4000)
               for k in ALL_KINDS}
    print(f"T2 = {t2:.4f}: worst |z| STANDARD {np.abs(curves2['STANDARD']['zscore']).max():.2f}, "
          f"PROJECTOR {np.abs(curves2['PROJECTOR']['zscore']).max():.2f}, "
          f"ANALOG {np.abs(curves2['ANALOG']['zscore']).max():.2f}")
    print("saved", plot_decay(curves2, 1.0 / t2, IMAGES_DIR / "trajectory_t2_decay.png",
                              f"T2 decay, T1 = {t1}, Tphi = {tphi}", r"$\langle X\rangle$"))

    rows = [dephasing_variance_ratio(g) for g in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]
    print("\n  gamma t   measured   analytic")
    for r in rows:
        print(f"  {r['gamma_t']:<9.2f} {r['ratio']:<10.2f} {r['analytic']:.2f}")
    print("saved", plot_variance_reduction(rows))