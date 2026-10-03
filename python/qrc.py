"""High-level Python interface to the quantum reservoir computing module (Tier 4).

Wraps build/quantum_sim_qrc (src/bindings_qrc.cpp, src/Reservoir.h).

    from qrc import default_4q, run_memory_capacity, run_mackey_glass
    mc = run_memory_capacity(n_qubits=4, virtual_nodes=4)
    pr = run_mackey_glass(n_qubits=4)

Run as a script to regenerate docs/images/qrc_*.png.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
IMAGES_DIR = ROOT / "docs" / "images"

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import quantum_sim_qrc as _qrc
except ImportError as exc:
    raise ImportError(
        f"quantum_sim_qrc not found in {BUILD_DIR}. Build it with the command "
        "at the top of src/bindings_qrc.cpp."
    ) from exc

Observable = _qrc.Observable
ReservoirConfig = _qrc.ReservoirConfig
QuantumReservoir = _qrc.QuantumReservoir
LinearReadout = _qrc.LinearReadout
MemoryCapacityResult = _qrc.MemoryCapacityResult
PredictionResult = _qrc.PredictionResult
nmse = _qrc.nmse
squared_correlation = _qrc.squared_correlation
scale_to_unit = _qrc.scale_to_unit
mackey_glass = _qrc.mackey_glass

__all__ = [
    "Observable", "ReservoirConfig", "QuantumReservoir", "LinearReadout",
    "MemoryCapacityResult", "PredictionResult",
    "nmse", "squared_correlation", "scale_to_unit", "mackey_glass",
    "make_config", "default_4q", "large_6q",
    "run_memory_capacity", "run_mackey_glass",
    "plot_memory_capacity", "plot_mackey_glass_prediction",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _as_observables(observables):
    """Accept Observable members or their names ('Z', 'X', 'ZZ', 'XX')."""
    if observables is None:
        return [Observable.Z]
    if isinstance(observables, (str, Observable)):
        observables = [observables]
    out = []
    for o in observables:
        if isinstance(o, str):
            try:
                o = getattr(Observable, o.upper())
            except AttributeError:
                raise ValueError(f"unknown observable '{o}', expected Z, X, ZZ or XX")
        out.append(o)
    return out


def make_config(n_qubits=4, virtual_nodes=4, observables=None, coupling=1.0,
                field=1.0, field_disorder=0.5, tau=1.0, seed=1):
    """Build a ReservoirConfig. Validation happens when QuantumReservoir is constructed."""
    cfg = ReservoirConfig()
    cfg.n_qubits = int(n_qubits)
    cfg.virtual_nodes = int(virtual_nodes)
    cfg.observables = _as_observables(observables)
    cfg.coupling = float(coupling)
    cfg.field = float(field)
    cfg.field_disorder = float(field_disorder)
    cfg.tau = float(tau)
    cfg.seed = int(seed)
    return cfg


def default_4q(seed=1):
    """4 qubits, 4 virtual nodes, <Z_i> readout: 16 features per input."""
    return make_config(n_qubits=4, virtual_nodes=4, observables=["Z"], seed=seed)


def large_6q(seed=1):
    """6 qubits, 4 virtual nodes, <Z_i> and <Z_i Z_j> readout: 84 features per input."""
    return make_config(n_qubits=6, virtual_nodes=4, observables=["Z", "ZZ"], seed=seed)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_memory_capacity(n_qubits=4, virtual_nodes=4, observables=None, max_delay=20,
                        n_train=2000, n_test=1000, washout=100, seed=1,
                        task_seed=12345, ridge=None, **cfg_kwargs):
    """Short-term memory task on i.i.d. U[0, 1] input. Returns MemoryCapacityResult.

    seed fixes the reservoir (J_ij, h_i); task_seed fixes the input sequence.
    ridge=None uses the C++ default. Extra kwargs go to make_config
    (coupling, field, field_disorder, tau).
    """
    cfg = make_config(n_qubits, virtual_nodes, observables, seed=seed, **cfg_kwargs)
    res = QuantumReservoir(cfg)
    kw = {"washout": washout, "seed": task_seed}
    if ridge is not None:
        kw["ridge"] = ridge
    return _qrc.memory_capacity(res, max_delay, n_train, n_test, **kw)


def run_mackey_glass(n_qubits=4, n_samples=3000, horizon=1, washout=200, n_train=1500,
                     virtual_nodes=4, observables=None, seed=1, ridge=None,
                     mg_tau=17.0, **cfg_kwargs):
    """h-step-ahead Mackey-Glass prediction. Returns PredictionResult.

    Test rows = n_samples - horizon - washout - n_train, must be >= 2.
    """
    cfg = make_config(n_qubits, virtual_nodes, observables, seed=seed, **cfg_kwargs)
    res = QuantumReservoir(cfg)
    series = mackey_glass(n_samples, tau=mg_tau)
    args = (res, series, horizon, washout, n_train)
    return _qrc.predict_series(*args) if ridge is None else _qrc.predict_series(*args, ridge)


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


def plot_memory_capacity(result, path=IMAGES_DIR / "qrc_memory_capacity.png"):
    """Bar plot of MC_k against delay k, annotated with MC_total and the n_features bound."""
    plt, path = _prepare(path)
    mc = np.asarray(result.mc)
    k = np.arange(mc.size)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(k, mc, color="#3b6ea5", width=0.8)
    ax.set_xlabel("delay k")
    ax.set_ylabel(r"$MC_k$")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Memory capacity: total = {result.total:.2f} "
                 f"(bound {result.n_features} features)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mackey_glass_prediction(result, path=IMAGES_DIR / "qrc_mackey_glass.png", n_show=300):
    """Left: target vs prediction on the first n_show test points. Right: NMSE comparison."""
    plt, path = _prepare(path)
    target = np.asarray(result.target_test)
    pred = np.asarray(result.prediction_test)
    n_show = min(n_show, target.size)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.2, 1]})

    a0.plot(target[:n_show], color="black", lw=1.4, label="target")
    a0.plot(pred[:n_show], color="#d1495b", lw=1.0, ls="--", label="reservoir")
    a0.set_xlabel("test step")
    a0.set_ylabel("x (scaled)")
    a0.set_title("Mackey-Glass prediction (held-out)")
    a0.legend(frameon=False)

    labels = ["reservoir", "persistence"]
    vals = [result.nmse_test, result.nmse_persistence]
    a1.bar(labels, vals, color=["#3b6ea5", "#999999"])
    a1.set_yscale("log")
    a1.set_ylabel("test NMSE")
    a1.set_title("Test NMSE")
    for i, v in enumerate(vals):
        a1.text(i, v, f"{v:.1e}", ha="center", va="bottom", fontsize=9)
    a1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mc = run_memory_capacity()
    print(f"MC_total = {mc.total:.3f}  (n_features = {mc.n_features})")
    print("saved", plot_memory_capacity(mc))

    pr = run_mackey_glass()
    print(f"Mackey-Glass NMSE train {pr.nmse_train:.2e}  test {pr.nmse_test:.2e}  "
          f"persistence {pr.nmse_persistence:.2e}")
    print("saved", plot_mackey_glass_prediction(pr))