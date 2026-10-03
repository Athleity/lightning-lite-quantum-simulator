"""High-level Python interface to the gate-level algorithms (Tier 2).

Wraps build/quantum_sim_algorithms (src/bindings_algorithms.cpp, src/Algorithms.h).

    from algorithms import Backend, run_grover, run_maxcut, run_vqe
    g = run_grover(8, [37])                                   # StateVector backend
    q = run_maxcut([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], gammas=[0.6], betas=[0.4])
    v = run_vqe(2, hamiltonian_diag=ising_diagonal(2, zz=[(0, 1, 1.0)], z=[(0, 0.5)]))

Three interchangeable backends: Reference (scalar loop, up to 26 qubits), Eigen (vectorised,
up to 24) and StateVector (OpenMP, up to 25). Qubit q is bit q of the basis-state index.
"""

import enum
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import quantum_sim_algorithms as _alg
except ImportError as exc:
    raise ImportError(
        f"quantum_sim_algorithms not found in {BUILD_DIR}. Build it with the command "
        "at the top of src/bindings_algorithms.cpp."
    ) from exc

Graph = _alg.Graph
Grover = _alg.Grover
GroverResult = _alg.GroverResult
QAOA = _alg.QAOA
QAOAResult = _alg.QAOAResult
VQE = _alg.VQE
VQEResult = _alg.VQEResult

__all__ = [
    "Backend", "make_backend",
    "Graph", "Grover", "GroverResult", "QAOA", "QAOAResult", "VQE", "VQEResult",
    "run_grover", "run_maxcut", "run_vqe",
    "cut_values", "expected_cut", "ising_diagonal",
    "benchmark_backends",
]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend(enum.Enum):
    REFERENCE = "reference"
    EIGEN = "eigen"
    STATEVECTOR = "statevector"


_FACTORIES = {
    Backend.REFERENCE: _alg.make_reference_backend,
    Backend.EIGEN: _alg.make_eigen_backend,
    Backend.STATEVECTOR: _alg.make_statevector_backend,
}


def make_backend(kind, n_qubits):
    """Construct a backend. kind is a Backend member or its name ('eigen', 'STATEVECTOR', ...)."""
    if isinstance(kind, str):
        try:
            kind = Backend[kind.upper()]
        except KeyError:
            raise ValueError(f"unknown backend '{kind}', expected one of "
                             f"{[b.name.lower() for b in Backend]}")
    return _FACTORIES[kind](int(n_qubits))


# ---------------------------------------------------------------------------
# Grover
# ---------------------------------------------------------------------------

def run_grover(n_qubits, marked_states, backend=Backend.STATEVECTOR, iterations=-1):
    """Grover search for one or more marked basis states. Returns GroverResult.

    iterations = -1 picks round(pi/(4 theta) - 1/2), theta = asin sqrt(M/N).
    """
    if isinstance(marked_states, (int, np.integer)):
        marked_states = [marked_states]
    be = make_backend(backend, n_qubits)
    return Grover.run(be, int(n_qubits), [int(m) for m in marked_states], int(iterations))


# ---------------------------------------------------------------------------
# QAOA / MaxCut
# ---------------------------------------------------------------------------

def _make_graph(edges, n_nodes=None):
    norm = []
    for e in edges:
        i, j, w = (e[0], e[1], e[2]) if len(e) > 2 else (e[0], e[1], 1.0)
        norm.append((int(i), int(j), float(w)))
    if n_nodes is None:
        if not norm:
            raise ValueError("n_nodes is required for a graph without edges")
        n_nodes = max(max(i, j) for i, j, _ in norm) + 1
    return Graph(int(n_nodes), norm)


def run_maxcut(edges, gammas, betas, backend=Backend.STATEVECTOR, n_nodes=None):
    """QAOA for MaxCut with p = len(gammas) layers. Returns QAOAResult.

    edges: list of (i, j, weight) tuples (weight defaults to 1.0 for (i, j)).
    n_nodes defaults to max node index + 1; pass it if the graph has isolated nodes.
    """
    g = _make_graph(edges, n_nodes)
    be = make_backend(backend, g.n_nodes)
    return QAOA.run(be, g, [float(a) for a in gammas], [float(b) for b in betas])


def cut_values(edges, n_nodes=None):
    """Cut weight of every bitstring, array of size 2^n_nodes indexed by basis state."""
    g = _make_graph(edges, n_nodes)
    z = np.arange(1 << g.n_nodes, dtype=np.int64)
    cut = np.zeros(z.size)
    for i, j, w in g.edges:
        cut += w * (((z >> i) & 1) != ((z >> j) & 1))
    return cut


def expected_cut(result, edges, n_nodes=None):
    """<H_C> = sum_z P(z) cut(z) of a QAOAResult; compare against cut_values(...).max()."""
    return float(np.dot(result.probabilities, cut_values(edges, n_nodes)))


# ---------------------------------------------------------------------------
# VQE
# ---------------------------------------------------------------------------

def ising_diagonal(n_qubits, zz=(), z=()):
    """Diagonal of H = sum J_ij Z_i Z_j + sum h_i Z_i, array of size 2^n_qubits.

    zz: list of (i, j, J); z: list of (i, h). Z eigenvalue of qubit q is 1 - 2 * bit_q.
    """
    idx = np.arange(1 << int(n_qubits), dtype=np.int64)
    s = [1.0 - 2.0 * ((idx >> q) & 1) for q in range(int(n_qubits))]
    diag = np.zeros(idx.size)
    for i, j, coupling in zz:
        diag += coupling * s[i] * s[j]
    for i, field in z:
        diag += field * s[i]
    return diag


def run_vqe(n_qubits, energy_fn=None, n_layers=2, backend=Backend.STATEVECTOR, max_iter=200,
            lr=0.05, seed=1, hamiltonian_diag=None):
    """Hardware-efficient VQE (RY layers + CNOT ring). Returns VQEResult.

    Give either energy_fn or hamiltonian_diag:
      energy_fn(params: ndarray) -> float   black box; it owns state preparation and measurement
      hamiltonian_diag                       array of size 2^n_qubits (see ising_diagonal); the
                                             energy is sum_z P(z) diag[z] on the prepared state
    Gradients by the parameter-shift rule. On return the best state is prepared on the backend.
    """
    if (energy_fn is None) == (hamiltonian_diag is None):
        raise ValueError("give exactly one of energy_fn and hamiltonian_diag")

    vqe = VQE(int(n_qubits), int(n_layers), int(seed))
    be = make_backend(backend, n_qubits)

    if hamiltonian_diag is not None:
        diag = np.asarray(hamiltonian_diag, dtype=float)
        if diag.shape != (1 << int(n_qubits),):
            raise ValueError(f"hamiltonian_diag must have length 2^n_qubits = {1 << int(n_qubits)}")

        def energy_fn(params):
            vqe.prepare(be, params)
            return float(np.dot(be.probabilities(), diag))

    return vqe.optimize(be, energy_fn, int(max_iter), float(lr))


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_backends(fn, n_range, backends=None, repeats=3, reference_max_n=16):
    """Best-of-`repeats` wall time of fn(n_qubits, backend) on every backend and size.

    Returns {backend name: [time in ms for each n in n_range]}. Reference is skipped above
    reference_max_n (NaN entries). Example:

        benchmark_backends(lambda n, b: run_grover(n, [1], backend=b, iterations=10), [10, 14, 18])
    """
    kinds = list(Backend) if backends is None else [Backend[b.upper()] if isinstance(b, str) else b
                                                    for b in backends]
    out = {}
    for kind in kinds:
        row = []
        for n in n_range:
            if kind is Backend.REFERENCE and n > reference_max_n:
                row.append(float("nan"))
                continue
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn(n, kind)
                best = min(best, time.perf_counter() - t0)
            row.append(1e3 * best)
        out[kind.name.lower()] = row
    return out


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    r = run_grover(6, [37])
    print(f"Grover n=6: found {r.marked_index} in {r.iterations} iterations, "
          f"P = {r.probabilities[r.marked_index]:.4f}")

    tri = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0)]
    q = run_maxcut(tri, gammas=[0.6], betas=[0.4])
    print(f"QAOA p=1: best cut {q.best_cut}, <cut> = {expected_cut(q, tri):.3f}, "
          f"optimum {cut_values(tri).max()}")

    diag = ising_diagonal(3, zz=[(0, 1, 1.0), (1, 2, 1.0)], z=[(0, 0.5)])
    v = run_vqe(3, hamiltonian_diag=diag, n_layers=2, max_iter=300)
    print(f"VQE: E = {v.energy:.4f} after {v.iterations} steps (exact ground {diag.min():.4f})")