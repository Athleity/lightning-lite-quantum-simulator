"""High-level Python API over the quantum_sim_noise C++ module.

Conventions (same as the C++ core):
  * Qubit q is bit q of the basis-state index, qubit 0 least significant.
  * For a gate on `targets`, bit j of the operator index acts on targets[j].
    CNOT is therefore gate("CNOT", control, target).
  * Times are in seconds. Purcell rates are in 1/s; the raw PurcellModel takes
    g, delta and kappa in rad/s, purcell_from_frequencies takes cyclic Hz.
  * Density matrices are limited to MAX_DM_QUBITS qubits. Bad input raises
    ValueError, never a silent fallback.
"""

from __future__ import annotations

import math
import operator
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import quantum_sim_noise as _core
except ImportError:
    _BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
    if str(_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(_BUILD_DIR))
    try:
        import quantum_sim_noise as _core
    except ImportError as exc:
        raise ImportError(
            "quantum_sim_noise not found. Build it with the g++ line at the top of "
            "src/bindings_noise.cpp and put the .so in build/ or on PYTHONPATH."
        ) from exc

KrausChannel = _core.KrausChannel
DepolarizingChannel = _core.DepolarizingChannel
T1T2Model = _core.T1T2Model
PurcellModel = _core.PurcellModel
density_from_statevector = _core.density_from_statevector
ground_state_density = _core.ground_state_density
is_valid_density_matrix = _core.is_valid_density_matrix

MAX_DM_QUBITS = _core.MAX_DM_QUBITS
DEFAULT_TOL = _core.DEFAULT_TOL

TWO_PI = 2.0 * math.pi
NS = 1e-9
US = 1e-6
MHZ = 1e6
GHZ = 1e9

# Typical transmon gate durations. Override per NoiseModel for a specific device.
DEFAULT_GATE_TIME_1Q = 30 * NS
DEFAULT_GATE_TIME_2Q = 250 * NS

_UNITARY_TOL = 1e-10
_NORM_TOL = 1e-9
_CACHE_LIMIT = 4096

__all__ = [
    "KrausChannel", "DepolarizingChannel", "T1T2Model", "PurcellModel",
    "density_from_statevector", "ground_state_density", "is_valid_density_matrix",
    "MAX_DM_QUBITS", "DEFAULT_TOL", "TWO_PI", "NS", "US", "MHZ", "GHZ",
    "DEFAULT_GATE_TIME_1Q", "DEFAULT_GATE_TIME_2Q",
    "gate_matrix", "QubitParams", "NoiseModel", "DensityMatrixSimulator",
    "purcell_from_frequencies", "purcell_t1_limit_sweep",
    "depolarizing_p_from_avg_fidelity", "coherence_limited_fidelity",
    "t1_experiment", "ramsey_experiment", "t1_analytic", "ramsey_analytic",
]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_PX = np.array([[0, 1], [1, 0]], dtype=complex)
_PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PZ = np.array([[1, 0], [0, -1]], dtype=complex)

_FIXED_GATES = {
    "I": _I2,
    "X": _PX,
    "Y": _PY,
    "Z": _PZ,
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0),
    "S": np.diag([1.0, 1j]).astype(complex),
    "T": np.diag([1.0, np.exp(1j * math.pi / 4.0)]).astype(complex),
    # Operator index = control + 2 * target, so basis states 1 and 3 swap.
    "CNOT": np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex),
    "CZ": np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
}
_ROTATION_AXES = {"RX": _PX, "RY": _PY, "RZ": _PZ}


def gate_matrix(name: str, theta: float | None = None) -> np.ndarray:
    """Matrix for a named gate. RX, RY, RZ need theta and use exp(-i theta P / 2)."""
    key = name.upper()
    if key in _ROTATION_AXES:
        if theta is None:
            raise ValueError(f"gate {name!r} needs an angle")
        return math.cos(theta / 2.0) * _I2 - 1j * math.sin(theta / 2.0) * _ROTATION_AXES[key]
    if theta is not None:
        raise ValueError(f"gate {name!r} takes no angle")
    try:
        return _FIXED_GATES[key].copy()
    except KeyError:
        known = ", ".join(sorted(list(_FIXED_GATES) + list(_ROTATION_AXES)))
        raise ValueError(f"unknown gate {name!r}; known gates: {known}") from None


def _check_targets(targets: Iterable[int], n_qubits: int) -> list[int]:
    out = [operator.index(q) for q in targets]
    if not out:
        raise ValueError("no target qubits given")
    if len(set(out)) != len(out):
        raise ValueError(f"repeated target qubit in {out}")
    for q in out:
        if not 0 <= q < n_qubits:
            raise ValueError(f"qubit {q} outside register of {n_qubits} qubits")
    return out


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def depolarizing_p_from_avg_fidelity(avg_fidelity: float, num_qubits: int) -> float:
    """Pauli error probability p of DepolarizingChannel for a target average fidelity.

    The channel has process fidelity 1 - p, and F_avg = (d (1 - p) + 1) / (d + 1)
    with d = 2^m, so p = (1 - F_avg) (d + 1) / d.
    """
    if num_qubits not in (1, 2):
        raise ValueError("num_qubits must be 1 or 2")
    if not 0.0 <= avg_fidelity <= 1.0:
        raise ValueError("avg_fidelity must lie in [0, 1]")
    d = 1 << num_qubits
    p = (1.0 - avg_fidelity) * (d + 1) / d
    if p > 1.0:
        raise ValueError(f"average fidelity {avg_fidelity} is below the depolarizing range")
    return p


def coherence_limited_fidelity(model: T1T2Model, duration: float) -> float:
    """Average fidelity of an ideal identity over `duration` with the given T1, T2.

    The Pauli transfer matrix diagonal is (1, c, c, s) with c = exp(-t/T2) and
    s = exp(-t/T1), so F_pro = (1 + 2c + s) / 4 and F_avg = (3 + 2c + s) / 6.
    """
    s = model.excited_population(duration)
    c = model.coherence_magnitude(duration)
    return (3.0 + 2.0 * c + s) / 6.0


def purcell_from_frequencies(g_hz: float, delta_hz: float, kappa_hz: float,
                             filter_db: float = 0.0) -> PurcellModel:
    """PurcellModel from cyclic frequencies in Hz.

    kappa_hz is the resonator linewidth as a frequency (kappa / 2 pi). The
    resulting rate is gamma_P = 2 pi * kappa_hz * (g / delta)^2 in 1/s, times
    10^(-filter_db / 10) when a filter attenuation is given.
    """
    model = PurcellModel(TWO_PI * g_hz, TWO_PI * delta_hz, TWO_PI * kappa_hz)
    return model.with_filter_db(filter_db)


def purcell_t1_limit_sweep(g_hz: float, kappa_hz: float, deltas_hz: Sequence[float],
                           filter_db: float = 0.0, on_invalid: str = "raise") -> np.ndarray:
    """Purcell-limited T1 (seconds) for each qubit-resonator detuning.

    Detunings outside the dispersive range (g / |delta| above MAX_DISPERSIVE_RATIO)
    raise ValueError, or give NaN when on_invalid="nan".
    """
    if on_invalid not in ("raise", "nan"):
        raise ValueError("on_invalid must be 'raise' or 'nan'")
    out = np.empty(len(deltas_hz))
    for k, delta in enumerate(deltas_hz):
        try:
            out[k] = purcell_from_frequencies(g_hz, float(delta), kappa_hz, filter_db).t1_limit()
        except ValueError:
            if on_invalid == "raise":
                raise
            out[k] = np.nan
    return out


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class QubitParams:
    """Intrinsic T1, T2 (seconds) and an optional Purcell channel for one qubit.

    With a PurcellModel attached, the effective relaxation is
    1/T1_eff = 1/T1 + gamma_P with the intrinsic pure dephasing time held fixed.
    """
    t1: float
    t2: float
    purcell: PurcellModel | None = None

    def __post_init__(self):
        self.effective()  # the C++ constructor validates T1, T2

    def effective(self) -> T1T2Model:
        base = T1T2Model(self.t1, self.t2)
        return base if self.purcell is None else self.purcell.combined_with(base)


class NoiseModel:
    """Per-qubit decoherence plus depolarizing gate error for a register.

    A gate is applied as an ideal unitary, followed by T1/T2 (and Purcell) decay
    on each target qubit for the gate duration, followed by depolarizing noise
    with probability p1q or p2q. This is the usual gate-then-error approximation,
    not a master-equation simulation of the pulse.
    """

    def __init__(self, qubits: Sequence[QubitParams], p1q: float = 0.0, p2q: float = 0.0,
                 gate_time_1q: float = DEFAULT_GATE_TIME_1Q,
                 gate_time_2q: float = DEFAULT_GATE_TIME_2Q):
        self._qubits = tuple(qubits)
        if not 1 <= len(self._qubits) <= MAX_DM_QUBITS:
            raise ValueError(f"need between 1 and {MAX_DM_QUBITS} qubits")
        for name, p in (("p1q", p1q), ("p2q", p2q)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name, t in (("gate_time_1q", gate_time_1q), ("gate_time_2q", gate_time_2q)):
            if not (t >= 0.0 and math.isfinite(t)):
                raise ValueError(f"{name} must be finite and >= 0")
        self.p1q, self.p2q = p1q, p2q
        self.gate_time_1q, self.gate_time_2q = gate_time_1q, gate_time_2q
        self._effective = [qp.effective() for qp in self._qubits]
        self._idle_cache: dict[tuple[int, float], KrausChannel] = {}
        self._depol_cache: dict[tuple[int, ...], KrausChannel] = {}

    @classmethod
    def uniform(cls, n_qubits: int, t1: float, t2: float,
                purcell: PurcellModel | None = None, **kwargs) -> "NoiseModel":
        return cls([QubitParams(t1, t2, purcell)] * n_qubits, **kwargs)

    @property
    def n_qubits(self) -> int:
        return len(self._qubits)

    def effective_model(self, qubit: int) -> T1T2Model:
        q = _check_targets([qubit], self.n_qubits)[0]
        return self._effective[q]

    def idle_channel(self, qubit: int, duration: float) -> KrausChannel:
        q = _check_targets([qubit], self.n_qubits)[0]
        key = (q, float(duration))
        channel = self._idle_cache.get(key)
        if channel is None:
            channel = self._effective[q].channel(duration, q)
            if len(self._idle_cache) >= _CACHE_LIMIT:
                self._idle_cache.clear()
            self._idle_cache[key] = channel
        return channel

    def gate_channels(self, targets: Sequence[int], duration: float | None = None
                      ) -> list[KrausChannel]:
        targets = _check_targets(targets, self.n_qubits)
        m = len(targets)
        if m not in (1, 2):
            raise ValueError("gate noise is defined for 1 and 2 qubit gates; decompose larger gates")
        t = duration if duration is not None else (self.gate_time_1q if m == 1 else self.gate_time_2q)
        channels = [self.idle_channel(q, t) for q in targets] if t > 0.0 else []
        p = self.p1q if m == 1 else self.p2q
        if p > 0.0:
            key = tuple(targets)
            depol = self._depol_cache.get(key)
            if depol is None:
                depol = DepolarizingChannel(p, m).to_kraus(targets)
                if len(self._depol_cache) >= _CACHE_LIMIT:
                    self._depol_cache.clear()
                self._depol_cache[key] = depol
            channels.append(depol)
        return channels

    def __repr__(self) -> str:
        return (f"NoiseModel(n_qubits={self.n_qubits}, p1q={self.p1q}, p2q={self.p2q}, "
                f"gate_time_1q={self.gate_time_1q}, gate_time_2q={self.gate_time_2q})")


# ---------------------------------------------------------------------------
# Density-matrix simulator
# ---------------------------------------------------------------------------

class DensityMatrixSimulator:
    """Density-matrix register with optional noise, built on KrausChannel.apply.

    Unitaries go through the same bit-block routine as the noise channels, so no
    2^n x 2^n operator is built for any gate.
    """

    def __init__(self, n_qubits: int, noise: NoiseModel | None = None, initial=None):
        if not 1 <= n_qubits <= MAX_DM_QUBITS:
            raise ValueError(f"n_qubits must be in [1, {MAX_DM_QUBITS}]")
        if noise is not None and noise.n_qubits != n_qubits:
            raise ValueError(f"noise model covers {noise.n_qubits} qubits, register has {n_qubits}")
        self._n = n_qubits
        self._noise = noise
        self._rho = self._prepare(initial)

    def _prepare(self, initial) -> np.ndarray:
        if initial is None:
            return ground_state_density(self._n)
        arr = np.asarray(initial, dtype=complex)
        dim = 1 << self._n
        if arr.ndim == 1:
            if arr.shape[0] != dim:
                raise ValueError(f"state vector has length {arr.shape[0]}, expected {dim}")
            return density_from_statevector(arr)
        if arr.ndim == 2:
            if arr.shape != (dim, dim):
                raise ValueError(f"density matrix has shape {arr.shape}, expected {(dim, dim)}")
            if not is_valid_density_matrix(arr):
                raise ValueError("initial matrix is not a valid density matrix")
            return np.array(arr, dtype=complex, order="F")
        raise ValueError("initial must be a state vector or a density matrix")

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def noise(self) -> NoiseModel | None:
        return self._noise

    @property
    def rho(self) -> np.ndarray:
        return self._rho.copy()

    def reset(self, initial=None) -> None:
        self._rho = self._prepare(initial)

    # -- evolution ---------------------------------------------------------

    def apply_channel(self, channel: KrausChannel) -> None:
        self._rho = channel.apply(self._rho, self._n)

    def apply_unitary(self, U, targets: Sequence[int], duration: float | None = None,
                      noisy: bool = True) -> None:
        targets = _check_targets(targets, self._n)
        U = np.asarray(U, dtype=complex)
        d = 1 << len(targets)
        if U.shape != (d, d):
            raise ValueError(f"unitary has shape {U.shape}, expected {(d, d)} for {len(targets)} target(s)")
        if not np.allclose(U.conj().T @ U, np.eye(d), atol=_UNITARY_TOL, rtol=0.0):
            raise ValueError("matrix is not unitary")
        # Build the noise channels first so a bad request leaves the state untouched.
        noise_channels = []
        if noisy and self._noise is not None:
            noise_channels = self._noise.gate_channels(targets, duration)
        self.apply_channel(KrausChannel([U], targets))
        for channel in noise_channels:
            self.apply_channel(channel)

    def gate(self, name: str, *targets: int, theta: float | None = None,
             duration: float | None = None, noisy: bool = True) -> None:
        self.apply_unitary(gate_matrix(name, theta), targets, duration, noisy)

    def idle(self, duration: float, qubits: Sequence[int] | None = None) -> None:
        """Let qubits sit for `duration`. A no-op when the simulator has no noise model."""
        if self._noise is None:
            return
        qs = range(self._n) if qubits is None else _check_targets(qubits, self._n)
        for q in qs:
            self.apply_channel(self._noise.idle_channel(q, duration))

    # -- readout -----------------------------------------------------------

    def populations(self) -> np.ndarray:
        return np.real(np.diag(self._rho)).copy()

    def trace(self) -> float:
        return float(np.real(np.trace(self._rho)))

    def purity(self) -> float:
        return float(np.real(np.vdot(self._rho, self._rho)))  # Tr(rho^2) for Hermitian rho

    def is_valid(self, tol: float = DEFAULT_TOL) -> bool:
        return is_valid_density_matrix(self._rho, tol)

    def reduced_density(self, qubit: int) -> np.ndarray:
        n = self._n
        q = _check_targets([qubit], n)[0]
        # C-order reshape: axis i is bit (n - 1 - i) of the row index, axis n + i of the column.
        t = self._rho.reshape((2,) * (2 * n))
        ax = n - 1 - q
        col = [i if i != ax else n + ax for i in range(n)]
        return np.einsum(t, list(range(n)) + col, [ax, n + ax])

    def bloch_vector(self, qubit: int) -> np.ndarray:
        r = self.reduced_density(qubit)
        return np.array([2.0 * r[0, 1].real, -2.0 * r[0, 1].imag, (r[0, 0] - r[1, 1]).real])

    def expect_z(self, qubit: int) -> float:
        q = _check_targets([qubit], self._n)[0]
        idx = np.arange(1 << self._n)
        signs = 1.0 - 2.0 * ((idx >> q) & 1)
        return float(np.dot(signs, self.populations()))

    def fidelity_with_pure(self, psi) -> float:
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != (1 << self._n,):
            raise ValueError(f"state vector must have length {1 << self._n}")
        if abs(np.linalg.norm(psi) - 1.0) > _NORM_TOL:
            raise ValueError("state vector is not normalized")
        return float(np.real(np.vdot(psi, self._rho @ psi)))

    def __repr__(self) -> str:
        return f"DensityMatrixSimulator(n_qubits={self._n}, noise={self._noise!r})"


# ---------------------------------------------------------------------------
# Single-qubit experiments, run through the actual Kraus channel
# ---------------------------------------------------------------------------

def _as_times(times) -> np.ndarray:
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if not (np.all(np.isfinite(t)) and np.all(t >= 0.0)):
        raise ValueError("times must be finite and >= 0")
    return t


def t1_experiment(model: T1T2Model, times) -> np.ndarray:
    """P(|1>) after each delay, starting from |1>, from the C++ channel."""
    times = _as_times(times)
    one = np.zeros((2, 2), dtype=complex)
    one[1, 1] = 1.0
    return np.array([model.channel(float(t), 0).apply(one, 1)[1, 1].real for t in times])


def ramsey_experiment(model: T1T2Model, times) -> np.ndarray:
    """|rho01(t)| / |rho01(0)| after each delay, starting from |+>, from the C++ channel."""
    times = _as_times(times)
    plus = np.full((2, 2), 0.5, dtype=complex)
    return np.array([2.0 * abs(model.channel(float(t), 0).apply(plus, 1)[0, 1]) for t in times])


def t1_analytic(model: T1T2Model, times) -> np.ndarray:
    return np.exp(-_as_times(times) / model.t1())


def ramsey_analytic(model: T1T2Model, times) -> np.ndarray:
    return np.exp(-_as_times(times) / model.t2())


# ---------------------------------------------------------------------------
# Self-test: python python/noise_models.py
# ---------------------------------------------------------------------------

def _expect_error(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def _self_test() -> None:
    t1, t2 = 100 * US, 80 * US
    model = T1T2Model(t1, t2)
    times = np.array([0.0, 25 * US, 50 * US, 100 * US])

    assert np.max(np.abs(t1_experiment(model, times) - t1_analytic(model, times))) < 1e-12
    assert np.max(np.abs(ramsey_experiment(model, times) - ramsey_analytic(model, times))) < 1e-12

    # Noiseless Bell state
    bell = np.array([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2.0)
    sim = DensityMatrixSimulator(2)
    sim.gate("H", 0)
    sim.gate("CNOT", 0, 1)
    assert abs(sim.fidelity_with_pure(bell) - 1.0) < 1e-12
    assert abs(sim.purity() - 1.0) < 1e-12
    assert abs(sim.bloch_vector(0)[2]) < 1e-12  # reduced state is maximally mixed

    # X flips the Bloch z component
    flip = DensityMatrixSimulator(3)
    flip.gate("X", 1)
    assert abs(flip.bloch_vector(1)[2] + 1.0) < 1e-12
    assert abs(flip.expect_z(1) + 1.0) < 1e-12 and abs(flip.expect_z(0) - 1.0) < 1e-12

    # Idle decay through the simulator
    noise = NoiseModel.uniform(1, t1, t2)
    decay = DensityMatrixSimulator(1, noise, initial=np.array([0.0, 1.0]))
    decay.idle(50 * US)
    assert abs(decay.populations()[1] - math.exp(-50 * US / t1)) < 1e-12

    # Coherence-limited fidelity against the Choi state of the idle channel
    n2 = NoiseModel.uniform(2, t1, t2)
    choi = DensityMatrixSimulator(2)
    choi.gate("H", 0)
    choi.gate("CNOT", 0, 1)
    dt = 40 * US
    choi.apply_channel(n2.idle_channel(0, dt))
    f_pro = choi.fidelity_with_pure(bell)
    s, c = math.exp(-dt / t1), math.exp(-dt / t2)
    assert abs(f_pro - (1.0 + 2.0 * c + s) / 4.0) < 1e-12
    assert abs((2.0 * f_pro + 1.0) / 3.0 - coherence_limited_fidelity(model, dt)) < 1e-12

    # Purcell: frequency helper, filter, and the effective T1 seen by the simulator
    g, delta, kappa = 100 * MHZ, 1.5 * GHZ, 1 * MHZ
    pm = purcell_from_frequencies(g, delta, kappa)
    assert abs(pm.rate() / (TWO_PI * kappa * (g / delta) ** 2) - 1.0) < 1e-14
    assert abs(purcell_from_frequencies(g, delta, kappa, filter_db=10.0).rate() / pm.rate() - 0.1) < 1e-14
    purcell_noise = NoiseModel.uniform(1, t1, t2, purcell=pm)
    ps = DensityMatrixSimulator(1, purcell_noise, initial=np.array([0.0, 1.0]))
    ps.idle(20 * US)
    assert abs(ps.populations()[1] - math.exp(-20 * US * (1.0 / t1 + pm.rate()))) < 1e-12
    eff = purcell_noise.effective_model(0)
    assert eff.t2() <= 2.0 * eff.t1() * (1.0 + 1e-9)

    sweep = purcell_t1_limit_sweep(g, kappa, [1.0 * GHZ, 2.0 * GHZ])
    assert abs(sweep[1] / sweep[0] - 4.0) < 1e-12  # 1 / delta^2 scaling

    # Fidelity <-> depolarizing p round trip
    assert abs(depolarizing_p_from_avg_fidelity(0.999, 1) - 0.0015) < 1e-15
    assert abs(depolarizing_p_from_avg_fidelity(0.99, 2) - 0.0125) < 1e-15

    # Errors
    assert _expect_error(sim.gate, "CNOT", 0, 0)
    assert _expect_error(sim.gate, "CNOT", 0, 5)
    assert _expect_error(sim.gate, "RX", 0)
    assert _expect_error(sim.gate, "FOO", 0)
    assert _expect_error(sim.apply_unitary, np.array([[1, 1], [0, 1]]), [0])
    assert _expect_error(DensityMatrixSimulator, 0)
    assert _expect_error(DensityMatrixSimulator, 2, NoiseModel.uniform(3, t1, t2))
    assert _expect_error(NoiseModel.uniform, 2, t1, 3 * t1)
    assert _expect_error(purcell_t1_limit_sweep, g, kappa, [0.1 * GHZ])

    print("noise_models: self-test passed")


if __name__ == "__main__":
    _self_test()