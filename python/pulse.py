# python/pulse.py
"""
Pulse-level transmon control: thin Python layer over the C++ quantum_sim_pulse module.

Units follow src/Transmon.h:
    E/h quantities (EJ, EC, frequencies, anharmonicity) in Hz
    angular quantities (drive_freq, omega_*, envelopes) in rad/s
    times in seconds
Use the constants below, e.g. Transmon(ej=15 * GHZ, ec=300 * MHZ), duration=8 * NS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BUILD = Path(__file__).resolve().parents[1] / "build"
if str(_BUILD) not in sys.path:
    sys.path.insert(0, str(_BUILD))

try:
    import quantum_sim_pulse as _qsp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"quantum_sim_pulse not found in {_BUILD}. Build it with the command at the top "
        "of src/bindings_pulse.cpp."
    ) from exc

# --- units ---------------------------------------------------------------------------
MHZ = 1e6
GHZ = 1e9
NS = 1e-9
US = 1e-6
TWO_PI = _qsp.TWO_PI

AUTO_STEPS = _qsp.AUTO_STEPS
DriveModel = _qsp.DriveModel
DrivePulse = _qsp.DrivePulse
Envelope = _qsp.Envelope
LindbladOp = _qsp.LindbladOp
LindbladSolver = _qsp.LindbladSolver

__all__ = [
    "MHZ", "GHZ", "NS", "US", "TWO_PI", "AUTO_STEPS",
    "Transmon", "DriveModel", "DrivePulse", "Envelope", "LindbladOp", "LindbladSolver",
    "pulse_from_gaussian", "pulse_from_drag", "pulse_from_square",
    "beta_sweep", "leak_vs_nlevels",
]


class Transmon(_qsp.Transmon):
    """quantum_sim_pulse.Transmon plus a few unit-convenient helpers."""

    def __init__(self, ej, ec, n_levels=3, charge_cutoff=40, ng=0.0):
        super().__init__(ej, ec, n_levels, charge_cutoff, ng)

    @property
    def alpha_mhz(self) -> float:
        """Anharmonicity alpha/h in MHz (negative for a transmon)."""
        return self.anharmonicity() / MHZ

    @property
    def f01_ghz(self) -> float:
        """0-1 transition frequency in GHz."""
        return self.frequency_01() / GHZ

    def drive_frequency(self) -> float:
        """Resonant 0-1 drive frequency in rad/s."""
        return self.omega_01()


# --- pulse constructors ------------------------------------------------------------------
def pulse_from_gaussian(theta, duration, sigma, drive_freq):
    """Truncated Gaussian rotation by `theta` (rad). drive_freq in rad/s."""
    env = _qsp.gaussian_envelope(theta, duration, sigma)
    return DrivePulse(env, duration, drive_freq)


def pulse_from_drag(theta, duration, sigma, beta, drive_freq, transmon):
    """First-order DRAG pulse; the anharmonicity is taken from `transmon`."""
    env = _qsp.drag_envelope(theta, duration, sigma, beta, transmon.anharmonicity_angular())
    return DrivePulse(env, duration, drive_freq)


def pulse_from_square(theta, duration, drive_freq):
    """Constant-amplitude rotation by `theta` (rad)."""
    return DrivePulse(_qsp.square_envelope(theta, duration), duration, drive_freq)


# --- sweeps ----------------------------------------------------------------------------------
def beta_sweep(transmon, theta, duration, sigma, beta_array,
               initial_level=1, model=DriveModel.RWA, steps=AUTO_STEPS):
    """Leakage out of {|0>,|1>} for a resonant DRAG pulse at each beta. Returns ndarray."""
    betas = np.asarray(beta_array, dtype=float).ravel()
    wd = transmon.omega_01()
    out = np.empty(betas.size)
    for k, b in enumerate(betas):
        p = pulse_from_drag(theta, duration, sigma, b, wd, transmon)
        out[k] = transmon.leakage(p, initial_level, model, steps)
    return out


def leak_vs_nlevels(ej, ec, theta, duration, sigma, beta, n_levels_array,
                    initial_level=1, model=DriveModel.RWA, steps=AUTO_STEPS,
                    charge_cutoff=40, ng=0.0):
    """Leakage vs number of retained transmon levels (convergence check). Returns ndarray."""
    ns = np.asarray(n_levels_array, dtype=int).ravel()
    out = np.empty(ns.size)
    for k, n in enumerate(ns):
        tr = Transmon(ej, ec, int(n), charge_cutoff, ng)
        p = pulse_from_drag(theta, duration, sigma, beta, tr.omega_01(), tr)
        out[k] = tr.leakage(p, initial_level, model, steps)
    return out