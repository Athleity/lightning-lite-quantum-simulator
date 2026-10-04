"""Lattice surgery on rotated surface code patches: a fault-tolerant logical CNOT.

Physics
-------
Why surgery. A planar surface code patch has no transversal two-qubit gate, and two
patches cannot be laid on top of each other. What a 2D nearest-neighbour array does
well is measure stabilizers and change *which* stabilizers it measures. Merging two
patches into one larger patch and splitting it again measures a joint logical Pauli
of the two patches. Joint Pauli measurements, plus single-patch preparation and
readout, are enough for the whole Clifford group, and the outcomes only update a
classical Pauli frame, so no physical correction gate is ever applied.

Boundaries. Data qubits sit on integer (row, col). A check ("plaquette") sits between
four data qubits, labelled by its top-left corner (i, j), and is an X check when
i + j is even, a Z check otherwise. A patch keeps every weight-4 check inside it, plus
weight-2 X checks on its top and bottom edges and weight-2 Z checks on its left and
right edges. Call an edge an X boundary or a Z boundary after the weight-2 checks it
hosts.
    logical X  a column of X, running from the top X boundary to the bottom one. A
               chain of X flips Z checks only at its two ends, so it can only end
               where a Z check is missing, and that is an X boundary.
    logical Z  a row of Z, running from the left Z boundary to the right one.
The brief paired rough/smooth with check types the other way round. The pairing above
is the one the stabilizer algebra forces; _check_geometry verifies the commutation
relations, logical operators and seam products for every region used here.

Merge. Stack patch B under patch A with one row of fresh data qubits (the strip)
between them, so that an X boundary faces an X boundary. Then measure the checks of
the combined rectangle. Away from the seam nothing changes. At the seam the weight-2 X
checks of the two boundaries become weight-4 X checks reaching into the strip, and a
row of new Z checks appears across it. The strip is prepared in |+>, so the new X
checks are already known (old boundary check times the strip's +1 X values) while the
new Z checks are random. Two facts follow:
  * Every strip qubit lies in exactly two seam Z checks, so the product of all seam Z
    checks is Z_A (one row of A) times Z_B (one row of B) with the strip cancelled. The
    product of the random outcomes is therefore the value of Z_A Z_B. That is the only
    logical information they carry. Each single outcome is just a random bit that the
    merged code now treats as a stabilizer.
  * In the merged rectangle every Z row is the same logical operator, so the two
    patches now hold one logical qubit. The merge projected onto an eigenspace of
    Z_A Z_B and erased exactly one bit: the parity, not the individual values.
Merging side by side along Z boundaries does the same for X_A X_B, with the strip in |0>.

Split. Measure the strip qubits out in the basis they were prepared in. Their product
with the weight-4 seam X checks restores the old weight-2 boundary checks (this is the
rule used for the first detectors after a split), and the random seam Z checks are
dropped. The logical information is not touched. The one subtlety is the strip's own outcome. The
merged logical operator of the other type runs through patch, strip and patch (an X
column after the ZZ merge, a Z row after the XX merge), so once the strip is read out
its outcome in that column or row joins the Pauli frame. It is a real term, not
bookkeeping noise: leaving it out makes the X-sector CNOT wrong in half the shots.

CNOT from three measurements. Control C, target T, ancilla A prepared in |+>:
    M_ZZ(C, A) -> m1,    M_XX(A, T) -> m2,    M_Z(A) -> m3.
Following each logical Pauli through the three measurements gives
    X_C -> (-1)^m2 X_C X_T,   Z_C -> Z_C,   X_T -> X_T,   Z_T -> (-1)^(m1 + m3) Z_C Z_T.
That is CNOT followed by Z_C^m2 X_T^(m1 + m3), so the outcomes are a Pauli frame:
    |c, t>  ->  |c, c xor t>   after   X_T^(m1 xor m3),   Z_C^m2   (tracked, not applied).
On the surface code the strip outcomes join the frame as well (see Split), and the
logical operators must be read on consistent rows and columns: a different row or
column differs by stabilizers whose values are random bits after a merge and split.
The brief's five-step AuxCNOT (grow, split, merge, conditional Z, shrink) is a more
compact patch schedule. I did not use it. This is the textbook version, because every
step can be checked against the flow table above.

Layout (d = distance, W = 2 d + 1 rows and columns of room):
        cols 0..d-1                 col d      cols d+1..2d
    C   rows 0..d-1
    g1  row d          (strip for the ZZ merge, prepared in |+>)
    A   rows d+1..2d      +   g2 (strip, |0>)   +   T   rows d+1..2d
C-A-g1 is one rectangle of 2 d + 1 rows, A-g2-T one rectangle of 2 d + 1 columns. Each
patch origin has row + col even, which keeps the check parity of every merged
rectangle identical to the parity of its parts.

Syndrome circuit. X checks use the order NW, NE, SW, SE and Z checks NW, SW, NE, SE.
Two checks never touch one data qubit in the same step, and every X/Z pair that
shares two data qubits meets them in the same relative order, so the checks commute
under the circuit. A mid-circuit ancilla fault spreads to the last two qubits of its
check: a horizontal pair for X checks and a vertical pair for Z checks. Both are
perpendicular to the logical operator they could extend (X columns, Z rows), so hook
errors do not cut the distance.

Fault tolerance. Each stage (prepare, merge ZZ, merge XX) runs `rounds` (default d)
rounds of stabilizer measurement. Detectors compare each check with the previous round.
Across a stage change they compare it with the check it was built from (merge: the old
boundary check, split: the larger check minus the strip outcomes), so every detector
is deterministic. The logical outcome m1 is the product of the seam Z checks in the
last ZZ-merge round and m2 the product of the seam X checks in the last XX-merge round.
Both are folded into the logical observables, so the decoder is scored on whether it
infers the Pauli frame correctly.

Readout. Two sectors are decoded, because the frame bits act on different Paulis.
  Z sector  C, T start in |c>, |t>; read everything in Z.
            obs0 = Z_C (a row of C)             must equal c
            obs1 = (row through A, g2, T) xor m1  must equal c xor t
  X sector  C, T start in |c_x>, |t_x> (X eigenstates, 1 = minus); read everything in X.
            obs0 = X_C (column d - 1 of C) xor m2 xor (g1 outcome in column d - 1)
                                                must equal c_x xor t_x
            obs1 = X_T (a column of T)          must equal t_x
Together the two sectors check the CNOT on all four Z-basis and all four X-basis inputs.
The logical error rate reported is the sum of the two sector failure probabilities, an
upper bound on the probability that a random CNOT is wrong in either basis.

Why this is the base for Pangaea. A merge is a block of space-time in which the code
is larger, a split closes it. A circuit of merges and splits is a 3D object: patches are
sheets extruded in time and a merge is a pipe joining two sheets. The brief describes
Qarakal's Pangaea as the 3D generalization of that picture. I took that from the brief
and did not check it. The mechanics are the same: seam checks, strip preparation and
readout, detectors across the change of check set, and a frame that carries the outcomes.

Noise model. The same SD6 parameterization as SurfaceCode.py (NOISE_MODELS), one
parameter p on every location. Strip qubits get a reset error when prepared and a
measurement error when read out. The same caveats apply: numbers are not comparable to
SI1000 devices at the same nominal p.

Validate against (all run by the self-test): at p = 0 the detectors never fire and
the observables never flip; the CNOT truth table holds exactly in both sectors for all
four inputs; the uncorrected readout is wrong in about half the shots while the frame
corrected one is never wrong; m1 and m2 are random, repeat in every round of their
merge, and equal the joint operator read off the final data; at p > 0 the detector
error model decomposes into graphlike errors and the failure rate rises with p.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

try:
    import stim
except ImportError as exc:
    raise ImportError("LatticeSurgery needs stim: pip install stim pymatching") from exc

try:
    from SurfaceCode import DEFAULT_P, NOISE_MODELS, make_decoder
except ImportError:  # imported from another working directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from SurfaceCode import DEFAULT_P, NOISE_MODELS, make_decoder

__all__ = [
    "LatticeSurgery", "CNOTResult", "CNOTExperiment", "Region", "Stabilizer",
    "TruthTableRow", "TruthTableReport", "MergeReport",
    "region_stabilizers", "run_scan",
]

MIN_DISTANCE = 3
DEFAULT_BATCH = 50_000
DEFAULT_SHOTS = 20_000
SECTORS = ("z", "x")
MIN_SHOTS_FOR_BRANCH_CHECK = 16  # below this a correct frame may never flip, by chance

# Offsets (di, dj) of the four data qubits of plaquette (i, j) in measurement order.
_ORDER = {
    "X": ((0, 0), (0, 1), (1, 0), (1, 1)),  # NW, NE, SW, SE
    "Z": ((0, 0), (1, 0), (0, 1), (1, 1)),  # NW, SW, NE, SE
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """Rectangle of data qubits, rows r0..r1 and cols c0..c1 inclusive.

    One region is one code: the stabilizers follow from the rectangle alone. The
    origin must have r0 + c0 even, which fixes the corner pattern of the weight-2
    boundary checks (top-left corner free of a weight-2 check).
    """

    r0: int
    r1: int
    c0: int
    c1: int

    def __post_init__(self) -> None:
        if self.r1 - self.r0 < 1 or self.c1 - self.c0 < 1:
            raise ValueError("a region needs at least 2 rows and 2 columns")
        if (self.r0 + self.c0) % 2:
            raise ValueError(f"region origin ({self.r0}, {self.c0}) must have even row + col")

    def contains(self, r: int, c: int) -> bool:
        return self.r0 <= r <= self.r1 and self.c0 <= c <= self.c1

    def qubits(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.r0, self.r1 + 1) for c in range(self.c0, self.c1 + 1)]


@dataclass(frozen=True)
class Stabilizer:
    """One check: type, plaquette index (i, j), data qubits in measurement order.

    `corners` always has four entries in the circuit's step order, None where the
    plaquette sticks out of the region (weight-2 boundary checks).
    """

    kind: str
    i: int
    j: int
    corners: tuple
    support: frozenset


@lru_cache(maxsize=None)
def region_stabilizers(region: Region) -> tuple[Stabilizer, ...]:
    """All checks of the rotated surface code on `region`.

    A weight-4 plaquette is always kept. A weight-2 plaquette on a top or bottom edge
    is kept only if it is an X check, one on a left or right edge only if it is a Z
    check. A d x d region gives d^2 - 1 checks and one logical qubit.
    """
    out = []
    for i in range(region.r0 - 1, region.r1 + 1):
        for j in range(region.c0 - 1, region.c1 + 1):
            kind = "X" if (i + j) % 2 == 0 else "Z"
            corners = tuple((i + di, j + dj) if region.contains(i + di, j + dj) else None
                            for di, dj in _ORDER[kind])
            present = [q for q in corners if q is not None]
            if len(present) == 4:
                keep = True
            elif len(present) == 2:
                same_row = present[0][0] == present[1][0]
                keep = (kind == "X") if same_row else (kind == "Z")
            else:
                keep = False
            if keep:
                out.append(Stabilizer(kind, i, j, corners, frozenset(present)))
    return tuple(out)


def _layout(d: int) -> dict:
    return {
        "C": Region(0, d - 1, 0, d - 1),
        "A": Region(d + 1, 2 * d, 0, d - 1),
        "T": Region(d + 1, 2 * d, d + 1, 2 * d),
        "CA": Region(0, 2 * d, 0, d - 1),       # C + strip g1 + A
        "AT": Region(d + 1, 2 * d, 0, 2 * d),   # A + strip g2 + T
        "g1": [(d, c) for c in range(d)],
        "g2": [(r, d) for r in range(d + 1, 2 * d + 1)],
    }


def _check_geometry(d: int) -> None:
    """Assert the algebra the physics section relies on, for every region used."""
    lay = _layout(d)
    for name in ("C", "A", "T", "CA", "AT"):
        reg = lay[name]
        stabs = region_stabilizers(reg)
        assert len(stabs) == len(reg.qubits()) - 1, f"{name}: not a one-logical-qubit code"
        for a in stabs:
            for b in stabs:
                if a.kind != b.kind:
                    assert len(a.support & b.support) % 2 == 0, f"{name}: X and Z checks anticommute"
        z_row = frozenset((reg.r0, c) for c in range(reg.c0, reg.c1 + 1))
        x_col = frozenset((r, reg.c0) for r in range(reg.r0, reg.r1 + 1))
        assert all(len(z_row & s.support) % 2 == 0 for s in stabs if s.kind == "X"), name
        assert all(len(x_col & s.support) % 2 == 0 for s in stabs if s.kind == "Z"), name
        assert len(z_row & x_col) == 1, f"{name}: logical X and Z must anticommute"

    def seam_product(reg: Region, kind: str, strip: list) -> frozenset:
        acc: frozenset = frozenset()
        for s in region_stabilizers(reg):
            if s.kind == kind and s.support & set(strip):
                acc = acc ^ s.support
        return acc

    # ZZ merge: product of seam Z checks = Z on the bottom row of C and the top row of A.
    want = frozenset((d - 1, c) for c in range(d)) | frozenset((d + 1, c) for c in range(d))
    assert seam_product(lay["CA"], "Z", lay["g1"]) == want, "seam Z product is not Z_C Z_A"
    # XX merge: product of seam X checks = X on the right column of A and left column of T.
    want = (frozenset((r, d - 1) for r in range(d + 1, 2 * d + 1))
            | frozenset((r, d + 1) for r in range(d + 1, 2 * d + 1)))
    assert seam_product(lay["AT"], "X", lay["g2"]) == want, "seam X product is not X_A X_T"


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

class _Builder:
    """Emit stim instructions and track which earlier outcomes predict each check.

    State between rounds:
      last     (kind, support) -> measurement index of the previous round
      fresh    data qubit -> basis it was just reset in (valid for one round)
      retired  data qubit -> (basis, measurement index) of a strip readout
    """

    def __init__(self, distance: int, noise: dict):
        self.d = distance
        self.W = 2 * distance + 1
        self.p_gate = float(noise.get("after_clifford_depolarization", 0.0))
        self.p_idle = float(noise.get("before_round_data_depolarization", 0.0))
        self.p_reset = float(noise.get("after_reset_flip_probability", 0.0))
        self.p_meas = float(noise.get("before_measure_flip_probability", 0.0))
        self.body = stim.Circuit()
        self.n_meas = 0
        self.t = 0
        self.last: dict = {}
        self.last_pos: dict = {}
        self.fresh: dict = {}
        self.retired: dict = {}
        self.coords: dict = {}
        self.round_log: list = []

    # -- indexing ----------------------------------------------------------
    def data(self, q: tuple[int, int]) -> int:
        r, c = q
        idx = r * self.W + c
        self.coords[idx] = (2 * c, 2 * r)
        return idx

    def anc(self, i: int, j: int) -> int:
        idx = self.W * self.W + (i + 1) * (self.W + 1) + (j + 1)
        self.coords[idx] = (2 * j + 1, 2 * i + 1)
        return idx

    # -- emission helpers --------------------------------------------------
    def _gate(self, name: str, targets: list) -> None:
        if targets:
            self.body.append(name, targets)

    def _noise(self, name: str, targets: list, p: float) -> None:
        if p > 0.0 and targets:
            self.body.append(name, targets, p)

    def _rec(self, idx: int):
        return stim.target_rec(idx - self.n_meas)

    def _detector(self, terms: Sequence[int], i: int, j: int) -> None:
        self.body.append("DETECTOR", [self._rec(k) for k in terms], [2 * j + 1, 2 * i + 1, self.t])

    # -- data qubit operations ---------------------------------------------
    def reset(self, qubits: list, basis: str) -> None:
        ids = [self.data(q) for q in qubits]
        self._gate("R" if basis == "z" else "RX", ids)
        self._noise("X_ERROR" if basis == "z" else "Z_ERROR", ids, self.p_reset)
        for q in qubits:
            self.fresh[q] = basis

    def flip(self, qubits: list, gate: str) -> None:
        """Noiseless Pauli used only to choose the input state (X on a logical-X column)."""
        self._gate(gate, [self.data(q) for q in qubits])

    def measure(self, qubits: list, basis: str, retire: bool = False) -> dict:
        ids = [self.data(q) for q in qubits]
        self._noise("X_ERROR" if basis == "z" else "Z_ERROR", ids, self.p_meas)
        self._gate("M" if basis == "z" else "MX", ids)
        out = {q: self.n_meas + k for k, q in enumerate(qubits)}
        self.n_meas += len(ids)
        if retire:
            for q in qubits:
                self.retired[q] = (basis, out[q])
        return out

    # -- detectors ---------------------------------------------------------
    def _reference(self, s: Stabilizer):
        """Earlier outcomes whose parity equals this check, [] if known, None if random."""
        key = (s.kind, s.support)
        if key in self.last:
            return [self.last[key]]
        basis = s.kind.lower()
        fresh = [q for q in s.support if q in self.fresh]
        if fresh:
            # Prepare or merge: qubits just reset in this check's own basis have a known
            # value (+1), so the check equals the old check on the remaining qubits.
            if any(self.fresh[q] != basis for q in fresh):
                return None
            rest = s.support - set(fresh)
            if not rest:
                return []
            old = self.last.get((s.kind, frozenset(rest)))
            return None if old is None else [old]
        # Split: a larger check last round, whose extra qubits were read out in this basis.
        for (kind, big), idx in self.last.items():
            if kind == s.kind and s.support < big:
                gone = sorted(big - s.support)
                if all(q in self.retired and self.retired[q][0] == basis for q in gone):
                    return [idx] + [self.retired[q][1] for q in gone]
        return None

    def round(self, regions: Sequence[Region], label: str = "") -> dict:
        """One round of syndrome extraction on every region. Returns (kind, support) -> index."""
        stabs = [s for reg in regions for s in region_stabilizers(reg)]
        qubits = sorted({q for reg in regions for q in reg.qubits()})
        data_ids = [self.data(q) for q in qubits]
        anc_ids = [self.anc(s.i, s.j) for s in stabs]
        if len(set(anc_ids)) != len(anc_ids):
            raise ValueError("regions overlap: two checks share an ancilla")
        x_anc = [a for a, s in zip(anc_ids, stabs) if s.kind == "X"]

        self._noise("DEPOLARIZE1", data_ids, self.p_idle)
        self._gate("R", anc_ids)
        self._noise("X_ERROR", anc_ids, self.p_reset)
        self._gate("H", x_anc)
        self._noise("DEPOLARIZE1", x_anc, self.p_gate)
        self.body.append("TICK")
        for step in range(4):
            pairs: list = []
            for a, s in zip(anc_ids, stabs):
                q = s.corners[step]
                if q is None:
                    continue
                dq = self.data(q)
                pairs += [a, dq] if s.kind == "X" else [dq, a]
            self._gate("CX", pairs)
            self._noise("DEPOLARIZE2", pairs, self.p_gate)
            self.body.append("TICK")
        self._gate("H", x_anc)
        self._noise("DEPOLARIZE1", x_anc, self.p_gate)
        self._noise("X_ERROR", anc_ids, self.p_meas)
        base = self.n_meas
        self._gate("M", anc_ids)
        self.n_meas += len(anc_ids)
        self.body.append("TICK")

        current = {}
        n_det = 0
        for k, s in enumerate(stabs):
            idx = base + k
            current[(s.kind, s.support)] = idx
            ref = self._reference(s)
            if ref is not None:
                self._detector([idx] + ref, s.i, s.j)
                n_det += 1
        self.last = current
        self.last_pos = {(s.kind, s.support): (s.i, s.j) for s in stabs}
        self.fresh = {}
        self.retired = {}
        self.round_log.append((label, len(stabs), n_det))
        self.t += 1
        return current

    def final_detectors(self, basis: str, readout: dict) -> None:
        """Compare each last-round check of the readout basis with the data outcomes."""
        kind = basis.upper()
        for (k, support), idx in self.last.items():
            if k == kind:
                i, j = self.last_pos[(k, support)]
                self._detector([idx] + [readout[q] for q in sorted(support)], i, j)
        self.last = {}

    def observable(self, index: int, terms: Sequence[int]) -> None:
        self.body.append("OBSERVABLE_INCLUDE", [self._rec(k) for k in terms], index)

    def finish(self) -> stim.Circuit:
        out = stim.Circuit()
        for q, (x, y) in sorted(self.coords.items()):
            out.append("QUBIT_COORDS", [q], [x, y])
        out += self.body
        return out


def _seam(round_map: dict, kind: str, strip: list) -> tuple[int, ...]:
    """Indices of the checks of `kind` that touch the strip, in a stable order."""
    s = set(strip)
    return tuple(sorted(idx for (k, support), idx in round_map.items() if k == kind and support & s))


@dataclass
class CNOTExperiment:
    """A built circuit plus the measurement indices needed to read its logical outcomes.

    sector      "z" or "x" (see module docstring)
    c, t        input bits: Z eigenvalue bits in the z sector, X eigenvalue bits in the x sector
    stop        "cnot" (full circuit) or "merge1" (ZZ merge only, read out in Z)
    m1_rounds   per ZZ-merge round, indices of the seam Z checks (product = Z_C Z_A)
    m2_rounds   per XX-merge round, indices of the seam X checks (product = X_A X_T)
    readout     data coordinate -> index of its final measurement
    strip1      strip g1 coordinate -> index of its X readout at the first split
    round_log   (stage, checks, detectors) per round, to see where detectors are missing
    """

    circuit: stim.Circuit
    distance: int
    sector: str
    c: int
    t: int
    stop: str
    m1_rounds: tuple
    m2_rounds: tuple
    readout: dict
    strip1: dict
    round_log: tuple

    def _par(self, rec: np.ndarray, coords: Sequence[tuple[int, int]] = (),
             extra: Sequence[int] = ()) -> np.ndarray:
        idx = [self.readout[q] for q in coords] + list(extra)
        if not idx:
            return np.zeros(rec.shape[0], dtype=bool)
        return (np.count_nonzero(rec[:, idx], axis=1) & 1).astype(bool)

    def outputs(self, rec: np.ndarray):
        """Logical outcomes from raw measurement records, shape (shots, num_measurements).

        Returns (out_control, out_target, raw, frame). `raw` is the readout of the qubit
        the frame acts on (target in the z sector, control in the x sector) before the
        frame bit is applied, so out = raw xor frame for that qubit.
        """
        d = self.distance
        if self.sector == "z":
            m1 = self._par(rec, extra=self.m1_rounds[-1])
            out_c = self._par(rec, [(d - 1, col) for col in range(d)])
            row = self._par(rec, [(d + 1, col) for col in range(2 * d + 1)])
            raw = self._par(rec, [(d + 1, col) for col in range(d + 1, 2 * d + 1)])
            out_t = row ^ m1
            return out_c, out_t, raw, out_t ^ raw
        # The merged logical X after the ZZ merge is a column through C, strip g1 and A, so
        # after the split the strip's X outcome in that column joins the frame. The column
        # must be the one the XX merge uses on A (d - 1): the restored X checks of A next
        # to g1 equal products of random strip outcomes, so other columns of A differ.
        frame = self._par(rec, extra=list(self.m2_rounds[-1]) + [self.strip1[(d, d - 1)]])
        raw = self._par(rec, [(r, d - 1) for r in range(d)])
        out_t = self._par(rec, [(r, d + 1) for r in range(d + 1, 2 * d + 1)])
        return raw ^ frame, out_t, raw, frame

    def expected(self) -> tuple[int, int]:
        if self.sector == "z":
            return self.c, self.c ^ self.t
        return self.c ^ self.t, self.t


def _build(distance: int, noise: dict, rounds: int, c: int, t: int,
           sector: str, stop: str) -> CNOTExperiment:
    d = distance
    lay = _layout(d)
    C, A, T, CA, AT = lay["C"], lay["A"], lay["T"], lay["CA"], lay["AT"]
    g1, g2 = lay["g1"], lay["g2"]
    b = _Builder(d, noise)

    # Prepare. C and T in the sector basis, A in |+>.
    b.reset(C.qubits(), sector)
    b.reset(T.qubits(), sector)
    b.reset(A.qubits(), "x")
    for bit, reg in ((c, C), (t, T)):
        if bit:
            if sector == "z":  # X on a logical-X column flips the logical Z value
                b.flip([(r, reg.c0) for r in range(reg.r0, reg.r1 + 1)], "X")
            else:              # Z on a logical-Z row flips the logical X value
                b.flip([(reg.r0, col) for col in range(reg.c0, reg.c1 + 1)], "Z")
    for _ in range(rounds):
        b.round([C, A, T], "prepare")

    # ZZ merge of C and A through strip g1 (prepared in |+>): measures Z_C Z_A.
    b.reset(g1, "x")
    m1_rounds = []
    for _ in range(rounds):
        m1_rounds.append(_seam(b.round([CA, T], "merge ZZ"), "Z", g1))

    if stop == "merge1":
        readout = b.measure(CA.qubits() + T.qubits(), "z")
        b.final_detectors("z", readout)
        return CNOTExperiment(b.finish(), d, sector, c, t, stop, tuple(m1_rounds), (),
                              readout, {}, tuple(b.round_log))

    # Split: read the strip out in X, restoring the boundary checks of C and A.
    strip1 = b.measure(g1, "x", retire=True)

    # XX merge of A and T through strip g2 (prepared in |0>): measures X_A X_T.
    b.reset(g2, "z")
    m2_rounds = []
    for _ in range(rounds):
        m2_rounds.append(_seam(b.round([C, AT], "merge XX"), "X", g2))

    # Split and read out in one step: measure every remaining data qubit.
    readout = b.measure(C.qubits() + AT.qubits(), sector)
    b.final_detectors(sector, readout)

    if sector == "z":
        b.observable(0, [readout[(d - 1, col)] for col in range(d)])
        b.observable(1, [readout[(d + 1, col)] for col in range(2 * d + 1)] + list(m1_rounds[-1]))
    else:
        b.observable(0, [readout[(r, d - 1)] for r in range(d)] + list(m2_rounds[-1])
                     + [strip1[(d, d - 1)]])
        b.observable(1, [readout[(r, d + 1)] for r in range(d + 1, 2 * d + 1)])
    return CNOTExperiment(b.finish(), d, sector, c, t, stop, tuple(m1_rounds), tuple(m2_rounds),
                          readout, strip1, tuple(b.round_log))


def _graphlike_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    try:
        return circuit.detector_error_model(decompose_errors=True)
    except ValueError as exc:
        raise ValueError("the CNOT circuit has an error that cannot be split into "
                         "graphlike pieces; MWPM needs a graphlike detector error model") from exc


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruthTableRow:
    sector: str
    inputs: tuple[int, int]
    expected: tuple[int, int]
    shots: int
    failures: int      # shots where (control, target) after the frame differ from `expected`
    frame_flips: int   # shots where the frame bit was 1, i.e. the correction acted

    @property
    def ok(self) -> bool:
        if self.failures:
            return False
        if self.shots >= MIN_SHOTS_FOR_BRANCH_CHECK:
            return 0 < self.frame_flips < self.shots  # both frame values exercised
        return True


@dataclass(frozen=True)
class TruthTableReport:
    rows: tuple

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.rows)

    def __str__(self) -> str:
        lines = []
        for r in self.rows:
            name = "|{}{}>".format(*r.inputs) if r.sector == "z" else "|{}{}>_x".format(*r.inputs)
            out = "|{}{}>".format(*r.expected) if r.sector == "z" else "|{}{}>_x".format(*r.expected)
            lines.append(f"{r.sector}-sector {name} -> {out}: failures {r.failures}/{r.shots}, "
                         f"frame acted in {r.frame_flips}/{r.shots} shots")
        return "\n".join(lines)


@dataclass(frozen=True)
class MergeReport:
    shots: int
    m1_repeats: bool      # every ZZ-merge round gives the same product
    m1_is_joint: bool     # m1 == Z_C xor Z_A read off the final data
    m1_random: bool       # m1 takes both values (A started in |+>, so Z_C Z_A is not fixed)
    z_c_fixed: bool       # Z_C alone is unchanged by the merge
    z_a_random: bool      # Z_A alone is not defined: only the parity is
    m2_repeats: bool
    m2_is_joint: bool     # m2 == X_A xor X_T read off the final data
    m2_random: bool

    @property
    def passed(self) -> bool:
        return all((self.m1_repeats, self.m1_is_joint, self.m1_random, self.z_c_fixed,
                    self.z_a_random, self.m2_repeats, self.m2_is_joint, self.m2_random))


@dataclass(frozen=True)
class CNOTResult:
    distance: int
    p: float
    shots: int                    # shots per sector
    truth_table_failures: int     # z-sector shots with a wrong control or target
    error_rate: float             # z-sector rate + x-sector rate, per CNOT
    error_rate_se: float
    rounds: int = 0
    x_sector_failures: int = 0
    seconds: float = 0.0

    def __str__(self) -> str:
        return (f"d={self.distance} p={self.p:g} shots={self.shots}/sector "
                f"fail(z)={self.truth_table_failures} fail(x)={self.x_sector_failures} "
                f"CNOT error={self.error_rate:.3e} +/- {self.error_rate_se:.1e}")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LatticeSurgery:
    """Logical CNOT between two distance-d rotated surface code patches by lattice surgery.

    Three patches (control, ancilla, target) and two strips, merged in the order ZZ then
    XX (see module docstring). d >= 3 and odd. `rounds` defaults to d and is used for
    preparation and for each merge. `noise` is a SurfaceCode noise model name ("sd6" or
    "readout"), `decoder` is "mwpm", "correlated" or a Decoder object.
    """

    def __init__(self, distance: int, p: float = DEFAULT_P, rounds: int | None = None,
                 noise: str = "sd6", decoder="mwpm"):
        if distance < MIN_DISTANCE or distance % 2 == 0:
            raise ValueError(f"distance must be odd and >= {MIN_DISTANCE}, got {distance}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {p}")
        rounds = distance if rounds is None else rounds
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        if noise not in NOISE_MODELS:
            raise ValueError(f"unknown noise model {noise!r}; use one of {sorted(NOISE_MODELS)}")
        self.distance, self.p, self.rounds, self.noise = distance, p, rounds, noise
        self.decoder_spec = decoder

    # -- circuits ----------------------------------------------------------
    def experiment(self, c: int = 0, t: int = 0, sector: str = "z", stop: str = "cnot",
                   noiseless: bool = False) -> CNOTExperiment:
        """Build one circuit. `noiseless` forces p = 0 whatever self.p is."""
        if c not in (0, 1) or t not in (0, 1):
            raise ValueError("c and t must be 0 or 1")
        if sector not in SECTORS:
            raise ValueError("sector must be 'z' or 'x'")
        if stop not in ("cnot", "merge1"):
            raise ValueError("stop must be 'cnot' or 'merge1'")
        if stop == "merge1" and sector != "z":
            raise ValueError("the ZZ-merge-only circuit exists in the z sector only")
        p = 0.0 if noiseless else self.p
        return _build(self.distance, NOISE_MODELS[self.noise](p), self.rounds, c, t, sector, stop)

    def cnot_circuit(self, c: int = 0, t: int = 0, sector: str = "z") -> stim.Circuit:
        """The full noisy CNOT circuit for input |c, t> (z sector) or its X-basis twin."""
        return self.experiment(c, t, sector).circuit

    @property
    def num_qubits(self) -> int:
        return len(self.cnot_circuit().get_final_qubit_coordinates())

    @property
    def num_detectors(self) -> int:
        return self.cnot_circuit().num_detectors

    def circuit_distance(self, sector: str = "z") -> int:
        """Smallest number of circuit faults that flip an observable and fire no detector.

        Uses stim's graphlike search, so it can overestimate when the true minimum fault
        is a hyperedge. Slow for d >= 5.
        """
        errors = self.cnot_circuit(0, 0, sector).search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=3,
            dont_explore_edges_increasing_symptom_degree=True,
            canonicalize_circuit_errors=True)
        return len(errors)

    # -- exact checks at p = 0 -------------------------------------------------
    def truth_table_report(self, shots: int = 64, seed: int | None = 0) -> TruthTableReport:
        """Run all four inputs in both sectors without noise and compare with the CNOT."""
        if shots < 1:
            raise ValueError("shots must be >= 1")
        rows = []
        for sector in SECTORS:
            for c, t in itertools.product((0, 1), repeat=2):
                exp = self.experiment(c, t, sector, noiseless=True)
                rec = exp.circuit.compile_sampler(seed=seed).sample(shots)
                out_c, out_t, _, frame = exp.outputs(rec)
                want_c, want_t = exp.expected()
                wrong = (out_c != bool(want_c)) | (out_t != bool(want_t))
                rows.append(TruthTableRow(sector, (c, t), (want_c, want_t), shots,
                                          int(np.count_nonzero(wrong)),
                                          int(np.count_nonzero(frame))))
        return TruthTableReport(tuple(rows))

    def verify_truth_table(self, shots: int = 64, seed: int | None = 0) -> bool:
        """True iff the CNOT truth table holds exactly at p = 0 (both sectors, all inputs)."""
        return self.truth_table_report(shots, seed).passed

    def merge_report(self, shots: int = 64, seed: int | None = 0) -> MergeReport:
        """Check that each merge measures a joint operator and nothing else, at p = 0."""
        d = self.distance
        s_flags = dict(m1_repeats=True, m1_is_joint=True, m1_random=False, z_c_fixed=True,
                       z_a_random=False)
        for c in (0, 1):
            exp = self.experiment(c, 0, "z", stop="merge1", noiseless=True)
            rec = exp.circuit.compile_sampler(seed=seed).sample(shots)
            rounds = [exp._par(rec, extra=r) for r in exp.m1_rounds]
            z_c = exp._par(rec, [(d - 1, col) for col in range(d)])
            z_a = exp._par(rec, [(d + 1, col) for col in range(d)])
            s_flags["m1_repeats"] &= all(np.array_equal(rounds[0], r) for r in rounds[1:])
            s_flags["m1_is_joint"] &= bool(np.array_equal(rounds[-1], z_c ^ z_a))
            s_flags["z_c_fixed"] &= bool(np.all(z_c == bool(c)))
            s_flags["m1_random"] |= bool(rounds[-1].any() and not rounds[-1].all())
            s_flags["z_a_random"] |= bool(z_a.any() and not z_a.all())
        exp = self.experiment(0, 0, "x", noiseless=True)
        rec = exp.circuit.compile_sampler(seed=seed).sample(shots)
        rounds = [exp._par(rec, extra=r) for r in exp.m2_rounds]
        x_a = exp._par(rec, [(r, d - 1) for r in range(d + 1, 2 * d + 1)])
        x_t = exp._par(rec, [(r, d + 1) for r in range(d + 1, 2 * d + 1)])
        return MergeReport(
            shots=shots,
            m2_repeats=all(np.array_equal(rounds[0], r) for r in rounds[1:]),
            m2_is_joint=bool(np.array_equal(rounds[-1], x_a ^ x_t)),
            m2_random=bool(rounds[-1].any() and not rounds[-1].all()),
            **s_flags)

    # -- noisy sampling --------------------------------------------------------
    def _sector_failures(self, sector: str, n_shots: int, seed: int | None, batch: int) -> int:
        # The four inputs differ only by noiseless X/Z gates on the input state, and stim
        # reports detectors and observables as flips against the noiseless run, so one
        # detector error model and one decoder serve all of them.
        inputs = list(itertools.product((0, 1), repeat=2))
        exps = [self.experiment(c, t, sector) for c, t in inputs]
        decoder = make_decoder(self.decoder_spec, _graphlike_dem(exps[0].circuit))
        quota = [n_shots // 4 + (1 if k < n_shots % 4 else 0) for k in range(4)]
        failures = 0
        for k, (exp, n) in enumerate(zip(exps, quota)):
            if n == 0:
                continue
            sampler = exp.circuit.compile_detector_sampler(
                seed=None if seed is None else seed + 17 * k + (1000 if sector == "x" else 0))
            done = 0
            while done < n:
                m = min(batch, n - done)
                detections, observables = sampler.sample(m, separate_observables=True)
                predictions = decoder.decode_batch(detections)
                failures += int(np.count_nonzero(np.any(predictions != observables, axis=1)))
                done += m
        return failures

    def sample_cnot(self, n_shots: int = DEFAULT_SHOTS, seed: int | None = None,
                    batch: int = DEFAULT_BATCH) -> CNOTResult:
        """Decode `n_shots` CNOTs in each sector and count wrong outputs.

        A shot fails when the decoder's inferred frame leaves the control or target
        wrong. Inputs cycle through all four basis states. At p = 0 use
        verify_truth_table instead, since the detector error model is empty.
        """
        if n_shots < 4 or batch < 1:
            raise ValueError("n_shots must be >= 4 and batch >= 1")
        if self.p == 0.0:
            raise ValueError("sample_cnot needs p > 0; use verify_truth_table for the exact check")
        t0 = time.perf_counter()
        zf = self._sector_failures("z", n_shots, seed, batch)
        xf = self._sector_failures("x", n_shots, seed, batch)
        pz, px = zf / n_shots, xf / n_shots
        se = math.hypot(math.sqrt(pz * (1 - pz) / n_shots), math.sqrt(px * (1 - px) / n_shots))
        return CNOTResult(self.distance, self.p, n_shots, zf, pz + px, se, self.rounds, xf,
                          time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def run_scan(distances: Sequence[int] = (3, 5), p: float = 0.002, n_shots: int = DEFAULT_SHOTS,
             seed: int | None = 0, verbose: bool = True) -> list[CNOTResult]:
    """Logical CNOT error against distance at one p. Below threshold it should fall with d."""
    results = []
    for d in sorted(set(distances)):
        r = LatticeSurgery(d, p).sample_cnot(n_shots, seed=None if seed is None else seed + d)
        results.append(r)
        if verbose:
            print(r)
    if verbose:
        for a, b in zip(results, results[1:]):
            if a.error_rate > 0 and b.error_rate > 0:
                print(f"d={a.distance} -> {b.distance}: CNOT error ratio "
                      f"{a.error_rate / b.error_rate:.2f}")
    return results


def _self_test() -> None:
    # Algebra of every region: counts, commutation, logical operators, seam products.
    for d in (3, 5, 7):
        _check_geometry(d)
    for bad in ((0, 1, 1, 2), (0, 0, 0, 3), (1, 3, 0, 3)):
        try:
            Region(*bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

    # Noiseless: the CNOT truth table holds exactly, in both sectors and for all inputs.
    ls = LatticeSurgery(3, 0.0)
    table = ls.truth_table_report(shots=64)
    assert table.passed, "\n" + str(table)
    assert ls.verify_truth_table()
    assert len(table.rows) == 8 and all(r.failures == 0 for r in table.rows)

    # Each merge measures a joint operator: repeated rounds agree, the product equals the
    # joint operator read off the data, and neither single-patch operator is defined.
    merge = ls.merge_report(shots=64)
    assert merge.passed, merge

    # The correction flips the target (z sector) or control (x sector) exactly when needed:
    # without the frame some shots are wrong, with it none are, and both values occur.
    for sector, c, t in (("z", 1, 1), ("z", 0, 1), ("x", 1, 0)):
        exp = ls.experiment(c, t, sector, noiseless=True)
        rec = exp.circuit.compile_sampler(seed=5).sample(200)
        out_c, out_t, raw, frame = exp.outputs(rec)
        want_c, want_t = exp.expected()
        corrected = out_t if sector == "z" else out_c
        want = want_t if sector == "z" else want_c
        assert np.all(corrected == bool(want))
        assert np.any(raw != bool(want)) and np.any(raw == bool(want))
        assert np.array_equal(raw ^ frame, corrected)

    # Detectors and observables are deterministic: nothing fires at p = 0.
    for d in (3, 5):
        for sector in SECTORS:
            exp = LatticeSurgery(d, 0.0).experiment(1, 1, sector, noiseless=True)
            det, obs = exp.circuit.compile_detector_sampler(seed=1).sample(
                100, separate_observables=True)
            assert not det.any() and not obs.any(), (d, sector)
            assert exp.circuit.num_observables == 2

    # With p > 0 the error model splits into graphlike errors and the rate rises with p.
    for d in (3, 5):
        _graphlike_dem(LatticeSurgery(d, 0.001).cnot_circuit())
    lo = LatticeSurgery(3, 0.001).sample_cnot(2000, seed=1)
    hi = LatticeSurgery(3, 0.004).sample_cnot(2000, seed=1)
    assert hi.error_rate > lo.error_rate

    # Input checks.
    for bad in (lambda: LatticeSurgery(4), lambda: LatticeSurgery(1),
                lambda: LatticeSurgery(3, -0.1), lambda: LatticeSurgery(3, rounds=0),
                lambda: LatticeSurgery(3, noise="nope"),
                lambda: LatticeSurgery(3).experiment(2, 0),
                lambda: LatticeSurgery(3).experiment(0, 0, "y"),
                lambda: LatticeSurgery(3).experiment(0, 0, "x", stop="merge1"),
                lambda: LatticeSurgery(3, 0.0).sample_cnot(100),
                lambda: LatticeSurgery(3, 0.001).sample_cnot(2)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

    print("LatticeSurgery: self-test passed")


if __name__ == "__main__":
    _self_test()
    print()
    run_scan()