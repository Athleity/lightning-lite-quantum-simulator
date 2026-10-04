"""3D toric code memory on a periodic d x d x d lattice: both sectors, all three logicals.

Physics
-------
Lattice. Qubits sit on the 3 d^3 edges of a cubic lattice with periodic boundaries. A vertex
stabilizer A_v is the product of X on the 6 edges at v. A face stabilizer B_f is the product
of Z on the 4 edges around a face f. A vertex and a face share 0 or 2 edges, so all checks
commute. There are d^3 vertex checks and 3 d^3 face checks, with relations among them
(the 6 faces of a cube multiply to the identity), and the code encodes 3 logical qubits
(self-test: k = n - rank(A) - rank(B) = 3, by GF(2) elimination).

Excitations. A Z error on an edge flips the 2 vertex checks at its ends, so Z errors give
point-like defects and a Z chain is a string whose endpoints are the defects. An X error on an
edge flips the 4 faces containing it, so X errors give loop-like defects and an X membrane
is a surface whose boundary is a loop of flipped faces.

Logical operators are asymmetric:
    logical Z_k  a string of Z along a non-contractible cycle in direction k (weight d)
    logical X_k  a membrane of X over a non-contractible plane transverse to k (weight d^2)
The string crosses its own membrane once (Z_k with X_k anticommute) and crosses the others
zero times.

Two sectors, two circuits. A memory experiment protects one error type at a time.
  sector "z"  prepare |+>, measure the d^3 X vertex checks each round, measure data in X.
              Errors that matter: Z. Observable k = X parity over the membrane of direction-k
              edges at coordinate 0 (weight d^2); a Z string along k flips it.
  sector "x"  prepare |0>, measure the 3 d^3 Z face checks each round, measure data in Z.
              Errors that matter: X. Observable k = Z parity over the line of direction-k
              edges (weight d); an X membrane over the plane at coordinate 0 flips it.
Detectors compare each check with the previous round (first round: with its known value),
and the last round with the data readout. The other error type is in the circuit but flips
no detector or observable of its sector.

Decoding. Z sector: every single fault flips at most two detectors (decompose_errors splits
the rest), so the syndrome is a (3 + 1)-dimensional matching problem, and MWPM applies.
X sector: one X error flips four faces, a syndrome is a closed loop, and the optimal recovery
is a minimum-weight surface bounded by that loop, a different optimisation problem from
matching. The detector error model is therefore not graphlike and MWPM is not available
here. This module does not ship a decoder for it: pass a Decoder object (anything with `name`
and `decode_batch`, e.g. a trained neural decoder), or TrivialDecoder for the undecoded
baseline. The redundancy among face checks (cube relations) is what makes single-shot
decoding possible for this sector. It is not used here.

Statistical mechanics. Under code-capacity noise with optimal decoding the Z sector maps to
the 3D random-bond Ising model and the X sector to the 3D random plaquette gauge model
(Dennis, Kitaev, Landahl, Preskill 2002; Wang, Harrington, Preskill 2003). The literature
thresholds, as recalled and to be checked before quoting, are about 23 % for the point-like
sector and about 3.3 % for the loop-like sector, both with perfect measurement. They are not
circuit-level numbers. noise="code_capacity" (one round, perfect measurement) is the setting
closest to them, and with MWPM it measures a matching threshold, which is below the optimal
one. Circuit-level (sd6) thresholds are much lower.

Why 3D. The brief describes Qarakal's Pangaea as the 3D generalisation of lattice surgery
(see LatticeSurgery.py). That was taken from the brief and not checked. This module is a
standard 3D toric code memory, not a model of Pangaea.

Noise models (p on every listed location; flips are Z-type in sector z and X-type in x,
because resets and measurements are in the matching basis):
    sd6                 gate depolarization, data idle depolarization, reset and measurement flips
    phenomenological    flip on every data qubit each round and a measurement flip, perfect gates
    code_capacity       one round, one flip per data qubit, perfect measurement

Limitations. Sector x has no built-in decoder. The two sectors are simulated in separate
circuits, so correlations between them (Y errors) are not modelled, and sample_both_sectors
reports a union bound. Several logicals are scored together (a shot fails if any observable
is wrong), and the per-round conversion eps assumes one bit, so eps is approximate for
logicals = 3. Periodic lattices are the torus, not a planar patch with boundaries.
Mid-round ancilla faults are in the circuit (SD6), but hook errors that matter do not occur
in either sector, since the relevant Pauli of an ancilla fault does not propagate to the data.

Validate against (self-test): lattice algebra and a conflict-free gate schedule; k = 3 logical
qubits; detector and qubit counts; p = 0 is silent; a single Z error fires 2 detectors and a
single X error 4; strings and membranes are invisible to the checks and flip exactly their own
observable; stabilizers (a face, a vertex star) flip nothing; failure rate rises with p; same
seed gives identical counts; invalid inputs raise ValueError. Threshold scans are printed,
not asserted.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import time

import numpy as np

try:
    import stim
except ImportError as exc:
    raise ImportError("ThreeDSurfaceCode needs stim: pip install stim pymatching") from exc

try:
    from SurfaceCode import (DEFAULT_BATCH, DEFAULT_P, NOISE_MODELS, LogicalErrorRate,
                             _logical_rate, make_decoder)
except ImportError:  # imported from another working directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from SurfaceCode import (DEFAULT_BATCH, DEFAULT_P, NOISE_MODELS, LogicalErrorRate,
                             _logical_rate, make_decoder)

__all__ = ["ThreeDSurfaceCode", "TrivialDecoder", "combined_p_logical", "threshold_estimate",
           "run_scan"]

MIN_DISTANCE = 3
SECTORS = ("z", "x")
THREED_NOISE = ("sd6", "phenomenological", "code_capacity")
TWO_D_THRESHOLD = 0.0130      # correlated MWPM, circuit level (benchmarks/qec_correlated.py)
TWO_D_THRESHOLD_MWPM = 0.0111
LIT_POINT_CODE_CAPACITY = 0.23  # as recalled, check before quoting
LIT_LOOP_CODE_CAPACITY = 0.033  # as recalled, check before quoting

_UNIT = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_PLANES = ((0, 1), (0, 2), (1, 2))


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------

def _vertex(L: int, x: int, y: int, z: int) -> int:
    return ((x % L) * L + (y % L)) * L + (z % L)


def _edge(L: int, x: int, y: int, z: int, a: int) -> int:
    """Edge from vertex (x, y, z) to its neighbour along axis a (periodic)."""
    return _vertex(L, x, y, z) * 3 + a


def _star(L: int, x: int, y: int, z: int) -> tuple[int, ...]:
    """The 6 edges at a vertex, in circuit step order."""
    return (_edge(L, x, y, z, 0), _edge(L, x, y, z, 1), _edge(L, x, y, z, 2),
            _edge(L, x - 1, y, z, 0), _edge(L, x, y - 1, z, 1), _edge(L, x, y, z - 1, 2))


def _m(a: int, b: int) -> int:
    return 0 if b == (a + 1) % 3 else 1


def _face(L: int, x: int, y: int, z: int, a: int, b: int) -> tuple[int, ...]:
    """The 4 edges of the face spanned by axes a < b at base (x, y, z), in circuit step order.

    The slot of an edge is 2 m(axis, partner) + (0 for the low edge, 1 for the high edge).
    Because m(a, b) != m(b, a) and m(a, b) != m(a, c), every edge lands in a different step
    in each of its 4 faces, and the 4 edges of one face take 4 different steps.
    """
    ea, eb = _UNIT[a], _UNIT[b]
    slots = [0] * 4
    slots[2 * _m(a, b)] = _edge(L, x, y, z, a)
    slots[2 * _m(a, b) + 1] = _edge(L, x + eb[0], y + eb[1], z + eb[2], a)
    slots[2 * _m(b, a)] = _edge(L, x, y, z, b)
    slots[2 * _m(b, a) + 1] = _edge(L, x + ea[0], y + ea[1], z + ea[2], b)
    return tuple(slots)


def _checks(L: int, sector: str) -> list:
    """[(ancilla coordinates, edges in step order)] for the checks measured in a sector."""
    out = []
    for x, y, z in itertools.product(range(L), repeat=3):
        if sector == "z":
            out.append(((x, y, z), _star(L, x, y, z)))
        else:
            for a, b in _PLANES:
                ea, eb = _UNIT[a], _UNIT[b]
                out.append(((x + 0.5 * (ea[0] + eb[0]), y + 0.5 * (ea[1] + eb[1]),
                             z + 0.5 * (ea[2] + eb[2])), _face(L, x, y, z, a, b)))
    return out


def _plane_pos(axis: int, t: int, u: int, w: int) -> list[int]:
    pos = [0, 0, 0]
    others = [a for a in range(3) if a != axis]
    pos[axis], pos[others[0]], pos[others[1]] = t, u, w
    return pos


def _membrane(L: int, axis: int) -> frozenset:
    """Direction-`axis` edges at coordinate 0 along `axis`: the X membrane, weight L^2."""
    return frozenset(_edge(L, *_plane_pos(axis, 0, u, w), axis)
                     for u in range(L) for w in range(L))


def _line(L: int, axis: int, u: int = 1, w: int = 2) -> frozenset:
    """A non-contractible Z string of direction-`axis` edges, weight L."""
    return frozenset(_edge(L, *_plane_pos(axis, t, u, w), axis) for t in range(L))


def _gf2_rank(supports) -> int:
    basis: dict = {}
    rank = 0
    for s in supports:
        r = 0
        for e in s:
            r |= 1 << e
        while r:
            top = r.bit_length() - 1
            if top in basis:
                r ^= basis[top]
            else:
                basis[top] = r
                rank += 1
                break
    return rank


def _check_geometry(L: int) -> None:
    """Assert the algebra the physics section relies on."""
    n_edges = 3 * L ** 3
    vchk, fchk = _checks(L, "z"), _checks(L, "x")
    stars = [frozenset(es) for _, es in vchk]
    faces = [frozenset(es) for _, es in fchk]
    assert len(stars) == L ** 3 and len(faces) == 3 * L ** 3
    assert all(len(s) == 6 for s in stars) and all(len(f) == 4 for f in faces)
    cs, cf = np.zeros(n_edges, int), np.zeros(n_edges, int)
    for s in stars:
        cs[list(s)] += 1
    for f in faces:
        cf[list(f)] += 1
    assert np.all(cs == 2) and np.all(cf == 4), "edge incidence is wrong"
    assert all(len(s & f) % 2 == 0 for s in stars for f in faces), "A_v and B_f anticommute"

    for chk in (vchk, fchk):  # no data qubit is used twice in one circuit step
        for s in range(len(chk[0][1])):
            col = [es[s] for _, es in chk]
            assert len(set(col)) == len(col), "gate schedule has a conflict"

    mems = [_membrane(L, k) for k in range(3)]
    lines = [_line(L, k) for k in range(3)]
    assert all(len(m) == L * L for m in mems) and all(len(l) == L for l in lines)
    assert all(len(m & f) % 2 == 0 for m in mems for f in faces), "X membrane is detected"
    assert all(len(l & s) % 2 == 0 for l in lines for s in stars), "Z string is detected"
    for i in range(3):
        for j in range(3):
            assert len(lines[i] & mems[j]) == (1 if i == j else 0), "logical algebra is wrong"

    k = n_edges - _gf2_rank(stars) - _gf2_rank(faces)
    assert k == 3, f"expected 3 logical qubits, got {k}"


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

def _noise_params(name: str, p: float) -> dict:
    if name == "sd6":
        m = NOISE_MODELS["sd6"](p)
        return dict(gate=m["after_clifford_depolarization"],
                    idle_depol=m["before_round_data_depolarization"], idle_flip=0.0,
                    reset=m["after_reset_flip_probability"],
                    meas=m["before_measure_flip_probability"])
    if name == "phenomenological":
        return dict(gate=0.0, idle_depol=0.0, idle_flip=p, reset=0.0, meas=p)
    return dict(gate=0.0, idle_depol=0.0, idle_flip=p, reset=0.0, meas=0.0)  # code_capacity


def _build(L: int, rounds: int, noise: dict, sector: str, logicals: int,
           inject: tuple = ()) -> stim.Circuit:
    """Memory circuit. `inject` lists edges that get a deterministic Z_ERROR(1) (sector z) or
    X_ERROR(1) (sector x) right after preparation, used to test syndromes. A probability-1
    error channel is used instead of a Z/X gate because stim reports detection events relative
    to the noiseless reference run, which would absorb a gate."""
    zs = sector == "z"
    n_data = 3 * L ** 3
    chk = _checks(L, sector)
    n_chk, n_steps = len(chk), len(chk[0][1])
    data = list(range(n_data))
    ancs = [n_data + i for i in range(n_chk)]
    prep, meas = ("RX", "MX") if zs else ("R", "M")
    flip = "Z_ERROR" if zs else "X_ERROR"
    c = stim.Circuit()

    for x, y, z in itertools.product(range(L), repeat=3):
        for a in range(3):
            c.append("QUBIT_COORDS", [_edge(L, x, y, z, a)],
                     [x + 0.5 * (a == 0), y + 0.5 * (a == 1), z + 0.5 * (a == 2)])
    for i, (xyz, _) in enumerate(chk):
        c.append("QUBIT_COORDS", [ancs[i]], list(xyz))

    def noise_op(name, targets, p):
        if p > 0.0:
            c.append(name, targets, p)

    n_meas = 0

    def rec(idx):
        return stim.target_rec(idx - n_meas)

    c.append(prep, data)
    noise_op(flip, data, noise["reset"])
    for e in inject:
        c.append("Z_ERROR" if zs else "X_ERROR", [e], 1.0)
    c.append("TICK")

    prev = None
    for r in range(rounds):
        noise_op("DEPOLARIZE1", data, noise["idle_depol"])
        noise_op(flip, data, noise["idle_flip"])
        c.append(prep, ancs)
        noise_op(flip, ancs, noise["reset"])
        c.append("TICK")
        for s in range(n_steps):
            pairs = []
            for i, (_, es) in enumerate(chk):
                pairs += [ancs[i], es[s]] if zs else [es[s], ancs[i]]
            c.append("CX", pairs)
            noise_op("DEPOLARIZE2", pairs, noise["gate"])
            c.append("TICK")
        noise_op(flip, ancs, noise["meas"])
        c.append(meas, ancs)
        base = n_meas
        n_meas += n_chk
        for i, (xyz, _) in enumerate(chk):
            terms = [rec(base + i)] + ([rec(prev + i)] if prev is not None else [])
            c.append("DETECTOR", terms, list(xyz) + [r])
        prev = base
        c.append("TICK")

    noise_op(flip, data, noise["meas"])
    c.append(meas, data)
    base_d = n_meas
    n_meas += n_data
    for i, (xyz, es) in enumerate(chk):
        c.append("DETECTOR", [rec(prev + i)] + [rec(base_d + e) for e in es],
                 list(xyz) + [rounds])
    for k in range(logicals):
        edges = _membrane(L, k) if zs else _line(L, k)
        c.append("OBSERVABLE_INCLUDE", [rec(base_d + e) for e in sorted(edges)], k)
    return c


# ---------------------------------------------------------------------------
# Decoders and public class
# ---------------------------------------------------------------------------

class TrivialDecoder:
    """Predicts no logical flip, so its failure rate is the undecoded rate. Satisfies the
    Decoder protocol of SurfaceCode.py. Use it as a baseline or to exercise the plumbing."""

    name = "trivial"

    def __init__(self, num_observables: int = 1):
        if num_observables < 1:
            raise ValueError("num_observables must be >= 1")
        self.num_observables = num_observables

    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        return np.zeros((np.asarray(detections).shape[0], self.num_observables), dtype=np.uint8)


def combined_p_logical(z: LogicalErrorRate, x: LogicalErrorRate) -> float:
    """Union bound on a full-code memory failure: P(Z sector fails) + P(X sector fails), capped
    at 1. Exact only if the two failure events are disjoint, otherwise an upper bound."""
    if (z.distance, z.rounds) != (x.distance, x.rounds):
        raise ValueError("sector results must share distance and rounds")
    return min(1.0, z.p_logical + x.p_logical)


class ThreeDSurfaceCode:
    """3D toric code memory on a periodic d x d x d lattice, one error sector per instance.

    `distance` >= 3 is the lattice size. `sector` is "z" (point-like defects, MWPM available)
    or "x" (loop-like defects, needs a Decoder object, see module docstring). `rounds`
    defaults to d (1 for code_capacity, where any other value is rejected). `logicals` is 3
    (all logical qubits) or 1 (direction 0 only). Raises ValueError for invalid arguments,
    for a string decoder in sector x, and for a sector-z circuit whose detector error model is
    not graphlike.
    """

    def __init__(self, distance: int, p: float = DEFAULT_P, rounds: int | None = None,
                 noise: str = "sd6", decoder="mwpm", sector: str = "z", logicals: int = 3):
        if distance < MIN_DISTANCE:
            raise ValueError(f"distance must be >= {MIN_DISTANCE}, got {distance}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {p}")
        if noise not in THREED_NOISE:
            raise ValueError(f"unknown noise model {noise!r}; use one of {list(THREED_NOISE)}")
        if sector not in SECTORS:
            raise ValueError("sector must be 'z' or 'x'")
        if logicals not in (1, 3):
            raise ValueError("logicals must be 1 or 3")
        if rounds is None:
            rounds = 1 if noise == "code_capacity" else distance
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        if noise == "code_capacity" and rounds != 1:
            raise ValueError("code_capacity noise has perfect measurement and needs rounds = 1")
        if sector == "x" and isinstance(decoder, str):
            raise ValueError("sector 'x' has loop-like syndromes and no matching decoder; pass a "
                             "Decoder object (a trained decoder, or TrivialDecoder as baseline)")
        self.distance, self.p, self.rounds = distance, p, rounds
        self.noise, self.sector, self.logicals = noise, sector, logicals
        self.circuit = _build(distance, rounds, _noise_params(noise, p), sector, logicals)
        if sector == "z":
            try:
                self.detector_error_model = self.circuit.detector_error_model(decompose_errors=True)
            except ValueError as exc:
                raise ValueError("the 3D circuit has an error that cannot be split into graphlike "
                                 "pieces; MWPM needs a graphlike detector error model") from exc
        else:
            self.detector_error_model = self.circuit.detector_error_model(decompose_errors=False)
        self.decoder = make_decoder(decoder, self.detector_error_model)

    @property
    def num_data_qubits(self) -> int:
        """3 d^3 edge qubits."""
        return 3 * self.distance ** 3

    @property
    def num_ancillas(self) -> int:
        """d^3 vertex ancillas (sector z) or 3 d^3 face ancillas (sector x)."""
        return (1 if self.sector == "z" else 3) * self.distance ** 3

    @property
    def num_qubits(self) -> int:
        return self.num_data_qubits + self.num_ancillas

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    @property
    def num_logicals(self) -> int:
        """Logical observables scored in this instance (3 = all logical qubits of the code)."""
        return self.circuit.num_observables

    def circuit_with_errors(self, edges) -> stim.Circuit:
        """Circuit with a deterministic Z_ERROR(1) (sector z) or X_ERROR(1) (sector x) on each
        listed edge after preparation, and no other noise. For tests."""
        edges = tuple(edges)
        if any(not 0 <= e < self.num_data_qubits for e in edges):
            raise ValueError("edge index out of range")
        return _build(self.distance, self.rounds, _noise_params(self.noise, 0.0), self.sector,
                      self.logicals, edges)

    def sample(self, n_shots: int, seed: int | None = None,
               batch: int = DEFAULT_BATCH) -> LogicalErrorRate:
        """Decode `n_shots` shots; a shot fails if any scored observable is mispredicted."""
        if n_shots < 1 or batch < 1:
            raise ValueError("n_shots and batch must be >= 1")
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        t0 = time.perf_counter()
        errors = done = 0
        while done < n_shots:
            m = min(batch, n_shots - done)
            detections, observables = sampler.sample(m, separate_observables=True)
            predictions = np.asarray(self.decoder.decode_batch(detections))
            errors += int(np.count_nonzero(np.any(predictions != observables, axis=1)))
            done += m
        return _logical_rate(self.distance, self.rounds, self.p, n_shots, errors,
                             time.perf_counter() - t0)

    def sample_both_sectors(self, n_shots: int, seed: int | None = None, x_decoder=None,
                            batch: int = DEFAULT_BATCH):
        """(sector z rate, sector x rate) at this distance, p, rounds, noise and logicals.

        Sector z uses MWPM (or this instance's decoder if it is the z sector). Sector x uses
        `x_decoder`, required unless this instance is the x sector. Combine with
        combined_p_logical for a union bound on a full-code memory failure."""
        if self.sector == "x":
            x_decoder = self.decoder
        elif x_decoder is None:
            raise ValueError("pass x_decoder: sector x has no matching decoder")
        z_dec = self.decoder if self.sector == "z" else "mwpm"
        kw = dict(rounds=self.rounds, noise=self.noise, logicals=self.logicals)
        zc = ThreeDSurfaceCode(self.distance, self.p, decoder=z_dec, sector="z", **kw)
        xc = ThreeDSurfaceCode(self.distance, self.p, decoder=x_decoder, sector="x", **kw)
        s0 = None if seed is None else seed
        s1 = None if seed is None else seed + 1
        return zc.sample(n_shots, s0, batch), xc.sample(n_shots, s1, batch)


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------

def threshold_estimate(ps, small: list, large: list):
    """p where eps(small d) = eps(large d), by log-log interpolation of the ratio, or None.

    Below threshold the larger lattice is better (ratio > 1). Needs a sign change of
    log(eps_small / eps_large) between consecutive scanned p with errors on both sides."""
    pts = [(p, a.eps / b.eps) for p, a, b in zip(ps, small, large) if a.errors > 0 and b.errors > 0]
    for (p0, r0), (p1, r1) in zip(pts, pts[1:]):
        if r0 > 1.0 >= r1:
            f = math.log(r0) / (math.log(r0) - math.log(r1))
            return math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0)))
    return None


def run_scan(distances=(3, 5, 7), ps=(0.002, 0.005, 0.01, 0.02), n_shots: int = 5000,
             noise: str = "sd6", seed: int | None = 0, verbose: bool = True) -> dict:
    """Sector-z memory at every (d, p). Returns {(d, p): LogicalErrorRate} plus estimates.

    Prints eps per round, the suppression eps_d / eps_{d+2}, and a crossing estimate for
    each neighbouring pair (None when the scan has no crossing or too few errors)."""
    ds = sorted(set(distances))
    res: dict = {}
    for d in ds:
        for p in ps:
            res[(d, p)] = ThreeDSurfaceCode(d, p, noise=noise).sample(
                n_shots, seed=None if seed is None else seed + d)
    est = {}
    for a, b in zip(ds, ds[1:]):
        est[(a, b)] = threshold_estimate(ps, [res[(a, p)] for p in ps], [res[(b, p)] for p in ps])
    if verbose:
        print(f"sector z, noise={noise}, {n_shots} shots per point, 3 logicals scored")
        print(f"{'p':>8}" + "".join(f"{'d=' + str(d):>14}" for d in ds)
              + "".join(f"{f'{a}->{b}':>10}" for a, b in zip(ds, ds[1:])))
        for p in ps:
            row = f"{p:>8g}" + "".join(f"{res[(d, p)].eps:>14.3e}" for d in ds)
            for a, b in zip(ds, ds[1:]):
                ra, rb = res[(a, p)], res[(b, p)]
                row += f"{ra.eps / rb.eps:>10.2f}" if ra.errors and rb.errors else f"{'n/a':>10}"
            print(row)
        for (a, b), e in est.items():
            print(f"  crossing d={a} vs d={b}: " + (f"p ~ {e:.4f}" if e else "none in range"))
    return {"results": res, "crossings": est}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    for L in (3, 4, 5):
        _check_geometry(L)
    print("[1] lattice algebra, conflict-free schedules and k = 3 logical qubits: d = 3, 4, 5")

    L = 3
    cz = ThreeDSurfaceCode(L, 0.001)
    cx = ThreeDSurfaceCode(L, 0.001, sector="x", decoder=TrivialDecoder(3))
    assert cz.num_data_qubits == 81 and cx.num_data_qubits == 81
    assert cz.num_ancillas == 27 and cx.num_ancillas == 81
    assert cz.num_qubits == 108 and cx.num_qubits == 162
    assert cz.num_detectors == 27 * (L + 1) and cx.num_detectors == 81 * (L + 1)
    assert cz.num_logicals == 3 and cx.num_logicals == 3
    assert len(cz.circuit.get_final_qubit_coordinates()) == cz.num_qubits
    assert len(cx.circuit.get_final_qubit_coordinates()) == cx.num_qubits
    print(f"[2] d=3 z: {cz.num_qubits} qubits, {cz.num_detectors} detectors (graphlike DEM); "
          f"x: {cx.num_qubits} qubits, {cx.num_detectors} detectors; 3 observables each")

    for code in (ThreeDSurfaceCode(L, 0.0), ThreeDSurfaceCode(L, 0.0, sector="x",
                                                                 decoder=TrivialDecoder(3))):
        det, obs = code.circuit.compile_detector_sampler(seed=1).sample(50, separate_observables=True)
        assert not det.any() and not obs.any()
        assert code.sample(500, seed=1).errors == 0
    print("[3] p = 0: no detector fires, no observable flips (both sectors)")

    def run(code, edges, seed=2):
        return code.circuit_with_errors(edges).compile_detector_sampler(seed=seed).sample(
            8, separate_observables=True)

    onehot = [np.eye(3, dtype=bool)[k] for k in range(3)]
    for k in range(3):
        e_in = _edge(L, *_plane_pos(k, 0, 1, 1), k)
        e_out = _edge(L, *_plane_pos(k, 1, 1, 1), k)
        det, obs = run(cz, [e_in])
        assert np.all(det.sum(axis=1) == 2) and np.all(obs == onehot[k]), ("Z", k)
        det, obs = run(cz, [e_out])
        assert np.all(det.sum(axis=1) == 2) and not obs.any()
        e_in = _edge(L, *_plane_pos(k, 1, 1, 2), k)
        e_out = _edge(L, *_plane_pos(k, 1, 0, 0), k)
        det, obs = run(cx, [e_in])
        assert np.all(det.sum(axis=1) == 4) and np.all(obs == onehot[k]), ("X", k)
        det, obs = run(cx, [e_out])
        assert np.all(det.sum(axis=1) == 4) and not obs.any()
    print("[4] single Z error: 2 detectors; single X error: 4 detectors; observable k flips "
          "only for an edge on its membrane (z) or line (x)")

    for k in range(3):
        det, obs = run(cz, sorted(_line(L, k)), seed=3)
        assert not det.any() and np.all(obs == onehot[k]), ("string", k)
        det, obs = run(cx, sorted(_membrane(L, k)), seed=3)
        assert not det.any() and np.all(obs == onehot[k]), ("membrane", k)
    det, obs = run(cz, list(_face(L, 1, 1, 1, 0, 1)))                    # Z on a face: stabilizer
    assert not det.any() and not obs.any()
    det, obs = run(cx, list(_star(L, 1, 1, 1)))                         # X on a star: stabilizer
    assert not det.any() and not obs.any()
    print("[5] Z strings along x, y, z and X membranes: no detectors, exactly their own observable; "
          "face (Z) and star (X) stabilizers: nothing")

    lo = ThreeDSurfaceCode(L, 0.002).sample(20000, seed=4).errors
    hi = ThreeDSurfaceCode(L, 0.008).sample(20000, seed=4).errors
    assert hi > lo, (lo, hi)
    xlo = ThreeDSurfaceCode(L, 0.002, sector="x", decoder=TrivialDecoder(3)).sample(20000, seed=4)
    xhi = ThreeDSurfaceCode(L, 0.008, sector="x", decoder=TrivialDecoder(3)).sample(20000, seed=4)
    assert 0 < xlo.errors < xhi.errors
    print(f"[6] failures rise with p: z (MWPM) {lo} -> {hi}; x (undecoded) {xlo.errors} -> {xhi.errors}")

    a = ThreeDSurfaceCode(L, 0.004).sample(5000, seed=9)
    b = ThreeDSurfaceCode(L, 0.004).sample(5000, seed=9)
    assert a.errors == b.errors
    for noise in ("phenomenological", "code_capacity"):
        ThreeDSurfaceCode(L, 0.01, noise=noise).sample(2000, seed=1)
    zr, xr = ThreeDSurfaceCode(L, 0.004).sample_both_sectors(4000, seed=5, x_decoder=TrivialDecoder(3))
    print(f"[7] same seed gives identical counts; both sectors at p=0.004: z {zr.p_logical:.3e}, "
          f"x (undecoded) {xr.p_logical:.3e}, union bound {combined_p_logical(zr, xr):.3e}")

    bad_cases = (
        lambda: ThreeDSurfaceCode(2), lambda: ThreeDSurfaceCode(3, -0.1),
        lambda: ThreeDSurfaceCode(3, 1.5), lambda: ThreeDSurfaceCode(3, rounds=0),
        lambda: ThreeDSurfaceCode(3, noise="nope"), lambda: ThreeDSurfaceCode(3, decoder="nope"),
        lambda: ThreeDSurfaceCode(3, sector="y"), lambda: ThreeDSurfaceCode(3, logicals=2),
        lambda: ThreeDSurfaceCode(3, sector="x"),
        lambda: ThreeDSurfaceCode(3, sector="x", decoder="mwpm"),
        lambda: ThreeDSurfaceCode(3, noise="code_capacity", rounds=3),
        lambda: ThreeDSurfaceCode(3, 0.001).sample(0),
        lambda: ThreeDSurfaceCode(3, 0.001).circuit_with_errors([10 ** 6]),
        lambda: ThreeDSurfaceCode(3, 0.001).sample_both_sectors(100),
        lambda: TrivialDecoder(0))
    for bad in bad_cases:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    print(f"[8] {len(bad_cases)} invalid inputs raise ValueError")

    # Timing against the stated targets (printed, machine dependent).
    for d, target in ((3, 30), (5, 120)):
        t0 = time.perf_counter()
        ThreeDSurfaceCode(d, 0.003).sample(50_000, seed=1)
        dt = time.perf_counter() - t0
        print(f"[9] d={d}, 50000 shots incl. DEM build: {dt:.1f} s (target {target} s: "
              f"{'met' if dt < target else 'NOT met'})")

    print("\n[10] threshold scans (printed, not asserted)")
    shots = int(os.environ.get("THREED_SCAN_SHOTS", "5000"))
    run_scan(noise="sd6", n_shots=shots)
    print()
    cc = run_scan(ps=(0.10, 0.14, 0.18, 0.22), noise="code_capacity", n_shots=shots)

    print("\nComparison (sector z only; sector x has no decoder here)")
    print(f"  2D surface code, circuit level (qec_correlated): {TWO_D_THRESHOLD:.2%} correlated "
          f"MWPM, {TWO_D_THRESHOLD_MWPM:.2%} MWPM")
    print(f"  3D toric code, code capacity, MWPM, this module: "
          + ", ".join(f"d={a}/{b}: " + (f"{e:.1%}" if e else "no crossing")
                      for (a, b), e in cc["crossings"].items()))
    print(f"  literature, code capacity, optimal decoding (as recalled, check): point-like "
          f"~{LIT_POINT_CODE_CAPACITY:.0%}, loop-like ~{LIT_LOOP_CODE_CAPACITY:.1%}")
    print("  circuit-level and code-capacity numbers are different quantities, do not compare directly")
    print("\nThreeDSurfaceCode: self-test passed")


if __name__ == "__main__":
    _self_test()