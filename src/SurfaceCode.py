"""Rotated surface code memory experiment: stim sampling, PyMatching MWPM decoding.

Physics
-------
Surface code. The surface code is the planar version of Kitaev's toric code. Data
qubits sit on a d x d grid. Weight-4 (weight-2 on the boundary) X-type and Z-type
stabilizers, measured by ancilla qubits, tile the grid in a checkerboard. A single
X error flips the two Z stabilizers next to it, a single Z error flips two X
stabilizers, so errors show up as pairs of "defects". The rotated layout encodes
one logical qubit in d^2 data qubits and d^2 - 1 ancillas (2 d^2 - 1 in total),
about half the cost of the unrotated layout at the same distance. Logical
operators are strings across the patch of length d, so a logical error needs a
chain of at least (d + 1) / 2 physical errors that the decoder misreads.

Memory experiment. Prepare the logical |0> (z basis), run `rounds` cycles of
stabilizer measurement, measure every data qubit, and ask whether the logical Z
value is still the one we started with. Detectors are parities of consecutive
stabilizer outcomes (round r against r - 1) and are deterministic without noise,
so a detector that fires marks a space-time defect. One logical observable
records the logical Z parity.

Decoder. With independent Pauli noise each error mechanism flips at most two
detectors of one stabilizer type, so decoding is "pair up the defects with
minimum total weight". The weight of an edge is log((1 - q) / q) for an error
with probability q, so the cheapest matching is the most likely error pattern.
Minimum-weight perfect matching (MWPM) finds this pairing in polynomial time
(blossom algorithm), and PyMatching implements it fast enough for d = 7 at
millions of shots. MWPM is the standard decoder because the matching structure
is exact for this code and noise class (Y errors, which flip both types, are
decomposed into an X and a Z part and the correlation between the two is
dropped, which costs a small constant in the effective threshold). Correlated
matching (MatchingDecoder with correlated=True) puts part of that correlation back
with a second matching pass.

Threshold. Below a critical physical error rate p_th the logical error rate falls
exponentially with distance,
    eps_d ~ C / Lambda^((d + 1) / 2),    Lambda ~ p_th / p,
and above p_th adding qubits makes things worse. Lambda > 1 is the operating
condition for error correction. Circuit-level thresholds for this noise model are
near 0.7 % to 1 %, far below the ~10 % of phenomenological noise models.

Hardware fit. Every stabilizer circuit uses only nearest-neighbour two-qubit
gates on a square lattice (each ancilla touches its four data neighbours), so the
code maps straight onto a 2D transmon array with fixed couplers, with no
long-range wiring. Syndrome extraction needs only CZ/CNOT, H, reset and
measurement, all native on superconducting chips.

Noise model. One parameter p drives every location, which is the SD6 convention
(standard depolarizing noise, six-step cycle):
    after_clifford_depolarization    DEPOLARIZE1/2(p) after every gate
    before_round_data_depolarization DEPOLARIZE1(p) on data qubits each round (idle)
    after_reset_flip_probability     X_ERROR(p) after every reset
    before_measure_flip_probability  X_ERROR(p) before every measurement
stim's generator puts the idle error on the data qubits once per round and adds no
separate idle error on the ancillas, which is a slight simplification of full SD6.
Google's SI1000 model uses different, hardware-shaped rates per location, so
numbers here are not directly comparable to Willow at the same nominal p.

Reference numbers (Google Quantum AI, "Quantum error correction below the surface
code threshold", Nature 2025, as recalled, check against the paper before quoting):
    Lambda = 2.14 +/- 0.02
    d = 7 logical lifetime ~ 291 us against ~ 119 us for the best physical qubit
    (ratio 2.4), cycle time ~ 1.1 us, mean physical T1 ~ 68 us

Validate against: with p = 0 the detectors never fire and no shot fails, the
logical error rate rises monotonically with p, falls with d below threshold, and
the per-round conversion reproduces eps exactly for a known P_L. For correlated
matching: both decoders agree shot for shot when the noise has no Y-type
mechanisms, and correlated matching fails less often on identical shots at d >= 5.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

try:
    import pymatching
    import stim
except ImportError as exc:
    raise ImportError("SurfaceCode needs stim and pymatching: pip install stim pymatching") from exc

__all__ = [
    "SurfaceCodeMemory", "LogicalErrorRate", "PairedComparison", "Decoder", "MatchingDecoder",
    "make_decoder", "LambdaEstimate", "BreakevenResult",
    "per_round_error", "per_round_error_se", "lambda_pairwise", "lambda_fit",
    "logical_lifetime", "breakeven", "run_scan",
]

MIN_DISTANCE = 3
DEFAULT_P = 0.003
DEFAULT_BATCH = 50_000
DEFAULT_MAX_SHOTS = 2_000_000
DEFAULT_MAX_ERRORS = 200

# Willow reference values (see module comment).
WILLOW_LAMBDA = 2.14
WILLOW_LAMBDA_SE = 0.02
WILLOW_T1_S = 68e-6
WILLOW_BEST_PHYSICAL_S = 119e-6
WILLOW_CYCLE_S = 1.1e-6
WILLOW_D7_RATIO = 2.4


# ---------------------------------------------------------------------------
# Logical error statistics
# ---------------------------------------------------------------------------

def per_round_error(p_logical: float, rounds: int) -> float:
    """Per-round logical error eps from the failure probability over `rounds` rounds.

    The logical bit flips independently each round with probability eps, so
    1 - 2 P_L = (1 - 2 eps)^rounds, giving eps = (1 - (1 - 2 P_L)^(1/rounds)) / 2.
    P_L >= 1/2 means the memory is fully randomized and returns 1/2.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not 0.0 <= p_logical <= 1.0:
        raise ValueError("p_logical must lie in [0, 1]")
    if p_logical >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_logical) ** (1.0 / rounds))


def per_round_error_se(p_logical: float, se_logical: float, rounds: int) -> float:
    """Standard error of eps by propagating the binomial error of P_L.

    d eps / d P_L = (1 - 2 P_L)^(1/rounds - 1) / rounds.
    """
    if p_logical >= 0.5:
        return float("nan")
    return se_logical * (1.0 - 2.0 * p_logical) ** (1.0 / rounds - 1.0) / rounds


@dataclass(frozen=True)
class LogicalErrorRate:
    distance: int
    rounds: int
    p: float
    shots: int
    errors: int
    p_logical: float   # failure probability over `rounds` rounds
    se_logical: float  # binomial standard error of p_logical
    eps: float         # per-round logical error
    eps_se: float
    seconds: float

    def __str__(self) -> str:
        return (f"d={self.distance} p={self.p:g} shots={self.shots} errors={self.errors} "
                f"P_L({self.rounds} rounds)={self.p_logical:.3e} "
                f"eps={self.eps:.3e} +/- {self.eps_se:.1e}")


def _logical_rate(distance: int, rounds: int, p: float, shots: int, errors: int,
                  seconds: float) -> LogicalErrorRate:
    p_l = errors / shots
    se = math.sqrt(p_l * (1.0 - p_l) / shots)
    return LogicalErrorRate(
        distance=distance, rounds=rounds, p=p, shots=shots, errors=errors,
        p_logical=p_l, se_logical=se,
        eps=per_round_error(p_l, rounds),
        eps_se=per_round_error_se(p_l, se, rounds),
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def _sd6(p: float) -> dict:
    return dict(after_clifford_depolarization=p, before_round_data_depolarization=p,
                after_reset_flip_probability=p, before_measure_flip_probability=p)


def _readout(p: float) -> dict:
    # Only reset and measurement flips. Each is a single X flip, so no error
    # mechanism flips detectors of both types and there are no Y-type hyperedges.
    # Correlated matching has nothing to exploit here, which makes this model the
    # negative control for it: both decoders must give identical predictions.
    return dict(after_reset_flip_probability=p, before_measure_flip_probability=p)


NOISE_MODELS = {"sd6": _sd6, "readout": _readout}


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class Decoder(Protocol):
    """Anything that maps detection events to a predicted logical flip.

    `detections` is a (shots, num_detectors) boolean array and the return value a
    (shots, num_observables) array of predicted observable flips. A shot counts
    as a logical failure when the prediction differs from the sampled observable.
    """

    name: str

    def decode_batch(self, detections: np.ndarray) -> np.ndarray: ...


class MatchingDecoder:
    """PyMatching MWPM, plain or with two-pass correlated matching.

    Plain matching treats the X-type and Z-type syndrome graphs as independent. A
    Y error, though, flips detectors in both, and stim's decompose_errors splits
    it into an X-type edge and a Z-type edge joined by "^" in the detector error
    model. Once a first matching has picked the X-type edge, the Z-type edge it is
    paired with is more likely than its stand-alone weight says: an X-type flip
    from a single-qubit depolarizing error is a Y error about half the time, and
    then the Z-type edge fires with it.

    Correlated matching (Fowler, arXiv:1310.0863) exploits this in two passes:
      1. match with the stand-alone weights;
      2. for every edge in that solution, lower the weight of the edges it was
         decomposed together with, and match again.
    The second pass is the output. It is a heuristic, not exact maximum-likelihood
    decoding: if the first pass is wrong, the reweighting can reinforce the wrong
    answer. It costs about two to three times the decoding time of plain matching.
    """

    def __init__(self, dem: stim.DetectorErrorModel, correlated: bool = False):
        self.correlated = bool(correlated)
        self.name = "correlated" if self.correlated else "mwpm"
        # The matching graph must be built with correlations enabled for the
        # second pass to know which edges belong together.
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=self.correlated)

    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        return self.matching.decode_batch(detections, enable_correlations=self.correlated)


def make_decoder(spec, dem: stim.DetectorErrorModel) -> Decoder:
    """Build a decoder from "mwpm", "correlated", or pass an object through.

    A custom object needs a `name` and a `decode_batch` method (see Decoder).
    """
    if isinstance(spec, str):
        if spec not in ("mwpm", "correlated"):
            raise ValueError(f"unknown decoder {spec!r}; use 'mwpm', 'correlated' or a Decoder object")
        return MatchingDecoder(dem, correlated=(spec == "correlated"))
    if not (hasattr(spec, "decode_batch") and hasattr(spec, "name")):
        raise ValueError("a custom decoder needs a `name` and a `decode_batch` method")
    return spec


@dataclass(frozen=True)
class PairedComparison:
    """Several decoders run on the same sampled shots at one (d, p).

    Decoding identical syndromes removes the sampling noise from the comparison:
    the difference in failures is set by the shots on which the decoders
    disagree, not by how many shots each happened to fail on. The first decoder
    is the reference. Sampling stops when the reference reaches max_errors, so
    all decoders see the same shot count.
    """

    distance: int
    rounds: int
    p: float
    shots: int
    reference: str
    rates: dict           # decoder name -> LogicalErrorRate (seconds = decode time)
    only_reference: dict  # name -> shots where the reference fails and `name` succeeds
    only_other: dict      # name -> shots where `name` fails and the reference succeeds

    def reduction(self, name: str) -> tuple[float, float, float]:
        """(fractional drop in failures against the reference, standard error, McNemar z).

        With b = only_reference and c = only_other, the failure count falls by b - c.
        The disagreeing shots are binomial with variance ~ b + c, so
        z = (b - c) / sqrt(b + c) tests whether the two decoders really differ, and
        se = sqrt(b + c) / k_ref. A negative value means `name` is worse.
        """
        k0 = self.rates[self.reference].errors
        if name == self.reference:
            return 0.0, 0.0, 0.0
        if k0 == 0:
            return float("nan"), float("nan"), float("nan")
        b, c = self.only_reference[name], self.only_other[name]
        spread = math.sqrt(b + c)
        z = (b - c) / spread if spread > 0 else 0.0
        return (b - c) / k0, spread / k0, z


class SurfaceCodeMemory:
    """Distance-d rotated surface code memory under SD6 circuit-level noise.

    d >= 3 and odd. `rounds` defaults to d. `basis` is "z" (logical |0>, protected
    against X errors) or "x" (logical |+>, protected against Z errors). Under
    symmetric depolarizing noise the two give the same rate.

    `decoder` is "mwpm" (plain matching, the default), "correlated" (two-pass
    correlated matching) or a Decoder object. `noise` is "sd6" or "readout"
    (reset and measurement flips only, a control with no Y-type correlations).
    `compare` runs several decoders on identical shots.
    """

    def __init__(self, distance: int, p: float = DEFAULT_P, rounds: int | None = None,
                 basis: str = "z", decoder="mwpm", noise: str = "sd6"):
        if distance < MIN_DISTANCE or distance % 2 == 0:
            raise ValueError(f"distance must be odd and >= {MIN_DISTANCE}, got {distance}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {p}")
        rounds = distance if rounds is None else rounds
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        if basis not in ("z", "x"):
            raise ValueError("basis must be 'z' or 'x'")
        if noise not in NOISE_MODELS:
            raise ValueError(f"unknown noise model {noise!r}; use one of {sorted(NOISE_MODELS)}")
        self.distance, self.p, self.rounds, self.basis, self.noise = distance, p, rounds, basis, noise

        self.circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis}",
            distance=distance,
            rounds=rounds,
            **NOISE_MODELS[noise](p),
        )
        # decompose_errors splits Y-type and other multi-detector errors into
        # graphlike pieces so every mechanism is an edge for the matching graph.
        self.detector_error_model = self.circuit.detector_error_model(decompose_errors=True)
        self.decoder = make_decoder(decoder, self.detector_error_model)

    @property
    def num_qubits(self) -> int:
        # circuit.num_qubits is max index + 1 and stim leaves gaps in the layout,
        # so count the qubits that actually have coordinates.
        return len(self.circuit.get_final_qubit_coordinates())

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    def sample(self, max_shots: int = DEFAULT_MAX_SHOTS, max_errors: int = DEFAULT_MAX_ERRORS,
               batch: int = DEFAULT_BATCH, seed: int | None = None) -> LogicalErrorRate:
        """Sample, decode and count logical failures.

        Stops after `max_errors` failures (relative error ~ 1/sqrt(max_errors)) or
        `max_shots` shots, whichever comes first. At p = 0 it runs all max_shots
        and reports zero errors, with a zero standard error, so check `errors`
        before trusting a rate.
        """
        if max_shots < 1 or batch < 1 or max_errors < 1:
            raise ValueError("max_shots, batch and max_errors must be >= 1")
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        t0 = time.perf_counter()
        shots = errors = 0
        while shots < max_shots and errors < max_errors:
            n = min(batch, max_shots - shots)
            detections, observables = sampler.sample(n, separate_observables=True)
            predictions = self.decoder.decode_batch(detections)
            errors += int(np.count_nonzero(np.any(predictions != observables, axis=1)))
            shots += n
        return _logical_rate(self.distance, self.rounds, self.p, shots, errors,
                             time.perf_counter() - t0)

    def compare(self, decoders: Sequence = ("mwpm", "correlated"),
                max_shots: int = DEFAULT_MAX_SHOTS, max_errors: int = DEFAULT_MAX_ERRORS,
                batch: int = DEFAULT_BATCH, seed: int | None = None) -> PairedComparison:
        """Decode the same shots with every decoder in `decoders`.

        The first decoder is the reference and sets the stopping rule (max_errors
        failures or max_shots shots). Each decoder's `seconds` is its own decoding
        time, with sampling excluded.
        """
        if max_shots < 1 or batch < 1 or max_errors < 1:
            raise ValueError("max_shots, batch and max_errors must be >= 1")
        if len(decoders) < 2:
            raise ValueError("compare needs at least two decoders")
        decs = [make_decoder(d, self.detector_error_model) for d in decoders]
        names = [d.name for d in decs]
        if len(set(names)) != len(names):
            raise ValueError(f"decoder names must be distinct, got {names}")
        ref = names[0]

        sampler = self.circuit.compile_detector_sampler(seed=seed)
        shots = 0
        errors = {n: 0 for n in names}
        seconds = {n: 0.0 for n in names}
        only_ref = {n: 0 for n in names[1:]}
        only_other = {n: 0 for n in names[1:]}
        while shots < max_shots and errors[ref] < max_errors:
            n_batch = min(batch, max_shots - shots)
            detections, observables = sampler.sample(n_batch, separate_observables=True)
            failed = {}
            for dec in decs:
                t0 = time.perf_counter()
                predictions = dec.decode_batch(detections)
                seconds[dec.name] += time.perf_counter() - t0
                failed[dec.name] = np.any(predictions != observables, axis=1)
                errors[dec.name] += int(np.count_nonzero(failed[dec.name]))
            for n in names[1:]:
                only_ref[n] += int(np.count_nonzero(failed[ref] & ~failed[n]))
                only_other[n] += int(np.count_nonzero(~failed[ref] & failed[n]))
            shots += n_batch
        rates = {n: _logical_rate(self.distance, self.rounds, self.p, shots, errors[n], seconds[n])
                 for n in names}
        return PairedComparison(self.distance, self.rounds, self.p, shots, ref, rates,
                                only_ref, only_other)


# ---------------------------------------------------------------------------
# Error suppression factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LambdaEstimate:
    label: str
    value: float
    se: float

    def __str__(self) -> str:
        return f"{self.label}: Lambda = {self.value:.3f} +/- {self.se:.3f}"


def lambda_pairwise(small: LogicalErrorRate, large: LogicalErrorRate) -> LambdaEstimate:
    """Lambda = eps_{d-2} / eps_d from two neighbouring distances.

    The relative errors add in quadrature. Raises ValueError if either side saw no
    logical errors, because the ratio is then undefined.
    """
    if large.distance != small.distance + 2:
        raise ValueError("distances must differ by 2")
    if small.errors == 0 or large.errors == 0:
        raise ValueError("no logical errors observed; raise shots or p")
    lam = small.eps / large.eps
    se = lam * math.hypot(small.eps_se / small.eps, large.eps_se / large.eps)
    return LambdaEstimate(f"d={small.distance} -> {large.distance}", lam, se)


def lambda_fit(results: Sequence[LogicalErrorRate]) -> LambdaEstimate:
    """Weighted fit of ln eps_d = ln C - ((d + 1) / 2) ln Lambda over all distances.

    Weights are 1 / sigma^2 with sigma = eps_se / eps. Needs at least 3 distances
    to say anything beyond the pairwise ratios.
    """
    pts = [r for r in results if r.errors > 0]
    if len(pts) < 3:
        raise ValueError("need at least three distances with observed errors")
    x = np.array([(r.distance + 1) / 2 for r in pts])
    y = np.log([r.eps for r in pts])
    w = 1.0 / np.array([r.eps_se / r.eps for r in pts]) ** 2
    xm = np.sum(w * x) / np.sum(w)
    sxx = np.sum(w * (x - xm) ** 2)
    slope = np.sum(w * (x - xm) * y) / sxx
    return LambdaEstimate("weighted fit", math.exp(-slope), math.exp(-slope) / math.sqrt(sxx))


# ---------------------------------------------------------------------------
# Breakeven
# ---------------------------------------------------------------------------

def logical_lifetime(eps: float, cycle_time: float) -> float:
    """Time constant of the logical <Z> decay, in seconds.

    <Z>(n) = (1 - 2 eps)^n = exp(-n / N), so N = -1 / ln(1 - 2 eps) cycles, which
    is ~ 1 / (2 eps) for small eps. Physical qubits are compared through their
    own <Z> decay time, T1.
    """
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    if cycle_time <= 0.0:
        raise ValueError("cycle_time must be positive")
    return cycle_time * (-1.0 / math.log1p(-2.0 * eps))


@dataclass(frozen=True)
class BreakevenResult:
    distance: int
    eps: float
    logical_lifetime_s: float
    physical_lifetime_s: float
    ratio: float
    ratio_se: float

    @property
    def beyond_breakeven(self) -> bool:
        return self.ratio > 1.0

    def __str__(self) -> str:
        return (f"d={self.distance}: logical lifetime {self.logical_lifetime_s * 1e6:.1f} us vs "
                f"physical {self.physical_lifetime_s * 1e6:.1f} us -> ratio "
                f"{self.ratio:.2f} +/- {self.ratio_se:.2f} "
                f"({'above' if self.beyond_breakeven else 'below'} breakeven)")


def breakeven(result: LogicalErrorRate, cycle_time: float = WILLOW_CYCLE_S,
              physical_lifetime: float = WILLOW_T1_S) -> BreakevenResult:
    """Logical lifetime over the lifetime of a physical qubit.

    Defaults are Willow's cycle time and mean T1. Google quotes 2.4x for d = 7
    against its best physical qubit (~119 us), not against the 68 us mean, so use
    physical_lifetime=WILLOW_BEST_PHYSICAL_S to compare with that number. Here the
    simulated p is not tied to any T1, so the ratio is a statement about the
    chosen cycle time and p, not about a device.
    """
    if physical_lifetime <= 0.0:
        raise ValueError("physical_lifetime must be positive")
    tau = logical_lifetime(result.eps, cycle_time)
    ratio = tau / physical_lifetime
    # tau ~ cycle_time / (2 eps) for small eps, so its relative error is that of eps.
    return BreakevenResult(result.distance, result.eps, tau, physical_lifetime, ratio,
                           ratio * result.eps_se / result.eps)


# ---------------------------------------------------------------------------
# Scan and report
# ---------------------------------------------------------------------------

def run_scan(distances: Sequence[int] = (3, 5, 7), p: float = DEFAULT_P,
             max_shots: int = DEFAULT_MAX_SHOTS, max_errors: int = DEFAULT_MAX_ERRORS,
             seed: int | None = 0, verbose: bool = True) -> dict:
    """Run every distance at one p. Returns results, pairwise Lambdas, the fit and breakeven."""
    ds = sorted(set(distances))
    results = []
    for d in ds:
        r = SurfaceCodeMemory(d, p).sample(max_shots=max_shots, max_errors=max_errors,
                                           seed=None if seed is None else seed + d)
        results.append(r)
        if verbose:
            print(r)

    lambdas = []
    for a, b in zip(results, results[1:]):
        if b.distance == a.distance + 2:
            try:
                lambdas.append(lambda_pairwise(a, b))
            except ValueError as exc:
                if verbose:
                    print(f"d={a.distance} -> {b.distance}: skipped ({exc})")
    fit = None
    try:
        fit = lambda_fit(results)
    except ValueError:
        pass

    top = [r for r in results if r.errors > 0]
    be = breakeven(top[-1]) if top else None

    if verbose:
        print()
        for lam in lambdas:
            print(lam)
        if fit:
            print(fit)
        print(f"Willow reference: Lambda = {WILLOW_LAMBDA} +/- {WILLOW_LAMBDA_SE}")
        if be:
            print(be)
            print(f"Willow d=7: {WILLOW_D7_RATIO}x against its best physical qubit")
    return {"results": results, "lambdas": lambdas, "fit": fit, "breakeven": be}


def _self_test() -> None:
    # Per-round conversion round trip.
    for eps in (1e-4, 3e-3, 0.05):
        for rounds in (1, 5, 25):
            p_l = 0.5 * (1.0 - (1.0 - 2.0 * eps) ** rounds)
            assert abs(per_round_error(p_l, rounds) - eps) < 1e-12
    assert per_round_error(0.5, 7) == 0.5

    # Noise-free memory never fails and every detector stays quiet.
    clean = SurfaceCodeMemory(3, 0.0)
    r0 = clean.sample(max_shots=2000, batch=1000, seed=1)
    assert r0.errors == 0 and r0.eps == 0.0
    assert clean.circuit.num_observables == 1

    # Rotated code: 2 d^2 - 1 qubits.
    assert SurfaceCodeMemory(5, 0.001).num_qubits == 2 * 25 - 1

    # Monotone in p at fixed d, and suppression with d below threshold.
    lo = SurfaceCodeMemory(3, 0.002).sample(20000, 10**9, seed=2).errors
    hi = SurfaceCodeMemory(3, 0.006).sample(20000, 10**9, seed=2).errors
    assert hi > lo
    scan = run_scan((3, 5), p=0.003, max_shots=100000, max_errors=100, verbose=False)
    r3, r5 = scan["results"]
    assert r5.eps < r3.eps, "no suppression with distance"
    assert scan["lambdas"][0].value > 1.0

    # Input checks.
    for bad in (lambda: SurfaceCodeMemory(4), lambda: SurfaceCodeMemory(1),
                lambda: SurfaceCodeMemory(3, -0.1), lambda: SurfaceCodeMemory(3, 1.5),
                lambda: logical_lifetime(0.0, 1e-6), lambda: per_round_error(0.1, 0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

    # Decoder plumbing.
    mem = SurfaceCodeMemory(3, 0.002, decoder="correlated")
    assert mem.decoder.name == "correlated"
    custom = make_decoder(MatchingDecoder(mem.detector_error_model), mem.detector_error_model)
    assert custom.name == "mwpm"
    for bad in (lambda: SurfaceCodeMemory(3, 0.001, decoder="nope"),
                lambda: SurfaceCodeMemory(3, 0.001, noise="nope"),
                lambda: make_decoder(object(), mem.detector_error_model),
                lambda: mem.compare(decoders=("mwpm",)),
                lambda: mem.compare(decoders=("mwpm", "mwpm"))):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

    # Negative control: with only reset/measurement flips there are no Y-type
    # mechanisms, so correlated matching must agree with plain MWPM shot for shot.
    ctrl = SurfaceCodeMemory(5, 0.01, noise="readout").compare(
        max_shots=20000, max_errors=10**9, batch=10000, seed=3)
    assert ctrl.only_reference["correlated"] == 0 and ctrl.only_other["correlated"] == 0

    # Under SD6 noise correlated matching beats plain MWPM at d = 5, on identical shots.
    cmp5 = SurfaceCodeMemory(5, 0.004).compare(max_shots=100000, max_errors=10**9, seed=4)
    red, se, z = cmp5.reduction("correlated")
    assert red > 0.15 and z > 4, (red, se, z)
    assert cmp5.rates["correlated"].shots == cmp5.rates["mwpm"].shots

    print("SurfaceCode: self-test passed")


if __name__ == "__main__":
    _self_test()
    print()
    run_scan()