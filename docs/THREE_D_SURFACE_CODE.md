# 3D toric code

`src/ThreeDSurfaceCode.py` simulates a memory experiment on the 3D toric code: a periodic
d x d x d cubic lattice with a qubit on every edge. It encodes 3 logical qubits, has two error
sectors with different decoding problems, and is built on the same stim circuit and noise
machinery as the 2D surface code in `src/SurfaceCode.py`.

This is a standard 3D toric code memory. It is not a model of any specific 3D lattice-surgery
architecture (see [Scope](#scope-and-limitations)).

## Lattice and stabilizers

- Qubits: 3 d^3 edges.
- Vertex check A_v: product of X on the 6 edges at a vertex (d^3 checks).
- Face check B_f: product of Z on the 4 edges around a face (3 d^3 checks).
- A vertex and a face share 0 or 2 edges, so all checks commute.
- The checks are not independent (the 6 faces of a cube multiply to the identity), and the code
  encodes k = n - rank(A) - rank(B) = 3 logical qubits. The module verifies k = 3 by GF(2)
  elimination at d = 3, 4, 5.

## Excitations and logical operators

| Error | Flips | Defect | Logical operator |
|---|---|---|---|
| Z on an edge | the 2 vertex checks at its ends | point-like; a Z chain is a string whose endpoints are the defects | logical Z_k: string of Z along a non-contractible cycle in direction k, weight d |
| X on an edge | the 4 faces containing it | loop-like; an X membrane is a surface whose boundary is a loop of flipped faces | logical X_k: membrane of X over a non-contractible plane transverse to k, weight d^2 |

The string of direction k crosses the membrane of direction k once (they anticommute) and the
other two membranes zero times.

## Two sectors, two circuits

A memory experiment protects one error type at a time, so each sector has its own circuit.

| | Sector `"z"` | Sector `"x"` |
|---|---|---|
| Prepare | \|+> | \|0> |
| Checks measured each round | d^3 vertex (X) checks | 3 d^3 face (Z) checks |
| Data readout | X basis | Z basis |
| Errors that matter | Z | X |
| Observable k | X parity over the membrane of direction-k edges at coordinate 0 (weight d^2) | Z parity over the line of direction-k edges (weight d) |
| Decoder | MWPM (built in) | none built in: pass a `Decoder` object |

Detectors compare each check with the previous round (first round: its known value) and the
last round with the data readout. The other error type is in the circuit but flips no detector
or observable of its sector.

### Why sector x has no decoder here

In sector z every single fault flips at most two detectors (after `decompose_errors`), so the
syndrome is a (3+1)-dimensional matching problem and MWPM applies. In sector x one X error
flips four faces, a syndrome is a closed loop, and the optimal recovery is a minimum-weight
surface bounded by that loop. That is not a matching problem, the detector error model is not
graphlike, and `ThreeDSurfaceCode(..., sector="x")` with a string decoder raises `ValueError`.
Pass any object with `name` and `decode_batch` (for example a trained neural decoder), or
`TrivialDecoder` for the undecoded baseline. The redundancy among face checks, which makes
single-shot decoding possible for this sector, is not used.

## Noise models

`p` is applied at every listed location. Flips are Z-type in sector z and X-type in sector x,
because resets and measurements are in the matching basis.

| `noise` | Contents |
|---|---|
| `"sd6"` | gate depolarization, data idle depolarization, reset and measurement flips |
| `"phenomenological"` | flip on every data qubit each round plus a measurement flip, perfect gates |
| `"code_capacity"` | one round, one flip per data qubit, perfect measurement (`rounds` must be 1) |

`rounds` defaults to d (1 for `code_capacity`). Mid-round ancilla faults are in the SD6
circuit, but hook errors that matter do not occur in either sector, since the relevant Pauli of
an ancilla fault does not propagate to the data.

## API

```python
from ThreeDSurfaceCode import ThreeDSurfaceCode, TrivialDecoder, combined_p_logical

code = ThreeDSurfaceCode(distance=5, p=0.003, noise="sd6", sector="z", logicals=3)
rate = code.sample(50_000, seed=1)          # LogicalErrorRate
print(rate.p_logical, rate.eps, rate.eps_se)

# Both sectors (sector x needs a decoder object): union bound on a full-code failure
zr, xr = code.sample_both_sectors(10_000, seed=2, x_decoder=TrivialDecoder(3))
print(combined_p_logical(zr, xr))
```

- `distance >= 3` is the lattice size.
- `logicals` is 3 (all logical qubits) or 1 (direction 0 only). A shot fails if any scored
  observable is mispredicted.
- Properties: `num_data_qubits`, `num_ancillas`, `num_qubits`, `num_detectors`, `num_logicals`.
- `circuit_with_errors(edges)` returns the circuit with a deterministic Z (sector z) or X
  (sector x) error on each listed edge and no other noise. It is used by the self-test.
- Invalid arguments raise `ValueError`.

## Benchmark: `benchmarks/qec_3d.py`

Sector z, `noise="sd6"`, MWPM, all 3 logicals scored, d = 3, 5, 7, p from 0.1 % to 2 % on a log
grid (plus p = 0.3 %, where Lambda is evaluated with twice the shot budget).

```
python benchmarks/qec_3d.py            # full budget, long (d = 7 with SD6 is slow at low p)
python benchmarks/qec_3d.py --quick    # small budget, noisier
```

Outputs: the table `p | eps_d3 | eps_d5 | eps_d7 | ratio 3->5 | ratio 5->7`, threshold
crossings with a 16-84 % resampling interval, Lambda at p = 0.3 %, the 2D comparison,
`docs/images/qec_3d_threshold.png` and `docs/qec_3d_benchmark.json`. Exit status 0 means all
checks passed.

### Results

| Quantity | Value |
|---|---|
| Threshold, d = 3/5 crossing | 0.71 % [0.69, 0.72] |
| Threshold, d = 5/7 crossing | 0.61 % [0.60, 0.62] |
| Lambda(3 -> 5) at p = 0.3 % | 7.36 |
| Lambda(5 -> 7) at p = 0.3 % | 5.81 |

The intervals come from resampling each eps within its standard error. They do not include
finite-size drift. The d = 3/5 crossing sits above the d = 5/7 crossing, so the estimate is
still moving with d. Treat 0.61 % as an estimate for this lattice size, not the asymptotic
threshold.

Checks (all pass): eps rises with p at every d (whole-range test on points with failures and
p_logical < 0.5); eps_3 > eps_5 > eps_7 below threshold; both crossings found inside the scan
range; Lambda(5 -> 7) > 1 at p = 0.3 %.

### Comparison with the 2D surface code

| Code | Decoder | Threshold (circuit level, SD6) |
|---|---|---|
| 2D surface code | plain MWPM | 1.11 % |
| 2D surface code | correlated MWPM | 1.30 % |
| 3D toric code, sector z | MWPM | 0.61 % (d = 5/7), 0.71 % (d = 3/5) |

The 3D d = 5/7 threshold is about 0.55 times the 2D plain-MWPM value and about 0.47 times the
2D correlated value. Lambda here is measured at p = 0.3 %, while the 2D Lambda values in
`docs/QEC.md` are at p = 0.2 %, so the two sets of Lambda values are not directly comparable.

The literature code-capacity thresholds for the 3D toric code under optimal decoding are about
23 % (point-like sector) and about 3.3 % (loop-like sector). These numbers are as recalled and
should be checked before quoting. They are a different quantity from a circuit-level MWPM
threshold: code capacity has perfect measurement, optimal decoding is not matching, and SD6
noise adds gate and idle faults. Do not compare them to the 0.61 % above.

## Scope and limitations

- Sector x has no built-in decoder. The benchmark covers sector z only.
- The two sectors are simulated in separate circuits, so correlations between them (Y errors)
  are not modelled. `sample_both_sectors` reports a union bound.
- With 3 logicals scored together, the per-round eps assumes one logical bit and is
  approximate. Crossings are less affected, because every distance is converted the same way.
- The lattice is the torus, not a planar patch with boundaries.
- The module is a standard 3D toric code memory. The brief describes Qarakal's Pangaea as a 3D
  generalisation of lattice surgery. That description was taken from the brief and not checked,
  and this module does not model it.
- Under code-capacity noise with optimal decoding, sector z maps to the 3D random-bond Ising
  model and sector x to the 3D random plaquette gauge model (Dennis, Kitaev, Landahl, Preskill
  2002; Wang, Harrington, Preskill 2003). With MWPM, the code-capacity setting measures a
  matching threshold, which is below the optimal one.

## Validation

`python src/ThreeDSurfaceCode.py` runs the self-test:

- lattice algebra, conflict-free gate schedule, and k = 3 logical qubits at d = 3, 4, 5
- detector and qubit counts
- p = 0 is silent in both sectors
- a single Z error fires 2 detectors and a single X error fires 4
- strings and membranes are invisible to the checks and flip exactly their own observable
- stabilizers (a face, a vertex star) flip nothing
- failure rate rises with p; the same seed gives identical counts
- invalid inputs raise `ValueError`

The self-test also prints threshold scans for sector z (SD6 and code capacity). Those are
printed, not asserted.

## Files

| File | Role |
|---|---|
| `src/ThreeDSurfaceCode.py` | lattice, circuits, decoders, self-test |
| `benchmarks/qec_3d.py` | threshold sweep and 2D comparison |
| `docs/images/qec_3d_threshold.png` | eps vs p for d = 3, 5, 7 with the 2D thresholds marked |
| `docs/qec_3d_benchmark.json` | raw results, crossings, Lambda, checks |