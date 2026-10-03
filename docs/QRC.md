# Quantum Reservoir Computing

Time-multiplexed quantum reservoir with a ridge-regression readout, simulated exactly with density matrices. Implemented in `src/Reservoir.h` (single header), exposed through `src/bindings_qrc.cpp` as `quantum_sim_qrc`, and wrapped in `python/qrc.py`.

## 1. Overview

Reservoir computing trains only a linear readout on top of a fixed nonlinear dynamical system. Here the system is a small spin network (N qubits) driven by a scalar input stream. Following Fujii and Nakajima, the input is injected into one qubit, the network evolves unitarily, and observables are measured at V equally spaced instants inside each input interval (virtual nodes). Each input therefore produces `n_features = n_observables * V` features, from a Hilbert space of dimension 2^N, with no trainable quantum parameters.

```
s_t --> inject --> U (V sub-steps, measure after each) --> features f_t --> ridge readout --> y_t
```

## 2. Reservoir dynamics

Hamiltonian (transverse-field Ising with random all-to-all couplings):

```
H = sum_{i<j} J_ij X_i X_j + sum_i h_i Z_i
```

- `J_ij` drawn uniformly in `[-1, 1]` and scaled by `coupling`
- `h_i = field * (1 + field_disorder * u_i)` with `u_i` uniform in `[-1, 1]`
- `seed` fixes both, so a reservoir is reproducible

Per-step unitary, built once by diagonalising the real symmetric `H`:

```
U = exp(-i H tau / V)
```

**Replace-injection.** For each input `s in [0, 1]`, the input qubit is traced out and replaced by a pure state parameterised by `s`, so the rest of the register keeps its memory of earlier inputs while the input qubit carries only the new value. Then `rho -> U rho U^+` is applied V times, with the observables measured after each application.

**Virtual nodes.** The V measurements per input are taken at times `tau/V, 2 tau/V, ..., tau` after injection. They sample the transient at several times and multiply the feature count by V without adding qubits.

**Observables.** `Z_i` and `X_i` (N each), `Z_iZ_j` and `X_iX_j` (N(N-1)/2 each, i < j). All expectation values lie in [-1, 1].

## 3. Ridge readout

Features `F` (T x m) are centred, and the bias is not penalised:

```
w = (Fc^T Fc + lambda * tr(Fc^T Fc)/m * I)^-1 Fc^T yc
b = mean(y) - mean(F) . w
```

The ridge `lambda` is relative to the mean diagonal of the centred Gram matrix, so one value works across feature scales. The solve uses LDLT and raises `RuntimeError` if the Gram matrix is singular (collinear features with too small a ridge).

## 4. Memory capacity

Following Jaeger (2002), the reservoir is driven by i.i.d. `s_t ~ U[0, 1]`. For each delay k a separate readout is trained to reproduce `s_{t-k}`, and

```
MC_k     = squared correlation(y_hat_k, s_{t-k})  on held-out data
MC_total = sum_k MC_k
```

For a linear readout on m features, `MC_total <= m` in the infinite-data limit. In the code, `n_features` is reported as this bound; finite samples can push the estimate slightly above it, so the benchmark allows 0.5 of slack. `MC_total` depends on `max_delay`, `n_train` and `washout`. N=4, V=4 gives 8.25 at `max_delay=20` and 5.28 at `max_delay=5`.

## 5. Mackey-Glass benchmark

The delay differential equation

```
dx/dt = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x(t),   beta=0.2, gamma=0.1, n=10, tau=17
```

is integrated in the chaotic regime and sampled at `dt = 1`, after discarding a 200-sample transient. The series is min-max scaled to [0, 1], and the features at step t are regressed onto `x_{t+h}`. The baseline is persistence, `x_{t+h} = x_t`. Setup: 3000 samples, washout 200, 1500 training rows, ridge default.

## 6. Results

### Memory capacity, MC_total over (N, V)

Observable: `Z_i`. Seed 1, `max_delay=20`, `n_train=1000`, `n_test=500`.

| N \ V |  1  |  2  |  3  |  4  |  6   |  8   |
|-------|-----|-----|-----|-----|------|------|
| 1     | 1.1 | 1.1 | 1.1 | 1.1 | 1.1  | 1.1  |
| 2     | 1.1 | 1.2 | 1.2 | 1.2 | 1.2  | 1.2  |
| 3     | 2.4 | 4.7 | 5.8 | 6.8 | 7.4  | 7.5  |
| 4     | 3.5 | 6.1 | 7.2 | 8.4 | 9.1  | 9.2  |
| 5     | 3.5 | 6.4 | 7.9 | 8.7 | 9.3  | 9.4  |
| 6     | 4.4 | 6.8 | 8.3 | 9.2 | 10.1 | 10.3 |

Best configuration: N=6, V=8, MC_total = 10.25 against a bound of 48 features.

Observations:

- N=1 sits at about 1.1 because `H = hZ` commutes with the measured `Z`, so the qubit retains only the last input. N=2 adds almost nothing, and the jump appears at N=3.
- Capacity grows with V and saturates: from V=6 to V=8 the gain is at most 0.2. The sub-step measurements are strongly correlated, so MC_total reaches only a fraction of the N·V bound.
- Beyond N=3, additional qubits help less than additional virtual nodes at fixed cost.

Figures: `docs/images/qrc_mc_vs_qubits.png` (V=2 and V=4 curves with the N·V bound dotted) and `docs/images/qrc_mc_heatmap.png`.

### MC_k profile

N=4, V=4 (`max_delay=20`, `n_train=2000`, `n_test=1000`):

| k    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
|------|------|------|------|------|------|------|------|------|
| MC_k | 1.00 | 0.98 | 0.96 | 0.92 | 0.76 | 0.66 | 0.66 | 0.45 |

MC_k decays smoothly to about 0 by delay 20; MC_total = 8.25. The profile for the best configuration (N=6, V=8) is in `docs/images/qrc_features.png`.

### Mackey-Glass, one-step prediction (N=4, V=4)

| Metric                     | Value    |
|----------------------------|----------|
| NMSE train                 | 3.90e-4  |
| NMSE test                  | 3.78e-4  |
| NMSE persistence (test)    | 2.15e-2  |
| Improvement over baseline  | 56.9x    |

The reservoir beats persistence at every horizon h = 1..10. The per-horizon NMSE values are printed by the benchmark and plotted in `docs/images/qrc_mg_horizon.png`; the h=1 row is the table above. Test NMSE is close to train NMSE, so the readout is not overfitting.

Figures: `docs/images/qrc_mg_prediction.png` (target vs reservoir on held-out data, plus an NMSE bar chart) and `docs/images/qrc_mg_horizon.png`.

## 7. Validation

`src/Reservoir.h` carries its own test suite, built with `-DLL_TEST` (22 checks, all pass):

```
g++ -O2 -std=c++17 -DLL_TEST -I/usr/include/eigen3 -x c++ src/Reservoir.h -o build/reservoir_test
./build/reservoir_test
```

| Group | What is checked |
|-------|-----------------|
| 1 | N=1 memory capacity: MC_0 = 1, MC_k (k >= 1) at the finite-sample noise floor |
| 2 | Closed-form expectation values: N=1 `H = hZ`, N=2 `H = J XX`, injection and bit ordering with `H = 0` (to 1e-12) |
| 3 | Readout: recovery of known weights and bias, held-out error, collinear features, `predict_series` identity |
| 4 | Physical invariants over long runs: trace preservation, Hermiticity, positivity of rho, features in [-1, 1], unitarity of U, symmetry of H |
| 5 | Memory-capacity sanity: exceeds the single-qubit value, stays within the feature bound |
| 6 | Mackey-Glass: fixed point x* = 1 preserved, attractor inside [0.2, 1.8] |
| 7 | Every documented exception is raised (bad config, out-of-range input, unfitted readout, bad horizon) |

The Python benchmark adds 8 further checks, all passing (`benchmarks/qrc_mackey_glass.py`).

## 8. API examples

```python
import sys; sys.path.insert(0, "python")
import qrc

# Presets
cfg = qrc.default_4q()      # N=4, V=4, <Z_i>: 16 features
cfg = qrc.large_6q()        # N=6, V=4, <Z_i> + <Z_iZ_j>: 84 features

# Memory capacity
mc = qrc.run_memory_capacity(n_qubits=4, virtual_nodes=4, max_delay=20)
print(mc.total, mc.n_features, list(mc.mc)[:5])

# Mackey-Glass, 1-step
pr = qrc.run_mackey_glass(n_qubits=4, horizon=1)
print(pr.nmse_test, pr.nmse_persistence)

# Plots
qrc.plot_memory_capacity(mc)
qrc.plot_mackey_glass_prediction(pr)

# Lower level: drive a reservoir and fit a readout yourself
res = qrc.QuantumReservoir(qrc.make_config(n_qubits=3, virtual_nodes=4, observables=["Z", "ZZ"]))
F = res.run(qrc.scale_to_unit(qrc.mackey_glass(1500)))     # (T, n_features)
ro = qrc.LinearReadout(1e-8)
ro.fit(F[200:1200], target[200:1200])                     # target: your own series
y_hat = ro.predict(F[1200:])
```

Rebuild the extension with the command in the header of `src/bindings_qrc.cpp`. Full figure regeneration:

```
python3 benchmarks/qrc_mackey_glass.py               # N = 1..6, about 15-20 min
QRC_QUICK=1 python3 benchmarks/qrc_mackey_glass.py   # N = 1..4, short run
```

Quick mode overwrites the same PNG files, so commit the full-run figures first.

## 9. Limitations

- **Size.** `MAX_RESERVOIR_QUBITS = 8`. The state is a dense 2^N x 2^N density matrix and each step is a dense matrix product, so cost grows roughly as 8^N per step (N=6 is about 8x slower than N=5).
- **Exact expectations.** Features are exact expectation values with no shot noise. On hardware, finite sampling would add noise of order 1/sqrt(shots) to every feature, which would raise the achievable NMSE and reduce the effective memory capacity.
- **Unitary dynamics only.** No decoherence or dissipation. Real devices have T1/T2 loss (see `docs/HARDWARE_NOISE.md`), and coupling this reservoir to the Tier 1 noise channels is not done yet.
- **Single seed.** The reported grid uses one reservoir realisation. Seed-to-seed spread is not quantified in these tables.
- **No advantage claim.** This shows that a small quantum reservoir has a memory capacity and prediction accuracy that are well above the baseline. It does not show an advantage over a classical echo-state network of equal feature count; that comparison is not run here.

## 10. Next

- Tier 6: interview prep (12-slide deck, Q&A, `tests/test_integration.py`)
- Tier 2: algorithms (Grover, QAOA, VQE)
- Tier 5: Qiskit, PennyLane and OpenQASM integration