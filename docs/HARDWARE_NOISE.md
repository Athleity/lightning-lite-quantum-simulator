# Hardware noise in Lightning-Lite 2.0

The noise module adds density-matrix simulation of superconducting-qubit decoherence to Lightning-Lite: relaxation, dephasing, Purcell decay through the readout resonator, and depolarizing gate error. Every closed form below is checked against the code by `benchmarks/t1_t2_decay.py`.

## 1. What is implemented

| Component | Location | Role |
|---|---|---|
| `KrausChannel` | `src/Noise.{h,cpp}` | CPTP map on m target qubits, applied to a density matrix |
| `DepolarizingChannel` | `src/Noise.{h,cpp}` | Symmetric Pauli noise on 1 or 2 qubits |
| `T1T2Model` | `src/Noise.{h,cpp}` | Amplitude damping plus pure dephasing from T1 and T2 |
| `PurcellModel` | `src/Noise.{h,cpp}` | Dispersive-limit Purcell rate with a scalar filter factor, and `combined_with` for intrinsic T1/T2 |
| `QubitParams`, `NoiseModel` | `python/noise_models.py` | Per-qubit T1, T2, optional Purcell channel, gate times, 1q and 2q depolarizing error |
| `DensityMatrixSimulator` | `python/noise_models.py` | Register of up to 10 qubits, gates and idle noise, populations, purity, Bloch vector, fidelity |

The bindings in `src/bindings_noise.cpp` build the Python module `quantum_sim_noise`. `lambda` is a Python keyword, so `KrausChannel.phase_damping` and `T1T2Model.lambda_` take the trailing underscore.

**Applying a channel.** No 2^n x 2^n operator is ever built. Split a basis index into the m target bits and the remaining n - m bits. For every pair (r1, r2) of non-target index groups, the entries rho[(r1, .), (r2, .)] form a 2^m x 2^m block B. The update is B -> sum_k K_k B K_k^dagger, using two small Eigen products per Kraus operator, followed by a scatter back into rho. There are 4^(n-m) independent blocks, so the loop runs under OpenMP. Cost is O(4^n 2^m K) for K Kraus operators. Gates go through the same routine, with a single unitary as the only Kraus operator.

Conventions: qubit q is bit q of the basis-state index (qubit 0 least significant). For a multi-qubit operator, bit j of the operator index acts on `targets[j]`, so `gate("CNOT", control, target)`.

## 2. Physics

### 2.1 Relaxation

Amplitude damping has Kraus operators

    K0 = [[1, 0], [0, sqrt(1 - gamma)]],    K1 = [[0, sqrt(gamma)], [0, 0]].

It maps rho11 to (1 - gamma) rho11 and rho01 to sqrt(1 - gamma) rho01. Requiring P1(t) = P1(0) exp(-t/T1) gives

    gamma = 1 - exp(-t / T1).

### 2.2 Dephasing

Phase damping, K0 = diag(1, sqrt(1 - lambda)) and K1 = diag(0, sqrt(lambda)), leaves populations alone and scales rho01 by sqrt(1 - lambda). The Ramsey time T2 combines both processes:

    1/T2 = 1/(2 T1) + 1/Tphi.

The coherence after both channels is sqrt(1 - gamma) sqrt(1 - lambda) = exp(-t/(2 T1)) sqrt(1 - lambda). Matching exp(-t/T2) requires sqrt(1 - lambda) = exp(-t/Tphi), so

    lambda = 1 - exp(-2 t / Tphi).

The factor 2 is there because lambda is a probability while Tphi is the decay time of the amplitude itself. Since 1/Tphi >= 0, T2 <= 2 T1, and the `T1T2Model` constructor throws for T2 above that bound. At T2 = 2 T1, Tphi is infinite.

### 2.3 Depolarizing error

On m = 1 or 2 qubits,

    rho' = (1 - p) rho + p / (4^m - 1) * sum over non-identity Paulis P of (P rho P).

Here p is the total probability of a non-identity Pauli error. Consequences, with d = 2^m:

- The process fidelity is 1 - p, and the average gate fidelity is F_avg = (d (1 - p) + 1) / (d + 1), which is 1 - 2p/3 for one qubit and 1 - 4p/5 for two.
- The one-qubit Bloch vector shrinks by 1 - 4p/3.
- The fully depolarizing point is p = (4^m - 1) / 4^m, or 3/4 and 15/16. Larger p is still a valid CPTP map.

`depolarizing_p_from_avg_fidelity` converts a benchmarked average fidelity into p.

### 2.4 Purcell decay

In the dispersive regime the qubit-like eigenstate carries a resonator-photon admixture of amplitude g / Delta, where Delta = omega_q - omega_r. A photon in the resonator leaks into the line at rate kappa, so the qubit inherits

    gamma_P = kappa * (g / Delta)^2.

The rate is linear in kappa and falls as 1/Delta^2. g, Delta and kappa are angular frequencies. `purcell_from_frequencies` takes cyclic values in Hz and multiplies by 2 pi, so for kappa/2pi in Hz, gamma_P = 2 pi kappa_Hz (g/Delta)^2.

### 2.5 Combining with intrinsic T1 and T2

Purcell decay is an energy-loss channel. It adds to 1/T1 and leaves the intrinsic pure dephasing time unchanged:

    Tphi     = 1 / (1/T2 - 1/(2 T1))        (intrinsic, fixed)
    1/T1_eff = 1/T1 + gamma_P
    1/T2_eff = 1/(2 T1_eff) + 1/Tphi

Holding T2 fixed while T1 drops would break T2 <= 2 T1_eff. Rebuilding T2_eff from the same Tphi satisfies the bound by construction, because 1/T2_eff >= 1/(2 T1_eff).

Worked example at g/2pi = 100 MHz, Delta/2pi = 1 GHz, kappa/2pi = 10 MHz, T1 = 100 us, T2 = 80 us (analytic values):

| Filter (dB) | gamma_P (1/s) | 1/gamma_P (us) | T1_eff (us) |
|---:|---:|---:|---:|
| 0 | 6.283e5 | 1.59 | 1.57 |
| 10 | 6.283e4 | 15.92 | 13.73 |
| 20 | 6.283e3 | 159.15 | 61.41 |
| 30 | 6.283e2 | 1591.55 | 94.09 |
| 40 | 6.283e1 | 15915.5 | 99.38 |

## 3. Dispersive-limit approximation and validity

`gamma_P = kappa (g/Delta)^2` is the leading term in g/Delta. For the two-level Jaynes-Cummings mixing angle, the photon admixture of the qubit-like state is

    sin^2(theta) = (1/2) (1 - 1 / sqrt(1 + 4 (g/Delta)^2)) = (g/Delta)^2 - 3 (g/Delta)^4 + ...

so the leading term overestimates by about 3 (g/Delta)^2 in relative terms: roughly 3% at g/Delta = 0.1, 11% at 0.2, and 41% at 0.5.

`PurcellModel` throws if g / |Delta| exceeds `MAX_DISPERSIVE_RATIO` = 0.5, which stops the model from returning a rate far outside its range. Values above about 0.2 should be read as qualitative. The model also leaves out higher transmon levels and anharmonicity corrections, counter-rotating terms, and any frequency dependence of kappa beyond the single filter factor below.

## 4. Pi-filter as a scalar suppression on kappa

A filter on the output line changes the impedance Z(omega) the resonator sees, which changes how strongly the line couples to the qubit at the qubit frequency. This module models that with one number:

    gamma_P = s * kappa * (g / Delta)^2,    s = 10^(-A / 10)  for A dB of power attenuation at omega_q,

with s in (0, 1] and s = 1 meaning no filter. `with_filter(s)` and `with_filter_db(A)` replace the factor, they do not multiply the existing one.

**This is an approximation.** It is a dispersive-limit scalar model. It does not compute Z(omega), does not derive s from circuit parameters, and does not model the filter's band edges or ripple. The user supplies A from a circuit simulation or a measurement. s enters only gamma_P, so the readout linewidth and dispersive shift are untouched.

Qarakal's first published work is on broadband Purcell protection with pi-filters. This module gives the simulator a place to put a filter's attenuation and shows what it does to T1_eff and T2_eff. It does not reproduce that paper's results.
<!-- Add a link to the Qarakal pi-filter paper here -->

## 5. Validation

`python benchmarks/t1_t2_decay.py` enforces the 13 checks below, prints a max error per check, and exits 0 when all pass. Parameters: T1 = 100 us, T2 = 80 us, g/2pi = 100 MHz, Delta/2pi = 1 GHz, kappa/2pi = 10 MHz. A check passes when its max error is at or below the tolerance.

| # | Check | Analytic reference | Tolerance |
|--:|---|---|---|
| 1 | T1 decay, P1(t) from the excited state, t in 0 to 300 us | exp(-t / T1) | 1e-12 abs |
| 2 | T2 coherence, rho01 magnitude from \|+>, t in 0 to 300 us | exp(-t / T2) | 1e-12 abs |
| 3 | Filter factor, 0 to 40 dB | 10^(-dB / 10) | 1e-12 rel |
| 4 | Purcell T1 limit vs filter attenuation | 1 / (s kappa (g/Delta)^2) | 1e-12 rel |
| 5 | Combined T1_eff vs filter attenuation | 1 / (1/T1 + gamma_P) | 1e-12 rel |
| 6 | Purcell rate vs detuning, 0 and 20 dB, Delta/2pi 0.25 to 3 GHz | s kappa (g/Delta)^2 | 1e-12 rel |
| 7 | Rate scaling exponent in Delta | log-log slope = -2 | 1e-9 abs |
| 8 | Combined T1_eff vs g/Delta, 0.01 to 0.40, 4 filters | 1 / (1/T1 + gamma_P) | 1e-12 rel |
| 9 | Combined T2_eff vs g/Delta, 4 filters | 1 / (1/(2 T1_eff) + 1/Tphi) | 1e-9 rel |
| 10 | Intrinsic Tphi preserved by `combined_with` | Tphi_eff / Tphi = 1 | 1e-9 |
| 11 | T2_eff <= 2 T1_eff | T2_eff / (2 T1_eff) - 1 <= 0 | 1e-9 |
| 12 | Simulator idle decay at 0 and 20 dB, 16 steps | exp(-t / T1_eff), exp(-t / T2_eff) | 1e-11 abs |
| 13 | Kraus trace preservation, T1/T2 and Purcell channels | sum of K^dagger K = I | exact |

The C++ unit tests in `src/Noise.cpp` cover the bit-block application against a dense 2^n x 2^n reference, the T1/T2 and depolarizing closed forms, the Purcell rate and its detuning and filter scaling, `combined_with`, and the invalid-input paths. Build and run:

    g++ -O2 -std=c++17 -fopenmp -DLL_TEST -I/usr/include/eigen3 src/Noise.cpp -o noise_test && ./noise_test

## 6. API examples

### Python

```python
import sys
sys.path[:0] = ["python", "build"]

import numpy as np
import noise_models as nm

T1, T2 = 100e-6, 80e-6

# 20 dB filter, parameters in cyclic Hz
purcell = nm.purcell_from_frequencies(g_hz=100e6, delta_hz=1e9, kappa_hz=10e6, filter_db=20.0)
print(purcell.rate(), purcell.t1_limit())

eff = purcell.combined_with(nm.T1T2Model(T1, T2))
print(eff.t1(), eff.t2(), eff.t_phi())

# Two-qubit Bell state with decoherence and gate error
noise = nm.NoiseModel.uniform(2, T1, T2, purcell=purcell, p1q=5e-4, p2q=5e-3)
sim = nm.DensityMatrixSimulator(2, noise)
sim.gate("H", 0)
sim.gate("CNOT", 0, 1)
sim.idle(20e-6)

bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
print(sim.fidelity_with_pure(bell), sim.purity(), sim.bloch_vector(0))
```

### C++

```cpp
#include <cmath>
#include <cstdio>

#include "Noise.h"

int main() {
    using namespace ll;
    const double two_pi = 6.283185307179586;

    PurcellModel purcell(two_pi * 100e6, two_pi * 1e9, two_pi * 10e6);
    T1T2Model eff = purcell.with_filter_db(20.0).combined_with(T1T2Model(100e-6, 80e-6));

    // 3-qubit GHZ state
    CVector psi = CVector::Zero(8);
    psi(0) = psi(7) = 1.0 / std::sqrt(2.0);
    CMatrix rho = density_from_statevector(psi);

    eff.channel(30e-6, 1).apply(rho, 3);                         // idle on qubit 1
    DepolarizingChannel(2e-3, 2).to_kraus({0, 2}).apply(rho, 3); // 2q gate error on qubits 0, 2

    std::printf("trace %.12f  valid %d\n", rho.trace().real(), is_valid_density_matrix(rho));
    return 0;
}
```

Build with `g++ -O2 -std=c++17 -fopenmp -I/usr/include/eigen3 -Isrc example.cpp src/Noise.cpp`.

## 7. Limitations

- **No Z(omega).** The filter is one scalar on kappa, not a frequency-dependent impedance (section 4).
- **No crosstalk.** Noise is independent per qubit. There is no ZZ coupling, no correlated noise, no leakage out of the computational subspace.
- **No Lindblad evolution.** Noise is applied as discrete channels, once per gate and once per idle window, after an ideal unitary. There is no master equation integrated during a pulse, and T1, T2 and gamma_P are constants.
- **No trajectories.** The simulator is density-matrix only and capped at 10 qubits. There is no Monte Carlo wavefunction method.
- **Zero-temperature relaxation only.** Amplitude damping goes to the ground state, so there is no thermal excitation and no readout error.

## 8. Next

1. **Pulse level (Tier 3):** the transmon Hamiltonian and its diagonalization, DRAG envelopes, and a Schrodinger solver, to compare DRAG and square-pulse leakage against an analytic three-level result.
2. **Quantum reservoir computing (Tier 4):** a reservoir with an echo-state readout, validated by memory capacity and a Mackey-Glass task.

## References

- A. A. Houck et al., "Controlling the spontaneous emission of a superconducting transmon qubit," Phys. Rev. Lett. 101, 080502 (2008).
- E. A. Sete, J. M. Martinis, A. N. Korotkov, "Quantum theory of a bandpass Purcell filter for qubit readout," Phys. Rev. A 92, 012325 (2015).
- M. A. Nielsen, "A simple formula for the average gate fidelity of a quantum dynamical operation," Phys. Lett. A 303, 249 (2002).