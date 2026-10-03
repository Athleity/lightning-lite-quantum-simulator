# Backend Crossover Measurements

Grover's algorithm, single marked state, `iterations = round(pi/(4 theta))`.
Three backends behind one `ll::Backend` interface.

| n | Eigen (ms) | StateVec (ms) | Speedup | Winner |
|---|-----------|---------------|---------|--------|
| 10 | 1.2 | 5.7 | 0.22x | Eigen |
| 12 | 47.4 | 61.4 | 0.77x | Eigen |
| 14 | 389.0 | 310.5 | 1.25x | StateVec |
| 15 | 362.6 | 254.5 | 1.42x | StateVec |
| 16 | 1294.6 | 976.1 | 1.33x | StateVec |
| 17 | 3029.2 | 2257.1 | 1.34x | StateVec |
| 18 | 9340.9 | 6030.1 | 1.55x | StateVec |
| 20 | 87045.1 | 54518.6 | 1.60x | StateVec |

## Interpretation

- Crossover at n = 13-14, matching StateVectorBackend::PAR_MIN = 2^14.
- Below crossover: Eigen's SIMD wins (no thread startup cost).
- Above crossover: OpenMP + cache-blocked iteration dominates.
- At n = 20: 1.60x faster than Eigen.
- All three backends agree on Grover's peak probability to machine precision.
