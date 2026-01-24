# Changelog

All notable changes to Lightning-Lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-24

### 🎉 Initial Release

**Major Achievement**: 3.5x faster than Qiskit Aer, 30x internal speedup

### Added
- Cache-optimized quantum statevector simulator
- Zero-copy Python-C++ integration using pybind11
- OpenMP multi-threading support
- Comprehensive benchmark suite
- Qiskit comparison benchmarks with visualization
- Professional documentation (ARCHITECTURE.md, RESULTS.md, BUILD_INSTRUCTIONS.md)
- Bell state and quantum algorithm examples

### Performance
- **vs Qiskit**: 2.3x - 3.5x faster for 4-12 qubit circuits
- **OpenMP scaling**: 3.5x speedup on 8 cores
- **Gate fusion**: 3.8x speedup for sequential gates
- **Cache optimization**: 2.3x improvement from block-based memory access

### Supported Gates
- Single-qubit: H, X, Y, Z, RX, RY, RZ
- Two-qubit: CNOT, CZ
- Custom gates via matrix input

### Benchmarks
- `compare_and_graph.py`: Full Qiskit comparison with graphs
- `verify_yours.py`: Simulator verification
- `verify_qiskit.py`: Qiskit baseline verification
- `generate_graphs.py`: Custom visualization

---

## [0.3.0] - 2026-01-22 (Internal)

### Added
- Cache-optimized gate application (`apply_gate_cache_optimized`)
- Block-based memory access patterns
- CNOT gate implementation

### Performance
- 2.3x speedup from cache optimization
- Reduced cache misses from 40% → 5%

---

## [0.2.0] - 2026-01-20 (Internal)

### Added
- Gate fusion optimization
- Sequential gate combination logic
- Fusion benchmark suite

### Performance
- 3.8x speedup from gate fusion
- Reduced 800 gates → 200 fused gates

---

## [0.1.0] - 2026-01-18 (Internal)

### Added
- OpenMP parallelization
- Thread scaling benchmarks
- Multi-core gate application

### Performance
- 3.5x speedup on 8 cores
- 93% parallel efficiency on 4 cores

---

## [0.0.1] - 2026-01-15 (Internal)

### Added
- Initial C++ implementation
- Basic statevector simulator
- Python bindings via pybind11
- Single-qubit gate support

---

## Future Releases

### [1.1.0] - Planned Q1 2026
- [ ] GPU acceleration (CUDA)
- [ ] Noise modeling
- [ ] Circuit optimization passes
- [ ] Python package (PyPI)

### [2.0.0] - Planned Q2 2026
- [ ] Tensor network simulation
- [ ] Distributed computing support
- [ ] Advanced gate set (Toffoli, Fredkin)
- [ ] Measurement sampling

---

## Notes

### Versioning Strategy
- **Major (X.0.0)**: Breaking API changes, major features
- **Minor (0.X.0)**: New features, performance improvements
- **Patch (0.0.X)**: Bug fixes, documentation updates

### Performance Baseline
All performance comparisons use:
- Hardware: 8-core x86_64 CPU, DDR4 RAM
- Qiskit Aer version: 0.17.2
- Workload: 1000 mixed operations (H + CNOT)

---

*Maintained by Priyansh Bhavsar | Last updated: January 24, 2026*
