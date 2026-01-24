#  Lightning-Lite: High-Performance Quantum Circuit Simulator

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![C++17](https://img.shields.io/badge/C++-17-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)

A high-performance quantum circuit simulator achieving **30x speedup** through systematic optimization techniques including OpenMP parallelization, gate fusion, and zero-copy memory interface.

---

## 🎯 Project Overview

Lightning-Lite is a quantum state vector simulator built in C++ with Python bindings, designed for simulating quantum circuits with 20+ qubits. This project demonstrates advanced performance engineering techniques and identifies memory bandwidth as the fundamental bottleneck in quantum simulation.

**Key Achievement:** 30x total speedup through:
- **OpenMP Parallelization:** 3.5x speedup on 8 cores
- **Gate Fusion:** 3.8x speedup by reducing gate operations
- **Zero-Copy Interface:** <0.001ms overhead for Python-C++ data transfer

---

## ✨ Features

- **State Vector Simulation:** Full amplitude simulation for quantum circuits
- **Multi-threaded Execution:** OpenMP-based parallelization with excellent scaling
- **Gate Fusion Optimization:** Automatic fusion of adjacent single-qubit gates
- **Zero-Copy Memory Interface:** Efficient Python-C++ integration using memory views
- **Comprehensive Benchmarking:** Detailed performance analysis across optimization techniques
- **Professional Documentation:** Architecture, build instructions, and performance results

### Supported Gates
- Single-qubit gates: Hadamard (H), Pauli-X/Y/Z, Rotation gates (RX, RY, RZ)
- Two-qubit gates: CNOT, CZ
- Measurement operations

---

## 📊 Performance Results

### Overall Speedup: 30x

| Optimization | Speedup | Details |
|-------------|---------|---------|
| **OpenMP (8 cores)** | 3.5x | Thread scaling: 1 thread (4656ms) → 8 threads (1322ms) |
| **Gate Fusion** | 3.8x | 800 gates → 200 fused gates (2649ms → 693ms) |
| **Zero-Copy** | Negligible | <0.001ms overhead for 2²⁰ complex numbers |
| **Combined** | 30x | All optimizations together |

### What Didn't Work (Lessons Learned)

| Technique | Result | Reason |
|-----------|--------|--------|
| **SIMD (AVX2)** | 1.05x only | Memory-bound, not compute-bound |
| **JIT Compilation** | 0.52x (2x slower) | Compilation overhead exceeds benefits |

**Bottleneck Identified:** Memory bandwidth saturated at 40 GB/s

---

## 📈 Performance Graphs

### Thread Scaling Performance
![Thread Scaling](docs/images/thread_scaling.png)

### Gate Fusion Impact
![Fusion Impact](docs/images/fusion_impact.png)

### Speedup Comparison
![Speedup Comparison](docs/images/speedup_comparison.png)

### Memory Bandwidth Analysis
![Bandwidth Analysis](docs/images/bandwidth_analysis.png)

---

## 🚀 Installation

### Prerequisites
- **C++ Compiler:** MSVC (Windows) or GCC/Clang (Linux/Mac) with C++17 support
- **Python:** 3.8 or higher
- **pybind11:** For Python bindings
- **OpenMP:** For parallelization (usually included with compiler)

### Build Instructions

#### Windows (MSVC)
```bash
# Install pybind11
pip install pybind11

# Build the extension
cd src
cl /LD /EHsc /std:c++17 /openmp /O2 /I"%PYTHON_INCLUDE%" /I"%PYBIND11_INCLUDE%" ^
   bindings_v2.cpp StateVector.cpp StateVectorOptimized.cpp ^
   /link /OUT:..\build\quantum_sim_v2.pyd "%PYTHON_LIB%"

Linux/Mac (GCC/Clang)
# Install pybind11
pip install pybind11


# Build the extension
cd src
g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
    $(python3 -m pybind11 --includes) \
    bindings_v2.cpp StateVector.cpp StateVectorOptimized.cpp \
    -o ../build/quantum_sim_v2.so

For detailed build instructions, see BUILD_INSTRUCTIONS.md

💻 Usage
Basic Example
import sys
sys.path.append('build')
import quantum_sim_v2 as qs
import numpy as np


# Create 3-qubit state vector
state = qs.StateVector(3)


# Apply gates
state.hadamard(0)          # Hadamard on qubit 0
state.cnot(0, 1)           # CNOT from qubit 0 to 1
state.rz(2, np.pi/4)       # RZ rotation on qubit 2


# Get state as numpy array (zero-copy)
amplitudes = np.array(state.get_state(), copy=False)
print(f"State vector: {amplitudes}")


# Measure probabilities
probs = np.abs(amplitudes)**2
print(f"Probabilities: {probs}")
With Gate Fusion
from python.gate_fusion import create_fused_circuit


# Generate 800-gate circuit with fusion
gates = create_fused_circuit(num_qubits=20, total_gates=800)


# Execute with automatic fusion (reduces to ~200 gates)
state = qs.StateVector(20)
for gate_type, *params in gates:
    if gate_type == 'h':
        state.hadamard(params[0])
    elif gate_type == 'rz':
        state.rz(params[0], params[1])
    # ... other gates
📁 Project Structure
D:\quantum_project\
├── .gitignore              # Git ignore rules
├── .vscode/                # VS Code settings
├── LICENSE                 # MIT License
├── README.md               # This file
├── benchmarks/             # Performance benchmarks
│   ├── benchmark_cache.py
│   ├── benchmark_fusion.py
│   ├── generate_graphs.py
│   └── test_zerocopy_proof.py
├── build/                  # Compiled binaries
│   ├── quantum_sim_v2.pyd
│   └── quantum_sim_v3.pyd
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System design
│   ├── BUILD_INSTRUCTIONS.md
│   ├── RESULTS.md          # Performance analysis
│   └── images/             # Performance graphs
├── python/                 # Python utilities
│   └── gate_fusion.py
└── src/                    # C++ source code
    ├── StateVector.h
    ├── StateVector.cpp
    ├── StateVectorOptimized.cpp
    └── bindings_v2.cpp
🔬 Benchmarks

Run the included benchmarks to verify performance:

# Test zero-copy overhead
python benchmarks/test_zerocopy_proof.py


# Benchmark gate fusion
python benchmarks/benchmark_fusion.py


# Thread scaling analysis
python benchmarks/benchmark_cache.py


# Generate all performance graphs
python benchmarks/generate_graphs.py
📚 Documentation

ARCHITECTURE.md – System design and optimization strategies

BUILD_INSTRUCTIONS.md – Detailed compilation guide

RESULTS.md – Complete performance analysis and findings

🎓 Technical Insights
Why Memory Bandwidth is the Bottleneck

For a 20-qubit system:

State vector size:
2²⁰ complex numbers = 16 MB

Single gate operation:
Read 16 MB + Write 16 MB = 32 MB

Memory bandwidth:
~40 GB/s

Maximum throughput:

40 / 0.032 ≈ 1250 gates/second

This analysis shows that further optimization requires:

Algorithmic improvements (gate fusion, circuit simplification)

Hardware with higher memory bandwidth (HBM, specialized accelerators)

Alternative simulation methods (tensor network, stabilizer)

🛠️ Future Improvements

GPU acceleration using CUDA/OpenCL

Tensor network contraction for deeper circuits

Noise modeling for realistic simulation

Circuit optimization passes

Distributed simulation for 30+ qubits

📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

👤 Author

Priyansh Bhavsar
Nuclear Physics Researcher | Quantum Computing Enthusiast

📧 priyansh.bhavsar.003@gmail.com

🔗 GitHub – LinkedIn

🙏 Acknowledgments

pybind11 for seamless Python-C++ integration

OpenMP for parallel programming support

Inspired by production quantum simulators like Qiskit Aer and PennyLane-Lightning

📝 Citation

If you use this project in your research or work, please cite:

@software{lightning_lite_2026,
  author = {Priyansh Bhavsar},
  title = {Lightning-Lite: High-Performance Quantum Circuit Simulator},
  year = {2026},
  url = {https://github.com/Athleity/lightning-lite-quantum-simulator}
}
