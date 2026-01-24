# Build Instructions for Lightning-Lite

Complete guide for building Lightning-Lite quantum simulator on Windows, Linux, and macOS.

---

## Prerequisites

### Required
- **C++ Compiler**: GCC 8+, Clang 7+, or MSVC 2019+ with C++17 support
- **Python**: 3.8 or higher with development headers
- **pybind11**: Python-C++ binding library
- **OpenMP**: For parallelization (usually included with compiler)

### Optional (for benchmarking)
- **NumPy**: Array operations
- **Matplotlib**: Visualization
- **Qiskit + Qiskit-Aer**: Comparison benchmarks

---

## Installation

### 1. Install Python Dependencies

```bash
# Core dependencies
pip install numpy pybind11

# Optional: For benchmarking
pip install matplotlib qiskit qiskit-aer
```

---

## 2. Build Instructions by Platform

### Windows (MSYS2/MinGW) - Recommended

#### Step 1: Install MSYS2
Download from [https://www.msys2.org/](https://www.msys2.org/) and install to `C:\msys64`

#### Step 2: Install Build Tools
Open MSYS2 MSYS terminal:
```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-python
```

#### Step 3: Find Your Python Paths
Open your **Anaconda/Miniconda prompt** (not MSYS2):
```bash
# Find Python executable
python -c "import sys; print(sys.executable)"

# Find include directory
python -c "import sysconfig; print(sysconfig.get_paths()['include'])"

# Find pybind11 include
python -c "import pybind11; print(pybind11.get_include())"

# Check for library
dir C:\Users\YOUR_USER\miniconda3\envs\py312\libs\python312.lib
```

#### Step 4: Build
Replace paths with your actual paths from Step 3:

```cmd
cd D:\quantum_project

REM Build v2 (zero-copy version)
C:\msys64\mingw64\bin\g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC ^
    src/bindings_v2.cpp -o build/quantum_sim_v2.pyd ^
    -I "C:\Users\YOUR_USER\miniconda3\envs\py312\Include" ^
    -I "C:\Users\YOUR_USER\miniconda3\envs\py312\Lib\site-packages\pybind11\include" ^
    -L "C:\Users\YOUR_USER\miniconda3\envs\py312\libs" -lpython312

REM Build v3 (optimized version)
C:\msys64\mingw64\bin\g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC ^
    src/StateVectorOptimized.cpp -o build/quantum_sim_v3.pyd ^
    -I "C:\Users\YOUR_USER\miniconda3\envs\py312\Include" ^
    -I "C:\Users\YOUR_USER\miniconda3\envs\py312\Lib\site-packages\pybind11\include" ^
    -L "C:\Users\YOUR_USER\miniconda3\envs\py312\libs" -lpython312
```

#### Step 5: Verify
```cmd
cd build
python -c "import quantum_sim_v3; print('✓ Success!')"
```

---

### Linux (Ubuntu/Debian)

#### Step 1: Install Dependencies
```bash
sudo apt update
sudo apt install g++ python3-dev python3-pip libomp-dev
pip3 install pybind11 numpy matplotlib
```

#### Step 2: Build
```bash
cd quantum_project

# Build v2
g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
    src/bindings_v2.cpp -o build/quantum_sim_v2.so \
    $(python3 -m pybind11 --includes) $(python3-config --ldflags)

# Build v3
g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
    src/StateVectorOptimized.cpp -o build/quantum_sim_v3.so \
    $(python3 -m pybind11 --includes) $(python3-config --ldflags)
```

#### Step 3: Verify
```bash
cd build
python3 -c "import quantum_sim_v3; print('✓ Success!')"
```

---

### macOS

#### Step 1: Install Homebrew & Tools
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install GCC with OpenMP support
brew install gcc cmake python3
pip3 install pybind11 numpy matplotlib
```

#### Step 2: Build
```bash
cd quantum_project

# Build v2 (use g++-13 or your GCC version)
g++-13 -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
    src/bindings_v2.cpp -o build/quantum_sim_v2.so \
    $(python3 -m pybind11 --includes) $(python3-config --ldflags)

# Build v3
g++-13 -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
    src/StateVectorOptimized.cpp -o build/quantum_sim_v3.so \
    $(python3 -m pybind11 --includes) $(python3-config --ldflags)
```

#### Step 3: Verify
```bash
cd build
python3 -c "import quantum_sim_v3; print('✓ Success!')"
```

---

## 3. Verification & Testing

### Quick Test
```python
import sys
sys.path.insert(0, 'build')
import numpy as np
from quantum_sim_v3 import StateVector

# Create 2-qubit state
state = np.zeros(4, dtype=np.complex128)
state[0] = 1.0
sim = StateVector(state)

# Apply Hadamard
H = [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2)]
sim.apply_gate_cache_optimized(0, H)

print("✓ Simulator works!")
print(sim.get_numpy_view())
```

**Expected output:**
```
✓ Simulator works!
[0.707+0.j 0.707+0.j 0.   +0.j 0.   +0.j]
```

### Run Full Benchmark Suite
```bash
cd benchmarks

# Compare vs Qiskit
python compare_and_graph.py

# Expected: 3.5x faster than Qiskit for 4-10 qubits
```

---

## 4. Troubleshooting

### Error: "Module Not Found"
**Solution**: Add build directory to Python path
```python
import sys
sys.path.insert(0, 'build')
```

### Error: "pybind11.h not found"
**Solution**: Install pybind11
```bash
pip install pybind11
```

### Error: "Python.h not found"
**Solution**: Install Python development headers
- **Windows**: Already included in conda/miniconda
- **Linux**: `sudo apt install python3-dev`
- **macOS**: `brew install python3`

### Error: "DLL load failed" (Windows)
**Solution 1**: Run from build directory
```cmd
cd build
python -c "import quantum_sim_v3; print('Success')"
```

**Solution 2**: Add MSYS2 to PATH
```cmd
set PATH=C:\msys64\mingw64\bin;%PATH%
```

### Error: "Import hangs" (Windows)
**Cause**: Python version mismatch between compiler and runtime

**Solution**: Rebuild with YOUR Python paths (see Step 3 above)

### Error: "Access Violation / Segfault"
**Cause**: Module name mismatch between .cpp and .pyd filename

**Check**: 
- `bindings_v2.cpp` has `PYBIND11_MODULE(quantum_sim_v2, m)`
- `StateVectorOptimized.cpp` has `PYBIND11_MODULE(quantum_sim_v3, m)`
- Filenames match: `quantum_sim_v2.pyd` and `quantum_sim_v3.pyd`

---

## 5. Build Optimization Flags

### Debug Build
```bash
g++ -O0 -g -Wall -shared -std=c++17 -fopenmp -fPIC ...
```

### Release Build (Default)
```bash
g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC ...
```

### Maximum Optimization
```bash
g++ -O3 -march=native -mtune=native -Wall -shared -std=c++17 -fopenmp -fPIC ...
```

**Warning**: `-march=native` makes binary non-portable

---

## 6. CMake Build (Alternative)

### Create CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(lightning_lite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(quantum_sim_v3 src/StateVectorOptimized.cpp)
```

### Build
```bash
mkdir build && cd build
cmake ..
make
```

---

## 7. Performance Tuning

### OpenMP Thread Count
```bash
# Set number of threads
export OMP_NUM_THREADS=8

# Run benchmark
python benchmarks/compare_and_graph.py
```

### CPU Affinity (Linux)
```bash
# Pin to physical cores
taskset -c 0-7 python benchmarks/compare_and_graph.py
```

---

## Contact

For build issues, open an issue on GitHub or contact:
📧 priyansh.bhavsar.003@gmail.com

---

*Last updated: January 24, 2026*
