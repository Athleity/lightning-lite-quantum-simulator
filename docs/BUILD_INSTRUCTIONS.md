# Build Instructions for Lightning-Lite

## Prerequisites

- **C++ Compiler:** g++ with C++17 and OpenMP support
- **Python:** 3.8+ with development headers
- **Libraries:** pybind11, NumPy, Matplotlib

---

## Windows (MSYS2/MinGW)

### Install MSYS2

Download from https://www.msys2.org/ and install to `C:\msys64`

### Install Tools

Open MSYS2 MSYS terminal:
```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-python mingw-w64-x86_64-python-pip
```

### Install Python Packages
```bash
pip install pybind11 numpy matplotlib
```

### Compile the Project
```cmd
REM Zero-copy version
C:\msys64\mingw64\bin\g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC src/bindings_v2.cpp -o build/quantum_sim_v2.pyd -I C:\msys64\mingw64\include\python3.12 -I C:\msys64\mingw64\lib\python3.12\site-packages\pybind11\include -L C:\msys64\mingw64\lib -lpython3.12

REM Optimized version
C:\msys64\mingw64\bin\g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC src/StateVectorOptimized.cpp -o build/quantum_sim_v3.pyd -I C:\msys64\mingw64\include\python3.12 -I C:\msys64\mingw64\lib\python3.12\site-packages\pybind11\include -L C:\msys64\mingw64\lib -lpython3.12
```

---

## Linux (Ubuntu/Debian)

### Install Dependencies
```bash
sudo apt update
sudo apt install g++ cmake python3-dev python3-pip libomp-dev
pip3 install pybind11 numpy matplotlib
```

### Compile
```bash
g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC src/bindings_v2.cpp -o build/quantum_sim_v2.so $(python3 -m pybind11 --includes) $(python3-config --ldflags)

g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC src/StateVectorOptimized.cpp -o build/quantum_sim_v3.so $(python3 -m pybind11 --includes) $(python3-config --ldflags)
```

---

## macOS

### Install Homebrew & Tools
```bash
brew install gcc cmake python3
pip3 install pybind11 numpy matplotlib
```

### Compile
```bash
g++-13 -O3 -Wall -shared -std=c++17 -fopenmp -fPIC src/bindings_v2.cpp -o build/quantum_sim_v2.so $(python3 -m pybind11 --includes) $(python3-config --ldflags)
```

---

## Running Benchmarks

### Set Python Path

**Windows:**
```cmd
set PYTHONPATH=%CD%\build;%PYTHONPATH%
```

**Linux/Mac:**
```bash
export PYTHONPATH=$(pwd)/build:$PYTHONPATH
```

### Run Tests
```bash
cd benchmarks

# Zero-copy verification
python test_zerocopy_proof.py

# Gate fusion benchmark
python benchmark_fusion.py

# Generate graphs
python generate_graphs.py
```

---

## Verification

Test that everything works:
```python
import sys
sys.path.insert(0, 'build')
import numpy as np
import quantum_sim_v3

# Create 3-qubit state
state = np.zeros(8, dtype=np.complex128)
state[0] = 1.0

# Initialize simulator
sim = quantum_sim_v3.StateVector(state)

# Apply X gate
X_gate = [0+0j, 1+0j, 1+0j, 0+0j]
sim.apply_gate_cache_optimized(0, X_gate)

print("✓ Simulator works!")
print(state)
```

**Expected output:**
```
✓ Simulator works!
[0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
```

---

## Troubleshooting

### Module Not Found
```
ModuleNotFoundError: No module named 'quantum_sim_v3'
```

**Solution:** Add build/ to Python path:
```python
import sys
sys.path.insert(0, 'build')
```

### Compilation Errors

**"pybind11.h not found"**
```bash
pip install pybind11
```

**"Python.h not found"**
- **Windows:** `pacman -S mingw-w64-x86_64-python`
- **Linux:** `sudo apt install python3-dev`
- **Mac:** `brew install python3`

### Runtime Errors

**"DLL load failed" (Windows)**

Add MSYS2 to PATH:
```cmd
set PATH=C:\msys64\mingw64\bin;%PATH%
```

---

## Contact

For issues or questions, open an issue on GitHub.