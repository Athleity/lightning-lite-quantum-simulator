import numpy as np
import quantum_sim_v3
import time

num_qubits = 20
num_gates = 100

print(f"=== Cache Optimization Benchmark ===")
print(f"System: {num_qubits} qubits, {num_gates} gates")
print(f"State vector size: {2**num_qubits:,} amplitudes\n")

X_gate = [0+0j, 1+0j, 1+0j, 0+0j]

# Test 1: Standard method
print("Testing standard method...")
state1 = np.zeros(2**num_qubits, dtype=np.complex128)
state1[0] = 1.0
sim1 = quantum_sim_v3.StateVector(state1)

start = time.time()
for i in range(num_gates):
    sim1.apply_gate_standard(i % num_qubits, X_gate)
time_standard = time.time() - start

print(f"Standard method: {time_standard*1000:.1f} ms\n")

# Test 2: Cache-optimized method
print("Testing cache-optimized method...")
state2 = np.zeros(2**num_qubits, dtype=np.complex128)
state2[0] = 1.0
sim2 = quantum_sim_v3.StateVector(state2)

start = time.time()
for i in range(num_gates):
    sim2.apply_gate_cache_optimized(i % num_qubits, X_gate)
time_optimized = time.time() - start

print(f"Cache-optimized method: {time_optimized*1000:.1f} ms\n")

# Results
print("="*50)
print(f"Standard:        {time_standard*1000:.1f} ms")
print(f"Cache-optimized: {time_optimized*1000:.1f} ms")
print(f"\nSpeedup: {time_standard/time_optimized:.2f}x")
print(f"Performance improvement: {(1 - time_optimized/time_standard)*100:.1f}%")
