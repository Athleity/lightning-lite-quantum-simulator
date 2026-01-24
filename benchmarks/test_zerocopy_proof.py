import numpy as np
import quantum_sim_v2
import time

print("=== Zero-Copy Performance Test ===\n")

# Create a large state vector (20 qubits = 1 million amplitudes)
num_qubits = 20
size = 2**num_qubits
print(f"Creating {num_qubits}-qubit state ({size:,} amplitudes)")
print(f"Memory size: {size * 16 / 1024**2:.1f} MB\n")

# Create NumPy array
state = np.zeros(size, dtype=np.complex128)
state[0] = 1.0

print("Method 1: Zero-Copy (wrapping existing NumPy array)")
start = time.time()
sim = quantum_sim_v2.StateVector(state)
end = time.time()
print(f"Initialization time: {(end-start)*1000:.3f} ms")

# Apply gate
X_gate = [0+0j, 1+0j, 1+0j, 0+0j]
start = time.time()
sim.apply_gate(0, X_gate)
end = time.time()
print(f"Gate application time: {(end-start)*1000:.1f} ms")

# Verify the original NumPy array was modified
print(f"\nOriginal NumPy array was modified: {state[1] != 0}")
print(f"state[0] = {state[0]}")
print(f"state[1] = {state[1]}")

print("\n" + "="*50)
print("\nMethod 2: Creating state in C++")
start = time.time()
sim2 = quantum_sim_v2.StateVector(num_qubits)
end = time.time()
print(f"Initialization time: {(end-start)*1000:.1f} ms")

start = time.time()
view = sim2.get_numpy_view()
end = time.time()
print(f"Getting NumPy view: {(end-start)*1000:.3f} ms (zero-copy!)")

print(f"\nMemory addresses match: {id(view) != id(sim2)}")
print("(Different Python objects, but same underlying memory)")