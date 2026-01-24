import numpy as np
import quantum_sim_v3
import time
from gate_fusion import GateFuser, GATES

num_qubits = 20
num_iterations = 10

print("=== Gate Fusion Performance Benchmark ===\n")
print(f"System: {num_qubits} qubits")
print(f"Circuit: H-X-S-T repeated {num_iterations} times on each qubit\n")

# Build a circuit with lots of gates on same qubits
circuit = []
for iteration in range(num_iterations):
    for q in range(num_qubits):
        circuit.append((q, GATES['H']))
        circuit.append((q, GATES['X']))
        circuit.append((q, GATES['S']))
        circuit.append((q, GATES['T']))

print(f"Total gates before fusion: {len(circuit)}")

# Fuse the circuit
fused_circuit = GateFuser.fuse_circuit(circuit)
print(f"Total gates after fusion: {len(fused_circuit)}")
print(f"Reduction: {(1 - len(fused_circuit)/len(circuit))*100:.1f}%\n")

# Test 1: Without fusion (apply each gate separately)
print("Method 1: Without fusion (applying each gate)...")
state1 = np.zeros(2**num_qubits, dtype=np.complex128)
state1[0] = 1.0
sim1 = quantum_sim_v3.StateVector(state1)

start = time.time()
for target, gate in circuit:
    sim1.apply_gate_cache_optimized(target, gate)
time_unfused = time.time() - start

print(f"Time: {time_unfused*1000:.1f} ms\n")

# Test 2: With fusion (apply fused gates)
print("Method 2: With gate fusion...")
state2 = np.zeros(2**num_qubits, dtype=np.complex128)
state2[0] = 1.0
sim2 = quantum_sim_v3.StateVector(state2)

start = time.time()
for target, gate in fused_circuit:
    sim2.apply_gate_cache_optimized(target, gate)
time_fused = time.time() - start

print(f"Time: {time_fused*1000:.1f} ms\n")

# Results
print("="*60)
print(f"Without fusion: {time_unfused*1000:.1f} ms ({len(circuit)} gates)")
print(f"With fusion:    {time_fused*1000:.1f} ms ({len(fused_circuit)} gates)")
print(f"\nSpeedup: {time_unfused/time_fused:.2f}x")
print(f"Performance gain: {(1 - time_fused/time_unfused)*100:.1f}%")

# Verify results are the same
print(f"\nResults match: {np.allclose(state1, state2)}")
print(f"  state1[0] = {state1[0]}")
print(f"  state2[0] = {state2[0]}")
