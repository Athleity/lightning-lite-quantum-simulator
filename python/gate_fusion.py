import numpy as np

class GateFuser:
    """Pre-processor that fuses sequential gates on the same qubit"""
    
    @staticmethod
    def multiply_gates(gate1, gate2):
        """Multiply two 2x2 gate matrices"""
        g1 = np.array(gate1).reshape(2, 2)
        g2 = np.array(gate2).reshape(2, 2)
        result = g2 @ g1  # Note: reverse order for correct quantum operation
        return result.flatten().tolist()
    
    @staticmethod
    def fuse_circuit(gates):
        """
        gates: list of (target_qubit, gate_matrix) tuples
        returns: fused list with fewer operations
        """
        if not gates:
            return []
        
        fused = []
        current_target, current_gate = gates[0]
        
        for target, gate in gates[1:]:
            if target == current_target:
                # Same qubit - fuse the gates
                current_gate = GateFuser.multiply_gates(current_gate, gate)
            else:
                # Different qubit - save current and start new
                fused.append((current_target, current_gate))
                current_target = target
                current_gate = gate
        
        fused.append((current_target, current_gate))
        return fused
    
    @staticmethod
    def analyze_circuit(gates):
        """Show fusion statistics"""
        fused = GateFuser.fuse_circuit(gates)
        original_count = len(gates)
        fused_count = len(fused)
        reduction = (1 - fused_count / original_count) * 100
        
        print(f"Original gates: {original_count}")
        print(f"After fusion: {fused_count}")
        print(f"Reduction: {reduction:.1f}%")
        
        return fused

# Common gates
GATES = {
    'X': [0+0j, 1+0j, 1+0j, 0+0j],
    'H': [1/np.sqrt(2)+0j, 1/np.sqrt(2)+0j, 
          1/np.sqrt(2)+0j, -1/np.sqrt(2)+0j],
    'S': [1+0j, 0+0j, 0+0j, 0+1j],
    'T': [1+0j, 0+0j, 0+0j, np.exp(1j*np.pi/4)],
}

# Test
if __name__ == "__main__":
    print("=== Gate Fusion Demo ===\n")
    
    # Example circuit: H-X-H on qubit 0, then X on qubit 1
    circuit = [
        (0, GATES['H']),
        (0, GATES['X']),
        (0, GATES['H']),
        (1, GATES['X']),
        (0, GATES['S']),
        (0, GATES['T']),
    ]
    
    print("Original circuit:")
    for i, (target, gate) in enumerate(circuit):
        gate_name = [k for k, v in GATES.items() if v == gate][0]
        print(f"  Gate {i+1}: {gate_name} on qubit {target}")
    
    print()
    fused = GateFuser.analyze_circuit(circuit)
    
    print("\nFused circuit:")
    for i, (target, gate) in enumerate(fused):
        print(f"  Fused gate {i+1} on qubit {target}")