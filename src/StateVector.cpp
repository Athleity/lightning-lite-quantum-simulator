#include "StateVector.h"
#include <omp.h>

StateVector::StateVector(int n) {
    num_qubits = n;
    int size = 1 << n;
    state.resize(size);
    
    for (int i = 0; i < size; i++) {
        state[i] = std::complex<double>(0.0, 0.0);
    }
    
    state[0] = std::complex<double>(1.0, 0.0);
}

void StateVector::print() const {
    std::cout << "State vector (" << num_qubits << " qubits):\n";
    for (int i = 0; i < state.size(); i++) {
        std::cout << "  |" << i << ">: " << state[i] << "\n";
    }
}

int StateVector::getNumQubits() const {
    return num_qubits;
}

std::complex<double> StateVector::getAmplitude(int index) const {
    return state[index];
}

int StateVector::getSize() const {
    return state.size();
}

void StateVector::applyGate(int target_qubit, const std::complex<double> gate[4]) {
    int size = state.size();
    int target_bit = 1 << target_qubit;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i += 2 * target_bit) {
        for (int j = 0; j < target_bit; j++) {
            int idx0 = i + j;
            int idx1 = idx0 + target_bit;
            
            std::complex<double> a0 = state[idx0];
            std::complex<double> a1 = state[idx1];
            
            state[idx0] = gate[0] * a0 + gate[1] * a1;
            state[idx1] = gate[2] * a0 + gate[3] * a1;
        }
    }
}