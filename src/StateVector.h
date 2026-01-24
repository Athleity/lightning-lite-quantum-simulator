#ifndef STATEVECTOR_H
#define STATEVECTOR_H

#include <complex>
#include <vector>
#include <iostream>
#include <cmath>

class StateVector {
private:
    int num_qubits;
    std::vector<std::complex<double>> state;

public:
    StateVector(int n);
    void print() const;
    int getNumQubits() const;
    std::complex<double> getAmplitude(int index) const;
    int getSize() const;
    void applyGate(int target_qubit, const std::complex<double> gate[4]);
};

#endif