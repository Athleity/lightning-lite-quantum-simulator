#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <complex>
#include <vector>
#include <omp.h>

namespace py = pybind11;

class StateVectorZeroCopy {
private:
    std::complex<double>* data;
    size_t size;
    int num_qubits;
    bool owns_memory;

public:
    StateVectorZeroCopy(int n) : num_qubits(n), owns_memory(true) {
        size = 1ULL << n;
        data = new std::complex<double>[size];
        data[0] = {1.0, 0.0};
        for (size_t i = 1; i < size; i++) {
            data[i] = {0.0, 0.0};
        }
    }
    
    StateVectorZeroCopy(py::array_t<std::complex<double>> numpy_array) 
        : owns_memory(false) {
        py::buffer_info buf = numpy_array.request();
        
        if (buf.ndim != 1) {
            throw std::runtime_error("NumPy array must be 1-dimensional");
        }
        
        data = static_cast<std::complex<double>*>(buf.ptr);
        size = buf.shape[0];
        
        num_qubits = 0;
        size_t temp = size;
        while (temp > 1) {
            num_qubits++;
            temp >>= 1;
        }
        
        if ((1ULL << num_qubits) != size) {
            throw std::runtime_error("Array size must be power of 2");
        }
    }
    
    ~StateVectorZeroCopy() {
        if (owns_memory && data != nullptr) {
            delete[] data;
        }
    }
    
    void apply_gate_optimized(int target, const std::complex<double> gate[4]) {
        const size_t target_mask = 1ULL << target;
        
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            if ((i & target_mask) == 0) {
                size_t i0 = i;
                size_t i1 = i | target_mask;
                
                std::complex<double> a0 = data[i0];
                std::complex<double> a1 = data[i1];
                
                data[i0] = gate[0] * a0 + gate[1] * a1;
                data[i1] = gate[2] * a0 + gate[3] * a1;
            }
        }
    }
    
    py::array_t<std::complex<double>> get_numpy_view() {
        return py::array_t<std::complex<double>>(
            {static_cast<py::ssize_t>(size)},
            {sizeof(std::complex<double>)},
            data,
            py::cast(*this)
        );
    }
    
    int get_num_qubits() const { return num_qubits; }
    size_t get_size() const { return size; }
};

PYBIND11_MODULE(quantum_sim_v2, m) {
    m.doc() = "Zero-copy quantum state vector simulator";
    
    py::class_<StateVectorZeroCopy>(m, "StateVector")
        .def(py::init<int>())
        .def(py::init<py::array_t<std::complex<double>>>())
        .def("apply_gate", [](StateVectorZeroCopy &self, int target, py::list gate_list) {
            std::complex<double> gate[4];
            for (int i = 0; i < 4; i++) {
                gate[i] = gate_list[i].cast<std::complex<double>>();
            }
            self.apply_gate_optimized(target, gate);
        })
        .def("get_numpy_view", &StateVectorZeroCopy::get_numpy_view)
        .def("get_num_qubits", &StateVectorZeroCopy::get_num_qubits)
        .def("get_size", &StateVectorZeroCopy::get_size);
}