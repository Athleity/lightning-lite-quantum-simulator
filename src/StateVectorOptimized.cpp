#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <complex>
#include <omp.h>

namespace py = pybind11;

class StateVectorOptimized {
private:
    std::complex<double>* data;
    size_t size;
    int num_qubits;
    bool owns_memory;

public:
    StateVectorOptimized(int n) : num_qubits(n), owns_memory(true) {
        size = 1ULL << n;
        data = new std::complex<double>[size];
        data[0] = {1.0, 0.0};
        for (size_t i = 1; i < size; i++) {
            data[i] = {0.0, 0.0};
        }
    }
    
    StateVectorOptimized(py::array_t<std::complex<double>> numpy_array) 
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
    
    ~StateVectorOptimized() {
        if (owns_memory && data != nullptr) {
            delete[] data;
        }
    }
    
    // Cache-optimized gate application using bit manipulation
    void apply_gate_cache_optimized(int target, const std::complex<double> gate[4]) {
        const size_t k = static_cast<size_t>(target);
        const size_t stride = 1ULL << k;
        const size_t num_blocks = size >> (k + 1);
        const size_t block_size = 1ULL << (k + 1);
        
        #pragma omp parallel for schedule(static)
        for (size_t block = 0; block < num_blocks; block++) {
            size_t base_idx = block * block_size;
            
            // Process stride elements contiguously for better cache locality
            for (size_t j = 0; j < stride; j++) {
                size_t idx0 = base_idx + j;
                size_t idx1 = idx0 + stride;
                
                std::complex<double> a0 = data[idx0];
                std::complex<double> a1 = data[idx1];
                
                data[idx0] = gate[0] * a0 + gate[1] * a1;
                data[idx1] = gate[2] * a0 + gate[3] * a1;
            }
        }
    }
    
    // Original method for comparison
    void apply_gate_standard(int target, const std::complex<double> gate[4]) {
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

PYBIND11_MODULE(quantum_sim_v3, m) {
    m.doc() = "Cache-optimized quantum state vector simulator";
    
    py::class_<StateVectorOptimized>(m, "StateVector")
        .def(py::init<int>())
        .def(py::init<py::array_t<std::complex<double>>>())
        .def("apply_gate_cache_optimized", [](StateVectorOptimized &self, int target, py::list gate_list) {
            std::complex<double> gate[4];
            for (int i = 0; i < 4; i++) {
                gate[i] = gate_list[i].cast<std::complex<double>>();
            }
            self.apply_gate_cache_optimized(target, gate);
        })
        .def("apply_gate_standard", [](StateVectorOptimized &self, int target, py::list gate_list) {
            std::complex<double> gate[4];
            for (int i = 0; i < 4; i++) {
                gate[i] = gate_list[i].cast<std::complex<double>>();
            }
            self.apply_gate_standard(target, gate);
        })
        .def("get_numpy_view", &StateVectorOptimized::get_numpy_view)
        .def("get_num_qubits", &StateVectorOptimized::get_num_qubits)
        .def("get_size", &StateVectorOptimized::get_size);
}