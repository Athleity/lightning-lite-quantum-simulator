// Python bindings for the quantum reservoir computing module (Tier 4).
//
// Build:
//   g++ -O3 -Wall -shared -std=c++17 -fPIC -I/usr/include/eigen3 \
//       $(python3 -m pybind11 --includes) \
//       src/bindings_qrc.cpp \
//       -o build/quantum_sim_qrc$(python3-config --extension-suffix) \
//       $(python3-config --ldflags)

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>

#include "Reservoir.h"

namespace py = pybind11;
using namespace ll;

PYBIND11_MODULE(quantum_sim_qrc, m) {

    m.doc() = "Quantum reservoir computing: Fujii-Nakajima time-multiplexed reservoir with ridge readout";

    py::enum_<Observable>(m, "Observable")
        .value("Z", Observable::Z)    // <Z_i>, N features
        .value("X", Observable::X)    // <X_i>, N features
        .value("ZZ", Observable::ZZ)  // <Z_i Z_j>, i < j, N(N-1)/2 features
        .value("XX", Observable::XX); // <X_i X_j>, i < j, N(N-1)/2 features

    py::class_<ReservoirConfig>(m, "ReservoirConfig")
        .def(py::init<>())
        .def_readwrite("n_qubits", &ReservoirConfig::n_qubits)
        .def_readwrite("coupling", &ReservoirConfig::coupling)
        .def_readwrite("field", &ReservoirConfig::field)
        .def_readwrite("field_disorder", &ReservoirConfig::field_disorder)
        .def_readwrite("tau", &ReservoirConfig::tau)
        .def_readwrite("virtual_nodes", &ReservoirConfig::virtual_nodes)
        .def_readwrite("observables", &ReservoirConfig::observables)
        .def_readwrite("seed", &ReservoirConfig::seed)
        .def("__repr__", [](const ReservoirConfig& c) {
            std::ostringstream os;
            os << "ReservoirConfig(n_qubits=" << c.n_qubits << ", coupling=" << c.coupling
               << ", field=" << c.field << ", field_disorder=" << c.field_disorder
               << ", tau=" << c.tau << ", virtual_nodes=" << c.virtual_nodes
               << ", n_observable_types=" << c.observables.size() << ", seed=" << c.seed << ")";
            return os.str();
        });

    // Accessors return copies: the C++ object owns the matrices and may be reset or destroyed.
    constexpr auto copy = py::return_value_policy::copy;

    py::class_<QuantumReservoir>(m, "QuantumReservoir")
        .def(py::init<const ReservoirConfig&>(), py::arg("cfg"))
        .def("reset", &QuantumReservoir::reset)
        .def("step", &QuantumReservoir::step, py::arg("s"),
             "Inject s in [0, 1], evolve, return the n_features() features of this input.")
        .def("run", &QuantumReservoir::run, py::arg("inputs"),
             py::call_guard<py::gil_scoped_release>(),
             "Reset, then drive with `inputs`; returns a (T, n_features) matrix.")
        .def("measure", &QuantumReservoir::measure,
             "Expectation values of the configured observables in the current state.")
        .def("config", &QuantumReservoir::config, copy)
        .def_property_readonly("n_qubits", &QuantumReservoir::n_qubits)
        .def_property_readonly("dim", &QuantumReservoir::dim)
        .def_property_readonly("n_observables", &QuantumReservoir::n_observables)
        .def_property_readonly("n_features", &QuantumReservoir::n_features)
        .def_property_readonly("hamiltonian", &QuantumReservoir::hamiltonian, copy)
        .def_property_readonly("couplings", &QuantumReservoir::couplings, copy)
        .def_property_readonly("fields", &QuantumReservoir::fields, copy)
        .def_property_readonly("step_unitary", &QuantumReservoir::step_unitary, copy)
        .def_property_readonly("state", &QuantumReservoir::state, copy);

    py::class_<LinearReadout>(m, "LinearReadout")
        .def(py::init<double>(), py::arg("ridge") = DEFAULT_RIDGE)
        .def("fit", &LinearReadout::fit, py::arg("features"), py::arg("target"),
             py::call_guard<py::gil_scoped_release>())
        .def("predict", &LinearReadout::predict, py::arg("features"))
        .def_property_readonly("weights", &LinearReadout::weights, copy)
        .def_property_readonly("bias", &LinearReadout::bias)
        .def_property_readonly("ridge", &LinearReadout::ridge);

    py::class_<MemoryCapacityResult>(m, "MemoryCapacityResult")
        .def(py::init<>())
        .def_readonly("mc", &MemoryCapacityResult::mc)
        .def_readonly("total", &MemoryCapacityResult::total)
        .def_readonly("n_features", &MemoryCapacityResult::n_features);

    py::class_<PredictionResult>(m, "PredictionResult")
        .def(py::init<>())
        .def_readonly("nmse_train", &PredictionResult::nmse_train)
        .def_readonly("nmse_test", &PredictionResult::nmse_test)
        .def_readonly("nmse_persistence", &PredictionResult::nmse_persistence)
        .def_readonly("target_test", &PredictionResult::target_test)
        .def_readonly("prediction_test", &PredictionResult::prediction_test);

    m.def("nmse", &nmse, py::arg("pred"), py::arg("target"));
    m.def("squared_correlation", &squared_correlation, py::arg("a"), py::arg("b"));
    m.def("scale_to_unit", &scale_to_unit, py::arg("x"));

    m.def("memory_capacity", &memory_capacity,
          py::arg("res"), py::arg("max_delay"), py::arg("n_train"), py::arg("n_test"),
          py::arg("washout") = 100, py::arg("seed") = 12345, py::arg("ridge") = DEFAULT_RIDGE,
          py::call_guard<py::gil_scoped_release>());

    m.def("mackey_glass", &mackey_glass,
          py::arg("n"), py::arg("tau") = 17.0, py::arg("beta") = 0.2, py::arg("gamma") = 0.1,
          py::arg("exponent") = 10.0, py::arg("x0") = 1.2, py::arg("transient") = 200,
          py::arg("sample_dt") = 1.0, py::arg("substeps") = 10);

    m.def("predict_series", &predict_series,
          py::arg("res"), py::arg("series"), py::arg("horizon"), py::arg("washout"),
          py::arg("n_train"), py::arg("ridge") = DEFAULT_RIDGE,
          py::call_guard<py::gil_scoped_release>());
}