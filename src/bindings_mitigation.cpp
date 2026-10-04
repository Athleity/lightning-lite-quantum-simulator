// src/bindings_mitigation.cpp
//
// Python bindings for error mitigation (module: quantum_sim_mitigation).
//
// DensityNoiseBackend and zne_exact are defined in src/Mitigation.cpp, which has no header.
// This file includes that .cpp directly so the module builds from the single command below
// (Mitigation.cpp must NOT also be added to the link line, and LL_TEST must not be defined).
//
// Build:
//   g++ -O2 -std=c++17 -fopenmp -I/usr/include/eigen3 -c src/Noise.cpp -o build/Noise.o
//   g++ -O3 -shared -std=c++17 -fopenmp -fPIC -I/usr/include/eigen3 \
//       $(python3 -m pybind11 --includes) \
//       src/bindings_mitigation.cpp build/Noise.o \
//       -o build/quantum_sim_mitigation$(python3-config --extension-suffix) \
//       $(python3-config --ldflags)
//
// Callables. circuit_fn(backend) and observable_fn(backend) are Python callables. They are
// wrapped by hand instead of through pybind11/functional.h: a C++ reference argument passed
// to a Python callable is copied by pybind11 (and Backend is abstract), so the backend is
// passed by pointer, which pybind11 hands to Python as a non-owning reference. The backend
// is valid only for the duration of the call; do not store it. Callables run with the GIL
// held (extrapolate does not release it), so Python exceptions propagate cleanly and the
// scale guards in ZNE / zne_exact restore the noise scale.
//
// Lifetime. NoisyBackend keeps a reference to its inner backend; the binding ties the inner
// object's lifetime to the NoisyBackend (keep_alive).

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "Mitigation.cpp"  // DensityNoiseBackend, zne_exact (see note above)

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

using namespace ll;

py::array_t<double> to_array(const std::vector<double>& v) {
    return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
}

py::array_t<std::complex<double>> to_array(const std::vector<std::complex<double>>& v) {
    return py::array_t<std::complex<double>>(static_cast<py::ssize_t>(v.size()), v.data());
}

CircuitFn to_circuit(const py::object& f) {
    if (!PyCallable_Check(f.ptr())) throw py::type_error("circuit_fn must be callable");
    py::function fn = py::reinterpret_borrow<py::function>(f);
    return [fn](Backend& b) { fn(&b); };
}

ObservableFn to_observable(const py::object& f) {
    if (!PyCallable_Check(f.ptr())) throw py::type_error("observable_fn must be callable");
    py::function fn = py::reinterpret_borrow<py::function>(f);
    return [fn](const Backend& b) -> double { return py::cast<double>(fn(&b)); };
}

std::string result_repr(const ZNEResult& r) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "ZNEResult(value=%.10g, std_error=%.3g, n_scales=%zu, trajectories=%d)",
                  r.value, r.std_error, r.scales.size(), r.trajectories);
    return buf;
}

}  // namespace

PYBIND11_MODULE(quantum_sim_mitigation, m) {
    m.doc() = "Lightning-Lite error mitigation: zero-noise extrapolation by gate folding, "
              "sampled (NoisyBackend) and exact density-matrix (DensityNoiseBackend)";

    // ---- constants -----------------------------------------------------------------------
    m.attr("MAX_ZNE_SCALES") = MAX_ZNE_SCALES;
    m.attr("MAX_ZNE_SCALE") = MAX_ZNE_SCALE;
    m.attr("DEFAULT_ZNE_TRAJECTORIES") = DEFAULT_ZNE_TRAJECTORIES;
    m.attr("MAX_DM_QUBITS") = MAX_DM_QUBITS;

    // ---- Backend (abstract base; gate methods dispatch virtually) --------------------------
    py::class_<Backend>(m, "Backend", "Gate-level backend interface. Little-endian qubit order.")
        .def("reset", &Backend::reset, "n_qubits"_a, "Allocate n_qubits and set |0...0>.")
        .def_property_readonly("n_qubits", &Backend::n_qubits)
        .def("h", &Backend::h, "q"_a)
        .def("x", &Backend::x, "q"_a)
        .def("rx", &Backend::rx, "q"_a, "theta"_a)
        .def("ry", &Backend::ry, "q"_a, "theta"_a)
        .def("rz", &Backend::rz, "q"_a, "theta"_a)
        .def("cnot", &Backend::cnot, "control"_a, "target"_a)
        .def("mcz", &Backend::mcz, "qubits"_a)
        .def("probabilities", [](const Backend& b) { return to_array(b.probabilities()); })
        .def("state", [](const Backend& b) { return to_array(b.state()); },
             "Amplitudes (pure-state backends only; DensityNoiseBackend raises RuntimeError).");

    py::class_<ReferenceBackend, Backend>(m, "ReferenceBackend", "Dense state-vector reference backend.")
        .def(py::init<int>(), "n_qubits"_a = 1);

    // ---- NoisyBackend ------------------------------------------------------------------------
    py::class_<NoisyBackend, Backend>(
        m, "NoisyBackend",
        "Wraps a Backend; folds each gate U -> U (U^+ U)^k and applies sampled depolarizing noise "
        "(total Pauli probability p) after every application.")
        .def(py::init<Backend&, double, std::uint64_t>(), "inner"_a, "p"_a, "seed"_a = 1,
             py::keep_alive<1, 2>())
        .def_property_readonly("probability", &NoisyBackend::probability)
        .def_property_readonly("inner", [](NoisyBackend& b) -> Backend& { return b.inner(); },
                               py::return_value_policy::reference_internal)
        .def("reseed", &NoisyBackend::reseed, "seed"_a)
        .def("set_scale", &NoisyBackend::set_scale, "scale"_a, "Odd integer noise scale 2k + 1.")
        .def_property_readonly("scale", &NoisyBackend::scale)
        .def_property_readonly("logical_gates", &NoisyBackend::logical_gates)
        .def_property_readonly("physical_gates", &NoisyBackend::physical_gates);

    // ---- DensityNoiseBackend ----------------------------------------------------------------------
    py::class_<DensityNoiseBackend, Backend>(
        m, "DensityNoiseBackend",
        "Exact density-matrix counterpart of NoisyBackend (Kraus channels, no sampling noise).")
        .def(py::init<double, int>(), "p"_a, "n_qubits"_a = 1)
        .def_property_readonly("probability", &DensityNoiseBackend::probability)
        .def("density", [](const DensityNoiseBackend& b) { return CMatrix(b.density()); })
        .def("set_scale", &DensityNoiseBackend::set_scale, "scale"_a)
        .def_property_readonly("scale", &DensityNoiseBackend::scale)
        .def_property_readonly("logical_gates", &DensityNoiseBackend::logical_gates)
        .def_property_readonly("physical_gates", &DensityNoiseBackend::physical_gates);

    // ---- ZNEResult -------------------------------------------------------------------------------------
    py::class_<ZNEResult>(m, "ZNEResult")
        .def(py::init<>())
        .def_readonly("value", &ZNEResult::value)
        .def_readonly("std_error", &ZNEResult::std_error)
        .def_readonly("scales", &ZNEResult::scales)
        .def_readonly("values", &ZNEResult::values)
        .def_readonly("std_errors", &ZNEResult::std_errors)
        .def_readonly("weights", &ZNEResult::weights)
        .def_readonly("trajectories", &ZNEResult::trajectories)
        .def("__repr__", &result_repr);

    // ---- ZNE ------------------------------------------------------------------------------------------------
    py::class_<ZNE>(m, "ZNE", "Zero-noise extrapolation with Richardson weights over folded noise scales.")
        .def(py::init<int>(), "trajectories"_a = DEFAULT_ZNE_TRAJECTORIES)
        .def_property_readonly("trajectories", &ZNE::trajectories)
        .def_static("default_scales", &ZNE::default_scales)
        .def_static("validate_scales", &ZNE::validate_scales, "scales"_a)
        .def_static("richardson_weights", &ZNE::richardson_weights, "scales"_a)
        .def_static("richardson", &ZNE::richardson, "scales"_a, "values"_a)
        .def(
            "extrapolate",
            [](const ZNE& z, NoisyBackend& noisy, const py::object& circuit_fn,
               const py::object& observable_fn, const std::vector<double>& scales) {
                return z.extrapolate(noisy, to_circuit(circuit_fn), to_observable(observable_fn), scales);
            },
            "noisy"_a, "circuit_fn"_a, "observable_fn"_a, "scales"_a = ZNE::default_scales(),
            "circuit_fn(backend) applies gates; observable_fn(backend) returns a float. The backend "
            "is reset to |0...0> (at its current size) before every trajectory.");

    // ---- zne_exact --------------------------------------------------------------------------------------------------
    m.def(
        "zne_exact",
        [](DensityNoiseBackend& b, const py::object& circuit_fn, const py::object& observable_fn,
           const std::vector<double>& scales) {
            return zne_exact(b, to_circuit(circuit_fn), to_observable(observable_fn), scales);
        },
        "backend"_a, "circuit_fn"_a, "observable_fn"_a, "scales"_a = ZNE::default_scales(),
        "Exact ZNE on the density matrix: no statistical error, isolates the extrapolation bias.");
}