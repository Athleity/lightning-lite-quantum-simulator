// Python bindings for the gate-level algorithms (Tier 2).
//
// Build:
//   g++ -O3 -shared -std=c++17 -fopenmp -march=native -fPIC -I/usr/include/eigen3 \
//       $(python3 -m pybind11 --includes) \
//       src/bindings_algorithms.cpp \
//       -o build/quantum_sim_algorithms$(python3-config --extension-suffix) \
//       $(python3-config --ldflags)

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>

#include "Algorithms.h"
#include "EigenBackend.h"
#include "StateVectorBackend.h"

namespace py = pybind11;
using namespace ll;

namespace {

Eigen::VectorXd to_vec(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
}

}  // namespace

PYBIND11_MODULE(quantum_sim_algorithms, m) {
    m.doc() = "Grover, QAOA and VQE on interchangeable gate-level backends";

    // ---- Backends ---------------------------------------------------------
    // Abstract: instances come from the factories below. Accessors return numpy copies.
    py::class_<Backend>(m, "Backend")
        .def("n_qubits", &Backend::n_qubits)
        .def("probabilities", [](const Backend& b) { return to_vec(b.probabilities()); },
             "|amplitude|^2 of every basis state, qubit q = bit q of the index.")
        .def("state", [](const Backend& b) {
            const auto s = b.state();
            return Eigen::VectorXcd(Eigen::Map<const Eigen::VectorXcd>(s.data(), static_cast<Eigen::Index>(s.size())));
        }, "Complex amplitudes, size 2^n.");

    m.def("make_reference_backend",
          [](int n_qubits) { return std::unique_ptr<Backend>(new ReferenceBackend(n_qubits)); },
          py::arg("n_qubits"), "Scalar loop reference, up to 26 qubits.");
    m.def("make_eigen_backend",
          [](int n_qubits) { return std::unique_ptr<Backend>(new EigenBackend(n_qubits)); },
          py::arg("n_qubits"), "Eigen-vectorised, up to 24 qubits.");
    m.def("make_statevector_backend",
          [](int n_qubits) { return std::unique_ptr<Backend>(new StateVectorBackend(n_qubits)); },
          py::arg("n_qubits"), "OpenMP block-iteration kernel, up to 25 qubits.");

    // ---- Grover -----------------------------------------------------------
    py::class_<GroverResult>(m, "GroverResult")
        .def_property_readonly("probabilities", [](const GroverResult& r) { return to_vec(r.probabilities); })
        .def_readonly("marked_index", &GroverResult::marked_index)
        .def_readonly("iterations", &GroverResult::iterations);

    py::class_<Grover>(m, "Grover")
        .def_static("run", &Grover::run,
                    py::arg("backend"), py::arg("n_qubits"), py::arg("marked_states"),
                    py::arg("iterations") = -1,
                    py::call_guard<py::gil_scoped_release>(),
                    "iterations = -1 picks round(pi/(4 theta) - 1/2), theta = asin sqrt(M/N).");

    // ---- QAOA -------------------------------------------------------------
    // `edges` returns a copy; use add_edge() or the constructor to build a graph.
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def(py::init([](int n_nodes, std::vector<std::tuple<int, int, double>> edges) {
                 Graph g;
                 g.n_nodes = n_nodes;
                 g.edges = std::move(edges);
                 return g;
             }),
             py::arg("n_nodes"), py::arg("edges"))
        .def_readwrite("n_nodes", &Graph::n_nodes)
        .def_readwrite("edges", &Graph::edges)
        .def("add_edge", [](Graph& g, int i, int j, double w) { g.edges.emplace_back(i, j, w); },
             py::arg("i"), py::arg("j"), py::arg("weight") = 1.0);

    py::class_<QAOAResult>(m, "QAOAResult")
        .def_property_readonly("probabilities", [](const QAOAResult& r) { return to_vec(r.probabilities); })
        .def_readonly("best_cut", &QAOAResult::best_cut);

    py::class_<QAOA>(m, "QAOA")
        .def_static("run", &QAOA::run,
                    py::arg("backend"), py::arg("graph"), py::arg("gammas"), py::arg("betas"),
                    py::call_guard<py::gil_scoped_release>(),
                    "p = len(gammas) = len(betas) layers of U_C(gamma) then U_B(beta) on |+>^n.")
        .def_static("cut_value", &QAOA::cut_value, py::arg("graph"), py::arg("z"),
                    "Cut weight of the bitstring with basis index z.");

    // ---- VQE --------------------------------------------------------------
    py::class_<VQEResult>(m, "VQEResult")
        .def_readonly("energy", &VQEResult::energy)
        .def_property_readonly("params", [](const VQEResult& r) { return to_vec(r.params); })
        .def_readonly("iterations", &VQEResult::iterations);

    py::class_<VQE>(m, "VQE")
        .def(py::init<int, int, std::uint64_t>(),
             py::arg("n_qubits"), py::arg("n_layers"), py::arg("seed") = 1)
        .def_property_readonly("n_qubits", &VQE::n_qubits)
        .def_property_readonly("n_layers", &VQE::n_layers)
        .def_property_readonly("n_params", &VQE::n_params)
        .def("prepare", &VQE::prepare, py::arg("backend"), py::arg("params"),
             py::call_guard<py::gil_scoped_release>(),
             "Reset the backend and apply the ansatz; params ordered layer by layer, theta[l * n + q].")
        // energy_fn calls back into Python, so the GIL stays held for the whole optimisation.
        .def("optimize",
             [](const VQE& vqe, Backend& backend, const py::function& energy_fn, int max_iter, double lr) {
                 auto fn = [&energy_fn](const std::vector<double>& th) {
                     py::array_t<double> arr(static_cast<py::ssize_t>(th.size()), th.data());
                     return energy_fn(arr).cast<double>();
                 };
                 return vqe.optimize(backend, fn, max_iter, lr);
             },
             py::arg("backend"), py::arg("energy_fn"), py::arg("max_iter") = 200, py::arg("lr") = 0.05,
             "energy_fn(params: ndarray) -> float; typically vqe.prepare(backend, params), then <H> "
             "from backend.probabilities(). Gradients by the parameter-shift rule.");
}