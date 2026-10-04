// Python bindings for the variance-reduced quantum trajectory solver (Tier 3).
//
// Module: quantum_sim_trajectory
//
// Build:
//   g++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC -I/usr/include/eigen3 -Isrc \
//       $(python3 -m pybind11 --includes) \
//       src/bindings_trajectory.cpp \
//       -o build/quantum_sim_trajectory$(python3-config --extension-suffix) \
//       $(python3-config --ldflags)
//
// GIL handling. The solver runs without the GIL so its OpenMP loops use all cores. Python
// callables (a time-dependent H(t), an observable) are wrapped so that every call re-acquires the
// GIL, which also makes calls from OpenMP worker threads safe. Python callbacks therefore run one
// at a time; estimate_expectation takes a matrix observable and avoids them entirely.
// A wrapper is built while the GIL is still held, because copying a Python object needs it.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <memory>
#include <sstream>

#include "TrajectorySolver.h"

namespace py = pybind11;
using namespace ll;

namespace {

using Obs = std::function<double(const CVector&)>;

// Shared handle whose last release re-acquires the GIL, wherever it happens.
std::shared_ptr<py::object> keep(const py::object& o) {
    return std::shared_ptr<py::object>(new py::object(o), [](py::object* p) {
        py::gil_scoped_acquire gil;
        delete p;
    });
}

HamiltonianFn wrap_hamiltonian(const py::object& f) {
    if (!PyCallable_Check(f.ptr())) throw py::type_error("H must be a matrix or a callable t -> matrix");
    auto h = keep(f);
    return [h](double t) {
        py::gil_scoped_acquire gil;
        return (*h)(t).cast<CMatrix>();
    };
}

Obs wrap_observable(const py::object& f) {
    if (!PyCallable_Check(f.ptr())) throw py::type_error("obs must be callable: ndarray -> float");
    auto h = keep(f);
    return [h](const CVector& psi) {
        py::gil_scoped_acquire gil;
        return (*h)(psi).cast<double>();
    };
}

Obs matrix_observable(const CMatrix& op, int dim) {
    if (op.rows() != dim || op.cols() != dim) throw std::invalid_argument("estimate_expectation: operator must be dim x dim");
    if ((op - op.adjoint()).norm() > 1e-9 * std::max(1.0, op.norm()))
        throw std::invalid_argument("estimate_expectation: operator is not Hermitian");
    return [op](const CVector& psi) { return psi.dot(op * psi).real(); };
}

}  // namespace

PYBIND11_MODULE(quantum_sim_trajectory, m) {
    m.doc() = "Quantum trajectories with standard, projector and diffusive unravelings of the Lindblad equation";

    py::enum_<Unraveling>(m, "Unraveling")
        .value("STANDARD", Unraveling::STANDARD)    // quantum jumps; a Pauli channel applies P at rate gamma
        .value("PROJECTOR", Unraveling::PROJECTOR)  // Pauli-like channels rewritten with C = sqrt(g)(1 - P)
        .value("ANALOG", Unraveling::ANALOG);       // homodyne diffusion, no jumps

    py::class_<TrajectorySolver::Stats>(m, "Stats")
        .def(py::init<>())
        .def_readonly("mean", &TrajectorySolver::Stats::mean)
        .def_readonly("std_error", &TrajectorySolver::Stats::std_error)
        .def_readonly("n", &TrajectorySolver::Stats::n)
        .def("__repr__", [](const TrajectorySolver::Stats& s) {
            std::ostringstream os;
            os << "Stats(mean=" << s.mean << ", std_error=" << s.std_error << ", n=" << s.n << ")";
            return os.str();
        });

    py::class_<TrajectorySolver> cls(m, "TrajectorySolver");
    cls.def(py::init<int>(), py::arg("n_qubits"))
        .def("set_hamiltonian", [](TrajectorySolver& s, const CMatrix& h) { s.set_hamiltonian(h); }, py::arg("H"),
             "Constant Hermitian matrix.")
        .def("set_hamiltonian",
             [](TrajectorySolver& s, const py::object& f) { s.set_hamiltonian(HamiltonianFn(wrap_hamiltonian(f))); },
             py::arg("H"), "Callable t -> Hermitian matrix; sampled once per call at step endpoints and midpoints.")
        .def("add_lindblad", &TrajectorySolver::add_lindblad, py::arg("L"), py::arg("gamma"),
             "Adds gamma (L rho L^+ - 1/2 {L^+ L, rho}).")
        .def("clear_lindblad", &TrajectorySolver::clear_lindblad)
        .def("projector_applies", &TrajectorySolver::projector_applies, py::arg("channel"),
             "True if PROJECTOR rewrites this channel (Hermitian L with L^2 = c 1).")
        .def_property_readonly("n_qubits", &TrajectorySolver::n_qubits)
        .def_property_readonly("dim", &TrajectorySolver::dim)
        .def_property_readonly("n_channels", &TrajectorySolver::n_channels)
        .def("reset", &TrajectorySolver::reset)
        .def("state", [](const TrajectorySolver& s) { return CVector(s.state()); })
        .def("run_one", &TrajectorySolver::run_one,
             py::arg("psi0"), py::arg("duration"), py::arg("dt"), py::arg("seed"),
             py::arg("kind") = Unraveling::STANDARD, py::call_guard<py::gil_scoped_release>(),
             "One trajectory; returns the normalised final state.")
        .def("evolve_ensemble", &TrajectorySolver::evolve_ensemble,
             py::arg("psi0"), py::arg("duration"), py::arg("dt"), py::arg("n_trajectories"),
             py::arg("kind") = Unraveling::STANDARD, py::arg("seed") = std::uint64_t{1},
             py::call_guard<py::gil_scoped_release>(),
             "rho = mean of |psi><psi| over n_trajectories.")
        .def("estimate_observable",
             [](const TrajectorySolver& s, const CVector& psi0, double duration, double dt, int n,
                const py::object& obs, Unraveling kind, std::uint64_t seed) {
                 const Obs f = wrap_observable(obs);
                 py::gil_scoped_release release;
                 return s.estimate_observable(psi0, duration, dt, n, kind, f, seed);
             },
             py::arg("psi0"), py::arg("duration"), py::arg("dt"), py::arg("n_trajectories"), py::arg("obs"),
             py::arg("kind") = Unraveling::STANDARD, py::arg("seed") = std::uint64_t{1},
             "obs(psi: ndarray) -> float, evaluated on every final state.")
        .def("estimate_expectation",
             [](const TrajectorySolver& s, const CVector& psi0, double duration, double dt, int n,
                const CMatrix& op, Unraveling kind, std::uint64_t seed) {
                 const Obs f = matrix_observable(op, s.dim());
                 py::gil_scoped_release release;
                 return s.estimate_observable(psi0, duration, dt, n, kind, f, seed);
             },
             py::arg("psi0"), py::arg("duration"), py::arg("dt"), py::arg("n_trajectories"), py::arg("op"),
             py::arg("kind") = Unraveling::STANDARD, py::arg("seed") = std::uint64_t{1},
             "Mean and standard error of <psi|op|psi> for a Hermitian matrix; runs without Python callbacks.");
    cls.attr("MAX_QUBITS") = TrajectorySolver::MAX_QUBITS;
}