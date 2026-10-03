// Build:
//   g++ -O3 -shared -std=c++17 -fopenmp -fPIC -I/usr/include/eigen3 \
//       $(python3 -m pybind11 --includes) \
//       src/Noise.cpp src/bindings_noise.cpp \
//       -o quantum_sim_noise$(python3-config --extension-suffix)

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <vector>

#include "Noise.h"

namespace py = pybind11;
using namespace ll;

namespace {

template <class... Args>
std::string repr(const char* name, Args&&... args) {
    std::ostringstream os;
    os.precision(6);
    os << name << "(";
    const char* sep = "";
    ((os << sep << args, sep = ", "), ...);
    os << ")";
    return os.str();
}

}  // namespace

PYBIND11_MODULE(quantum_sim_noise, m) {
    m.doc() =
        "Density-matrix noise channels for Lightning-Lite: Kraus maps, depolarizing, "
        "T1/T2 and Purcell decay. Errors raise ValueError.";

    m.attr("MAX_DM_QUBITS") = MAX_DM_QUBITS;
    m.attr("DEFAULT_TOL") = DEFAULT_TOL;

    // ---------------------------------------------------------------- KrausChannel
    // NumPy arrays are copied into Eigen on the way in, so a plain Eigen::MatrixXcd&
    // argument would modify a temporary and the caller's array would stay unchanged.
    // apply() therefore returns a new array. apply_inplace() takes an Eigen::Ref and
    // writes through, which only works for a Fortran-ordered complex128 array
    // (np.asfortranarray); any other layout raises TypeError instead of copying.
    py::class_<KrausChannel>(m, "KrausChannel",
                             "CPTP map rho -> sum_k K_k rho K_k^dagger on a set of target qubits.")
        .def(py::init<std::vector<CMatrix>, std::vector<int>>(), py::arg("ops"),
             py::arg("targets"))
        .def(
            "apply",
            [](const KrausChannel& self, CMatrix rho, int n_qubits) {
                self.apply(rho, n_qubits);
                return rho;
            },
            py::arg("rho"), py::arg("n_qubits"), py::call_guard<py::gil_scoped_release>(),
            "Return the channel applied to rho. The input array is not modified.")
        .def(
            "apply_inplace",
            [](const KrausChannel& self, Eigen::Ref<CMatrix> rho, int n_qubits) {
                CMatrix work = rho;
                self.apply(work, n_qubits);  // throws before rho is touched
                rho = work;
            },
            py::arg("rho"), py::arg("n_qubits"), py::call_guard<py::gil_scoped_release>(),
            "Apply the channel to rho in place. rho must be a Fortran-ordered complex128 array.")
        .def("is_trace_preserving", &KrausChannel::is_trace_preserving,
             py::arg("tol") = DEFAULT_TOL)
        .def("compose", &KrausChannel::compose, py::arg("other"),
             "Channel that applies self first, then other. Requires identical targets.")
        .def("operators", &KrausChannel::operators)
        .def("targets", &KrausChannel::targets)
        .def("num_targets", &KrausChannel::num_targets)
        .def_static("amplitude_damping", &KrausChannel::amplitude_damping, py::arg("gamma"),
                    py::arg("qubit"))
        // "lambda" is a Python keyword, so the argument is lambda_.
        .def_static("phase_damping", &KrausChannel::phase_damping, py::arg("lambda_"),
                    py::arg("qubit"))
        .def_static("identity", &KrausChannel::identity, py::arg("qubit"))
        .def("__repr__", [](const KrausChannel& self) {
            return repr("KrausChannel", "num_ops=" + std::to_string(self.operators().size()),
                        "num_targets=" + std::to_string(self.num_targets()));
        });

    // ------------------------------------------------------------ DepolarizingChannel
    py::class_<DepolarizingChannel>(
        m, "DepolarizingChannel",
        "rho -> (1-p) rho + p/(4^m-1) sum_{P != I} P rho P for m = 1 or 2 qubits.")
        .def(py::init<double, int>(), py::arg("p"), py::arg("num_qubits"))
        .def("probability", &DepolarizingChannel::probability)
        .def("num_qubits", &DepolarizingChannel::num_qubits)
        .def("to_kraus", &DepolarizingChannel::to_kraus, py::arg("targets"))
        .def("__repr__", [](const DepolarizingChannel& self) {
            return repr("DepolarizingChannel", "p=" + std::to_string(self.probability()),
                        "num_qubits=" + std::to_string(self.num_qubits()));
        });

    // -------------------------------------------------------------------- T1T2Model
    py::class_<T1T2Model>(m, "T1T2Model",
                          "Single-qubit relaxation and dephasing, times in seconds, T2 <= 2 T1.")
        .def(py::init<double, double>(), py::arg("t1"), py::arg("t2"))
        .def("t1", &T1T2Model::t1)
        .def("t2", &T1T2Model::t2)
        .def("t_phi", &T1T2Model::t_phi)
        .def("gamma", &T1T2Model::gamma, py::arg("duration"))
        // Same keyword issue: model.lambda(t) is a syntax error, hence lambda_.
        .def("lambda_", &T1T2Model::lambda, py::arg("duration"))
        .def("channel", &T1T2Model::channel, py::arg("duration"), py::arg("qubit"))
        .def("excited_population", &T1T2Model::excited_population, py::arg("duration"))
        .def("coherence_magnitude", &T1T2Model::coherence_magnitude, py::arg("duration"))
        .def("__repr__", [](const T1T2Model& self) {
            return repr("T1T2Model", "t1=" + std::to_string(self.t1()),
                        "t2=" + std::to_string(self.t2()));
        });

    // ----------------------------------------------------------------- PurcellModel
    py::class_<PurcellModel>(m, "PurcellModel",
                             "Dispersive-limit Purcell decay gamma_P = s * kappa * (g/delta)^2, "
                             "rates in rad/s, s = filter suppression in (0, 1].")
        .def(py::init<double, double, double, double>(), py::arg("g"), py::arg("delta"),
             py::arg("kappa"), py::arg("filter_suppression") = 1.0)
        .def("dispersive_ratio", &PurcellModel::dispersive_ratio)
        .def("rate", &PurcellModel::rate)
        .def("t1_limit", &PurcellModel::t1_limit)
        .def("filter_suppression", &PurcellModel::filter_suppression)
        .def("with_filter", &PurcellModel::with_filter, py::arg("suppression"))
        .def("with_filter_db", &PurcellModel::with_filter_db, py::arg("attenuation_db"))
        .def("channel", &PurcellModel::channel, py::arg("duration"), py::arg("qubit"))
        .def("combined_with", &PurcellModel::combined_with, py::arg("intrinsic"),
             "Intrinsic T1/T2 plus Purcell decay with the intrinsic Tphi held fixed.")
        .def("__repr__", [](const PurcellModel& self) {
            return repr("PurcellModel", "rate=" + std::to_string(self.rate()),
                        "filter_suppression=" + std::to_string(self.filter_suppression()));
        });

    // -------------------------------------------------------------------- Functions
    m.def("density_from_statevector", &density_from_statevector, py::arg("psi"),
          "|psi><psi| for a normalized state vector of length 2^n, n <= MAX_DM_QUBITS.");
    m.def("ground_state_density", &ground_state_density, py::arg("n_qubits"),
          "|0...0><0...0| on n_qubits qubits.");
    m.def("is_valid_density_matrix", &is_valid_density_matrix, py::arg("rho"),
          py::arg("tol") = DEFAULT_TOL,
          "Hermitian, unit trace and positive semidefinite to within tol.");
}