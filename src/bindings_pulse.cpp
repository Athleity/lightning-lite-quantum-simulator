// src/bindings_pulse.cpp
//
// Python bindings for Tier 3 (module: quantum_sim_pulse).
//
// Build:
//   g++ -O2 -std=c++17 -shared -fPIC -fopenmp \
//       $(python3 -m pybind11 --includes) -I/usr/include/eigen3 \
//       src/bindings_pulse.cpp src/Transmon.cpp src/DRAG.cpp src/SchrodingerSolver.cpp \
//       -o build/quantum_sim_pulse$(python3-config --extension-suffix)
//   (compile with NO -DLL_TEST: each .cpp has its own test main)
//
// Conventions: scalar parameters are read-only properties, everything else is a method.
// Units are those of Transmon.h (Hz for E/h, rad/s for omega, seconds for time).
//
// GIL: long-running C++ (propagator, leakage, evolve) releases it. Every Python callable
// (envelope, H(t)) is wrapped so it re-acquires the GIL per call and is destroyed under it.
// Envelopes made by gaussian_envelope / drag_envelope / square_envelope stay native: a
// DrivePulse built from one never calls back into Python.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Transmon.h"

namespace py = pybind11;
using namespace pybind11::literals;

// ---------------------------------------------------------------------------
// Declarations of code that lives only in .cpp files (no header yet).
// The LindbladSolver definition below must stay token-identical to the one in
// SchrodingerSolver.cpp (ODR); move both into SchrodingerSolver.h when convenient.
// ---------------------------------------------------------------------------
namespace ll {

Envelope gaussian_envelope(double theta, double duration, double sigma);
Envelope drag_envelope(double theta, double duration, double sigma, double beta,
                       double anharmonicity_angular);
Envelope square_envelope(double theta, double duration);

constexpr int SOLVER_MAX_QUBITS = 10;
constexpr int SOLVER_RATE_SAMPLES = 1025;
constexpr double SOLVER_STATE_TOL = 1e-9;

using HamiltonianFn = std::function<CMatrix(double)>;

struct LindbladOp {
    CMatrix L;
    double rate;
};

class LindbladSolver {
public:
    LindbladSolver(int dim, HamiltonianFn h, std::vector<LindbladOp> ops = {});

    int dim() const { return dim_; }

    CMatrix evolve(const CMatrix& rho0, double t_start, double t_end,
                   int steps = AUTO_STEPS) const;

    double rate_bound(double t_start, double t_end) const;

private:
    CMatrix eval_h(double t, const char* who) const;
    CMatrix deriv(const CMatrix& h, const CMatrix& rho) const;

    int dim_;
    HamiltonianFn h_;
    std::vector<LindbladOp> ops_;
    CMatrix gamma_;
    double diss_rate_;
};

}  // namespace ll

namespace {

using namespace ll;

// Opaque handle so native envelopes pass through Python without a round trip.
struct EnvelopeHandle {
    Envelope fn;
};

// Keeps a Python object alive; the last reference is dropped with the GIL held.
std::shared_ptr<py::object> hold(py::object f) {
    return std::shared_ptr<py::object>(new py::object(std::move(f)), [](py::object* o) {
        py::gil_scoped_acquire gil;
        delete o;
    });
}

void require_callable(const py::object& f, const char* what) {
    if (!PyCallable_Check(f.ptr())) throw py::type_error(std::string(what) + " must be callable");
}

Envelope to_envelope(const py::object& obj) {
    if (py::isinstance<EnvelopeHandle>(obj)) return obj.cast<const EnvelopeHandle&>().fn;
    require_callable(obj, "envelope");
    auto h = hold(obj);
    return [h](double t) -> cplx {
        py::gil_scoped_acquire gil;
        return py::cast<cplx>((*h)(t));  // float or complex return both accepted
    };
}

HamiltonianFn to_hamiltonian(const py::object& obj) {
    if (PyCallable_Check(obj.ptr())) {
        auto h = hold(obj);
        return [h](double t) -> CMatrix {
            py::gil_scoped_acquire gil;
            return py::cast<CMatrix>((*h)(t));
        };
    }
    const CMatrix h0 = py::cast<CMatrix>(obj);  // constant H given as a matrix
    return [h0](double) { return h0; };
}

std::string transmon_repr(const Transmon& t) {
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "Transmon(ej=%.6g, ec=%.6g, n_levels=%d, charge_cutoff=%d, ng=%.6g)", t.ej(),
                  t.ec(), t.n_levels(), t.charge_cutoff(), t.ng());
    return buf;
}

}  // namespace

PYBIND11_MODULE(quantum_sim_pulse, m) {
    m.doc() = "Lightning-Lite Tier 3: transmon spectrum, pulse-level dynamics, DRAG, Lindblad solver";

    // ---- constants ---------------------------------------------------------------------
    m.attr("TWO_PI") = TWO_PI;
    m.attr("MIN_LEVELS") = MIN_LEVELS;
    m.attr("DEFAULT_LEVELS") = DEFAULT_LEVELS;
    m.attr("MAX_LEVELS") = MAX_LEVELS;
    m.attr("MIN_CHARGE_CUTOFF") = MIN_CHARGE_CUTOFF;
    m.attr("DEFAULT_CHARGE_CUTOFF") = DEFAULT_CHARGE_CUTOFF;
    m.attr("CHARGE_EDGE_TOL") = CHARGE_EDGE_TOL;
    m.attr("TRANSMON_REGIME_RATIO") = TRANSMON_REGIME_RATIO;
    m.attr("AUTO_STEPS") = AUTO_STEPS;
    m.attr("RK4_TARGET_PHASE_STEP") = RK4_TARGET_PHASE_STEP;
    m.attr("RK4_MAX_PHASE_STEP") = RK4_MAX_PHASE_STEP;
    m.attr("ENVELOPE_SAMPLES") = ENVELOPE_SAMPLES;
    m.attr("MAX_PULSE_STEPS") = MAX_PULSE_STEPS;
    m.attr("SOLVER_MAX_QUBITS") = SOLVER_MAX_QUBITS;

    // ---- DriveModel --------------------------------------------------------------------
    py::enum_<DriveModel>(m, "DriveModel")
        .value("RWA", DriveModel::RWA)
        .value("Full", DriveModel::Full);

    // ---- Envelope ------------------------------------------------------------------------
    py::class_<EnvelopeHandle>(m, "Envelope",
                               "Complex baseband envelope Omega(t) = Omega_I + i Omega_Q in rad/s.")
        .def(py::init([](const py::object& f) { return EnvelopeHandle{to_envelope(f)}; }),
             "callable"_a)
        .def("__call__", [](const EnvelopeHandle& e, double t) { return e.fn(t); }, "t"_a)
        .def(
            "sample",
            [](const EnvelopeHandle& e, const Eigen::VectorXd& t) {
                CVector out(t.size());
                for (Eigen::Index k = 0; k < t.size(); ++k) out(k) = e.fn(t(k));
                return out;
            },
            "t"_a, "Evaluate at an array of times; returns a complex array.");

    // ---- DrivePulse ----------------------------------------------------------------------
    py::class_<DrivePulse>(m, "DrivePulse")
        .def(py::init([](const py::object& envelope, double duration, double drive_frequency) {
                 return DrivePulse{to_envelope(envelope), duration, drive_frequency};
             }),
             "envelope"_a, "duration"_a, "drive_frequency"_a)
        .def_property(
            "envelope", [](const DrivePulse& p) { return EnvelopeHandle{p.envelope}; },
            [](DrivePulse& p, const py::object& f) { p.envelope = to_envelope(f); })
        .def_readwrite("duration", &DrivePulse::duration)
        .def_readwrite("drive_frequency", &DrivePulse::drive_frequency)
        .def("__repr__", [](const DrivePulse& p) {
            char buf[120];
            std::snprintf(buf, sizeof buf, "DrivePulse(duration=%.6g, drive_frequency=%.9g)",
                          p.duration, p.drive_frequency);
            return std::string(buf);
        });

    // ---- Transmon ------------------------------------------------------------------------
    py::class_<Transmon>(m, "Transmon")
        .def(py::init<double, double, int, int, double>(), "ej"_a, "ec"_a,
             "n_levels"_a = DEFAULT_LEVELS, "charge_cutoff"_a = DEFAULT_CHARGE_CUTOFF,
             "ng"_a = 0.0)
        // parameters
        .def_property_readonly("ej", &Transmon::ej)
        .def_property_readonly("ec", &Transmon::ec)
        .def_property_readonly("ng", &Transmon::ng)
        .def_property_readonly("n_levels", &Transmon::n_levels)
        .def_property_readonly("charge_cutoff", &Transmon::charge_cutoff)
        .def_property_readonly("charge_dim", &Transmon::charge_dim)
        .def_property_readonly("ej_ec_ratio", &Transmon::ej_ec_ratio)
        .def_property_readonly("is_transmon_regime", &Transmon::is_transmon_regime)
        // spectrum
        .def("energies", [](const Transmon& t) { return Eigen::VectorXd(t.energies()); })
        .def("energy", &Transmon::energy, "level"_a)
        .def("frequency", &Transmon::frequency, "i"_a, "j"_a)
        .def("omega", &Transmon::omega, "i"_a, "j"_a)
        .def("frequency_01", &Transmon::frequency_01)
        .def("omega_01", &Transmon::omega_01)
        .def("anharmonicity", &Transmon::anharmonicity)
        .def("anharmonicity_angular", &Transmon::anharmonicity_angular)
        // operators
        .def("hamiltonian_charge_basis", &Transmon::hamiltonian_charge_basis)
        .def("eigenstate", &Transmon::eigenstate, "level"_a)
        .def("edge_weight", &Transmon::edge_weight, "level"_a)
        .def("n_matrix", [](const Transmon& t) { return Eigen::MatrixXd(t.n_matrix()); })
        .def("n_element", &Transmon::n_element, "i"_a, "j"_a)
        .def("coupling_matrix", &Transmon::coupling_matrix)
        // transmon-limit formulas
        .def("plasma_frequency", &Transmon::plasma_frequency)
        .def("frequency_01_asymptotic", &Transmon::frequency_01_asymptotic)
        .def("anharmonicity_leading", &Transmon::anharmonicity_leading)
        .def("anharmonicity_asymptotic", &Transmon::anharmonicity_asymptotic)
        .def("n01_asymptotic", &Transmon::n01_asymptotic)
        // driven dynamics
        .def("rotating_hamiltonian", &Transmon::rotating_hamiltonian, "t"_a, "pulse"_a,
             "model"_a = DriveModel::RWA)
        .def("propagator", &Transmon::propagator, "pulse"_a, "model"_a = DriveModel::RWA,
             "steps"_a = AUTO_STEPS, py::call_guard<py::gil_scoped_release>())
        .def("leakage", &Transmon::leakage, "pulse"_a, "initial_level"_a = 1,
             "model"_a = DriveModel::RWA, "steps"_a = AUTO_STEPS,
             py::call_guard<py::gil_scoped_release>())
        .def("__repr__", &transmon_repr);

    // ---- DRAG / pulse shapes ------------------------------------------------------------------
    m.def("gaussian_envelope",
          [](double theta, double duration, double sigma) {
              return EnvelopeHandle{gaussian_envelope(theta, duration, sigma)};
          },
          "theta"_a, "duration"_a, "sigma"_a,
          "Truncated zero-offset Gaussian with area theta; Omega_Q = 0.");
    m.def("drag_envelope",
          [](double theta, double duration, double sigma, double beta,
             double anharmonicity_angular) {
              return EnvelopeHandle{
                  drag_envelope(theta, duration, sigma, beta, anharmonicity_angular)};
          },
          "theta"_a, "duration"_a, "sigma"_a, "beta"_a, "anharmonicity_angular"_a,
          "First-order DRAG: Omega = x(t) - i beta x'(t) / alpha.");
    m.def("square_envelope",
          [](double theta, double duration) {
              return EnvelopeHandle{square_envelope(theta, duration)};
          },
          "theta"_a, "duration"_a, "Constant real Omega = theta / duration.");

    // ---- Lindblad solver ------------------------------------------------------------------------
    py::class_<LindbladOp>(m, "LindbladOp", "Jump operator L (dim x dim) with rate gamma in 1/s.")
        .def(py::init([](const CMatrix& L, double rate) { return LindbladOp{L, rate}; }), "L"_a,
             "rate"_a)
        .def(py::init([](const py::tuple& t) {
                 if (t.size() != 2) throw py::value_error("expected an (L, rate) tuple");
                 return LindbladOp{t[0].cast<CMatrix>(), t[1].cast<double>()};
             }),
             "pair"_a)
        .def_readwrite("L", &LindbladOp::L)
        .def_readwrite("rate", &LindbladOp::rate);
    py::implicitly_convertible<py::tuple, LindbladOp>();

    py::class_<LindbladSolver>(m, "LindbladSolver",
                               "RK4 Lindblad master-equation solver on the d x d density matrix.")
        .def(py::init([](int dim, const py::object& hamiltonian, std::vector<LindbladOp> ops) {
                 return LindbladSolver(dim, to_hamiltonian(hamiltonian), std::move(ops));
             }),
             "dim"_a, "hamiltonian"_a, "ops"_a = std::vector<LindbladOp>{},
             "hamiltonian: callable t -> (dim x dim) matrix in rad/s, or a constant matrix. "
             "ops: list of LindbladOp or (L, rate) tuples.")
        .def_static(
            "from_pulse",
            [](const Transmon& tr, const DrivePulse& pulse, DriveModel model,
               std::vector<LindbladOp> ops) {
                return LindbladSolver(
                    tr.n_levels(),
                    [tr, pulse, model](double t) { return tr.rotating_hamiltonian(t, pulse, model); },
                    std::move(ops));
            },
            "transmon"_a, "pulse"_a, "model"_a = DriveModel::RWA,
            "ops"_a = std::vector<LindbladOp>{},
            "Solver with H(t) = transmon.rotating_hamiltonian(t, pulse, model).")
        .def_property_readonly("dim", &LindbladSolver::dim)
        .def("rate_bound", &LindbladSolver::rate_bound, "t_start"_a, "t_end"_a,
             py::call_guard<py::gil_scoped_release>())
        .def("evolve", &LindbladSolver::evolve, "rho0"_a, "t_start"_a, "t_end"_a,
             "steps"_a = AUTO_STEPS, py::call_guard<py::gil_scoped_release>());
}