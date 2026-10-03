#pragma once

#include <Eigen/Dense>

#include <complex>
#include <functional>
#include <stdexcept>

namespace ll {

// Same aliases as Noise.h. Declaring an alias twice for the same type is legal,
// so either header can be included on its own.
using cplx = std::complex<double>;
using CMatrix = Eigen::MatrixXcd;
using CVector = Eigen::VectorXcd;

inline constexpr double TWO_PI = 6.283185307179586476925286766559;

// Units used by everything below
//   Energies (EJ, EC, E_i, alpha) are E/h in Hz, so EC = 300e6 means EC/h = 300 MHz.
//   Angular quantities (omega_ij, pulse envelopes, drive frequency) are rad/s,
//   with omega = 2 pi * (E/h). hbar never appears explicitly.
//   Times are in seconds.

// Eigenlevels kept in the projected operators. At least 3, because the |2>
// level is needed for the anharmonicity and for leakage.
constexpr int MIN_LEVELS = 3;
constexpr int DEFAULT_LEVELS = 3;
constexpr int MAX_LEVELS = 16;

// Charge basis |n>, n = -N..N, dimension 2N+1. N is the "charge cutoff".
constexpr int MIN_CHARGE_CUTOFF = 8;
constexpr int DEFAULT_CHARGE_CUTOFF = 40;

// A retained eigenstate with more than this probability on the outermost
// charge states |n| = N means the cutoff is too small, and the constructor throws.
constexpr double CHARGE_EDGE_TOL = 1e-12;

// EJ/EC above which the transmon-limit expansions are meaningful.
constexpr double TRANSMON_REGIME_RATIO = 20.0;

// Pulse integration (fixed-step RK4 on the rotating-frame Schrodinger equation).
// Local RK4 error scales like (rate * dt)^5 / 120, where rate bounds the fastest
// frequency in H_rot(t), including explicit time dependence. Steps with
// rate * dt above RK4_MAX_PHASE_STEP are rejected, not run.
constexpr int AUTO_STEPS = 0;                    // pass as `steps` to pick dt automatically
constexpr double RK4_TARGET_PHASE_STEP = 0.01;   // rate * dt aimed for when steps == AUTO_STEPS
constexpr double RK4_MAX_PHASE_STEP = 0.02;      // hard limit for a user-supplied step count
constexpr int ENVELOPE_SAMPLES = 4096;           // samples used to estimate peak |Omega|
constexpr int MAX_PULSE_STEPS = 5000000;

// Complex baseband envelope Omega(t) = Omega_I(t) + i Omega_Q(t) in rad/s.
// Omega is the Rabi frequency of the 0-1 transition: a constant real Omega over
// a time T rotates the qubit by an angle Omega * T, so a pi pulse has Omega*T = pi.
using Envelope = std::function<cplx(double)>;

// The drive is the lab-frame term
//     H_drive(t)/hbar = f(t) * C,   f(t) = Re[Omega(t) exp(-i omega_d t)]
//                                        = Omega_I cos(omega_d t) + Omega_Q sin(omega_d t),
// where C = n / |<0|n|1>| couples through the charge operator. Pulse shapes
// (DRAG and others) are built as Envelopes in DRAG.cpp.
struct DrivePulse {
    Envelope envelope;       // evaluated for 0 <= t <= duration
    double duration;         // seconds, > 0 and finite
    double drive_frequency;  // omega_d in rad/s, > 0 and finite
};

enum class DriveModel {
    // Frame rotating at omega_d, nearest-neighbour coupling, counter-rotating terms dropped.
    // Cheap and accurate while the drive is far weaker than omega_d.
    RWA,
    // Same frame, every matrix element of C kept and no rotating-wave approximation.
    // H_rot(t) then oscillates at up to n_levels * omega_d, so it needs far more steps.
    Full
};

// ---------------------------------------------------------------------------
// Transmon
// ---------------------------------------------------------------------------

// A superconducting transmon in the charge basis,
//
//     H = 4 EC (n - ng)^2 - EJ cos(phi),      [phi, n] = i,
//
// with EC = e^2 / (2 C_sigma) the charging energy, EJ the Josephson energy, n the
// Cooper-pair number operator, phi the superconducting phase across the junction,
// and ng the offset charge in units of 2e. Because exp(i phi) raises n by one,
// cos(phi) = (1/2) sum_n (|n><n+1| + |n+1><n|), and in the charge basis H is a
// real symmetric tridiagonal matrix:
//
//     H[n][n]   = 4 EC (n - ng)^2
//     H[n][n+1] = H[n+1][n] = -EJ / 2.
//
// The constructor diagonalizes it (Eigen SelfAdjointEigenSolver) and keeps the
// lowest n_levels eigenpairs, so every operator below lives in the eigenbasis
// |0>, |1>, ..., |n_levels-1>.
//
// Transmon limit, EJ/EC >> 1. Expanding cos(phi) around the minimum gives a
// weakly anharmonic oscillator with plasma frequency sqrt(8 EJ EC):
//
//     E_m      = -EJ + sqrt(8 EJ EC) (m + 1/2) - (EC/12)(6 m^2 + 6 m + 3)
//                - (EC^{3/2} / EJ^{1/2}) (2m+1)(m^2+m+1) / (16 sqrt 2) + ...
//     E_01     = sqrt(8 EJ EC) - EC - EC sqrt(EC/EJ) / (2 sqrt 2)
//     alpha    = (E_2 - E_1) - (E_1 - E_0)
//              = -EC [1 + (9 / (8 sqrt 2)) sqrt(EC/EJ) + O(EC/EJ)]
//     |<m+1|n|m>| = sqrt(m+1)/2 * (EJ/(2 EC))^{1/4}   (harmonic limit)
//
// The leading result alpha/h = -EC/h converges slowly: the first correction is
// about 0.80 sqrt(EC/EJ), so alpha is 11% larger in magnitude than EC at
// EJ/EC = 50 and 2.5% larger at EJ/EC = 1000. Compare against the corrected
// anharmonicity_asymptotic() at moderate EJ/EC.
//
// Validate against: alpha -> -EC as EJ/EC grows, E_01 against
// frequency_01_asymptotic(), n_01 against n01_asymptotic(), n_12 / n_01 -> sqrt(2),
// and, at ng = 0, parity: n_ij = 0 whenever i + j is even, because n is odd under
// the reflection n -> -n that maps eigenstate j to (-1)^j times itself.
class Transmon {
public:
    // ej, ec in Hz as E/h, both finite and > 0. n_levels in [MIN_LEVELS, MAX_LEVELS].
    // charge_cutoff >= MIN_CHARGE_CUTOFF with 2*charge_cutoff + 1 > n_levels.
    // ng finite. Throws std::invalid_argument for any violation, and also when a
    // retained eigenstate has weight above CHARGE_EDGE_TOL on the outermost charge
    // states (cutoff too small for these parameters). Throws std::runtime_error
    // if the eigensolver does not converge or an n_{i,i+1} element vanishes.
    Transmon(double ej, double ec, int n_levels = DEFAULT_LEVELS,
             int charge_cutoff = DEFAULT_CHARGE_CUTOFF, double ng = 0.0);

    // --- parameters ---
    double ej() const { return ej_; }
    double ec() const { return ec_; }
    double ng() const { return ng_; }
    int n_levels() const { return n_levels_; }
    int charge_cutoff() const { return cutoff_; }
    int charge_dim() const { return 2 * cutoff_ + 1; }
    double ej_ec_ratio() const { return ej_ / ec_; }
    bool is_transmon_regime() const { return ej_ec_ratio() >= TRANSMON_REGIME_RATIO; }

    // --- spectrum ---
    // E_i / h in Hz, ascending, i = 0..n_levels-1. Absolute energies, so E_0 is
    // close to -EJ. Only differences are physical.
    const Eigen::VectorXd& energies() const { return energies_; }
    double energy(int level) const;  // std::out_of_range if level is not in [0, n_levels)

    // (E_j - E_i) / h in Hz and the same gap as an angular frequency in rad/s.
    double frequency(int i, int j) const;
    double omega(int i, int j) const;

    double frequency_01() const { return frequency(0, 1); }
    double omega_01() const { return omega(0, 1); }

    // alpha / h = (E_2 - E_1)/h - (E_1 - E_0)/h in Hz. Negative for a transmon.
    // anharmonicity_angular() is alpha / hbar in rad/s, which equals 2 pi times
    // anharmonicity(), so the validation target is anharmonicity() ~ -ec().
    double anharmonicity() const;
    double anharmonicity_angular() const { return TWO_PI * anharmonicity(); }

    // --- operators ---
    // H / h in Hz, (2N+1) x (2N+1), charge basis, as written above.
    Eigen::MatrixXd hamiltonian_charge_basis() const;

    // Real charge-basis amplitudes <n|level>, n = -N..N (index 0 is n = -N).
    // Sign gauge: <level|n|level+1> > 0, which makes the coupling matrix below
    // unambiguous. std::out_of_range for a bad level.
    Eigen::VectorXd eigenstate(int level) const;

    // Probability of `level` on the two outermost charge states |n| = N.
    // This is the convergence measure the constructor checks against CHARGE_EDGE_TOL.
    double edge_weight(int level) const;

    // <i|n|j> in the eigenbasis, n_levels x n_levels, real symmetric, in the
    // gauge above. n_element throws std::out_of_range for bad indices.
    const Eigen::MatrixXd& n_matrix() const { return n_matrix_; }
    double n_element(int i, int j) const;

    // C = n / <0|n|1>, so C(0,1) = +1 and C(1,2) is close to sqrt(2) in the
    // transmon limit. This is the operator the drive couples through.
    Eigen::MatrixXd coupling_matrix() const;

    // --- transmon-limit formulas, for validation (see class comment) ---
    double plasma_frequency() const;            // sqrt(8 EJ EC), Hz
    double frequency_01_asymptotic() const;     // sqrt(8EJEC) - EC - EC sqrt(EC/EJ)/(2 sqrt 2), Hz
    double anharmonicity_leading() const;       // -EC, Hz
    double anharmonicity_asymptotic() const;    // -EC [1 + 9/(8 sqrt 2) sqrt(EC/EJ)], Hz
    double n01_asymptotic() const;              // (1/2)(EJ/(2 EC))^{1/4}

    // --- driven dynamics ---
    // H_rot(t) / hbar in rad/s, n_levels x n_levels, in the frame rotating at
    // omega_d (U = exp(i omega_d t sum_j j |j><j|)). With
    // Delta_j = omega_j - j omega_d and omega_j = 2 pi (E_j - E_0)/h:
    //
    //   RWA:   H_rot = sum_j Delta_j |j><j|
    //                  + (1/2) sum_j c_{j,j+1} [ Omega*(t) |j><j+1| + Omega(t) |j+1><j| ]
    //   Full:  H_rot = sum_j Delta_j |j><j|
    //                  + f(t) sum_{ij} c_ij exp(i (i - j) omega_d t) |i><j|
    //
    // with c = coupling_matrix() and f(t) = Re[Omega(t) exp(-i omega_d t)]. The
    // Full form contains the RWA terms plus counter-rotating ones at 2 omega_d.
    // Throws std::invalid_argument for an invalid pulse.
    CMatrix rotating_hamiltonian(double t, const DrivePulse& pulse,
                                 DriveModel model = DriveModel::RWA) const;

    // U with |psi(T)> = U |psi(0)> in the rotating frame, from fixed-step RK4 on
    // dU/dt = -i H_rot(t) U with U(0) = 1. The frame is diagonal in the level
    // basis, so populations |U_kj|^2 are the same as in the lab frame.
    //
    // steps == AUTO_STEPS picks dt so that rate_bound * dt <= RK4_TARGET_PHASE_STEP,
    // where rate_bound adds the largest |Delta_j|, peak|Omega| times the norm of
    // C, and for Full also n_levels * omega_d. Peak |Omega| is estimated by
    // sampling the envelope at ENVELOPE_SAMPLES points. A positive `steps` is used
    // as given, and throws std::invalid_argument if rate_bound * dt exceeds
    // RK4_MAX_PHASE_STEP. Also throws if the automatic count exceeds
    // MAX_PULSE_STEPS, if steps is negative, or if the envelope returns a
    // non-finite value.
    CMatrix propagator(const DrivePulse& pulse, DriveModel model = DriveModel::RWA,
                       int steps = AUTO_STEPS) const;

    // Population outside the computational subspace {|0>, |1>} after the pulse,
    // starting in |initial_level>:  sum_{k >= 2} |U_{k, initial_level}|^2.
    // With n_levels = 3 this is the leakage to |2>. initial_level must be 0 or 1,
    // else std::invalid_argument. Repeat with a larger n_levels to check that
    // population reaching |3> and above does not change the answer.
    //
    // Validate against: a constant Omega pulse in the RWA, where H_rot is time
    // independent and exp(-i H T) from a direct eigendecomposition is exact, and
    // the weak-drive scaling leakage ~ (Omega/alpha)^2 for a square pulse.
    double leakage(const DrivePulse& pulse, int initial_level = 1,
                   DriveModel model = DriveModel::RWA, int steps = AUTO_STEPS) const;

private:
    void diagonalize();
    void check_level(int level, const char* who) const;
    void check_pulse(const DrivePulse& pulse, const char* who) const;
    double rate_bound(const DrivePulse& pulse, double peak_omega, DriveModel model) const;

    double ej_;
    double ec_;
    double ng_;
    int n_levels_;
    int cutoff_;

    Eigen::VectorXd energies_;   // n_levels, E_i / h in Hz
    Eigen::MatrixXd states_;     // (2N+1) x n_levels, columns are eigenstates
    Eigen::MatrixXd n_matrix_;   // n_levels x n_levels, <i|n|j>
};

}  // namespace ll