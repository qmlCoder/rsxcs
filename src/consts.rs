//! Numerical constants used by libxc 7.1.2.
//!
//! The values are copied verbatim from `src/util.h` of libxc 7.1.2 so that
//! results agree with the reference implementation to machine precision.

/// `(3/(4 pi))^(1/3)`.
pub const RS_FACTOR: f64 = 0.6203504908994000166680068120477781673508;

/// `3/8 * (3/pi)^(1/3) * 4^(2/3)`; coefficient of the per-spin LDA exchange
/// energy density `-X_FACTOR_C * n_sigma^(4/3)`.
pub const X_FACTOR_C: f64 = 0.9305257363491000250020102180716672510262;

/// `3/10 * (6 pi^2)^(2/3)`; Thomas-Fermi kinetic-energy coefficient.
pub const K_FACTOR_C: f64 = 4.557799872345597137288163759599305358515;

/// `10/81`, the gradient coefficient of PBE-Sol / PBEsol exchange.
pub const MU_GE: f64 = 0.1234567901234567901234567901234567901235;

/// `1 / (2 (6 pi^2)^(1/3))`, converts `|grad n| / n^(4/3)` to the standard
/// reduced gradient `s`.
pub const X2S: f64 = 0.1282782438530421943003109254455883701296;

/// `2^(4/3) - 2`.
pub const FZETA_FACTOR: f64 = 0.5198420997897463295344212145564567011405;

/// Clamp applied to `1 + zeta` (libxc's default `zeta_threshold`).
pub const ZETA_THRESHOLD: f64 = f64::EPSILON;

/// The density threshold used throughout this crate (libxc uses 1e-15 for
/// the functionals implemented here).
pub const DENS_THRESHOLD: f64 = 1e-15;

/// Floor applied to `sigma = |grad rho|^2` (libxc derives this from the
/// density threshold as `(1e-15)^(4/3)`, squared).
pub const SIGMA_FLOOR: f64 = 1e-40;

/// Floor applied to the kinetic-energy density.
pub const TAU_FLOOR: f64 = 1e-20;
