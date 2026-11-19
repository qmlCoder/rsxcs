//! Meta-GGA exchange functionals.
//!
//! Layout of `v`:
//! * unpolarized: `[n, sigma, tau]`
//! * polarized:   `[n_up, n_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down]`
//!
//! TPSS does not depend on the Laplacian of the density.

use crate::consts::*;
use crate::info::Functional;
use crate::scalar::Scalar;

/// TPSS exchange enhancement factors (libxc `mgga_x_tpss`).
const TPSS_B: f64 = 0.40;
const TPSS_C: f64 = 1.59096;
const TPSS_E: f64 = 1.537;
const TPSS_KAPPA: f64 = 0.8040;
const TPSS_MU: f64 = 0.21951;
const TPSS_BLOC_A: f64 = 2.0;
const TPSS_BLOC_B: f64 = 0.0;

#[inline]
fn tpss_x_spin<S: Scalar>(n: S, sigma: S, tau: S) -> S {
    let n43 = n.powf(4.0 / 3.0);
    // raw x^2 = |grad n_s|^2 / n_s^(8/3)
    let x2 = sigma / (n43 * n43);
    // t = tau / n^(5/3) (libxc convention, mgga_alpha divides by K_FACTOR_C)
    let t = tau / n.powf(5.0 / 3.0);
    // p = (X2S x)^2 = s^2
    let p = S::from_f64(X2S * X2S) * x2;

    let alpha = (t - x2 / S::from_f64(8.0)) / S::from_f64(K_FACTOR_C);
    let z = x2 / (S::from_f64(8.0) * t);
    let z2 = z * z;

    // q_b, Eq. (7)
    let am1 = alpha - S::from_f64(1.0);
    let qb = S::from_f64(9.0 / 20.0) * am1
        / (S::from_f64(1.0) + S::from_f64(TPSS_B) * alpha * am1).sqrt()
        + S::from_f64(2.0 / 3.0) * p;

    let ff = TPSS_BLOC_A + TPSS_BLOC_B * z.value();
    let e_sqrt = S::from_f64(TPSS_E).sqrt();
    let mu = S::from_f64(TPSS_MU);

    // Eq. (10)
    let num = (S::from_f64(MU_GE)
        + S::from_f64(TPSS_C) * z.powf(ff) / ((S::from_f64(1.0) + z2) * (S::from_f64(1.0) + z2)))
        * p
        + S::from_f64(146.0 / 2025.0) * qb * qb
        - S::from_f64(73.0 / 405.0)
            * qb
            * (S::from_f64(0.5) * (S::from_f64(9.0 / 25.0) * z2 + p * p)).sqrt()
        + S::from_f64(MU_GE * MU_GE) / S::from_f64(TPSS_KAPPA) * p * p
        + S::from_f64(2.0) * e_sqrt * S::from_f64(MU_GE) * S::from_f64(9.0 / 25.0) * z2
        + S::from_f64(TPSS_E) * mu * p * p * p;
    let den = (S::from_f64(1.0) + e_sqrt * p) * (S::from_f64(1.0) + e_sqrt * p);
    let fx = num / den;

    // Eq. (5), in the cancellation-free algebraic form used by libxc 7.1.2.
    let kappa = S::from_f64(TPSS_KAPPA);
    let enh = S::from_f64(1.0) + kappa * fx / (kappa + fx);

    -S::from_f64(X_FACTOR_C) * n43 * enh
}

/// Energy density contribution of a meta-GGA functional.
pub(crate) fn energy<S: Scalar>(f: Functional, polarized: bool, v: &[S]) -> S {
    let half = S::from_f64(0.5);
    let quarter = S::from_f64(0.25);
    let (na, nb, saa, sbb, ta, tb) = if polarized {
        (v[0], v[1], v[2], v[4], v[5], v[6])
    } else {
        (
            v[0] * half,
            v[0] * half,
            v[1] * quarter,
            v[1] * quarter,
            v[2] * half,
            v[2] * half,
        )
    };

    match f {
        Functional::MggaXTpss => tpss_x_spin(na, saa, ta) + tpss_x_spin(nb, sbb, tb),
        _ => unreachable!("{f} is not a meta-GGA functional"),
    }
}
