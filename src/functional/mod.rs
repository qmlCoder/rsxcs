//! Functional implementations and hybrid expansion.

pub(crate) mod gga;
pub(crate) mod lda;
pub(crate) mod mgga;

use crate::info::{Family, Functional};
use crate::scalar::Scalar;

/// Semilocal components of a hybrid: `(coefficient, functional)`.
///
/// The exact-exchange fraction (`cam_alpha` in libxc) is not included; a
/// hybrid evaluator is expected to add the Hartree-Fock exchange separately.
fn mixture(f: Functional) -> &'static [(f64, Functional)] {
    match f {
        // PBE0: 0.25 HF + 0.75 PBE-X + PBE-C
        Functional::HybGgaXcPbeh => &[(0.75, Functional::GgaXPbe), (1.0, Functional::GgaCPbe)],
        // B3LYP: 0.20 HF + 0.08 LDA-X + 0.72 B88 + 0.19 VWN-RPA + 0.81 LYP
        Functional::HybGgaXcB3lyp => &[
            (0.08, Functional::LdaX),
            (0.72, Functional::GgaXB88),
            (0.19, Functional::LdaCVwnRpa),
            (0.81, Functional::GgaCLyp),
        ],
        _ => &[],
    }
}

/// Energy density of a single (non-hybrid) functional in the given layout.
fn component<S: Scalar>(coef: f64, f: Functional, pol: bool, v: &[S]) -> S {
    let e = match f.family() {
        Family::Lda => lda::energy(f, pol, v),
        Family::Gga => gga::energy(f, pol, v),
        Family::Mgga => mgga::energy(f, pol, v),
        Family::HybGga => unreachable!("nested hybrids are not supported"),
    };
    e * S::from_f64(coef)
}

/// Energy density of any implemented functional.
pub(crate) fn energy<S: Scalar>(f: Functional, pol: bool, v: &[S]) -> S {
    if f.family() == Family::HybGga {
        let mut e = S::from_f64(0.0);
        for &(coef, sub) in mixture(f) {
            e = e + component(coef, sub, pol, v);
        }
        e
    } else {
        component(1.0, f, pol, v)
    }
}
