//! LDA exchange and correlation functionals.
//!
//! The formulas are direct ports of the SymPy sources shipped with libxc
//! 7.1.2 (`python/lda_exc/*.py` and `python/vwn.py`).

use crate::consts::*;
use crate::info::Functional;
use crate::scalar::Scalar;

/// libxc's `opz_pow_n(z, n) = (1 + z)^n` with the zeta clamp.
#[inline]
pub(crate) fn opz<S: Scalar>(z: S, n: f64) -> S {
    let t = S::from_f64(1.0) + z;
    if t.value() <= ZETA_THRESHOLD {
        S::from_f64(ZETA_THRESHOLD.powf(n))
    } else {
        t.powf(n)
    }
}

/// `(1 + z)^n - 1` evaluated with libxc's clamp and an expm1/log1p chain.
#[inline]
fn opz_m1<S: Scalar>(z: S, n: f64) -> S {
    if z.value() <= ZETA_THRESHOLD - 1.0 {
        S::from_f64(ZETA_THRESHOLD.powf(n) - 1.0)
    } else {
        (z.log1p() * S::from_f64(n)).expm1()
    }
}

/// Spin-interpolation function `f(zeta)` (libxc `f_zeta`).
#[inline]
pub(crate) fn f_zeta<S: Scalar>(z: S) -> S {
    let four_thirds = 4.0 / 3.0;
    (opz_m1(z, four_thirds) + opz_m1(-z, four_thirds)) / S::from_f64(FZETA_FACTOR)
}

/// `1 - zeta^n`, factored to avoid cancellation (libxc `one_minus_z_pow_n`).
#[inline]
fn one_minus_z_pow_n<S: Scalar>(z: S, n: u32) -> S {
    let one = S::from_f64(1.0);
    match n {
        2 => (one - z) * (one + z),
        4 => (one - z) * (one + z) * (one + z * z),
        _ => one - z.powf(n as f64),
    }
}

/// Per-spin LDA exchange energy density: `-X_FACTOR_C n^(4/3)`.
#[inline]
pub(crate) fn x_spin<S: Scalar>(n: S) -> S {
    -S::from_f64(X_FACTOR_C) * n.powf(4.0 / 3.0)
}

// ---------------------------------------------------------------------------
// Perdew-Wang 92
// ---------------------------------------------------------------------------

struct PwParams {
    a: [f64; 3],
    alpha1: [f64; 3],
    beta1: [f64; 3],
    beta2: [f64; 3],
    beta3: [f64; 3],
    beta4: [f64; 3],
    fz20: f64,
}

const PW: PwParams = PwParams {
    a: [0.031091, 0.015545, 0.016887],
    alpha1: [0.21370, 0.20548, 0.11125],
    beta1: [7.5957, 14.1189, 10.357],
    beta2: [3.5876, 6.1977, 3.6231],
    beta3: [1.6382, 3.3662, 0.88026],
    beta4: [0.49294, 0.62517, 0.49671],
    fz20: 1.709921,
};

const PW_MOD: PwParams = PwParams {
    a: [0.0310907, 0.01554535, 0.0168869],
    alpha1: [0.21370, 0.20548, 0.11125],
    beta1: [7.5957, 14.1189, 10.357],
    beta2: [3.5876, 6.1977, 3.6231],
    beta3: [1.6382, 3.3662, 0.88026],
    beta4: [0.49294, 0.62517, 0.49671],
    fz20: 1.709920934161365617563962776245,
};

#[inline]
fn pw_g_aux<S: Scalar>(p: &PwParams, k: usize, rs: S) -> S {
    S::from_f64(p.beta1[k]) * rs.sqrt()
        + S::from_f64(p.beta2[k]) * rs
        + S::from_f64(p.beta3[k]) * rs.powf(1.5)
        + S::from_f64(p.beta4[k]) * rs * rs
}

#[inline]
fn pw_g<S: Scalar>(p: &PwParams, k: usize, rs: S) -> S {
    let a = S::from_f64(p.a[k]);
    -S::from_f64(2.0)
        * a
        * (S::from_f64(1.0) + S::from_f64(p.alpha1[k]) * rs)
        * (S::from_f64(1.0) / (S::from_f64(2.0) * a * pw_g_aux(p, k, rs))).log1p()
}

/// Per-particle PW92 correlation energy.
#[inline]
pub(crate) fn f_pw<S: Scalar>(rs: S, z: S) -> S {
    f_pw_impl(&PW, rs, z)
}

pub(crate) fn f_pw_modified<S: Scalar>(rs: S, z: S) -> S {
    f_pw_impl(&PW_MOD, rs, z)
}

fn f_pw_impl<S: Scalar>(p: &PwParams, rs: S, z: S) -> S {
    let g1 = pw_g(p, 0, rs);
    let g2 = pw_g(p, 1, rs);
    let g3 = pw_g(p, 2, rs);
    let fz = f_zeta(z);
    g1 + z.powf(4.0) * fz * (g2 - g1 + g3 / S::from_f64(p.fz20)) - fz * g3 / S::from_f64(p.fz20)
}

// ---------------------------------------------------------------------------
// Perdew-Zunger 81
// ---------------------------------------------------------------------------

const PZ_GAMMA: [f64; 2] = [-0.1423, -0.0843];
const PZ_BETA1: [f64; 2] = [1.0529, 1.3981];
const PZ_BETA2: [f64; 2] = [0.3334, 0.2611];
const PZ_A: [f64; 2] = [0.0311, 0.01555];
const PZ_B: [f64; 2] = [-0.048, -0.0269];
const PZ_C: [f64; 2] = [0.0020, 0.0007];
const PZ_D: [f64; 2] = [-0.0116, -0.0048];

#[inline]
fn pz_ec<S: Scalar>(i: usize, rs: S) -> S {
    let one = S::from_f64(1.0);
    if rs.value() >= 1.0 {
        S::from_f64(PZ_GAMMA[i])
            / (one + S::from_f64(PZ_BETA1[i]) * rs.sqrt() + S::from_f64(PZ_BETA2[i]) * rs)
    } else {
        S::from_f64(PZ_A[i]) * rs.ln()
            + S::from_f64(PZ_B[i])
            + S::from_f64(PZ_C[i]) * rs * rs.ln()
            + S::from_f64(PZ_D[i]) * rs
    }
}

/// Per-particle PZ81 correlation energy.
#[inline]
pub(crate) fn f_pz<S: Scalar>(rs: S, z: S) -> S {
    let e0 = pz_ec(0, rs);
    let e1 = pz_ec(1, rs);
    e0 + (e1 - e0) * f_zeta(z)
}

// ---------------------------------------------------------------------------
// VWN
// ---------------------------------------------------------------------------

const A_VWN: [f64; 3] = [
    0.0310907,
    0.01554535,
    -0.016886863940389630, // -1/(6 pi^2)
];
const B_VWN: [f64; 3] = [3.72744, 7.06042, 1.13107];
const C_VWN: [f64; 3] = [12.9352, 18.0578, 13.0045];
const X0_VWN: [f64; 3] = [-0.10498, -0.32500, -0.0047584];

const A_RPA: [f64; 3] = [
    0.0310907,
    0.01554535,
    -0.016886863940389630, // -1/(6 pi^2)
];
const B_RPA: [f64; 3] = [13.0720, 20.1231, 1.06835];
const C_RPA: [f64; 3] = [42.7198, 101.578, 11.4813];
const X0_RPA: [f64; 3] = [-0.409286, -0.743294, -0.228344];

/// `4 / (9 (2^(1/3) - 1))`.
const FPP_VWN: f64 = 4.0 / (9.0 * (1.259921049894873164767210607278228350570 - 1.0));

#[inline]
fn f_aux<S: Scalar>(a: f64, b: f64, c_: f64, x0: f64, rs: S) -> S {
    let q = S::from_f64(4.0 * c_ - b * b).sqrt();
    let f1 = S::from_f64(2.0 * b) / q;
    let f2 = S::from_f64(b * x0) / S::from_f64(x0 * x0 + b * x0 + c_);
    let f3 = S::from_f64(2.0 * (2.0 * x0 + b)) / q;
    let fx = rs + S::from_f64(b) * rs.sqrt() + S::from_f64(c_);
    let sq = rs.sqrt();
    S::from_f64(a)
        * (((rs - fx) / fx).log1p()
            + (f1 - f2 * f3) * (q / (S::from_f64(2.0) * sq + S::from_f64(b))).atan()
            - f2 * (((sq - S::from_f64(x0)) * (sq - S::from_f64(x0)) - fx) / fx).log1p())
}

#[inline]
fn vwn_dmc<S: Scalar>(rs: S) -> S {
    f_aux(A_VWN[1], B_VWN[1], C_VWN[1], X0_VWN[1], rs)
        - f_aux(A_VWN[0], B_VWN[0], C_VWN[0], X0_VWN[0], rs)
}

#[inline]
fn vwn_drpa<S: Scalar>(rs: S) -> S {
    f_aux(A_RPA[1], B_RPA[1], C_RPA[1], X0_RPA[1], rs)
        - f_aux(A_RPA[0], B_RPA[0], C_RPA[0], X0_RPA[0], rs)
}

/// Per-particle VWN correlation energy.  `variant` follows the libxc numbers:
/// 7 = VWN5, 8 = RPA, 28..31 = VWN1..VWN4.
fn f_vwn<S: Scalar>(variant: u32, rs: S, z: S) -> S {
    let (a1, b1, c1, x01) = (A_VWN[0], B_VWN[0], C_VWN[0], X0_VWN[0]);
    let fz = f_zeta(z);
    let f0 = f_aux(a1, b1, c1, x01, rs);
    match variant {
        7 => {
            let f3 = f_aux(A_VWN[2], B_VWN[2], C_VWN[2], X0_VWN[2], rs);
            let om = one_minus_z_pow_n(z, 4);
            f0 + f3 * fz * om / S::from_f64(FPP_VWN) + vwn_dmc(rs) * fz * z.powf(4.0)
        }
        8 => {
            let f0r = f_aux(A_RPA[0], B_RPA[0], C_RPA[0], X0_RPA[0], rs);
            let f1r = f_aux(A_RPA[1], B_RPA[1], C_RPA[1], X0_RPA[1], rs);
            f0r * (S::from_f64(1.0) - fz) + f1r * fz
        }
        28 => {
            let f1 = f_aux(A_VWN[1], B_VWN[1], C_VWN[1], X0_VWN[1], rs);
            f0 * (S::from_f64(1.0) - fz) + f1 * fz
        }
        29 => {
            let f3r = f_aux(A_RPA[2], B_RPA[2], C_RPA[2], X0_RPA[2], rs);
            let om = one_minus_z_pow_n(z, 4);
            f0 + f3r * fz * om / S::from_f64(FPP_VWN) - vwn_drpa(rs) * fz * om + vwn_dmc(rs) * fz
        }
        30 => {
            let f3r = f_aux(A_RPA[2], B_RPA[2], C_RPA[2], X0_RPA[2], rs);
            let om = one_minus_z_pow_n(z, 4);
            f0 + vwn_dmc(rs) / vwn_drpa(rs) * f3r * fz * om / S::from_f64(FPP_VWN)
                + vwn_dmc(rs) * fz * z.powf(4.0)
        }
        31 => {
            let f3r = f_aux(A_RPA[2], B_RPA[2], C_RPA[2], X0_RPA[2], rs);
            let om = one_minus_z_pow_n(z, 4);
            f0 + f3r * fz * om / S::from_f64(FPP_VWN) + vwn_dmc(rs) * fz * z.powf(4.0)
        }
        _ => unreachable!("unknown VWN variant {variant}"),
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Energy density contribution of an LDA functional.
///
/// Layout of `v`: unpolarized `[n]`, polarized `[n_up, n_down]`.
pub(crate) fn energy<S: Scalar>(f: Functional, polarized: bool, v: &[S]) -> S {
    let half = S::from_f64(0.5);
    let (na, nb) = if polarized {
        (v[0], v[1])
    } else {
        (v[0] * half, v[0] * half)
    };
    let n = na + nb;
    let rs = S::from_f64(RS_FACTOR) / n.cbrt();
    let zeta = (na - nb) / n;

    match f {
        Functional::LdaX => x_spin(na) + x_spin(nb),
        Functional::LdaCVwn => n * f_vwn(7, rs, zeta),
        Functional::LdaCVwnRpa => n * f_vwn(8, rs, zeta),
        Functional::LdaCPz => n * f_pz(rs, zeta),
        Functional::LdaCPw => n * f_pw(rs, zeta),
        Functional::LdaCPwMod => n * f_pw_modified(rs, zeta),
        Functional::LdaCVwn1 => n * f_vwn(28, rs, zeta),
        Functional::LdaCVwn2 => n * f_vwn(29, rs, zeta),
        Functional::LdaCVwn3 => n * f_vwn(30, rs, zeta),
        Functional::LdaCVwn4 => n * f_vwn(31, rs, zeta),
        _ => unreachable!("{f} is not an LDA functional"),
    }
}
