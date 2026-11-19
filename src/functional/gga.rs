//! GGA exchange, GGA correlation and the hybrid functionals built from them.
//!
//! Layout of `v`:
//! * unpolarized: `[n, sigma]`
//! * polarized:   `[n_up, n_down, sigma_uu, sigma_ud, sigma_dd]`
//!
//! The energy is always returned as an energy *density* (hartree), so that
//! differentiating with respect to the density gives `vxc` directly.

use crate::consts::*;
use crate::functional::lda;
use crate::info::Functional;
use crate::scalar::Scalar;

// ---------------------------------------------------------------------------
// Exchange
// ---------------------------------------------------------------------------

/// Per-spin GGA exchange energy density.
#[inline]
fn x_spin<S: Scalar>(f: Functional, n: S, sigma: S) -> S {
    let n43 = n.powf(4.0 / 3.0);
    // xs = |grad n_sigma| / n_sigma^(4/3), p = (X2S * xs)^2 = s^2
    let xs2 = sigma / (n43 * n.powf(4.0 / 3.0)); // sigma / n^(8/3)
    let p = S::from_f64(X2S * X2S) * xs2;

    let enh = match f {
        Functional::GgaXPbe => pbe_enh(p, 0.8040, 0.2195149727645171),
        Functional::GgaXPbeR => pbe_enh(p, 1.245, 0.2195149727645171),
        Functional::GgaXPbeSol => pbe_enh(p, 0.8040, MU_GE),
        Functional::GgaXB88 => S::from_f64(1.0) + b88_m1(xs2),
        Functional::GgaXPw91 => pw91_enh(p.sqrt()),
        _ => unreachable!("{f} is not a GGA exchange functional"),
    };
    -S::from_f64(X_FACTOR_C) * n43 * enh
}

#[inline]
fn pbe_enh<S: Scalar>(p: S, kappa: f64, mu: f64) -> S {
    // 1 + kappa mu s^2 / (kappa + mu s^2), the cancellation-free form used by
    // libxc 7.1.2.
    let m = S::from_f64(mu) * p;
    S::from_f64(1.0) + S::from_f64(kappa) * m / (S::from_f64(kappa) + m)
}

#[inline]
fn b88_m1<S: Scalar>(x2: S) -> S {
    // libxc's B88 is written in terms of the *raw* reduced gradient
    // x = |grad n| / n^(4/3), not the standard s = X2S * x.
    let beta = 0.0042;
    let x = x2.sqrt();
    S::from_f64(beta / X_FACTOR_C) * x2
        / (S::from_f64(1.0) + S::from_f64(6.0 * beta) * x * x.asinh())
}

#[inline]
fn pw91_enh<S: Scalar>(s: S) -> S {
    let s2 = s * s;
    let fnum = S::from_f64(1.0)
        + s * S::from_f64(0.19645) * (S::from_f64(7.7956) * s).asinh()
        + (S::from_f64(0.2743) - S::from_f64(0.1508) * (S::from_f64(-100.0) * s2).exp()) * s2;
    let fden = S::from_f64(1.0)
        + s * S::from_f64(0.19645) * (S::from_f64(7.7956) * s).asinh()
        + S::from_f64(0.004) * s2 * s2;
    fnum / fden
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

/// Per-particle GGA correlation energy.
fn c_eps<S: Scalar>(f: Functional, rs: S, zeta: S, xt: S, xs0: S, xs1: S) -> S {
    match f {
        Functional::GgaCPbe => f_pbe(rs, zeta, xt),
        Functional::GgaCLyp => f_lyp(rs, zeta, xt, xs0, xs1),
        Functional::GgaCP86 => f_p86(rs, zeta, xt, 1.745 * 0.11),
        Functional::GgaCP86Ft => f_p86(rs, zeta, xt, 1.7454151061251239789 * 0.11),
        _ => unreachable!("{f} is not a GGA correlation functional"),
    }
}

#[inline]
fn mphi<S: Scalar>(z: S) -> S {
    S::from_f64(0.5) * (lda::opz(z, 2.0 / 3.0) + lda::opz(-z, 2.0 / 3.0))
}

#[inline]
fn pbe_t<S: Scalar>(rs: S, zeta: S, xt: S) -> S {
    let two_13 = 1.259921049894873164767210607278228350570;
    xt / (S::from_f64(4.0 * two_13) * mphi(zeta) * rs.sqrt())
}

fn f_pbe<S: Scalar>(rs: S, zeta: S, xt: S) -> S {
    let beta = 0.06672455060314922;
    let gamma = (1.0 - std::f64::consts::LN_2) / std::f64::consts::PI.powi(2);

    let eps_pw = lda::f_pw_modified(rs, zeta);
    let phi3 = mphi(zeta).powf(3.0);

    let t = pbe_t(rs, zeta, xt);
    let t2 = t * t;
    let a =
        S::from_f64(beta) / (S::from_f64(gamma) * (-eps_pw / (S::from_f64(gamma) * phi3)).expm1());
    let f1 = t2 + a * t2 * t2;
    let f2 = S::from_f64(beta) * f1 / (S::from_f64(gamma) * (S::from_f64(1.0) + a * f1));
    let f_h = S::from_f64(gamma) * phi3 * f2.log1p();

    eps_pw + f_h
}

#[inline]
fn lyp_cf() -> f64 {
    0.3 * (3.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0)
}

fn f_lyp<S: Scalar>(rs: S, z: S, xt: S, xs0: S, xs1: S) -> S {
    let (a, b, c_, d) = (0.04918, 0.132, 0.2533, 0.349);
    let rr = rs / S::from_f64(RS_FACTOR);
    let cf = lyp_cf();

    let omega =
        S::from_f64(b) * (S::from_f64(-c_) * rr).exp() / (S::from_f64(1.0) + S::from_f64(d) * rr);
    let delta = (S::from_f64(c_) + S::from_f64(d) / (S::from_f64(1.0) + S::from_f64(d) * rr)) * rr;

    let omz2 = S::from_f64(1.0) - z * z; // 1 - zeta^2
    let f1d = S::from_f64(1.0) / (S::from_f64(1.0) + S::from_f64(d) * rr);

    let t1 = -omz2 * f1d;
    let t2 = -(xt * xt)
        * (omz2 * (S::from_f64(47.0) - S::from_f64(7.0) * delta) / S::from_f64(72.0)
            - S::from_f64(2.0 / 3.0));
    let t3 = -S::from_f64(cf / 2.0) * omz2 * (lda::opz(z, 8.0 / 3.0) + lda::opz(-z, 8.0 / 3.0));

    let aux6 = 0.5f64.powf(8.0 / 3.0);
    let aux4 = aux6 / 4.0;
    let aux5 = aux4 / 18.0;

    let xs0sq = xs0 * xs0;
    let xs1sq = xs1 * xs1;
    let p8 = lda::opz(z, 8.0 / 3.0);
    let m8 = lda::opz(-z, 8.0 / 3.0);
    let p11 = lda::opz(z, 11.0 / 3.0);
    let m11 = lda::opz(-z, 11.0 / 3.0);

    let t4 = S::from_f64(aux4)
        * omz2
        * (S::from_f64(2.5) - delta / S::from_f64(18.0))
        * (xs0sq * p8 + xs1sq * m8);
    let t5 = S::from_f64(aux5) * omz2 * (delta - S::from_f64(11.0)) * (xs0sq * p11 + xs1sq * m11);
    let t6 = -S::from_f64(aux6)
        * (S::from_f64(2.0 / 3.0) * (xs0sq * p8 + xs1sq * m8)
            - lda::opz(z, 2.0) * xs1sq * m8 / S::from_f64(4.0)
            - lda::opz(-z, 2.0) * xs0sq * p8 / S::from_f64(4.0));

    S::from_f64(a) * (t1 + omega * (t2 + t3 + t4 + t5 + t6))
}

fn f_p86<S: Scalar>(rs: S, z: S, xt: S, ftilde: f64) -> S {
    // Parameters of the original P86 (libxc `p86_val`).
    let (aa, bb, malpha, mbeta, mgamma, mdelta) =
        (0.001667, 0.002568, 0.023266, 7.389e-6, 8.723, 0.472);

    let dd = (lda::opz(z, 5.0 / 3.0) + lda::opz(-z, 5.0 / 3.0)).sqrt() / S::from_f64(2.0f64.sqrt());
    let rs2 = rs * rs;
    let cc = S::from_f64(aa)
        + (S::from_f64(bb) + S::from_f64(malpha) * rs + S::from_f64(mbeta) * rs2)
            / (S::from_f64(1.0)
                + S::from_f64(mgamma) * rs
                + S::from_f64(mdelta) * rs2
                + S::from_f64(1.0e4 * mbeta) * rs2 * rs);
    let cc_inf = aa + bb;

    let x1 = xt / (rs / S::from_f64(RS_FACTOR)).sqrt();
    let m_phi = S::from_f64(ftilde * cc_inf) / cc * x1;
    let h = x1 * x1 * (-m_phi).exp() * cc / dd;

    lda::f_pz(rs, z) + h
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Energy density contribution of a GGA (or hybrid-GGA component) functional.
pub(crate) fn energy<S: Scalar>(f: Functional, polarized: bool, v: &[S]) -> S {
    let quarter = S::from_f64(0.25);
    let half = S::from_f64(0.5);

    let (na, nb, saa, sab, sbb) = if polarized {
        (v[0], v[1], v[2], v[3], v[4])
    } else {
        (
            v[0] * half,
            v[0] * half,
            v[1] * quarter,
            v[1] * quarter,
            v[1] * quarter,
        )
    };

    match f.kind() {
        crate::info::Kind::Exchange => x_spin(f, na, saa) + x_spin(f, nb, sbb),
        crate::info::Kind::Correlation => {
            let n = na + nb;
            let zeta = (na - nb) / n;
            let rs = S::from_f64(RS_FACTOR) / n.cbrt();
            let n43 = n.powf(4.0 / 3.0);
            let na43 = na.powf(4.0 / 3.0);
            let nb43 = nb.powf(4.0 / 3.0);
            let st = (saa + S::from_f64(2.0) * sab + sbb).max(S::from_f64(0.0));
            let xt = st.sqrt() / n43;
            let xs0 = saa.sqrt() / na43;
            let xs1 = sbb.sqrt() / nb43;
            n * c_eps(f, rs, zeta, xt, xs0, xs1)
        }
        crate::info::Kind::ExchangeCorrelation => {
            // Only hybrids land here; they are expanded by `Xc` before this
            // function is called.  As a fallback, treat the functional as the
            // sum of its exchange and correlation parts if such an entry ever
            // exists.
            unreachable!("{f} must be expanded into components before evaluation")
        }
    }
}
