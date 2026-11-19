//! A tiny forward-mode automatic-differentiation engine.
//!
//! Every functional is written once as a generic function over [`Scalar`].
//! Evaluating it with [`f64`] gives the energy; evaluating it with [`Jet<N>`]
//! gives the energy density together with its gradient and Hessian with
//! respect to the `N` independent variables of the functional family
//! (spin densities, sigma components and, for meta-GGAs, the kinetic-energy
//! densities).  This is how `vxc` and `fxc` are obtained without
//! hand-differentiating every parametrisation, and it is why adding a new
//! functional only requires writing its energy expression.

// The loops below index fixed-size arrays deliberately: the Hessian is a
// dense N x N array and indexing keeps the formulas readable.
#![allow(clippy::needless_range_loop)]

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Scalar type used by the functional implementations.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn from_f64(x: f64) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn powf(self, p: f64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn expm1(self) -> Self;
    fn log1p(self) -> Self;
    fn asinh(self) -> Self;
    fn atan(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    /// Value of the scalar (used for the piecewise clamps of libxc).
    fn value(self) -> f64;
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn cbrt(self) -> Self {
        self.cbrt()
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        self.powf(p)
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn expm1(self) -> Self {
        self.exp_m1()
    }
    #[inline]
    fn log1p(self) -> Self {
        self.ln_1p()
    }
    #[inline]
    fn asinh(self) -> Self {
        self.asinh()
    }
    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn value(self) -> f64 {
        self
    }
}

/// Value, gradient and Hessian with respect to `N` variables.
///
/// The Hessian is stored in full; it is symmetric by construction.
#[derive(Clone, Copy, Debug)]
pub struct Jet<const N: usize> {
    pub v: f64,
    pub g: [f64; N],
    pub h: [[f64; N]; N],
}

impl<const N: usize> Jet<N> {
    pub fn zero() -> Self {
        Self {
            v: 0.0,
            g: [0.0; N],
            h: [[0.0; N]; N],
        }
    }

    /// Variable number `i` with value `v`.
    pub fn seed(v: f64, i: usize) -> Self {
        let mut j = Self::zero();
        j.v = v;
        j.g[i] = 1.0;
        j
    }

    #[inline]
    fn unary<F, G, H>(self, f: F, df: G, d2f: H) -> Self
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
        H: Fn(f64) -> f64,
    {
        let v = f(self.v);
        let fp = df(self.v);
        let fpp = d2f(self.v);
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = fp * self.g[i];
            for j in 0..N {
                h[i][j] = fp * self.h[i][j] + fpp * self.g[i] * self.g[j];
            }
        }
        Self { v, g, h }
    }
}

impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = self.g[i] + rhs.g[i];
            for j in 0..N {
                h[i][j] = self.h[i][j] + rhs.h[i][j];
            }
        }
        Self {
            v: self.v + rhs.v,
            g,
            h,
        }
    }
}

impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = self.g[i] - rhs.g[i];
            for j in 0..N {
                h[i][j] = self.h[i][j] - rhs.h[i][j];
            }
        }
        Self {
            v: self.v - rhs.v,
            g,
            h,
        }
    }
}

impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = -self.g[i];
            for j in 0..N {
                h[i][j] = -self.h[i][j];
            }
        }
        Self { v: -self.v, g, h }
    }
}

impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = self.g[i] * rhs.v + self.v * rhs.g[i];
            for j in 0..N {
                h[i][j] = self.h[i][j] * rhs.v
                    + self.v * rhs.h[i][j]
                    + self.g[i] * rhs.g[j]
                    + rhs.g[i] * self.g[j];
            }
        }
        Self {
            v: self.v * rhs.v,
            g,
            h,
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // a / b = a * (1/b)
        let inv = rhs.recip();
        self * inv
    }
}

impl<const N: usize> Jet<N> {
    #[inline]
    fn recip(self) -> Self {
        // d(1/x) = -1/x^2, d^2(1/x) = 2/x^3
        let v = 1.0 / self.v;
        let fp = -v * v;
        let fpp = 2.0 * v * v * v;
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = fp * self.g[i];
            for j in 0..N {
                h[i][j] = fp * self.h[i][j] + fpp * self.g[i] * self.g[j];
            }
        }
        Self { v, g, h }
    }
}

impl<const N: usize> Scalar for Jet<N> {
    #[inline]
    fn from_f64(x: f64) -> Self {
        let mut j = Self::zero();
        j.v = x;
        j
    }

    #[inline]
    fn sqrt(self) -> Self {
        // Guard the derivative at exactly zero: the functionals are written so
        // that the singular pieces cancel, but the cancellation is only
        // meaningful in exact arithmetic.  A finite (large) derivative keeps
        // the result finite instead of NaN on grid points with zero gradient.
        let v = self.v.sqrt();
        let fp = if self.v > 0.0 {
            0.5 / v
        } else {
            1.0 / SIGMA_EPS
        };
        let fpp = if self.v > 0.0 {
            -0.25 / (v * v * v)
        } else {
            0.0
        };
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = fp * self.g[i];
            for j in 0..N {
                h[i][j] = fp * self.h[i][j] + fpp * self.g[i] * self.g[j];
            }
        }
        Self { v, g, h }
    }

    #[inline]
    fn cbrt(self) -> Self {
        self.unary(
            f64::cbrt,
            |x| 1.0 / (3.0 * x.cbrt() * x.cbrt()),
            |x| -2.0 / (9.0 * x.cbrt().powi(5)),
        )
    }

    #[inline]
    fn powf(self, p: f64) -> Self {
        if p == 0.0 {
            return Self::from_f64(1.0);
        }
        let v = self.v.powf(p);
        let fp = p * self.v.powf(p - 1.0);
        let fpp = p * (p - 1.0) * self.v.powf(p - 2.0);
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for i in 0..N {
            g[i] = fp * self.g[i];
            for j in 0..N {
                h[i][j] = fp * self.h[i][j] + fpp * self.g[i] * self.g[j];
            }
        }
        Self { v, g, h }
    }

    #[inline]
    fn exp(self) -> Self {
        self.unary(f64::exp, f64::exp, f64::exp)
    }

    #[inline]
    fn ln(self) -> Self {
        self.unary(f64::ln, |x| 1.0 / x, |x| -1.0 / (x * x))
    }

    #[inline]
    fn expm1(self) -> Self {
        self.unary(f64::exp_m1, f64::exp, f64::exp)
    }

    #[inline]
    fn log1p(self) -> Self {
        self.unary(
            f64::ln_1p,
            |x| 1.0 / (1.0 + x),
            |x| -1.0 / ((1.0 + x) * (1.0 + x)),
        )
    }

    #[inline]
    fn asinh(self) -> Self {
        self.unary(
            f64::asinh,
            |x| 1.0 / (1.0 + x * x).sqrt(),
            |x| {
                let u = 1.0 + x * x;
                -x / (u * u.sqrt())
            },
        )
    }

    #[inline]
    fn atan(self) -> Self {
        self.unary(
            f64::atan,
            |x| 1.0 / (1.0 + x * x),
            |x| -2.0 * x / ((1.0 + x * x) * (1.0 + x * x)),
        )
    }

    #[inline]
    fn abs(self) -> Self {
        if self.v < 0.0 {
            -self
        } else {
            self
        }
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.v < other.v {
            other
        } else {
            self
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.v > other.v {
            other
        } else {
            self
        }
    }

    #[inline]
    fn value(self) -> f64 {
        self.v
    }
}

/// Small positive scale used by the guarded `sqrt` derivative.
const SIGMA_EPS: f64 = 1e-30;
