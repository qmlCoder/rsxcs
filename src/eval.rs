//! Evaluation driver: turns the generic energy expressions into the
//! libxc-compatible `exc`, `vxc` and `fxc` arrays.

use crate::consts::*;
use crate::functional;
use crate::info::{Family, Functional, Info, Kind, Spin};
use crate::scalar::Jet;

/// Highest derivative to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Energy only (`zk` / `exc`).
    Exc,
    /// Energy and first derivatives (`vrho`, `vsigma`, `vtau`).
    Vxc,
    /// Energy, first and second derivatives.
    Fxc,
}

/// Second derivatives, in libxc's layout.  Arrays that do not apply to the
/// functional family are left empty.
#[derive(Debug, Clone, Default)]
pub struct Fxc {
    /// `nspin=1`: 1 entry.  `nspin=2`: `(uu, ud, dd)`.
    pub v2rho2: Vec<f64>,
    /// `nspin=1`: 1 entry.  `nspin=2`: `(u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)`.
    pub v2rhosigma: Vec<f64>,
    /// `nspin=1`: 1 entry.  `nspin=2`: `(uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)`.
    pub v2sigma2: Vec<f64>,
    /// Meta-GGAs only (currently always zero: no implemented functional
    /// depends on the Laplacian).
    pub v2rholapl: Vec<f64>,
    /// Meta-GGAs only.  `nspin=1`: 1 entry.  `nspin=2`:
    /// `(u_u, u_d, d_u, d_d)`.
    pub v2rhotau: Vec<f64>,
    /// Meta-GGAs only (currently always zero).
    pub v2sigmalapl: Vec<f64>,
    /// Meta-GGAs only.  `nspin=1`: 1 entry.  `nspin=2`:
    /// `(uu_u, uu_d, ud_u, ud_d, dd_u, dd_d)`.
    pub v2sigmatau: Vec<f64>,
    /// Meta-GGAs only (currently always zero).
    pub v2lapl2: Vec<f64>,
    /// Meta-GGAs only (currently always zero).
    pub v2lapltau: Vec<f64>,
    /// Meta-GGAs only.  `nspin=1`: 1 entry.  `nspin=2`: `(uu, ud, dd)`.
    pub v2tau2: Vec<f64>,
}

/// Result of an evaluation.  `exc` is always filled; `vxc` is empty for
/// [`Order::Exc`]; `fxc` is present only for [`Order::Fxc`].
#[derive(Debug, Clone, Default)]
pub struct Evaluation {
    pub exc: Vec<f64>,
    /// Derivatives with respect to the spin densities.
    pub vxc: Vec<f64>,
    /// Derivatives with respect to the sigma components (GGA and meta-GGA).
    pub vsigma: Vec<f64>,
    /// Derivatives with respect to tau (meta-GGA).
    pub vtau: Vec<f64>,
    pub fxc: Option<Fxc>,
}

/// Errors returned by [`Xc::eval`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    UnknownId(u32),
    UnknownName(String),
    BadLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    MissingInput(&'static str),
    NoPoints,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownId(id) => write!(f, "unknown functional id {id}"),
            Error::UnknownName(name) => write!(f, "unknown functional name '{name}'"),
            Error::BadLength {
                what,
                expected,
                got,
            } => write!(f, "bad length for {what}: expected {expected}, got {got}"),
            Error::MissingInput(what) => write!(f, "missing required input '{what}'"),
            Error::NoPoints => write!(f, "the input contains no grid points"),
        }
    }
}

impl std::error::Error for Error {}

/// An exchange-correlation functional ready to be evaluated.
///
/// ```
/// use rsxcs::{Order, Spin, Xc};
/// let xc = Xc::new(101).unwrap(); // gga_x_pbe
/// let rho = [0.5];
/// let sigma = [0.05];
/// let out = xc
///     .eval(Spin::Unpolarized, &rho, Some(&sigma), None, None, Order::Vxc)
///     .unwrap();
/// assert!(out.exc[0] < 0.0);
/// assert_eq!(out.vxc.len(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xc {
    functional: Functional,
}

impl Xc {
    /// Build a functional from its libxc number.
    pub fn new(id: u32) -> Result<Self, Error> {
        Functional::from_id(id)
            .map(Self::from_functional)
            .ok_or(Error::UnknownId(id))
    }

    /// Build a functional from its libxc short name (case-insensitive).
    pub fn from_name(name: &str) -> Result<Self, Error> {
        Functional::from_name(name)
            .map(Self::from_functional)
            .ok_or_else(|| Error::UnknownName(name.to_string()))
    }

    pub fn from_functional(functional: Functional) -> Self {
        Self { functional }
    }

    #[inline]
    pub fn functional(&self) -> Functional {
        self.functional
    }

    #[inline]
    pub fn info(&self) -> Info {
        self.functional.info()
    }

    #[inline]
    pub fn family(&self) -> Family {
        self.functional.family()
    }

    #[inline]
    pub fn kind(&self) -> Kind {
        self.functional.kind()
    }

    /// Number of independent variables per grid point for this functional and
    /// spin treatment.
    pub fn n_vars(&self, spin: Spin) -> usize {
        let ns = spin.n_spin();
        match self.family().semilocal() {
            Family::Lda => ns,
            Family::Gga => ns + if ns == 2 { 3 } else { 1 },
            Family::Mgga => 2 * ns + if ns == 2 { 3 } else { 1 },
            Family::HybGga => unreachable!(),
        }
    }

    /// Evaluate the functional on a grid.
    ///
    /// * `rho` has length `npoints * nspin`.
    /// * `sigma` (GGA and meta-GGA) has length `npoints * (1 or 3)` with the
    ///   ordering `(uu)`, or `(uu, ud, dd)`.
    /// * `lapl` is accepted for API compatibility but currently ignored (none
    ///   of the implemented functionals depend on it).
    /// * `tau` (meta-GGA only) has length `npoints * nspin`.
    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        spin: Spin,
        rho: &[f64],
        sigma: Option<&[f64]>,
        lapl: Option<&[f64]>,
        tau: Option<&[f64]>,
        order: Order,
    ) -> Result<Evaluation, Error> {
        let ns = spin.n_spin();
        let pol = spin.is_polarized();
        if rho.len() % ns != 0 {
            return Err(Error::BadLength {
                what: "rho",
                expected: rho.len() - rho.len() % ns,
                got: rho.len(),
            });
        }
        let np = rho.len() / ns;
        if np == 0 {
            return Err(Error::NoPoints);
        }

        let family = self.family().semilocal();
        let n_sigma = if pol { 3 } else { 1 };

        let sigma = match family {
            Family::Lda => sigma,
            Family::Gga | Family::Mgga => Some(sigma.ok_or(Error::MissingInput("sigma"))?),
            Family::HybGga => unreachable!(),
        };
        if let Some(s) = sigma {
            let expected = np * n_sigma;
            if s.len() != expected {
                return Err(Error::BadLength {
                    what: "sigma",
                    expected,
                    got: s.len(),
                });
            }
        }
        let tau = match family {
            Family::Mgga => Some(tau.ok_or(Error::MissingInput("tau"))?),
            _ => tau,
        };
        if let Some(t) = tau {
            let expected = np * ns;
            if t.len() != expected {
                return Err(Error::BadLength {
                    what: "tau",
                    expected,
                    got: t.len(),
                });
            }
        }
        if let Some(l) = lapl {
            let expected = np * ns;
            if l.len() != expected {
                return Err(Error::BadLength {
                    what: "lapl",
                    expected,
                    got: l.len(),
                });
            }
        }

        let mut out = Evaluation {
            exc: vec![0.0; np],
            vxc: if order == Order::Exc {
                Vec::new()
            } else {
                vec![0.0; np * ns]
            },
            vsigma: if order == Order::Exc || family == Family::Lda {
                Vec::new()
            } else {
                vec![0.0; np * n_sigma]
            },
            vtau: if order == Order::Exc || family != Family::Mgga {
                Vec::new()
            } else {
                vec![0.0; np * ns]
            },
            fxc: if order == Order::Fxc {
                Some(self.alloc_fxc(np, ns))
            } else {
                None
            },
        };

        match (family, pol) {
            (Family::Lda, false) => self.run::<1>(&spin, np, rho, sigma, tau, &mut out),
            (Family::Lda, true) => self.run::<2>(&spin, np, rho, sigma, tau, &mut out),
            (Family::Gga, false) => self.run::<2>(&spin, np, rho, sigma, tau, &mut out),
            (Family::Gga, true) => self.run::<5>(&spin, np, rho, sigma, tau, &mut out),
            (Family::Mgga, false) => self.run::<3>(&spin, np, rho, sigma, tau, &mut out),
            (Family::Mgga, true) => self.run::<7>(&spin, np, rho, sigma, tau, &mut out),
            (Family::HybGga, _) => unreachable!(),
        }

        Ok(out)
    }

    fn alloc_fxc(&self, np: usize, ns: usize) -> Fxc {
        let pol = ns == 2;
        let lda = Family::Lda == self.family().semilocal();
        let mgga = Family::Mgga == self.family().semilocal();
        let n2 = if pol { 3 } else { 1 };
        let nrs = if pol { 6 } else { 1 };
        let nlt = if pol { 4 } else { 1 };
        Fxc {
            v2rho2: vec![0.0; np * n2],
            v2rhosigma: if lda { Vec::new() } else { vec![0.0; np * nrs] },
            v2sigma2: if lda { Vec::new() } else { vec![0.0; np * nrs] },
            v2rholapl: if mgga {
                vec![0.0; np * nlt]
            } else {
                Vec::new()
            },
            v2rhotau: if mgga {
                vec![0.0; np * nlt]
            } else {
                Vec::new()
            },
            v2sigmalapl: if mgga {
                vec![0.0; np * nrs]
            } else {
                Vec::new()
            },
            v2sigmatau: if mgga {
                vec![0.0; np * nrs]
            } else {
                Vec::new()
            },
            v2lapl2: if mgga { vec![0.0; np * n2] } else { Vec::new() },
            v2lapltau: if mgga {
                vec![0.0; np * nlt]
            } else {
                Vec::new()
            },
            v2tau2: if mgga { vec![0.0; np * n2] } else { Vec::new() },
        }
    }

    /// Evaluate one grid point, returning the energy density, gradient and
    /// Hessian with respect to the `N` variables of the layout.
    fn point<const N: usize>(&self, pol: bool, x: [f64; N]) -> (f64, [f64; N], [[f64; N]; N]) {
        let v: [Jet<N>; N] = std::array::from_fn(|i| Jet::seed(x[i], i));
        let e = functional::energy(self.functional, pol, &v);
        (e.v, e.g, e.h)
    }

    fn run<const N: usize>(
        &self,
        spin: &Spin,
        np: usize,
        rho: &[f64],
        sigma: Option<&[f64]>,
        tau: Option<&[f64]>,
        out: &mut Evaluation,
    ) {
        let pol = spin.is_polarized();
        let ns = spin.n_spin();

        for ip in 0..np {
            let x = match self.gather::<N>(pol, ns, ip, rho, sigma, tau) {
                Some(x) => x,
                None => continue,
            };
            let (e, g, h) = self.point::<N>(pol, x);
            let n_tot = if pol { x[0] + x[1] } else { x[0] };
            let h_flat: Vec<f64> = h.iter().flatten().copied().collect();
            self.emit(ip, ns, n_tot, e, &g, &h_flat, out);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gather<const N: usize>(
        &self,
        pol: bool,
        ns: usize,
        ip: usize,
        rho: &[f64],
        sigma: Option<&[f64]>,
        tau: Option<&[f64]>,
    ) -> Option<[f64; N]> {
        let mut x = [0.0; N];
        let family = self.family().semilocal();

        if !pol {
            let n = rho[ip];
            if n < DENS_THRESHOLD {
                return None;
            }
            x[0] = n.max(DENS_THRESHOLD);
            if family != Family::Lda {
                x[1] = sigma.unwrap()[ip].max(SIGMA_FLOOR);
            }
            if family == Family::Mgga {
                x[2] = tau.unwrap()[ip].max(TAU_FLOOR);
            }
        } else {
            let na = rho[2 * ip];
            let nb = rho[2 * ip + 1];
            if na + nb < DENS_THRESHOLD {
                return None;
            }
            let na = na.max(DENS_THRESHOLD);
            let nb = nb.max(DENS_THRESHOLD);
            x[0] = na;
            x[1] = nb;
            if family != Family::Lda {
                let s = &sigma.unwrap()[3 * ip..3 * ip + 3];
                let saa = s[0].max(SIGMA_FLOOR);
                let sbb = s[2].max(SIGMA_FLOOR);
                let half = 0.5 * (saa + sbb);
                let sab = s[1].clamp(-half, half);
                x[2] = saa;
                x[3] = sab;
                x[4] = sbb;
            }
            if family == Family::Mgga {
                x[5] = tau.unwrap()[2 * ip].max(TAU_FLOOR);
                x[6] = tau.unwrap()[2 * ip + 1].max(TAU_FLOOR);
            }
        }
        let _ = ns;
        Some(x)
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn emit(
        &self,
        ip: usize,
        ns: usize,
        n_tot: f64,
        e: f64,
        g: &[f64],
        h: &[f64],
        out: &mut Evaluation,
    ) {
        let pol = ns == 2;
        let nv = g.len();
        let hij = |i: usize, j: usize| h[i * nv + j];

        out.exc[ip] = e / n_tot;

        if !out.vxc.is_empty() {
            if pol {
                out.vxc[2 * ip] = g[0];
                out.vxc[2 * ip + 1] = g[1];
            } else {
                out.vxc[ip] = g[0];
            }
        }

        if !out.vsigma.is_empty() {
            if pol {
                out.vsigma[3 * ip] = g[2];
                out.vsigma[3 * ip + 1] = g[3];
                out.vsigma[3 * ip + 2] = g[4];
            } else {
                out.vsigma[ip] = g[1];
            }
        }

        if !out.vtau.is_empty() {
            if pol {
                out.vtau[2 * ip] = g[5];
                out.vtau[2 * ip + 1] = g[6];
            } else {
                out.vtau[ip] = g[2];
            }
        }

        let Some(fxc) = out.fxc.as_mut() else {
            return;
        };

        // v2rho2
        if pol {
            fxc.v2rho2[3 * ip] = hij(0, 0);
            fxc.v2rho2[3 * ip + 1] = hij(0, 1);
            fxc.v2rho2[3 * ip + 2] = hij(1, 1);
        } else {
            fxc.v2rho2[ip] = hij(0, 0);
        }

        let family = self.family().semilocal();
        if family == Family::Lda {
            return;
        }

        // sigma indices: unpolarized -> 1, polarized -> 2,3,4
        let sig: &[usize] = if pol { &[2, 3, 4] } else { &[1] };

        if !fxc.v2rhosigma.is_empty() {
            if pol {
                for ir in 0..2 {
                    for (is, &js) in sig.iter().enumerate() {
                        fxc.v2rhosigma[6 * ip + ir * 3 + is] = hij(ir, js);
                    }
                }
            } else {
                fxc.v2rhosigma[ip] = hij(0, sig[0]);
            }
        }

        if !fxc.v2sigma2.is_empty() {
            if pol {
                let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
                for (k, (i, j)) in pairs.iter().enumerate() {
                    fxc.v2sigma2[6 * ip + k] = hij(sig[*i], sig[*j]);
                }
            } else {
                fxc.v2sigma2[ip] = hij(sig[0], sig[0]);
            }
        }

        if family != Family::Mgga {
            return;
        }

        // tau indices: unpolarized vars [n, sigma, tau] -> 2;
        // polarized vars [na, nb, saa, sab, sbb, ta, tb] -> 5, 6.
        let tau: &[usize] = if pol { &[5, 6] } else { &[2] };

        if !fxc.v2rhotau.is_empty() {
            if pol {
                for ir in 0..2 {
                    for it in 0..2 {
                        fxc.v2rhotau[4 * ip + ir * 2 + it] = hij(ir, tau[it]);
                    }
                }
            } else {
                fxc.v2rhotau[ip] = hij(0, tau[0]);
            }
        }

        if !fxc.v2sigmatau.is_empty() {
            if pol {
                for (is, &js) in sig.iter().enumerate() {
                    for it in 0..2 {
                        fxc.v2sigmatau[6 * ip + is * 2 + it] = hij(js, tau[it]);
                    }
                }
            } else {
                fxc.v2sigmatau[ip] = hij(sig[0], tau[0]);
            }
        }

        if !fxc.v2tau2.is_empty() {
            if pol {
                fxc.v2tau2[3 * ip] = hij(tau[0], tau[0]);
                fxc.v2tau2[3 * ip + 1] = hij(tau[0], tau[1]);
                fxc.v2tau2[3 * ip + 2] = hij(tau[1], tau[1]);
            } else {
                fxc.v2tau2[ip] = hij(tau[0], tau[0]);
            }
        }
    }
}
