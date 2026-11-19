//! Safe wrapper around the c2rust-translated libxc 7.1.2.
//!
//! The generated crate (`../libxc`) is a mechanical C-to-Rust translation of
//! all 332 translation units of libxc 7.1.2.  Because every C file gets its
//! own copy of the header types, the wrapper converts between the
//! layout-identical `xc_func_type` definitions with raw-pointer casts.
//!
//! Build with the pinned toolchain in `c2rust/rust-toolchain.toml`:
//!
//! ```text
//! cd c2rust && cargo +nightly-2023-04-15 run --example compare
//! ```
//!
//! This gives access to *all* libxc functionals and their analytic FHC
//! derivatives (the generated code was built with `XC_MAXORDER=2`).

use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr;

use libxc_c2rust::src;

/// Spin treatment.  Unlike `rsxcs`, a libxc functional object is created for
/// one spin treatment, so it is passed to [`Xc::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spin {
    Unpolarized,
    Polarized,
}

impl Spin {
    pub fn n_spin(self) -> usize {
        match self {
            Spin::Unpolarized => 1,
            Spin::Polarized => 2,
        }
    }
    fn as_c(self) -> c_int {
        self.n_spin() as c_int
    }
}

/// Semilocal family used for evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Lda,
    Gga,
    Mgga,
}

/// Highest derivative to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Order {
    Exc,
    Vxc,
    Fxc,
}

/// Second derivatives in libxc's layout.  All arrays are allocated with the
/// dimension reported by libxc (`func.dim`), so the ordering is exactly the
/// one documented by libxc/pylibxc.
#[derive(Debug, Clone, Default)]
pub struct Second {
    pub v2rho2: Vec<f64>,
    pub v2rhosigma: Vec<f64>,
    pub v2rholapl: Vec<f64>,
    pub v2rhotau: Vec<f64>,
    pub v2sigma2: Vec<f64>,
    pub v2sigmalapl: Vec<f64>,
    pub v2sigmatau: Vec<f64>,
    pub v2lapl2: Vec<f64>,
    pub v2lapltau: Vec<f64>,
    pub v2tau2: Vec<f64>,
}

impl Second {
    fn alloc(d: &src::functionals::xc_dimensions, np: usize) -> Self {
        let n = |v: c_int| np * v.max(0) as usize;
        Self {
            v2rho2: vec![0.0; n(d.v2rho2)],
            v2rhosigma: vec![0.0; n(d.v2rhosigma)],
            v2rholapl: vec![0.0; n(d.v2rholapl)],
            v2rhotau: vec![0.0; n(d.v2rhotau)],
            v2sigma2: vec![0.0; n(d.v2sigma2)],
            v2sigmalapl: vec![0.0; n(d.v2sigmalapl)],
            v2sigmatau: vec![0.0; n(d.v2sigmatau)],
            v2lapl2: vec![0.0; n(d.v2lapl2)],
            v2lapltau: vec![0.0; n(d.v2lapltau)],
            v2tau2: vec![0.0; n(d.v2tau2)],
        }
    }
}

/// Evaluation result.
#[derive(Debug, Clone, Default)]
pub struct Output {
    /// Energy per particle, `npoints` values.
    pub exc: Vec<f64>,
    pub vxc: Vec<f64>,
    pub vsigma: Vec<f64>,
    pub vlapl: Vec<f64>,
    pub vtau: Vec<f64>,
    pub fxc: Option<Second>,
}

#[derive(Debug)]
pub enum Error {
    Init(c_int),
    UnknownName(String),
    BadLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    MissingInput(&'static str),
    UnsupportedFamily(c_int),
    NoPoints,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Init(code) => write!(f, "xc_func_init failed with code {code}"),
            Error::UnknownName(name) => write!(f, "unknown functional name '{name}'"),
            Error::BadLength {
                what,
                expected,
                got,
            } => write!(f, "bad length for {what}: expected {expected}, got {got}"),
            Error::MissingInput(what) => write!(f, "missing required input '{what}'"),
            Error::UnsupportedFamily(fam) => write!(f, "unsupported libxc family {fam}"),
            Error::NoPoints => write!(f, "the input contains no grid points"),
        }
    }
}

impl std::error::Error for Error {}

/// A libxc functional initialized for one spin treatment.
pub struct Xc {
    func: Box<src::functionals::xc_func_type>,
    family: Family,
    nspin: usize,
}

impl Xc {
    /// Create a functional from its libxc number.
    pub fn new(id: u32, spin: Spin) -> Result<Self, Error> {
        let mut func: Box<src::functionals::xc_func_type> =
            Box::new(unsafe { std::mem::zeroed() });
        let rc = unsafe {
            src::functionals::xc_func_init(&mut *func, id as c_int, spin.as_c())
        };
        if rc != 0 {
            return Err(Error::Init(rc));
        }
        let family_code = unsafe { (*(*func).info).family };
        let family = match family_code {
            src::functionals::XC_FAMILY_LDA | src::functionals::XC_FAMILY_HYB_LDA => Family::Lda,
            src::functionals::XC_FAMILY_GGA | src::functionals::XC_FAMILY_HYB_GGA => Family::Gga,
            src::functionals::XC_FAMILY_MGGA | src::functionals::XC_FAMILY_HYB_MGGA => Family::Mgga,
            other => {
                unsafe { src::functionals::xc_func_end(&mut *func) };
                return Err(Error::UnsupportedFamily(other));
            }
        };
        Ok(Self {
            func,
            family,
            nspin: spin.n_spin(),
        })
    }

    /// Create a functional from its libxc short name (case-insensitive).
    pub fn from_name(name: &str, spin: Spin) -> Result<Self, Error> {
        let cname = CString::new(name).map_err(|_| Error::UnknownName(name.to_string()))?;
        let id = unsafe { src::functionals::xc_functional_get_number(cname.as_ptr()) };
        if id < 0 {
            return Err(Error::UnknownName(name.to_string()));
        }
        Self::new(id as u32, spin)
    }

    pub fn id(&self) -> u32 {
        unsafe { (*(*self.func).info).number as u32 }
    }

    pub fn name(&self) -> &str {
        unsafe {
            let p = (*(*self.func).info).name;
            if p.is_null() {
                return "";
            }
            CStr::from_ptr(p as *const c_char).to_str().unwrap_or("")
        }
    }

    pub fn family(&self) -> Family {
        self.family
    }

    pub fn n_spin(&self) -> usize {
        self.nspin
    }

    /// Evaluate on a grid.  Input layout is identical to libxc/`pylibxc`:
    /// `rho` is `npoints*nspin`, `sigma` is `npoints*(1 or 3)`, `tau` is
    /// `npoints*nspin`, and `lapl` is optional.
    pub fn eval(
        &self,
        rho: &[f64],
        sigma: Option<&[f64]>,
        lapl: Option<&[f64]>,
        tau: Option<&[f64]>,
        order: Order,
    ) -> Result<Output, Error> {
        let ns = self.nspin;
        if rho.len() % ns != 0 {
            return Err(Error::BadLength {
                what: "rho",
                expected: rho.len() / ns * ns,
                got: rho.len(),
            });
        }
        let np = rho.len() / ns;
        if np == 0 {
            return Err(Error::NoPoints);
        }
        let d = &self.func.dim;
        let nsigma = d.sigma.max(0) as usize;

        let sigma = match self.family {
            Family::Lda => None,
            Family::Gga | Family::Mgga => Some(sigma.ok_or(Error::MissingInput("sigma"))?),
        };
        if let Some(s) = sigma {
            let expected = np * nsigma;
            if s.len() != expected {
                return Err(Error::BadLength {
                    what: "sigma",
                    expected,
                    got: s.len(),
                });
            }
        }
        let tau = match self.family {
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

        let n = |v: c_int| np * v.max(0) as usize;
        let mut exc = vec![0.0; n(d.zk)];
        let mut vxc = vec![0.0; n(d.vrho)];
        let mut vsigma_out = vec![0.0; n(d.vsigma)];
        let mut vlapl = vec![0.0; n(d.vlapl)];
        let mut vtau = vec![0.0; n(d.vtau)];
        let mut second = if order == Order::Fxc {
            Some(Second::alloc(d, np))
        } else {
            None
        };

        let lapl_ptr = lapl.map_or(ptr::null(), |v| v.as_ptr());
        let sigma_ptr = sigma.map_or(ptr::null(), |v| v.as_ptr());
        let tau_ptr = tau.map_or(ptr::null(), |v| v.as_ptr());

        // The generated crate defines one `xc_func_type` per C translation
        // unit.  They are `#[repr(C)]` identical, so the casts are sound.
        unsafe {
            match self.family {
                Family::Lda => {
                    let p = &*self.func as *const _ as *const src::lda::xc_func_type;
                    match order {
                        Order::Exc => src::lda::xc_lda_exc(
                            p,
                            np,
                            rho.as_ptr(),
                            exc.as_mut_ptr(),
                        ),
                        Order::Vxc => src::lda::xc_lda_exc_vxc(
                            p,
                            np,
                            rho.as_ptr(),
                            exc.as_mut_ptr(),
                            vxc.as_mut_ptr(),
                        ),
                        Order::Fxc => {
                            let s = second.as_mut().unwrap();
                            src::lda::xc_lda_exc_vxc_fxc(
                                p,
                                np,
                                rho.as_ptr(),
                                exc.as_mut_ptr(),
                                vxc.as_mut_ptr(),
                                s.v2rho2.as_mut_ptr(),
                            );
                        }
                    }
                }
                Family::Gga => {
                    let p = &*self.func as *const _ as *const src::gga::xc_func_type;
                    match order {
                        Order::Exc => src::gga::xc_gga_exc(
                            p,
                            np,
                            rho.as_ptr(),
                            sigma_ptr,
                            exc.as_mut_ptr(),
                        ),
                        Order::Vxc => src::gga::xc_gga_exc_vxc(
                            p,
                            np,
                            rho.as_ptr(),
                            sigma_ptr,
                            exc.as_mut_ptr(),
                            vxc.as_mut_ptr(),
                            vsigma_out.as_mut_ptr(),
                        ),
                        Order::Fxc => {
                            let s = second.as_mut().unwrap();
                            src::gga::xc_gga_exc_vxc_fxc(
                                p,
                                np,
                                rho.as_ptr(),
                                sigma_ptr,
                                exc.as_mut_ptr(),
                                vxc.as_mut_ptr(),
                                vsigma_out.as_mut_ptr(),
                                s.v2rho2.as_mut_ptr(),
                                s.v2rhosigma.as_mut_ptr(),
                                s.v2sigma2.as_mut_ptr(),
                            );
                        }
                    }
                }
                Family::Mgga => {
                    let p = &*self.func as *const _ as *const src::mgga::xc_func_type;
                    match order {
                        Order::Exc => src::mgga::xc_mgga_exc(
                            p,
                            np,
                            rho.as_ptr(),
                            sigma_ptr,
                            lapl_ptr,
                            tau_ptr,
                            exc.as_mut_ptr(),
                        ),
                        Order::Vxc => src::mgga::xc_mgga_exc_vxc(
                            p,
                            np,
                            rho.as_ptr(),
                            sigma_ptr,
                            lapl_ptr,
                            tau_ptr,
                            exc.as_mut_ptr(),
                            vxc.as_mut_ptr(),
                            vsigma_out.as_mut_ptr(),
                            vlapl.as_mut_ptr(),
                            vtau.as_mut_ptr(),
                        ),
                        Order::Fxc => {
                            let s = second.as_mut().unwrap();
                            src::mgga::xc_mgga_exc_vxc_fxc(
                                p,
                                np,
                                rho.as_ptr(),
                                sigma_ptr,
                                lapl_ptr,
                                tau_ptr,
                                exc.as_mut_ptr(),
                                vxc.as_mut_ptr(),
                                vsigma_out.as_mut_ptr(),
                                vlapl.as_mut_ptr(),
                                vtau.as_mut_ptr(),
                                s.v2rho2.as_mut_ptr(),
                                s.v2rhosigma.as_mut_ptr(),
                                s.v2rholapl.as_mut_ptr(),
                                s.v2rhotau.as_mut_ptr(),
                                s.v2sigma2.as_mut_ptr(),
                                s.v2sigmalapl.as_mut_ptr(),
                                s.v2sigmatau.as_mut_ptr(),
                                s.v2lapl2.as_mut_ptr(),
                                s.v2lapltau.as_mut_ptr(),
                                s.v2tau2.as_mut_ptr(),
                            );
                        }
                    }
                }
            }
        }

        Ok(Output {
            exc,
            vxc,
            vsigma: vsigma_out,
            vlapl,
            vtau,
            fxc: second,
        })
    }
}

impl Drop for Xc {
    fn drop(&mut self) {
        unsafe { src::functionals::xc_func_end(&mut *self.func) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbe_values_match_safe_crate() {
        let xc = Xc::from_name("gga_x_pbe", Spin::Unpolarized).unwrap();
        let out = xc
            .eval(&[0.5], Some(&[0.05]), None, None, Order::Fxc)
            .unwrap();
        assert!((out.exc[0] - -0.5872591953957921).abs() < 1e-13);
        assert!((out.vxc[0] - -0.7801794371835453).abs() < 1e-13);
        let fxc = out.fxc.unwrap();
        assert!((fxc.v2rho2[0] - -0.5276055576791105).abs() < 1e-12);
    }

    #[test]
    fn all_families_initialize() {
        for (name, spin) in [
            ("lda_x", Spin::Unpolarized),
            ("lda_c_vwn", Spin::Polarized),
            ("gga_x_pbe", Spin::Polarized),
            ("gga_c_lyp", Spin::Unpolarized),
            ("mgga_x_tpss", Spin::Polarized),
            ("mgga_c_tpss", Spin::Unpolarized),
            ("hyb_gga_xc_b3lyp", Spin::Unpolarized),
            ("hyb_mgga_xc_b98", Spin::Unpolarized),
        ] {
            let xc = Xc::from_name(name, spin)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let ns = spin.n_spin();
            let rho = vec![0.5 / ns as f64; ns];
            let sigma = vec![0.05; if ns == 2 { 3 } else { 1 }];
            let tau = vec![0.4 / ns as f64; ns];
            let out = xc
                .eval(&rho, Some(&sigma), None, Some(&tau), Order::Vxc)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(out.exc[0].is_finite(), "{name}");
            assert!(out.vxc.iter().all(|v| v.is_finite()), "{name}");
        }
    }
}
