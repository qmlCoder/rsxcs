//! `rsxcs` — exchange-correlation functionals for density-functional theory.
//!
//! This crate is a Rust port of a subset of [libxc 7.1.2](https://gitlab.com/libxc/libxc).
//! It implements the LDA, GGA and meta-GGA machinery together with
//! `exc`, `vxc` and `fxc` evaluation, and a registry keyed by the same
//! functional numbers and short names used by libxc.
//!
//! ```
//! use rsxcs::{Functional, Order, Spin, Xc};
//!
//! let xc = Xc::new(101).unwrap();                 // gga_x_pbe
//! assert_eq!(xc.info().name, "gga_x_pbe");
//!
//! let rho = [0.8, 0.2];                           // spin-up, spin-down
//! let sigma = [0.1, 0.01, 0.05];                  // uu, ud, dd
//! let out = xc
//!     .eval(Spin::Polarized, &rho, Some(&sigma), None, None, Order::Fxc)
//!     .unwrap();
//!
//! assert_eq!(out.exc.len(), 1);
//! assert_eq!(out.vxc.len(), 2);
//! assert!(out.fxc.is_some());
//! ```
//!
//! # Design
//!
//! Each functional is written once as an energy density generic over
//! [`scalar::Scalar`].  Evaluating it with `f64` gives the energy; evaluating
//! it with [`scalar::Jet`] propagates first and second derivatives
//! (forward-mode automatic differentiation), producing `vxc` and `fxc`
//! without any hand-written derivative code.  The libxc conventions for the
//! input layout, the meaning of `sigma`, the spin scaling and the output
//! orderings are preserved.

#![forbid(unsafe_code)]
// Constants are copied verbatim from libxc (more digits than f64 can hold).
#![allow(clippy::excessive_precision)]

pub mod consts;
mod eval;
mod functional;
pub mod info;
pub mod scalar;

pub use eval::{Error, Evaluation, Fxc, Order, Xc};
pub use info::{Family, Functional, Info, Kind, Spin};
