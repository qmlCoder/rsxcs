# rsxcs

Exchange–correlation functionals for density-functional theory, written in
Rust. This crate is a port of a subset of
[libxc 7.1.2](https://gitlab.com/libxc/libxc/-/releases) and keeps libxc's
functional numbers, short names, spin conventions, input ordering and output
array layouts, so results can be checked directly against libxc.

It is **not** a complete translation of libxc (which contains 800+
functionals and ~50 MB of generated code). The infrastructure, however, is
complete for the LDA / GGA / meta-GGA families, and adding a functional only
requires writing its energy expression — first and second derivatives are
obtained by automatic differentiation.

```rust
use rsxcs::{Order, Spin, Xc};

let xc = Xc::from_name("gga_x_pbe").unwrap();   // or Xc::new(101).unwrap()

let rho   = [0.8, 0.2];                    // spin-up, spin-down
let sigma = [0.1, 0.01, 0.05];             // uu, ud, dd
let out = xc
    .eval(Spin::Polarized, &rho, Some(&sigma), None, None, Order::Fxc)
    .unwrap();

println!("exc     = {}", out.exc[0]);
println!("vrho    = {:?}", out.vxc);
println!("vsigma  = {:?}", out.vsigma);
println!("v2rho2  = {:?}", out.fxc.as_ref().unwrap().v2rho2);
```

## Implemented functionals

| libxc ID | name | family | kind |
|---:|---|---|---|
| 1   | `lda_x`          | LDA  | exchange |
| 7   | `lda_c_vwn`      | LDA  | correlation (VWN5) |
| 8   | `lda_c_vwn_rpa`  | LDA  | correlation (VWN-RPA) |
| 9   | `lda_c_pz`       | LDA  | correlation |
| 12  | `lda_c_pw`       | LDA  | correlation (PW92) |
| 13  | `lda_c_pw_mod`   | LDA  | correlation (PW92, modified) |
| 28–31 | `lda_c_vwn_1..4` | LDA | correlation (VWN1–VWN4) |
| 101 | `gga_x_pbe`      | GGA  | exchange |
| 102 | `gga_x_pbe_r`    | GGA  | exchange (revPBE) |
| 106 | `gga_x_b88`      | GGA  | exchange |
| 109 | `gga_x_pw91`     | GGA  | exchange |
| 116 | `gga_x_pbe_sol`  | GGA  | exchange |
| 130 | `gga_c_pbe`      | GGA  | correlation |
| 131 | `gga_c_lyp`      | GGA  | correlation |
| 132 | `gga_c_p86`      | GGA  | correlation |
| 217 | `gga_c_p86_ft`   | GGA  | correlation |
| 202 | `mgga_x_tpss`    | MGGA | exchange |
| 402 | `hyb_gga_xc_b3lyp` | hybrid GGA | exchange–correlation (semilocal part) |
| 406 | `hyb_gga_xc_pbeh`  | hybrid GGA | exchange–correlation (semilocal part) |

For hybrids, only the semilocal part is returned; the exact-exchange fraction
(`cam_alpha`: 0.20 for B3LYP, 0.25 for PBE0) has to be added by the host code.

## Input and output conventions

`Xc::eval(spin, rho, sigma, lapl, tau, order)` follows libxc:

* `rho`: `npoints * nspin`, spin-resolved densities.
* `sigma`: `npoints * (1 or 3)`, with the ordering `(uu)` or `(uu, ud, dd)`;
  `sigma_ud` may be negative, with `|sigma_ud| <= (sigma_uu + sigma_dd)/2`.
  Required for GGA and meta-GGA.
* `lapl`: accepted for API compatibility. None of the implemented functionals
  depends on the Laplacian, and the corresponding second derivatives are
  returned as zeros.
* `tau`: `npoints * nspin`, required for meta-GGA. Use the *positive*
  kinetic-energy density (`tau = 1/2 sum_i |grad psi_i|^2`).
* `order`: `Exc`, `Vxc` or `Fxc`.

`Evaluation` contains:

| field | size | ordering |
|---|---|---|
| `exc` | `npoints` | energy per particle |
| `vxc` | `npoints * nspin` | `(u, d)` |
| `vsigma` | `npoints * (1 or 3)` | `(uu, ud, dd)` |
| `vtau` | `npoints * nspin` | `(u, d)` |
| `fxc.v2rho2` | `npoints * (1 or 3)` | `(uu, ud, dd)` |
| `fxc.v2rhosigma` | `npoints * (1 or 6)` | `(u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)` |
| `fxc.v2sigma2` | `npoints * (1 or 6)` | `(uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)` |
| `fxc.v2rhotau` | `npoints * (1 or 4)` | `(u_u, u_d, d_u, d_d)` |
| `fxc.v2sigmatau` | `npoints * (1 or 6)` | `(uu_u, uu_d, ud_u, ud_d, dd_u, dd_d)` |
| `fxc.v2tau2` | `npoints * (1 or 3)` | `(uu, ud, dd)` |
| `fxc.v2rho{lapl}`, `v2sigma{lapl}`, `v2lapl*` | zeros | meta-GGA only |

The `sigma` ordering and the second-derivative orderings match libxc/pylibxc.

## How the derivatives are obtained

Each functional is implemented once as an energy-density expression generic
over a scalar type (`src/functional/*.rs`). Evaluating it with `f64` gives the
energy; evaluating it with `Jet<N>` (`src/scalar.rs`) propagates the gradient
and Hessian with respect to the `N` layout variables by forward-mode automatic
differentiation. The driver then maps those derivatives onto libxc's arrays
(`src/eval.rs`).

This means:

* there is exactly one implementation of each parametrisation — no separate
  hand-written `vxc`/`fxc` kernels that can drift out of sync;
* adding a functional is a matter of adding its energy expression and an entry
  in `src/info.rs`;
* near zero gradients the automatic differentiation can lose accuracy in the
  second derivative (the analytic cancellation is only exact in real
  arithmetic), although the energy and `vxc` remain well behaved. Grid points
  with `sigma = 0` are floored, as in libxc.

## Validation against libxc 7.1.2

`tests/data/libxc_7_1_2_reference.txt` contains `exc`, `vxc`, `vsigma`,
`vtau` and all implemented second derivatives computed with libxc 7.1.2 for
four unpolarized and four polarized points per functional. The test suite
compares every value with a relative tolerance of `5e-10`:

```console
$ cargo test
```

The reference file was generated by `tools/gen_reference.py`; regenerating it
requires a libxc 7.1.2 build and the matching `pylibxc` sources:

```console
$ cmake -S libxc-7.1.2 -B build -DENABLE_FORTRAN=OFF -DBUILD_TESTING=OFF \
        -DENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON
$ cmake --build build -j
$ cp build/libxc.so.15 libxc-7.1.2/pylibxc/
$ ln -sf libxc.so.15 libxc-7.1.2/pylibxc/libxc.so
$ PYTHONPATH=libxc-7.1.2 python3 tools/gen_reference.py
```

All listed functionals are covered by the test.

## Example

```console
$ cargo run --example grid
```

## Adding a functional

1. Implement its energy density (hartree) as
   `fn my_f<S: Scalar>(...) -> S` in `src/functional/{lda,gga,mgga}.rs`.
2. Add a variant to `define_functionals!` in `src/info.rs` with the libxc
   number, name, family and kind.
3. Add the dispatch arm in the corresponding module.
4. Optionally regenerate the reference data (see above) and add the name to
   `FUNCTIONALS` in `tools/gen_reference.py`.

## Full libxc backend through c2rust

The safe crate above implements a curated subset.  For **all** libxc 7.1.2
functionals there is a complete mechanical translation generated with
`c2rust` in [`c2rust/`](c2rust/):

* all 332 libxc translation units (~53 MB of Rust), including analytic `fxc`;
* `c2rust/generate.sh` reproduces the transpilation from the 7.1.2 release
  (requires c2rust 0.22.1 + `nightly-2023-04-15`);
* `c2rust/libxc-wrapper` is a small safe wrapper (`Xc::new`, `Xc::from_name`,
  `eval` for LDA/GGA/meta-GGA and hybrids);
* `cargo run --example compare` checks the full backend against `rsxcs`: the
  maximum relative difference over all overlapping functionals, spin
  treatments and derivatives is **3.8e-14**.

Why keep both?  The hand-written crate is stable, reviewable and easy to
extend; the c2rust backend is complete but large, unsafe and pinned to an old
nightly.  Use the former for the implemented functionals and the latter when
you need libxc's full catalogue.

```console
$ ./c2rust/generate.sh
$ cd c2rust/libxc-wrapper && cargo run --example compare
```

## Limitations

* Only the functionals listed above are implemented; the rest of libxc is not
  (yet) translated.
* No Laplacian-dependent meta-GGAs; `lapl` is ignored.
* No external parameters (e.g. the `alpha` of `lda_c_xalpha`), no range
  separation, no kinetic-energy functionals, no GGA/MGGA hybrids beyond the
  two listed.
* Exact exchange is not evaluated: hybrids return only their semilocal
  contribution.
* Meta-GGA input points must satisfy `tau >= tau_W` (no Fermi-hole-curvature
  enforcement is applied). libxc does not enforce it for TPSS either, but the
  algebraic rearrangements differ in that unphysical regime.
* Densities below `1e-15`, `sigma` below `1e-40` and `tau` below `1e-20` are
  clamped, mirroring libxc's screening thresholds for these functionals.

## License

Mozilla Public License 2.0, the same license as libxc. The functional formulas
and constants are derived from libxc 7.1.2 (Copyright the libxc authors,
M. A. L. Marques, S. Lehtola, X. Andrade and contributors).
