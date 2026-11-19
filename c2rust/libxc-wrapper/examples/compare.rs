//! Compare the c2rust-backed full libxc port with the hand-written safe
//! subset (`rsxcs`) on every functional that both support.
//!
//! ```text
//! cd c2rust && cargo +nightly-2023-04-15 run --example compare
//! ```

use libxc_full::{Order, Spin, Xc as FullXc};
use rsxcs::{Order as SafeOrder, Spin as SafeSpin, Xc as SafeXc};

const NAMES: &[&str] = &[
    "lda_x",
    "lda_c_vwn",
    "lda_c_vwn_rpa",
    "lda_c_pz",
    "lda_c_pw",
    "lda_c_pw_mod",
    "lda_c_vwn_1",
    "lda_c_vwn_2",
    "lda_c_vwn_3",
    "lda_c_vwn_4",
    "gga_x_pbe",
    "gga_x_pbe_r",
    "gga_x_pbe_sol",
    "gga_x_b88",
    "gga_x_pw91",
    "gga_c_pbe",
    "gga_c_lyp",
    "gga_c_p86",
    "gga_c_p86_ft",
    "mgga_x_tpss",
    "hyb_gga_xc_b3lyp",
    "hyb_gga_xc_pbeh",
];

fn max_rel_err(got: &[f64], want: &[f64], acc: &mut f64) {
    for (g, w) in got.iter().zip(want) {
        let denom = w.abs().max(1.0);
        *acc = acc.max((g - w).abs() / denom);
    }
}

fn main() {
    // (rho, sigma, tau) unpolarized and (na, nb, saa, sab, sbb, ta, tb).
    let unpol = [
        (0.5, 0.05, 0.40),
        (2.0, 2.00, 5.00),
        (0.02, 0.002, 0.05),
        (8.0, 60.0, 40.00),
    ];
    let pol = [
        (0.4, 0.1, 0.05, 0.01, 0.02, 0.30, 0.10),
        (1.2, 0.8, 2.00, 0.50, 1.00, 3.00, 2.00),
        (0.015, 0.005, 0.002, 0.0005, 0.001, 0.05, 0.04),
        (6.0, 2.0, 50.0, 10.0, 20.0, 30.0, 12.00),
    ];

    let mut global = 0.0f64;
    println!(
        "{:<22} {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "functional", "spin", "exc", "vxc", "vsigma", "vtau", "fxc"
    );

    for name in NAMES {
        for (spin, safe_spin) in [
            (Spin::Unpolarized, SafeSpin::Unpolarized),
            (Spin::Polarized, SafeSpin::Polarized),
        ] {
            let full = FullXc::from_name(name, spin).unwrap();
            let safe = SafeXc::from_name(name).unwrap();
            let mut errs = [0.0f64; 5];

            match spin {
                Spin::Unpolarized => {
                    for &(n, s, t) in &unpol {
                        let a = full
                            .eval(&[n], Some(&[s]), None, Some(&[t]), Order::Fxc)
                            .unwrap();
                        let b = safe
                            .eval(
                                safe_spin,
                                &[n],
                                Some(&[s]),
                                None,
                                Some(&[t]),
                                SafeOrder::Fxc,
                            )
                            .unwrap();
                        max_rel_err(&a.exc, &b.exc, &mut errs[0]);
                        max_rel_err(&a.vxc, &b.vxc, &mut errs[1]);
                        max_rel_err(&a.vsigma, &b.vsigma, &mut errs[2]);
                        max_rel_err(&a.vtau, &b.vtau, &mut errs[3]);
                        let (fa, fb) = (a.fxc.as_ref().unwrap(), b.fxc.as_ref().unwrap());
                        max_rel_err(&fa.v2rho2, &fb.v2rho2, &mut errs[4]);
                        max_rel_err(&fa.v2rhosigma, &fb.v2rhosigma, &mut errs[4]);
                        max_rel_err(&fa.v2sigma2, &fb.v2sigma2, &mut errs[4]);
                        max_rel_err(&fa.v2rhotau, &fb.v2rhotau, &mut errs[4]);
                        max_rel_err(&fa.v2sigmatau, &fb.v2sigmatau, &mut errs[4]);
                        max_rel_err(&fa.v2tau2, &fb.v2tau2, &mut errs[4]);
                    }
                }
                Spin::Polarized => {
                    for &(na, nb, saa, sab, sbb, ta, tb) in &pol {
                        let a = full
                            .eval(
                                &[na, nb],
                                Some(&[saa, sab, sbb]),
                                None,
                                Some(&[ta, tb]),
                                Order::Fxc,
                            )
                            .unwrap();
                        let b = safe
                            .eval(
                                safe_spin,
                                &[na, nb],
                                Some(&[saa, sab, sbb]),
                                None,
                                Some(&[ta, tb]),
                                SafeOrder::Fxc,
                            )
                            .unwrap();
                        max_rel_err(&a.exc, &b.exc, &mut errs[0]);
                        max_rel_err(&a.vxc, &b.vxc, &mut errs[1]);
                        max_rel_err(&a.vsigma, &b.vsigma, &mut errs[2]);
                        max_rel_err(&a.vtau, &b.vtau, &mut errs[3]);
                        let (fa, fb) = (a.fxc.as_ref().unwrap(), b.fxc.as_ref().unwrap());
                        max_rel_err(&fa.v2rho2, &fb.v2rho2, &mut errs[4]);
                        max_rel_err(&fa.v2rhosigma, &fb.v2rhosigma, &mut errs[4]);
                        max_rel_err(&fa.v2sigma2, &fb.v2sigma2, &mut errs[4]);
                        max_rel_err(&fa.v2rhotau, &fb.v2rhotau, &mut errs[4]);
                        max_rel_err(&fa.v2sigmatau, &fb.v2sigmatau, &mut errs[4]);
                        max_rel_err(&fa.v2tau2, &fb.v2tau2, &mut errs[4]);
                    }
                }
            }

            for e in errs {
                global = global.max(e);
            }
            println!(
                "{:<22} {:>4}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}",
                name,
                if spin == Spin::Polarized { "pol" } else { "unpol" },
                errs[0],
                errs[1],
                errs[2],
                errs[3],
                errs[4]
            );
        }
    }

    println!("\nmax relative error: {global:.3e}");
    assert!(
        global < 1e-8,
        "c2rust backend and rsxcs disagree (max rel err {global:.3e})"
    );
}
