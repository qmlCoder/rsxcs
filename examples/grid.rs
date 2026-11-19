//! Evaluate a few functionals on a small unpolarized grid.
//!
//! Run with:
//!
//! ```text
//! cargo run --example grid
//! ```

use rsxcs::{Order, Spin, Xc};

fn main() {
    // rho, sigma = |grad rho|^2, tau
    let grid = [(0.5, 0.05, 0.40), (2.0, 2.00, 5.00), (8.0, 60.0, 40.00)];

    let names = [
        "lda_c_pw",
        "gga_x_pbe",
        "gga_c_pbe",
        "mgga_x_tpss",
        "hyb_gga_xc_pbeh",
    ];

    for name in names {
        let xc = Xc::from_name(name).unwrap();
        println!("\n{} ({})", xc.info().name, xc.info().id);
        for &(n, sigma, tau) in &grid {
            let out = xc
                .eval(
                    Spin::Unpolarized,
                    &[n],
                    Some(&[sigma]),
                    None,
                    Some(&[tau]),
                    Order::Vxc,
                )
                .unwrap();
            print!(
                "  rho={n:<5} exc={:<14.8} vrho={:<14.8}",
                out.exc[0], out.vxc[0]
            );
            if let Some(v) = out.vsigma.first() {
                print!(" vsigma={v:<14.8}");
            }
            if let Some(v) = out.vtau.first() {
                print!(" vtau={v:<14.8}");
            }
            println!();
        }
    }
}
