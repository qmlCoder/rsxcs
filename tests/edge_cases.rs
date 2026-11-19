//! Robustness and consistency tests that do not need libxc.

use rsxcs::{Error, Functional, Order, Spin, Xc};

fn finite(x: f64) -> bool {
    x.is_finite()
}

#[test]
fn all_functionals_are_finite_on_degenerate_points() {
    for &f in Functional::ALL {
        let xc = Xc::from_functional(f);
        for spin in [Spin::Unpolarized, Spin::Polarized] {
            let ns = spin.n_spin();
            let rho = vec![0.35; ns];
            let sigma = vec![0.0; if ns == 2 { 3 } else { 1 }];
            let tau = vec![0.2; ns];
            let out = xc
                .eval(spin, &rho, Some(&sigma), None, Some(&tau), Order::Vxc)
                .unwrap_or_else(|e| panic!("{f} {spin:?}: {e}"));
            assert!(finite(out.exc[0]), "{f} {spin:?}: non-finite exc");
            assert!(
                out.vxc.iter().all(|v| finite(*v)),
                "{f} {spin:?}: non-finite vxc"
            );
            assert!(
                out.vsigma.iter().all(|v| finite(*v)),
                "{f} {spin:?}: non-finite vsigma"
            );
            assert!(
                out.vtau.iter().all(|v| finite(*v)),
                "{f} {spin:?}: non-finite vtau"
            );
        }
    }
}

#[test]
fn zero_density_is_zero() {
    for &f in Functional::ALL {
        let xc = Xc::from_functional(f);
        let out = xc
            .eval(
                Spin::Unpolarized,
                &[0.0],
                Some(&[0.0]),
                None,
                Some(&[0.0]),
                Order::Exc,
            )
            .unwrap();
        assert_eq!(out.exc[0], 0.0, "{f}");
    }
}

#[test]
fn unpolarized_matches_equal_spin_polarized() {
    let n = 0.7;
    let sigma = 0.4;
    let tau = 0.9;
    for &f in Functional::ALL {
        let xc = Xc::from_functional(f);
        let unpol = xc
            .eval(
                Spin::Unpolarized,
                &[n],
                Some(&[sigma]),
                None,
                Some(&[tau]),
                Order::Exc,
            )
            .unwrap();
        let pol = xc
            .eval(
                Spin::Polarized,
                &[n / 2.0, n / 2.0],
                Some(&[sigma / 4.0, sigma / 4.0, sigma / 4.0]),
                None,
                Some(&[tau / 2.0, tau / 2.0]),
                Order::Exc,
            )
            .unwrap();
        assert!(
            (unpol.exc[0] - pol.exc[0]).abs() < 1e-12,
            "{f}: unpol {} vs pol {}",
            unpol.exc[0],
            pol.exc[0]
        );
    }
}

#[test]
fn errors_are_reported() {
    assert!(matches!(Xc::new(999_999), Err(Error::UnknownId(_))));
    assert!(matches!(
        Xc::from_name("not_a_functional"),
        Err(Error::UnknownName(_))
    ));

    let pbe = Xc::new(101).unwrap();
    assert!(matches!(
        pbe.eval(Spin::Unpolarized, &[0.5], None, None, None, Order::Exc),
        Err(Error::MissingInput("sigma"))
    ));
    assert!(matches!(
        pbe.eval(
            Spin::Polarized,
            &[0.5, 0.5],
            Some(&[0.1]),
            None,
            None,
            Order::Exc
        ),
        Err(Error::BadLength { what: "sigma", .. })
    ));
    assert!(matches!(
        Xc::new(12)
            .unwrap()
            .eval(Spin::Unpolarized, &[], None, None, None, Order::Exc),
        Err(Error::NoPoints)
    ));
}
