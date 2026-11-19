# Full libxc 7.1.2 via c2rust

`rsxcs` (the crate in the repository root) is the hand-written, safe subset.
This directory contains a **complete mechanical translation** of libxc 7.1.2
produced with [c2rust](https://c2rust.com/), plus a small safe wrapper.

What the pipeline produces:

* all **332 C translation units** of libxc 7.1.2 transpiled to Rust
  (~53 MB of `src/*.rs`), including every LDA/GGA/meta-GGA/hybrid functional
  and its analytic derivatives;
* the generated code was built with `XC_MAXORDER=2`, so `exc`, `vxc` and
  `fxc` are available (no `kxc`/`lxc`);
* no semantic edits are made: only 22 `f128` constant expressions are
  simplified to `f64` literals (the old `f128` crate cannot evaluate them in
  `const` context) and the package is renamed.

## Generate the backend

Requirements: `cmake`, a C compiler, `c2rust` 0.22.1 and the
`nightly-2023-04-15` toolchain (c2rust's output uses the `extern_types` and
`raw_ref_op` feature gates).

```console
$ ./c2rust/generate.sh
```

Without an argument the script downloads the libxc 7.1.2 release tarball; you
can also pass an existing source tree:

```console
$ ./c2rust/generate.sh /path/to/libxc-7.1.2
```

The generated crate is written to `c2rust/libxc/` and is gitignored. The
script needs roughly 5 minutes and ~1 GB of temporary space.

## Build and test the wrapper

The wrapper is a normal Rust crate depending on `../libxc` by path:

```console
$ cd c2rust/libxc-wrapper
$ cargo test                    # picks up c2rust/rust-toolchain.toml
$ cargo run --example compare
```

`compare` evaluates all 22 functionals of the safe `rsxcs` crate, for both
spin treatments and four points each, with `exc`, `vxc`, `vsigma`, `vtau` and
all second derivatives, and compares the two implementations. The observed
maximum relative difference is **3.8e-14** (pure round-off).

## API

```rust
use libxc_full::{Order, Spin, Xc};

let xc = Xc::from_name("mgga_c_tpss", Spin::Unpolarized).unwrap();
let out = xc.eval(
    &[0.5],              // rho
    Some(&[0.05]),       // sigma (GGA/MGGA)
    None,                // lapl (optional)
    Some(&[0.4]),        // tau (MGGA)
    Order::Fxc,
).unwrap();
println!("{}", out.exc[0]);
```

The wrapper allocates all output arrays from the dimensions libxc reports
(`func.dim`), so the layouts are exactly libxc's, including the full
meta-GGA `v2rholapl / v2sigmalapl / v2lapl2 / ...` set.

## Why a wrapper?

c2rust emits one copy of the header types (`xc_func_type`, `xc_func_info_type`,
...) per translation unit. All copies are `#[repr(C)]`-identical but nominally
distinct, and the generated functions are `pub unsafe extern "C"`. The wrapper:

* initializes/frees the functional object (`xc_func_init` / `xc_func_end`);
* casts between the layout-identical `xc_func_type` copies at the call
  boundary;
* dispatches to `xc_lda_*`, `xc_gga_*` or `xc_mgga_*` based on the family,
  including `HYB_LDA`, `HYB_GGA` and `HYB_MGGA`;
* owns the output `Vec`s and reports errors instead of returning C codes.

## Caveats

* The generated crate is large, unsafe, nightly-only and produces thousands of
  warnings. It is not intended to be edited by hand.
* Build it with `debug = 0` (already set in the wrapper). The old
  `nightly-2023-04-15` LLVM segfaults when emitting DWARF for the whole crate
  with debuginfo enabled; a release build with `debuginfo=0` works.
* The generated code contains `genwiki`, `xc-sanity` and `xc-threshold` main
  functions as modules; `autobins = false` in the generated `Cargo.toml` keeps
  them from being built.
* Hybrid functionals expose only the semilocal part, exactly like libxc's
  `xc_gga_exc`/`xc_mgga_exc`; exact exchange is the caller's responsibility.
* The safe `rsxcs` crate remains the recommended API for the functionals it
  implements (stable Rust, reviewable code, forward-mode AD derivatives).

## License

The generated code is derived from libxc 7.1.2 and is under the Mozilla Public
License 2.0, like libxc itself.
