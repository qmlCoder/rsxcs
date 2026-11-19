#!/usr/bin/env python3
"""Generate reference values for the `rsxcs` test suite with libxc 7.1.2.

This script is *not* needed to build or use the crate.  It was used once to
produce `tests/data/libxc_7_1_2_reference.txt`, which is checked in so that
`cargo test` does not require libxc.

Requirements
------------
A libxc 7.1.2 shared library plus the matching `pylibxc` Python package, e.g.

    cmake -S libxc-7.1.2 -B build -DENABLE_FORTRAN=OFF -DBUILD_TESTING=OFF \
          -DENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON
    cmake --build build -j
    cp build/libxc.so.15 libxc-7.1.2/pylibxc/
    ln -sf libxc.so.15 libxc-7.1.2/pylibxc/libxc.so
    PYTHONPATH=libxc-7.1.2 python3 tools/gen_reference.py

File format
-----------
One header per case::

    # name id nspin npoints family

followed by one line per grid point::

    rho0 rho1 sig0 sig1 sig2 tau0 tau1 lapl0 lapl1
    zk vrho0 vrho1 vsigma0 vsigma1 vsigma2 vtau0 vtau1
    v2rho2_0 v2rho2_1 v2rho2_2
    v2rhosigma_0 ... v2rhosigma_5
    v2sigma2_0 ... v2sigma2_5
    v2rhotau_0 ... v2rhotau_3
    v2sigmatau_0 ... v2sigmatau_5
    v2tau2_0 v2tau2_1 v2tau2_2

(all on one physical line).  Entries that do not exist for a family/spin are
written as 0.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import pylibxc
except ImportError:  # pragma: no cover
    sys.exit("pylibxc is required: set PYTHONPATH to the pylibxc source tree")
try:
    from pylibxc import flags
except ImportError:  # pragma: no cover
    flags = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "data", "libxc_7_1_2_reference.txt")

# Functionals implemented by rsxcs (libxc short names).
FUNCTIONALS = [
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
]

# (rho, sigma, tau)
UNPOLARIZED = [
    (0.5, 0.05, 0.4),
    (2.0, 2.0, 5.0),
    (0.02, 0.002, 0.05),
    (8.0, 60.0, 40.0),
]

# (rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down)
POLARIZED = [
    (0.4, 0.1, 0.05, 0.01, 0.02, 0.3, 0.1),
    (1.2, 0.8, 2.0, 0.5, 1.0, 3.0, 2.0),
    (0.015, 0.005, 0.002, 0.0005, 0.001, 0.05, 0.04),
    (6.0, 2.0, 50.0, 10.0, 20.0, 30.0, 12.0),
]

FAMILY_NAMES = {1: "lda", 2: "gga", 4: "mgga", 32: "gga"}


def family_of(func) -> int:
    fam = func.get_family()
    if flags is not None:
        return {flags.XC_FAMILY_LDA: 1, flags.XC_FAMILY_GGA: 2,
                flags.XC_FAMILY_MGGA: 4, flags.XC_FAMILY_HYB_GGA: 2}[fam]
    return FAMILY_NAMES[fam]


def vec(d, key, n, default=0.0):
    if key not in d:
        return np.full(n, default)
    a = np.asarray(d[key], dtype=float).reshape(-1)
    out = np.full(n, default)
    out[: min(n, a.size)] = a[:n]
    return out


def one_case(name: str, spin: str, points) -> str:
    func = pylibxc.LibXCFunctional(name, spin)
    fam = family_of(func)
    pol = spin == "polarized"
    nspin = 2 if pol else 1

    lines = [f"# {name} {func.get_number()} {nspin} {len(points)} {FAMILY_NAMES[fam]}"]

    for pt in points:
        if pol:
            na, nb, saa, sab, sbb, ta, tb = pt
            inp = {
                "rho": np.array([na, nb]),
                "sigma": np.array([saa, sab, sbb]),
                "tau": np.array([ta, tb]),
                "lapl": np.zeros(2),
            }
        else:
            n, s, t = pt
            inp = {
                "rho": np.array([n]),
                "sigma": np.array([s]),
                "tau": np.array([t]),
                "lapl": np.zeros(1),
            }

        if fam == 1:
            inp.pop("sigma")
            inp.pop("tau")
            inp.pop("lapl")
        elif fam == 2:
            inp.pop("tau")
            inp.pop("lapl")

        out = func.compute(inp, do_exc=True, do_vxc=True, do_fxc=True)

        zk = vec(out, "zk", 1)
        vrho = vec(out, "vrho", 2)
        vsigma = vec(out, "vsigma", 3)
        vtau = vec(out, "vtau", 2)
        v2rho2 = vec(out, "v2rho2", 3)
        v2rhosigma = vec(out, "v2rhosigma", 6)
        v2sigma2 = vec(out, "v2sigma2", 6)
        v2rhotau = vec(out, "v2rhotau", 4)
        v2sigmatau = vec(out, "v2sigmatau", 6)
        v2tau2 = vec(out, "v2tau2", 3)

        values = np.concatenate(
            [
                np.array([na, nb, saa, sab, sbb, ta, tb], dtype=float)
                if pol
                else np.array([pt[0], 0.0, pt[1], 0.0, 0.0, pt[2], 0.0], dtype=float),
                np.zeros(2),
                zk,
                vrho,
                vsigma,
                vtau,
                v2rho2,
                v2rhosigma,
                v2sigma2,
                v2rhotau,
                v2sigmatau,
                v2tau2,
            ]
        )
        lines.append(" ".join(f"{v:.17g}" for v in values))

    return "\n".join(lines)


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    chunks = [
        "# Reference values generated with libxc 7.1.2 (pylibxc).",
        "# See tools/gen_reference.py for the column layout.",
        "",
    ]
    for name in FUNCTIONALS:
        chunks.append(one_case(name, "unpolarized", UNPOLARIZED))
        chunks.append("")
        chunks.append(one_case(name, "polarized", POLARIZED))
        chunks.append("")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(chunks))
    print(f"wrote {OUT} ({len(FUNCTIONALS)} functionals)")


if __name__ == "__main__":
    main()
