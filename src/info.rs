//! Functional metadata and the registry of implemented functionals.

/// Spin treatment requested for an evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spin {
    Unpolarized,
    Polarized,
}

impl Spin {
    /// Number of spin channels (`1` or `2`).
    #[inline]
    pub const fn n_spin(self) -> usize {
        match self {
            Spin::Unpolarized => 1,
            Spin::Polarized => 2,
        }
    }

    #[inline]
    pub const fn is_polarized(self) -> bool {
        matches!(self, Spin::Polarized)
    }
}

/// libxc functional family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Lda,
    Gga,
    Mgga,
    /// Hybrid GGA.  The semilocal part is evaluated with the GGA machinery;
    /// the exact-exchange fraction is the caller's responsibility.
    HybGga,
}

impl Family {
    /// Family used for the semilocal evaluation (hybrids are GGAs here).
    #[inline]
    pub const fn semilocal(self) -> Family {
        match self {
            Family::HybGga => Family::Gga,
            other => other,
        }
    }
}

/// Type of the functional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Exchange,
    Correlation,
    ExchangeCorrelation,
}

/// Static description of a functional.
#[derive(Debug, Clone, Copy)]
pub struct Info {
    /// libxc functional number (e.g. `101` for `gga_x_pbe`).
    pub id: u32,
    /// libxc short name (e.g. `"gga_x_pbe"`).
    pub name: &'static str,
    pub kind: Kind,
    pub family: Family,
    pub description: &'static str,
    /// Primary references, following the libxc entry.
    pub references: &'static [&'static str],
}

macro_rules! define_functionals {
    ($(
        $variant:ident => {
            id: $id:expr,
            name: $name:expr,
            kind: $kind:expr,
            family: $family:expr,
            desc: $desc:expr,
            refs: $refs:expr $(,)?
        }
    ),* $(,)?) => {
        /// All functionals implemented by this crate.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Functional {
            $($variant),*
        }

        impl Functional {
            /// Every implemented functional.
            pub const ALL: &'static [Functional] = &[$(Functional::$variant),*];

            /// Static metadata.
            pub const fn info(self) -> Info {
                match self {
                    $(Functional::$variant => Info {
                        id: $id,
                        name: $name,
                        kind: $kind,
                        family: $family,
                        description: $desc,
                        references: $refs,
                    }),*
                }
            }

            /// Look up a functional by its libxc number.
            pub fn from_id(id: u32) -> Option<Self> {
                match id {
                    $($id => Some(Functional::$variant),)*
                    _ => None,
                }
            }

            /// Look up a functional by its libxc short name (case-insensitive).
            pub fn from_name(name: &str) -> Option<Self> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|f| f.info().name.eq_ignore_ascii_case(name))
            }
        }
    };
}

define_functionals! {
    LdaX => {
        id: 1, name: "lda_x", kind: Kind::Exchange, family: Family::Lda,
        desc: "Slater exchange",
        refs: &["P. A. M. Dirac, Proc. Cambridge Philos. Soc. 26, 376 (1930)"],
    },
    LdaCVwn => {
        id: 7, name: "lda_c_vwn", kind: Kind::Correlation, family: Family::Lda,
        desc: "Vosko, Wilk & Nusair (VWN5)",
        refs: &["S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)"],
    },
    LdaCVwnRpa => {
        id: 8, name: "lda_c_vwn_rpa", kind: Kind::Correlation, family: Family::Lda,
        desc: "Vosko, Wilk & Nusair (VWN RPA)",
        refs: &["S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)"],
    },
    LdaCPz => {
        id: 9, name: "lda_c_pz", kind: Kind::Correlation, family: Family::Lda,
        desc: "Perdew & Zunger",
        refs: &["J. P. Perdew, A. Zunger, Phys. Rev. B 23, 5048 (1981)"],
    },
    LdaCPw => {
        id: 12, name: "lda_c_pw", kind: Kind::Correlation, family: Family::Lda,
        desc: "Perdew & Wang (PW92)",
        refs: &["J. P. Perdew, Y. Wang, Phys. Rev. B 45, 13244 (1992)"],
    },
    LdaCPwMod => {
        id: 13, name: "lda_c_pw_mod", kind: Kind::Correlation, family: Family::Lda,
        desc: "Perdew & Wang (PW92), modified",
        refs: &["J. P. Perdew, Y. Wang, Phys. Rev. B 45, 13244 (1992)"],
    },
    LdaCVwn1 => {
        id: 28, name: "lda_c_vwn_1", kind: Kind::Correlation, family: Family::Lda,
        desc: "Vosko, Wilk & Nusair (VWN1)",
        refs: &["S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)"],
    },
    LdaCVwn2 => {
        id: 29, name: "lda_c_vwn_2", kind: Kind::Correlation, family: Family::Lda,
        desc: "Vosko, Wilk & Nusair (VWN2)",
        refs: &["S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)"],
    },
    LdaCVwn3 => {
        id: 30, name: "lda_c_vwn_3", kind: Kind::Correlation, family: Family::Lda,
        desc: "Vosko, Wilk & Nusair (VWN3)",
        refs: &["S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)"],
    },
    LdaCVwn4 => {
        id: 31, name: "lda_c_vwn_4", kind: Kind::Correlation, family: Family::Lda,
        desc: "Vosko, Wilk & Nusair (VWN4)",
        refs: &["S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)"],
    },
    GgaXPbe => {
        id: 101, name: "gga_x_pbe", kind: Kind::Exchange, family: Family::Gga,
        desc: "Perdew, Burke & Ernzerhof exchange",
        refs: &["J. P. Perdew, K. Burke, M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)"],
    },
    GgaXPbeR => {
        id: 102, name: "gga_x_pbe_r", kind: Kind::Exchange, family: Family::Gga,
        desc: "Perdew, Burke & Ernzerhof exchange (revised)",
        refs: &["Y. Zhang, W. Yang, Phys. Rev. Lett. 80, 890 (1998)"],
    },
    GgaXB88 => {
        id: 106, name: "gga_x_b88", kind: Kind::Exchange, family: Family::Gga,
        desc: "Becke 88 exchange",
        refs: &["A. D. Becke, Phys. Rev. A 38, 3098 (1988)"],
    },
    GgaXPw91 => {
        id: 109, name: "gga_x_pw91", kind: Kind::Exchange, family: Family::Gga,
        desc: "Perdew & Wang 91 exchange",
        refs: &["J. P. Perdew et al., Phys. Rev. B 46, 6671 (1992)"],
    },
    GgaXPbeSol => {
        id: 116, name: "gga_x_pbe_sol", kind: Kind::Exchange, family: Family::Gga,
        desc: "PBEsol exchange",
        refs: &["J. P. Perdew et al., Phys. Rev. Lett. 100, 136406 (2008)"],
    },
    GgaCPbe => {
        id: 130, name: "gga_c_pbe", kind: Kind::Correlation, family: Family::Gga,
        desc: "Perdew, Burke & Ernzerhof correlation",
        refs: &["J. P. Perdew, K. Burke, M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)"],
    },
    GgaCLyp => {
        id: 131, name: "gga_c_lyp", kind: Kind::Correlation, family: Family::Gga,
        desc: "Lee, Yang & Parr correlation",
        refs: &["C. Lee, W. Yang, R. G. Parr, Phys. Rev. B 37, 785 (1988)"],
    },
    GgaCP86 => {
        id: 132, name: "gga_c_p86", kind: Kind::Correlation, family: Family::Gga,
        desc: "Perdew 86 correlation",
        refs: &["J. P. Perdew, Phys. Rev. B 33, 8822 (1986)"],
    },
    GgaCP86Ft => {
        id: 217, name: "gga_c_p86_ft", kind: Kind::Correlation, family: Family::Gga,
        desc: "Perdew 86 correlation with a more accurate ftilde",
        refs: &["J. P. Perdew, Phys. Rev. B 33, 8822 (1986)"],
    },
    MggaXTpss => {
        id: 202, name: "mgga_x_tpss", kind: Kind::Exchange, family: Family::Mgga,
        desc: "Tao, Perdew, Staroverov & Scuseria exchange",
        refs: &[
            "J. Tao et al., Phys. Rev. Lett. 91, 146401 (2003)",
            "J. P. Perdew et al., Phys. Rev. B 69, 6898 (2004)",
        ],
    },
    HybGgaXcB3lyp => {
        id: 402, name: "hyb_gga_xc_b3lyp", kind: Kind::ExchangeCorrelation, family: Family::HybGga,
        desc: "B3LYP hybrid",
        refs: &["P. J. Stephens et al., J. Phys. Chem. 98, 11623 (1994)"],
    },
    HybGgaXcPbeh => {
        id: 406, name: "hyb_gga_xc_pbeh", kind: Kind::ExchangeCorrelation, family: Family::HybGga,
        desc: "PBEH (PBE0)",
        refs: &[
            "C. Adamo, V. Barone, J. Chem. Phys. 110, 6158 (1999)",
            "M. Ernzerhof, G. E. Scuseria, J. Chem. Phys. 110, 5029 (1999)",
        ],
    },
}

impl Functional {
    /// Convenience accessors.
    #[inline]
    pub const fn id(self) -> u32 {
        self.info().id
    }
    #[inline]
    pub const fn name(self) -> &'static str {
        self.info().name
    }
    #[inline]
    pub const fn kind(self) -> Kind {
        self.info().kind
    }
    #[inline]
    pub const fn family(self) -> Family {
        self.info().family
    }
}

impl std::fmt::Display for Functional {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.info().name, self.info().id)
    }
}
