use crate::disp::{D3Params, D4Params, Dispersion, GcpParams, SrbParams};

#[derive(Debug, Clone, Copy)]
pub struct Composite {
    pub keyword: &'static str,
    pub functional: &'static str,
    pub basis: &'static str,
    pub basis_label: &'static str,
    pub grid_level: usize,
    pub dispersion: Dispersion,
    pub disp_label: &'static str,
    pub gcp: Option<GcpParams>,
    pub srb: Option<SrbParams>,
}

pub const COMPOSITES: &[Composite] = &[
    Composite {
        keyword: "r2scan-3c",
        functional: "r2scan",
        basis: "def2-mtzvpp",
        basis_label: "def2-mTZVPP",
        // Production grid level 3, not the dense reference-quality level 4. r2SCAN is the
        // re-regularized SCAN (Furness et al., J. Phys. Chem. Lett. 11, 8208 (2020)), whose
        // re-regularization was designed to tame SCAN's grid sensitivity, so the
        // point-efficient pruned level-3 grid integrates it to ~1e-5 Eh of the reference —
        // well inside chemical accuracy and matching the integration-accuracy tradeoff
        // mainstream composite implementations ship as their default. Reference quality is
        // still reachable on demand via grid level 4 / `--grid 4`.
        grid_level: 3,
        dispersion: Dispersion::D4(D4Params::R2SCAN_3C),
        disp_label: "D4  (r2scan-3c)",
        gcp: Some(GcpParams::R2SCAN_3C),
        srb: None,
    },
    Composite {
        keyword: "b3lyp-3c",
        functional: "b3lyp5",
        basis: "def2-msvp",
        basis_label: "def2-mSVP",
        grid_level: 3,
        dispersion: Dispersion::D3(D3Params::B3LYP_3C),
        disp_label: "D3(BJ)-ATM",
        gcp: Some(GcpParams::B3LYP_3C),
        srb: None,
    },
    Composite {
        keyword: "b97-3c",
        functional: "gga_xc_b97_3c",
        basis: "def2-mtzvp",
        basis_label: "mTZVP (def2-mTZVP)",
        grid_level: 3,
        dispersion: Dispersion::D3(D3Params::B97_3C),
        disp_label: "D3(BJ)-ATM",
        gcp: None,
        srb: Some(SrbParams::B97_3C),
    },
    Composite {
        keyword: "pbeh-3c",
        functional: "hyb_gga_xc_pbeh_3c",
        basis: "def2-msvp",
        basis_label: "def2-mSVP",
        grid_level: 3,
        dispersion: Dispersion::D3(D3Params::PBEH_3C),
        disp_label: "D3(BJ)-ATM",
        gcp: Some(GcpParams::PBEH_3C),
        srb: None,
    },
];

pub fn composite(keyword: &str) -> Option<&'static Composite> {
    COMPOSITES
        .iter()
        .find(|c| c.keyword.eq_ignore_ascii_case(keyword))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive_and_complete() {
        assert_eq!(composite("R2SCAN-3C").unwrap().basis, "def2-mtzvpp");
        assert_eq!(composite("b3lyp-3c").unwrap().functional, "b3lyp5");
        assert_eq!(composite("B97-3c").unwrap().basis, "def2-mtzvp");
        assert_eq!(
            composite("PBEh-3c").unwrap().functional,
            "hyb_gga_xc_pbeh_3c"
        );
        assert!(composite("pbe").is_none());
    }

    #[test]
    fn entries_reference_existing_parts() {
        for c in COMPOSITES {
            assert!(
                crate::basis::BasisSet::load(c.basis).is_ok(),
                "{}: basis {} not loadable",
                c.keyword,
                c.basis
            );
            assert!(
                crate::dft::FunctionalSpec::parse(c.functional).is_ok(),
                "{}: functional {} not parsable",
                c.keyword,
                c.functional
            );
            assert!(c.keyword.ends_with("-3c"));
            assert!(c.grid_level <= 4);
            assert!(
                c.gcp.is_some() || c.srb.is_some(),
                "{}: no short-range correction term",
                c.keyword
            );
        }
    }

    #[test]
    fn b97_3c_uses_srb_not_gcp_and_pbeh_3c_uses_gcp() {
        let b97 = composite("b97-3c").unwrap();
        assert!(b97.gcp.is_none() && b97.srb == Some(SrbParams::B97_3C));
        assert!(matches!(
            b97.dispersion,
            Dispersion::D3(p) if p == D3Params::B97_3C
        ));
        let pbeh = composite("pbeh-3c").unwrap();
        assert!(pbeh.srb.is_none() && pbeh.gcp == Some(GcpParams::PBEH_3C));
        assert!(matches!(
            pbeh.dispersion,
            Dispersion::D3(p) if p == D3Params::PBEH_3C && p.s8 == 0.0
        ));
    }
}
