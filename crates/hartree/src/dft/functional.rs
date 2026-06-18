use xcx::{
    CamParams, DispersionModel, DoubleHybridParams, Functional, FunctionalId, Spin, Vv10Params,
};

use crate::dft::error::DftError;

#[derive(Debug, Clone)]
enum Recipe {
    Single(FunctionalId),
    Mix(Vec<(f64, FunctionalId)>),
}

impl Recipe {
    fn construct(&self, spin: Spin) -> Result<Functional, xcx::XcError> {
        match self {
            Recipe::Single(id) => Functional::new(*id, spin),
            Recipe::Mix(parts) => {
                let mut built = Vec::with_capacity(parts.len());
                for (w, id) in parts {
                    built.push((*w, Functional::new(*id, spin)?));
                }
                Functional::mix(built)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionalSpec {
    name: &'static str,
    recipe: Recipe,
    exx_fraction: f64,
    needs_sigma: bool,
    needs_tau: bool,
    cam: Option<CamParams>,
    vv10: Option<Vv10Params>,
    grid_level: usize,
    grid_sensitive: bool,
    double_hybrid: Option<DoubleHybridParams>,
    d4_param_set: Option<&'static str>,
}

const TAU_ALLOWED: &[&str] = &[
    "tpss", "r2scan", "m06-2x", "pw6b95", "b97m-v", "wb97m-v", "pwpb95", "wb97m(2)",
];

impl FunctionalSpec {
    pub fn parse(name: &str) -> Result<Self, DftError> {
        use FunctionalId::*;
        let lower = name.to_ascii_lowercase();
        let (canonical, recipe): (&'static str, Recipe) = match lower.as_str() {
            "svwn" | "lda" => ("svwn", Recipe::Mix(vec![(1.0, LdaX), (1.0, LdaCVwn)])),
            "blyp" => ("blyp", Recipe::Mix(vec![(1.0, GgaXB88), (1.0, GgaCLyp)])),
            "pbe" => ("pbe", Recipe::Mix(vec![(1.0, GgaXPbe), (1.0, GgaCPbe)])),
            "b3lyp" => ("b3lyp", Recipe::Single(HybGgaXcB3lyp)),
            "b3lyp5" => ("b3lyp5", Recipe::Single(HybGgaXcB3lyp5)),
            "pbe0" | "pbeh" => ("pbe0", Recipe::Single(HybGgaXcPbeh)),
            "tpss" => (
                "tpss",
                Recipe::Mix(vec![(1.0, MggaXTpss), (1.0, MggaCTpss)]),
            ),
            "r2scan" => (
                "r2scan",
                Recipe::Mix(vec![(1.0, MggaXR2scan), (1.0, MggaCR2scan)]),
            ),
            "m06-2x" | "m062x" | "m06_2x" => (
                "m06-2x",
                Recipe::Mix(vec![(1.0, HybMggaXM062x), (1.0, MggaCM062x)]),
            ),
            "pw6b95" => ("pw6b95", Recipe::Single(HybMggaXcPw6b95)),
            "b97m-v" | "b97m_v" | "b97mv" => ("b97m-v", Recipe::Single(MggaXcB97mV)),
            "wb97x-v" | "wb97x_v" | "wb97xv" => ("wb97x-v", Recipe::Single(HybGgaXcWb97xV)),
            "wb97m-v" | "wb97m_v" | "wb97mv" => ("wb97m-v", Recipe::Single(HybMggaXcWb97mV)),
            "b2plyp" => ("b2plyp", Recipe::Single(HybGgaXcB2plyp)),
            "revdsd-pbep86" | "revdsd_pbep86" | "revdsdpbep86" => {
                ("revdsd-pbep86", Recipe::Single(HybGgaXcRevdsdPbep86D4))
            }
            "pwpb95" => ("pwpb95", Recipe::Single(HybMggaXcPwpb95)),
            "wb97m(2)" | "wb97m2" | "wb97m-2" | "wb97m_2" => {
                ("wb97m(2)", Recipe::Single(HybMggaXcWb97m2))
            }
            "m06-l" | "m06l" | "m06_l" => {
                return Err(DftError::NeedsTau("m06-l".to_string()));
            }
            other => match FunctionalId::from_name(other) {
                Some(id) => (id.name(), Recipe::Single(id)),
                None => return Err(DftError::UnknownFunctional(name.to_string())),
            },
        };

        let probe = recipe.construct(Spin::Unpolarized)?;
        let info = probe.info();
        if info.needs_tau && !TAU_ALLOWED.contains(&canonical) {
            return Err(DftError::NeedsTau(canonical.to_string()));
        }
        if info.needs_lapl {
            return Err(DftError::NeedsLapl(canonical.to_string()));
        }
        let (cam, vv10) = match info.hybrid {
            Some(h) => (h.cam, h.vv10),
            None => (None, None),
        };
        let double_hybrid = info.double_hybrid();
        let d4_param_set = match &recipe {
            Recipe::Single(_) => info.dispersion(),
            Recipe::Mix(parts) => parts.iter().find_map(|(_, id)| {
                Functional::new(*id, Spin::Unpolarized)
                    .ok()
                    .and_then(|f| f.info().dispersion())
            }),
        }
        .filter(|d| d.model == DispersionModel::D4)
        .map(|d| d.param_set);

        let grid = match &recipe {
            Recipe::Single(_) => info.grid(),
            Recipe::Mix(parts) => {
                let mut rec = info.grid();
                for (_, id) in parts {
                    let g = Functional::new(*id, Spin::Unpolarized)?.info().grid();
                    rec.level = rec.level.max(g.level);
                    rec.grid_sensitive = rec.grid_sensitive || g.grid_sensitive;
                }
                rec
            }
        };

        Ok(Self {
            name: canonical,
            exx_fraction: probe.exx_fraction(),
            needs_sigma: info.needs_sigma,
            needs_tau: info.needs_tau,
            cam,
            vv10,
            grid_level: grid.level as usize,
            grid_sensitive: grid.grid_sensitive,
            double_hybrid,
            d4_param_set,
            recipe,
        })
    }

    pub fn build(&self, spin: Spin) -> Result<Functional, DftError> {
        Ok(self.recipe.construct(spin)?)
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn exx_fraction(&self) -> f64 {
        self.exx_fraction
    }

    pub fn needs_sigma(&self) -> bool {
        self.needs_sigma
    }

    pub fn needs_tau(&self) -> bool {
        self.needs_tau
    }

    pub fn cam(&self) -> Option<CamParams> {
        self.cam
    }

    pub fn vv10(&self) -> Option<Vv10Params> {
        self.vv10
    }

    pub fn recommended_grid_level(&self) -> usize {
        self.grid_level
    }

    pub fn grid_sensitive(&self) -> bool {
        self.grid_sensitive
    }

    pub fn double_hybrid(&self) -> Option<DoubleHybridParams> {
        self.double_hybrid
    }

    pub fn d4_param_set(&self) -> Option<&'static str> {
        self.d4_param_set
    }

    pub fn libxc_id(&self) -> Option<u32> {
        match &self.recipe {
            Recipe::Single(id) => Some(id.as_u32()),
            Recipe::Mix(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xcx::XcInput;

    const TABLE_NAMES: &[&str] = &[
        "svwn", "lda", "blyp", "pbe", "b3lyp", "b3lyp5", "pbe0", "pbeh", "tpss", "r2scan",
    ];

    #[test]
    fn every_table_name_resolves_and_evaluates() {
        for &name in TABLE_NAMES {
            let spec = FunctionalSpec::parse(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            for spin in [Spin::Unpolarized, Spin::Polarized] {
                let f = spec
                    .build(spin)
                    .unwrap_or_else(|e| panic!("{name}/{spin:?}: {e}"));
                let np = 3;
                let ns = spin.channels();
                let rho: Vec<f64> = (0..ns * np).map(|i| 0.1 + 0.01 * i as f64).collect();
                let sigma: Vec<f64> = (0..(2 * ns - 1) * np)
                    .map(|i| 0.02 + 0.01 * i as f64)
                    .collect();
                let tau: Vec<f64> = (0..ns * np).map(|i| 0.05 + 0.01 * i as f64).collect();
                let input = match (spec.needs_sigma(), spec.needs_tau()) {
                    (true, true) => XcInput::gga(&rho, &sigma).with_tau(&tau),
                    (true, false) => XcInput::gga(&rho, &sigma),
                    _ => XcInput::lda(&rho),
                };
                let out = f
                    .eval(np, &input)
                    .unwrap_or_else(|e| panic!("{name}/{spin:?} eval: {e}"));
                assert_eq!(out.exc.len(), np, "{name}/{spin:?}: exc length");
                assert!(
                    out.exc.iter().all(|v| v.is_finite()),
                    "{name}/{spin:?}: non-finite exc {:?}",
                    out.exc
                );
                assert!(
                    out.vrho.iter().all(|v| v.is_finite()),
                    "{name}/{spin:?}: non-finite vrho"
                );
                assert!(
                    out.vsigma.iter().all(|v| v.is_finite()),
                    "{name}/{spin:?}: non-finite vsigma"
                );
            }
        }
    }

    #[test]
    fn b3lyp_is_libxc_402_with_20_percent_exx() {
        let spec = FunctionalSpec::parse("B3LYP").unwrap(); // case-insensitive
        assert_eq!(spec.name(), "b3lyp");
        assert_eq!(spec.libxc_id(), Some(402));
        assert!((spec.exx_fraction() - 0.20).abs() < 1e-12);
    }

    #[test]
    fn b3lyp5_is_libxc_475() {
        let spec = FunctionalSpec::parse("b3lyp5").unwrap();
        assert_eq!(spec.libxc_id(), Some(475));
        assert!((spec.exx_fraction() - 0.20).abs() < 1e-12);
    }

    #[test]
    fn pbe0_is_libxc_406_with_25_percent_exx() {
        for name in ["pbe0", "PBE0", "pbeh"] {
            let spec = FunctionalSpec::parse(name).unwrap();
            assert_eq!(spec.name(), "pbe0", "{name}");
            assert_eq!(spec.libxc_id(), Some(406), "{name}");
            assert!((spec.exx_fraction() - 0.25).abs() < 1e-12, "{name}");
        }
    }

    #[test]
    fn pure_functionals_have_zero_exx() {
        for name in ["svwn", "lda", "blyp", "pbe"] {
            let spec = FunctionalSpec::parse(name).unwrap();
            assert_eq!(spec.exx_fraction(), 0.0, "{name}");
            assert_eq!(spec.libxc_id(), None, "{name}");
        }
    }

    #[test]
    fn pbe_needs_sigma_svwn_does_not() {
        assert!(FunctionalSpec::parse("pbe").unwrap().needs_sigma());
        assert!(FunctionalSpec::parse("blyp").unwrap().needs_sigma());
        assert!(FunctionalSpec::parse("b3lyp").unwrap().needs_sigma());
        assert!(!FunctionalSpec::parse("svwn").unwrap().needs_sigma());
        assert!(!FunctionalSpec::parse("lda").unwrap().needs_sigma());
    }

    #[test]
    fn canonical_libxc_names_pass_through() {
        let pbe_x = FunctionalSpec::parse("gga_x_pbe").unwrap();
        assert_eq!(pbe_x.name(), "gga_x_pbe");
        assert_eq!(pbe_x.libxc_id(), Some(101));
        assert_eq!(pbe_x.exx_fraction(), 0.0);

        let slater = FunctionalSpec::parse("lda_x").unwrap();
        assert_eq!(slater.libxc_id(), Some(1));
    }

    #[test]
    fn unknown_name_errors() {
        assert!(matches!(
            FunctionalSpec::parse("not_a_functional"),
            Err(DftError::UnknownFunctional(_))
        ));
    }

    #[test]
    fn meta_gga_is_rejected() {
        for name in [
            "mgga_x_tpss",
            "mgga_c_tpss",
            "mgga_x_r2scan",
            "mgga_c_r2scan",
            "mgga_x_m06_l",
            "mgga_c_m06_l",
            "m06-l",
            "M06-L",
            "m06l",
        ] {
            let got = FunctionalSpec::parse(name);
            assert!(
                matches!(got, Err(DftError::NeedsTau(_))),
                "{name} (meta-GGA) should be rejected with NeedsTau, got {got:?}"
            );
        }
    }

    #[test]
    fn round03_functionals_metadata() {
        let m062x = FunctionalSpec::parse("M06-2X").unwrap();
        assert_eq!(m062x.name(), "m06-2x");
        assert!((m062x.exx_fraction() - 0.54).abs() < 1e-12);
        assert!(m062x.needs_tau() && m062x.needs_sigma());
        assert!(m062x.cam().is_none() && m062x.vv10().is_none());
        assert!(m062x.grid_sensitive());
        assert_eq!(m062x.recommended_grid_level(), 4);

        let pw6b95 = FunctionalSpec::parse("pw6b95").unwrap();
        assert_eq!(pw6b95.libxc_id(), Some(451));
        assert!((pw6b95.exx_fraction() - 0.28).abs() < 1e-12);
        assert!(pw6b95.cam().is_none() && pw6b95.vv10().is_none());
        assert!(!pw6b95.grid_sensitive());

        let b97mv = FunctionalSpec::parse("b97m-v").unwrap();
        assert_eq!(b97mv.libxc_id(), Some(254));
        assert_eq!(b97mv.exx_fraction(), 0.0);
        assert!(b97mv.cam().is_none());
        let vv10 = b97mv.vv10().unwrap();
        assert_eq!((vv10.b, vv10.c), (6.0, 0.01));
        assert!(b97mv.grid_sensitive());

        let wb97xv = FunctionalSpec::parse("wb97x-v").unwrap();
        assert_eq!(wb97xv.libxc_id(), Some(466));
        assert!(!wb97xv.needs_tau());
        let cam = wb97xv.cam().unwrap();
        assert_eq!((cam.omega, cam.alpha, cam.beta), (0.30, 0.167, 0.833));
        assert!((wb97xv.exx_fraction() - cam.alpha).abs() < 1e-12);
        assert!(wb97xv.vv10().is_some());

        let wb97mv = FunctionalSpec::parse("wb97m-v").unwrap();
        assert_eq!(wb97mv.libxc_id(), Some(531));
        assert!(wb97mv.needs_tau());
        let cam = wb97mv.cam().unwrap();
        assert_eq!((cam.omega, cam.alpha, cam.beta), (0.30, 0.15, 0.85));
        assert!(wb97mv.vv10().is_some());

        for name in ["m06-2x", "pw6b95", "b97m-v", "wb97x-v", "wb97m-v"] {
            let spec = FunctionalSpec::parse(name).unwrap();
            for spin in [Spin::Unpolarized, Spin::Polarized] {
                spec.build(spin)
                    .unwrap_or_else(|e| panic!("{name}/{spin:?}: {e}"));
            }
        }

        let pbe0 = FunctionalSpec::parse("pbe0").unwrap();
        assert!(pbe0.cam().is_none() && pbe0.vv10().is_none());
        assert!(!pbe0.grid_sensitive());
        assert_eq!(pbe0.recommended_grid_level(), 3);
    }

    #[test]
    fn composite_3c_functionals_resolve_with_correct_metadata() {
        let b97 = FunctionalSpec::parse("gga_xc_b97_3c").unwrap();
        assert_eq!(b97.name(), "gga_xc_b97_3c");
        assert_eq!(b97.libxc_id(), Some(327));
        assert_eq!(b97.exx_fraction(), 0.0);
        assert!(b97.needs_sigma() && !b97.needs_tau());
        assert!(b97.cam().is_none() && b97.vv10().is_none());
        assert!(!b97.grid_sensitive());
        assert_eq!(b97.recommended_grid_level(), 3);

        let pbeh = FunctionalSpec::parse("hyb_gga_xc_pbeh_3c").unwrap();
        assert_eq!(pbeh.name(), "hyb_gga_xc_pbeh_3c");
        assert_eq!(pbeh.libxc_id(), Some(100_005));
        assert!((pbeh.exx_fraction() - 0.42).abs() < 1e-12);
        assert!(pbeh.needs_sigma() && !pbeh.needs_tau());
        assert!(pbeh.cam().is_none() && pbeh.vv10().is_none());
        assert!(pbeh.double_hybrid().is_none());
        assert_eq!(pbeh.recommended_grid_level(), 3);

        for name in ["gga_xc_b97_3c", "hyb_gga_xc_pbeh_3c"] {
            let spec = FunctionalSpec::parse(name).unwrap();
            for spin in [Spin::Unpolarized, Spin::Polarized] {
                let f = spec.build(spin).unwrap();
                let ns = spin.channels();
                let rho: Vec<f64> = (0..ns * 2).map(|i| 0.2 + 0.05 * i as f64).collect();
                let sigma: Vec<f64> = (0..(2 * ns - 1) * 2)
                    .map(|i| 0.03 + 0.01 * i as f64)
                    .collect();
                let out = f.eval(2, &XcInput::gga(&rho, &sigma)).unwrap();
                assert!(out.exc.iter().all(|v| v.is_finite()), "{name}/{spin:?}");
            }
        }
    }

    #[test]
    fn tpss_and_r2scan_construct() {
        for name in ["tpss", "TPSS", "r2scan", "R2SCAN"] {
            let spec = FunctionalSpec::parse(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(spec.exx_fraction(), 0.0, "{name}");
            assert!(spec.needs_sigma(), "{name}");
            assert!(spec.needs_tau(), "{name}");
            assert_eq!(spec.libxc_id(), None, "{name}: composite, no single id");
            for spin in [Spin::Unpolarized, Spin::Polarized] {
                spec.build(spin)
                    .unwrap_or_else(|e| panic!("{name}/{spin:?}: {e}"));
            }
        }
        assert!(!FunctionalSpec::parse("pbe").unwrap().needs_tau());
        assert!(!FunctionalSpec::parse("b3lyp").unwrap().needs_tau());
    }
}
