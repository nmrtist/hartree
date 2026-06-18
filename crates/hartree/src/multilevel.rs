use crate::basis::BasisSet;
use crate::core::Molecule;
use crate::dft::FunctionalSpec;
use crate::disp::{Dispersion, GcpParams, SrbParams};
use crate::ext::ensemble::Ensemble;

use crate::job::{FrequencyData, Job, JobOptions, JobResult, Method};

#[derive(Debug, Clone)]
pub struct Level {
    pub label: String,
    pub method: Method,
    pub basis: String,
    pub dispersion: Option<Dispersion>,
    pub gcp: Option<GcpParams>,
    pub srb: Option<SrbParams>,
    pub grid_level: usize,
}

#[derive(Debug, Clone)]
pub struct MultiLevelSpec {
    pub sp: Level,
    pub opt: Level,
}

impl MultiLevelSpec {
    pub fn label(&self) -> String {
        format!("{} // {}", self.sp.label, self.opt.label)
    }
}

pub fn is_multilevel(spec: &str) -> bool {
    spec.contains("//")
}

pub fn parse_spec(spec: &str, multiplicity: u32) -> Result<Option<MultiLevelSpec>, String> {
    if !is_multilevel(spec) {
        return Ok(None);
    }
    let parts: Vec<&str> = spec.split("//").collect();
    if parts.len() != 2 {
        return Err(format!(
            "multi-level spec {spec:?}: expected exactly one '//' separating \
             SP_SPEC // OPT_SPEC (got {})",
            parts.len() - 1
        ));
    }
    let sp = parse_level(parts[0], multiplicity)?;
    let opt = parse_level(parts[1], multiplicity)?;
    Ok(Some(MultiLevelSpec { sp, opt }))
}

pub fn parse_level(token: &str, multiplicity: u32) -> Result<Level, String> {
    let token = token.trim();
    if token.is_empty() {
        return Err(
            "empty level spec: each side of '//' must be method[/basis], \
             e.g. wb97m-v/def2-tzvpp // r2scan-3c"
                .into(),
        );
    }
    let mut parts = token.splitn(3, '/');
    let method_tok = parts
        .next()
        .expect("splitn yields at least one part")
        .trim()
        .to_ascii_lowercase();
    let basis_tok = parts.next().map(|s| s.trim().to_ascii_lowercase());
    if parts.next().is_some() {
        return Err(format!(
            "level spec {token:?} has more than one '/': expected method[/basis]"
        ));
    }
    if method_tok.is_empty() {
        return Err(format!("level spec {token:?} has an empty method"));
    }
    if basis_tok.as_deref() == Some("") {
        return Err(format!(
            "level spec {token:?} has an empty basis after '/': \
             write method/basis or drop the '/'"
        ));
    }

    let (base_method, want_disp) = if let Some(base) = method_tok.strip_suffix("-d3") {
        (base.to_string(), Some(false))
    } else if let Some(base) = method_tok.strip_suffix("-d4") {
        (base.to_string(), Some(true))
    } else {
        (method_tok.clone(), None)
    };

    if let Some(c) = crate::composite::composite(&base_method) {
        if let Some(b) = &basis_tok {
            return Err(format!(
                "{} is a composite method and carries its own basis ({}); \
                 drop \"/{b}\" from the level spec",
                c.keyword, c.basis_label
            ));
        }
        if want_disp.is_some() {
            return Err(format!(
                "{} defines its own {} and short-range corrections; \
                 a -d3/-d4 suffix is not allowed",
                c.keyword,
                c.dispersion.label()
            ));
        }
        return Ok(Level {
            label: c.keyword.to_string(),
            method: Method::Dft(FunctionalSpec::parse(c.functional).map_err(|e| e.to_string())?),
            basis: c.basis.to_string(),
            dispersion: Some(c.dispersion),
            gcp: c.gcp,
            srb: c.srb,
            grid_level: c.grid_level,
        });
    }

    let method = match base_method.as_str() {
        "rhf" => Method::Rhf,
        "uhf" => Method::Uhf,
        "hf" => {
            if multiplicity > 1 {
                Method::Uhf
            } else {
                Method::Rhf
            }
        }
        "rohf" => Method::Rohf,
        "mp2" => Method::Mp2,
        "ccsd" => Method::Ccsd,
        "ccsd(t)" | "ccsdt" => Method::CcsdT,
        other => match FunctionalSpec::parse(other) {
            Ok(spec) => Method::Dft(spec),
            Err(crate::dft::DftError::UnknownFunctional(_)) => {
                return Err(format!(
                    "unknown method {other:?} in level spec {token:?} (expected hf, rhf, uhf, \
                     rohf, mp2, ccsd, ccsd(t), a functional like pbe/b3lyp/r2scan, or a \
                     composite like r2scan-3c; -d3/-d4 suffixes are allowed)"
                ));
            }
            Err(err) => return Err(err.to_string()),
        },
    };

    let basis = basis_tok.ok_or_else(|| {
        format!(
            "method {method_tok:?} requires an explicit basis in a multi-level spec: \
             write \"{method_tok}/<basis>\" (only the 3c composites carry their own basis)"
        )
    })?;
    BasisSet::load(&basis).map_err(|e| format!("level spec {token:?}: {e}"))?;

    let dispersion = match want_disp {
        None => None,
        Some(d4) => {
            let (suffix, model) = if d4 { ("-d4", "D4") } else { ("-d3", "D3(BJ)") };
            if matches!(method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
                return Err(format!(
                    "{suffix} is not supported for post-HF methods ({base_method}); \
                     the {model} correction applies to HF and DFT functionals"
                ));
            }
            let param_key = match &method {
                Method::Rhf | Method::Uhf | Method::Rohf => "hf".to_string(),
                Method::Dft(spec) => spec
                    .d4_param_set()
                    .map(str::to_string)
                    .unwrap_or_else(|| spec.name().to_string()),
                _ => unreachable!("post-HF rejected above"),
            };
            match Dispersion::for_method(d4, &param_key) {
                Some(d) => Some(d),
                None => {
                    return Err(format!(
                        "no {model} parametrization exists for {param_key} \
                         (supported: pbe, blyp, b3lyp, b3lyp5, pbe0, tpss, r2scan, hf; \
                         D4 additionally: b2plyp, revdsd-pbep86, pwpb95)"
                    ));
                }
            }
        }
    };

    let grid_level = match &method {
        Method::Dft(spec) if spec.grid_sensitive() => spec.recommended_grid_level(),
        _ => 3,
    };

    Ok(Level {
        label: format!("{method_tok}/{basis}"),
        method,
        basis,
        dispersion,
        gcp: None,
        srb: None,
        grid_level,
    })
}

#[derive(Debug, Clone)]
pub struct MultiLevelOptions {
    pub compute_frequencies: bool,
    pub symmetry_number: u32,
    pub qrrho_w0_cm1: f64,
    pub grid_override: Option<usize>,
    pub all_electron: bool,
    pub ri_mp2: bool,
    pub solvent_eps: Option<f64>,
    pub smd: Option<String>,
    pub alpb: Option<String>,
    pub gbsa: Option<String>,
}

impl Default for MultiLevelOptions {
    fn default() -> Self {
        Self {
            compute_frequencies: false,
            symmetry_number: 1,
            qrrho_w0_cm1: crate::props::thermo::QRRHO_W0_DEFAULT_CM1,
            grid_override: None,
            all_electron: false,
            ri_mp2: false,
            solvent_eps: None,
            smd: None,
            alpb: None,
            gbsa: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompositeThermo {
    pub e_low: f64,
    pub h_corr: f64,
    pub g_corr: f64,
    pub g_corr_qrrho: f64,
    pub enthalpy: f64,
    pub gibbs: f64,
    pub gibbs_qrrho: f64,
    pub freq: FrequencyData,
}

#[derive(Debug, Clone)]
pub struct MultiLevelResult {
    pub geometry: Molecule,
    pub opt: JobResult,
    pub sp: JobResult,
    pub e_low: f64,
    pub e_high: f64,
    pub thermo: Option<CompositeThermo>,
}

fn stage_options(level: &Level, opts: &MultiLevelOptions) -> JobOptions {
    JobOptions {
        all_electron: opts.all_electron,
        grid_level: opts.grid_override.unwrap_or(level.grid_level),
        dispersion: level.dispersion,
        gcp: level.gcp,
        srb: level.srb,
        solvent_eps: opts.solvent_eps,
        smd: opts.smd.clone(),
        alpb: opts.alpb.clone(),
        gbsa: opts.gbsa.clone(),
        symmetry_number: opts.symmetry_number,
        qrrho_w0_cm1: opts.qrrho_w0_cm1,
        ..JobOptions::default()
    }
}

pub fn run_multilevel(
    molecule: &Molecule,
    spec: &MultiLevelSpec,
    opts: &MultiLevelOptions,
) -> Result<MultiLevelResult, String> {
    let lo = &spec.opt;
    let hi = &spec.sp;

    let opt = Job {
        molecule: molecule.clone(),
        basis: lo.basis.clone(),
        method: lo.method.clone(),
        options: JobOptions {
            optimize_geometry: true,
            ..stage_options(lo, opts)
        },
    }
    .run()
    .map_err(|e| format!("multi-level OPT step ({}): {e}", lo.label))?;
    let opt_geo = opt
        .optimized_geometry
        .as_ref()
        .ok_or_else(|| format!("multi-level OPT step ({}) returned no geometry", lo.label))?;
    if !opt_geo.converged {
        return Err(format!(
            "multi-level OPT step ({}) did not converge",
            lo.label
        ));
    }
    let atoms = molecule
        .atoms
        .iter()
        .zip(&opt_geo.positions)
        .map(|(a, p)| crate::core::Atom::new(a.element, *p))
        .collect();
    let geometry = Molecule::new(atoms, molecule.charge, molecule.multiplicity);
    let e_low = opt.best_energy();

    let low_freq = if opts.compute_frequencies {
        let fj = Job {
            molecule: geometry.clone(),
            basis: lo.basis.clone(),
            method: lo.method.clone(),
            options: JobOptions {
                compute_frequencies: true,
                ..stage_options(lo, opts)
            },
        }
        .run()
        .map_err(|e| format!("multi-level FREQ step ({}): {e}", lo.label))?;
        if !fj.converged() {
            return Err(format!(
                "multi-level FREQ step ({}) did not converge",
                lo.label
            ));
        }
        let e_low_freq = fj.best_energy();
        let fd = fj.frequencies.clone().ok_or_else(|| {
            format!(
                "multi-level FREQ step ({}) returned no frequencies",
                lo.label
            )
        })?;
        Some((e_low_freq, fd))
    } else {
        None
    };

    let sp = Job {
        molecule: geometry.clone(),
        basis: hi.basis.clone(),
        method: hi.method.clone(),
        options: JobOptions {
            ri_mp2: opts.ri_mp2,
            ..stage_options(hi, opts)
        },
    }
    .run()
    .map_err(|e| format!("multi-level SP step ({}): {e}", hi.label))?;
    if !sp.converged() {
        return Err(format!(
            "multi-level SP step ({}) did not converge",
            hi.label
        ));
    }
    let e_high = sp.best_energy();

    let thermo = low_freq.map(|(e_low_freq, fd)| {
        let t = &fd.thermochemistry;
        let h_corr = t.enthalpy - e_low_freq;
        let g_corr = t.gibbs - e_low_freq;
        let g_corr_qrrho = t.gibbs_qrrho - e_low_freq;
        CompositeThermo {
            e_low: e_low_freq,
            h_corr,
            g_corr,
            g_corr_qrrho,
            enthalpy: e_high + h_corr,
            gibbs: e_high + g_corr,
            gibbs_qrrho: e_high + g_corr_qrrho,
            freq: fd,
        }
    });

    Ok(MultiLevelResult {
        geometry,
        opt,
        sp,
        e_low,
        e_high,
        thermo,
    })
}

pub const ENSEMBLE_RERANK_CAP: usize = 6;

#[derive(Debug, Clone)]
pub struct RankedConformer {
    pub molecule: Molecule,
    pub e_low: f64,
    pub e_high: f64,
    pub weight: f64,
}

pub fn rerank_ensemble(
    ensemble: &Ensemble,
    spec: &MultiLevelSpec,
    opts: &MultiLevelOptions,
    cap: usize,
) -> Result<Vec<RankedConformer>, String> {
    if ensemble.is_empty() {
        return Err("multi-level re-ranking needs a non-empty conformer ensemble".into());
    }
    let ml_opts = MultiLevelOptions {
        compute_frequencies: false,
        ..opts.clone()
    };
    let mut ranked = Vec::new();
    for (i, conf) in ensemble.conformers.iter().take(cap.max(1)).enumerate() {
        let res = run_multilevel(&conf.molecule, spec, &ml_opts)
            .map_err(|e| format!("conformer {} re-ranking: {e}", i + 1))?;
        ranked.push(RankedConformer {
            molecule: res.geometry,
            e_low: res.e_low,
            e_high: res.e_high,
            weight: 0.0,
        });
    }
    ranked.sort_by(|a, b| a.e_high.partial_cmp(&b.e_high).expect("finite energies"));
    let weights = Ensemble::new(
        ranked
            .iter()
            .map(|r| crate::ext::ensemble::Conformer {
                molecule: r.molecule.clone(),
                energy: r.e_high,
            })
            .collect(),
    )
    .boltzmann_weights(298.15);
    for (r, w) in ranked.iter_mut().zip(weights) {
        r.weight = w;
    }
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_level_spec_is_not_multilevel() {
        assert!(parse_spec("hf", 1).unwrap().is_none());
        assert!(parse_spec("b3lyp/def2-svp", 1).unwrap().is_none());
        assert!(!is_multilevel("r2scan-3c"));
    }

    #[test]
    fn valid_plain_spec_parses() {
        let s = parse_spec("hf/6-31g // hf/sto-3g", 1).unwrap().unwrap();
        assert!(matches!(s.sp.method, Method::Rhf));
        assert_eq!(s.sp.basis, "6-31g");
        assert!(matches!(s.opt.method, Method::Rhf));
        assert_eq!(s.opt.basis, "sto-3g");
        assert_eq!(s.label(), "hf/6-31g // hf/sto-3g");
        assert!(s.sp.dispersion.is_none() && s.sp.gcp.is_none() && s.sp.srb.is_none());
    }

    #[test]
    fn hf_keyword_resolves_by_multiplicity() {
        let s = parse_spec("hf/sto-3g // hf/sto-3g", 2).unwrap().unwrap();
        assert!(matches!(s.sp.method, Method::Uhf));
        assert!(matches!(s.opt.method, Method::Uhf));
    }

    #[test]
    fn composite_levels_carry_their_own_parts() {
        let s = parse_spec("wb97m-v/def2-tzvpp // r2scan-3c", 1)
            .unwrap()
            .unwrap();
        assert_eq!(s.opt.label, "r2scan-3c");
        assert_eq!(s.opt.basis, "def2-mtzvpp");
        assert!(s.opt.dispersion.is_some() && s.opt.gcp.is_some());
        assert_eq!(s.opt.grid_level, 4);
        assert!(matches!(s.sp.method, Method::Dft(_)));
        assert_eq!(s.sp.basis, "def2-tzvpp");
    }

    #[test]
    fn composite_with_explicit_basis_is_rejected() {
        let err = parse_spec("hf/sto-3g // r2scan-3c/def2-svp", 1).unwrap_err();
        assert!(
            err.contains("carries its own basis"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn composite_with_dispersion_suffix_is_rejected() {
        let err = parse_level("b97-3c-d4", 1).unwrap_err();
        assert!(err.contains("-d3/-d4 suffix"), "unexpected error: {err}");
    }

    #[test]
    fn plain_method_without_basis_is_rejected() {
        let err = parse_spec("pbe // hf/sto-3g", 1).unwrap_err();
        assert!(
            err.contains("requires an explicit basis"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn malformed_specs_are_rejected_with_named_errors() {
        let err = parse_spec("a // b // c", 1).unwrap_err();
        assert!(err.contains("exactly one '//'"), "unexpected error: {err}");
        let err = parse_spec("hf/sto-3g // ", 1).unwrap_err();
        assert!(err.contains("empty level spec"), "unexpected error: {err}");
        let err = parse_level("hf/sto-3g/extra", 1).unwrap_err();
        assert!(err.contains("more than one '/'"), "unexpected error: {err}");
        let err = parse_level("hf/", 1).unwrap_err();
        assert!(err.contains("empty basis"), "unexpected error: {err}");
        let err = parse_spec("nosuchmethod/sto-3g // hf/sto-3g", 1).unwrap_err();
        assert!(err.contains("unknown method"), "unexpected error: {err}");
        assert!(parse_level("hf/no-such-basis", 1).is_err());
    }

    #[test]
    fn dispersion_suffixes_resolve_per_level() {
        let s = parse_spec("pbe-d4/def2-svp // hf-d3/sto-3g", 1)
            .unwrap()
            .unwrap();
        assert!(matches!(s.sp.dispersion, Some(Dispersion::D4(_))));
        assert!(matches!(s.opt.dispersion, Some(Dispersion::D3(_))));
        let err = parse_level("mp2-d3/sto-3g", 1).unwrap_err();
        assert!(err.contains("post-HF"), "unexpected error: {err}");
    }
}
