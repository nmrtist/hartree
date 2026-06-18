use crate::core::Molecule;

use crate::job::{FrequencyData, Job, JobOptions, JobResult, Method, PostHfResult};
use crate::multilevel::{Level, parse_level};

pub const W1_KEYWORD: &str = "hartree-w1";

pub fn basis_cardinal(basis: &str) -> Option<u32> {
    match basis.to_ascii_lowercase().as_str() {
        "cc-pvdz" | "def2-svp" => Some(2),
        "cc-pvtz" | "aug-cc-pvtz" | "def2-tzvp" | "def2-tzvpp" => Some(3),
        "cc-pvqz" | "def2-qzvp" | "def2-qzvpp" => Some(4),
        _ => None,
    }
}

pub fn extrapolate_hf_karton_martin(e_lo: f64, e_hi: f64, l_lo: u32, l_hi: u32) -> f64 {
    assert!(l_lo < l_hi, "cardinal numbers must be increasing");
    let f = |l: u32| (l as f64 + 1.0) * (-9.0 * (l as f64).sqrt()).exp();
    let (f_lo, f_hi) = (f(l_lo), f(l_hi));
    (e_hi * f_lo - e_lo * f_hi) / (f_lo - f_hi)
}

pub fn extrapolate_corr_n3(e_lo: f64, e_hi: f64, l_lo: u32, l_hi: u32) -> f64 {
    assert!(l_lo < l_hi, "cardinal numbers must be increasing");
    let (c_lo, c_hi) = ((l_lo as f64).powi(3), (l_hi as f64).powi(3));
    (c_hi * e_hi - c_lo * e_lo) / (c_hi - c_lo)
}

#[derive(Debug, Clone)]
pub struct W1Options {
    pub opt_level: String,
    pub basis_small: String,
    pub basis_large: String,
    pub compute_frequencies: bool,
    pub symmetry_number: u32,
    pub qrrho_w0_cm1: f64,
    pub all_electron: bool,
}

impl Default for W1Options {
    fn default() -> Self {
        Self {
            opt_level: "b3lyp/cc-pvtz".into(),
            basis_small: "cc-pvtz".into(),
            basis_large: "cc-pvqz".into(),
            compute_frequencies: true,
            symmetry_number: 1,
            qrrho_w0_cm1: crate::props::thermo::QRRHO_W0_DEFAULT_CM1,
            all_electron: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct W1Thermo {
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
pub struct W1Result {
    pub geometry: Molecule,
    pub opt_label: String,
    pub e_opt: f64,
    pub basis_small: String,
    pub basis_large: String,
    pub cardinal_small: u32,
    pub cardinal_large: u32,
    pub e_hf_small: f64,
    pub e_hf_large: f64,
    pub e_hf_cbs: f64,
    pub e_ccsd_corr_small: f64,
    pub e_ccsd_corr_large: f64,
    pub e_ccsd_corr_cbs: f64,
    pub e_t_small: f64,
    pub n_frozen: usize,
    pub thermo: Option<W1Thermo>,
}

impl W1Result {
    pub fn electronic_energy(&self) -> f64 {
        self.e_hf_cbs + self.e_ccsd_corr_cbs + self.e_t_small
    }
}

fn resolve_pair(opts: &W1Options) -> Result<(u32, u32), String> {
    for b in [&opts.basis_small, &opts.basis_large] {
        crate::basis::BasisSet::load(b)
            .map_err(|e| format!("hartree-W1 extrapolation pair: {e}"))?;
    }
    let lo = basis_cardinal(&opts.basis_small).ok_or_else(|| {
        format!(
            "hartree-W1: basis {:?} has no defined cardinal number for CBS extrapolation \
             (use a cc-pVnZ or def2 n-zeta set)",
            opts.basis_small
        )
    })?;
    let hi = basis_cardinal(&opts.basis_large).ok_or_else(|| {
        format!(
            "hartree-W1: basis {:?} has no defined cardinal number for CBS extrapolation \
             (use a cc-pVnZ or def2 n-zeta set)",
            opts.basis_large
        )
    })?;
    if lo >= hi {
        return Err(format!(
            "hartree-W1: the extrapolation pair must have increasing cardinal numbers \
             (got {:?} (n={lo}) and {:?} (n={hi}))",
            opts.basis_small, opts.basis_large
        ));
    }
    Ok((lo, hi))
}

fn run_stage(label: &str, job: Job) -> Result<JobResult, String> {
    let res = job
        .run()
        .map_err(|e| format!("hartree-W1 {label} stage: {e}"))?;
    if !res.converged() {
        return Err(format!("hartree-W1 {label} stage did not converge"));
    }
    Ok(res)
}

pub fn run_w1(molecule: &Molecule, opts: &W1Options) -> Result<W1Result, String> {
    if molecule.multiplicity != 1 {
        return Err(format!(
            "hartree-W1 requires a closed-shell molecule (RHF-reference CCSD(T)); got \
             multiplicity {} — open-shell coupled cluster is not implemented",
            molecule.multiplicity
        ));
    }
    let (l_small, l_large) = resolve_pair(opts)?;
    let opt_level: Level = parse_level(&opts.opt_level, molecule.multiplicity)
        .map_err(|e| format!("hartree-W1 opt level: {e}"))?;

    let stage_opts = |level: &Level| JobOptions {
        grid_level: level.grid_level,
        dispersion: level.dispersion,
        gcp: level.gcp,
        srb: level.srb,
        symmetry_number: opts.symmetry_number,
        qrrho_w0_cm1: opts.qrrho_w0_cm1,
        ..JobOptions::default()
    };
    let opt = run_stage(
        "OPT",
        Job {
            molecule: molecule.clone(),
            basis: opt_level.basis.clone(),
            method: opt_level.method.clone(),
            options: JobOptions {
                optimize_geometry: true,
                ..stage_opts(&opt_level)
            },
        },
    )?;
    let positions = &opt
        .optimized_geometry
        .as_ref()
        .ok_or_else(|| {
            format!(
                "hartree-W1 OPT stage ({}) returned no geometry",
                opt_level.label
            )
        })?
        .positions;
    let atoms = molecule
        .atoms
        .iter()
        .zip(positions)
        .map(|(a, p)| crate::core::Atom::new(a.element, *p))
        .collect();
    let geometry = Molecule::new(atoms, molecule.charge, molecule.multiplicity);
    let e_opt = opt.best_energy();

    let low_freq = if opts.compute_frequencies {
        let fj = run_stage(
            "FREQ",
            Job {
                molecule: geometry.clone(),
                basis: opt_level.basis.clone(),
                method: opt_level.method.clone(),
                options: JobOptions {
                    compute_frequencies: true,
                    ..stage_opts(&opt_level)
                },
            },
        )?;
        let e_low = fj.best_energy();
        let fd = fj.frequencies.clone().ok_or_else(|| {
            format!(
                "hartree-W1 FREQ stage ({}) returned no frequencies",
                opt_level.label
            )
        })?;
        Some((e_low, fd))
    } else {
        None
    };

    let cc_options = JobOptions {
        all_electron: opts.all_electron,
        ..JobOptions::default()
    };
    let small = run_stage(
        "CCSD(T)/small",
        Job {
            molecule: geometry.clone(),
            basis: opts.basis_small.clone(),
            method: Method::CcsdT,
            options: cc_options.clone(),
        },
    )?;
    let (e_ccsd_corr_small, e_t_small, n_frozen) = match &small.post_hf {
        Some(PostHfResult::CcsdT { result, n_frozen }) => (
            result.ccsd.correlation_energy,
            result.triples_energy,
            *n_frozen,
        ),
        _ => return Err("hartree-W1 CCSD(T)/small stage returned no CCSD(T) result".into()),
    };
    let e_hf_small = small.scf.energy;

    let large = run_stage(
        "CCSD/large",
        Job {
            molecule: geometry.clone(),
            basis: opts.basis_large.clone(),
            method: Method::Ccsd,
            options: cc_options,
        },
    )?;
    let e_ccsd_corr_large = match &large.post_hf {
        Some(PostHfResult::Ccsd { result, .. }) => result.correlation_energy,
        _ => return Err("hartree-W1 CCSD/large stage returned no CCSD result".into()),
    };
    let e_hf_large = large.scf.energy;

    let e_hf_cbs = extrapolate_hf_karton_martin(e_hf_small, e_hf_large, l_small, l_large);
    let e_ccsd_corr_cbs =
        extrapolate_corr_n3(e_ccsd_corr_small, e_ccsd_corr_large, l_small, l_large);
    let e_final = e_hf_cbs + e_ccsd_corr_cbs + e_t_small;

    let thermo = low_freq.map(|(e_low, fd)| {
        let t = &fd.thermochemistry;
        let h_corr = t.enthalpy - e_low;
        let g_corr = t.gibbs - e_low;
        let g_corr_qrrho = t.gibbs_qrrho - e_low;
        W1Thermo {
            e_low,
            h_corr,
            g_corr,
            g_corr_qrrho,
            enthalpy: e_final + h_corr,
            gibbs: e_final + g_corr,
            gibbs_qrrho: e_final + g_corr_qrrho,
            freq: fd,
        }
    });

    Ok(W1Result {
        geometry,
        opt_label: opt_level.label,
        e_opt,
        basis_small: opts.basis_small.clone(),
        basis_large: opts.basis_large.clone(),
        cardinal_small: l_small,
        cardinal_large: l_large,
        e_hf_small,
        e_hf_large,
        e_hf_cbs,
        e_ccsd_corr_small,
        e_ccsd_corr_large,
        e_ccsd_corr_cbs,
        e_t_small,
        n_frozen,
        thermo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corr_n3_hand_computed() {
        let cbs = extrapolate_corr_n3(-1.0, -1.1, 3, 4);
        assert!((cbs - (-43.4 / 37.0)).abs() < 1e-15, "cbs = {cbs}");
        assert!(cbs < -1.1);
        assert_eq!(extrapolate_corr_n3(-2.5, -2.5, 2, 3), -2.5);
    }

    #[test]
    fn hf_karton_martin_hand_computed() {
        let f3 = 4.0 * (-9.0 * 3.0_f64.sqrt()).exp();
        let f4 = 5.0 * (-9.0 * 4.0_f64.sqrt()).exp();
        let expected = (-100.01 * f3 - (-100.0) * f4) / (f3 - f4);
        let cbs = extrapolate_hf_karton_martin(-100.0, -100.01, 3, 4);
        assert!((cbs - expected).abs() < 1e-12, "cbs = {cbs}");
        assert!(cbs < -100.01 && cbs > -100.0125, "cbs = {cbs}");
        assert_eq!(extrapolate_hf_karton_martin(-1.0, -1.0, 3, 4), -1.0);
    }

    #[test]
    fn cardinals_cover_the_bundled_families() {
        assert_eq!(basis_cardinal("cc-pVTZ"), Some(3));
        assert_eq!(basis_cardinal("cc-pvqz"), Some(4));
        assert_eq!(basis_cardinal("def2-tzvpp"), Some(3));
        assert_eq!(basis_cardinal("def2-qzvpp"), Some(4));
        assert_eq!(basis_cardinal("sto-3g"), None);
        assert_eq!(basis_cardinal("def2-mtzvpp"), None);
    }

    #[test]
    fn pair_validation_rejects_bad_pairs() {
        let opts = W1Options {
            basis_large: "no-such-basis".into(),
            ..W1Options::default()
        };
        let err = resolve_pair(&opts).unwrap_err();
        assert!(err.contains("unknown basis set"), "unexpected error: {err}");
        let opts = W1Options {
            basis_small: "sto-3g".into(),
            ..W1Options::default()
        };
        let err = resolve_pair(&opts).unwrap_err();
        assert!(err.contains("cardinal number"), "unexpected error: {err}");
        let opts = W1Options {
            basis_small: "cc-pvqz".into(),
            basis_large: "cc-pvtz".into(),
            ..W1Options::default()
        };
        let err = resolve_pair(&opts).unwrap_err();
        assert!(
            err.contains("increasing cardinal numbers"),
            "unexpected error: {err}"
        );
        assert_eq!(resolve_pair(&W1Options::default()).unwrap(), (3, 4));
    }
}
