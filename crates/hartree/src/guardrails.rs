use crate::job::{Job, Method};

const SMALL_BASES: &[&str] = &["sto-3g", "6-31g", "6-311g"];

fn non_dft_basis_family(lower: &str) -> Option<&'static str> {
    if lower.starts_with("6-31") {
        Some("Pople")
    } else if lower.contains("cc-p") {
        Some("correlation-consistent (cc-pVnZ)")
    } else {
        None
    }
}

fn is_composite(job: &Job) -> bool {
    let Method::Dft(spec) = &job.method else {
        return false;
    };
    crate::composite::COMPOSITES.iter().any(|c| {
        c.basis.eq_ignore_ascii_case(&job.basis)
            && crate::dft::FunctionalSpec::parse(c.functional)
                .is_ok_and(|f| f.name() == spec.name())
    })
}

pub fn assess_job(job: &Job) -> Vec<String> {
    let mut warnings = Vec::new();
    let basis_lower = job.basis.to_ascii_lowercase();
    let composite = is_composite(job);
    let dft_spec = match &job.method {
        Method::Dft(spec) => Some(spec),
        _ => None,
    };

    if let Some(family) = non_dft_basis_family(&basis_lower)
        && dft_spec.is_some()
        && !composite
    {
        warnings.push(format!(
            "warning: {} basis {} with DFT — the def2 family (e.g. def2-TZVP) is \
             recommended for production DFT",
            family, job.basis
        ));
    }

    if SMALL_BASES.contains(&basis_lower.as_str()) {
        warnings.push(format!(
            "warning: basis {} is minimal/unpolarized — too small for production \
             energies (use at least a polarized double-zeta, e.g. def2-SVP; \
             def2-TZVP or better for benchmarks)",
            job.basis
        ));
    }

    if let Some(spec) = dft_spec {
        if spec.exx_fraction() == 0.0 && spec.cam().is_none() && !spec.needs_tau() {
            warnings.push(format!(
                "note: {} is a pure GGA/LDA — GGAs systematically underestimate \
                 reaction barriers; for kinetics use a hybrid or RS hybrid such as \
                 wB97M-V (see --recommend barriers)",
                spec.name()
            ));
        }

        if job.options.dispersion.is_none()
            && spec.vv10().is_none()
            && let Some(key) = spec.d4_param_set()
        {
            warnings.push(format!(
                "warning: {name} run without a dispersion correction — the functional \
                 metadata recommends D4 (parameter set \"{key}\"); add the -d4 suffix \
                 (--method {name}-d4)",
                name = spec.name()
            ));
        }

        if spec.grid_sensitive() && job.options.grid_level < spec.recommended_grid_level() {
            warnings.push(format!(
                "warning: {} is grid-sensitive and grid level {} is below the \
                 recommended level {} — use --grid {}",
                spec.name(),
                job.options.grid_level,
                spec.recommended_grid_level(),
                spec.recommended_grid_level()
            ));
        }
    }

    if matches!(job.method, Method::Rhf | Method::Uhf | Method::Rohf) {
        warnings.push(
            "note: HF neglects electron correlation — HF energetics are qualitative \
             only (see --recommend general)"
                .to_string(),
        );
    }

    warnings
}

#[derive(Debug, Clone, Copy)]
pub struct Recommendation {
    pub task: &'static str,
    pub aliases: &'static [&'static str],
    pub level: &'static str,
    pub rationale: &'static str,
    pub invocation: &'static [&'static str],
    pub notes: &'static [&'static str],
}

pub const RECOMMENDATIONS: &[Recommendation] = &[
    Recommendation {
        task: "general",
        aliases: &["geometry", "default"],
        level: "r2scan-3c (geometry optimization + frequencies)",
        rationale: "r2scan-3c (Grimme et al., J. Chem. Phys. 154, 064103 (2021)) is the \
                    recommended general-purpose composite: r2scan/def2-mTZVPP + D4 + gCP \
                    gives near-hybrid accuracy for geometries, conformers, and reaction \
                    energies at a fraction of the cost.",
        invocation: &[
            "hartree molecule.xyz --method r2scan-3c --opt",
            "hartree optimized.xyz --method r2scan-3c --freq",
        ],
        notes: &[
            "run --freq at the optimized geometry to confirm a minimum and obtain \
                  RRHO/mRRHO thermochemistry",
        ],
    },
    Recommendation {
        task: "barriers",
        aliases: &["kinetics", "ts", "barrier"],
        level: "wb97m-v/def2-TZVPP single point on a r2scan-3c geometry",
        rationale: "GGAs (and to a lesser degree global hybrids) systematically \
                    underestimate reaction barriers; the range-separated hybrid \
                    wb97m-v with its VV10 nonlocal correlation is among the most \
                    accurate functionals for kinetics (Mardirossian & Head-Gordon 2016).",
        invocation: &["hartree molecule.xyz --method \"wb97m-v/def2-tzvpp // r2scan-3c\""],
        notes: &[
            "analytic gradients are unavailable for RS hybrids — optimize at the \
             r2scan-3c level and take the wb97m-v energy on that geometry (the \
             multi-level '//' workflow above does exactly this)",
        ],
    },
    Recommendation {
        task: "nci",
        aliases: &["noncovalent", "non-covalent"],
        level: "wb97m-v/def2-TZVPP single point on a r2scan-3c geometry",
        rationale: "for non-covalent interactions the VV10-containing RS hybrid \
                    wb97m-v is the benchmark-recommended choice; def2-TZVPP keeps the \
                    basis-set superposition error small.",
        invocation: &["hartree molecule.xyz --method \"wb97m-v/def2-tzvpp // r2scan-3c\""],
        notes: &[
            "analytic gradients are unavailable for RS hybrids — use the multi-level \
             '//' workflow (optimize at r2scan-3c, single-point at wb97m-v)",
            "for interaction energies of complexes consider the counterpoise driver \
             (--cp) at the single-point level",
        ],
    },
    Recommendation {
        task: "thermochemistry",
        aliases: &["thermo", "free-energy", "gibbs"],
        level: "wb97m-v/def2-TZVPP // r2scan-3c with --freq (composite free energy)",
        rationale: "the standard composite protocol: geometry and thermal corrections \
                    (RRHO/mRRHO) at the cheap r2scan-3c level, electronic energy from \
                    the accurate RS hybrid — G = E_high + (G_low - E_low).",
        invocation: &["hartree molecule.xyz --method \"wb97m-v/def2-tzvpp // r2scan-3c\" --freq"],
        notes: &[
            "the mRRHO (quasi-RRHO) free energy is the recommended value; set \
                  --symmetry-number for symmetric molecules",
        ],
    },
];

pub fn recommend(task: &str) -> Option<&'static Recommendation> {
    RECOMMENDATIONS.iter().find(|r| {
        r.task.eq_ignore_ascii_case(task) || r.aliases.iter().any(|a| a.eq_ignore_ascii_case(task))
    })
}

pub fn recommendation_tasks() -> Vec<&'static str> {
    RECOMMENDATIONS.iter().map(|r| r.task).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JobOptions, Molecule};

    fn h2() -> Molecule {
        Molecule::from_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap()
    }

    fn job(method: Method, basis: &str, options: JobOptions) -> Job {
        Job {
            molecule: h2(),
            basis: basis.into(),
            method,
            options,
        }
    }

    fn dft(name: &str) -> Method {
        Method::Dft(crate::dft::FunctionalSpec::parse(name).unwrap())
    }

    fn has(warnings: &[String], needle: &str) -> bool {
        warnings.iter().any(|w| w.contains(needle))
    }

    #[test]
    fn pople_and_cc_bases_warn_with_dft_only() {
        for basis in ["6-31g", "cc-pvdz"] {
            let w = assess_job(&job(dft("pbe"), basis, JobOptions::default()));
            assert!(has(&w, "def2 family"), "{basis}: {w:?}");
        }
        let w = assess_job(&job(Method::Mp2, "cc-pvdz", JobOptions::default()));
        assert!(!has(&w, "def2 family"), "{w:?}");
        let w = assess_job(&job(dft("pbe"), "def2-svp", JobOptions::default()));
        assert!(!has(&w, "def2 family"), "{w:?}");
    }

    #[test]
    fn small_basis_warns_and_polarized_does_not() {
        let w = assess_job(&job(Method::Rhf, "sto-3g", JobOptions::default()));
        assert!(has(&w, "minimal/unpolarized"), "{w:?}");
        let w = assess_job(&job(Method::Rhf, "def2-svp", JobOptions::default()));
        assert!(!has(&w, "minimal/unpolarized"), "{w:?}");
    }

    #[test]
    fn pure_gga_carries_barrier_note_hybrids_do_not() {
        let w = assess_job(&job(dft("pbe"), "def2-svp", JobOptions::default()));
        assert!(has(&w, "underestimate"), "{w:?}");
        for name in ["b3lyp", "wb97m-v", "tpss"] {
            let w = assess_job(&job(dft(name), "def2-svp", JobOptions::default()));
            assert!(!has(&w, "underestimate"), "{name}: {w:?}");
        }
    }

    #[test]
    fn missing_dispersion_warns_from_metadata() {
        let w = assess_job(&job(dft("pbe"), "def2-svp", JobOptions::default()));
        assert!(has(&w, "without a dispersion correction"), "{w:?}");
        assert!(has(&w, "\"pbe\""), "{w:?}");

        let with_d4 = JobOptions {
            dispersion: crate::disp::Dispersion::for_method(true, "pbe"),
            ..JobOptions::default()
        };
        let w = assess_job(&job(dft("pbe"), "def2-svp", with_d4));
        assert!(!has(&w, "without a dispersion correction"), "{w:?}");

        let w = assess_job(&job(dft("wb97m-v"), "def2-svp", JobOptions::default()));
        assert!(!has(&w, "without a dispersion correction"), "{w:?}");
    }

    #[test]
    fn hf_notes_missing_correlation_dft_does_not() {
        for m in [Method::Rhf, Method::Uhf, Method::Rohf] {
            let w = assess_job(&job(m, "def2-svp", JobOptions::default()));
            assert!(has(&w, "neglects electron correlation"), "{w:?}");
        }
        let w = assess_job(&job(dft("pbe0"), "def2-svp", JobOptions::default()));
        assert!(!has(&w, "neglects electron correlation"), "{w:?}");
    }

    #[test]
    fn coarse_grid_under_grid_sensitive_functional_warns() {
        let coarse = JobOptions {
            grid_level: 2,
            ..JobOptions::default()
        };
        let w = assess_job(&job(dft("m06-2x"), "def2-svp", coarse));
        assert!(has(&w, "grid-sensitive"), "{w:?}");
        let fine = JobOptions {
            grid_level: 4,
            ..JobOptions::default()
        };
        let w = assess_job(&job(dft("m06-2x"), "def2-svp", fine));
        assert!(!has(&w, "grid-sensitive"), "{w:?}");
        let w = assess_job(&job(
            dft("pbe0"),
            "def2-svp",
            JobOptions {
                grid_level: 1,
                ..JobOptions::default()
            },
        ));
        assert!(!has(&w, "grid-sensitive"), "{w:?}");
    }

    #[test]
    fn r2scan_3c_job_is_warning_clean() {
        let c = crate::composite::composite("r2scan-3c").unwrap();
        let options = JobOptions {
            grid_level: c.grid_level,
            dispersion: Some(c.dispersion),
            gcp: c.gcp,
            srb: c.srb,
            ..JobOptions::default()
        };
        let w = assess_job(&job(dft(c.functional), c.basis, options));
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn recommendation_table_is_consistent() {
        assert!(!RECOMMENDATIONS.is_empty());
        for r in RECOMMENDATIONS {
            assert!(!r.level.is_empty() && !r.rationale.is_empty());
            assert!(!r.invocation.is_empty(), "{}: no invocation", r.task);
            for inv in r.invocation {
                assert!(inv.starts_with("hartree "), "{}: {inv}", r.task);
            }
        }
        assert_eq!(recommend("GENERAL").unwrap().task, "general");
        assert_eq!(recommend("kinetics").unwrap().task, "barriers");
        assert_eq!(recommend("thermo").unwrap().task, "thermochemistry");
        assert!(recommend("nope").is_none());
        assert_eq!(
            recommendation_tasks(),
            vec!["general", "barriers", "nci", "thermochemistry"]
        );
        assert!(recommend("general").unwrap().level.contains("r2scan-3c"));
        assert!(recommend("barriers").unwrap().level.contains("wb97m-v"));
        assert!(recommend("nci").unwrap().level.contains("wb97m-v"));
    }
}
