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
            "composite 3c methods carry their own basis and corrections (r2scan-3c = \
             r2scan/def2-mTZVPP + D4 + gCP); do not pass a separate --basis or -d4",
            "dispersion should essentially always be on — the 3c composites include D4 \
             already; for a plain functional add the -d4 suffix (or use a VV10 functional \
             such as wb97m-v / b97m-v)",
        ],
    },
    Recommendation {
        task: "optimization",
        aliases: &["geometry-only", "optimize", "geom", "opt"],
        level: "r2scan-3c geometry + frequencies (b97-3c or tpss-d4/def2-TZVP as alternatives)",
        rationale: "for structures, conformers, and harmonic frequencies a meta-GGA \
                    composite is the efficient sweet spot: r2scan-3c is the all-round \
                    first choice, b97-3c is a faster GGA-level alternative; both have \
                    analytic gradients so --opt/--freq run directly. Reserve barriers \
                    for a range-separated hybrid single point (see --recommend barriers).",
        invocation: &[
            "hartree molecule.xyz --method r2scan-3c --opt",
            "hartree molecule.xyz --method b97-3c --opt --freq",
            "hartree molecule.xyz --method tpss-d4 --basis def2-tzvp --opt",
        ],
        notes: &[
            "avoid pure GGAs (and especially BLYP) for barriers — they underestimate them; \
             a meta-GGA/composite is fine for geometries and frequencies",
            "pbeh-3c (DZ) is a cheap polarized-double-zeta hybrid for SIE-prone or polar \
             systems and TS pre-optimizations; b3lyp-3c (DZ) is tuned for IR spectra",
            "def2-SVP is the practical lower bound (structures only, with dispersion/gCP); \
             use def2-TZVP for production geometries and as the minimum for energies",
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
            "wb97x-v is a lighter range-separated VV10 alternative; avoid pure GGAs \
             (and global hybrids to a lesser degree) for kinetics",
            "for the highest accuracy on small systems, a CCSD(T) reference (see \
             --recommend reference) or the hartree-W1 protocol settles the barrier",
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
             (--cp <n_atoms_A>) at the single-point level to remove BSSE; def2-TZVPP \
             keeps the raw BSSE small to begin with",
            "for anions or systems where the diffuse tail matters, switch to a diffuse \
             basis such as def2-TZVPD (see --recommend anions)",
        ],
    },
    Recommendation {
        task: "anions",
        aliases: &["diffuse", "dipoles", "polarizability", "electron-affinity"],
        level: "wb97m-v/def2-TZVPD single point on a r2scan-3c geometry (diffuse basis)",
        rationale: "anions, dipole moments, and polarizabilities have a diffuse electron \
                    tail that a valence basis truncates; def2-TZVPD adds diffuse functions \
                    to def2-TZVP so the loosely bound density is described. wb97m-v remains \
                    an excellent choice for the electronic energy.",
        invocation: &[
            "hartree anion.xyz --method \"wb97m-v/def2-tzvpd // r2scan-3c\" --charge -1",
            "hartree molecule.xyz --method wb97m-v --basis def2-tzvpd",
        ],
        notes: &[
            "hartree bundles the Karlsruhe diffuse/property sets def2-SVPD, def2-TZVPD, and \
             def2-TZVPPD (no minimally augmented ma-def2 sets); def2-TZVPD is the balanced \
             default for anions and response properties",
            "diffuse functions raise the basis-set linear dependence and slow SCF \
             convergence — keep a tight integral grid and watch the SCF",
            "set the molecular charge with --charge (e.g. --charge -1 for a singly \
             charged anion)",
        ],
    },
    Recommendation {
        task: "reference",
        aliases: &[
            "benchmark",
            "gold-standard",
            "ccsdt",
            "coupled-cluster",
            "high-level",
        ],
        level: "CCSD(T)/large basis single point on a converged geometry (single-reference)",
        rationale: "for single-reference systems CCSD(T) is the gold standard; climb the \
                    MP2 -> CCSD -> CCSD(T) ladder in a converged basis (def2-TZVPP and up, \
                    extrapolating toward def2-QZVP/cc-pVQZ). CCSD(T) has no analytic gradient \
                    here, so run it as a single point on a DFT or HF geometry.",
        invocation: &[
            "hartree molecule.xyz --method \"ccsd(t)/def2-tzvpp // r2scan-3c\"",
            "hartree molecule.xyz --method ccsd --basis def2-qzvpp",
            "hartree molecule.xyz --protocol w1",
        ],
        notes: &[
            "the hartree-W1 protocol (--protocol w1) is a W1-STYLE staged workflow built \
             from hartree's native methods (B3LYP/cc-pVTZ geometry + frequencies, then \
             HF/CBS + CCSD/(T) extrapolation across cc-pVTZ/cc-pVQZ) — it is NOT literal \
             W1/W1-F12, and there is no explicitly correlated F12 path",
            "post-HF methods are energy-only (no --opt/--freq, no dispersion or solvation, \
             no ECP atoms) — always take them as single points on a separate geometry",
            "hartree has no DLPNO-CCSD(T) (the canonical CCSD(T) scales steeply, so keep the \
             system small) and no multireference WFT (CASSCF/CASPT2); use --recommend \
             multireference to screen for static correlation first",
        ],
    },
    Recommendation {
        task: "multireference",
        aliases: &["diagnostic", "fod", "static-correlation", "radicals"],
        level: "FOD fractional-occupation-number diagnostic (single-reference screen)",
        rationale: "diradicals, some transition-metal complexes, and bond-breaking \
                    transition states carry static correlation that single-reference \
                    CCSD(T)/DFT describe poorly. The FOD (fractional-occupation-number \
                    weighted density) is a cheap DFT-based indicator: a large integrated \
                    N_FOD flags multireference character.",
        invocation: &[
            "hartree molecule.xyz --method tpss --basis def2-tzvp --fod",
            "hartree molecule.xyz --method pbe0 --basis def2-tzvp --fod --fod-cube fod.cube",
        ],
        notes: &[
            "a small HOMO-LUMO gap or a slow/oscillating SCF are softer hints of static \
             correlation; --smear (Fermi smearing) can stabilize such an SCF",
            "FOD is SCF-level only (HF or DFT, conventional in-core backend; not post-HF, \
             --direct/--ri, ECP atoms, or double hybrids); --fod-cube writes the FOD density \
             for visualization",
            "hartree has no multireference WFT (no CASSCF/CASPT2/NEVPT2) — a high N_FOD means \
             the result needs an external multireference treatment, not a hartree single point",
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
            "the mRRHO (quasi-RRHO) free energy is the recommended value, especially with \
                  many low-frequency modes; set --symmetry-number for symmetric molecules",
            "for thermochemistry on a non-equilibrium or SQM (xtb) structure, use the \
             single-point-Hessian (--sph) frequencies instead of a full re-optimization",
            "double hybrids (b2plyp, revdsd-pbep86, pwpb95, wb97m(2)) give the best small-\
             molecule energies but need a large/QZ basis (e.g. def2-QZVPP) and are energy-\
             only (closed-shell, no --opt/--freq); the global hybrids pw6b95 and pbe0 are \
             solid reaction-energy choices",
            "for heavy elements (Z > 36) the def2 family supplies small-core ECPs \
             automatically (scalar relativity + reduced cost); reserve --x2c (all-electron, \
             energy-only scalar-relativistic) for cases needing explicit relativistic cores",
            "the Minnesota functional m06-2x is accurate but grid/basis sensitive — keep a \
             fine grid and add dispersion (m06-l is not available in hartree)",
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
            vec![
                "general",
                "optimization",
                "barriers",
                "nci",
                "anions",
                "reference",
                "multireference",
                "thermochemistry",
            ]
        );
        assert!(recommend("general").unwrap().level.contains("r2scan-3c"));
        assert!(recommend("barriers").unwrap().level.contains("wb97m-v"));
        assert!(recommend("nci").unwrap().level.contains("wb97m-v"));
    }

    #[test]
    fn task_names_and_aliases_are_unique_and_resolvable() {
        // No name/alias collides with another entry's name or aliases, and every
        // token resolves back to its own entry.
        let mut seen: Vec<&str> = Vec::new();
        for r in RECOMMENDATIONS {
            for tok in std::iter::once(&r.task).chain(r.aliases) {
                assert!(
                    !seen.contains(tok),
                    "duplicate task/alias token {tok:?} (in {})",
                    r.task
                );
                seen.push(tok);
                assert_eq!(recommend(tok).unwrap().task, r.task, "{tok} -> {}", r.task);
                // Lookups are case-insensitive.
                assert_eq!(recommend(&tok.to_uppercase()).unwrap().task, r.task);
            }
        }
    }

    #[test]
    fn new_tasks_resolve_and_stay_on_supported_methods() {
        // The four original tasks keep their core level strings.
        assert!(
            recommend("optimization")
                .unwrap()
                .level
                .contains("r2scan-3c")
        );
        assert_eq!(recommend("geometry-only").unwrap().task, "optimization");
        assert_eq!(recommend("opt").unwrap().task, "optimization");

        // Anions get a diffuse Karlsruhe basis that hartree actually bundles.
        let anions = recommend("anions").unwrap();
        assert_eq!(recommend("diffuse").unwrap().task, "anions");
        assert!(anions.level.contains("def2-TZVPD"));
        assert!(
            crate::basis::BasisSet::load("def2-tzvpd").is_ok(),
            "anions basis must be loadable"
        );

        // Reference ladder names a supported post-HF method and the W1 protocol.
        let reference = recommend("reference").unwrap();
        assert_eq!(recommend("ccsdt").unwrap().task, "reference");
        assert!(reference.level.contains("CCSD(T)"));
        assert!(
            reference
                .invocation
                .iter()
                .any(|i| i.contains("--protocol w1"))
        );

        // The multireference entry is the FOD diagnostic, not a (nonexistent) CASSCF.
        let mr = recommend("multireference").unwrap();
        assert_eq!(recommend("fod").unwrap().task, "multireference");
        assert!(mr.invocation.iter().any(|i| i.contains("--fod")));
        assert!(mr.notes.iter().any(|n| n.contains("CASSCF")));
    }
}
