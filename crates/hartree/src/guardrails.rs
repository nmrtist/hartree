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

    if job.options.transition_state {
        warnings.push(transition_state_note(job));
    }

    warnings
}

/// Cost/usage guidance for a `--ts` run: the finite-difference Hessian dominates,
/// and the right algorithm depends on how good the guess is. P-RFO refines a Hessian
/// and needs a guess already in the saddle basin; the dimer method is Hessian-free
/// during the search and forgiving of a rough guess (it still verifies with one
/// Hessian). Tailored to the selected algorithm so the advice is actionable.
fn transition_state_note(job: &Job) -> String {
    use crate::opt::ts::TsAlgorithm;
    let mut note = String::from(
        "note: a transition-state search finite-differences a Hessian \
         (~6·natom gradient evaluations per build, the dominant cost); ",
    );
    match job.options.ts_options.algorithm {
        TsAlgorithm::Prfo => note.push_str(
            "the default P-RFO follows a Hessian eigenvector uphill and needs a guess \
             already inside the saddle's quadratic basin — the dimer method \
             (--ts-algo dimer) is Hessian-free during the search and more forgiving of \
             a rough guess. --ts-recalc-hessian rebuilds a fresh Hessian every N \
             accepted steps (default 0 maintains it by a quasi-Newton update); raise it \
             on a rugged surface.",
        ),
        _ => note.push_str(
            "the dimer method is Hessian-free during the search and forgiving of a rough \
             guess, but still finite-differences one Hessian to verify the saddle \
             (--ts-recalc-hessian does not apply).",
        ),
    }
    note
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
            "heavy elements (Z > 36) use def2-ECP small-core potentials: hartree covers the \
             full def2-ECP range Rb–Rn (37–86, including the 4f lanthanides) with the def2-SVP \
             and def2-TZVP orbital basis — the 3c composites (def2-mTZVPP) and the other def2 \
             bases do not carry the heavy orbital set, so choose def2-SVP/def2-TZVP for a \
             heavy-element job. For all-electron scalar relativity use --x2c (energy-only \
             scalar-relativistic)",
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
#[path = "guardrails_tests.rs"]
mod tests;
