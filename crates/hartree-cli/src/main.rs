use std::process::ExitCode;

use hartree::core::Molecule;
use hartree::core::units::{ANGSTROM_TO_BOHR, AU_DIPOLE_TO_DEBYE};
use hartree::dft::FunctionalSpec;
use hartree::disp::Dispersion;
use hartree::opt::OptResult;
use hartree::opt::ts::{TsResult, TsStatus};
use hartree::props::frequencies::FrequencyResult;
use hartree::props::population::PopulationAnalysis;
use hartree::props::thermo::ThermoResult;
use hartree::scf::{Reference, ScfResult, Smearing};
use hartree::{DftDiagnostics, Job, JobOptions, Method, PostHfResult};
use serde::Serialize;

mod periodic;
mod ts_flags;

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(true);
    }

    if periodic::is_periodic(&args) {
        return periodic::run_periodic_cli(&args);
    }

    let mut xyz_path: Option<String> = None;
    let mut basis = String::from("sto-3g");
    let mut basis_explicit = false;
    let mut method_str = String::from("rhf");
    let mut charge: i32 = 0;
    let mut multiplicity: u32 = 1;
    let mut do_opt = false;
    let mut do_ts = false;
    let mut ts_options = hartree::opt::ts::TsOptions::default();
    let mut ts_flag_seen = false;
    let mut irc_subflag_seen = false;
    let mut ts_product_path: Option<String> = None;
    let mut ts_use_neb = false;
    let mut ts_neb_images: Option<usize> = None;
    let mut ts_scan_points: Option<usize> = None;
    let mut ts_scan_coord: Option<hartree::CoordScanSpec> = None;
    let mut ts_output_path: Option<String> = None;
    let mut all_electron = false;
    let mut direct = false;
    let mut ri = false;
    let mut ri_mp2 = false;
    let mut cosx = false;
    let mut x2c = false;
    let mut do_properties = false;
    let mut do_freq = false;
    let mut do_sph = false;
    let mut conformers: Option<Option<String>> = None;
    let mut conformers_out: Option<String> = None;
    let mut rerank = false;
    let mut symmetry_number: u32 = 1;
    let mut qrrho_w0: f64 = hartree::props::thermo::QRRHO_W0_DEFAULT_CM1;
    let mut grid: Option<usize> = None;
    let mut solvent: Option<String> = None;
    let mut eps: Option<f64> = None;
    let mut smd: Option<String> = None;
    let mut alpb: Option<String> = None;
    let mut gbsa: Option<String> = None;
    let mut cosmo_file: Option<String> = None;
    let mut smear: Option<f64> = None;
    let mut method_explicit = false;
    let mut fod = false;
    let mut fod_cube: Option<String> = None;
    let mut cp_na: Option<usize> = None;
    let mut cp_charges: Option<(i32, i32)> = None;
    let mut cp_mults: Option<(u32, u32)> = None;
    let mut gcp_keyword: Option<String> = None;
    let mut no_method_warnings = false;
    let mut recommend_task: Option<String> = None;
    let mut protocol: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--basis" => {
                basis = take(&args, &mut i, "--basis")?;
                basis_explicit = true;
            }
            "--method" => {
                method_str = take(&args, &mut i, "--method")?.to_ascii_lowercase();
                method_explicit = true;
            }
            "--charge" => {
                charge = take(&args, &mut i, "--charge")?
                    .parse()
                    .map_err(|_| "--charge must be an integer".to_string())?
            }
            "--spin" | "--mult" => {
                multiplicity = take(&args, &mut i, "--spin")?
                    .parse()
                    .map_err(|_| "--spin must be a positive integer (2S+1)".to_string())?
            }
            "--opt" => do_opt = true,
            "--ts" => do_ts = true,
            "--ts-product" => {
                ts_product_path = Some(take(&args, &mut i, "--ts-product")?);
                ts_flag_seen = true;
            }
            "--ts-neb" => {
                ts_use_neb = true;
                ts_flag_seen = true;
            }
            "--ts-neb-images" => {
                let n: usize = take(&args, &mut i, "--ts-neb-images")?
                    .parse()
                    .map_err(|_| "--ts-neb-images must be a positive integer".to_string())?;
                if n == 0 {
                    return Err("--ts-neb-images must be a positive integer (>= 1)".into());
                }
                ts_neb_images = Some(n);
                ts_flag_seen = true;
            }
            "--ts-scan" => {
                let n: usize = take(&args, &mut i, "--ts-scan")?
                    .parse()
                    .map_err(|_| "--ts-scan must be an integer point count".to_string())?;
                if n < 3 {
                    return Err("--ts-scan needs at least 3 path points".into());
                }
                ts_scan_points = Some(n);
                ts_flag_seen = true;
            }
            "--ts-scan-coord" => {
                let spec = take(&args, &mut i, "--ts-scan-coord")?;
                ts_scan_coord = Some(ts_flags::parse_scan_coord(&spec)?);
                ts_flag_seen = true;
            }
            "--ts-output" => {
                let path = take(&args, &mut i, "--ts-output")?;
                if path.trim().is_empty() {
                    return Err("--ts-output needs a non-empty file path".into());
                }
                ts_output_path = Some(path);
                ts_flag_seen = true;
            }
            "--all-electron" => all_electron = true,
            "--direct" => direct = true,
            "--ri" => ri = true,
            "--ri-mp2" => ri_mp2 = true,
            "--cosx" => cosx = true,
            "--x2c" => x2c = true,
            "--properties" | "--props" => do_properties = true,
            "--freq" => do_freq = true,
            "--sph" => do_sph = true,
            "--conformers" => {
                if args.get(i + 1).map(|s| s.as_str()) == Some("crest") {
                    i += 1;
                    conformers = Some(Some("crest".to_string()));
                } else {
                    conformers = Some(None);
                }
            }
            "--conformers-out" => conformers_out = Some(take(&args, &mut i, "--conformers-out")?),
            "--rerank" => rerank = true,
            "--qrrho-w0" => {
                qrrho_w0 = take(&args, &mut i, "--qrrho-w0")?
                    .parse()
                    .map_err(|_| "--qrrho-w0 must be a frequency in cm^-1".to_string())?;
                if !qrrho_w0.is_finite() || qrrho_w0 <= 0.0 {
                    return Err("--qrrho-w0 must be a positive frequency in cm^-1".into());
                }
            }
            "--symmetry-number" => {
                symmetry_number = take(&args, &mut i, "--symmetry-number")?
                    .parse()
                    .map_err(|_| "--symmetry-number must be a positive integer".to_string())?
            }
            "--grid" => {
                grid = Some(
                    take(&args, &mut i, "--grid")?
                        .parse()
                        .map_err(|_| "--grid must be an integer 0..=4".to_string())?,
                )
            }
            "--solvent" => solvent = Some(take(&args, &mut i, "--solvent")?.to_ascii_lowercase()),
            "--smd" => smd = Some(take(&args, &mut i, "--smd")?.to_ascii_lowercase()),
            "--alpb" => alpb = Some(take(&args, &mut i, "--alpb")?.to_ascii_lowercase()),
            "--gbsa" => gbsa = Some(take(&args, &mut i, "--gbsa")?.to_ascii_lowercase()),
            "--cosmo-file" => cosmo_file = Some(take(&args, &mut i, "--cosmo-file")?),
            "--smear" => {
                let t: f64 = take(&args, &mut i, "--smear")?
                    .parse()
                    .map_err(|_| "--smear must be a temperature in kelvin".to_string())?;
                if !t.is_finite() || t <= 0.0 {
                    return Err("--smear temperature must be positive".into());
                }
                smear = Some(t);
            }
            "--cp" => {
                let n: usize = take(&args, &mut i, "--cp")?.parse().map_err(|_| {
                    "--cp must be the number of atoms in fragment A (the first n atoms \
                     of the XYZ, in input order)"
                        .to_string()
                })?;
                if n == 0 {
                    return Err("--cp needs at least one atom in fragment A".into());
                }
                cp_na = Some(n);
            }
            "--cp-charges" => {
                let v = take(&args, &mut i, "--cp-charges")?;
                cp_charges = Some(parse_pair(&v, "--cp-charges")?);
            }
            "--cp-mults" => {
                let v = take(&args, &mut i, "--cp-mults")?;
                let (a, b): (u32, u32) = parse_pair(&v, "--cp-mults")?;
                if a == 0 || b == 0 {
                    return Err("--cp-mults multiplicities must be >= 1".into());
                }
                cp_mults = Some((a, b));
            }
            "--gcp" => gcp_keyword = Some(take(&args, &mut i, "--gcp")?.to_ascii_lowercase()),
            "--protocol" => {
                protocol = Some(take(&args, &mut i, "--protocol")?.to_ascii_lowercase())
            }
            "--no-method-warnings" => no_method_warnings = true,
            "--recommend" => {
                recommend_task = Some(take(&args, &mut i, "--recommend")?.to_ascii_lowercase())
            }
            "--fod" => fod = true,
            "--fod-cube" => fod_cube = Some(take(&args, &mut i, "--fod-cube")?),
            "--eps" => {
                eps = Some(
                    take(&args, &mut i, "--eps")?
                        .parse()
                        .map_err(|_| "--eps must be a number (dielectric constant)".to_string())?,
                )
            }
            "--ts-irc" => {
                ts_options.confirm_irc = true;
                ts_flag_seen = true;
            }
            f if ts_flags::TS_VALUE_FLAGS.contains(&f) => {
                let flag = args[i].clone();
                let value = take(&args, &mut i, &flag)?;
                ts_flags::apply_ts_option(&mut ts_options, &flag, &value)?;
                ts_flag_seen = true;
                // The IRC sub-flags only take effect once `--ts-irc` enables the trace.
                irc_subflag_seen |= flag.starts_with("--ts-irc-");
            }
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            path => xyz_path = Some(path.to_string()),
        }
        i += 1;
    }

    if ts_product_path.is_some() && !do_ts {
        return Err(
            "--ts-product gives the product endpoint for a two-endpoint transition-state \
             search; add --ts to request the search"
                .into(),
        );
    }
    if ts_use_neb && ts_product_path.is_none() {
        return Err(
            "--ts-neb relaxes a band between two endpoints; provide the product geometry \
             with --ts-product <file.xyz>"
                .into(),
        );
    }
    if ts_neb_images.is_some() && !ts_use_neb {
        eprintln!("warning: --ts-neb-images is ignored without --ts-neb");
    }
    if ts_scan_points.is_some() && ts_product_path.is_none() {
        return Err(
            "--ts-scan places the guess at the energy peak between two endpoints; provide \
             the product with --ts-product <file.xyz>"
                .into(),
        );
    }
    if ts_scan_points.is_some() && ts_use_neb {
        eprintln!("warning: --ts-scan is ignored with --ts-neb (the band already finds the peak)");
    }
    if ts_scan_coord.is_some() && ts_product_path.is_some() {
        return Err(
            "--ts-scan-coord is a single-ended distinguished-coordinate scan; it is mutually \
             exclusive with the two-endpoint --ts-product route"
                .into(),
        );
    }
    if ts_flag_seen && !do_ts {
        eprintln!("warning: --ts-* flags are ignored without --ts");
    } else if irc_subflag_seen && !ts_options.confirm_irc {
        eprintln!("warning: --ts-irc-* flags are ignored without --ts-irc");
    }

    if let Some(task) = &recommend_task {
        return report_recommendation(task);
    }

    let xyz_path = xyz_path.ok_or("missing XYZ file argument (try --help)")?;
    let xyz = std::fs::read_to_string(&xyz_path).map_err(|e| format!("reading {xyz_path}: {e}"))?;
    let molecule = Molecule::from_xyz(&xyz)
        .map_err(|e| e.to_string())?
        .with_charge(charge)
        .with_multiplicity(multiplicity);
    molecule.validate().map_err(|e| e.to_string())?;

    // Two-endpoint transition-state input: the main XYZ is the reactant, --ts-product
    // is the product. A reaction conserves charge and multiplicity, so the product
    // takes the reactant's. The guess shares the reactant's composition.
    let ts_guess = if let Some(path) = &ts_product_path {
        let pxyz = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
        let product = Molecule::from_xyz(&pxyz)
            .map_err(|e| e.to_string())?
            .with_charge(charge)
            .with_multiplicity(multiplicity);
        product.validate().map_err(|e| e.to_string())?;
        let mut input = hartree::TsGuessInput::new(product);
        input.use_neb = ts_use_neb;
        // Permute the product onto the reactant's atom order before building the band, the
        // same correspondence the IDPP route applies, so a product whose atoms are listed
        // in a different order than the reactant is handled identically on both routes.
        input.neb_options.map_atoms = true;
        if let Some(n) = ts_neb_images {
            input.neb_options.n_images = n;
        }
        // The energy-peaked scan applies only to the IDPP route; the band finds its own
        // peak, so it is left unset under --ts-neb.
        if !ts_use_neb {
            input.scan_points = ts_scan_points;
        }
        Some(input)
    } else {
        None
    };

    if do_sph && !do_freq {
        return Err("--sph modifies --freq; pass --freq as well".into());
    }

    let solvent_eps = match (&solvent, eps) {
        (Some(_), Some(_)) => {
            return Err(
                "--solvent and --eps are mutually exclusive (a named solvent already fixes ε)"
                    .into(),
            );
        }
        (Some(name), None) => Some(hartree::solv::solvent_epsilon(name).ok_or_else(|| {
            let names: Vec<&str> = hartree::solv::SOLVENTS.iter().map(|(n, _)| *n).collect();
            format!(
                "unknown solvent {name:?} (available: {}; or give --eps <value> directly)",
                names.join(", ")
            )
        })?),
        (None, Some(e)) => Some(e),
        (None, None) => None,
    };

    if smd.is_some() && (solvent.is_some() || eps.is_some()) {
        return Err(
            "--smd and --solvent/--eps are mutually exclusive (SMD fixes both the dielectric constant and the non-electrostatic parameterization)"
                .into(),
        );
    }
    if let Some(name) = &smd
        && hartree::solv::smd_solvent(name).is_none()
    {
        let names: Vec<&str> = hartree::solv::SMD_SOLVENTS.iter().map(|s| s.name).collect();
        return Err(format!(
            "unknown SMD solvent {name:?} (available: {})",
            names.join(", ")
        ));
    }

    let n_solv = [
        solvent.is_some() || eps.is_some(),
        smd.is_some(),
        alpb.is_some(),
        gbsa.is_some(),
        cosmo_file.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count();
    if n_solv > 1 {
        return Err(
            "the solvation options are mutually exclusive: choose at most one of \
             --solvent/--eps (C-PCM), --smd, --alpb, --gbsa, or --cosmo-file"
                .into(),
        );
    }
    if let Some(name) = &alpb
        && hartree::solv::alpb_solvent(name).is_none()
    {
        return Err(format!(
            "unknown ALPB solvent {name:?} (available: {})",
            hartree::solv::alpb_solvent_names().join(", ")
        ));
    }
    if let Some(name) = &gbsa
        && hartree::solv::gbsa_solvent(name).is_none()
    {
        return Err(format!(
            "unknown GBSA solvent {name:?} (available: {})",
            hartree::solv::gbsa_solvent_names().join(", ")
        ));
    }

    if let Some(name) = &protocol {
        if !matches!(name.as_str(), "w1" | "hartree-w1") {
            return Err(format!(
                "unknown protocol {name:?} (available: w1 — the hartree-W1 composite \
                 thermochemistry protocol)"
            ));
        }
        if method_explicit || basis_explicit {
            return Err(
                "--protocol w1 defines its own methods and bases (B3LYP/cc-pVTZ geometry \
                 + frequencies; CCSD(T)/cc-pVTZ; CCSD/cc-pVQZ; CBS extrapolation); drop \
                 --method/--basis"
                    .into(),
            );
        }
        if do_opt || do_freq {
            return Err(
                "--protocol w1 already optimizes the geometry and computes frequencies as \
                 protocol stages; drop --opt/--freq"
                    .into(),
            );
        }
        for (on, what) in [
            (direct, "--direct"),
            (ri, "--ri"),
            (ri_mp2, "--ri-mp2"),
            (cosx, "--cosx"),
            (x2c, "--x2c"),
            (fod, "--fod"),
            (smear.is_some(), "--smear"),
            (do_properties, "--properties"),
            (do_sph, "--sph"),
            (conformers.is_some(), "--conformers"),
            (rerank, "--rerank"),
            (cosmo_file.is_some(), "--cosmo-file"),
            (gcp_keyword.is_some(), "--gcp"),
            (cp_na.is_some(), "--cp"),
            (grid.is_some(), "--grid"),
            (solvent_eps.is_some(), "--solvent/--eps"),
            (smd.is_some(), "--smd"),
            (alpb.is_some(), "--alpb"),
            (gbsa.is_some(), "--gbsa"),
        ] {
            if on {
                return Err(format!(
                    "{what} is not supported with --protocol w1 (the protocol is a fixed \
                     gas-phase staged workflow); run the stages as separate jobs instead"
                ));
            }
        }
        let w1_opts = hartree::w1::W1Options {
            symmetry_number,
            qrrho_w0_cm1: qrrho_w0,
            all_electron,
            ..hartree::w1::W1Options::default()
        };
        let res = hartree::w1::run_w1(&molecule, &w1_opts)?;
        report_w1(&molecule, &w1_opts, &res);
        return Ok(true);
    }

    let ml_spec = hartree::multilevel::parse_spec(&method_str, multiplicity)?;
    if rerank && ml_spec.is_none() {
        return Err(
            "--rerank requires --conformers and a multi-level '//' --method spec \
             (e.g. --method \"r2scan-3c // hf/sto-3g\")"
                .into(),
        );
    }
    if let Some(spec) = ml_spec {
        if basis_explicit {
            return Err(
                "--basis conflicts with a multi-level '//' spec: each level carries its \
                 basis inside the spec itself (method/basis, or a composite keyword)"
                    .into(),
            );
        }
        if do_opt {
            return Err(
                "--opt is redundant with a multi-level '//' spec: the OPT level (right of \
                 '//') already optimizes the geometry; drop --opt"
                    .into(),
            );
        }
        for (on, what) in [
            (direct, "--direct"),
            (ri, "--ri"),
            (cosx, "--cosx"),
            (x2c, "--x2c"),
            (fod, "--fod"),
            (smear.is_some(), "--smear"),
            (do_properties, "--properties"),
            (do_sph, "--sph"),
            (cosmo_file.is_some(), "--cosmo-file"),
            (gcp_keyword.is_some(), "--gcp"),
            (cp_na.is_some(), "--cp"),
        ] {
            if on {
                return Err(format!(
                    "{what} is not supported with a multi-level '//' spec; \
                     run the stages as separate single-level jobs instead"
                ));
            }
        }
        if let Some(g) = grid
            && g > 4
        {
            return Err("--grid must be an integer 0..=4".into());
        }
        let ml_opts = hartree::multilevel::MultiLevelOptions {
            compute_frequencies: do_freq,
            symmetry_number,
            qrrho_w0_cm1: qrrho_w0,
            grid_override: grid,
            all_electron,
            ri_mp2,
            solvent_eps,
            smd: smd.clone(),
            alpb: alpb.clone(),
            gbsa: gbsa.clone(),
        };
        if let Some(engine) = &conformers {
            if !rerank {
                return Err(
                    "--conformers with a multi-level '//' spec needs --rerank (the CENSO-lite \
                     re-ranking hook); without it the spec is ambiguous for the screening step"
                        .into(),
                );
            }
            return run_multilevel_conformers(
                &molecule,
                &spec,
                &ml_opts,
                engine.as_deref(),
                alpb.clone(),
                conformers_out.as_deref(),
            );
        }
        let res = hartree::multilevel::run_multilevel(&molecule, &spec, &ml_opts)?;
        report_multilevel(&molecule, &spec, &res);
        if !no_method_warnings {
            let mut warnings = res.sp.method_warnings.clone();
            for w in &res.opt.method_warnings {
                if !warnings.contains(w) {
                    warnings.push(w.clone());
                }
            }
            print_method_warnings(&warnings);
        }
        return Ok(true);
    }

    if let Some(engine) = &conformers {
        return run_conformers(
            &molecule,
            &method_str,
            &basis,
            engine.as_deref(),
            alpb.clone(),
            conformers_out.as_deref(),
        );
    }

    if let Some(xtb_method) = hartree::ext::xtb::XtbMethod::from_keyword(&method_str) {
        return run_xtb(&molecule, &method_str, xtb_method, do_opt, alpb.clone());
    }

    if fod_cube.is_some() && !fod {
        return Err("--fod-cube requires --fod".into());
    }
    if fod {
        if method_explicit {
            if method_str != "tpss" {
                eprintln!(
                    "warning: FOD analysis is calibrated at TPSS/def2-TZVP (T_el = 5000 K); \
                     using --method {method_str} — interpret N_FOD thresholds with care"
                );
            }
        } else {
            method_str = String::from("tpss");
        }
        if basis_explicit {
            if !basis.eq_ignore_ascii_case("def2-tzvp") {
                eprintln!(
                    "warning: FOD analysis is calibrated at TPSS/def2-TZVP; using --basis {basis}"
                );
            }
        } else {
            basis = String::from("def2-tzvp");
        }
    }

    {
        let gcp_is_dftc = gcp_keyword
            .as_deref()
            .is_some_and(|k| k.replace('-', "") == "dftc");
        if method_str.replace('-', "").contains("dftc") || gcp_is_dftc {
            return Err(
                "DFT-C is not implemented: its parameter tables (the 1296 pairwise \
                 c_AB/alpha_AB/beta_AB values for H\u{2013}Kr plus the fitted atomic and \
                 gCP/def2-SVPD parameters) are published only in the AIP supplementary \
                 material of J. Chem. Phys. 146, 234105 (2017), which hartree cannot \
                 vendor. Obtain the SI zip in a browser (https://doi.org/10.1063/1.4986962) \
                 or email the corresponding author. The B97M-V/def2-SVPD NCI protocol \
                 runs without DFT-C: --method b97m-v --basis def2-svpd."
                    .into(),
            );
        }
    }

    let (base_method_str, want_disp) = if let Some(base) = method_str.strip_suffix("-d3") {
        (base.to_string(), Some(false))
    } else if let Some(base) = method_str.strip_suffix("-d4") {
        (base.to_string(), Some(true))
    } else {
        (method_str.clone(), None)
    };

    let composite = hartree::composite::composite(&base_method_str);
    if let Some(c) = composite {
        if want_disp.is_some() {
            return Err(format!(
                "{} defines its own {} and short-range corrections; a -d3/-d4 suffix is not                  allowed",
                c.keyword,
                c.dispersion.label()
            ));
        }
        if basis_explicit && !basis.eq_ignore_ascii_case(c.basis) {
            return Err(format!(
                "{} implies the {} basis; --basis {basis} conflicts \
                 (omit --basis or pass {})",
                c.keyword, c.basis_label, c.basis
            ));
        }
        basis = String::from(c.basis);
    }

    let method = if let Some(c) = composite {
        Method::Dft(FunctionalSpec::parse(c.functional).map_err(|e| e.to_string())?)
    } else {
        match base_method_str.as_str() {
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
                Err(hartree::dft::DftError::UnknownFunctional(_)) => {
                    return Err(format!(
                        "unknown method {other:?} (expected hf, rhf, uhf, rohf, mp2, ccsd, ccsd(t), \
                     or a functional like svwn/pbe/blyp/b3lyp/pbe0/tpss/r2scan, \
                     optionally with a -d3 or -d4 suffix)"
                    ));
                }
                Err(err) => return Err(err.to_string()),
            },
        }
    };

    let dispersion = if let Some(c) = composite {
        Some(c.dispersion)
    } else if let Some(d4) = want_disp {
        let (suffix, model) = if d4 { ("-d4", "D4") } else { ("-d3", "D3(BJ)") };
        if matches!(method, Method::Mp2 | Method::Ccsd | Method::CcsdT) {
            return Err(format!(
                "{suffix} is not supported for post-HF methods ({base_method_str}); \
                 the {model} correction applies to HF and DFT functionals"
            ));
        }
        let param_key = match &method {
            Method::Rhf | Method::Uhf | Method::Rohf => "hf".to_string(),
            Method::Dft(spec) => spec
                .d4_param_set()
                .map(str::to_string)
                .unwrap_or_else(|| spec.name().to_string()),
            _ => unreachable!(),
        };
        match Dispersion::for_method(d4, &param_key) {
            Some(d) => Some(d),
            None => {
                return Err(format!(
                    "no {model} parametrization exists for {param_key} \
                     (supported: pbe, blyp, b3lyp, b3lyp5, pbe0, tpss, r2scan, hf; \n                     D4 additionally: b2plyp, revdsd-pbep86, pwpb95)"
                ));
            }
        }
    } else {
        None
    };

    let standalone_gcp = match &gcp_keyword {
        Some(k) => {
            if composite.is_some() {
                return Err(format!(
                    "{} defines its own gCP treatment; --gcp is not allowed with a composite \
                     method",
                    base_method_str
                ));
            }
            Some(hartree::disp::GcpParams::by_keyword(k).ok_or_else(|| {
                format!(
                    "unknown gCP parameter set {k:?} (available: r2scan-3c / def2-mtzvpp, \
                     dft/sv(p) / b3lyp-3c, pbeh-3c / def2-msvp; only the sets vendored from \
                     the reference implementation mctc-gcp are bundled)"
                )
            })?)
        }
        None => None,
    };

    if grid.is_some() && !matches!(method, Method::Dft(_)) {
        return Err("--grid requires a functional method (e.g. --method pbe)".into());
    }
    let metadata_grid = match &method {
        Method::Dft(spec) if grid.is_none() && spec.grid_sensitive() => {
            let level = spec.recommended_grid_level();
            eprintln!(
                "note: {} is grid-sensitive; defaulting to grid level {level} \
                 (override with --grid)",
                spec.name()
            );
            Some(level)
        }
        _ => None,
    };
    let grid_level = grid
        .or(metadata_grid)
        .unwrap_or(composite.map_or(3, |c| c.grid_level));
    if grid_level > 4 {
        return Err("--grid must be an integer 0..=4".into());
    }

    if do_opt && (solvent_eps.is_some() || smd.is_some()) {
        eprintln!("note: solvated geometry optimization uses finite-difference gradients (slower)");
    }

    if do_opt && do_freq {
        eprintln!(
            "note: --freq is not computed in the same run as --opt for single-level jobs; \
             re-run --freq at the printed optimized geometry (multi-level '//' specs and \
             --protocol w1 combine the two)"
        );
    }

    if !method_explicit && !fod && !no_method_warnings {
        eprintln!(
            "note: no --method given; defaulting to RHF (the historic default). For \
             production work r2scan-3c is recommended — see `hartree --recommend general` \
             (suppress this note with --no-method-warnings)"
        );
    }

    let job = Job {
        molecule: molecule.clone(),
        basis: basis.clone(),
        method,
        options: JobOptions {
            all_electron,
            direct,
            ri,
            compute_properties: do_properties,
            compute_frequencies: do_freq,
            single_point_hessian: do_sph,
            optimize_geometry: do_opt,
            transition_state: do_ts,
            symmetry_number,
            qrrho_w0_cm1: qrrho_w0,
            grid_level,
            dispersion,
            solvent_eps,
            smd: smd.clone(),
            alpb: alpb.clone(),
            gbsa: gbsa.clone(),
            cosmo_file: cosmo_file.as_ref().map(Into::into),
            gcp: composite.and_then(|c| c.gcp).or(standalone_gcp),
            srb: composite.and_then(|c| c.srb),
            smearing: smear.map(|temperature_k| Smearing::Fermi { temperature_k }),
            fod,
            fod_cube: fod_cube.as_ref().map(Into::into),
            ri_mp2,
            cosx,
            x2c,
            ts_options,
            ts_guess,
            ts_coord_scan: ts_scan_coord.clone(),
            // The CLI runs one job per process, so it leaves rayon on its global
            // pool and sets no in-process memory budget; both knobs exist for
            // library/embedding callers that drive several jobs in one process.
            n_threads: None,
            mem_budget_bytes: None,
        },
    };

    if let Some(na) = cp_na {
        if na >= molecule.len() {
            return Err(format!(
                "--cp {na}: fragment A must leave at least one atom for fragment B \
                 (the complex has {} atoms)",
                molecule.len()
            ));
        }
        let (qa, qb) = cp_charges.unwrap_or((0, 0));
        let (ma, mb) = cp_mults.unwrap_or((1, 1));
        let frags = hartree::CpFragments {
            fragment_a: (0..na).collect(),
            charge_a: qa,
            multiplicity_a: ma,
            charge_b: qb,
            multiplicity_b: mb,
        };
        let cp = hartree::counterpoise(&job, &frags)?;
        println!(
            "hartree -- counterpoise (Boys-Bernardi) -- {} / {}",
            method_str.to_ascii_uppercase(),
            basis
        );
        println!(
            "complex: {} atoms (charge {}, mult {});  fragment A: atoms 1-{na} \
             (charge {qa}, mult {ma});  fragment B: atoms {}-{} (charge {qb}, mult {mb})",
            molecule.len(),
            molecule.charge,
            molecule.multiplicity,
            na + 1,
            molecule.len()
        );
        println!();
        println!("  E_AB^(AB) (complex)      {:>20.12} Eh", cp.e_complex);
        println!(
            "  E_A^(AB)  (A + ghost-B)  {:>20.12} Eh",
            cp.e_a_in_dimer_basis
        );
        println!(
            "  E_B^(AB)  (ghost-A + B)  {:>20.12} Eh",
            cp.e_b_in_dimer_basis
        );
        println!("  E_A^(A)   (A alone)      {:>20.12} Eh", cp.e_a);
        println!("  E_B^(B)   (B alone)      {:>20.12} Eh", cp.e_b);
        println!();
        let to_kcal = 627.509474063;
        println!(
            "  dE_int (uncorrected)     {:>20.12} Eh  ({:>10.4} kcal/mol)",
            cp.interaction_uncorrected(),
            cp.interaction_uncorrected() * to_kcal
        );
        println!(
            "  dE_int^CP (corrected)    {:>20.12} Eh  ({:>10.4} kcal/mol)",
            cp.interaction_cp(),
            cp.interaction_cp() * to_kcal
        );
        println!(
            "  delta_BSSE               {:>20.12} Eh  ({:>10.4} kcal/mol)",
            cp.bsse(),
            cp.bsse() * to_kcal
        );
        return Ok(true);
    }
    if cp_charges.is_some() || cp_mults.is_some() {
        return Err("--cp-charges/--cp-mults require --cp <nA>".into());
    }

    let result = job.run()?;

    let ecp_atoms = hartree::BasisSet::load(&basis)
        .map(|set| hartree::ecp_summary(&molecule, &set))
        .unwrap_or_default();

    if let Some(opt) = &result.optimized_geometry {
        report_optimization(&molecule, &basis, &method_str, &ecp_atoms, opt);
        if !no_method_warnings {
            print_method_warnings(&result.method_warnings);
        }
        return Ok(opt.converged);
    }

    if let Some(ts) = &result.transition_state {
        if ts_product_path.is_some() {
            let route = if ts_use_neb {
                "climbing-image NEB band"
            } else if ts_scan_points.is_some() {
                "an energy-peaked IDPP path scan"
            } else {
                "IDPP interpolation"
            };
            println!(
                "note: two-endpoint search -- guess built by {route} between the reactant \
                 and --ts-product, then refined\n"
            );
            print_mapping_confidence(result.mapping_confidence.as_ref());
        } else if ts_scan_coord.is_some() {
            println!(
                "note: single-ended search -- guess built by a distinguished-coordinate scan \
                 (one internal coordinate driven, the rest relaxed), then refined\n"
            );
        }
        report_transition_state(&molecule, &basis, &method_str, &ecp_atoms, ts);
        if let Some(path) = &ts_output_path {
            write_ts_json(
                path,
                &molecule,
                &basis,
                &method_str,
                ts,
                result.mapping_confidence.as_ref(),
            );
        }
        if !no_method_warnings {
            print_method_warnings(&result.method_warnings);
        }
        return Ok(ts.converged());
    }

    report(
        &molecule,
        &basis,
        &method_str,
        &ecp_atoms,
        &result.scf,
        result.ri.as_ref(),
        result.cosx.as_ref(),
    );
    if x2c {
        println!();
        println!("X2C-1e scalar-relativistic Hamiltonian: active");
        println!(
            "  spin-free one-electron X2C (c = 137.035999084 a.u., CODATA 2018); \
             two-electron integrals nonrelativistic"
        );
        if do_properties {
            println!(
                "  note: property operators are NOT picture-change corrected (X2C-1e caveat); \
                 dipole moments and populations use the nonrelativistic operators"
            );
        }
    }
    if let Some(dft) = &result.dft {
        report_dft(dft, &result.scf);
    }
    if let (Some(e_nl), None) = (result.vv10_energy, &result.double_hybrid) {
        println!(
            "  E_nl (VV10, non-SC) {:>20.12} Eh   (post-SCF; included in total below)",
            e_nl
        );
        println!(
            "  total energy + E_nl {:>20.12} Eh",
            result.scf.energy + e_nl
        );
    }
    if let Some((occ_a, occ_b)) = &result.scf.occupations {
        let frac = |occ: &[f64]| occ.iter().filter(|&&f| f > 1e-6 && f < 1.0 - 1e-6).count();
        let ts = result.scf.electronic_entropy.unwrap_or(0.0);
        let free = result.scf.free_energy.unwrap_or(result.scf.energy);
        let t_el = smear
            .or(result.fod.as_ref().map(|f| f.temperature_k))
            .expect("occupations imply a smearing temperature");
        println!();
        println!("Fermi smearing (T = {:.1} K):", t_el);
        println!(
            "  fractionally occupied   {:>10} α  /  {:>5} β  (of {} orbitals each)",
            frac(occ_a),
            frac(occ_b),
            occ_a.len()
        );
        println!("  T*S_el              {:>20.12} Eh", ts);
        println!("  free energy F=E-T*S {:>20.12} Eh", free);
    }
    if let Some(f) = &result.fod {
        println!();
        println!("FOD analysis (Grimme):");
        println!(
            "  T_el                {:>20.1} K   (T = 5000 K + 20000 K * a_x)",
            f.temperature_k
        );
        println!(
            "  N_FOD               {:>20.6}     ({:.6} alpha, {:.6} beta)",
            f.n_fod, f.n_fod_alpha, f.n_fod_beta
        );
        if let Some(path) = &fod_cube {
            println!("  rho_FOD cube written to {path}");
        }
        if f.n_fod >= 1.0 {
            println!(
                "  WARNING: N_FOD = {:.3} >= 1.0 — strong static correlation \
                 (multireference character); single-reference results (DFT, MP2, CCSD(T)) \
                 may be unreliable for this system",
                f.n_fod
            );
        } else if f.n_fod >= 0.5 {
            println!(
                "  note: N_FOD = {:.3} is in the borderline 0.5–1.0 range — mild static \
                 correlation; inspect rho_FOD and frontier occupations",
                f.n_fod
            );
        } else {
            println!(
                "  N_FOD < 0.5: no significant static correlation indicated \
                 (single-reference methods are appropriate)"
            );
        }
    }
    if let Some(smd) = &result.smd {
        const KCAL: f64 = 627.509_451;
        println!();
        println!("SMD solvation (Marenich, Cramer, Truhlar 2009):");
        println!("  solvent             {:>20}", smd.solvent);
        println!("  dielectric ε        {:>20.4}", smd.epsilon);
        println!("  E (gas, this geom)  {:>20.12} Eh", smd.e_gas);
        println!(
            "  E (solution)        {:>20.12} Eh   (SCF total in solvent)",
            smd.e_solution
        );
        println!(
            "  ΔG_EP               {:>20.12} Eh   ({:>9.3} kcal/mol)",
            smd.g_ep,
            smd.g_ep * KCAL
        );
        println!(
            "  G_CDS               {:>20.12} Eh   ({:>9.3} kcal/mol)",
            smd.g_cds,
            smd.g_cds * KCAL
        );
        println!(
            "  ΔG_solv             {:>20.12} Eh   ({:>9.3} kcal/mol)",
            smd.dg_solv,
            smd.dg_solv * KCAL
        );
        println!(
            "  standard state: 298 K, fixed 1 mol/L in gas and solution (no concentration term)"
        );
    } else if let Some(g) = &result.gbsa {
        const KCAL: f64 = 627.509_451;
        println!();
        println!(
            "{} solvation (xtb GFN2 parameters; post-SCF on Mulliken charges):",
            g.model
        );
        println!("  solvent             {:>20}", g.solvent);
        println!("  dielectric ε        {:>20.4}", g.epsilon);
        println!(
            "  G_born              {:>20.12} Eh   ({:>9.3} kcal/mol)",
            g.g_born,
            g.g_born * KCAL
        );
        if g.g_hb != 0.0 {
            println!(
                "  G_hb                {:>20.12} Eh   ({:>9.3} kcal/mol)",
                g.g_hb,
                g.g_hb * KCAL
            );
        }
        println!(
            "  G_sasa              {:>20.12} Eh   ({:>9.3} kcal/mol)",
            g.g_sasa,
            g.g_sasa * KCAL
        );
        println!(
            "  G_shift             {:>20.12} Eh   ({:>9.3} kcal/mol)",
            g.g_shift,
            g.g_shift * KCAL
        );
        println!(
            "  ΔG_solv             {:>20.12} Eh   ({:>9.3} kcal/mol)",
            g.g_solv,
            g.g_solv * KCAL
        );
        println!(
            "  caveat: GFN2-fit parameters applied to ab-initio Mulliken charges (provenance \
             documented)"
        );
    } else if let Some(e_solv) = result.scf.solvation_energy {
        println!();
        if let Some(path) = &cosmo_file {
            println!("C-PCM (ideal conductor, ε = ∞) for COSMO-RS export:");
            println!(
                "  E_diel              {:>20.12} Eh   (included in total energy)",
                e_solv
            );
            println!("  .cosmo file written to {path}");
        } else {
            println!("C-PCM solvation (electrostatics only):");
            if let Some(name) = &solvent {
                println!("  solvent             {:>20}", name);
            }
            println!("  dielectric ε        {:>20.4}", solvent_eps.unwrap());
            println!(
                "  E_solv              {:>20.12} Eh   (included in total energy)",
                e_solv
            );
        }
    }
    if let Some(c) = composite {
        let e_disp = result
            .dispersion_energy
            .expect("every composite carries a dispersion correction");
        println!();
        println!("{} composite:", c.keyword);
        println!(
            "  {:<20}{:>20.12} Eh",
            format!("E_SCF ({})", c.functional),
            result.scf.energy
        );
        println!(
            "  {:<20}{:>20.12} Eh",
            format!("E_{}", c.disp_label),
            e_disp
        );
        if let (Some(e_gcp), Some(gcp)) = (result.gcp_energy, c.gcp) {
            println!(
                "  {:<20}{:>20.12} Eh",
                format!("E_gCP ({})", gcp.label),
                e_gcp
            );
        }
        if let Some(e_srb) = result.srb_energy {
            println!("  E_SRB               {:>20.12} Eh", e_srb);
        }
        println!(
            "  composite total     {:>20.12} Eh",
            result.scf.energy
                + e_disp
                + result.gcp_energy.unwrap_or(0.0)
                + result.srb_energy.unwrap_or(0.0)
        );
    } else if let (Some(e_disp), None) = (result.dispersion_energy, &result.double_hybrid) {
        let label = dispersion
            .expect("dispersion energy implies a model")
            .label();
        println!();
        println!("dispersion {label:<9}{:>20.12} Eh", e_disp);
        println!(
            "total energy + disp {:>20.12} Eh",
            result.scf.energy + e_disp
        );
    }
    if composite.is_none()
        && let (Some(e_gcp), Some(p)) = (result.gcp_energy, standalone_gcp)
    {
        println!();
        println!("gCP ({})  {:>20.12} Eh", p.label, e_gcp);
        println!(
            "total energy incl. corrections {:>20.12} Eh",
            result.best_energy()
        );
    }
    if let Some(dh) = &result.double_hybrid {
        println!();
        println!(
            "double hybrid {} (PT2 on {} orbitals):",
            dh.functional_name, dh.scf_functional_name
        );
        println!("  E_SCF (DH, no PT2)  {:>20.12} Eh", dh.e_scf);
        println!(
            "  E_PT2 os            {:>20.12} Eh   (c_os = {:.5} x {:.12})",
            dh.c_os * dh.e_os,
            dh.c_os,
            dh.e_os
        );
        println!(
            "  E_PT2 ss            {:>20.12} Eh   (c_ss = {:.5} x {:.12})",
            dh.c_ss * dh.e_ss,
            dh.c_ss,
            dh.e_ss
        );
        match &dh.pt2_aux_basis {
            Some(aux) => println!("  PT2 backend         {:>20}", format!("RI-MP2 ({aux})")),
            None => println!("  PT2 backend         {:>20}", "conventional MP2"),
        }
        println!("  frozen core         {:>20} orbitals", dh.n_frozen);
        if let Some(e_nl) = result.vv10_energy {
            println!("  E_nl (VV10 x {:.5}){:>20.12} Eh", dh.vv10_scale, e_nl);
        }
        if let Some(e_disp) = result.dispersion_energy {
            println!("  E_disp (D4)         {:>20.12} Eh", e_disp);
        }
        println!("  total energy        {:>20.12} Eh", result.best_energy());
    }
    if let Some(post) = &result.post_hf {
        match post {
            PostHfResult::Mp2 {
                result: r,
                n_frozen,
            } => report_mp2(*n_frozen, r, result.scf.reference),
            PostHfResult::RiMp2 {
                result: r,
                n_frozen,
                aux_basis,
            } => report_ri_mp2(*n_frozen, r, aux_basis, result.scf.reference),
            PostHfResult::Ccsd {
                result: r,
                n_frozen,
            } => report_ccsd(*n_frozen, r),
            PostHfResult::CcsdT {
                result: r,
                n_frozen,
            } => report_ccsdt(*n_frozen, r),
        }
    }
    if let Some(props) = &result.properties {
        report_properties(&molecule, props.dipole_au, &props.population);
    }
    if let Some(freq) = &result.frequencies {
        if freq.is_sph {
            println!();
            println!(
                "NOTE: single-point Hessian (SPH) — geometry taken as-is, gradient direction \
                 projected out (Spicher & Grimme 2021). SPH frequencies are APPROXIMATE."
            );
        }
        report_frequencies(&freq.frequencies);
        report_thermo(&freq.thermochemistry);
    }
    if !no_method_warnings {
        print_method_warnings(&result.method_warnings);
    }

    Ok(result.converged())
}

fn print_method_warnings(warnings: &[String]) {
    if warnings.is_empty() {
        return;
    }
    println!();
    println!("method-quality assessment (informational; suppress with --no-method-warnings):");
    for w in warnings {
        println!("  {w}");
    }
}

fn report_recommendation(task: &str) -> Result<bool, String> {
    let rec = hartree::guardrails::recommend(task).ok_or_else(|| {
        format!(
            "unknown --recommend task {task:?} (available: {})",
            hartree::guardrails::recommendation_tasks().join(", ")
        )
    })?;
    println!("hartree -- recommended level of theory: {}", rec.task);
    println!();
    println!("  level:     {}", rec.level);
    println!("  rationale: {}", rec.rationale);
    println!();
    println!("  run:");
    for inv in rec.invocation {
        println!("    {inv}");
    }
    if !rec.notes.is_empty() {
        println!();
        for note in rec.notes {
            println!("  note: {note}");
        }
    }
    Ok(true)
}

fn parse_pair<T: std::str::FromStr>(value: &str, flag: &str) -> Result<(T, T), String> {
    let mut parts = value.split(',');
    let err = || format!("{flag} expects two comma-separated values, e.g. {flag} 0,1");
    let a = parts.next().ok_or_else(err)?.trim();
    let b = parts.next().ok_or_else(err)?.trim();
    if parts.next().is_some() {
        return Err(err());
    }
    Ok((a.parse().map_err(|_| err())?, b.parse().map_err(|_| err())?))
}

pub(crate) fn take(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

const GAP_WARN_HF: f64 = 0.1;
const GAP_WARN_KS: f64 = 0.05;

const T1_WARN: f64 = 0.02;

fn report(
    molecule: &Molecule,
    basis: &str,
    method: &str,
    ecp_atoms: &[(String, u32, u32)],
    result: &ScfResult,
    ri: Option<&hartree::RiDiagnostics>,
    cosx: Option<&hartree::CosxDiagnostics>,
) {
    println!("hartree -- {} / {}", method.to_ascii_uppercase(), basis);
    println!(
        "atoms: {}   charge: {}   multiplicity: {}   electrons: {} ({} alpha, {} beta)   basis fns: {}",
        molecule.len(),
        molecule.charge,
        molecule.multiplicity,
        result.n_alpha + result.n_beta,
        result.n_alpha,
        result.n_beta,
        result.n_basis,
    );
    print_ecp_line(ecp_atoms);
    if let Some(ri) = ri {
        println!(
            "RI-JK density fitting: aux basis {} ({} aux fns)",
            ri.aux_basis, ri.naux
        );
    }
    if let Some(cosx) = cosx {
        let rs_note = match cosx.rs_omega {
            Some(omega) => format!(", RS Coulomb + erf(omega={omega}) kernels"),
            None => String::new(),
        };
        println!(
            "COSX semi-numerical exchange: grid {} ({} points{}{})",
            cosx.grid,
            cosx.n_points,
            if cosx.overlap_fitted {
                ", overlap-fitted"
            } else {
                ""
            },
            rs_note
        );
    }
    println!("nuclear repulsion: {:.10} Eh", result.nuclear_repulsion);
    println!();

    println!("{:>5}  {:>22}  {:>11}", "iter", "energy / Eh", "|error|");
    for step in &result.history {
        println!(
            "{:>5}  {:>22.12}  {:>11.2e}",
            step.iteration, step.energy, step.error_norm
        );
    }
    println!();

    if result.converged {
        println!("converged in {} iterations", result.iterations);
    } else {
        println!("NOT CONVERGED after {} iterations", result.iterations);
    }
    println!();
    println!("total energy        {:>20.12} Eh", result.energy);
    println!("electronic          {:>20.12} Eh", result.electronic_energy);
    println!("nuclear repulsion   {:>20.12} Eh", result.nuclear_repulsion);
    if result.reference != Reference::Rhf {
        println!("<S^2>               {:>20.6}", result.spin_squared);
    }

    let (gap_alpha, gap_beta) = result.homo_lumo_gap();
    let is_ks = result.xc_energy.is_some();
    let threshold = if is_ks { GAP_WARN_KS } else { GAP_WARN_HF };
    if result.reference == Reference::Uhf {
        if let Some(g) = gap_alpha {
            println!("HOMO-LUMO gap (α)   {:>20.6} Eh", g);
        }
        if let Some(g) = gap_beta {
            println!("HOMO-LUMO gap (β)   {:>20.6} Eh", g);
        }
    } else if let Some(g) = gap_alpha {
        println!("HOMO-LUMO gap       {:>20.6} Eh", g);
    }
    for (label, gap) in [("α", gap_alpha), ("β", gap_beta)] {
        if let Some(g) = gap {
            if g < threshold {
                println!(
                    "warning: small HOMO-LUMO gap ({g:.4} Eh < {threshold} Eh{}) — possible \
                     multi-reference character or SCF instability",
                    if result.reference == Reference::Uhf {
                        format!(", {label} channel")
                    } else {
                        String::new()
                    }
                );
                if result.reference != Reference::Uhf {
                    break; // α and β are the same channel; warn once
                }
            }
        }
    }
}

fn report_dft(dft: &DftDiagnostics, scf: &ScfResult) {
    let ks = if scf.reference == Reference::Uhf {
        "UKS"
    } else {
        "RKS"
    };
    println!();
    println!("Kohn-Sham DFT: {} ({})", dft.functional_name, ks);
    println!(
        "  grid level {}   ({} points)",
        dft.grid_level, dft.n_grid_points
    );
    if let Some(exc) = scf.xc_energy {
        println!("  E_xc                {:>20.12} Eh", exc);
    }
    if let Some(ne) = scf.n_elec_grid {
        println!(
            "  grid electrons ∫ρ   {:>20.10}   (N = {})",
            ne,
            scf.n_alpha + scf.n_beta
        );
    }
    if dft.exx_fraction > 0.0 {
        println!("  exact exchange c_x  {:>20.4}", dft.exx_fraction);
    }
}

fn report_mp2(n_frozen: usize, mp2: &hartree::cc::Mp2Result, reference: hartree::scf::Reference) {
    let kind = if reference == hartree::scf::Reference::Uhf {
        "UHF"
    } else {
        "RHF"
    };
    println!();
    println!(
        "{kind}-MP2  (frozen core: {} orbital{})",
        n_frozen,
        if n_frozen == 1 { "" } else { "s" }
    );
    println!("  opposite-spin       {:>20.12} Eh", mp2.opposite_spin);
    println!("  same-spin           {:>20.12} Eh", mp2.same_spin);
    println!("  correlation energy  {:>20.12} Eh", mp2.correlation_energy);
    println!("  total energy        {:>20.12} Eh", mp2.total_energy);
}

fn report_ri_mp2(
    n_frozen: usize,
    mp2: &hartree::cc::RiMp2Result,
    aux_basis: &str,
    reference: hartree::scf::Reference,
) {
    let kind = if reference == hartree::scf::Reference::Uhf {
        "UHF"
    } else {
        "RHF"
    };
    println!();
    println!(
        "{kind}-RI-MP2  (frozen core: {} orbital{})",
        n_frozen,
        if n_frozen == 1 { "" } else { "s" }
    );
    println!(
        "  aux basis (MP2-fit) {:>20}   ({} aux fns)",
        aux_basis, mp2.naux
    );
    println!("  opposite-spin       {:>20.12} Eh", mp2.opposite_spin);
    println!("  same-spin           {:>20.12} Eh", mp2.same_spin);
    println!("  correlation energy  {:>20.12} Eh", mp2.correlation_energy);
    println!("  total energy        {:>20.12} Eh", mp2.total_energy);
}

fn report_ccsd(n_frozen: usize, cc: &hartree::cc::CcsdResult) {
    println!();
    println!(
        "RHF-CCSD (spin-adapted; frozen core: {} orbital{})",
        n_frozen,
        if n_frozen == 1 { "" } else { "s" }
    );
    if cc.converged {
        println!("  converged in {} iterations", cc.iterations);
    } else {
        println!("  NOT CONVERGED after {} iterations", cc.iterations);
    }
    println!("  MP2 (iter-0) check  {:>20.12} Eh", cc.mp2_correlation);
    println!("  correlation energy  {:>20.12} Eh", cc.correlation_energy);
    println!("  total energy        {:>20.12} Eh", cc.total_energy);
    println!("  T1 diagnostic       {:>20.6}", cc.t1_diagnostic);
    if cc.t1_diagnostic > T1_WARN {
        println!(
            "warning: T1 diagnostic {:.4} > {T1_WARN} — significant singles amplitudes; \
             a single-reference CC treatment may be unreliable",
            cc.t1_diagnostic
        );
    }
}

fn report_ccsdt(n_frozen: usize, r: &hartree::cc::CcsdTResult) {
    report_ccsd(n_frozen, &r.ccsd);
    println!();
    println!(
        "RHF-CCSD(T)  (frozen core: {} orbital{})",
        n_frozen,
        if n_frozen == 1 { "" } else { "s" }
    );
    println!("  triples correction  {:>20.12} Eh", r.triples_energy);
    println!("  total energy        {:>20.12} Eh", r.total_energy);
}

fn print_ecp_line(ecp_atoms: &[(String, u32, u32)]) {
    if ecp_atoms.is_empty() {
        return;
    }
    let list = ecp_atoms
        .iter()
        .map(|(sym, z, n_core)| format!("{sym} (Z={z}, {n_core} core electrons replaced)"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("effective core potentials (def2-ECP): {list}");
}

fn report_optimization(
    molecule: &Molecule,
    basis: &str,
    method: &str,
    ecp_atoms: &[(String, u32, u32)],
    result: &OptResult,
) {
    println!(
        "hartree -- {} / {}   [geometry optimization]",
        method.to_ascii_uppercase(),
        basis
    );
    println!(
        "atoms: {}   charge: {}   multiplicity: {}",
        molecule.len(),
        molecule.charge,
        molecule.multiplicity,
    );
    print_ecp_line(ecp_atoms);
    println!();

    println!(
        "{:>5}  {:>22}  {:>11}  {:>11}  {:>11}",
        "step", "energy / Eh", "max force", "rms force", "max disp"
    );
    for step in &result.history {
        println!(
            "{:>5}  {:>22.12}  {:>11.2e}  {:>11.2e}  {:>11.2e}",
            step.iteration, step.energy, step.max_force, step.rms_force, step.max_disp
        );
    }
    println!();

    if result.converged {
        println!("optimization converged in {} steps", result.iterations);
    } else {
        println!(
            "optimization NOT CONVERGED after {} steps",
            result.iterations
        );
    }
    println!();

    println!("optimized geometry (angstrom):");
    for (atom, pos) in molecule.atoms.iter().zip(&result.positions) {
        println!(
            "{:<2}  {:>14.8}  {:>14.8}  {:>14.8}",
            atom.element.symbol(),
            pos[0] / ANGSTROM_TO_BOHR,
            pos[1] / ANGSTROM_TO_BOHR,
            pos[2] / ANGSTROM_TO_BOHR,
        );
    }
    println!();
    println!("total energy        {:>20.12} Eh", result.energy);
}

/// Print a one-line warning when a two-endpoint atom mapping was ambiguous (symmetric or
/// equivalent atoms it could not uniquely resolve), so the user knows the reaction
/// coordinate and guess may rest on an arbitrary choice among interchangeable atoms.
/// Threshold mirrors a "less than fully unique" mapping; a confident or absent mapping
/// prints nothing.
fn print_mapping_confidence(confidence: Option<&hartree::opt::ts::guess::MappingConfidence>) {
    if let Some(c) = confidence {
        if c.confidence < 1.0 {
            println!(
                "note: atom mapping confidence low ({:.2}); {} atom(s) ambiguous \
                 (symmetric/equivalent atoms) -- inspect the mapping if the guess looks wrong\n",
                c.confidence,
                c.ambiguous.len()
            );
        }
    }
}

fn report_transition_state(
    molecule: &Molecule,
    basis: &str,
    method: &str,
    ecp_atoms: &[(String, u32, u32)],
    result: &TsResult,
) {
    println!(
        "hartree -- {} / {}   [transition-state search]",
        method.to_ascii_uppercase(),
        basis
    );
    println!(
        "atoms: {}   charge: {}   multiplicity: {}",
        molecule.len(),
        molecule.charge,
        molecule.multiplicity,
    );
    print_ecp_line(ecp_atoms);
    println!();

    println!(
        "{:>5}  {:>22}  {:>11}  {:>11}  {:>11}",
        "step", "energy / Eh", "max force", "rms force", "max disp"
    );
    for step in &result.history {
        println!(
            "{:>5}  {:>22.12}  {:>11.2e}  {:>11.2e}  {:>11.2e}",
            step.iteration, step.energy, step.max_force, step.rms_force, step.max_disp
        );
    }
    println!();

    match result.status {
        TsStatus::Converged => {
            println!("transition state converged in {} steps", result.iterations)
        }
        TsStatus::NotConverged => println!(
            "transition-state search NOT CONVERGED after {} steps",
            result.iterations
        ),
        TsStatus::WrongImaginaryModeCount => println!(
            "geometry converged in {} steps, but it is NOT a first-order saddle \
             (wrong number of imaginary modes)",
            result.iterations
        ),
        TsStatus::StoppedEarly => {
            println!("transition-state search stopped early by an observer")
        }
        _ => println!("transition-state search finished with an unrecognized status"),
    }

    if !result.converged() {
        if let Some(reason) = &result.diagnostic {
            println!("reason: {reason}");
        }
    }

    if let Some(v) = &result.verification {
        let n_neg = v.negative_eigenvalues.len();
        println!(
            "saddle check: {} imaginary mode{}",
            n_neg,
            if n_neg == 1 { "" } else { "s" }
        );
        if let Some(freq) = v.imaginary_frequency_cm1 {
            println!("  imaginary frequency {:>13.2} cm^-1", freq);
        }
    }
    println!();

    println!("transition-state geometry (angstrom):");
    for (atom, pos) in molecule.atoms.iter().zip(&result.positions) {
        println!(
            "{:<2}  {:>14.8}  {:>14.8}  {:>14.8}",
            atom.element.symbol(),
            pos[0] / ANGSTROM_TO_BOHR,
            pos[1] / ANGSTROM_TO_BOHR,
            pos[2] / ANGSTROM_TO_BOHR,
        );
    }
    println!();
    println!("total energy        {:>20.12} Eh", result.energy);

    if let Some(irc) = &result.irc {
        println!();
        println!(
            "IRC endpoint confirmation (mass-weighted downhill trace along the reaction mode):"
        );
        let df = irc.forward_energy - result.energy;
        let dr = irc.reverse_energy - result.energy;
        let tag = |converged: bool, steps: usize| {
            if converged {
                format!("minimum, {steps} steps")
            } else {
                format!("step cap, {steps} steps")
            }
        };
        println!(
            "  forward endpoint energy  {:>20.12} Eh   ΔE = {:>9.2e} Eh   ({})",
            irc.forward_energy,
            df,
            tag(irc.forward_converged, irc.forward_steps)
        );
        println!(
            "  reverse endpoint energy  {:>20.12} Eh   ΔE = {:>9.2e} Eh   ({})",
            irc.reverse_energy,
            dr,
            tag(irc.reverse_converged, irc.reverse_steps)
        );
        if df <= 0.0 && dr <= 0.0 {
            println!("  both endpoints relaxed below the saddle (ΔE <= 0).");
        } else {
            println!(
                "  note: an endpoint did not relax below the saddle (ΔE > 0); \
                 inspect the geometry."
            );
        }
    }
}

/// One atom in the serialized transition-state geometry: its element symbol and
/// Cartesian coordinates in both angstrom and bohr (atomic units), so a consumer
/// need not know hartree's internal unit convention.
#[derive(Serialize)]
struct TsJsonAtom {
    element: String,
    xyz_angstrom: [f64; 3],
    xyz_bohr: [f64; 3],
}

/// The harmonic-verification summary of a serialized transition state: the
/// imaginary frequency (cm^-1, negative by the usual convention) and the number of
/// negative (imaginary) Hessian modes. Present only when the post-convergence
/// verification ran.
#[derive(Serialize)]
struct TsJsonVerification {
    imaginary_frequency_cm1: Option<f64>,
    n_imaginary_modes: usize,
}

/// The two-endpoint IRC summary of a serialized transition state: the relaxed
/// forward/reverse endpoint energies (Eh), whether each reached a minimum, and the
/// step counts. Present only when an IRC trace ran.
#[derive(Serialize)]
struct TsJsonIrc {
    forward_energy_eh: f64,
    forward_converged: bool,
    forward_steps: usize,
    reverse_energy_eh: f64,
    reverse_converged: bool,
    reverse_steps: usize,
}

/// Machine-readable record of a transition-state search, written by `--ts-output`.
/// A thin wrapper over [`TsResult`] that fixes units explicitly and folds in the
/// CLI-side context (level of theory, atom-mapping confidence) the solver does not
/// carry.
#[derive(Serialize)]
struct TsJson {
    method: String,
    basis: String,
    status: String,
    converged: bool,
    energy_eh: f64,
    n_steps: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    geometry: Vec<TsJsonAtom>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verification: Option<TsJsonVerification>,
    #[serde(skip_serializing_if = "Option::is_none")]
    irc: Option<TsJsonIrc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mapping_confidence: Option<f64>,
}

/// A short, stable string for a [`TsStatus`], for the JSON `status` field.
fn ts_status_str(status: TsStatus) -> &'static str {
    match status {
        TsStatus::Converged => "converged",
        TsStatus::NotConverged => "not_converged",
        TsStatus::WrongImaginaryModeCount => "wrong_imaginary_mode_count",
        TsStatus::StoppedEarly => "stopped_early",
        _ => "unknown",
    }
}

/// Serialize a completed transition-state search to `path` as JSON. A write failure
/// is reported but does not fail an otherwise-successful run.
fn write_ts_json(
    path: &str,
    molecule: &Molecule,
    basis: &str,
    method: &str,
    result: &TsResult,
    mapping_confidence: Option<&hartree::opt::ts::guess::MappingConfidence>,
) {
    let geometry: Vec<TsJsonAtom> = molecule
        .atoms
        .iter()
        .zip(&result.positions)
        .map(|(atom, pos)| TsJsonAtom {
            element: atom.element.symbol().to_string(),
            xyz_angstrom: [
                pos[0] / ANGSTROM_TO_BOHR,
                pos[1] / ANGSTROM_TO_BOHR,
                pos[2] / ANGSTROM_TO_BOHR,
            ],
            xyz_bohr: *pos,
        })
        .collect();

    let verification = result.verification.as_ref().map(|v| TsJsonVerification {
        imaginary_frequency_cm1: v.imaginary_frequency_cm1,
        n_imaginary_modes: v.negative_eigenvalues.len(),
    });

    let irc = result.irc.as_ref().map(|i| TsJsonIrc {
        forward_energy_eh: i.forward_energy,
        forward_converged: i.forward_converged,
        forward_steps: i.forward_steps,
        reverse_energy_eh: i.reverse_energy,
        reverse_converged: i.reverse_converged,
        reverse_steps: i.reverse_steps,
    });

    let record = TsJson {
        method: method.to_string(),
        basis: basis.to_string(),
        status: ts_status_str(result.status).to_string(),
        converged: result.converged(),
        energy_eh: result.energy,
        n_steps: result.iterations,
        reason: result.diagnostic.clone(),
        geometry,
        verification,
        irc,
        mapping_confidence: mapping_confidence.map(|c| c.confidence),
    };

    match serde_json::to_string_pretty(&record) {
        Ok(mut json) => {
            json.push('\n');
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("warning: could not write --ts-output {path}: {e}");
            } else {
                println!("wrote machine-readable transition-state result to {path}");
            }
        }
        Err(e) => eprintln!("error: could not serialize transition-state result: {e}"),
    }
}

fn report_properties(molecule: &Molecule, mu_au: [f64; 3], pop: &PopulationAnalysis) {
    println!();
    println!("one-electron properties:");
    let mu_debye: [f64; 3] = mu_au.map(|x| x * AU_DIPOLE_TO_DEBYE);
    let mu_mag = mu_debye.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!(
        "  dipole moment:  x = {:>10.6} D   y = {:>10.6} D   z = {:>10.6} D",
        mu_debye[0], mu_debye[1], mu_debye[2]
    );
    println!("  |μ|            {:>10.6} Debye", mu_mag);
    println!();
    println!("  atom  element      Mulliken      Löwdin");
    for (i, atom) in molecule.atoms.iter().enumerate() {
        println!(
            "  {:>4}  {:>7}    {:>10.6}    {:>10.6}",
            i + 1,
            atom.element.symbol(),
            pop.mulliken_charges[i],
            pop.lowdin_charges[i],
        );
    }
    println!();
    println!("  Mayer bond orders:");
    let n = molecule.len();
    for i in 0..n {
        for j in i + 1..n {
            let b = pop.mayer_bond_orders[i][j];
            if b > 0.1 {
                println!(
                    "    {:>2}({}) – {:>2}({})   {:>8.4}",
                    i + 1,
                    molecule.atoms[i].element.symbol(),
                    j + 1,
                    molecule.atoms[j].element.symbol(),
                    b
                );
            }
        }
    }
}

fn run_xtb(
    molecule: &Molecule,
    method_str: &str,
    method: hartree::ext::xtb::XtbMethod,
    do_opt: bool,
    alpb: Option<String>,
) -> Result<bool, String> {
    use hartree::ext::xtb::{XtbInput, XtbRun, run};
    let run_kind = if do_opt { XtbRun::Opt } else { XtbRun::Energy };
    let input = XtbInput::from_molecule(method, molecule, alpb);
    let workdir = std::env::temp_dir().join(format!("hartree_xtb_{}", std::process::id()));
    let result = run(molecule, &input, run_kind, &workdir).map_err(|e| e.to_string())?;
    println!(
        "hartree -- {} (external xtb)",
        method_str.to_ascii_uppercase()
    );
    println!(
        "atoms: {}   charge: {}   multiplicity: {}",
        molecule.len(),
        molecule.charge,
        molecule.multiplicity
    );
    println!();
    println!("total energy        {:>20.12} Eh", result.energy);
    if let Some(opt) = &result.optimized {
        println!();
        println!("optimized geometry (angstrom):");
        for atom in &opt.atoms {
            println!(
                "{:<2}  {:>14.8}  {:>14.8}  {:>14.8}",
                atom.element.symbol(),
                atom.position[0] / ANGSTROM_TO_BOHR,
                atom.position[1] / ANGSTROM_TO_BOHR,
                atom.position[2] / ANGSTROM_TO_BOHR,
            );
        }
    }
    Ok(true)
}

fn run_conformers(
    molecule: &Molecule,
    method_str: &str,
    basis: &str,
    engine: Option<&str>,
    alpb: Option<String>,
    out_path: Option<&str>,
) -> Result<bool, String> {
    use hartree::ext::ensemble::Ensemble;

    let ensemble: Ensemble = if engine == Some("crest") {
        use hartree::ext::crest::{CrestInput, run};
        let method = hartree::ext::xtb::XtbMethod::from_keyword(method_str)
            .unwrap_or(hartree::ext::xtb::XtbMethod::Gfn2Xtb);
        let input = CrestInput::from_molecule(method, molecule, alpb);
        let workdir = std::env::temp_dir().join(format!("hartree_crest_{}", std::process::id()));
        println!(
            "hartree -- conformer ensemble (external CREST, {})",
            method.keyword()
        );
        run(molecule, &input, &workdir).map_err(|e| e.to_string())?
    } else {
        use hartree::ext::confgen::{ConfGenOptions, generate_conformers};
        let method = simple_method(method_str)?;
        let opts = ConfGenOptions::default();
        println!(
            "hartree -- conformer ensemble (fallback torsion-driving generator, {} / {})",
            method_str.to_ascii_uppercase(),
            basis
        );
        let res = generate_conformers(molecule, &opts, |m| {
            let job = Job {
                molecule: m.clone(),
                basis: basis.to_string(),
                method: method.clone(),
                options: JobOptions::default(),
            };
            job.run().ok().and_then(|r| {
                if r.converged() {
                    Some(r.best_energy())
                } else {
                    None
                }
            })
        })
        .map_err(|e| e.to_string())?;
        println!(
            "rotatable bonds: {}   driven: {}   candidates: {}   screened: {}",
            res.rotatable_bonds.len(),
            res.driven_bonds.len(),
            res.n_candidates,
            res.n_screened
        );
        res.ensemble
    };

    if ensemble.is_empty() {
        return Err("conformer search produced no valid conformers".into());
    }
    let rel_kcal = ensemble.relative_energies_kcal();
    let weights = ensemble.boltzmann_weights(298.15);
    println!();
    println!(
        "ensemble ({} conformers, Boltzmann weights at 298.15 K):",
        ensemble.len()
    );
    println!(
        "  {:>4}  {:>20}  {:>12}  {:>10}",
        "rank", "energy / Eh", "ΔE / kcal", "weight"
    );
    for (i, c) in ensemble.conformers.iter().enumerate() {
        println!(
            "  {:>4}  {:>20.12}  {:>12.4}  {:>10.4}",
            i + 1,
            c.energy,
            rel_kcal[i],
            weights[i]
        );
    }

    if let Some(path) = out_path {
        let mut text = String::new();
        for c in &ensemble.conformers {
            text.push_str(
                &hartree::ext::xyz::write_xyz(&c.molecule, &format!("{:.12}", c.energy))
                    .map_err(|e| e.to_string())?,
            );
        }
        std::fs::write(path, text).map_err(|e| format!("writing {path}: {e}"))?;
        println!();
        println!("ensemble written to {path}");
    }
    Ok(true)
}

fn report_w1(molecule: &Molecule, opts: &hartree::w1::W1Options, res: &hartree::w1::W1Result) {
    println!("hartree -- hartree-W1 composite thermochemistry protocol");
    println!("(a W1-STYLE protocol from hartree's native methods -- NOT literal W1/W1-F12)");
    println!(
        "atoms: {}   charge: {}   multiplicity: {}",
        molecule.len(),
        molecule.charge,
        molecule.multiplicity
    );
    println!();
    println!("protocol stages:");
    println!("  1. geometry + frequencies   {}", res.opt_label);
    println!(
        "  2. HF/CBS                   {} / {}  (Karton-Martin 2-pt, E(L) = E_CBS + A(L+1)e^(-9*sqrt(L)))",
        res.basis_small, res.basis_large
    );
    println!(
        "  3. CCSD corr/CBS            {} / {}  (Halkier 2-pt n^-3), frozen core: {}",
        res.basis_small, res.basis_large, res.n_frozen
    );
    println!(
        "  4. (T) additivity           {} (unextrapolated)",
        res.basis_small
    );
    println!("  omitted (documented): core-valence correlation, scalar relativity, spin-orbit");
    println!();
    println!("optimized geometry (angstrom):");
    for atom in &res.geometry.atoms {
        println!(
            "{:<2}  {:>14.8}  {:>14.8}  {:>14.8}",
            atom.element.symbol(),
            atom.position[0] / ANGSTROM_TO_BOHR,
            atom.position[1] / ANGSTROM_TO_BOHR,
            atom.position[2] / ANGSTROM_TO_BOHR,
        );
    }
    println!();
    println!("hartree-W1 energy breakdown:");
    println!(
        "  E_opt   ({:<})        {:>20.12} Eh   (at its optimized geometry)",
        res.opt_label, res.e_opt
    );
    println!(
        "  E_HF    ({:<})            {:>20.12} Eh",
        res.basis_small, res.e_hf_small
    );
    println!(
        "  E_HF    ({:<})            {:>20.12} Eh",
        res.basis_large, res.e_hf_large
    );
    println!(
        "  E_HF/CBS                       {:>20.12} Eh",
        res.e_hf_cbs
    );
    println!(
        "  E_corr(CCSD, {:<})        {:>20.12} Eh",
        res.basis_small, res.e_ccsd_corr_small
    );
    println!(
        "  E_corr(CCSD, {:<})        {:>20.12} Eh",
        res.basis_large, res.e_ccsd_corr_large
    );
    println!(
        "  E_corr(CCSD)/CBS               {:>20.12} Eh",
        res.e_ccsd_corr_cbs
    );
    println!(
        "  E_(T)   ({:<})            {:>20.12} Eh",
        res.basis_small, res.e_t_small
    );
    println!(
        "  E(hartree-W1)                    {:>20.12} Eh   = E_HF/CBS + E_corr/CBS + E_(T)",
        res.electronic_energy()
    );

    if let Some(t) = &res.thermo {
        report_frequencies(&t.freq.frequencies);
        report_thermo(&t.freq.thermochemistry);
        println!();
        println!(
            "hartree-W1 thermochemistry (thermal corrections at {}, sigma = {}):",
            res.opt_label, opts.symmetry_number
        );
        println!("  H_corr(low) = H-E     {:>20.12} Eh", t.h_corr);
        println!("  G_corr(low) = G-E     {:>20.12} Eh   (RRHO)", t.g_corr);
        println!("  G_corr(low, mRRHO)    {:>20.12} Eh", t.g_corr_qrrho);
        println!(
            "  H(hartree-W1)           {:>20.12} Eh   = E + H_corr(low)",
            t.enthalpy
        );
        println!(
            "  G(hartree-W1, RRHO)     {:>20.12} Eh   = E + G_corr(low)",
            t.gibbs
        );
        println!(
            "  G(hartree-W1, mRRHO)    {:>20.12} Eh   (recommended)",
            t.gibbs_qrrho
        );
    }
}

fn report_multilevel(
    molecule: &Molecule,
    spec: &hartree::multilevel::MultiLevelSpec,
    res: &hartree::multilevel::MultiLevelResult,
) {
    println!("hartree -- multi-level: {}", spec.label());
    println!(
        "atoms: {}   charge: {}   multiplicity: {}",
        molecule.len(),
        molecule.charge,
        molecule.multiplicity
    );
    println!("  SP  level (energy)    {}", spec.sp.label);
    println!("  OPT level (geometry)  {}", spec.opt.label);
    println!();

    let opt = res
        .opt
        .optimized_geometry
        .as_ref()
        .expect("multi-level result carries the optimization");
    println!(
        "geometry: optimized at {} ({} steps, converged)",
        spec.opt.label, opt.iterations
    );
    println!("optimized geometry (angstrom):");
    for atom in &res.geometry.atoms {
        println!(
            "{:<2}  {:>14.8}  {:>14.8}  {:>14.8}",
            atom.element.symbol(),
            atom.position[0] / ANGSTROM_TO_BOHR,
            atom.position[1] / ANGSTROM_TO_BOHR,
            atom.position[2] / ANGSTROM_TO_BOHR,
        );
    }
    println!();

    println!("multi-level energy breakdown:");
    println!(
        "  E_low   ({:<})  {:>20.12} Eh   (at its optimized geometry)",
        spec.opt.label, res.e_low
    );
    println!(
        "  E(SCF)  ({:<})  {:>20.12} Eh",
        spec.sp.label, res.sp.scf.energy
    );
    for (label, value) in [
        ("E_disp", res.sp.dispersion_energy),
        ("E_gCP", res.sp.gcp_energy),
        ("E_SRB", res.sp.srb_energy),
        ("E_nl (VV10)", res.sp.vv10_energy),
    ] {
        if let Some(v) = value {
            println!("  {label:<21} {v:>20.12} Eh   (SP level)");
        }
    }
    if let Some(post) = &res.sp.post_hf {
        println!(
            "  E_corr (post-HF)      {:>20.12} Eh   (SP level)",
            post.total_energy() - res.sp.scf.energy
        );
    }
    if let Some(dh) = &res.sp.double_hybrid {
        println!(
            "  E_PT2 ({:<})  {:>20.12} Eh   (SP level)",
            dh.functional_name,
            dh.pt2_energy()
        );
    }
    println!(
        "  E_high//low           {:>20.12} Eh   (final electronic energy)",
        res.e_high
    );

    if let Some(t) = &res.thermo {
        report_frequencies(&t.freq.frequencies);
        report_thermo(&t.freq.thermochemistry);
        println!();
        println!(
            "composite thermochemistry (thermal corrections at {}, electronic energy at {}):",
            spec.opt.label, spec.sp.label
        );
        println!("  E_high//low           {:>20.12} Eh", res.e_high);
        println!("  H_corr(low) = H-E     {:>20.12} Eh", t.h_corr);
        println!("  G_corr(low) = G-E     {:>20.12} Eh   (RRHO)", t.g_corr);
        println!("  G_corr(low, mRRHO)    {:>20.12} Eh", t.g_corr_qrrho);
        println!(
            "  H(composite)          {:>20.12} Eh   = E_high + H_corr(low)",
            t.enthalpy
        );
        println!(
            "  G(composite, RRHO)    {:>20.12} Eh   = E_high + G_corr(low)",
            t.gibbs
        );
        println!(
            "  G(composite, mRRHO)   {:>20.12} Eh   (recommended)",
            t.gibbs_qrrho
        );
    }
}

fn run_multilevel_conformers(
    molecule: &Molecule,
    spec: &hartree::multilevel::MultiLevelSpec,
    ml_opts: &hartree::multilevel::MultiLevelOptions,
    engine: Option<&str>,
    alpb: Option<String>,
    out_path: Option<&str>,
) -> Result<bool, String> {
    use hartree::ext::ensemble::Ensemble;
    use hartree::multilevel::ENSEMBLE_RERANK_CAP;

    let ensemble: Ensemble = if engine == Some("crest") {
        use hartree::ext::crest::{CrestInput, run};
        let method = hartree::ext::xtb::XtbMethod::Gfn2Xtb;
        let input = CrestInput::from_molecule(method, molecule, alpb);
        let workdir = std::env::temp_dir().join(format!("hartree_crest_{}", std::process::id()));
        println!(
            "hartree -- multi-level conformer re-ranking (external CREST, {})",
            method.keyword()
        );
        run(molecule, &input, &workdir).map_err(|e| e.to_string())?
    } else {
        use hartree::ext::confgen::{ConfGenOptions, generate_conformers};
        println!(
            "hartree -- multi-level conformer re-ranking (fallback generator, \
             screened at RHF/STO-3G)"
        );
        let res = generate_conformers(molecule, &ConfGenOptions::default(), |m| {
            let job = Job {
                molecule: m.clone(),
                basis: "sto-3g".into(),
                method: Method::Rhf,
                options: JobOptions::default(),
            };
            job.run().ok().and_then(|r| {
                if r.converged() {
                    Some(r.best_energy())
                } else {
                    None
                }
            })
        })
        .map_err(|e| e.to_string())?;
        res.ensemble
    };
    if ensemble.is_empty() {
        return Err("conformer search produced no valid conformers".into());
    }
    if ensemble.len() > ENSEMBLE_RERANK_CAP {
        println!(
            "note: re-ranking the {ENSEMBLE_RERANK_CAP} lowest of {} conformers \
             (the documented cap keeps the hook cheap)",
            ensemble.len()
        );
    }

    let ranked =
        hartree::multilevel::rerank_ensemble(&ensemble, spec, ml_opts, ENSEMBLE_RERANK_CAP)?;
    println!();
    println!(
        "re-ranked ensemble ({} conformers, {}; Boltzmann weights at 298.15 K \
         from the composite energies):",
        ranked.len(),
        spec.label()
    );
    println!(
        "  {:>4}  {:>20}  {:>20}  {:>12}  {:>10}",
        "rank", "E_low / Eh", "E_high//low / Eh", "ΔE / kcal", "weight"
    );
    const KCAL: f64 = 627.509_474_063;
    let e0 = ranked.first().map(|r| r.e_high).unwrap_or(0.0);
    for (i, r) in ranked.iter().enumerate() {
        println!(
            "  {:>4}  {:>20.12}  {:>20.12}  {:>12.4}  {:>10.4}",
            i + 1,
            r.e_low,
            r.e_high,
            (r.e_high - e0) * KCAL,
            r.weight
        );
    }

    if let Some(path) = out_path {
        let mut text = String::new();
        for r in &ranked {
            text.push_str(
                &hartree::ext::xyz::write_xyz(&r.molecule, &format!("{:.12}", r.e_high))
                    .map_err(|e| e.to_string())?,
            );
        }
        std::fs::write(path, text).map_err(|e| format!("writing {path}: {e}"))?;
        println!();
        println!("re-ranked ensemble written to {path}");
    }
    Ok(true)
}

fn simple_method(method_str: &str) -> Result<Method, String> {
    match method_str {
        "rhf" | "hf" => Ok(Method::Rhf),
        "uhf" => Ok(Method::Uhf),
        "rohf" => Ok(Method::Rohf),
        other => FunctionalSpec::parse(other).map(Method::Dft).map_err(|_| {
            format!("unsupported conformer-screen method {other:?} (use hf or a functional)")
        }),
    }
}

fn report_frequencies(freq: &FrequencyResult) {
    println!();
    println!("harmonic vibrational frequencies (numerical Hessian of the gradient):");
    for (i, &f) in freq.frequencies_cm1.iter().enumerate() {
        let tag = if f < -1.0 {
            "i"
        } else if f.abs() < 10.0 {
            "(trans/rot)"
        } else {
            ""
        };
        let f_display = if f < 0.0 {
            format!("{:.2}i", (-f))
        } else {
            format!("{:.2}", f)
        };
        println!("  {:>3}:  {:>12} cm⁻¹  {}", i + 1, f_display, tag);
    }
    println!();
    if freq.n_imaginary > 0 {
        println!(
            "  WARNING: {} imaginary mode(s) — not at a minimum",
            freq.n_imaginary
        );
    } else {
        let n_vib = freq.frequencies_cm1.iter().filter(|&&f| f >= 10.0).count();
        println!("  {n_vib} real vibrational modes (0 imaginary)");
    }
}

fn report_thermo(thermo: &ThermoResult) {
    println!();
    println!(
        "RRHO thermochemistry at {:.2} K (σ = {}, linear = {}):",
        thermo.temperature, thermo.symmetry_number, thermo.is_linear
    );
    println!(
        "  zero-point energy (ZPE)          {:>20.12} Eh",
        thermo.zpe
    );
    println!(
        "  thermal correction to E          {:>20.12} Eh",
        thermo.thermal_energy_corr
    );
    println!(
        "  thermal correction to H          {:>20.12} Eh",
        thermo.enthalpy_corr
    );
    println!(
        "  total enthalpy H({:.0} K)        {:>20.12} Eh",
        thermo.temperature, thermo.enthalpy
    );
    println!(
        "  entropy S                        {:>20.12} Eh/K",
        thermo.entropy
    );
    println!(
        "  T·S                              {:>20.12} Eh",
        thermo.temperature * thermo.entropy
    );
    println!(
        "  Gibbs free energy G({:.0} K)     {:>20.12} Eh",
        thermo.temperature, thermo.gibbs
    );
    println!();
    println!("quasi-RRHO (mRRHO) thermochemistry, Grimme Chem. Eur. J. 18, 9955 (2012),");
    println!(
        "w0 = {:.1} cm⁻¹ (entropy-only convention: H is the harmonic RRHO enthalpy):",
        thermo.qrrho_w0_cm1
    );
    println!(
        "  entropy S(mRRHO)                 {:>20.12} Eh/K",
        thermo.entropy_qrrho
    );
    println!(
        "  T·S(mRRHO)                       {:>20.12} Eh",
        thermo.temperature * thermo.entropy_qrrho
    );
    println!(
        "  Gibbs free energy G(mRRHO)       {:>20.12} Eh   (recommended)",
        thermo.gibbs_qrrho
    );
}

fn print_usage() {
    println!(
        "hartree -- pure-Rust quantum chemistry\n\n\
         USAGE:\n    hartree <molecule.xyz> [options]\n\n\
         OPTIONS:\n\
         \x20   --basis <name>          basis set [default: sto-3g]. Bundled: sto-3g, 6-31g;\n\
         \x20                           6-311g, 6-311g(d,p), 6-311+g(d,p), 6-311++g(d,p)\n\
         \x20                           (omits He); cc-pvdz, cc-pvtz, cc-pvqz; aug-cc-pvtz\n\
         \x20                           (all H–Ar); the Karlsruhe family, all-electron H–Kr:\n\
         \x20                           def2-svp, def2-svpd, def2-tzvp, def2-tzvpp, def2-tzvpd,\n\
         \x20                           def2-tzvppd, def2-qzvp, def2-qzvpp, def2-mtzvpp,\n\
         \x20                           def2-mtzvp (mTZVP), def2-msvp,\n\
         \x20                           ma-def2-svp, ma-def2-tzvp (minimally augmented);\n\
         \x20                           def2-svp and def2-tzvp also cover Ag, Sn, I, Au via\n\
         \x20                           the def2-ECP effective core potentials (auto-attached)\n\
         \x20   --method <name>         hf | rhf | uhf | rohf | mp2 | ccsd | ccsd(t) | <functional>\n\
         \x20                           (hf auto-selects rhf/uhf by multiplicity)\n\
         \x20                           [default: rhf]   (mp2 auto-selects an RHF or UHF\n\
         \x20                           reference by multiplicity; ccsd/ccsd(t) are\n\
         \x20                           closed-shell RHF single points; a functional name — svwn, pbe, blyp,\n\
         \x20                           b3lyp, pbe0, tpss, r2scan, m06-2x, pw6b95, b97m-v,\n\
         \x20                           wb97x-v, wb97m-v, … — runs Kohn–Sham DFT, auto RKS/UKS.\n\
         \x20                           Double hybrids b2plyp, revdsd-pbep86, pwpb95,
\n         \x20                           wb97m(2)/wb97m2: closed-shell in-core single points
\n         \x20                           (E = E_SCF + c_os*E_os + c_ss*E_ss, frozen-core PT2
\n         \x20                           on the converged KS orbitals; wb97m(2) is evaluated
\n         \x20                           non-SC on wb97m-v orbitals, VV10 scaled by 0.65904).
\n         \x20                           Range-separated (wb97x-v/wb97m-v) and -V (VV10)\n\
         \x20                           functionals: conventional in-core single points, or\n\
         \x20                           --ri with --cosx for the RS hybrids (RI-J Coulomb +\n\
         \x20                           semi-numerical K; --ri alone and --direct are\n\
         \x20                           rejected for RS; no --opt/--freq); E_nl is evaluated\n\
         \x20                           non-self-consistently after the SCF; grid-sensitive\n\
         \x20                           functionals default to grid level 4)\n\
         \x20                           A -d3 suffix (pbe-d3, b3lyp-d3, hf-d3, …) adds the\n\
         \x20                           D3(BJ) dispersion correction; a -d4 suffix (pbe-d4,\n\
         \x20                           r2scan-d4, …) adds DFT-D4 (EEQ charges + ATM term)\n\
         \x20                           instead (neither for svwn or post-HF, not both at once)\n\
         \x20                           Composite (\"3c\") methods bundle functional + basis +\n\
         \x20                           corrections under one keyword; the basis is implied\n\
         \x20                           and -d3/-d4 suffixes are rejected (each defines its\n\
         \x20                           own corrections):\n\
         \x20                           r2scan-3c: r2scan/def2-mTZVPP + D4 (method-specific\n\
         \x20                           parameters) + gCP, grid defaults to 4\n\
         \x20                           b3lyp-3c: b3lyp5/def2-mSVP + D3(BJ)-ATM + gCP\n\
         \x20                           (DFT/SV(P) parameters)\n\
         \x20                           b97-3c: refitted B97 GGA/mTZVP + D3(BJ)-ATM + SRB\n\
         \x20                           (short-range bond-length correction; no gCP)\n\
         \x20                           pbeh-3c: modified PBE hybrid (42% EXX)/def2-mSVP\n\
         \x20                           + D3(BJ)-ATM + damped gCP\n\
         \x20                           Multi-level (\"//\") specs: --method \"SP // OPT\" runs\n\
         \x20                           the multi-level workflow — optimize at the OPT (low)\n\
         \x20                           level, final single point at the SP (high) level on\n\
         \x20                           that geometry (ORCA notation), e.g.\n\
         \x20                           --method \"wb97m-v/def2-tzvpp // r2scan-3c\". Each\n\
         \x20                           side is method[/basis]: composites carry their own\n\
         \x20                           basis (an explicit one is rejected); plain methods\n\
         \x20                           require one; -d3/-d4 suffixes apply per level. With\n\
         \x20                           --freq the frequencies/thermal corrections run at the\n\
         \x20                           OPT level and G = E_high + (G_low - E_low) is\n\
         \x20                           reported (composite free energy). Solvation flags\n\
         \x20                           apply to every stage by that stage's own rules;\n\
         \x20                           --opt/--basis and the single-point-only backends\n\
         \x20                           (--direct/--ri/--cosx/--x2c/--fod/--smear/--cp/\n\
         \x20                           --props/--sph/--gcp) are rejected with the spec\n\
         \x20   --protocol <name>       composite thermochemistry protocol. w1 (hartree-W1):\n\
         \x20                           a W1-STYLE staged workflow (NOT literal W1/W1-F12):\n\
         \x20                           B3LYP/cc-pVTZ geometry +\n\
         \x20                           frequencies/RRHO; HF/CBS from cc-pVTZ/cc-pVQZ\n\
         \x20                           (2-pt Karton-Martin); frozen-core CCSD-corr/CBS from\n\
         \x20                           the same pair (2-pt Halkier n^-3); (T) at cc-pVTZ\n\
         \x20                           (additivity). Closed-shell molecules only. Prints the\n\
         \x20                           stage breakdown and E/H/G. Core-valence correlation,\n\
         \x20                           scalar relativity, and spin-orbit are omitted\n\
         \x20                           (documented). Honours --symmetry-number, --qrrho-w0,\n\
         \x20                           --all-electron; rejects --method/--basis/--opt/--freq\n\
         \x20                           and the backend/solvation flags\n\
         \x20   --rerank                with --conformers and a multi-level \"//\" spec:\n\
         \x20                           CENSO-lite ensemble re-ranking — optimize each\n\
         \x20                           conformer at the OPT level, single-point at the SP\n\
         \x20                           level, Boltzmann weights from the composite energies\n\
         \x20                           (at most 6 conformers, the documented cap; the\n\
         \x20                           generation step is screened at RHF/STO-3G)\n\
         \x20   --grid <0..4>           DFT integration grid level [default: 3]\n\
         \x20                           (only valid with a functional method)\n\
         \x20   --charge <int>          net charge [default: 0]\n\
         \x20   --spin <int>            spin multiplicity 2S+1 [default: 1]\n\
         \x20   --all-electron          correlate all orbitals for mp2/ccsd/ccsd(t) (default:\n\
         \x20                           noble-gas frozen core)\n\
         \x20   --opt                   optimize the geometry (analytic gradient for rhf/uhf,\n\
         \x20                           finite differences for rohf and DFT functionals)\n\
         \x20   --ts                    locate a transition state (first-order saddle) by P-RFO\n\
         \x20                           eigenvector-following from the input geometry, which must\n\
         \x20                           be a guess near the saddle; verifies one imaginary mode\n\
         \x20                           and reports its frequency (rhf/uhf/rohf/DFT; mutually\n\
         \x20                           exclusive with --opt; no post-HF/RI/direct/COSX/X2C/\n\
         \x20                           smearing/implicit-solvent)\n\
         \x20   --ts-product <file.xyz>  two-endpoint TS search: the main XYZ is the reactant\n\
         \x20                           and this is the product. Builds a near-saddle guess\n\
         \x20                           between them (a single IDPP guess by default) and seeds\n\
         \x20                           the reaction coordinate before refining (requires --ts;\n\
         \x20                           product takes the reactant's charge/multiplicity)\n\
         \x20   --ts-neb                with --ts-product, relax a climbing-image NEB band onto\n\
         \x20                           the minimum-energy path and refine its climbing image\n\
         \x20                           instead of a single IDPP guess (more robust, costlier)\n\
         \x20   --ts-neb-images <int>   interior images for the --ts-neb band [default: 8] (>= 1)\n\
         \x20   --ts-scan <int>         with --ts-product (IDPP route), place the guess at the\n\
         \x20                           energy maximum of the path: evaluate the surface at <int>\n\
         \x20                           images and parabola-fit the peak (>= 3; better guess at\n\
         \x20                           <int> extra single-point energies)\n\
         \x20   --ts-scan-coord <spec>  single-ended distinguished-coordinate scan: drive one\n\
         \x20                           internal coordinate of the input geometry across a range,\n\
         \x20                           relaxing the rest at each point, and refine the saddle from\n\
         \x20                           the energy peak. spec is \"i,j[,k[,l]]:start:end:steps\":\n\
         \x20                           2 indices = bond i-j (range in angstrom), 3 = angle with\n\
         \x20                           centre k in the middle, 4 = dihedral about j-k (range in\n\
         \x20                           degrees); steps >= 3. Mutually exclusive with --ts-product\n\
         \x20   --ts-output <file.json>  after the search, write a machine-readable JSON\n\
         \x20                           record of the result (status, energy, final geometry\n\
         \x20                           in angstrom and bohr, imaginary frequency, IRC summary,\n\
         \x20                           step count; requires --ts)\n\
         \x20   --ts-irc                after a converged TS, trace the intrinsic reaction\n\
         \x20                           coordinate downhill in both senses of the reaction mode\n\
         \x20                           into the two basins and report the endpoint energies\n\
         \x20                           (confirms the minima the saddle joins)\n\
         \x20   --ts-irc-method <dvv|gs2|eulerpc>  IRC integrator [default: dvv]; dvv is\n\
         \x20                           Hessian-free, gs2 (Gonzalez–Schlegel) is constrained and\n\
         \x20                           most accurate, eulerpc reuses one cached Hessian\n\
         \x20   --ts-irc-step <step>    IRC arc-length step, mass-weighted (√amu·bohr)\n\
         \x20                           [default: 0.1]\n\
         \x20   --ts-irc-max-steps <int>  max IRC steps per endpoint [default: 150] (>= 1)\n\
         \x20   --ts-irc-gtol <a.u.>    IRC convergence threshold on the projected RMS force\n\
         \x20                           [default: 1e-3]\n\
         \x20   --ts-algo <prfo|dimer>  saddle-point algorithm [default: prfo]; both prfo\n\
         \x20                           (Hessian eigenvector-following) and dimer (Hessian-free,\n\
         \x20                           midpoint-gradient curvature estimate) are available\n\
         \x20   --ts-dimer-delta <bohr>  dimer half-separation for the curvature estimate\n\
         \x20                           [default: 1e-2] (--ts-algo dimer only)\n\
         \x20   --ts-max-iter <int>     max saddle-search iterations [default: 300] (>= 1)\n\
         \x20   --ts-trust <bohr>       initial trust radius for the climbing step [default: 0.2]\n\
         \x20   --ts-follow <int>       P-RFO mode to follow uphill, 0 = softest [default: 0]\n\
         \x20   --ts-recalc-hessian <int>  recompute the FD Hessian every N accepted steps;\n\
         \x20                           0 = compute once, then Bofill update [default: 0]\n\
         \x20   --ts-stall-refresh <int>  refresh the FD Hessian after N consecutive\n\
         \x20                           non-improving steps (P-RFO); a soft-surface aid for\n\
         \x20                           floppy systems, 0 = off [default: 0], try 5 if a search\n\
         \x20                           plateaus far from convergence\n\
         \x20   --ts-verify-hessian <strict|maintained|auto>  Hessian the post-convergence\n\
         \x20                           verification uses (P-RFO) [default: strict]; strict\n\
         \x20                           finite-differences a fresh one, maintained reuses the\n\
         \x20                           Bofill Hessian, auto reuses it unless a mode is near the\n\
         \x20                           threshold\n\
         \x20   --ts-coordinates <mass-weighted|internal>  coordinate frame the P-RFO climb\n\
         \x20                           steps in [default: mass-weighted]; internal uses redundant\n\
         \x20                           internal coordinates (bonds+angles) for better conditioning\n\
         \x20                           of soft reaction coordinates, falling back to mass-weighted\n\
         \x20                           when the internal set is incomplete\n\
         \x20   --ts-fd-step <bohr>     finite-difference step for gradients/Hessian [default:\n\
         \x20                           5e-3]\n\
         \x20   --ts-neg-tol <a.u.>     eigenvalue cutoff for a negative (reaction) mode\n\
         \x20                           [default: 1e-4]\n\
         \x20                           (TS runs raise the SCF cap to 400 and add a 0.3 level\n\
         \x20                           shift for small-gap stability)\n\
         \x20                           (the --ts-* knobs take effect only with --ts)\n\
         \x20   --direct                integral-direct SCF: recompute ERIs each iteration\n\
         \x20                           instead of storing the nao^4 tensor (rhf/uhf/rohf\n\
         \x20                           single points; reaches larger systems, slower)\n\
         \x20   --ri                    density-fitted (RI-JK) SCF: J/K from 3-center integrals\n\
         \x20                           over the def2-universal-jkfit auxiliary set (hf and all\n\
         \x20                           DFT functionals, single points; small fitting error\n\
         \x20                           ~1e-4 Eh; incompatible with --direct, --opt, post-HF,\n\
         \x20                           --properties, --freq)\n\
         \x20   --ri-mp2                RI-MP2: density-fitted MP2 over the matching def2 /C\n\
         \x20                           (MP2-fit) auxiliary set (<basis>/c; bundled: def2-svp/c,\n\
         \x20                           def2-tzvp/c — errors if no /C partner exists, never\n\
         \x20                           falls back to jkfit). --method mp2 only (RHF or UHF\n\
         \x20                           reference by multiplicity); the SCF step\n\
         \x20                           keeps its backend (default in-core, or --ri). Same\n\
         \x20                           frozen-core convention and E_OS/E_SS split as\n\
         \x20                           conventional MP2 (fitting error ~1e-5 Eh at def2-svp)\n\
         \x20   --cosx                  COSX semi-numerical exchange: K built on a coarse\n\
         \x20                           overlap-fitted grid (Neese et al. 2009 / Izsak & Neese\n\
         \x20                           2011); J keeps the configured path (in-core or --ri).\n\
         \x20                           HF, global-hybrid, and range-separated DFT single\n\
         \x20                           points (RS hybrids serve K_LR(omega) semi-numerically\n\
         \x20                           too, on either backend); rejected for post-HF,\n\
         \x20                           --direct, --opt, --freq, and --fod\n\
         \x20   --x2c                   scalar-relativistic X2C-1e Hamiltonian: T+V replaced\n\
         \x20                           by the picture-changed spin-free exact-two-component\n\
         \x20                           one-electron Hamiltonian (2e integrals stay\n\
         \x20                           nonrelativistic). HF and DFT on any backend; rejected\n\
         \x20                           with ECP atoms (double-counts relativity), --opt,\n\
         \x20                           --freq, and post-HF; properties carry a picture-change\n\
         \x20                           caveat\n\
         \x20   --solvent <name>        C-PCM implicit solvation (electrostatics only) in a\n\
         \x20                           named solvent: water, acetonitrile, methanol, dmso,\n\
         \x20                           chloroform, toluene (SCF-level methods, incl. --ri and\n\
         \x20                           -d3; E_solv included in the total; --opt uses FD\n\
         \x20                           gradients; --freq in solvent is rejected)\n\
         \x20   --eps <float>           C-PCM with an explicit dielectric constant instead of\n\
         \x20                           a named solvent (mutually exclusive with --solvent)\n\
             --smd <name>            SMD universal solvation model (Marenich/Cramer/
\n                                     Truhlar 2009): C-PCM electrostatics with SMD intrinsic
\n                                     Coulomb radii + the CDS surface-tension term; reports
\n                                     dG_EP, G_CDS, dG_solv (1 M -> 1 M standard state) vs a
\n                                     gas-phase reference SCF. 20 bundled solvents (water,
\n                                     methanol, ethanol, acetonitrile, dmso, acetone, thf,
\n                                     dmf, toluene, benzene, chloroform, ...). Same scope as
\n                                     C-PCM (SCF-level methods; --opt via FD; no --freq);
\n                                     mutually exclusive with --solvent/--eps
\n             --alpb <name>           ALPB implicit solvation (Ehlert/Stahn/Spicher/
\n                                     Grimme 2021), xtb GFN2 parameters. Post-SCF on
\n                                     SCF Mulliken charges; reports G_born/G_hb/G_sasa/
\n                                     G_shift and DeltaG_solv. Single points only.
\n                                     GFN2-fit params on ab-initio charges (caveat).
\n             --gbsa <name>           GBSA implicit solvation (generalized Born + SASA),
\n                                     xtb GFN2 parameters. Like --alpb but the Still
\n                                     kernel and no Poisson-Boltzmann shape term.
\n             --cosmo-file <path>     run C-PCM at the ideal-conductor limit (eps=inf)
\n                                     and write a .cosmo file (TURBOMOLE/COSMOtherm
\n                                     format) for COSMO-RS. Mutually exclusive with the
\n                                     other solvation options
\n         \x20   --smear <K>             Fermi-Dirac fractional-occupation smearing at the\n\
         \x20                           given electronic temperature (kelvin). Energy-only:\n\
         \x20                           rhf/uhf and DFT single points (no --opt, --freq, or\n\
         \x20                           post-HF). Reports occupations, T*S_el, and the free\n\
         \x20                           energy F = E - T*S_el\n\
         \x20   --fod                   Grimme FOD multireference diagnostic (Angew. Chem. Int.\n\
         \x20                           Ed. 54, 12308 (2015)): Fermi-smeared SCF at\n\
         \x20                           T_el = 5000 K + 20000 K*a_x, reporting N_FOD (the\n\
         \x20                           integral of the fractional-occupation weighted density;\n\
         \x20                           N_FOD >= 1 signals strong static correlation). Implies\n\
         \x20                           --method tpss --basis def2-tzvp unless given explicitly\n\
         \x20                           (warned); --smear overrides the temperature. In-core\n\
         \x20                           single points only (no --direct/--ri/--opt/--freq or\n\
         \x20                           post-HF)\n\
         \x20   --fod-cube <path>       with --fod, export rho_FOD (alpha+beta) as a Gaussian\n\
         \x20                           cube file (bounding box + 4 bohr margin, 0.2 bohr\n\
         \x20                           spacing)\n\
         \x20   --properties, --props   compute one-electron properties: dipole (Debye),\n\
         \x20                           Mulliken/Löwdin charges, Mayer bond orders\n\
         \x20   --freq                  compute harmonic frequencies (numerical Hessian of the\n\
         \x20                           gradient) and RRHO + quasi-RRHO (mRRHO, Grimme 2012)\n\
         \x20                           thermochemistry at 298.15 K. Works for rhf/uhf and DFT\n\
         \x20                           functionals incl. the 3c composites and ECP atoms;\n\
         \x20                           run at the optimized geometry for all-real modes\n\
         \x20   --qrrho-w0 <cm-1>       quasi-RRHO interpolation frequency w0 for the mRRHO\n\
         \x20                           entropy [default: 100]\n\
         \x20   --sph                   single-point Hessian (Spicher & Grimme, JCTC 2021):\n\
         \x20                           with --freq, take the geometry as-is (skip the\n\
         \x20                           stationarity assumption) and project the gradient\n\
         \x20                           direction out of the Hessian. Reduces to ordinary\n\
         \x20                           frequencies at a true minimum; SPH frequencies are\n\
         \x20                           APPROXIMATE (gradient-projection variant, not xtb\n\
         \x20                           --bhess; see crate docs)\n\
         \x20   --method gfn2-xtb       run the external `xtb` binary (GFN2-xTB); --method\n\
         \x20   --method gfn-ff         gfn-ff runs GFN-FF. Detected on PATH or via\n\
         \x20                           HARTREE_XTB_PATH; honours --charge/--spin/--alpb/--opt.\n\
         \x20                           A clear 'xtb binary not found' error with install\n\
         \x20                           guidance is raised when absent\n\
         \x20   --conformers [crest]    conformer ensemble search. Bare --conformers uses the\n\
         \x20                           built-in RDKit-free torsion-driving fallback (screened\n\
         \x20                           with --method/--basis single points; default RHF/\n\
         \x20                           STO-3G); `--conformers crest` runs the external CREST\n\
         \x20                           (HARTREE_CREST_PATH or PATH). Prints relative energies\n\
         \x20                           and Boltzmann weights at 298.15 K\n\
         \x20   --conformers-out <path> write the conformer ensemble as a multi-frame XYZ\n\
         \x20   --symmetry-number <int> rotational symmetry number σ for RRHO entropy\n\
         \x20                           [default: 1; use 2 for H₂O, HF, CO; 12 for CH₄]\n\
         \x20   --cp <nA>               Boys-Bernardi counterpoise correction for a\n\
         \x20                           two-fragment complex: the first nA atoms of the XYZ\n\
         \x20                           (input order) are fragment A, the rest fragment B.\n\
         \x20                           Runs the 5 required single points (complex; each\n\
         \x20                           fragment with the partner's ghost basis; each\n\
         \x20                           fragment alone) and reports dE_int^CP, the\n\
         \x20                           uncorrected dE_int, and the BSSE estimate. Works\n\
         \x20                           with any single-point-capable method (no --opt/\n\
         \x20                           --freq/--props/--fod/solvent)\n\
         \x20   --cp-charges <qA,qB>    per-fragment net charges for --cp [default: 0,0];\n\
         \x20                           must sum to --charge\n\
         \x20   --cp-mults <mA,mB>      per-fragment spin multiplicities for --cp\n\
         \x20                           [default: 1,1]\n\
         \x20   --gcp <set>             standalone geometric counterpoise (gCP) correction\n\
         \x20                           added to the total energy (outside the 3c\n\
         \x20                           composites). Parameter sets (vendored from the\n\
         \x20                           reference implementation mctc-gcp): r2scan-3c\n\
         \x20                           (alias def2-mtzvpp), dft/sv(p) (alias b3lyp-3c),\n\
         \x20                           pbeh-3c (alias def2-msvp). Ghost atoms are excluded\n\
         \x20   --recommend <task>      print the recommended level of theory for a task and\n\
         \x20                           exit (no XYZ needed): general (r2scan-3c opt+freq),\n\
         \x20                           barriers / nci (wb97m-v/def2-tzvpp single point on a\n\
         \x20                           r2scan-3c geometry via the multi-level '//' workflow),\n\
         \x20                           thermochemistry (the multi-level composite free\n\
         \x20                           energy with --freq)\n\
         \x20   --no-method-warnings    suppress the method-quality assessment section\n\
         \x20                           (informational warnings: Pople/cc bases with DFT,\n\
         \x20                           missing dispersion where the functional metadata\n\
         \x20                           recommends one, minimal/unpolarized bases, pure-GGA\n\
         \x20                           barrier caveats, HF-without-correlation, coarse grids\n\
         \x20                           under grid-sensitive functionals) and the no-method\n\
         \x20                           recommendation pointer. The warnings stay attached to\n\
         \x20                           the library JobResult either way\n\
         \n\
         \x20 PERIODIC (solid-state GPW DFT; selected by --cell / --kpoints / --cutoff):\n\
         \x20   --cell <spec>           unit cell: 'cubic <a>', or 1/3/6/9 numbers (Å, deg):\n\
         \x20                           1=cubic, 3=a b c (orthorhombic), 6=a b c α β γ,\n\
         \x20                           9=three row vectors. Omit the spec (or '--cell file')\n\
         \x20                           to read a Lattice=\"…\" line from the extended-XYZ file\n\
         \x20   --kpoints <n1 n2 n3>    Monkhorst–Pack mesh (or 'gamma') [default: gamma]\n\
         \x20   --cutoff <E>            plane-wave density cutoff in hartree [default: 280]\n\
         \x20   --basis <name>          GTH basis [default: DZVP-GTH-PADE; also SZV-GTH]\n\
         \x20   --xc <name>             pade (GTH-PADE LDA) | lda (Slater+PW92) [default: pade]\n\
         \x20   --pseudo <file>         user GTH potential file (CP2K format) [default: bundled]\n\
         \x20   --basis-file <file>     user GTH basis file (.gbs) [default: bundled]\n\
         \x20   --forces                also compute analytic forces\n\
         \x20   --stress                also compute the analytic stress tensor\n\
         \x20   --max-iter <n> --mixing <a>   SCF controls\n\
         \x20                           (v1: GTH PPs, spin-restricted, insulators/semiconductors;\n\
         \x20                           bundled elements: H, C, O, Si)\n\
         \x20   -h, --help              show this help\n\n\
         XYZ coordinates are in angstrom. Ghost atoms (counterpoise centers: basis\n\
         functions only, no nuclear charge, no electrons) are written Gh(O) or with\n\
         the @O shorthand in the element column; they are excluded from D3/D4/gCP/SRB\n\
         and from geometry jobs (--opt/--freq), and rejected with --props/solvent."
    );
}
