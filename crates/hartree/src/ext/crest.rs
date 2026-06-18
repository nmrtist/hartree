use std::path::{Path, PathBuf};

use crate::core::Molecule;

use crate::ext::ExtError;
use crate::ext::ensemble::{Conformer, Ensemble};
use crate::ext::xtb::XtbMethod;
use crate::ext::xyz::write_xyz;

pub const CREST_PATH_ENV: &str = "HARTREE_CREST_PATH";

#[derive(Debug, Clone)]
pub struct CrestInput {
    pub method: XtbMethod,
    pub charge: i32,
    pub n_unpaired: u32,
    pub alpb: Option<String>,
    pub threads: Option<usize>,
}

impl CrestInput {
    pub fn from_molecule(method: XtbMethod, molecule: &Molecule, alpb: Option<String>) -> Self {
        Self {
            method,
            charge: molecule.charge,
            n_unpaired: molecule.multiplicity.saturating_sub(1),
            alpb,
            threads: None,
        }
    }
}

pub fn crest_args(input: &CrestInput, xyz_file: &str) -> Vec<String> {
    let mut args = vec![xyz_file.to_string()];
    match input.method {
        XtbMethod::Gfn2Xtb => args.push("--gfn2".into()),
        XtbMethod::GfnFf => args.push("--gfnff".into()),
    }
    args.push("--chrg".into());
    args.push(input.charge.to_string());
    args.push("--uhf".into());
    args.push(input.n_unpaired.to_string());
    if let Some(s) = &input.alpb {
        args.push("--alpb".into());
        args.push(s.clone());
    }
    if let Some(t) = input.threads {
        args.push("-T".into());
        args.push(t.to_string());
    }
    args
}

pub fn find_crest() -> Result<PathBuf, ExtError> {
    crate::ext::xtb::find_binary(
        "crest",
        CREST_PATH_ENV,
        "https://github.com/crest-lab/crest",
        "crest",
    )
}

pub fn run(molecule: &Molecule, input: &CrestInput, workdir: &Path) -> Result<Ensemble, ExtError> {
    let exe = find_crest()?;
    std::fs::create_dir_all(workdir).map_err(|e| ExtError::io("creating crest workdir", e))?;
    let xyz_name = "hartree_crest_input.xyz";
    std::fs::write(workdir.join(xyz_name), write_xyz(molecule, "hartree")?)
        .map_err(|e| ExtError::io("writing crest input xyz", e))?;

    let args = crest_args(input, xyz_name);
    let output = std::process::Command::new(&exe)
        .args(&args)
        .current_dir(workdir)
        .output()
        .map_err(|e| ExtError::io(format!("spawning {}", exe.display()), e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ExtError::SubprocessFailed {
            program: "crest",
            status: output.status.to_string(),
            stderr_tail: stderr
                .chars()
                .rev()
                .take(800)
                .collect::<String>()
                .chars()
                .rev()
                .collect(),
        });
    }

    let ens_path = workdir.join("crest_conformers.xyz");
    let text = std::fs::read_to_string(&ens_path).map_err(|_| ExtError::MissingOutput {
        program: "crest",
        path: ens_path,
    })?;
    parse_crest_ensemble(&text, molecule.charge, molecule.multiplicity)
}

pub fn parse_crest_ensemble(
    text: &str,
    charge: i32,
    multiplicity: u32,
) -> Result<Ensemble, ExtError> {
    let frames = crate::ext::xyz::parse_multi_xyz(text)?;
    let mut conformers = Vec::with_capacity(frames.len());
    for (mol, comment) in frames {
        let energy = comment
            .split_whitespace()
            .next()
            .and_then(|t| t.replace(['D', 'd'], "E").parse::<f64>().ok())
            .ok_or_else(|| ExtError::Parse {
                what: "crest_conformers.xyz",
                message: format!("comment line has no leading energy: {comment:?}"),
            })?;
        conformers.push(Conformer {
            molecule: mol.with_charge(charge).with_multiplicity(multiplicity),
            energy,
        });
    }
    Ok(Ensemble::new(conformers))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENSEMBLE_FIXTURE: &str = "\
4
       -19.94120000
C     -1.50000000    -0.20000000     0.00000000
C     -0.50000000     0.50000000     0.00000000
C      0.50000000    -0.50000000     0.00000000
C      1.50000000     0.20000000     0.00000000
4
       -19.94050000
C     -1.50000000    -0.20000000     0.00000000
C     -0.50000000     0.50000000     0.00000000
C      0.50000000    -0.50000000     0.00000000
C      1.40000000     0.20000000     0.80000000
4
       -19.94050000
C     -1.50000000    -0.20000000     0.00000000
C     -0.50000000     0.50000000     0.00000000
C      0.50000000    -0.50000000     0.00000000
C      1.40000000     0.20000000    -0.80000000
";

    #[test]
    fn args_gfn2_alpb_threads() {
        let input = CrestInput {
            method: XtbMethod::Gfn2Xtb,
            charge: 0,
            n_unpaired: 0,
            alpb: Some("water".into()),
            threads: Some(4),
        };
        let args = crest_args(&input, "in.xyz");
        assert_eq!(
            args,
            vec![
                "in.xyz", "--gfn2", "--chrg", "0", "--uhf", "0", "--alpb", "water", "-T", "4"
            ]
        );
    }

    #[test]
    fn parse_ensemble_fixture() {
        let ens = parse_crest_ensemble(ENSEMBLE_FIXTURE, 0, 1).unwrap();
        assert_eq!(ens.len(), 3);
        assert!((ens.min_energy().unwrap() - (-19.9412)).abs() < 1e-8);
        let rel = ens.relative_energies();
        assert!(rel[0].abs() < 1e-12);
        assert!(rel[1] > 0.0);
        let w = ens.boltzmann_weights(298.15);
        assert!((w[1] - w[2]).abs() < 1e-12);
        assert!(w[0] > w[1]);
    }

    #[test]
    fn parse_ensemble_bad_comment_errors() {
        let bad = "2\nnot-a-number\nH 0 0 0\nH 0 0 0.74\n";
        assert!(parse_crest_ensemble(bad, 0, 1).is_err());
    }

    #[test]
    #[ignore = "requires the external crest binary; run with --include-ignored"]
    fn real_crest_ensemble() {
        if find_crest().is_err() {
            eprintln!("crest not found — skipping real-subprocess test");
            return;
        }
        let mol = Molecule::from_xyz("4\nbutane-frag\nC 0 0 0\nC 1.5 0 0\nC 3.0 0 0\nC 4.5 0 0\n")
            .unwrap();
        let input = CrestInput::from_molecule(XtbMethod::Gfn2Xtb, &mol, None);
        let workdir = std::env::temp_dir().join("hartree_crest_realtest");
        let ens = run(&mol, &input, &workdir).expect("crest run");
        assert!(!ens.is_empty());
    }

    #[test]
    fn find_crest_absent_named_error() {
        unsafe {
            std::env::set_var(CREST_PATH_ENV, "/nonexistent/crest/xyzzy");
        }
        match find_crest() {
            Err(ExtError::BinaryNotFound { program, .. }) => assert_eq!(program, "crest"),
            Err(other) => panic!("unexpected: {other}"),
            Ok(_) => {}
        }
        unsafe {
            std::env::remove_var(CREST_PATH_ENV);
        }
    }
}
