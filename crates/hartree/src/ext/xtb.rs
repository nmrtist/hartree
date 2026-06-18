use std::path::{Path, PathBuf};

use crate::core::Molecule;

use crate::ext::ExtError;
use crate::ext::xyz::write_xyz;

pub const XTB_PATH_ENV: &str = "HARTREE_XTB_PATH";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XtbMethod {
    Gfn2Xtb,
    GfnFf,
}

impl XtbMethod {
    pub fn keyword(self) -> &'static str {
        match self {
            XtbMethod::Gfn2Xtb => "gfn2-xtb",
            XtbMethod::GfnFf => "gfn-ff",
        }
    }

    pub fn from_keyword(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gfn2-xtb" | "gfn2" => Some(XtbMethod::Gfn2Xtb),
            "gfn-ff" | "gfnff" => Some(XtbMethod::GfnFf),
            _ => None,
        }
    }

    fn method_flags(self) -> Vec<String> {
        match self {
            XtbMethod::Gfn2Xtb => vec!["--gfn".into(), "2".into()],
            XtbMethod::GfnFf => vec!["--gfnff".into()],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XtbRun {
    Energy,
    Gradient,
    Opt,
}

#[derive(Debug, Clone)]
pub struct XtbInput {
    pub method: XtbMethod,
    pub charge: i32,
    pub n_unpaired: u32,
    pub alpb: Option<String>,
}

impl XtbInput {
    pub fn from_molecule(method: XtbMethod, molecule: &Molecule, alpb: Option<String>) -> Self {
        Self {
            method,
            charge: molecule.charge,
            n_unpaired: molecule.multiplicity.saturating_sub(1),
            alpb,
        }
    }
}

pub fn xtb_args(input: &XtbInput, xyz_file: &str, run: XtbRun) -> Vec<String> {
    let mut args = vec![xyz_file.to_string()];
    args.extend(input.method.method_flags());
    args.push("--chrg".into());
    args.push(input.charge.to_string());
    args.push("--uhf".into());
    args.push(input.n_unpaired.to_string());
    if let Some(s) = &input.alpb {
        args.push("--alpb".into());
        args.push(s.clone());
    }
    match run {
        XtbRun::Energy => {}
        XtbRun::Gradient => args.push("--grad".into()),
        XtbRun::Opt => args.push("--opt".into()),
    }
    args.push("--json".into());
    args
}

pub fn find_xtb() -> Result<PathBuf, ExtError> {
    find_binary(
        "xtb",
        XTB_PATH_ENV,
        "https://github.com/grimme-lab/xtb",
        "xtb",
    )
}

pub(crate) fn find_binary(
    program: &'static str,
    env_var: &'static str,
    install_hint: &'static str,
    conda_pkg: &'static str,
) -> Result<PathBuf, ExtError> {
    if let Some(p) = std::env::var_os(env_var) {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Some(p) = which_on_path(program) {
        return Ok(p);
    }
    Err(ExtError::BinaryNotFound {
        program,
        env_var,
        install_hint,
        conda_pkg,
    })
}

fn which_on_path(program: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.BAT;.CMD".into())
            .split(';')
            .map(|s| s.to_string())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path_var) {
        for ext in &exts {
            let cand = dir.join(format!("{program}{ext}"));
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct XtbResult {
    pub energy: f64,
    pub gradient: Option<Vec<[f64; 3]>>,
    pub optimized: Option<Molecule>,
}

pub fn run(
    molecule: &Molecule,
    input: &XtbInput,
    run_kind: XtbRun,
    workdir: &Path,
) -> Result<XtbResult, ExtError> {
    let exe = find_xtb()?;
    std::fs::create_dir_all(workdir).map_err(|e| ExtError::io("creating xtb workdir", e))?;
    let xyz_name = "hartree_xtb_input.xyz";
    let xyz_path = workdir.join(xyz_name);
    std::fs::write(&xyz_path, write_xyz(molecule, "hartree")?)
        .map_err(|e| ExtError::io("writing xtb input xyz", e))?;

    let args = xtb_args(input, xyz_name, run_kind);
    let output = std::process::Command::new(&exe)
        .args(&args)
        .current_dir(workdir)
        .output()
        .map_err(|e| ExtError::io(format!("spawning {}", exe.display()), e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .chars()
            .rev()
            .take(800)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(ExtError::SubprocessFailed {
            program: "xtb",
            status: output.status.to_string(),
            stderr_tail: if tail.is_empty() { stdout } else { tail },
        });
    }

    let energy = match std::fs::read_to_string(workdir.join("xtbout.json")) {
        Ok(j) => parse_json_energy(&j)?,
        Err(_) => parse_energy_stdout(&stdout)?,
    };

    let gradient = if run_kind == XtbRun::Gradient {
        let grad_path = workdir.join("gradient");
        let text = std::fs::read_to_string(&grad_path).map_err(|_| ExtError::MissingOutput {
            program: "xtb",
            path: grad_path,
        })?;
        Some(parse_turbomole_gradient(&text, molecule.len())?.gradient)
    } else {
        None
    };

    let optimized = if run_kind == XtbRun::Opt {
        let opt_path = workdir.join("xtbopt.xyz");
        let text = std::fs::read_to_string(&opt_path).map_err(|_| ExtError::MissingOutput {
            program: "xtb",
            path: opt_path,
        })?;
        let (mol, _e) = parse_xtbopt_xyz(&text, molecule.charge, molecule.multiplicity)?;
        Some(mol)
    } else {
        None
    };

    Ok(XtbResult {
        energy,
        gradient,
        optimized,
    })
}

#[derive(Debug, Clone)]
pub struct TurbomoleGradient {
    pub energy: f64,
    pub gradient: Vec<[f64; 3]>,
}

pub fn parse_turbomole_gradient(text: &str, n_atoms: usize) -> Result<TurbomoleGradient, ExtError> {
    let perr = |m: String| ExtError::Parse {
        what: "TURBOMOLE gradient file",
        message: m,
    };
    let mut lines = text.lines();
    let mut saw_grad = false;
    for l in lines.by_ref() {
        if l.trim_start().starts_with("$grad") {
            saw_grad = true;
            break;
        }
    }
    if !saw_grad {
        return Err(perr("no $grad block".into()));
    }
    let header = lines
        .next()
        .ok_or_else(|| perr("missing cycle header".into()))?;
    let energy = parse_scf_energy_from_header(header).ok_or_else(|| {
        perr(format!(
            "could not parse SCF energy from header: {header:?}"
        ))
    })?;

    let body: Vec<&str> = lines
        .take_while(|l| !l.trim_start().starts_with("$end"))
        .filter(|l| !l.trim().is_empty())
        .collect();
    if body.len() < 2 * n_atoms {
        return Err(perr(format!(
            "expected {} coordinate+gradient lines, found {}",
            2 * n_atoms,
            body.len()
        )));
    }
    let mut gradient = Vec::with_capacity(n_atoms);
    for line in &body[n_atoms..2 * n_atoms] {
        let vals =
            parse_three_floats(line).ok_or_else(|| perr(format!("bad gradient line: {line:?}")))?;
        gradient.push(vals);
    }
    Ok(TurbomoleGradient { energy, gradient })
}

fn parse_scf_energy_from_header(header: &str) -> Option<f64> {
    let idx = header.find("SCF energy")?;
    let after = &header[idx + "SCF energy".len()..];
    let after = after.trim_start().strip_prefix('=')?;
    let tok = after.split_whitespace().next()?;
    parse_fortran_float(tok)
}

fn parse_three_floats(line: &str) -> Option<[f64; 3]> {
    let mut it = line.split_whitespace();
    let x = parse_fortran_float(it.next()?)?;
    let y = parse_fortran_float(it.next()?)?;
    let z = parse_fortran_float(it.next()?)?;
    Some([x, y, z])
}

fn parse_fortran_float(tok: &str) -> Option<f64> {
    tok.replace(['D', 'd'], "E").parse().ok()
}

pub fn parse_xtbopt_xyz(
    text: &str,
    charge: i32,
    multiplicity: u32,
) -> Result<(Molecule, Option<f64>), ExtError> {
    let mol = Molecule::from_xyz(text)
        .map_err(|e| ExtError::Parse {
            what: "xtbopt.xyz",
            message: e.to_string(),
        })?
        .with_charge(charge)
        .with_multiplicity(multiplicity);
    let comment = text.lines().nth(1).unwrap_or("");
    let energy = comment
        .find("energy:")
        .and_then(|i| comment[i + "energy:".len()..].split_whitespace().next())
        .and_then(parse_fortran_float);
    Ok((mol, energy))
}

pub fn parse_json_energy(json: &str) -> Result<f64, ExtError> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| ExtError::Parse {
        what: "xtbout.json",
        message: e.to_string(),
    })?;
    v.get("total energy")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| ExtError::Parse {
            what: "xtbout.json",
            message: "missing numeric \"total energy\" key".into(),
        })
}

pub fn parse_energy_stdout(stdout: &str) -> Result<f64, ExtError> {
    for line in stdout.lines() {
        let t = line.trim_start_matches(['|', ' ']);
        if let Some(rest) = t.strip_prefix("TOTAL ENERGY") {
            for tok in rest.split_whitespace() {
                if let Some(f) = parse_fortran_float(tok) {
                    return Ok(f);
                }
            }
        }
    }
    Err(ExtError::Parse {
        what: "xtb stdout",
        message: "no '| TOTAL ENERGY … Eh |' line found".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRADIENT_FIXTURE: &str = "\
$grad
 cycle =      1    SCF energy =    -1.0599932109   |dE/dxyz| =  0.012345
      0.00000000000000      0.00000000000000     -0.69853235004775      h
      0.00000000000000      0.00000000000000      0.69853235004775      h
      0.00000000000000      0.00000000000000     -0.12000000000000E-01
      0.00000000000000      0.00000000000000      0.12000000000000E-01
$end
";

    const XTBOPT_FIXTURE: &str = "\
2
 energy: -1.067568856769 gnorm: 0.000034567890 xtb: 6.6.1 (8d0f1dd)
H         0.00000000000000    0.00000000000000   -0.37094772930000
H         0.00000000000000    0.00000000000000    0.37094772930000
";

    const XTBOUT_JSON_FIXTURE: &str = r#"{
  "total energy": -5.070387306651,
  "HOMO-LUMO gap/eV": 13.486,
  "electronic energy": -5.184,
  "dipole": [0.0, 0.0, 0.0]
}"#;

    const STDOUT_FIXTURE: &str = "\
          :::::::::::::::::::::::::::::::::::::::::::::::::::::
          ::                     SUMMARY                     ::
          :::::::::::::::::::::::::::::::::::::::::::::::::::::
          | TOTAL ENERGY               -5.070387306651 Eh   |
          | GRADIENT NORM               0.000339208090 Eh/α |
normal termination of xtb
";

    #[test]
    fn args_gfn2_with_charge_uhf_alpb() {
        let input = XtbInput {
            method: XtbMethod::Gfn2Xtb,
            charge: -1,
            n_unpaired: 2,
            alpb: Some("water".into()),
        };
        let args = xtb_args(&input, "mol.xyz", XtbRun::Gradient);
        assert_eq!(
            args,
            vec![
                "mol.xyz", "--gfn", "2", "--chrg", "-1", "--uhf", "2", "--alpb", "water", "--grad",
                "--json"
            ]
        );
    }

    #[test]
    fn args_gfnff_opt() {
        let input = XtbInput {
            method: XtbMethod::GfnFf,
            charge: 0,
            n_unpaired: 0,
            alpb: None,
        };
        let args = xtb_args(&input, "m.xyz", XtbRun::Opt);
        assert_eq!(
            args,
            vec![
                "m.xyz", "--gfnff", "--chrg", "0", "--uhf", "0", "--opt", "--json"
            ]
        );
    }

    #[test]
    fn method_keyword_round_trip() {
        assert_eq!(
            XtbMethod::from_keyword("gfn2-xtb"),
            Some(XtbMethod::Gfn2Xtb)
        );
        assert_eq!(XtbMethod::from_keyword("gfnff"), Some(XtbMethod::GfnFf));
        assert_eq!(XtbMethod::from_keyword("b3lyp"), None);
    }

    #[test]
    fn parse_gradient_fixture() {
        let g = parse_turbomole_gradient(GRADIENT_FIXTURE, 2).unwrap();
        assert!((g.energy - (-1.0599932109)).abs() < 1e-10);
        assert_eq!(g.gradient.len(), 2);
        assert!((g.gradient[0][2] - (-0.012)).abs() < 1e-12);
        assert!((g.gradient[1][2] - 0.012).abs() < 1e-12);
        assert!(g.gradient[0][0].abs() < 1e-15);
    }

    #[test]
    fn parse_gradient_wrong_atom_count_errors() {
        assert!(parse_turbomole_gradient(GRADIENT_FIXTURE, 3).is_err());
        assert!(parse_turbomole_gradient("no block here", 1).is_err());
    }

    #[test]
    fn parse_xtbopt_fixture() {
        let (mol, e) = parse_xtbopt_xyz(XTBOPT_FIXTURE, 0, 1).unwrap();
        assert_eq!(mol.len(), 2);
        assert_eq!(mol.atoms[0].element.symbol(), "H");
        assert!((e.unwrap() - (-1.067568856769)).abs() < 1e-12);
    }

    #[test]
    fn parse_json_fixture() {
        let e = parse_json_energy(XTBOUT_JSON_FIXTURE).unwrap();
        assert!((e - (-5.070387306651)).abs() < 1e-12);
        assert!(parse_json_energy("{}").is_err());
    }

    #[test]
    fn parse_stdout_fixture() {
        let e = parse_energy_stdout(STDOUT_FIXTURE).unwrap();
        assert!((e - (-5.070387306651)).abs() < 1e-12);
        assert!(parse_energy_stdout("nothing here").is_err());
    }

    #[test]
    fn fortran_float_d_exponent() {
        assert!((parse_fortran_float("1.5D-02").unwrap() - 0.015).abs() < 1e-15);
        assert!((parse_fortran_float("-3.0E0").unwrap() + 3.0).abs() < 1e-15);
    }

    #[test]
    #[ignore = "requires the external xtb binary; run with --include-ignored"]
    fn real_xtb_single_point_h2() {
        if find_xtb().is_err() {
            eprintln!("xtb not found — skipping real-subprocess test");
            return;
        }
        let mol = Molecule::from_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let input = XtbInput::from_molecule(XtbMethod::Gfn2Xtb, &mol, None);
        let workdir = std::env::temp_dir().join("hartree_xtb_realtest");
        let res = run(&mol, &input, XtbRun::Gradient, &workdir).expect("xtb run");
        assert!(res.energy < 0.0, "H2 GFN2 energy should be negative");
        assert!(res.gradient.is_some());
    }

    #[test]
    fn find_xtb_absent_gives_named_error() {
        unsafe {
            std::env::set_var(XTB_PATH_ENV, "/nonexistent/xtb/binary/xyzzy");
        }
        match find_xtb() {
            Err(ExtError::BinaryNotFound {
                program, env_var, ..
            }) => {
                assert_eq!(program, "xtb");
                assert_eq!(env_var, XTB_PATH_ENV);
            }
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => { /* xtb is on PATH on this machine */ }
        }
        unsafe {
            std::env::remove_var(XTB_PATH_ENV);
        }
    }
}
