use hartree::basis::{GthBasisSet, GthSet};
use hartree::core::Molecule;
use hartree::core::units::ANGSTROM_TO_BOHR;
use hartree::periodic::{Cell, KPoint, MonkhorstPack, PeriodicScfOptions};
use hartree::{PeriodicFunctional, PeriodicJob, run_periodic};

use crate::take;

pub fn is_periodic(args: &[String]) -> bool {
    args.iter().any(|a| {
        matches!(
            a.as_str(),
            "--cell" | "--kpoints" | "--kpts" | "--cutoff" | "--e-cut"
        )
    })
}

pub fn run_periodic_cli(args: &[String]) -> Result<bool, String> {
    let mut xyz_path: Option<String> = None;
    let mut basis = String::from("DZVP-GTH-PADE");
    let mut cell_spec: Option<String> = None;
    let mut kpoints_spec: Option<String> = None;
    let mut cutoff: Option<f64> = None;
    let mut xc = String::from("pade");
    let mut pseudo_file: Option<String> = None;
    let mut basis_file: Option<String> = None;
    let mut do_forces = false;
    let mut do_stress = false;
    let mut max_iter: Option<usize> = None;
    let mut mixing: Option<f64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--basis" => basis = take(args, &mut i, "--basis")?,
            "--cell" => {
                if let Some(v) = args.get(i + 1)
                    && cell_value_looks_like_spec(v)
                {
                    if !v.eq_ignore_ascii_case("file") {
                        cell_spec = Some(v.clone());
                    }
                    i += 1;
                }
            }
            "--kpoints" | "--kpts" => kpoints_spec = Some(take(args, &mut i, "--kpoints")?),
            "--cutoff" | "--e-cut" => {
                cutoff = Some(
                    take(args, &mut i, "--cutoff")?
                        .parse()
                        .map_err(|_| "--cutoff must be a number (hartree)".to_string())?,
                );
            }
            "--xc" => xc = take(args, &mut i, "--xc")?,
            "--pseudo" => pseudo_file = Some(take(args, &mut i, "--pseudo")?),
            "--basis-file" => basis_file = Some(take(args, &mut i, "--basis-file")?),
            "--forces" => do_forces = true,
            "--stress" => do_stress = true,
            "--max-iter" => {
                max_iter = Some(
                    take(args, &mut i, "--max-iter")?
                        .parse()
                        .map_err(|_| "--max-iter must be a positive integer".to_string())?,
                );
            }
            "--mixing" => {
                mixing = Some(
                    take(args, &mut i, "--mixing")?
                        .parse()
                        .map_err(|_| "--mixing must be a number in (0, 1]".to_string())?,
                );
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown periodic option {other} (see --help)"));
            }
            path => xyz_path = Some(path.to_string()),
        }
        i += 1;
    }

    let xyz_path = xyz_path.ok_or("missing XYZ file argument (try --help)")?;
    let xyz = std::fs::read_to_string(&xyz_path).map_err(|e| format!("reading {xyz_path}: {e}"))?;
    let (molecule, lattice_file) =
        Molecule::from_xyz_with_lattice(&xyz).map_err(|e| e.to_string())?;

    let cell = if let Some(spec) = &cell_spec {
        parse_cell_spec(spec)?
    } else if let Some(l) = lattice_file {
        Cell::from_vectors(l[0], l[1], l[2]).map_err(|e| e.to_string())?
    } else {
        return Err(
            "no unit cell: pass --cell \"cubic <a>\" / \"<9 numbers>\" (ångström) or give a \
             Lattice=\"…\" line in the extended-XYZ file"
                .into(),
        );
    };

    let basis_set = match &basis_file {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("reading {p}: {e}"))?;
            GthBasisSet::from_text("user", &text).map_err(|e| e.to_string())?
        }
        None => GthBasisSet::load_pade().map_err(|e| e.to_string())?,
    };
    let pseudo = match &pseudo_file {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("reading {p}: {e}"))?;
            GthSet::from_text("user", &text).map_err(|e| e.to_string())?
        }
        None => GthSet::load_pade().map_err(|e| e.to_string())?,
    };

    let kpoints = match &kpoints_spec {
        Some(s) => parse_kpoints_spec(s)?,
        None => vec![KPoint::gamma()],
    };

    let functional = PeriodicFunctional::from_name(&xc)?;

    let mut options = PeriodicScfOptions::default();
    if let Some(c) = cutoff {
        options.e_cut = c;
    }
    if let Some(m) = max_iter {
        options.max_iter = m;
    }
    if let Some(m) = mixing {
        options.mixing = m;
    }

    let job = PeriodicJob {
        molecule,
        cell,
        kpoints,
        basis_name: basis,
        basis_set,
        pseudo,
        functional,
        options,
        forces: do_forces,
        stress: do_stress,
    };

    let result = run_periodic(&job)?;
    report(&job, &result);
    Ok(result.scf.converged)
}

fn cell_value_looks_like_spec(v: &str) -> bool {
    v.eq_ignore_ascii_case("cubic") || v.eq_ignore_ascii_case("file") || {
        let mut chars = v.chars();
        match chars.next() {
            Some(c) if c.is_ascii_digit() || c == '.' => true,
            Some('+' | '-') => chars.next().is_some_and(|c| c.is_ascii_digit() || c == '.'),
            _ => false,
        }
    }
}

fn parse_cell_spec(spec: &str) -> Result<Cell, String> {
    let toks: Vec<&str> = spec
        .split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect();
    let s = ANGSTROM_TO_BOHR;
    if toks
        .first()
        .is_some_and(|t| t.eq_ignore_ascii_case("cubic"))
    {
        let a: f64 = toks
            .get(1)
            .ok_or("--cell cubic needs a lattice constant (ångström)")?
            .parse()
            .map_err(|_| "--cell cubic <a>: a must be a number".to_string())?;
        return Cell::cubic(a * s).map_err(|e| e.to_string());
    }
    let v: Vec<f64> = toks
        .iter()
        .map(|t| {
            t.parse::<f64>()
                .map_err(|_| format!("--cell: non-numeric token {t:?}"))
        })
        .collect::<Result<_, _>>()?;
    match v.len() {
        1 => Cell::cubic(v[0] * s).map_err(|e| e.to_string()),
        3 => Cell::from_vectors(
            [v[0] * s, 0.0, 0.0],
            [0.0, v[1] * s, 0.0],
            [0.0, 0.0, v[2] * s],
        )
        .map_err(|e| e.to_string()),
        6 => cell_from_parameters(v[0], v[1], v[2], v[3], v[4], v[5]),
        9 => Cell::from_vectors(
            [v[0] * s, v[1] * s, v[2] * s],
            [v[3] * s, v[4] * s, v[5] * s],
            [v[6] * s, v[7] * s, v[8] * s],
        )
        .map_err(|e| e.to_string()),
        n => Err(format!(
            "--cell expects 'cubic <a>' or 1/3/6/9 numbers (got {n}); 6 = a b c α β γ, 9 = three row vectors"
        )),
    }
}

fn cell_from_parameters(
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> Result<Cell, String> {
    let (al, be, ga) = (alpha.to_radians(), beta.to_radians(), gamma.to_radians());
    let s = ANGSTROM_TO_BOHR;
    let a1 = [a * s, 0.0, 0.0];
    let a2 = [b * s * ga.cos(), b * s * ga.sin(), 0.0];
    let cx = c * be.cos();
    let cy = c * (al.cos() - be.cos() * ga.cos()) / ga.sin();
    let cz2 = c * c - cx * cx - cy * cy;
    if cz2 <= 0.0 {
        return Err("--cell: non-physical a, b, c, α, β, γ (degenerate lattice)".into());
    }
    let a3 = [cx * s, cy * s, cz2.sqrt() * s];
    Cell::from_vectors(a1, a2, a3).map_err(|e| e.to_string())
}

fn parse_kpoints_spec(spec: &str) -> Result<Vec<KPoint>, String> {
    let s = spec.trim();
    if s.eq_ignore_ascii_case("gamma") || s.eq_ignore_ascii_case("g") {
        return Ok(vec![KPoint::gamma()]);
    }
    let mesh: Vec<usize> = s
        .split([',', ' ', '\t', 'x', 'X'])
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| format!("--kpoints: non-integer mesh token {t:?}"))
        })
        .collect::<Result<_, _>>()?;
    if mesh.len() != 3 {
        return Err(format!(
            "--kpoints needs 3 integers (n1 n2 n3) or 'gamma' (got {})",
            mesh.len()
        ));
    }
    if mesh.contains(&0) {
        return Err("--kpoints mesh dimensions must be positive".into());
    }
    MonkhorstPack::regular([mesh[0], mesh[1], mesh[2]]).map_err(|e| e.to_string())
}

fn report(job: &PeriodicJob, result: &hartree::PeriodicJobResult) {
    let natom = job.molecule.len();
    let scf = &result.scf;
    let c = &scf.components;
    println!(
        "== periodic GPW ({:?} XC, {} basis) ==",
        job.functional, job.basis_name
    );
    println!(
        "  cell volume   {:.4} bohr^3   atoms {natom}   k-points {}",
        job.cell.volume(),
        job.kpoints.len()
    );
    println!(
        "  SCF           {} in {} iters   N(grid) = {:.4}",
        if scf.converged {
            "converged"
        } else {
            "NOT converged"
        },
        scf.iterations,
        scf.n_elec_grid
    );
    println!("  E_total       {:.8} Ha", scf.energy);
    if natom > 0 {
        println!("  E/atom        {:.8} Ha", scf.energy / natom as f64);
    }
    println!("  components (Ha):");
    println!("    E_kin       {:.8}", c.e_kin);
    println!("    E_hartree   {:.8}", c.e_hartree);
    println!("    E_xc        {:.8}", c.e_xc);
    println!("    E_local_sr  {:.8}", c.e_local_sr);
    println!("    E_nonlocal  {:.8}", c.e_nonlocal);
    println!("    E_self      {:.8}", c.e_self);
    println!("    E_overlap   {:.8}", c.e_overlap);

    if let Some(forces) = &result.forces {
        println!("  forces (Ha/bohr):");
        for (i, f) in forces.iter().enumerate() {
            println!("    atom {i:3}   {:+.6}  {:+.6}  {:+.6}", f[0], f[1], f[2]);
        }
    }
    if let Some(sigma) = &result.stress {
        println!("  stress (Ha/bohr^3):");
        for row in sigma {
            println!("    {:+.6}  {:+.6}  {:+.6}", row[0], row[1], row[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| (*a).to_string()).collect()
    }

    #[test]
    fn detects_periodic_flags() {
        assert!(is_periodic(&s(&["foo.xyz", "--cell"])));
        assert!(is_periodic(&s(&["--cutoff", "100"])));
        assert!(is_periodic(&s(&["--kpoints", "2 2 2"])));
        assert!(!is_periodic(&s(&["foo.xyz", "--basis", "sto-3g"])));
    }

    #[test]
    fn cell_spec_forms_agree() {
        let a = 5.0 * ANGSTROM_TO_BOHR;
        let vol = a * a * a;
        for spec in [
            "cubic 5.0",
            "5.0",
            "5 5 5",
            "5 0 0 0 5 0 0 0 5",
            "5 5 5 90 90 90",
        ] {
            let c = parse_cell_spec(spec).unwrap();
            assert!(
                (c.volume() - vol).abs() < 1e-5,
                "{spec}: vol {}",
                c.volume()
            );
        }
        assert!(parse_cell_spec("5 5").is_err());
        assert!(parse_cell_spec("cubic").is_err());
    }

    #[test]
    fn kpoints_spec_forms() {
        assert_eq!(parse_kpoints_spec("gamma").unwrap().len(), 1);
        assert_eq!(parse_kpoints_spec("1x1x1").unwrap().len(), 1);
        assert_eq!(parse_kpoints_spec("2 2 2").unwrap().len(), 8);
        assert!(parse_kpoints_spec("2 2").is_err());
        assert!(parse_kpoints_spec("2 0 2").is_err());
    }

    #[test]
    fn cell_value_detection() {
        for ok in ["cubic", "file", "5.0", "-1", "+2", ".5"] {
            assert!(cell_value_looks_like_spec(ok), "{ok}");
        }
        for no in ["si.xyz", "input", "foo"] {
            assert!(!cell_value_looks_like_spec(no), "{no}");
        }
        assert!(!cell_value_looks_like_spec("--kpoints"));
    }
}
