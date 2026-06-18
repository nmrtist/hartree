use std::collections::HashMap;
use std::time::Instant;

use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc, MolecularGrid, XcContributor};
use hartree::integrals::ConventionalProvider;
use hartree::scf::{ScfOptions, run_rhf};
use serde::Deserialize;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");

#[derive(Deserialize)]
struct Geometries {
    molecules: HashMap<String, GeomEntry>,
}

#[derive(Deserialize)]
struct GeomEntry {
    charge: i32,
    multiplicity: u32,
    atoms: Vec<(String, f64, f64, f64)>,
}

impl GeomEntry {
    fn molecule(&self) -> Molecule {
        let atoms = self
            .atoms
            .iter()
            .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
            .collect();
        Molecule::new(atoms, self.charge, self.multiplicity)
    }
}

fn geometries() -> Geometries {
    serde_json::from_str(GEOMETRIES_JSON).expect("parse geometries.json")
}

fn rhf_alpha_density(mol: &Molecule, basis: &str) -> (Vec<f64>, usize) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let n = ao.n_ao();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let provider = ConventionalProvider::new(ao.into_integral(), charges);
    let scf = run_rhf(
        &provider,
        mol.n_electrons() as usize,
        mol.nuclear_repulsion(),
        &ScfOptions {
            energy_tol: 1e-10,
            error_tol: 1e-8,
            ..ScfOptions::default()
        },
    )
    .unwrap();
    assert!(scf.converged, "RHF reference did not converge");
    (scf.density_alpha, n)
}

const CASES: &[(&str, &str)] = &[
    ("water", "cc-pvdz"),
    ("water", "def2-svp"),
    ("ethylene", "cc-pvdz"),
    ("ethylene", "def2-svp"),
];

fn bench_grid_and_vxc() {
    let geoms = geometries();
    let spec = FunctionalSpec::parse("pbe").unwrap();

    println!(
        "\n{:<22} {:>6} {:>9} {:>12} {:>12}",
        "case", "level", "points", "grid build", "V_xc eval"
    );
    for &(name, basis) in CASES {
        let mol = geoms.molecules[name].molecule();
        let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
        let (d_a, n) = rhf_alpha_density(&mol, basis);

        for level in [3usize, 4] {
            let t0 = Instant::now();
            let grid = MolecularGrid::build(&mol, level).unwrap();
            let t_grid = t0.elapsed().as_secs_f64();
            let npts = grid.points.len();

            let xc = GridXc::new(&mol, &ao, &spec, level).unwrap();
            let _ = xc.eval(&d_a, &d_a, n, true);
            let reps = 3;
            let t1 = Instant::now();
            for _ in 0..reps {
                let _ = xc.eval(&d_a, &d_a, n, true);
            }
            let t_eval = t1.elapsed().as_secs_f64() / reps as f64;

            println!(
                "{:<22} {:>6} {:>9} {:>10.1} ms {:>10.1} ms",
                format!("{name}/{basis}"),
                level,
                npts,
                t_grid * 1e3,
                t_eval * 1e3,
            );
            assert!(t_grid.is_finite() && t_grid > 0.0);
            assert!(t_eval.is_finite() && t_eval > 0.0);
        }
    }
    println!();
}

fn bench_vxc_rayon_scaling() {
    let geoms = geometries();
    let mol = geoms.molecules["ethylene"].molecule();
    let basis = "cc-pvdz";
    let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
    let (d_a, n) = rhf_alpha_density(&mol, basis);
    let spec = FunctionalSpec::parse("pbe").unwrap();
    let xc = GridXc::new(&mol, &ao, &spec, 4).unwrap();
    let _ = xc.eval(&d_a, &d_a, n, true); // warm up

    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let reps = 3;
    let t1 = Instant::now();
    serial.install(|| {
        for _ in 0..reps {
            let _ = xc.eval(&d_a, &d_a, n, true);
        }
    });
    let one = t1.elapsed().as_secs_f64() / reps as f64;

    let tn = Instant::now();
    for _ in 0..reps {
        let _ = xc.eval(&d_a, &d_a, n, true);
    }
    let many = tn.elapsed().as_secs_f64() / reps as f64;

    let threads = rayon::current_num_threads();
    println!(
        "\nV_xc eval ethylene/cc-pvdz/pbe L4 ({} grid points): 1 thread {:.1} ms -> {} threads {:.1} ms  (speedup {:.1}x)\n",
        xc.n_points(),
        one * 1e3,
        threads,
        many * 1e3,
        one / many,
    );
    if threads > 1 {
        assert!(
            many <= one * 1.5,
            "V_xc eval anti-scaled: {:.1} ms on {threads} threads vs {:.1} ms serial",
            many * 1e3,
            one * 1e3
        );
    }
}

fn main() {
    bench_grid_and_vxc();
    bench_vxc_rayon_scaling();
}
