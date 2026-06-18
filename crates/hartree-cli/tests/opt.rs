use std::collections::HashMap;

use hartree::core::{Atom, Element, Molecule};
use hartree::dft::FunctionalSpec;
use hartree::opt::{OptError, OptOptions, Surface, optimize};
use hartree::scf::Reference;
use hartree::{HfSurface, optimize_geometry, optimize_geometry_dft};
use serde::Deserialize;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");
const OPT_REFERENCES_JSON: &str = include_str!("../../../tests/ref/opt_references.json");

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

#[derive(Deserialize)]
struct OptReferences {
    entries: Vec<OptRefEntry>,
}

#[derive(Deserialize)]
struct OptRefEntry {
    molecule: String,
    basis: String,
    method: String,
    energy: f64,
    geometry_bohr: Vec<(String, f64, f64, f64)>,
}

fn start_molecule(g: &GeomEntry) -> Molecule {
    let atoms = g
        .atoms
        .iter()
        .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
        .collect();
    Molecule::new(atoms, g.charge, g.multiplicity)
}

fn ref_positions(entry: &OptRefEntry) -> Vec<[f64; 3]> {
    entry
        .geometry_bohr
        .iter()
        .map(|(_, x, y, z)| [*x, *y, *z])
        .collect()
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn angle_deg(i: [f64; 3], k: [f64; 3], j: [f64; 3]) -> f64 {
    let u = [i[0] - k[0], i[1] - k[1], i[2] - k[2]];
    let v = [j[0] - k[0], j[1] - k[1], j[2] - k[2]];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

type Bonds = Vec<(usize, usize)>;
type Angles = Vec<(usize, usize, usize)>;

fn internals_for(molecule: &str) -> (Bonds, Angles) {
    match molecule {
        "h2" => (vec![(0, 1)], vec![]),
        "water" => (vec![(0, 1), (0, 2)], vec![(1, 0, 2)]),
        other => panic!("no internal-coordinate spec for {other}"),
    }
}

fn check(entry: &OptRefEntry) {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let start = start_molecule(&geoms.molecules[&entry.molecule]);
    assert_eq!(entry.method, "rhf");

    let result = optimize_geometry(&start, &entry.basis, Reference::Rhf, &OptOptions::default())
        .unwrap_or_else(|e| panic!("{}/{}: {e}", entry.molecule, entry.basis));
    assert!(
        result.converged,
        "{}/{} optimization did not converge in {} steps",
        entry.molecule, entry.basis, result.iterations
    );

    let de = result.energy - entry.energy;
    assert!(
        de.abs() < 1e-6,
        "{}/{} energy: hartree {:.10} vs ORCA {:.10} (Δ = {:.2e})",
        entry.molecule,
        entry.basis,
        result.energy,
        entry.energy,
        de
    );

    let hartree = &result.positions;
    let orca = ref_positions(entry);
    let (bonds, angles) = internals_for(&entry.molecule);
    for (i, j) in bonds {
        let dr = distance(hartree[i], hartree[j]) - distance(orca[i], orca[j]);
        assert!(
            dr.abs() < 1e-4,
            "{}/{} bond {i}-{j}: Δr = {dr:.2e} bohr (hartree {:.6}, ORCA {:.6})",
            entry.molecule,
            entry.basis,
            distance(hartree[i], hartree[j]),
            distance(orca[i], orca[j]),
        );
    }
    for (i, k, j) in angles {
        let dth =
            angle_deg(hartree[i], hartree[k], hartree[j]) - angle_deg(orca[i], orca[k], orca[j]);
        assert!(
            dth.abs() < 0.01,
            "{}/{} angle {i}-{k}-{j}: Δθ = {dth:.2e}° (hartree {:.4}, ORCA {:.4})",
            entry.molecule,
            entry.basis,
            angle_deg(hartree[i], hartree[k], hartree[j]),
            angle_deg(orca[i], orca[k], orca[j]),
        );
    }

    eprintln!(
        "{}/{} opt OK: ΔE = {de:.2e} Eh, geometry matched",
        entry.molecule, entry.basis
    );
}

fn references() -> Vec<OptRefEntry> {
    let refs: OptReferences = serde_json::from_str(OPT_REFERENCES_JSON).unwrap();
    refs.entries
}

#[test]
fn optimizes_to_orca_references_fast() {
    let mut checked = 0;
    for entry in references() {
        let slow = entry.molecule == "water" && entry.basis == "cc-pvdz";
        if slow {
            continue;
        }
        check(&entry);
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected ≥3 fast opt references, checked {checked}"
    );
}

struct FdOnly(HfSurface);
impl Surface for FdOnly {
    fn energy(&mut self, positions: &[[f64; 3]]) -> Result<f64, OptError> {
        self.0.energy(positions)
    }
    fn analytic_gradient(&mut self, _: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        None
    }
}

#[test]
fn fd_driven_optimization_matches_orca() {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let start = start_molecule(&geoms.molecules["h2"]);
    let reference = references()
        .into_iter()
        .find(|e| e.molecule == "h2" && e.basis == "sto-3g")
        .unwrap();

    let mut surface = FdOnly(HfSurface::new(&start, "sto-3g", Reference::Rhf).unwrap());
    let result = optimize(&start, &mut surface, &OptOptions::default()).unwrap();
    assert!(result.converged, "FD-driven opt did not converge");

    let hartree = distance(result.positions[0], result.positions[1]);
    let orca = distance(ref_positions(&reference)[0], ref_positions(&reference)[1]);
    assert!(
        (hartree - orca).abs() < 1e-4,
        "FD-driven H2/sto-3g bond {hartree:.6} vs ORCA {orca:.6}"
    );
    assert!(
        (result.energy - reference.energy).abs() < 1e-6,
        "FD-driven energy mismatch"
    );
}

#[test]
fn rohf_surface_has_no_analytic_gradient() {
    let oh = Molecule::new(
        vec![
            Atom::new(Element::from_symbol("O").unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_symbol("H").unwrap(), [0.0, 0.0, 1.83]),
        ],
        0,
        2,
    );
    let positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]];

    let mut rohf = HfSurface::new(&oh, "sto-3g", Reference::Rohf).unwrap();
    assert!(
        rohf.analytic_gradient(&positions).is_none(),
        "ROHF must have no analytic gradient (routes to FD)"
    );

    let mut uhf = HfSurface::new(&oh, "sto-3g", Reference::Uhf).unwrap();
    assert!(
        uhf.analytic_gradient(&positions).is_some(),
        "UHF should offer an analytic gradient"
    );
}

#[test]
#[ignore = "slow (many min): toy-size cc-pVDZ eri_grad per step; run with --ignored"]
fn optimizes_water_ccpvdz_flagship() {
    let entry = references()
        .into_iter()
        .find(|e| e.molecule == "water" && e.basis == "cc-pvdz")
        .expect("water/cc-pvdz opt reference");
    check(&entry);
}

#[test]
#[ignore = "slow: FD-gradient KS optimization runs many SCFs per step; run with --ignored"]
fn optimizes_water_pbe_fd() {
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).unwrap();
    let start = start_molecule(&geoms.molecules["water"]);

    let result = optimize_geometry_dft(
        &start,
        "6-31g",
        Reference::Rhf,
        FunctionalSpec::parse("pbe").unwrap(),
        3,
        &OptOptions::default(),
    )
    .expect("water/pbe/6-31g FD optimization");
    assert!(
        result.converged,
        "FD KS optimization did not converge in {} steps",
        result.iterations
    );

    assert!(
        result.energy < result.history.first().unwrap().energy,
        "optimized energy {} not below the starting energy {}",
        result.energy,
        result.history.first().unwrap().energy
    );

    let p = &result.positions;
    let r1 = distance(p[0], p[1]);
    let r2 = distance(p[0], p[2]);
    let theta = angle_deg(p[1], p[0], p[2]);
    eprintln!(
        "water/pbe/6-31g FD opt: O–H = {r1:.4}/{r2:.4} bohr, H–O–H = {theta:.2}°, E = {:.8}",
        result.energy
    );
    for r in [r1, r2] {
        assert!(
            (1.7..=2.0).contains(&r),
            "O–H bond {r:.4} bohr is not a sensible water minimum"
        );
    }
    assert!(
        (95.0..=115.0).contains(&theta),
        "H–O–H angle {theta:.2}° is not a sensible water minimum"
    );
}
