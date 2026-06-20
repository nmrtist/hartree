//! Tests for transition-state guess construction (the geometric [`build_ts_guess`] and
//! the energy-peaked [`build_ts_guess_scanned`]). Split out of `guess.rs` to keep it
//! under the module-size limit; as a child module it still sees the parent's private
//! items through `super`.

use super::*;
use crate::core::{Atom, Element};
use crate::opt::{OptError, Surface};

fn atom(z: u32, p: [f64; 3]) -> Atom {
    Atom::new(Element::from_z(z).unwrap(), p)
}

fn ethane() -> Molecule {
    let cc = 2.9;
    let ch = 2.06;
    Molecule::new(
        vec![
            atom(6, [0.0, 0.0, 0.0]),
            atom(6, [0.0, 0.0, cc]),
            atom(1, [ch, 0.0, -0.7]),
            atom(1, [-ch * 0.5, ch * 0.87, -0.7]),
            atom(1, [-ch * 0.5, -ch * 0.87, -0.7]),
            atom(1, [ch * 0.5, ch * 0.87, cc + 0.7]),
            atom(1, [ch * 0.5, -ch * 0.87, cc + 0.7]),
            atom(1, [-ch, 0.0, cc + 0.7]),
        ],
        0,
        1,
    )
}

fn methyl(center: [f64; 3]) -> Molecule {
    let ch = 2.06;
    Molecule::new(
        vec![
            atom(6, center),
            atom(1, [center[0] + ch, center[1], center[2]]),
            atom(1, [center[0] - ch * 0.5, center[1] + ch * 0.87, center[2]]),
            atom(1, [center[0] - ch * 0.5, center[1] - ch * 0.87, center[2]]),
        ],
        0,
        2,
    )
}

#[test]
fn rejects_mismatched_composition() {
    let product = ethane();
    let frag = methyl([0.0, 0.0, 0.0]);
    let err = build_ts_guess(&[frag], &product, &GuessOptions::default()).unwrap_err();
    assert!(matches!(err, GuessError::AtomCountMismatch { .. }));
}

#[test]
fn maps_atoms_and_finds_forming_bond() {
    let product = ethane();
    let frag_a = methyl([0.0, 0.0, 0.0]);
    let frag_b = methyl([10.0, 0.0, 0.0]);
    let guess = build_ts_guess(&[frag_a, frag_b], &product, &GuessOptions::default()).unwrap();

    let mut seen = vec![false; product.len()];
    for (r, &p) in guess.atom_map.iter().enumerate() {
        assert!(!seen[p], "product atom {p} mapped twice");
        seen[p] = true;
        assert_eq!(
            guess.molecule.atoms[r].element.z(),
            product.atoms[p].element.z()
        );
    }

    let forming: Vec<&ReactionBond> = guess
        .reaction_coordinate
        .iter()
        .filter(|b| b.kind == BondChange::Forming)
        .collect();
    assert_eq!(forming.len(), 1, "expected one forming bond");
    let (a, b) = forming[0].atoms;
    assert_eq!(guess.molecule.atoms[a].element.z(), 6);
    assert_eq!(guess.molecule.atoms[b].element.z(), 6);

    let cc = distance(
        guess.molecule.atoms[a].position,
        guess.molecule.atoms[b].position,
    );
    assert!(cc > 2.9, "forming C–C should be stretched, got {cc} bohr");
    let mut min_d = f64::INFINITY;
    for i in 0..guess.molecule.len() {
        for j in (i + 1)..guess.molecule.len() {
            min_d = min_d.min(distance(
                guess.molecule.atoms[i].position,
                guess.molecule.atoms[j].position,
            ));
        }
    }
    assert!(min_d > 1.0, "atoms collapsed (min distance {min_d} bohr)");
}

#[test]
fn breaks_cc_bond_and_maps_atoms() {
    // Reactant is one ethane fragment (C–C bonded). The product is the same
    // atoms but with the two methyls pulled ~10 Bohr apart, breaking the C–C.
    let reactant = ethane();
    let mut product_atoms = methyl([0.0, 0.0, 0.0]).atoms;
    product_atoms.extend(methyl([10.0, 0.0, 0.0]).atoms);
    let product = Molecule::new(product_atoms, 0, 1);

    let guess = build_ts_guess(&[reactant], &product, &GuessOptions::default()).unwrap();

    // Map is total + injective + element-respecting.
    let mut seen = vec![false; product.len()];
    for (r, &p) in guess.atom_map.iter().enumerate() {
        assert!(p < product.len(), "reactant atom {r} unassigned (got {p})");
        assert!(!seen[p], "product atom {p} mapped twice");
        seen[p] = true;
        assert_eq!(
            guess.molecule.atoms[r].element.z(),
            product.atoms[p].element.z(),
            "element mismatch at reactant atom {r}"
        );
    }

    // Exactly one breaking bond, between the two carbons, and no forming bond.
    let breaking: Vec<&ReactionBond> = guess
        .reaction_coordinate
        .iter()
        .filter(|b| b.kind == BondChange::Breaking)
        .collect();
    assert_eq!(breaking.len(), 1, "expected one breaking bond");
    let forming = guess
        .reaction_coordinate
        .iter()
        .filter(|b| b.kind == BondChange::Forming)
        .count();
    assert_eq!(forming, 0, "expected no forming bond");
    let (a, b) = breaking[0].atoms;
    assert_eq!(guess.molecule.atoms[a].element.z(), 6);
    assert_eq!(guess.molecule.atoms[b].element.z(), 6);

    // No atoms collapsed onto each other in the assembled guess.
    let mut min_d = f64::INFINITY;
    for i in 0..guess.molecule.len() {
        for j in (i + 1)..guess.molecule.len() {
            min_d = min_d.min(distance(
                guess.molecule.atoms[i].position,
                guess.molecule.atoms[j].position,
            ));
        }
    }
    assert!(min_d > 1.0, "atoms collapsed (min distance {min_d} bohr)");
}

#[test]
fn unimolecular_single_fragment_interpolates() {
    let reactant = methyl([0.0, 0.0, 0.0]);
    let mut product = methyl([0.0, 0.0, 0.0]);
    for a in &mut product.atoms {
        a.position[2] += 0.3;
    }
    let guess = build_ts_guess(&[reactant], &product, &GuessOptions::default()).unwrap();
    assert_eq!(guess.molecule.len(), 4);
    assert!(guess.reaction_coordinate.is_empty());
    // The geometric builder leaves the path tangent unset.
    assert!(guess.reaction_tangent.is_none());
}

#[test]
fn reaction_mode_seed_points_along_forming_bond() {
    let product = ethane();
    let frag_a = methyl([0.0, 0.0, 0.0]);
    let frag_b = methyl([10.0, 0.0, 0.0]);
    let guess = build_ts_guess(&[frag_a, frag_b], &product, &GuessOptions::default()).unwrap();

    let forming: Vec<&ReactionBond> = guess
        .reaction_coordinate
        .iter()
        .filter(|b| b.kind == BondChange::Forming)
        .collect();
    assert_eq!(forming.len(), 1);
    let (i, j) = forming[0].atoms;

    let seed = guess
        .reaction_mode_seed()
        .expect("a forming bond yields a seed");
    assert_eq!(seed.len(), guess.molecule.len());

    // The seed is the unit C–C axis carried antiparallel by the two carbons.
    let p = &guess.molecule.atoms;
    let d = [
        p[j].position[0] - p[i].position[0],
        p[j].position[1] - p[i].position[1],
        p[j].position[2] - p[i].position[2],
    ];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let axis = [d[0] / n, d[1] / n, d[2] / n];
    let oi: f64 = (0..3).map(|c| seed[i][c] * axis[c]).sum();
    let oj: f64 = (0..3).map(|c| seed[j][c] * axis[c]).sum();
    assert!(oi.abs() > 0.5 && oj.abs() > 0.5, "carbons carry the seed");
    assert!(oi * oj < 0.0, "carbons move antiparallel along the bond");

    // The seed is normalized, and the spectator hydrogens carry nothing.
    let total: f64 = seed.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
    assert!((total - 1.0).abs() < 1e-9, "seed not normalized ({total})");
    for (a, v) in seed.iter().enumerate() {
        if a != i && a != j {
            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(mag < 1e-9, "spectator atom {a} carries seed {mag}");
        }
    }
}

#[test]
fn reaction_mode_seed_none_without_reaction_bonds() {
    let reactant = methyl([0.0, 0.0, 0.0]);
    let mut product = methyl([0.0, 0.0, 0.0]);
    for a in &mut product.atoms {
        a.position[2] += 0.3;
    }
    let guess = build_ts_guess(&[reactant], &product, &GuessOptions::default()).unwrap();
    assert!(guess.reaction_coordinate.is_empty());
    assert!(guess.reaction_mode_seed().is_none());
}

// ---------------------------------------------------------------------------
// Energy-peaked scan (`build_ts_guess_scanned`).
// ---------------------------------------------------------------------------

/// A surface whose energy is a downward parabola in the distance between atoms
/// `i` and `j`, peaking at `target`. Energy-only (the scan never asks for a gradient).
struct PairPeak {
    i: usize,
    j: usize,
    target: f64,
}
impl Surface for PairPeak {
    fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
        Ok(-(distance(x[self.i], x[self.j]) - self.target).powi(2))
    }
    fn analytic_gradient(&mut self, _x: &[[f64; 3]]) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        None
    }
}

/// Two H atoms starting 6 Bohr apart (reactant) and ending 2 Bohr apart (product); a
/// single fragment, so the IDPP path is the straight approach and the H–H separation
/// interpolates linearly from 6 to 2.
fn approaching_pair() -> (Molecule, Molecule) {
    let reactant = Molecule::new(
        vec![atom(1, [0.0, 0.0, 0.0]), atom(1, [6.0, 0.0, 0.0])],
        0,
        1,
    );
    let product = Molecule::new(
        vec![atom(1, [2.0, 0.0, 0.0]), atom(1, [4.0, 0.0, 0.0])],
        0,
        1,
    );
    (reactant, product)
}

#[test]
fn scan_lands_the_guess_on_the_energy_peak() {
    // Energy peaks where the H–H distance is 4 Bohr, i.e. exactly the path midpoint
    // (d = 6 - 4λ ⇒ d = 4 at λ = 0.5). The scanned guess should sit there.
    let (reactant, product) = approaching_pair();
    let mut surface = PairPeak {
        i: 0,
        j: 1,
        target: 4.0,
    };
    let guess = build_ts_guess_scanned(
        std::slice::from_ref(&reactant),
        &product,
        &mut surface,
        &ScanOptions::default(),
    )
    .expect("scan succeeds");

    assert_eq!(guess.molecule.len(), 2);
    let d = distance(
        guess.molecule.atoms[0].position,
        guess.molecule.atoms[1].position,
    );
    assert!(
        (d - 4.0).abs() < 0.1,
        "scanned guess H–H distance {d:.4} is not at the energy peak (4.0)"
    );
}

#[test]
fn scan_tangent_is_the_path_direction_and_seeds_the_search() {
    let (reactant, product) = approaching_pair();
    let mut surface = PairPeak {
        i: 0,
        j: 1,
        target: 4.0,
    };
    let guess = build_ts_guess_scanned(
        std::slice::from_ref(&reactant),
        &product,
        &mut surface,
        &ScanOptions::default(),
    )
    .unwrap();

    // The scan supplies a path tangent, even though no covalent bond change was
    // detected (the H–H pair is never within the bond cutoff), and the seed comes from
    // it, not from the (empty) bond-vector set.
    assert!(guess.reaction_coordinate.is_empty());
    let tangent = guess
        .reaction_tangent
        .as_ref()
        .expect("scan sets a tangent");
    let seed = guess
        .reaction_mode_seed()
        .expect("tangent seeds the search");
    assert_eq!(seed, *tangent);

    // The tangent is a unit vector along x: atom 0 moves +x, atom 1 moves -x as the
    // pair approaches (closing the separation along the bond axis).
    let norm: f64 = tangent.iter().flatten().map(|c| c * c).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-9, "tangent not normalized ({norm})");
    assert!(
        tangent[0][0] * tangent[1][0] < 0.0,
        "the two atoms should move antiparallel along the bond axis: {tangent:?}"
    );
    assert!(
        tangent[0][1].abs() < 1e-6 && tangent[0][2].abs() < 1e-6,
        "tangent has spurious off-axis components: {tangent:?}"
    );
}

#[test]
fn scan_reuses_the_geometric_assembly() {
    // The scan and the geometric builder share the same atom mapping and reaction
    // coordinate (they differ only in where along the path the guess sits), so a
    // bond-forming case maps identically either way.
    let product = ethane();
    let frag_a = methyl([0.0, 0.0, 0.0]);
    let frag_b = methyl([10.0, 0.0, 0.0]);
    let geometric = build_ts_guess(
        &[frag_a.clone(), frag_b.clone()],
        &product,
        &GuessOptions::default(),
    )
    .unwrap();

    // Peak the surface on the forming C–C pair so the scan has a real barrier to find.
    let forming = geometric
        .reaction_coordinate
        .iter()
        .find(|b| b.kind == BondChange::Forming)
        .expect("a forming bond");
    let mut surface = PairPeak {
        i: forming.atoms.0,
        j: forming.atoms.1,
        target: 0.5 * (forming.reactant_distance + forming.product_distance),
    };
    let scanned = build_ts_guess_scanned(
        &[frag_a, frag_b],
        &product,
        &mut surface,
        &ScanOptions::default(),
    )
    .unwrap();

    assert_eq!(scanned.atom_map, geometric.atom_map);
    assert_eq!(
        scanned.reaction_coordinate.len(),
        geometric.reaction_coordinate.len()
    );
    assert!(scanned.reaction_tangent.is_some());
}

#[test]
fn scan_rejects_too_few_points() {
    let (reactant, product) = approaching_pair();
    let mut surface = PairPeak {
        i: 0,
        j: 1,
        target: 4.0,
    };
    let opts = ScanOptions {
        n_points: 2,
        ..ScanOptions::default()
    };
    let err = build_ts_guess_scanned(
        std::slice::from_ref(&reactant),
        &product,
        &mut surface,
        &opts,
    )
    .unwrap_err();
    assert!(matches!(err, ScanError::TooFewPoints(2)));
}
