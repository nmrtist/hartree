use super::*;
use crate::core::{Atom, Element};

fn h2o() -> Molecule {
    Molecule::new(
        vec![
            Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [1.80, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [-0.45, 1.74, 0.0]),
        ],
        0,
        1,
    )
}

/// A non-planar hydrogen peroxide (Bohr): two O–H bonds bridged by an O–O bond, with
/// the second O–H rotated out of the O–O–H plane so the H–O–O–H torsion is well
/// defined. Exercises dihedral generation and its B-matrix.
fn hooh() -> Vec<[f64; 3]> {
    vec![
        [-0.60, 1.70, 0.00], // H on O(2)
        [0.00, 0.00, 0.00],  // O
        [2.78, 0.00, 0.00],  // O
        [3.38, 1.40, 0.90],  // H on O(3), out of plane
    ]
}

#[test]
fn water_has_two_bonds_and_one_angle() {
    let defs = generate(&h2o());
    let bonds = defs
        .iter()
        .filter(|d| matches!(d, Internal::Bond(..)))
        .count();
    let angles = defs
        .iter()
        .filter(|d| matches!(d, Internal::Angle(..)))
        .count();
    assert_eq!(bonds, 2, "O–H bonds");
    assert_eq!(angles, 1, "H–O–H angle");
    assert!(
        !defs.iter().any(|d| matches!(d, Internal::Dihedral(..))),
        "water has no rotatable bond, so no dihedral"
    );
}

#[test]
fn stretched_diatomic_still_bonds() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 3.0]),
        ],
        0,
        1,
    );
    let defs = generate(&mol);
    assert_eq!(defs, vec![Internal::Bond(0, 1)]);
}

#[test]
fn hydrogen_peroxide_has_one_central_torsion() {
    let x = hooh();
    let mol = Molecule::new(
        x.iter()
            .enumerate()
            .map(|(idx, &p)| {
                let z = if idx == 1 || idx == 2 { 8 } else { 1 };
                Atom::new(Element::from_z(z).unwrap(), p)
            })
            .collect(),
        0,
        1,
    );
    let defs = generate(&mol);
    let dihedrals: Vec<_> = defs
        .iter()
        .filter_map(|d| match d {
            Internal::Dihedral(i, j, k, l) => Some((*i, *j, *k, *l)),
            _ => None,
        })
        .collect();
    assert_eq!(
        dihedrals.len(),
        1,
        "exactly one H–O–O–H torsion, got {dihedrals:?}"
    );
    let (i, j, k, l) = dihedrals[0];
    assert_eq!([j, k], [1, 2], "torsion is about the O–O bond");
    assert_eq!(
        [i.min(l), i.max(l)],
        [0, 3],
        "torsion ends are the two hydrogens"
    );
    // A complete set spans 3N−6 = 6 internal DOF, so the internal frame need not fall
    // back to Cartesian for this pure-torsion molecule.
    assert_eq!(
        internal_rank(&defs, &x),
        6,
        "bonds+angles+torsion span HOOH"
    );
}

#[test]
fn wilson_b_matches_finite_difference() {
    let mol = h2o();
    let defs = generate(&mol);
    let x: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let b = wilson_b(&defs, &x);
    let ndof = 3 * x.len();
    let h = 1e-6;
    for (row, _) in defs.iter().enumerate() {
        for atom in 0..x.len() {
            for c in 0..3 {
                let mut xp = x.clone();
                xp[atom][c] += h;
                let mut xm = x.clone();
                xm[atom][c] -= h;
                let qp = values(&defs, &xp)[row];
                let qm = values(&defs, &xm)[row];
                let fd = (qp - qm) / (2.0 * h);
                let analytic = b[row * ndof + (3 * atom + c)];
                assert!(
                    (fd - analytic).abs() < 1e-7,
                    "B[{row},{atom},{c}] analytic {analytic} vs FD {fd}"
                );
            }
        }
    }
}

/// The dihedral B-matrix row is the analytic gradient of the torsion value, checked
/// against central differences on the non-planar HOOH geometry. Its terminal valence
/// angles are oblique (not 90°), so the central-atom projections `p = (b1·b2)/|b2|²` and
/// `q = (b3·b2)/|b2|²` are both nonzero (~0.22 here) — exercising the `gj`/`gk` projection
/// terms a right-angled geometry (p = q = 0) would leave untested — while staying well
/// away from the linear/branch degeneracies that would spoil the FD.
#[test]
fn dihedral_wilson_b_matches_finite_difference() {
    let x = hooh();
    let defs = vec![Internal::Dihedral(0, 1, 2, 3)];
    let b = wilson_b(&defs, &x);
    let h = 1e-6;
    for atom in 0..x.len() {
        for c in 0..3 {
            let mut xp = x.clone();
            xp[atom][c] += h;
            let mut xm = x.clone();
            xm[atom][c] -= h;
            let fd = (values(&defs, &xp)[0] - values(&defs, &xm)[0]) / (2.0 * h);
            let analytic = b[3 * atom + c];
            assert!(
                (fd - analytic).abs() < 1e-7,
                "B[dihedral,{atom},{c}] analytic {analytic} vs FD {fd}"
            );
        }
    }
}

/// When a torsion's terminal valence angle is near-linear the dihedral plane is
/// undefined; its Wilson row must be left zero (not a divergent ~1/sin coupling), so
/// the locally redundant coordinate drops cleanly into G's null space.
#[test]
fn dihedral_b_row_zeroed_at_near_linear_terminal_angle() {
    // angle i–j–k ≈ 178° (sin ≈ 0.03 < LINEAR_SIN_TOL): the i–j–k plane is degenerate.
    let x = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.03, 0.0],
        [2.0, 1.0, 0.5],
    ];
    let defs = vec![Internal::Dihedral(0, 1, 2, 3)];
    let b = wilson_b(&defs, &x);
    assert!(
        b.iter().all(|&v| v == 0.0),
        "near-linear dihedral row should be all zero, got {b:?}"
    );
}

/// The far terminal angle `j–k–l` going near-linear must zero the row too — the second
/// half of the guard (`nn > LINEAR_SIN_TOL·|b2||b3|`), independent of the `i–j–k` arm. Here
/// `i–j–k` is a clean right angle while `j–k–l ≈ 178°`, so only the `nn` clause fails.
#[test]
fn dihedral_b_row_zeroed_at_near_linear_far_terminal_angle() {
    let x = vec![
        [0.0, 1.0, 0.5],  // i — off-axis so i–j–k is ~90°, not linear
        [0.0, 0.0, 0.0],  // j
        [1.0, 0.0, 0.0],  // k
        [2.0, 0.03, 0.0], // l — angle j–k–l ≈ 178° (sin ≈ 0.03 < LINEAR_SIN_TOL)
    ];
    let defs = vec![Internal::Dihedral(0, 1, 2, 3)];
    let b = wilson_b(&defs, &x);
    assert!(
        b.iter().all(|&v| v == 0.0),
        "near-linear far-terminal-angle dihedral row should be all zero, got {b:?}"
    );
}

/// The co-linear-bend B-matrix rows are the analytic gradients of `(e1+e2)·êₐ` at a
/// near-linear centre (terminal angle ≈ 178°, where the ordinary valence bend is
/// singular), checked element-by-element against central differences. Both reference
/// cardinals (the two axes perpendicular to the chain) are exercised.
#[test]
fn linear_bend_wilson_b_matches_finite_difference() {
    // i–k–j ≈ 178° about the x-axis chain: the two bends live in the y/z cardinals.
    let x = vec![
        [0.0, 0.0, 0.0],  // i
        [1.0, 0.03, 0.0], // k (centre), nudged off the line
        [2.0, 0.0, 0.0],  // j
    ];
    let defs = vec![
        Internal::LinearBend(0, 1, 2, 1),
        Internal::LinearBend(0, 1, 2, 2),
    ];
    let b = wilson_b(&defs, &x);
    let ndof = 3 * x.len();
    let h = 1e-6;
    let mut max_err = 0.0_f64;
    for (row, _) in defs.iter().enumerate() {
        for atom in 0..x.len() {
            for c in 0..3 {
                let mut xp = x.clone();
                xp[atom][c] += h;
                let mut xm = x.clone();
                xm[atom][c] -= h;
                let fd = (values(&defs, &xp)[row] - values(&defs, &xm)[row]) / (2.0 * h);
                let analytic = b[row * ndof + (3 * atom + c)];
                max_err = max_err.max((fd - analytic).abs());
                assert!(
                    (fd - analytic).abs() < 1e-7,
                    "B[linear,{row},{atom},{c}] analytic {analytic} vs FD {fd}"
                );
            }
        }
    }
    // Each row sums to zero (translation invariance).
    for row in 0..defs.len() {
        for c in 0..3 {
            let s: f64 = (0..x.len()).map(|a| b[row * ndof + 3 * a + c]).sum();
            assert!(
                s.abs() < 1e-12,
                "linear-bend row {row} component {c} sum = {s}"
            );
        }
    }
}

/// Propyne CH₃–C≡C–H (Bohr): the C≡C–H and C–C≡C arms are near-linear sp centres whose
/// degenerate valence bends are replaced by co-linear-bend pairs. With those in place the
/// redundant set must span all 3N−6 internal DOF, so the internal frame need not fall
/// back to Cartesian.
fn propyne() -> Molecule {
    // x-axis chain: C(methyl) – C – C – H, with three methyl hydrogens splayed back.
    Molecule::new(
        vec![
            Atom::new(Element::from_z(6).unwrap(), [0.0, 0.0, 0.0]), // 0 methyl C
            Atom::new(Element::from_z(6).unwrap(), [2.84, 0.0, 0.0]), // 1 sp C
            Atom::new(Element::from_z(6).unwrap(), [5.12, 0.0, 0.0]), // 2 sp C
            Atom::new(Element::from_z(1).unwrap(), [7.14, 0.0, 0.0]), // 3 acetylenic H
            Atom::new(Element::from_z(1).unwrap(), [-0.69, 1.94, 0.0]), // 4 methyl H
            Atom::new(Element::from_z(1).unwrap(), [-0.69, -0.97, 1.68]), // 5 methyl H
            Atom::new(Element::from_z(1).unwrap(), [-0.69, -0.97, -1.68]), // 6 methyl H
        ],
        0,
        1,
    )
}

/// The two sp centres in propyne each contribute a co-linear-bend pair (instead of a
/// dropped angle), completing the redundant set: rank == 3N−6 so the internal frame is
/// not forced back to Cartesian. Confirms the near-linear branch is generated.
#[test]
fn propyne_linear_bends_complete_the_internal_set() {
    let mol = propyne();
    let defs = generate(&mol);
    let x: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let n_linear = defs
        .iter()
        .filter(|d| matches!(d, Internal::LinearBend(..)))
        .count();
    assert_eq!(
        n_linear, 4,
        "two sp centres × two perpendicular bends = 4 linear-bend coordinates, got {n_linear}"
    );
    let dof = 3 * x.len() - 6; // non-linear molecule overall (the methyl bends it)
    assert_eq!(
        internal_rank(&defs, &x),
        dof,
        "linear bends must complete the set to 3N−6 = {dof}"
    );
}

#[test]
fn displacement_wraps_torsion_across_branch() {
    // 170° → −175° is physically a +15° step across the +π branch, not −345°.
    let dih = vec![Internal::Dihedral(0, 1, 2, 3)];
    let to = vec![(-175.0_f64).to_radians()];
    let from = vec![170.0_f64.to_radians()];
    let d = displacement(&dih, &to, &from);
    assert!(
        (d[0] - 15.0_f64.to_radians()).abs() < 1e-12,
        "wrapped torsion change = {} rad",
        d[0]
    );
    // Bond and angle differences pass through unwrapped.
    let bond = vec![Internal::Bond(0, 1)];
    assert!((displacement(&bond, &[2.0], &[1.4])[0] - 0.6).abs() < 1e-12);
}

#[test]
fn back_transform_diatomic_stretch() {
    let mol = Molecule::new(
        vec![
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
            Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 1.40]),
        ],
        0,
        1,
    );
    let defs = generate(&mol);
    let x: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let x_new = back_transform(&defs, &x, &[0.10]); // stretch by 0.1 bohr
    let r = distance(x_new[0], x_new[1]);
    assert!(
        (r - 1.50).abs() < 1e-10,
        "back-transformed bond = {r}, want 1.50"
    );
}

/// Back-transforming a torsion step reproduces the requested change even when it
/// crosses the ±π branch: a +0.3 rad twist from a torsion already near +π lands at the
/// wrapped target, confirming the residual is taken along the short arc.
#[test]
fn back_transform_torsion_across_branch() {
    let x = hooh();
    let defs = vec![Internal::Dihedral(0, 1, 2, 3)];
    let phi0 = values(&defs, &x)[0];
    // Push the starting torsion close to +π, then twist past it.
    let near_pi = std::f64::consts::PI - 0.15;
    let x_pre = back_transform(&defs, &x, &[near_pi - phi0]);
    let x_post = back_transform(&defs, &x_pre, &[0.30]);
    let got = values(&defs, &x_post)[0];
    let target = wrap_to_pi(near_pi + 0.30);
    assert!(
        wrap_to_pi(got - target).abs() < 1e-6,
        "torsion landed at {got} rad, want {target} (wrapped)"
    );
}
