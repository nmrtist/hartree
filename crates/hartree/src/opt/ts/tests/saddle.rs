use super::*;
use crate::core::{Atom, Element, Molecule};
use crate::opt::ts::numerics::{column, mw_projected_hessian};
use crate::opt::ts::{TsOptions, verify_saddle};

#[test]
fn verify_saddle_on_quadratic() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mol = h3_molecule(&x0);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let v = verify_saddle(&mol, &mut surf, &x0, &TsOptions::default()).unwrap();
    assert!(
        v.is_first_order_saddle(),
        "expected one negative mode, got {:?}",
        v.negative_eigenvalues
    );
    assert!(v.negative_eigenvalues[0] < 0.0);
    assert!(
        v.imaginary_frequency_cm1.unwrap() < 0.0,
        "imaginary frequency must be reported negative"
    );
    let overlap = mode_overlap(&v.reaction_mode.unwrap(), &basis[0]);
    assert!(overlap > 0.99, "reaction mode overlap with w1 = {overlap}");
}

/// A heteronuclear analytic saddle (H, C, O): the distinct masses make the
/// mass-weighted Hessian eigenvectors differ from the Cartesian ones, so the
/// reaction mode is only right if it is reported un-mass-weighted (Cartesian), as
/// documented — the equal-mass H3 fixtures cannot tell the two frames apart.
#[test]
fn reaction_mode_is_cartesian_for_heteronuclear() {
    let x0 = h3_positions();
    let atoms = [1u32, 6, 8]
        .iter()
        .zip(&x0)
        .map(|(&z, &p)| Atom::new(Element::from_z(z).unwrap(), p))
        .collect();
    let mol = Molecule::new(atoms, 0, 1);
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut surf = Quadratic {
        x0: x0.clone(),
        h: h.clone(),
    };
    let v = verify_saddle(&mol, &mut surf, &x0, &TsOptions::default()).unwrap();
    assert!(v.is_first_order_saddle());

    // Expected Cartesian reaction mode: un-mass-weight the lowest mass-weighted mode.
    let masses: Vec<f64> = mol.atoms.iter().map(|a| a.element.mass()).collect();
    let spec = mw_projected_hessian(&x0, &masses, &h).unwrap();
    let q = column(&spec.eigenvectors, 9, 0);
    let mut cart = q.clone();
    for a in 0..3 {
        let s = masses[a].sqrt();
        for c in 0..3 {
            cart[3 * a + c] /= s;
        }
    }
    let cn: f64 = cart.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut cart {
        *x /= cn;
    }

    let react: Vec<f64> = v.reaction_mode.unwrap().iter().flatten().copied().collect();
    let overlap_cart = react
        .iter()
        .zip(&cart)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .abs();
    let overlap_mw = react.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>().abs();
    // Matches the Cartesian mode; the raw mass-weighted mode is a different vector,
    // so a regression that returned it would fail here.
    assert!(
        overlap_cart > 0.999,
        "reaction mode not Cartesian (overlap {overlap_cart})"
    );
    assert!(
        overlap_mw < 0.99,
        "reaction mode looks mass-weighted (overlap {overlap_mw})"
    );
}

#[test]
fn verify_saddle_rejects_minimum() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    // All-positive curvatures: a minimum, not a saddle.
    let h = hessian_from(&basis, &[0.5, 0.6, 0.9]);
    let mol = h3_molecule(&x0);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let v = verify_saddle(&mol, &mut surf, &x0, &TsOptions::default()).unwrap();
    assert!(v.negative_eigenvalues.is_empty());
    assert!(v.reaction_mode.is_none());
    assert!(v.imaginary_frequency_cm1.is_none());
}
