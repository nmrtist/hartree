use super::*;
use crate::core::{Atom, Element, Molecule};
use crate::opt::ts::numerics::{column, mw_projected_hessian, spectrum_ambiguous};
use crate::opt::ts::{SaddleVerification, TsOptions, verify_saddle};

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

/// A higher-order (second-order) saddle has no single reaction coordinate, so the
/// reaction mode and imaginary frequency are withheld: `reaction_mode.is_some()`
/// must agree with `is_first_order_saddle()` (the consistency the two fields
/// promise), and both are `None` here even though the lowest mode is negative.
#[test]
fn higher_order_saddle_withholds_reaction_mode() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    // Two negative curvatures: a second-order saddle.
    let h = hessian_from(&basis, &[-0.4, -0.3, 0.9]);
    let mol = h3_molecule(&x0);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let v = verify_saddle(&mol, &mut surf, &x0, &TsOptions::default()).unwrap();
    assert_eq!(v.negative_eigenvalues.len(), 2);
    assert!(!v.is_first_order_saddle());
    assert!(
        v.reaction_mode.is_none(),
        "a second-order saddle must not report a single reaction mode"
    );
    assert!(v.imaginary_frequency_cm1.is_none());
    // The invariant the field docs promise.
    assert_eq!(v.reaction_mode.is_some(), v.is_first_order_saddle());
}

/// Verification carries the full harmonic spectrum (every Cartesian DOF) for free
/// from its eigendecomposition, so a converged saddle yields its vibrational
/// frequencies without a second Hessian.
#[test]
fn verification_carries_full_spectrum_and_frequencies() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mol = h3_molecule(&x0);
    let mut surf = Quadratic { x0: x0.clone(), h };
    let v = verify_saddle(&mol, &mut surf, &x0, &TsOptions::default()).unwrap();

    // One eigenvalue per Cartesian DOF (3 atoms ⇒ 9), ascending.
    assert_eq!(v.eigenvalues.len(), 9);
    assert!(v.eigenvalues.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    // The lone negative eigenvalue is the smallest and equals the reported one.
    assert_eq!(v.negative_eigenvalues.len(), 1);
    assert!((v.eigenvalues[0] - v.negative_eigenvalues[0]).abs() < 1e-12);

    // Frequencies drop the 6 trans/rot null modes ⇒ 3 physical modes, with exactly
    // one imaginary (negative) frequency equal to the reported imaginary one.
    let freqs = v.frequencies_cm1();
    assert_eq!(freqs.len(), 3, "expected 3 physical modes, got {freqs:?}");
    let imag: Vec<f64> = freqs.iter().copied().filter(|&f| f < 0.0).collect();
    assert_eq!(imag.len(), 1);
    assert!((imag[0] - v.imaginary_frequency_cm1.unwrap()).abs() < 1e-6);
    assert_eq!(freqs.iter().filter(|&&f| f > 0.0).count(), 2);
}

/// A `SaddleVerification` serialized before the full spectrum was stored (no
/// `eigenvalues` key) still deserializes — the field defaults to empty, and
/// `frequencies_cm1` is then empty — without disturbing the older fields.
#[test]
fn verification_spectrum_defaults_empty_for_legacy_record() {
    let json =
        r#"{"negative_eigenvalues":[-0.4],"reaction_mode":null,"imaginary_frequency_cm1":-100.0}"#;
    let v: SaddleVerification = serde_json::from_str(json).unwrap();
    assert!(v.eigenvalues.is_empty());
    assert!(v.frequencies_cm1().is_empty());
    assert!(v.is_first_order_saddle());
}

/// The ambiguity band that drives the [`Auto`](crate::opt::ts::VerifyHessian::Auto)
/// fall-back: a physical mode in `(-2·tol, +tol)` is too close to the `−tol` cut to
/// trust an approximate Hessian on; modes clearly past either side, and the near-zero
/// trans/rot nulls, are not flagged.
#[test]
fn spectrum_ambiguous_flags_near_threshold_modes() {
    let tol = 1e-4;
    // Clean: reaction mode below −2·tol, others above tol, a null mode exempt.
    assert!(!spectrum_ambiguous(&[-0.4, 1e-9, 0.5, 0.9], tol));
    // A small positive mode inside the band could flip negative under Hessian error.
    assert!(spectrum_ambiguous(&[-0.4, 5e-5, 0.9], tol));
    // A mode just past −tol (but above −2·tol) could flip non-negative.
    assert!(spectrum_ambiguous(&[-1.5e-4, 0.6, 0.9], tol));
    // A clearly-negative mode below −2·tol is trustworthy.
    assert!(!spectrum_ambiguous(&[-3e-4, 0.6, 0.9], tol));
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
