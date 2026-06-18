use crate::core::Molecule;

use crate::props::frequencies::{FrequencyResult, harmonic_frequencies_projected};

pub const SPH_GRADIENT_NORM_FLOOR: f64 = 1.0e-3;

pub fn sph_frequencies(
    molecule: &Molecule,
    hessian: &[f64],
    gradient_cart: &[f64],
) -> FrequencyResult {
    let natom = molecule.len();
    let ndof = 3 * natom;
    assert_eq!(gradient_cart.len(), ndof, "gradient must be length 3N");

    let masses: Vec<f64> = molecule.atoms.iter().map(|a| a.element.mass()).collect();
    let mut g_mw = vec![0.0f64; ndof];
    for i in 0..natom {
        let sm = masses[i].sqrt();
        for k in 0..3 {
            g_mw[3 * i + k] = gradient_cart[3 * i + k] / sm;
        }
    }
    let norm: f64 = g_mw.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm < SPH_GRADIENT_NORM_FLOOR {
        harmonic_frequencies_projected(molecule, hessian, &[])
    } else {
        for x in &mut g_mw {
            *x /= norm;
        }
        harmonic_frequencies_projected(molecule, hessian, &[g_mw])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};
    use crate::props::frequencies::harmonic_frequencies;
    use crate::props::hessian::numerical_hessian;

    fn toy() -> (Molecule, Vec<f64>) {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, -0.7]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.7]),
            ],
            0,
            1,
        );
        let k = 0.5;
        let mut h = vec![0.0f64; 36];
        let zz = [(2usize, 2usize, k), (2, 5, -k), (5, 2, -k), (5, 5, k)];
        for (i, j, v) in zz {
            h[i * 6 + j] = v;
        }
        (mol, h)
    }

    #[test]
    fn reduces_to_ordinary_at_zero_gradient() {
        let (mol, h) = toy();
        let zero_grad = vec![0.0f64; 6];
        let sph = sph_frequencies(&mol, &h, &zero_grad);
        let ord = harmonic_frequencies(&mol, &h);
        assert_eq!(sph.frequencies_cm1.len(), ord.frequencies_cm1.len());
        for (a, b) in sph.frequencies_cm1.iter().zip(&ord.frequencies_cm1) {
            assert!((a - b).abs() < 1e-9, "SPH {a} vs ordinary {b}");
        }
    }

    #[test]
    fn tiny_gradient_below_floor_is_noop() {
        let (mol, h) = toy();
        let grad = vec![1e-9; 6]; // norm below floor
        let sph = sph_frequencies(&mol, &h, &grad);
        let ord = harmonic_frequencies(&mol, &h);
        for (a, b) in sph.frequencies_cm1.iter().zip(&ord.frequencies_cm1) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn gradient_projection_removes_a_mode() {
        let (mol, h) = toy();
        let grad = vec![0.0, 0.0, 0.3, 0.0, 0.0, -0.3]; // along the stretch
        let sph = sph_frequencies(&mol, &h, &grad);
        assert_eq!(sph.n_imaginary, 0);
        let n_nonzero = sph
            .frequencies_cm1
            .iter()
            .filter(|f| f.abs() > 10.0)
            .count();
        let ord = harmonic_frequencies(&mol, &h);
        let n_nonzero_ord = ord
            .frequencies_cm1
            .iter()
            .filter(|f| f.abs() > 10.0)
            .count();
        assert_eq!(n_nonzero_ord, 1, "toy has 1 vibration");
        assert_eq!(n_nonzero, 0, "gradient mode projected out");
    }

    #[test]
    fn sph_water_displaced_no_spurious_imaginary() {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.20]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, 1.43, -0.90]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, -1.43, -0.90]),
            ],
            0,
            1,
        );
        let ref_pos: Vec<[f64; 3]> =
            vec![[0.0, 0.0, 0.12], [0.0, 1.43, -0.95], [0.0, -1.43, -0.95]];
        let k = 0.4;
        let grad_fn = |m: &Molecule| -> Vec<f64> {
            let mut g = vec![0.0f64; 9];
            for i in 0..3 {
                for c in 0..3 {
                    g[3 * i + c] = k * (m.atoms[i].position[c] - ref_pos[i][c]);
                }
            }
            g
        };
        let hess = numerical_hessian(&mol, 0.005, |m| grad_fn(m));
        let grad_here = grad_fn(&mol);
        let sph = sph_frequencies(&mol, &hess, &grad_here);
        assert!(sph.frequencies_cm1.iter().all(|f| f.is_finite()));
        assert_eq!(
            sph.n_imaginary, 0,
            "spurious imaginary: {:?}",
            sph.frequencies_cm1
        );
    }
}
