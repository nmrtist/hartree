use crate::core::Molecule;
use crate::core::units::{BOLTZMANN_HT, HARTREE_TO_KCAL_MOL};

#[derive(Debug, Clone)]
pub struct Conformer {
    pub molecule: Molecule,
    pub energy: f64,
}

#[derive(Debug, Clone)]
pub struct Ensemble {
    pub conformers: Vec<Conformer>,
}

impl Ensemble {
    pub fn new(mut conformers: Vec<Conformer>) -> Self {
        conformers.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        Self { conformers }
    }

    pub fn len(&self) -> usize {
        self.conformers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conformers.is_empty()
    }

    pub fn min_energy(&self) -> Option<f64> {
        self.conformers.first().map(|c| c.energy)
    }

    pub fn relative_energies(&self) -> Vec<f64> {
        let e0 = self.min_energy().unwrap_or(0.0);
        self.conformers.iter().map(|c| c.energy - e0).collect()
    }

    pub fn relative_energies_kcal(&self) -> Vec<f64> {
        self.relative_energies()
            .into_iter()
            .map(|d| d * HARTREE_TO_KCAL_MOL)
            .collect()
    }

    pub fn boltzmann_weights(&self, temperature_k: f64) -> Vec<f64> {
        let kt = BOLTZMANN_HT * temperature_k;
        let rel = self.relative_energies();
        let unnorm: Vec<f64> = rel.iter().map(|&d| (-d / kt).exp()).collect();
        let z: f64 = unnorm.iter().sum();
        if z <= 0.0 || !z.is_finite() {
            let n = self.len() as f64;
            return vec![1.0 / n; self.len()];
        }
        unnorm.into_iter().map(|x| x / z).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_mol() -> Molecule {
        Molecule::from_xyz("1\nx\nH 0 0 0\n").unwrap()
    }

    fn ens(energies: &[f64]) -> Ensemble {
        Ensemble::new(
            energies
                .iter()
                .map(|&e| Conformer {
                    molecule: dummy_mol(),
                    energy: e,
                })
                .collect(),
        )
    }

    #[test]
    fn sorts_and_relative() {
        let e = ens(&[-1.0, -1.5, -1.2]);
        assert_eq!(e.min_energy().unwrap(), -1.5);
        let rel = e.relative_energies();
        assert!(rel[0].abs() < 1e-15);
        assert!(rel.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn weights_sum_to_one_and_favor_lowest() {
        let e = ens(&[0.0, 0.001, 0.002]); // hartree
        let w = e.boltzmann_weights(298.15);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(w[0] > w[1] && w[1] > w[2]);
    }

    #[test]
    fn degenerate_equal_energies_uniform() {
        let e = ens(&[-5.0, -5.0, -5.0, -5.0]);
        let w = e.boltzmann_weights(298.15);
        for wi in w {
            assert!((wi - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn known_two_state_ratio() {
        let kt = BOLTZMANN_HT * 298.15;
        let de = kt * 2.0f64.ln();
        let e = ens(&[0.0, de]);
        let w = e.boltzmann_weights(298.15);
        assert!((w[0] / w[1] - 2.0).abs() < 1e-9, "ratio {}", w[0] / w[1]);
    }
}
