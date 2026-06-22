pub mod lebedev;

mod becke;
mod radial;

pub(crate) use becke::BeckePartition;

use crate::core::Molecule;
use rayon::prelude::*;

use crate::dft::error::DftError;

pub const MAX_LEVEL: usize = 4;

/// Heaviest element the DFT grid supports (Rn, Z = 86 — periods 1–6). The radial
/// (Treutler–Ahlrichs ξ) and partition (Bragg–Slater) tables cover this range.
pub const MAX_GRID_Z: u32 = 86;

const PARTITION_CUTOFF: f64 = 1e-14;

// Per-period grid density. Columns are period 1–6 (period_index); rows are grid
// levels 0–4. Columns 0–2 (Z ≤ 18) are unchanged; columns 3–5 (periods 4–6) adopt
// PySCF's default radial/angular plan, where radial points grow per period and the
// angular (Lebedev) order saturates after period 3.
pub const RAD_GRIDS: [[usize; 6]; 5] = [
    [10, 15, 20, 30, 35, 40],    // level 0
    [30, 40, 50, 60, 65, 70],    // level 1
    [40, 60, 65, 75, 80, 85],    // level 2
    [50, 75, 80, 90, 95, 100],   // level 3
    [60, 90, 95, 105, 110, 115], // level 4
];

pub const ANG_NPTS: [[usize; 6]; 5] = [
    [50, 86, 110, 110, 110, 110], // level 0  (degrees 11, 15, 17, 17, 17, 17)
    [110, 194, 194, 194, 194, 194], // level 1  (17, 23, 23, 23, 23, 23)
    [194, 302, 302, 302, 302, 302], // level 2  (23, 29, 29, 29, 29, 29)
    [302, 302, 434, 434, 434, 434], // level 3  (29, 29, 35, 35, 35, 35)
    [434, 590, 590, 590, 590, 590], // level 4  (35, 41, 41, 41, 41, 41)
];

fn period_index(z: u32) -> usize {
    match z {
        0..=2 => 0,   // period 1
        3..=10 => 1,  // period 2
        11..=18 => 2, // period 3 (unchanged for Z ≤ 18)
        19..=36 => 3, // period 4  K–Kr
        37..=54 => 4, // period 5  Rb–Xe
        _ => 5,       // period 6  Cs–Rn
    }
}

fn level_config(z: u32, level: usize) -> (usize, usize) {
    let p = period_index(z);
    (RAD_GRIDS[level][p], ANG_NPTS[level][p])
}

#[derive(Debug, Clone)]
pub struct MolecularGrid {
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub atom_of_point: Vec<usize>,
}

struct AtomBlock {
    points: Vec<[f64; 3]>,
    weights: Vec<f64>,
}

impl MolecularGrid {
    pub fn build(mol: &Molecule, level: usize) -> Result<Self, DftError> {
        if level > MAX_LEVEL {
            return Err(DftError::InvalidGridLevel(level));
        }
        for atom in &mol.atoms {
            let z = atom.element.z();
            if !(1..=MAX_GRID_Z).contains(&z) {
                return Err(DftError::UnsupportedElement(z));
            }
        }

        let partition = becke::BeckePartition::new(mol);
        let n_atoms = mol.atoms.len();

        let blocks: Vec<AtomBlock> = (0..n_atoms)
            .into_par_iter()
            .map(|ia| build_atom_block(mol, ia, level, &partition))
            .collect();

        let total: usize = blocks.iter().map(|b| b.weights.len()).sum();
        let mut points = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        let mut atom_of_point = Vec::with_capacity(total);
        for (ia, block) in blocks.into_iter().enumerate() {
            atom_of_point.resize(atom_of_point.len() + block.weights.len(), ia);
            points.extend(block.points);
            weights.extend(block.weights);
        }
        Ok(Self {
            points,
            weights,
            atom_of_point,
        })
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

fn build_atom_block(
    mol: &Molecule,
    ia: usize,
    level: usize,
    partition: &becke::BeckePartition,
) -> AtomBlock {
    let z = mol.atoms[ia].element.z();
    let center = mol.atoms[ia].position; // bitwise reuse of the molecule's coords
    let (n_rad, n_ang) = level_config(z, level);
    let (radii, rad_w) = radial::treutler_ahlrichs(z, n_rad);
    let ang = lebedev::LebedevGrid::new(n_ang)
        .expect("level-table angular count must be a shipped Lebedev grid");

    let n_atoms = partition.n_atoms();
    let mut dist = vec![0.0; n_atoms];
    let mut cell = vec![0.0; n_atoms];

    let mut points = Vec::new();
    let mut weights = Vec::new();
    for (&r, &rw) in radii.iter().zip(&rad_w) {
        for (u, &aw) in ang.points.iter().zip(&ang.weights) {
            let p = [
                center[0] + r * u[0],
                center[1] + r * u[1],
                center[2] + r * u[2],
            ];
            partition.weights_into(p, &mut dist, &mut cell);
            let w_atom = cell[ia];
            if w_atom < PARTITION_CUTOFF {
                continue;
            }
            points.push(p);
            weights.push(rw * aw * w_atom);
        }
    }
    AtomBlock { points, weights }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};
    use std::f64::consts::PI;

    fn gaussian_integral(alpha: f64) -> f64 {
        (PI / alpha).powf(1.5)
    }

    fn atom(z: u32, pos: [f64; 3]) -> Atom {
        Atom::new(Element::from_z(z).unwrap(), pos)
    }

    #[test]
    fn single_center_gaussian() {
        let mol = Molecule::new(vec![atom(8, [0.0, 0.0, 0.0])], 0, 1);
        for level in [3, 4] {
            let grid = MolecularGrid::build(&mol, level).unwrap();
            assert!(grid.weights.iter().all(|&w| w > 0.0), "negative weight");
            let mut max_rel = 0.0_f64;
            for &alpha in &[0.5_f64, 1.0, 2.0] {
                let quad: f64 = grid
                    .points
                    .iter()
                    .zip(&grid.weights)
                    .map(|(p, &w)| {
                        let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                        w * (-alpha * r2).exp()
                    })
                    .sum();
                let rel = (quad - gaussian_integral(alpha)).abs() / gaussian_integral(alpha);
                assert!(rel < 1e-10, "level {level} alpha {alpha}: rel err {rel:e}");
                max_rel = max_rel.max(rel);
            }
            println!("single-center O level {level}: max rel err {max_rel:e}");
        }
    }

    #[test]
    fn multi_center_gaussians() {
        let d = 0.74 * crate::core::units::ANGSTROM_TO_BOHR;
        let r1 = [0.0, 0.0, 0.0];
        let r2 = [0.0, 0.0, d];
        let mid = [0.0, 0.0, d / 2.0];
        let mol = Molecule::new(vec![atom(1, r1), atom(1, r2)], 0, 1);
        let grid = MolecularGrid::build(&mol, 3).unwrap();
        assert!(grid.weights.iter().all(|&w| w > 0.0));

        let g = |p: &[f64; 3], c: &[f64; 3], alpha: f64| {
            let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
            (-alpha * (dx * dx + dy * dy + dz * dz)).exp()
        };
        let mut max_rel = 0.0_f64;
        for &alpha in &[0.7_f64, 1.5] {
            let two: f64 = grid
                .points
                .iter()
                .zip(&grid.weights)
                .map(|(p, &w)| w * (g(p, &r1, alpha) + g(p, &r2, alpha)))
                .sum();
            let rel =
                (two - 2.0 * gaussian_integral(alpha)).abs() / (2.0 * gaussian_integral(alpha));
            assert!(rel < 1e-8, "two-center alpha {alpha}: rel err {rel:e}");
            max_rel = max_rel.max(rel);

            let off: f64 = grid
                .points
                .iter()
                .zip(&grid.weights)
                .map(|(p, &w)| w * g(p, &mid, alpha))
                .sum();
            let rel = (off - gaussian_integral(alpha)).abs() / gaussian_integral(alpha);
            assert!(rel < 1e-8, "off-center alpha {alpha}: rel err {rel:e}");
            max_rel = max_rel.max(rel);
        }
        println!("H₂ multi-center level 3: max rel err {max_rel:e}");
    }

    #[test]
    fn point_counts_grow_with_level() {
        let mol = Molecule::from_xyz(
            "3\nwater\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
        )
        .unwrap();
        let mut prev = 0usize;
        for level in 0..=MAX_LEVEL {
            let grid = MolecularGrid::build(&mol, level).unwrap();
            assert_eq!(grid.points.len(), grid.weights.len());
            assert_eq!(grid.atom_of_point.len(), grid.weights.len());
            assert!(grid.weights.iter().all(|&w| w > 0.0));
            assert!(
                grid.len() > prev,
                "level {level}: {} not > {prev}",
                grid.len()
            );
            prev = grid.len();
            println!("water level {level}: {} points", grid.len());
        }
    }

    #[test]
    fn rejects_bad_level_and_element() {
        let mol = Molecule::new(vec![atom(8, [0.0; 3])], 0, 1);
        assert!(matches!(
            MolecularGrid::build(&mol, 5),
            Err(DftError::InvalidGridLevel(5))
        ));
        // Period 7 (here U, Z=92) is past the supported range and is rejected cleanly.
        let beyond = Molecule::new(vec![atom(92, [0.0; 3])], 0, 1);
        assert!(matches!(
            MolecularGrid::build(&beyond, 3),
            Err(DftError::UnsupportedElement(92))
        ));
    }

    #[test]
    fn builds_grid_for_heavy_elements() {
        // Period 4–6 elements (Fe, Br, Au) now build a non-empty grid; before the
        // Bragg/ξ tables were extended these panicked or were rejected at Z > 18.
        for z in [26u32, 35, 79, MAX_GRID_Z] {
            let mol = Molecule::new(vec![atom(z, [0.0; 3])], 0, 1);
            let grid = MolecularGrid::build(&mol, 3).unwrap();
            assert!(!grid.is_empty(), "Z={z} produced an empty grid");
            assert!(
                grid.weights.iter().all(|&w| w.is_finite()),
                "Z={z} grid has non-finite weights"
            );
        }
    }
}
