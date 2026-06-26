pub mod lebedev;

mod becke;
mod radial;

pub(crate) use becke::BeckePartition;

use crate::core::Molecule;
use rayon::prelude::*;
use std::collections::BTreeMap;

use crate::dft::error::DftError;

pub const MAX_LEVEL: usize = 4;

/// Heaviest element the DFT grid supports (Rn, Z = 86 — periods 1–6). The radial
/// (Treutler–Ahlrichs ξ) and partition (Bragg–Slater) tables cover this range.
pub const MAX_GRID_Z: u32 = 86;

const PARTITION_CUTOFF: f64 = 1e-14;

/// The production default grid level.
const DEFAULT_LEVEL: usize = 3;

/// Grid levels that apply the five-zone angular pruning: the two production-tier
/// levels (2 and 3). The coarse presets (levels 0/1 — VV10's non-local grid and
/// COSX use level 1) and the reference-quality level 4 (the external-reference DFT
/// oracle and r2scan-3c) keep the full Lebedev order on every radial shell, so they
/// stay bit-for-bit as pinned.
fn prunes_angular(level: usize) -> bool {
    level == 2 || level == DEFAULT_LEVEL
}

// Per-period radial-point count (Treutler–Ahlrichs mapping). Columns are period
// 1–6 (period_index); rows are grid levels 0–4. The production default (level 3)
// pairs a moderate radial set (~60 shells for the second row) with the full
// five-zone angular pruning below; this is the point-efficient regime mainstream
// codes use for default energies — far fewer points than a saturated radial grid at
// the same converged accuracy. Level 4 keeps the dense reference radial set.
//
// Levels 2 and 3 share an identical radial set by design: by ~60 shells the radial
// dimension is already converged, so saturating it further buys no accuracy. The two
// production tiers are separated solely by angular order (see ANG_NPTS) — level 3's
// finer valence Lebedev grid is the single variable that lifts it above level 2
// (water: ~27k vs ~22k points), which keeps the cause of any level-2/3 result
// difference unambiguous.
pub const RAD_GRIDS: [[usize; 6]; 5] = [
    [10, 15, 20, 30, 35, 40],    // level 0
    [30, 40, 50, 60, 65, 70],    // level 1
    [40, 60, 65, 75, 80, 85],    // level 2
    [40, 60, 65, 75, 80, 85],    // level 3  (production default)
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

/// The Lebedev ladder the five-zone angular pruning steps along — the shipped
/// Lebedev–Laikov orders from 38 up (including the negative-weight 74/230/266 and
/// the 350 rules). A radial shell's zone index selects an order off this ladder
/// relative to the shell's base order `n_ang`.
const PRUNE_LADDER: [usize; 13] = [38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434];

/// Per-shell Lebedev orders for the five radial zones
/// [core, inner, inner-valence, outer-valence, tail] of an atom whose base angular
/// order is `n_ang`. The outer-valence zone carries the full order, the two
/// flanking zones drop one Lebedev rung, and the core and near-tail coarsen to a
/// fixed (50, 86) pair. This is the standard five-zone radial pruning: it roughly
/// halves the molecular point count while holding the grid-converged XC energy,
/// gradient, and density integrals, because the coarsened zones are either nearly
/// spherical (core) or carry little density (tail). For the base order 302 it gives
/// [50, 86, 266, 302, 266]; for 434, [50, 86, 350, 434, 350].
fn prune_zone_orders(n_ang: usize) -> [usize; 5] {
    if n_ang < 50 {
        return [n_ang; 5];
    }
    if n_ang == 50 {
        return [50, 74, 74, 74, 50];
    }
    let idx = PRUNE_LADDER
        .iter()
        .position(|&o| o == n_ang)
        .expect("base angular order must lie on the prune ladder");
    [
        PRUNE_LADDER[1],
        PRUNE_LADDER[3],
        PRUNE_LADDER[idx - 1],
        PRUNE_LADDER[idx],
        PRUNE_LADDER[idx - 1],
    ]
}

/// Pruning zone boundaries as fractions of the Bragg radius, selected by period.
/// A radial shell at fractional radius f sits in zone = #{α : f > α}, so the four
/// thresholds partition [0, ∞) into the five zones of `prune_zone_orders`.
fn prune_alphas(z: u32) -> [f64; 4] {
    match z {
        0..=2 => [0.25, 0.5, 1.0, 4.5],    // H, He
        3..=10 => [0.1667, 0.5, 0.9, 3.5], // Li–Ne
        _ => [0.1, 0.4, 0.8, 2.5],         // Na and heavier
    }
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

        // Angular pruning is on by default. The escape hatch restores the historical
        // unpruned grid (full Lebedev order on every radial shell) for bit-for-bit
        // reproduction of pre-pruning energies.
        let prune = std::env::var_os("HARTREE_GRID_UNPRUNED").is_none();

        let blocks: Vec<AtomBlock> = (0..n_atoms)
            .into_par_iter()
            .map(|ia| build_atom_block(mol, ia, level, &partition, prune))
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
    prune: bool,
) -> AtomBlock {
    let z = mol.atoms[ia].element.z();
    let center = mol.atoms[ia].position; // bitwise reuse of the molecule's coords
    let (n_rad, n_ang) = level_config(z, level);
    let (radii, rad_w) = radial::treutler_ahlrichs(z, n_rad);

    // Angular pruning (production-tier levels only): the Lebedev order varies per
    // radial shell so the nearly-spherical core and the low-density tail are not
    // over-sampled with the full valence-order sphere. The coarse and reference
    // levels keep the full order on each shell. Cache the distinct orders this atom
    // needs.
    let zone_orders = if prune && prunes_angular(level) {
        prune_zone_orders(n_ang)
    } else {
        [n_ang; 5]
    };
    let alphas = prune_alphas(z);
    let r_atom = becke::bragg_radius_bohr(z);
    let mut ang_cache: BTreeMap<usize, lebedev::LebedevGrid> = BTreeMap::new();
    for &order in &zone_orders {
        ang_cache.entry(order).or_insert_with(|| {
            lebedev::LebedevGrid::new(order)
                .expect("prune-zone angular count must be a shipped Lebedev grid")
        });
    }

    let n_atoms = partition.n_atoms();
    let mut dist = vec![0.0; n_atoms];
    let mut cell = vec![0.0; n_atoms];

    let mut points = Vec::new();
    let mut weights = Vec::new();
    for (&r, &rw) in radii.iter().zip(&rad_w) {
        let frac = r / r_atom;
        let zone = alphas.iter().filter(|&&a| frac > a).count(); // 0 (core) … 4 (tail)
        let ang = &ang_cache[&zone_orders[zone]];
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
            // The pruned default level uses the 266/350 Lebedev rules, which carry a
            // few small negative angular weights (a published property), so only
            // finiteness is asserted here; the integral accuracy is checked below.
            assert!(
                grid.weights.iter().all(|&w| w.is_finite()),
                "non-finite weight"
            );
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
        assert!(grid.weights.iter().all(|&w| w.is_finite()));

        let g = |p: &[f64; 3], c: &[f64; 3], alpha: f64| {
            let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
            (-alpha * (dx * dx + dy * dy + dz * dz)).exp()
        };
        // The production level-3 grid angular-prunes the core and tail of each atom,
        // so a bare-Gaussian quadrature centred off the dense valence shells
        // converges to ~1e-6 rather than the ~1e-10 of the unpruned reference grids —
        // more than enough for the ~3e-6 Eh energy gates the full grid has to meet.
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
            assert!(rel < 5e-6, "two-center alpha {alpha}: rel err {rel:e}");
            max_rel = max_rel.max(rel);

            let off: f64 = grid
                .points
                .iter()
                .zip(&grid.weights)
                .map(|(p, &w)| w * g(p, &mid, alpha))
                .sum();
            let rel = (off - gaussian_integral(alpha)).abs() / gaussian_integral(alpha);
            assert!(rel < 5e-6, "off-center alpha {alpha}: rel err {rel:e}");
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
            assert!(grid.weights.iter().all(|&w| w.is_finite()));
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
