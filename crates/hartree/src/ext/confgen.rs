use crate::core::Molecule;

use crate::ext::ExtError;
use crate::ext::ensemble::{Conformer, Ensemble};
use crate::ext::kabsch::kabsch_rmsd;

#[derive(Debug, Clone)]
pub struct ConfGenOptions {
    pub positions_per_bond: usize,
    pub max_candidates: usize,
    pub clash_factor: f64,
    pub rmsd_threshold_bohr: f64,
    pub energy_window_hartree: f64,
}

impl Default for ConfGenOptions {
    fn default() -> Self {
        Self {
            positions_per_bond: 3,
            max_candidates: 2000,
            clash_factor: 0.6,
            rmsd_threshold_bohr: 0.1,
            energy_window_hartree: 1.0e-4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfGenResult {
    pub ensemble: Ensemble,
    pub rotatable_bonds: Vec<(usize, usize)>,
    pub driven_bonds: Vec<(usize, usize)>,
    pub n_candidates: usize,
    pub n_screened: usize,
}

pub fn connectivity(molecule: &Molecule) -> Vec<Vec<usize>> {
    let n = molecule.len();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let ri = molecule.atoms[i].element.covalent_radius();
            let rj = molecule.atoms[j].element.covalent_radius();
            let d = dist(molecule.atoms[i].position, molecule.atoms[j].position);
            if d < 1.3 * (ri + rj) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

pub fn rotatable_bonds(molecule: &Molecule) -> Vec<(usize, usize)> {
    let adj = connectivity(molecule);
    let n = molecule.len();
    let is_heavy = |k: usize| molecule.atoms[k].element.z() > 1;
    let mut bonds = Vec::new();
    for a in 0..n {
        if !is_heavy(a) {
            continue;
        }
        for &b in &adj[a] {
            if b <= a || !is_heavy(b) {
                continue;
            }
            let a_has_other = adj[a].iter().any(|&k| k != b && is_heavy(k));
            let b_has_other = adj[b].iter().any(|&k| k != a && is_heavy(k));
            if !a_has_other || !b_has_other {
                continue;
            }
            if !in_ring(&adj, a, b) {
                bonds.push((a, b));
            }
        }
    }
    bonds
}

fn in_ring(adj: &[Vec<usize>], a: usize, b: usize) -> bool {
    let mut seen = vec![false; adj.len()];
    let mut stack = vec![b];
    seen[b] = true;
    while let Some(x) = stack.pop() {
        for &y in &adj[x] {
            if x == b && y == a {
                continue; // skip the direct edge once
            }
            if y == a {
                return true;
            }
            if !seen[y] {
                seen[y] = true;
                stack.push(y);
            }
        }
    }
    false
}

fn fragment_on_b_side(adj: &[Vec<usize>], a: usize, b: usize) -> Vec<usize> {
    let mut seen = vec![false; adj.len()];
    seen[a] = true; // never cross back to a
    seen[b] = true;
    let mut stack = vec![b];
    let mut frag = vec![b];
    while let Some(x) = stack.pop() {
        for &y in &adj[x] {
            if !seen[y] {
                seen[y] = true;
                frag.push(y);
                stack.push(y);
            }
        }
    }
    frag
}

fn rotate_fragment(
    molecule: &Molecule,
    atoms: &[usize],
    pivot: [f64; 3],
    axis: [f64; 3],
    angle: f64,
) -> Molecule {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let u = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let (s, c) = angle.sin_cos();
    let mut out = molecule.clone();
    for &i in atoms {
        let p = molecule.atoms[i].position;
        let r = [p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]];
        let cross = [
            u[1] * r[2] - u[2] * r[1],
            u[2] * r[0] - u[0] * r[2],
            u[0] * r[1] - u[1] * r[0],
        ];
        let dot = u[0] * r[0] + u[1] * r[1] + u[2] * r[2];
        let rot = [
            r[0] * c + cross[0] * s + u[0] * dot * (1.0 - c),
            r[1] * c + cross[1] * s + u[1] * dot * (1.0 - c),
            r[2] * c + cross[2] * s + u[2] * dot * (1.0 - c),
        ];
        out.atoms[i].position = [rot[0] + pivot[0], rot[1] + pivot[1], rot[2] + pivot[2]];
    }
    out
}

fn has_clash(molecule: &Molecule, adj: &[Vec<usize>], clash_factor: f64) -> bool {
    let n = molecule.len();
    #[allow(clippy::needless_range_loop)] // symmetric i<j pair scan with index bookkeeping
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i].contains(&j) {
                continue;
            }
            let d = dist(molecule.atoms[i].position, molecule.atoms[j].position);
            let cut = clash_factor
                * (vdw_radius_bohr(molecule.atoms[i].element.z())
                    + vdw_radius_bohr(molecule.atoms[j].element.z()));
            if d < cut {
                return true;
            }
        }
    }
    false
}

pub fn generate_conformers<F>(
    molecule: &Molecule,
    opts: &ConfGenOptions,
    mut energy_fn: F,
) -> Result<ConfGenResult, ExtError>
where
    F: FnMut(&Molecule) -> Option<f64>,
{
    if molecule.has_ghosts() {
        return Err(ExtError::ConfGen(
            "conformer generation does not support ghost atoms".into(),
        ));
    }
    if opts.positions_per_bond < 1 {
        return Err(ExtError::ConfGen(
            "positions_per_bond must be at least 1".into(),
        ));
    }
    let adj = connectivity(molecule);
    let rot_bonds = rotatable_bonds(molecule);

    let g = opts.positions_per_bond;
    let mut n_driven = rot_bonds.len();
    while n_driven > 0 && (g as u64).pow(n_driven as u32) > opts.max_candidates as u64 {
        n_driven -= 1;
    }
    let driven: Vec<(usize, usize)> = rot_bonds.iter().take(n_driven).copied().collect();

    struct Driver {
        atoms: Vec<usize>,
        pivot: [f64; 3],
        axis: [f64; 3],
    }
    let drivers: Vec<Driver> = driven
        .iter()
        .map(|&(a, b)| {
            let b_side = fragment_on_b_side(&adj, a, b);
            let (rot_atoms, pivot, axis) = if b_side.len() * 2 <= molecule.len() {
                let axis = vec_sub(molecule.atoms[b].position, molecule.atoms[a].position);
                (b_side, molecule.atoms[a].position, axis)
            } else {
                let a_side: Vec<usize> = (0..molecule.len())
                    .filter(|k| !b_side.contains(k))
                    .collect();
                let axis = vec_sub(molecule.atoms[a].position, molecule.atoms[b].position);
                (a_side, molecule.atoms[b].position, axis)
            };
            Driver {
                atoms: rot_atoms,
                pivot,
                axis,
            }
        })
        .collect();

    let total: usize = if drivers.is_empty() {
        1
    } else {
        (g).pow(drivers.len() as u32)
    };
    let mut candidates: Vec<Molecule> = Vec::new();
    for idx in 0..total {
        let mut cand = molecule.clone();
        let mut rem = idx;
        for d in &drivers {
            let step = rem % g;
            rem /= g;
            if step != 0 {
                let angle = 2.0 * std::f64::consts::PI * (step as f64) / (g as f64);
                cand = rotate_fragment(&cand, &d.atoms, d.pivot, d.axis, angle);
            }
        }
        candidates.push(cand);
    }
    let n_candidates = candidates.len();

    let mut scored: Vec<Conformer> = Vec::new();
    for cand in candidates {
        if has_clash(&cand, &adj, opts.clash_factor) {
            continue;
        }
        if let Some(e) = energy_fn(&cand) {
            scored.push(Conformer {
                molecule: cand,
                energy: e,
            });
        }
    }
    let n_screened = scored.len();

    scored.sort_by(|x, y| x.energy.partial_cmp(&y.energy).unwrap());
    let mut unique: Vec<Conformer> = Vec::new();
    for c in scored {
        let dup = unique.iter().any(|u| {
            (u.energy - c.energy).abs() < opts.energy_window_hartree && {
                let pu: Vec<[f64; 3]> = u.molecule.atoms.iter().map(|a| a.position).collect();
                let pc: Vec<[f64; 3]> = c.molecule.atoms.iter().map(|a| a.position).collect();
                kabsch_rmsd(&pu, &pc)
                    .map(|r| r < opts.rmsd_threshold_bohr)
                    .unwrap_or(false)
            }
        });
        if !dup {
            unique.push(c);
        }
    }

    Ok(ConfGenResult {
        ensemble: Ensemble::new(unique),
        rotatable_bonds: rot_bonds,
        driven_bonds: driven,
        n_candidates,
        n_screened,
    })
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = vec_sub(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn vec_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vdw_radius_bohr(z: u32) -> f64 {
    use crate::core::units::ANGSTROM_TO_BOHR;
    let angstrom = match z {
        1 => 1.20,  // H
        6 => 1.70,  // C
        7 => 1.55,  // N
        8 => 1.52,  // O
        9 => 1.47,  // F
        15 => 1.80, // P
        16 => 1.80, // S
        17 => 1.75, // Cl
        35 => 1.85, // Br
        53 => 1.98, // I
        _ => 2.0,
    };
    angstrom * ANGSTROM_TO_BOHR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    fn butane() -> Molecule {
        let xyz = "14
n-butane anti
C    -1.9255    -0.2545     0.0000
C    -0.6586     0.5872     0.0000
C     0.6586    -0.5872     0.0000
C     1.9255     0.2545     0.0000
H    -2.8190     0.3727     0.0000
H    -1.9698    -0.8907     0.8870
H    -1.9698    -0.8907    -0.8870
H    -0.6285     1.2335     0.8835
H    -0.6285     1.2335    -0.8835
H     0.6285    -1.2335     0.8835
H     0.6285    -1.2335    -0.8835
H     2.8190    -0.3727     0.0000
H     1.9698     0.8907     0.8870
H     1.9698     0.8907    -0.8870
";
        Molecule::from_xyz(xyz).unwrap()
    }

    #[test]
    fn butane_has_one_rotatable_bond() {
        let mol = butane();
        let bonds = rotatable_bonds(&mol);
        assert_eq!(bonds.len(), 1, "rotatable bonds: {bonds:?}");
        assert_eq!(bonds[0], (1, 2));
    }

    #[test]
    fn ethane_terminal_bond_not_rotatable() {
        let xyz = "8
ethane
C  0.000  0.000  0.000
C  1.530  0.000  0.000
H -0.380  1.020  0.000
H -0.380 -0.510  0.880
H -0.380 -0.510 -0.880
H  1.910 -1.020  0.000
H  1.910  0.510  0.880
H  1.910  0.510 -0.880
";
        let mol = Molecule::from_xyz(xyz).unwrap();
        assert!(rotatable_bonds(&mol).is_empty());
    }

    #[test]
    fn water_no_rotatable_bonds() {
        let mol =
            Molecule::from_xyz("3\nw\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap();
        assert!(rotatable_bonds(&mol).is_empty());
    }

    #[test]
    fn connectivity_butane_carbon_chain() {
        let mol = butane();
        let adj = connectivity(&mol);
        assert!(adj[0].contains(&1));
        assert!(adj[1].contains(&2));
        assert!(adj[2].contains(&3));
        assert_eq!(
            adj[1]
                .iter()
                .filter(|&&k| mol.atoms[k].element.z() > 1)
                .count(),
            2
        );
    }

    #[test]
    fn butane_torsion_drive_three_candidates() {
        let mol = butane();
        let opts = ConfGenOptions::default();
        let res = generate_conformers(&mol, &opts, |m| {
            let dih = dihedral(m, 0, 1, 2, 3);
            Some(dih.cos()) // minimized near φ = 180° (anti), where cos = −1
        })
        .unwrap();
        assert_eq!(res.driven_bonds.len(), 1);
        assert_eq!(res.n_candidates, 3);
        assert_eq!(res.ensemble.len(), 3, "ensemble size");
        let lo = &res.ensemble.conformers[0].molecule;
        let phi = dihedral(lo, 0, 1, 2, 3).abs();
        assert!(phi > 2.6, "anti dihedral {phi} rad (expected near π)");
    }

    fn dihedral(m: &Molecule, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let p = |x: usize| m.atoms[x].position;
        let b1 = vec_sub(p(j), p(i));
        let b2 = vec_sub(p(k), p(j));
        let b3 = vec_sub(p(l), p(k));
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let n1 = cross(b1, b2);
        let n2 = cross(b2, b3);
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let m1 = cross(n1, [b2[0], b2[1], b2[2]]);
        let b2n = dot(b2, b2).sqrt();
        let x = dot(n1, n2);
        let y = dot(m1, n2) / b2n;
        y.atan2(x)
    }

    #[test]
    fn no_rotatable_returns_single() {
        let mol =
            Molecule::from_xyz("3\nw\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n").unwrap();
        let res = generate_conformers(&mol, &ConfGenOptions::default(), |_| Some(-76.0)).unwrap();
        assert_eq!(res.n_candidates, 1);
        assert_eq!(res.ensemble.len(), 1);
    }

    #[test]
    fn clash_filter_rejects_overlap() {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(8).unwrap(), [0.2, 0.0, 0.0]),
            ],
            0,
            1,
        );
        let adj = vec![vec![], vec![]]; // declare them non-bonded
        assert!(has_clash(&mol, &adj, 0.6));
    }
}
