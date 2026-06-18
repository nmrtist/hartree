use std::f64::consts::PI;

use crate::integrals::integral::periodic::RealSpaceGrid;
use latx::Cell;

const CONV: f64 = 5.3;

#[derive(Debug, Clone)]
pub struct GthLocalAtom {
    pub center: [f64; 3],
    pub z_ion: f64,
    pub r_loc: f64,
    pub c: Vec<f64>,
}

#[must_use]
pub fn core_charge_density(grid: &RealSpaceGrid, atoms: &[GthLocalAtom]) -> Vec<f64> {
    let params: Vec<([f64; 3], f64, f64)> = atoms
        .iter()
        .map(|a| {
            let alpha = 1.0 / (2.0 * a.r_loc * a.r_loc);
            (a.center, alpha, -a.z_ion * (alpha / PI).powf(1.5))
        })
        .collect();
    let cell = grid.cell();
    grid.points()
        .iter()
        .map(|r| {
            let mut acc = 0.0;
            for &(center, alpha, pref) in &params {
                let dr = cell.min_image([r[0] - center[0], r[1] - center[1], r[2] - center[2]]);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                acc += pref * (-alpha * d2).exp();
            }
            acc
        })
        .collect()
}

#[must_use]
pub fn local_pp_short_range(grid: &RealSpaceGrid, atoms: &[GthLocalAtom]) -> Vec<f64> {
    let cell = grid.cell();
    grid.points()
        .iter()
        .map(|r| {
            let mut acc = 0.0;
            for a in atoms {
                if a.c.is_empty() {
                    continue;
                }
                let dr =
                    cell.min_image([r[0] - a.center[0], r[1] - a.center[1], r[2] - a.center[2]]);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                let r_loc2 = a.r_loc * a.r_loc;
                let t = d2 / r_loc2; // (r/r_loc)²
                let mut poly = 0.0;
                for &ci in a.c.iter().rev() {
                    poly = poly * t + ci;
                }
                acc += (-0.5 * t).exp() * poly;
            }
            acc
        })
        .collect()
}

#[must_use]
pub fn self_energy(atoms: &[GthLocalAtom]) -> f64 {
    let sqrt_pi = PI.sqrt();
    -atoms
        .iter()
        .map(|a| a.z_ion * a.z_ion / (2.0 * a.r_loc * sqrt_pi))
        .sum::<f64>()
}

#[must_use]
pub fn overlap_energy(cell: &Cell, atoms: &[GthLocalAtom]) -> f64 {
    if atoms.is_empty() {
        return 0.0;
    }
    let r_max = atoms.iter().map(|a| a.r_loc).fold(0.0_f64, f64::max);
    let gamma_min = 1.0 / (2.0 * r_max.max(1e-12)); // γ for the widest pair (r_I=r_J=r_max)
    let r_cut = CONV / gamma_min;
    let max_pair = max_pair_distance(&atoms.iter().map(|a| a.center).collect::<Vec<_>>());
    let images = cell.lattice_images(r_cut + max_pair);

    let mut e = 0.0;
    for (i, ai) in atoms.iter().enumerate() {
        for (j, aj) in atoms.iter().enumerate() {
            let qq = ai.z_ion * aj.z_ion;
            if qq == 0.0 {
                continue;
            }
            let gamma = 1.0 / (2.0 * (ai.r_loc * ai.r_loc + aj.r_loc * aj.r_loc)).sqrt();
            let d0 = [
                ai.center[0] - aj.center[0],
                ai.center[1] - aj.center[1],
                ai.center[2] - aj.center[2],
            ];
            for &(triple, r) in &images {
                if i == j && triple == [0, 0, 0] {
                    continue;
                }
                let d = [d0[0] + r[0], d0[1] + r[1], d0[2] + r[2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist < 1e-12 || gamma * dist > CONV {
                    continue;
                }
                e += qq * libm::erfc(gamma * dist) / dist;
            }
        }
    }
    0.5 * e
}

#[must_use]
pub fn overlap_forces(cell: &Cell, atoms: &[GthLocalAtom]) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0; 3]; atoms.len()];
    if atoms.is_empty() {
        return forces;
    }
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    let r_max = atoms.iter().map(|a| a.r_loc).fold(0.0_f64, f64::max);
    let gamma_min = 1.0 / (2.0 * r_max.max(1e-12));
    let r_cut = CONV / gamma_min;
    let max_pair = max_pair_distance(&atoms.iter().map(|a| a.center).collect::<Vec<_>>());
    let images = cell.lattice_images(r_cut + max_pair);

    for (k, ak) in atoms.iter().enumerate() {
        if ak.z_ion == 0.0 {
            continue;
        }
        for (j, aj) in atoms.iter().enumerate() {
            let qq = ak.z_ion * aj.z_ion;
            if qq == 0.0 {
                continue;
            }
            let gamma = 1.0 / (2.0 * (ak.r_loc * ak.r_loc + aj.r_loc * aj.r_loc)).sqrt();
            let d0 = [
                ak.center[0] - aj.center[0],
                ak.center[1] - aj.center[1],
                ak.center[2] - aj.center[2],
            ];
            for &(triple, r) in &images {
                if k == j && triple == [0, 0, 0] {
                    continue;
                }
                let d = [d0[0] + r[0], d0[1] + r[1], d0[2] + r[2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist < 1e-12 || gamma * dist > CONV {
                    continue;
                }
                let mag = qq
                    * (libm::erfc(gamma * dist) / (dist * dist)
                        + two_over_sqrt_pi * gamma * (-gamma * gamma * dist * dist).exp() / dist)
                    / dist;
                for axis in 0..3 {
                    forces[k][axis] += mag * d[axis];
                }
            }
        }
    }
    forces
}

#[must_use]
pub fn core_charge_forces(
    grid: &RealSpaceGrid,
    atoms: &[GthLocalAtom],
    v_h: &[f64],
) -> Vec<[f64; 3]> {
    assert_eq!(
        v_h.len(),
        grid.n_points(),
        "Hartree potential length must equal grid points"
    );
    let cell = grid.cell();
    let dv = grid.dv();
    let points = grid.points();
    let mut forces = vec![[0.0; 3]; atoms.len()];
    for (i, a) in atoms.iter().enumerate() {
        if a.z_ion == 0.0 {
            continue;
        }
        let alpha = 1.0 / (2.0 * a.r_loc * a.r_loc);
        let pref = -a.z_ion * (alpha / PI).powf(1.5);
        let two_alpha = 2.0 * alpha;
        let mut acc = [0.0_f64; 3];
        for (g, r) in points.iter().enumerate() {
            let d = cell.min_image([r[0] - a.center[0], r[1] - a.center[1], r[2] - a.center[2]]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let rho = pref * (-alpha * d2).exp();
            let w = v_h[g] * rho;
            for ax in 0..3 {
                acc[ax] += w * d[ax];
            }
        }
        for ax in 0..3 {
            forces[i][ax] = -dv * two_alpha * acc[ax];
        }
    }
    forces
}

#[must_use]
pub fn local_sr_forces(grid: &RealSpaceGrid, atoms: &[GthLocalAtom], n: &[f64]) -> Vec<[f64; 3]> {
    assert_eq!(
        n.len(),
        grid.n_points(),
        "density length must equal grid points"
    );
    let cell = grid.cell();
    let dv = grid.dv();
    let points = grid.points();
    let mut forces = vec![[0.0; 3]; atoms.len()];
    for (i, a) in atoms.iter().enumerate() {
        if a.c.is_empty() {
            continue;
        }
        let r_loc2 = a.r_loc * a.r_loc;
        let mut acc = [0.0_f64; 3];
        for (g, r) in points.iter().enumerate() {
            let d = cell.min_image([r[0] - a.center[0], r[1] - a.center[1], r[2] - a.center[2]]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let t = d2 / r_loc2;
            let mut poly = 0.0;
            for &ci in a.c.iter().rev() {
                poly = poly * t + ci;
            }
            let mut dpoly = 0.0;
            for j in (1..a.c.len()).rev() {
                dpoly = dpoly * t + (j as f64) * a.c[j];
            }
            let dv_dt = (-0.5 * t).exp() * (dpoly - 0.5 * poly);
            let w = n[g] * dv_dt * (2.0 / r_loc2);
            for ax in 0..3 {
                acc[ax] += w * d[ax];
            }
        }
        for ax in 0..3 {
            forces[i][ax] = dv * acc[ax];
        }
    }
    forces
}

#[must_use]
pub fn core_charge_stress(
    grid: &RealSpaceGrid,
    atoms: &[GthLocalAtom],
    v_h: &[f64],
) -> [[f64; 3]; 3] {
    assert_eq!(
        v_h.len(),
        grid.n_points(),
        "Hartree potential length must equal grid points"
    );
    let cell = grid.cell();
    let dv = grid.dv();
    let points = grid.points();
    let mut tau = [[0.0_f64; 3]; 3];
    for a in atoms {
        if a.z_ion == 0.0 {
            continue;
        }
        let alpha = 1.0 / (2.0 * a.r_loc * a.r_loc);
        let pref = -a.z_ion * (alpha / PI).powf(1.5);
        for (g, r) in points.iter().enumerate() {
            let d = cell.min_image([r[0] - a.center[0], r[1] - a.center[1], r[2] - a.center[2]]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let rho = pref * (-alpha * d2).exp();
            let w = v_h[g] * (-2.0 * alpha) * rho;
            for (al, ta) in tau.iter_mut().enumerate() {
                for (be, tab) in ta.iter_mut().enumerate() {
                    *tab += w * d[al] * d[be];
                }
            }
        }
    }
    for ta in &mut tau {
        for tab in ta.iter_mut() {
            *tab *= dv;
        }
    }
    tau
}

#[must_use]
pub fn local_sr_stress(grid: &RealSpaceGrid, atoms: &[GthLocalAtom], n: &[f64]) -> [[f64; 3]; 3] {
    assert_eq!(
        n.len(),
        grid.n_points(),
        "density length must equal grid points"
    );
    let cell = grid.cell();
    let dv = grid.dv();
    let points = grid.points();
    let mut tau = [[0.0_f64; 3]; 3];
    for a in atoms {
        if a.c.is_empty() {
            continue;
        }
        let r_loc2 = a.r_loc * a.r_loc;
        for (g, r) in points.iter().enumerate() {
            let d = cell.min_image([r[0] - a.center[0], r[1] - a.center[1], r[2] - a.center[2]]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let t = d2 / r_loc2;
            let mut poly = 0.0;
            for &ci in a.c.iter().rev() {
                poly = poly * t + ci;
            }
            let mut dpoly = 0.0;
            for j in (1..a.c.len()).rev() {
                dpoly = dpoly * t + (j as f64) * a.c[j];
            }
            let dv_dt = (-0.5 * t).exp() * (dpoly - 0.5 * poly);
            let w = n[g] * dv_dt * (2.0 / r_loc2);
            for (al, ta) in tau.iter_mut().enumerate() {
                for (be, tab) in ta.iter_mut().enumerate() {
                    *tab += w * d[al] * d[be];
                }
            }
        }
    }
    for ta in &mut tau {
        for tab in ta.iter_mut() {
            *tab *= dv;
        }
    }
    tau
}

#[must_use]
pub fn overlap_stress(cell: &Cell, atoms: &[GthLocalAtom]) -> [[f64; 3]; 3] {
    let mut tau = [[0.0_f64; 3]; 3];
    if atoms.is_empty() {
        return tau;
    }
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    let r_max = atoms.iter().map(|a| a.r_loc).fold(0.0_f64, f64::max);
    let gamma_min = 1.0 / (2.0 * r_max.max(1e-12));
    let r_cut = CONV / gamma_min;
    let max_pair = max_pair_distance(&atoms.iter().map(|a| a.center).collect::<Vec<_>>());
    let images = cell.lattice_images(r_cut + max_pair);

    for (i, ai) in atoms.iter().enumerate() {
        for (j, aj) in atoms.iter().enumerate() {
            let qq = ai.z_ion * aj.z_ion;
            if qq == 0.0 {
                continue;
            }
            let gamma = 1.0 / (2.0 * (ai.r_loc * ai.r_loc + aj.r_loc * aj.r_loc)).sqrt();
            let d0 = [
                ai.center[0] - aj.center[0],
                ai.center[1] - aj.center[1],
                ai.center[2] - aj.center[2],
            ];
            for &(triple, r) in &images {
                if i == j && triple == [0, 0, 0] {
                    continue;
                }
                let d = [d0[0] + r[0], d0[1] + r[1], d0[2] + r[2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist < 1e-12 || gamma * dist > CONV {
                    continue;
                }
                let fp = -(libm::erfc(gamma * dist) / (dist * dist)
                    + two_over_sqrt_pi * gamma * (-gamma * gamma * dist * dist).exp() / dist);
                let coeff = 0.5 * qq * fp / dist;
                for (a, ta) in tau.iter_mut().enumerate() {
                    for (b, tab) in ta.iter_mut().enumerate() {
                        *tab += coeff * d[a] * d[b];
                    }
                }
            }
        }
    }
    tau
}

fn max_pair_distance(positions: &[[f64; 3]]) -> f64 {
    let mut m = 0.0_f64;
    for (i, a) in positions.iter().enumerate() {
        for b in &positions[i + 1..] {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            m = m.max(d);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::integral::periodic::hartree;
    use crate::periodic::ewald_energy;

    fn atom(center: [f64; 3], z: f64, r_loc: f64) -> GthLocalAtom {
        GthLocalAtom {
            center,
            z_ion: z,
            r_loc,
            c: vec![],
        }
    }

    fn strain_dirs() -> [[[f64; 3]; 3]; 3] {
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.2, -0.1], [0.2, -0.3, 0.15], [-0.1, 0.15, 0.5]],
        ]
    }

    fn contract(tau: &[[f64; 3]; 3], m: &[[f64; 3]; 3]) -> f64 {
        (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| tau[a][b] * m[a][b])
            .sum()
    }

    #[test]
    fn electrostatics_combination_equals_ewald() {
        let a = 6.0;
        let cell = Cell::cubic(a).unwrap();
        let r_loc = 0.5;
        let frac_pos = [
            ([0.0, 0.0, 0.0], 2.0),
            ([0.5, 0.5, 0.0], 2.0),
            ([0.5, 0.0, 0.5], -2.0),
            ([0.0, 0.5, 0.5], -2.0),
        ];
        let atoms: Vec<GthLocalAtom> = frac_pos
            .iter()
            .map(|&(f, z)| atom([f[0] * a, f[1] * a, f[2] * a], z, r_loc))
            .collect();

        let grid = RealSpaceGrid::from_cutoff(cell, 200.0);
        let rho = core_charge_density(&grid, &atoms);
        let (_v, e_h_core) = hartree(&rho, &grid);
        let e_combo = e_h_core + self_energy(&atoms) + overlap_energy(&cell, &atoms);

        let positions: Vec<[f64; 3]> = atoms.iter().map(|a| a.center).collect();
        let charges: Vec<f64> = atoms.iter().map(|a| a.z_ion).collect();
        let e_ewald = ewald_energy(&cell, &positions, &charges);

        assert!(
            (e_combo - e_ewald).abs() < 1e-3,
            "E_H[ρcore]+E_self+E_ovrl = {e_combo} vs Ewald {e_ewald}"
        );
    }

    #[test]
    fn overlap_energy_matches_brute_force() {
        let cell = Cell::cubic(4.0).unwrap();
        let atoms = [
            atom([0.0, 0.0, 0.0], 4.0, 0.44),
            atom([2.0, 2.0, 2.0], 4.0, 0.44),
        ];
        let got = overlap_energy(&cell, &atoms);

        let mut brute = 0.0;
        for (i, ai) in atoms.iter().enumerate() {
            for (j, aj) in atoms.iter().enumerate() {
                let gamma = 1.0 / (2.0 * (ai.r_loc.powi(2) + aj.r_loc.powi(2))).sqrt();
                for h in -12..=12 {
                    for k in -12..=12 {
                        for l in -12..=12 {
                            if i == j && h == 0 && k == 0 && l == 0 {
                                continue;
                            }
                            let r = cell.frac_to_cart([f64::from(h), f64::from(k), f64::from(l)]);
                            let d = [
                                ai.center[0] - aj.center[0] + r[0],
                                ai.center[1] - aj.center[1] + r[1],
                                ai.center[2] - aj.center[2] + r[2],
                            ];
                            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                            if dist < 1e-12 {
                                continue;
                            }
                            brute += ai.z_ion * aj.z_ion * libm::erfc(gamma * dist) / dist;
                        }
                    }
                }
            }
        }
        brute *= 0.5;
        assert!((got - brute).abs() < 1e-9, "overlap {got} vs brute {brute}");
    }

    #[test]
    fn overlap_forces_match_finite_difference() {
        let cell = Cell::cubic(7.0).unwrap();
        let mk = |pos: &[[f64; 3]]| -> Vec<GthLocalAtom> {
            vec![atom(pos[0], 4.0, 0.44), atom(pos[1], 4.0, 0.44)]
        };
        let positions = [[1.0, 1.2, 0.8], [3.6, 3.1, 3.9]];
        let analytic = overlap_forces(&cell, &mk(&positions));
        let fd = crate::periodic::finite_difference_forces(&positions, 1e-4, |p| {
            Ok(overlap_energy(&cell, &mk(p)))
        })
        .unwrap();
        for k in 0..2 {
            for axis in 0..3 {
                assert!(
                    (analytic[k][axis] - fd[k][axis]).abs() < 1e-7,
                    "atom {k} axis {axis}: analytic {} vs FD {}",
                    analytic[k][axis],
                    fd[k][axis]
                );
            }
        }
        for (a, b) in analytic[0].iter().zip(&analytic[1]) {
            assert!((a + b).abs() < 1e-9);
        }
    }

    #[test]
    fn local_sr_forces_match_finite_difference() {
        let cell = Cell::cubic(7.0).unwrap();
        let grid = RealSpaceGrid::from_cutoff(cell, 150.0);
        let nc = [3.2, 3.6, 3.4];
        let n: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| {
                let d2 = (r[0] - nc[0]).powi(2) + (r[1] - nc[1]).powi(2) + (r[2] - nc[2]).powi(2);
                (-0.6 * d2).exp()
            })
            .collect();
        let mk = |pos: &[[f64; 3]]| -> Vec<GthLocalAtom> {
            vec![
                GthLocalAtom {
                    center: pos[0],
                    z_ion: 4.0,
                    r_loc: 0.44,
                    c: vec![-7.336_102_97],
                },
                GthLocalAtom {
                    center: pos[1],
                    z_ion: 4.0,
                    r_loc: 0.44,
                    c: vec![-7.336_102_97],
                },
            ]
        };
        let positions = [[2.8, 3.1, 2.9], [4.3, 3.9, 4.1]];
        let analytic = local_sr_forces(&grid, &mk(&positions), &n);
        let fd = crate::periodic::finite_difference_forces(&positions, 1e-4, |p| {
            let v = local_pp_short_range(&grid, &mk(p));
            Ok(grid.dv() * n.iter().zip(&v).map(|(&a, &b)| a * b).sum::<f64>())
        })
        .unwrap();
        for k in 0..2 {
            for ax in 0..3 {
                assert!(
                    (analytic[k][ax] - fd[k][ax]).abs() < 1e-6,
                    "atom {k} axis {ax}: analytic {} vs FD {}",
                    analytic[k][ax],
                    fd[k][ax]
                );
            }
        }
    }

    #[test]
    fn core_charge_forces_match_finite_difference() {
        let cell = Cell::cubic(7.0).unwrap();
        let grid = RealSpaceGrid::from_cutoff(cell, 150.0);
        let nc = [3.0, 3.5, 3.7];
        let n: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| {
                let d2 = (r[0] - nc[0]).powi(2) + (r[1] - nc[1]).powi(2) + (r[2] - nc[2]).powi(2);
                0.8 * (-0.5 * d2).exp()
            })
            .collect();
        let mk = |pos: &[[f64; 3]]| -> Vec<GthLocalAtom> {
            vec![atom(pos[0], 4.0, 0.44), atom(pos[1], 4.0, 0.44)]
        };
        let positions = [[2.7, 3.2, 3.0], [4.4, 3.7, 4.2]];
        let atoms = mk(&positions);
        let rho_tot: Vec<f64> = n
            .iter()
            .zip(&core_charge_density(&grid, &atoms))
            .map(|(&a, &b)| a + b)
            .collect();
        let (v_h, _) = hartree(&rho_tot, &grid);
        let analytic = core_charge_forces(&grid, &atoms, &v_h);
        let fd = crate::periodic::finite_difference_forces(&positions, 1e-4, |p| {
            let rho: Vec<f64> = n
                .iter()
                .zip(&core_charge_density(&grid, &mk(p)))
                .map(|(&a, &b)| a + b)
                .collect();
            Ok(hartree(&rho, &grid).1)
        })
        .unwrap();
        for k in 0..2 {
            for ax in 0..3 {
                assert!(
                    (analytic[k][ax] - fd[k][ax]).abs() < 1e-5,
                    "atom {k} axis {ax}: analytic {} vs FD {}",
                    analytic[k][ax],
                    fd[k][ax]
                );
            }
        }
    }

    #[test]
    fn overlap_stress_matches_finite_difference() {
        use crate::periodic::stress::finite_difference_strain_energy;
        let cell = Cell::cubic(7.0).unwrap();
        let positions = [[1.0, 1.2, 0.8], [3.6, 3.1, 3.9]];
        let mk = |p: &[[f64; 3]]| -> Vec<GthLocalAtom> {
            vec![atom(p[0], 4.0, 0.44), atom(p[1], 4.0, 0.44)]
        };
        let tau = overlap_stress(&cell, &mk(&positions));

        let dirs = [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.2, -0.1], [0.2, -0.3, 0.15], [-0.1, 0.15, 0.5]],
        ];
        for m in dirs {
            let fd = finite_difference_strain_energy(&cell, &positions, &m, 1e-4, |c, p| {
                Ok(overlap_energy(c, &mk(p)))
            })
            .unwrap();
            let analytic: f64 = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| tau[a][b] * m[a][b])
                .sum();
            assert!(
                (analytic - fd).abs() < 1e-7,
                "strain {m:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    #[test]
    fn local_sr_stress_matches_finite_difference() {
        use crate::periodic::stress::finite_difference_strain_energy;
        let cell = Cell::cubic(7.0).unwrap();
        let grid = RealSpaceGrid::from_cutoff(cell, 120.0);
        let dims = grid.n();
        let dv0 = grid.dv();
        let nc = [3.2, 3.6, 3.4];
        let n: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| {
                let d2 = (r[0] - nc[0]).powi(2) + (r[1] - nc[1]).powi(2) + (r[2] - nc[2]).powi(2);
                (-0.6 * d2).exp()
            })
            .collect();
        let mk = |p: &[[f64; 3]]| -> Vec<GthLocalAtom> {
            vec![
                GthLocalAtom {
                    center: p[0],
                    z_ion: 4.0,
                    r_loc: 0.44,
                    c: vec![-7.336_102_97],
                },
                GthLocalAtom {
                    center: p[1],
                    z_ion: 4.0,
                    r_loc: 0.44,
                    c: vec![-7.336_102_97],
                },
            ]
        };
        let positions = [[2.8, 3.1, 2.9], [4.3, 3.9, 4.1]];
        let tau = local_sr_stress(&grid, &mk(&positions), &n);
        for m in strain_dirs() {
            let fd = finite_difference_strain_energy(&cell, &positions, &m, 1e-4, |c, p| {
                let g = RealSpaceGrid::new(*c, dims);
                let v = local_pp_short_range(&g, &mk(p));
                Ok(dv0 * n.iter().zip(&v).map(|(&a, &b)| a * b).sum::<f64>())
            })
            .unwrap();
            let analytic = contract(&tau, &m);
            assert!(
                (analytic - fd).abs() < 1e-6,
                "strain {m:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    #[test]
    fn core_charge_stress_matches_finite_difference() {
        use crate::periodic::stress::finite_difference_strain_energy;
        let cell = Cell::cubic(7.0).unwrap();
        let grid = RealSpaceGrid::from_cutoff(cell, 120.0);
        let dims = grid.n();
        let dv0 = grid.dv();
        let vc = [3.0, 3.5, 3.7];
        let v_h: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| {
                let d2 = (r[0] - vc[0]).powi(2) + (r[1] - vc[1]).powi(2) + (r[2] - vc[2]).powi(2);
                0.5 * (-0.4 * d2).exp()
            })
            .collect();
        let mk = |p: &[[f64; 3]]| -> Vec<GthLocalAtom> {
            vec![atom(p[0], 4.0, 0.44), atom(p[1], 4.0, 0.44)]
        };
        let positions = [[2.7, 3.2, 3.0], [4.4, 3.7, 4.2]];
        let tau = core_charge_stress(&grid, &mk(&positions), &v_h);
        for m in strain_dirs() {
            let fd = finite_difference_strain_energy(&cell, &positions, &m, 1e-4, |c, p| {
                let g = RealSpaceGrid::new(*c, dims);
                let rho = core_charge_density(&g, &mk(p));
                Ok(dv0 * v_h.iter().zip(&rho).map(|(&a, &b)| a * b).sum::<f64>())
            })
            .unwrap();
            let analytic = contract(&tau, &m);
            assert!(
                (analytic - fd).abs() < 1e-6,
                "strain {m:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    #[test]
    fn core_charge_integrates_to_minus_z() {
        let cell = Cell::cubic(8.0).unwrap();
        let grid = RealSpaceGrid::from_cutoff(cell, 200.0);
        let atoms = [
            atom([4.0, 4.0, 4.0], 4.0, 0.44),
            atom([2.0, 4.0, 4.0], 6.0, 0.30),
        ];
        let rho = core_charge_density(&grid, &atoms);
        let total = rho.iter().sum::<f64>() * grid.dv();
        assert!(
            (total - (-10.0)).abs() < 1e-6,
            "∫ρ_core = {total}, want −10"
        );
    }
}
