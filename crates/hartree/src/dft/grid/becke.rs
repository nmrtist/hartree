use crate::core::Molecule;
use crate::core::units::ANGSTROM_TO_BOHR;

const BRAGG_ANGSTROM: [f64; 18] = [
    0.35, 1.40, // H, He
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50, // Li..Ne
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80, // Na..Ar
];

fn bragg_radius_bohr(z: u32) -> f64 {
    BRAGG_ANGSTROM[(z - 1) as usize] * ANGSTROM_TO_BOHR
}

#[inline]
fn becke_step(mu: f64) -> f64 {
    0.5 * mu * (3.0 - mu * mu)
}

#[inline]
fn becke_step_d(mu: f64) -> f64 {
    1.5 * (1.0 - mu * mu)
}

#[inline]
fn becke_smooth3(mu: f64) -> f64 {
    becke_step(becke_step(becke_step(mu)))
}

pub(crate) struct BeckePartition {
    centers: Vec<[f64; 3]>,
    n: usize,
    inv_dist: Vec<f64>,
    a: Vec<f64>,
}

impl BeckePartition {
    pub(crate) fn new(mol: &Molecule) -> Self {
        let n = mol.atoms.len();
        let centers: Vec<[f64; 3]> = mol.atoms.iter().map(|atom| atom.position).collect();
        let mut inv_dist = vec![0.0; n * n];
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            let ri = bragg_radius_bohr(mol.atoms[i].element.z());
            for j in 0..i {
                let rj = bragg_radius_bohr(mol.atoms[j].element.z());
                let (ci, cj) = (centers[i], centers[j]);
                let (dx, dy, dz) = (ci[0] - cj[0], ci[1] - cj[1], ci[2] - cj[2]);
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                inv_dist[i * n + j] = 1.0 / dist;
                a[i * n + j] = (0.25 * (rj / ri - ri / rj)).clamp(-0.5, 0.5);
            }
        }
        Self {
            centers,
            n,
            inv_dist,
            a,
        }
    }

    pub(crate) fn n_atoms(&self) -> usize {
        self.n
    }

    pub(crate) fn weights_into(&self, point: [f64; 3], dist: &mut [f64], out: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            let c = self.centers[k];
            let (dx, dy, dz) = (point[0] - c[0], point[1] - c[1], point[2] - c[2]);
            dist[k] = (dx * dx + dy * dy + dz * dz).sqrt();
            out[k] = 1.0;
        }
        for i in 0..n {
            for j in 0..i {
                let mu = (dist[i] - dist[j]) * self.inv_dist[i * n + j];
                let nu = mu + self.a[i * n + j] * (1.0 - mu * mu);
                let f = becke_smooth3(nu);
                out[i] *= 0.5 * (1.0 - f);
                out[j] *= 0.5 * (1.0 + f);
            }
        }
        let sum: f64 = out[..n].iter().sum();
        let inv = 1.0 / sum;
        for w in out.iter_mut().take(n) {
            *w *= inv;
        }
    }

    pub(crate) fn weight_derivatives(
        &self,
        point: [f64; 3],
        parent: usize,
        out: &mut [[f64; 3]],
    ) -> f64 {
        let n = self.n;
        debug_assert_eq!(out.len(), n);
        for g in out.iter_mut() {
            *g = [0.0; 3];
        }
        if n == 1 {
            return 1.0;
        }

        let mut dist = vec![0.0; n];
        let mut dirv = vec![[0.0; 3]; n];
        for k in 0..n {
            let c = self.centers[k];
            let d = [point[0] - c[0], point[1] - c[1], point[2] - c[2]];
            let dk = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            dist[k] = dk;
            dirv[k] = [d[0] / dk, d[1] / dk, d[2] / dk];
        }

        let mut fac = vec![1.0; n * n];
        let mut dfac = vec![0.0; n * n];
        let mut mu_t = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..i {
                let inv_r = self.inv_dist[i * n + j];
                let a = self.a[i * n + j];
                let mu = (dist[i] - dist[j]) * inv_r;
                let nu = mu + a * (1.0 - mu * mu);
                let f1 = becke_step(nu);
                let f2 = becke_step(f1);
                let f3 = becke_step(f2);
                let dp =
                    becke_step_d(nu) * becke_step_d(f1) * becke_step_d(f2) * (1.0 - 2.0 * a * mu);
                mu_t[i * n + j] = mu;
                fac[i * n + j] = 0.5 * (1.0 - f3);
                fac[j * n + i] = 0.5 * (1.0 + f3);
                dfac[i * n + j] = -0.5 * dp;
                dfac[j * n + i] = 0.5 * dp;
            }
        }

        let mut u = vec![1.0; n];
        for c in 0..n {
            for o in 0..n {
                if o != c {
                    u[c] *= fac[c * n + o];
                }
            }
        }
        let z: f64 = u.iter().sum();
        let p_parent = u[parent] / z;

        let u_excl = |c: usize, skip: usize| -> f64 {
            let mut prod = 1.0;
            for o in 0..n {
                if o != c && o != skip {
                    prod *= fac[c * n + o];
                }
            }
            prod
        };

        let dmu = |i: usize, j: usize, a: usize| -> [f64; 3] {
            let (hi, lo) = if i > j { (i, j) } else { (j, i) };
            let inv_r = self.inv_dist[hi * n + lo];
            let mu = mu_t[hi * n + lo];
            let (ci, cj) = (self.centers[i], self.centers[j]);
            let mut g = [0.0; 3];
            for k in 0..3 {
                let e_ij = (ci[k] - cj[k]) * inv_r; // unit vector R_i ← R_j
                if a == i {
                    g[k] = -dirv[i][k] * inv_r - mu * e_ij * inv_r;
                } else {
                    g[k] = dirv[j][k] * inv_r + mu * e_ij * inv_r;
                }
            }
            g
        };

        for a_at in 0..n {
            if a_at == parent {
                continue;
            }
            let mut du_b = [0.0; 3];
            let mut du_sum = [0.0; 3];
            for c in 0..n {
                let du_c: [f64; 3] = if c == a_at {
                    let mut g = [0.0; 3];
                    for d_at in 0..n {
                        if d_at == a_at {
                            continue;
                        }
                        let pref = u_excl(a_at, d_at) * dfac[a_at * n + d_at];
                        let dm = dmu(a_at.max(d_at), a_at.min(d_at), a_at);
                        for k in 0..3 {
                            g[k] += pref * dm[k];
                        }
                    }
                    g
                } else {
                    let pref = u_excl(c, a_at) * dfac[c * n + a_at];
                    let dm = dmu(c.max(a_at), c.min(a_at), a_at);
                    [pref * dm[0], pref * dm[1], pref * dm[2]]
                };
                if c == parent {
                    du_b = du_c;
                }
                for k in 0..3 {
                    du_sum[k] += du_c[k];
                }
            }
            for k in 0..3 {
                out[a_at][k] = (du_b[k] - p_parent * du_sum[k]) / z;
            }
        }
        let mut tot = [0.0; 3];
        for (a_at, g) in out.iter().enumerate() {
            if a_at != parent {
                for k in 0..3 {
                    tot[k] += g[k];
                }
            }
        }
        out[parent] = [-tot[0], -tot[1], -tot[2]];
        p_parent
    }

    #[cfg(test)]
    pub(crate) fn weights(&self, point: [f64; 3]) -> Vec<f64> {
        let mut dist = vec![0.0; self.n];
        let mut out = vec![0.0; self.n];
        self.weights_into(point, &mut dist, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    fn h2o() -> Molecule {
        Molecule::from_xyz(
            "3\nwater\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
        )
        .unwrap()
    }

    #[test]
    fn bragg_table_spot_check() {
        assert_eq!(BRAGG_ANGSTROM[0], 0.35); // H
        assert_eq!(BRAGG_ANGSTROM[7], 0.60); // O
        assert_eq!(BRAGG_ANGSTROM[16], 1.00); // Cl
        assert!((bragg_radius_bohr(8) - 0.60 * ANGSTROM_TO_BOHR).abs() < 1e-15);
    }

    #[test]
    fn partition_of_unity() {
        let mol = h2o();
        let part = BeckePartition::new(&mol);
        let probes = [
            [0.0, 0.0, 0.0],
            [1.3, -0.7, 0.2],
            [-2.1, 0.4, 1.9],
            [0.05, 0.9, -0.6],
            [10.0, -8.0, 5.0],
            [0.0, 0.0, 0.221], // ~on the O nucleus
        ];
        for p in probes {
            let w = part.weights(p);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "Σw = {sum} at {p:?}");
            assert!(w.iter().all(|&x| (0.0..=1.0 + 1e-12).contains(&x)), "{w:?}");
        }
    }

    #[test]
    fn weight_peaks_on_its_own_nucleus() {
        let mol = h2o();
        let part = BeckePartition::new(&mol);
        let on_o = part.weights(mol.atoms[0].position);
        assert!(on_o[0] > 1.0 - 1e-6, "O weight on O nucleus = {}", on_o[0]);
        let on_h = part.weights(mol.atoms[1].position);
        assert!(on_h[1] > 1.0 - 1e-6, "H weight on H nucleus = {}", on_h[1]);
    }

    #[test]
    fn weight_derivatives_match_fd() {
        let mol = h2o();
        let part = BeckePartition::new(&mol);
        let offsets = [[0.31, -0.22, 0.47], [-0.13, 0.41, 0.29]];
        let eps = 1e-6;
        let n = mol.atoms.len();
        let mut dp = vec![[0.0; 3]; n];
        for parent in 0..n {
            for off in offsets {
                let c = mol.atoms[parent].position;
                let point = [c[0] + off[0], c[1] + off[1], c[2] + off[2]];
                let p0 = part.weight_derivatives(point, parent, &mut dp);
                assert!((p0 - part.weights(point)[parent]).abs() < 1e-14);
                #[allow(clippy::needless_range_loop)] // a/axis index both dp and the FD probe
                for a in 0..n {
                    for axis in 0..3 {
                        let fd = {
                            let probe = |s: f64| {
                                let mut m = mol.clone();
                                m.atoms[a].position[axis] += s * eps;
                                let part2 = BeckePartition::new(&m);
                                let mut pt = point;
                                if a == parent {
                                    pt[axis] += s * eps; // point rides its parent
                                }
                                part2.weights(pt)[parent]
                            };
                            (probe(1.0) - probe(-1.0)) / (2.0 * eps)
                        };
                        let an = dp[a][axis];
                        assert!(
                            (fd - an).abs() < 1e-8,
                            "parent {parent} atom {a} axis {axis}: fd={fd:e} an={an:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn single_atom_owns_everything() {
        let mol = Molecule::new(vec![Atom::new(Element::from_z(8).unwrap(), [0.0; 3])], 0, 1);
        let part = BeckePartition::new(&mol);
        assert_eq!(part.n_atoms(), 1);
        assert_eq!(part.weights([1.2, -3.4, 0.5]), vec![1.0]);
    }
}
