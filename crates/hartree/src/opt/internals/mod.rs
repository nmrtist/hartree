//! Redundant internal coordinates: their definition, automatic generation from a
//! molecule's connectivity, the Wilson B-matrix, and the periodicity-aware change
//! between two coordinate vectors. The linear-algebra transforms that build on the
//! B-matrix (internal gradient/Hessian, completeness rank, and the iterative
//! back-transformation of a step) live in [`transform`]; the tests in [`tests`].

use crate::core::Molecule;

#[cfg(test)]
mod tests;
mod transform;

pub use transform::{back_transform, internal_gradient, internal_hessian, internal_rank};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Internal {
    Bond(usize, usize),
    Angle(usize, usize, usize),
    /// The dihedral (torsion) `i–j–k–l` about the central `j–k` bond, valued in
    /// `(−π, π]`. Completes the redundant set for any molecule with a rotatable bond,
    /// which bonds and valence angles alone cannot span.
    Dihedral(usize, usize, usize, usize),
    /// One of the two co-linear bending coordinates (Bakken–Helgaker, J. Chem. Phys.
    /// 117, 9160 (2002)) replacing the degenerate valence angle `i–k–j` at a near-linear
    /// centre `k`, where the ordinary bend is singular. `axis ∈ {0,1,2}` selects a FIXED
    /// Cartesian cardinal `êₐ`; the value `(e1+e2)·êₐ` (with `e1`,`e2` the unit vectors
    /// from `k` to its two neighbours) measures the bend projected onto that direction,
    /// smooth at and near 180°. Emitted as a perpendicular pair so the two bending DOF
    /// about the axis are both represented.
    LinearBend(usize, usize, usize, usize),
}

const LINEAR_SIN_TOL: f64 = 0.05;

pub fn generate(mol: &Molecule) -> Vec<Internal> {
    let natom = mol.len();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); natom];
    let mut bonds: Vec<(usize, usize)> = Vec::new();
    let connect = |i: usize, j: usize, adj: &mut Vec<Vec<usize>>, b: &mut Vec<(usize, usize)>| {
        b.push((i, j));
        adj[i].push(j);
        adj[j].push(i);
    };

    for i in 0..natom {
        for j in (i + 1)..natom {
            let cutoff = 1.3
                * (mol.atoms[i].element.covalent_radius() + mol.atoms[j].element.covalent_radius());
            if distance(pos[i], pos[j]) < cutoff {
                connect(i, j, &mut adjacency, &mut bonds);
            }
        }
    }

    while let Some((i, j)) = shortest_cross_fragment_pair(&adjacency, &pos, natom) {
        connect(i, j, &mut adjacency, &mut bonds);
    }

    let mut internals: Vec<Internal> = bonds.iter().map(|&(i, j)| Internal::Bond(i, j)).collect();

    for (k, neigh) in adjacency.iter().enumerate() {
        for a in 0..neigh.len() {
            for b in (a + 1)..neigh.len() {
                let (i, j) = (neigh[a], neigh[b]);
                let theta = angle(pos[i], pos[k], pos[j]);
                if theta.sin().abs() > LINEAR_SIN_TOL {
                    internals.push(Internal::Angle(i, k, j));
                } else {
                    // The valence bend at a near-linear (sp) centre is singular, so it is
                    // replaced by two co-linear bending coordinates (Bakken–Helgaker)
                    // measuring the bend in two perpendicular planes. Each references a
                    // FIXED Cartesian cardinal so the coordinate is smooth at 180°; the
                    // two cardinals chosen are those perpendicular to the chain axis (the
                    // axis most aligned with `j−i`), spanning the plane the bend lives in.
                    let chain = sub(pos[j], pos[i]);
                    let a_dom = (0..3)
                        .max_by(|&p, &q| chain[p].abs().total_cmp(&chain[q].abs()))
                        .unwrap();
                    for axis in (0..3).filter(|&ax| ax != a_dom) {
                        internals.push(Internal::LinearBend(i, k, j, axis));
                    }
                }
            }
        }
    }

    // Dihedrals about each bond `j–k`: every neighbour `i` of `j` paired with every
    // neighbour `l` of `k` (the four atoms kept distinct), skipping a torsion whose
    // terminal valence angle `i–j–k` or `j–k–l` is near-linear, where the dihedral is
    // ill-defined. The central bond's stored orientation fixes one direction per
    // torsion, so each appears once.
    for &(j, k) in &bonds {
        for &i in &adjacency[j] {
            if i == k {
                continue;
            }
            for &l in &adjacency[k] {
                if l == j || l == i {
                    continue;
                }
                if angle(pos[i], pos[j], pos[k]).sin().abs() <= LINEAR_SIN_TOL
                    || angle(pos[j], pos[k], pos[l]).sin().abs() <= LINEAR_SIN_TOL
                {
                    continue;
                }
                internals.push(Internal::Dihedral(i, j, k, l));
            }
        }
    }

    internals
}

fn shortest_cross_fragment_pair(
    adjacency: &[Vec<usize>],
    pos: &[[f64; 3]],
    natom: usize,
) -> Option<(usize, usize)> {
    let component = connected_components(adjacency, natom);
    let mut best: Option<(f64, usize, usize)> = None;
    for i in 0..natom {
        for j in (i + 1)..natom {
            if component[i] != component[j] {
                let d = distance(pos[i], pos[j]);
                if best.is_none_or(|(bd, _, _)| d < bd) {
                    best = Some((d, i, j));
                }
            }
        }
    }
    best.map(|(_, i, j)| (i, j))
}

fn connected_components(adjacency: &[Vec<usize>], natom: usize) -> Vec<usize> {
    let mut label = vec![usize::MAX; natom];
    let mut next = 0;
    for start in 0..natom {
        if label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = next;
        while let Some(v) = stack.pop() {
            for &w in &adjacency[v] {
                if label[w] == usize::MAX {
                    label[w] = next;
                    stack.push(w);
                }
            }
        }
        next += 1;
    }
    label
}

pub fn values(defs: &[Internal], coords: &[[f64; 3]]) -> Vec<f64> {
    defs.iter()
        .map(|d| match *d {
            Internal::Bond(i, j) => distance(coords[i], coords[j]),
            Internal::Angle(i, k, j) => angle(coords[i], coords[k], coords[j]),
            Internal::Dihedral(i, j, k, l) => {
                dihedral_angle(coords[i], coords[j], coords[k], coords[l])
            }
            Internal::LinearBend(i, k, j, axis) => {
                let e1 = unit(sub(coords[i], coords[k]));
                let e2 = unit(sub(coords[j], coords[k]));
                e1[axis] + e2[axis]
            }
        })
        .collect()
}

pub fn wilson_b(defs: &[Internal], coords: &[[f64; 3]]) -> Vec<f64> {
    let natom = coords.len();
    let ndof = 3 * natom;
    let nq = defs.len();
    let mut b = vec![0.0; nq * ndof];

    for (row, d) in defs.iter().enumerate() {
        let base = row * ndof;
        match *d {
            Internal::Bond(i, j) => {
                let e = unit(sub(coords[i], coords[j]));
                for c in 0..3 {
                    b[base + 3 * i + c] = e[c];
                    b[base + 3 * j + c] = -e[c];
                }
            }
            Internal::Angle(i, k, j) => {
                let (u, ri) = unit_len(sub(coords[i], coords[k]));
                let (v, rj) = unit_len(sub(coords[j], coords[k]));
                let cos = dot(u, v).clamp(-1.0, 1.0);
                let sin = (1.0 - cos * cos).sqrt().max(1e-12);
                let mut gi = [0.0; 3];
                let mut gj = [0.0; 3];
                for c in 0..3 {
                    gi[c] = (cos * u[c] - v[c]) / (ri * sin);
                    gj[c] = (cos * v[c] - u[c]) / (rj * sin);
                }
                for c in 0..3 {
                    b[base + 3 * i + c] = gi[c];
                    b[base + 3 * j + c] = gj[c];
                    b[base + 3 * k + c] = -(gi[c] + gj[c]);
                }
            }
            Internal::Dihedral(i, j, k, l) => {
                // Sequential bond vectors b1=j−i, b2=k−j (central), b3=l−k. With
                // m=b1×b2 and n=b2×b3, the derivative of the torsion is
                //   ∂φ/∂r_i = −|b2|/|m|² · m,   ∂φ/∂r_l = +|b2|/|n|² · n,
                // and the two central atoms follow by projecting these onto the
                // central bond (p, q) so the four rows sum to zero (rigid invariance).
                let b1 = sub(coords[j], coords[i]);
                let b2 = sub(coords[k], coords[j]);
                let b3 = sub(coords[l], coords[k]);
                let m = cross(b1, b2);
                let n = cross(b2, b3);
                // |m| = |b1|·|b2|·sin(i–j–k) and |n| = |b2|·|b3|·sin(j–k–l): a
                // near-linear terminal angle leaves the torsion plane — and so its
                // gradient — undefined. `generate` skips such torsions at the guess,
                // but a valence angle can swing through linear mid-search; leave the
                // row zero there so the (then locally redundant) coordinate drops
                // cleanly into `G`'s null space instead of injecting a divergent
                // ~1/sin coupling that would scramble the step and the mode tracking.
                let (b1n, b2n, b3n) = (norm(b1), norm(b2), norm(b3));
                let (mn, nn) = (norm(m), norm(n));
                if mn > LINEAR_SIN_TOL * b1n * b2n && nn > LINEAR_SIN_TOL * b2n * b3n {
                    let b2_len2 = b2n * b2n;
                    let gi = scale(m, -b2n / (mn * mn));
                    let gl = scale(n, b2n / (nn * nn));
                    let p = dot(b1, b2) / b2_len2;
                    let q = dot(b3, b2) / b2_len2;
                    for c in 0..3 {
                        let gj = -gi[c] - p * gi[c] + q * gl[c];
                        let gk = -gl[c] + p * gi[c] - q * gl[c];
                        b[base + 3 * i + c] = gi[c];
                        b[base + 3 * j + c] = gj;
                        b[base + 3 * k + c] = gk;
                        b[base + 3 * l + c] = gl[c];
                    }
                }
            }
            Internal::LinearBend(i, k, j, axis) => {
                // L = (e1+e2)·êₐ with e1,e2 the unit vectors from centre k. Each term's
                // gradient is that of a unit vector: ∂(e·êₐ)/∂r = (êₐ − (êₐ·e)e)/r, the
                // component of êₐ perpendicular to e scaled by 1/|r−k|. The centre row is
                // minus the sum of the two end rows, so the row sums to zero (the
                // coordinate is translation-invariant).
                let (e1, r1) = unit_len(sub(coords[i], coords[k]));
                let (e2, r2) = unit_len(sub(coords[j], coords[k]));
                let mut gi = [0.0; 3];
                let mut gj = [0.0; 3];
                for c in 0..3 {
                    let ea = if c == axis { 1.0 } else { 0.0 };
                    gi[c] = (ea - e1[axis] * e1[c]) / r1;
                    gj[c] = (ea - e2[axis] * e2[c]) / r2;
                }
                for c in 0..3 {
                    b[base + 3 * i + c] = gi[c];
                    b[base + 3 * j + c] = gj[c];
                    b[base + 3 * k + c] = -(gi[c] + gj[c]);
                }
            }
        }
    }
    b
}

/// Componentwise `q_to − q_from`, with dihedral components wrapped into `(−π, π]`
/// (bonds and valence angles are plain differences). Used wherever an internal-
/// coordinate *change* is formed by subtracting two coordinate vectors — the
/// quasi-Newton update's step and the back-transformation residual — so a torsion
/// stepping across the ±π branch contributes its true displacement, not a near-2π
/// jump. `defs` labels each component; the three slices share its length.
pub fn displacement(defs: &[Internal], q_to: &[f64], q_from: &[f64]) -> Vec<f64> {
    defs.iter()
        .zip(q_to)
        .zip(q_from)
        .map(|((d, &to), &from)| match d {
            Internal::Dihedral(..) => wrap_to_pi(to - from),
            _ => to - from,
        })
        .collect()
}

/// Wrap an angle difference into the principal interval (`|result| ≤ π`), so a torsion
/// that steps across the ±π branch reports the small true change rather than a near-full
/// turn. A half-turn maps to one representative (±π → ∓π); immaterial for a difference.
pub(super) fn wrap_to_pi(a: f64) -> f64 {
    use std::f64::consts::TAU;
    a - TAU * (a / TAU).round()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

fn unit_len(a: [f64; 3]) -> ([f64; 3], f64) {
    let n = norm(a);
    ([a[0] / n, a[1] / n, a[2] / n], n)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

fn angle(i: [f64; 3], k: [f64; 3], j: [f64; 3]) -> f64 {
    let u = unit(sub(i, k));
    let v = unit(sub(j, k));
    dot(u, v).clamp(-1.0, 1.0).acos()
}

/// The signed dihedral `i–j–k–l` in `(−π, π]` (atan2 form, robust near 0 and π). With
/// sequential bond vectors `b1=j−i`, `b2=k−j`, `b3=l−k`, it is the angle between the
/// planes `(i,j,k)` and `(j,k,l)`, signed by the right-hand rule about `b2`.
fn dihedral_angle(ri: [f64; 3], rj: [f64; 3], rk: [f64; 3], rl: [f64; 3]) -> f64 {
    let b1 = sub(rj, ri);
    let b2 = sub(rk, rj);
    let b3 = sub(rl, rk);
    let c12 = cross(b1, b2);
    let c23 = cross(b2, b3);
    let x = dot(c12, c23);
    let y = dot(b1, c23) * norm(b2);
    y.atan2(x)
}
