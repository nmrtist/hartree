use crate::core::Molecule;
use crate::linalg::{mat_from_row_major, mat_to_row_major, symmetric_eigh};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Internal {
    Bond(usize, usize),
    Angle(usize, usize, usize),
}

const LINEAR_SIN_TOL: f64 = 0.05;

const PINV_REL_TOL: f64 = 1e-8;

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
                }
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
        }
    }
    b
}

pub fn internal_gradient(b: &[f64], gx: &[f64], nq: usize, ndof: usize) -> Vec<f64> {
    let mut bg = vec![0.0; nq];
    for (i, slot) in bg.iter_mut().enumerate() {
        let row = i * ndof;
        let mut s = 0.0;
        for j in 0..ndof {
            s += b[row + j] * gx[j];
        }
        *slot = s;
    }
    let g = gram(b, nq, ndof);
    let ginv = pseudo_inverse(&g, nq);
    matvec(&ginv, &bg, nq)
}

pub fn back_transform(defs: &[Internal], x0: &[[f64; 3]], dq: &[f64]) -> Vec<[f64; 3]> {
    let natom = x0.len();
    let ndof = 3 * natom;
    let nq = defs.len();

    let q0 = values(defs, x0);
    let q_target: Vec<f64> = q0.iter().zip(dq).map(|(a, d)| a + d).collect();

    let mut x = x0.to_vec();
    for _ in 0..50 {
        let b = wilson_b(defs, &x);
        let g = gram(&b, nq, ndof);
        let ginv = pseudo_inverse(&g, nq);
        let qcur = values(defs, &x);
        let dq_rem: Vec<f64> = q_target.iter().zip(&qcur).map(|(t, c)| t - c).collect();
        let gd = matvec(&ginv, &dq_rem, nq); // nq
        let mut max_dx = 0.0_f64;
        for (a, xa) in x.iter_mut().enumerate() {
            for c in 0..3 {
                let mut s = 0.0;
                for i in 0..nq {
                    s += b[i * ndof + (3 * a + c)] * gd[i];
                }
                xa[c] += s;
                max_dx = max_dx.max(s.abs());
            }
        }
        if max_dx < 1e-11 {
            break;
        }
    }
    x
}

fn gram(b: &[f64], nq: usize, ndof: usize) -> Vec<f64> {
    let mut g = vec![0.0; nq * nq];
    for i in 0..nq {
        for j in i..nq {
            let mut s = 0.0;
            for d in 0..ndof {
                s += b[i * ndof + d] * b[j * ndof + d];
            }
            g[i * nq + j] = s;
            g[j * nq + i] = s;
        }
    }
    g
}

fn pseudo_inverse(g: &[f64], nq: usize) -> Vec<f64> {
    if nq == 0 {
        return Vec::new();
    }
    let eig = symmetric_eigh(&mat_from_row_major(nq, g));
    let vectors = mat_to_row_major(&eig.vectors); // column k = eigenvector k
    let lam_max = eig.values.iter().cloned().fold(0.0_f64, f64::max);
    let thresh = PINV_REL_TOL * lam_max.max(1e-300);

    let mut out = vec![0.0; nq * nq];
    for k in 0..nq {
        let lam = eig.values[k];
        if lam <= thresh {
            continue;
        }
        let inv = 1.0 / lam;
        for i in 0..nq {
            let vik = vectors[i * nq + k];
            if vik == 0.0 {
                continue;
            }
            for j in 0..nq {
                out[i * nq + j] += inv * vik * vectors[j * nq + k];
            }
        }
    }
    out
}

fn matvec(m: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for (i, slot) in y.iter_mut().enumerate() {
        let row = i * n;
        let mut s = 0.0;
        for j in 0..n {
            s += m[row + j] * x[j];
        }
        *slot = s;
    }
    y
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    fn h2o() -> Molecule {
        Molecule::new(
            vec![
                Atom::new(Element::from_z(8).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [1.80, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [-0.45, 1.74, 0.0]),
            ],
            0,
            1,
        )
    }

    #[test]
    fn water_has_two_bonds_and_one_angle() {
        let defs = generate(&h2o());
        let bonds = defs
            .iter()
            .filter(|d| matches!(d, Internal::Bond(..)))
            .count();
        let angles = defs
            .iter()
            .filter(|d| matches!(d, Internal::Angle(..)))
            .count();
        assert_eq!(bonds, 2, "O–H bonds");
        assert_eq!(angles, 1, "H–O–H angle");
    }

    #[test]
    fn stretched_diatomic_still_bonds() {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 3.0]),
            ],
            0,
            1,
        );
        let defs = generate(&mol);
        assert_eq!(defs, vec![Internal::Bond(0, 1)]);
    }

    #[test]
    fn wilson_b_matches_finite_difference() {
        let mol = h2o();
        let defs = generate(&mol);
        let x: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
        let b = wilson_b(&defs, &x);
        let ndof = 3 * x.len();
        let h = 1e-6;
        for (row, _) in defs.iter().enumerate() {
            for atom in 0..x.len() {
                for c in 0..3 {
                    let mut xp = x.clone();
                    xp[atom][c] += h;
                    let mut xm = x.clone();
                    xm[atom][c] -= h;
                    let qp = values(&defs, &xp)[row];
                    let qm = values(&defs, &xm)[row];
                    let fd = (qp - qm) / (2.0 * h);
                    let analytic = b[row * ndof + (3 * atom + c)];
                    assert!(
                        (fd - analytic).abs() < 1e-7,
                        "B[{row},{atom},{c}] analytic {analytic} vs FD {fd}"
                    );
                }
            }
        }
    }

    #[test]
    fn back_transform_diatomic_stretch() {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 1.40]),
            ],
            0,
            1,
        );
        let defs = generate(&mol);
        let x: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
        let x_new = back_transform(&defs, &x, &[0.10]); // stretch by 0.1 bohr
        let r = distance(x_new[0], x_new[1]);
        assert!(
            (r - 1.50).abs() < 1e-10,
            "back-transformed bond = {r}, want 1.50"
        );
    }
}
