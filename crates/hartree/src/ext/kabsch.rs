fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    let mut c = [0.0; 3];
    for p in points {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    for ck in &mut c {
        *ck /= n;
    }
    c
}

pub fn kabsch_rmsd(p: &[[f64; 3]], q: &[[f64; 3]]) -> Option<f64> {
    if p.len() != q.len() || p.is_empty() {
        return None;
    }
    let cp = centroid(p);
    let cq = centroid(q);
    let pc: Vec<[f64; 3]> = p.iter().map(|a| sub(*a, cp)).collect();
    let qc: Vec<[f64; 3]> = q.iter().map(|a| sub(*a, cq)).collect();

    let r = optimal_rotation(&pc, &qc);

    let n = p.len() as f64;
    let mut sum = 0.0;
    for (a, b) in pc.iter().zip(&qc) {
        let ra = matvec(&r, *a);
        let d = sub(ra, *b);
        sum += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
    Some((sum / n).sqrt())
}

pub fn optimal_rotation(p: &[[f64; 3]], q: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut h = [[0.0f64; 3]; 3];
    for (a, b) in p.iter().zip(q) {
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += a[i] * b[j];
            }
        }
    }
    let m = mat_mul(&transpose(&h), &h);
    let (eval, evec) = eigh3(&m);

    let mut m_inv_sqrt = [[0.0f64; 3]; 3];
    for k in 0..3 {
        let inv = if eval[k] > 1e-12 {
            1.0 / eval[k].sqrt()
        } else {
            0.0
        };
        for i in 0..3 {
            for j in 0..3 {
                m_inv_sqrt[i][j] += inv * evec[i][k] * evec[j][k];
            }
        }
    }
    let mut r = mat_mul(&h, &m_inv_sqrt);

    if det3(&r) < 0.0 {
        let mut corr = [[0.0f64; 3]; 3];
        let inv = if eval[0] > 1e-12 {
            1.0 / eval[0].sqrt()
        } else {
            0.0
        };
        for i in 0..3 {
            for j in 0..3 {
                corr[i][j] = m_inv_sqrt[i][j] - 2.0 * inv * evec[i][0] * evec[j][0];
            }
        }
        r = mat_mul(&h, &corr);
    }
    transpose(&r)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn matvec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[allow(clippy::needless_range_loop)] // explicit row/col indices read clearer for the Givens update
fn eigh3(a: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut m = *a;
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..50 {
        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
        if off < 1e-15 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if m[p][q].abs() < 1e-18 {
                continue;
            }
            let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let mpp = m[p][p];
            let mqq = m[q][q];
            let mpq = m[p][q];
            m[p][p] = c * c * mpp - 2.0 * s * c * mpq + s * s * mqq;
            m[q][q] = s * s * mpp + 2.0 * s * c * mpq + c * c * mqq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;
            for k in 0..3 {
                if k != p && k != q {
                    let mkp = m[k][p];
                    let mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[p][k] = m[k][p];
                    m[k][q] = s * mkp + c * mkq;
                    m[q][k] = m[k][q];
                }
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut eval = [m[0][0], m[1][1], m[2][2]];
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| eval[i].partial_cmp(&eval[j]).unwrap());
    let sorted_eval = [eval[idx[0]], eval[idx[1]], eval[idx[2]]];
    let mut sorted_vec = [[0.0f64; 3]; 3];
    for (col, &src) in idx.iter().enumerate() {
        for row in 0..3 {
            sorted_vec[row][col] = v[row][src];
        }
    }
    eval = sorted_eval;
    (eval, sorted_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sets_zero_rmsd() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(kabsch_rmsd(&p, &p).unwrap() < 1e-12);
    }

    #[test]
    fn rotated_set_zero_rmsd() {
        let p = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ];
        let q: Vec<[f64; 3]> = p.iter().map(|a| [-a[1], a[0], a[2]]).collect();
        let rmsd = kabsch_rmsd(&p, &q).unwrap();
        assert!(rmsd < 1e-9, "rmsd = {rmsd}");
    }

    #[test]
    fn translated_set_zero_rmsd() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let q: Vec<[f64; 3]> = p.iter().map(|a| [a[0] + 5.0, a[1] - 3.0, a[2]]).collect();
        assert!(kabsch_rmsd(&p, &q).unwrap() < 1e-9);
    }

    #[test]
    fn known_nonzero_rmsd() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let q = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.3, 0.0]];
        let rmsd = kabsch_rmsd(&p, &q).unwrap();
        assert!(rmsd > 0.0 && rmsd < 0.3, "rmsd = {rmsd}");
    }

    #[test]
    fn mismatched_lengths_none() {
        let p = [[0.0, 0.0, 0.0]];
        let q = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(kabsch_rmsd(&p, &q).is_none());
    }

    #[test]
    fn rotation_is_proper() {
        let p = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ];
        let q: Vec<[f64; 3]> = p.iter().map(|a| [-a[1], a[0], a[2]]).collect();
        let cp = centroid(&p);
        let cq = centroid(&q);
        let pc: Vec<[f64; 3]> = p.iter().map(|a| sub(*a, cp)).collect();
        let qc: Vec<[f64; 3]> = q.iter().map(|a| sub(*a, cq)).collect();
        let r = optimal_rotation(&pc, &qc);
        assert!((det3(&r) - 1.0).abs() < 1e-9, "det(R) = {}", det3(&r));
    }
}
