//! IDPP image: interpolate the interatomic-distance matrix between two endpoints
//! and relax an image to match.

use super::{GuessOptions, distance};

/// Relax an image toward the `λ`-interpolated target distances `S = Σ w_ij (r_ij -
/// d_ij)²` with `w_ij = 1/d_ij^4`, starting from the straight Cartesian interpolation.
pub(super) fn idpp_image(
    reactant: &[[f64; 3]],
    product: &[[f64; 3]],
    options: &GuessOptions,
) -> Vec<[f64; 3]> {
    let n = reactant.len();
    let lambda = options.interpolation.clamp(0.0, 1.0);

    let mut target = vec![0.0f64; n * n];
    let mut weight = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dr = distance(reactant[i], reactant[j]);
            let dp = distance(product[i], product[j]);
            let d = (1.0 - lambda) * dr + lambda * dp;
            // Coincident atoms carry no distance constraint (and would blow up 1/d^4).
            if d < 1e-3 {
                continue;
            }
            target[i * n + j] = d;
            target[j * n + i] = d;
            let w = 1.0 / d.powi(4);
            weight[i * n + j] = w;
            weight[j * n + i] = w;
        }
    }

    let mut x: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            [
                (1.0 - lambda) * reactant[i][0] + lambda * product[i][0],
                (1.0 - lambda) * reactant[i][1] + lambda * product[i][1],
                (1.0 - lambda) * reactant[i][2] + lambda * product[i][2],
            ]
        })
        .collect();

    let mut objective = idpp_objective(&x, &target, &weight, n);
    let mut step = 0.05;
    for _ in 0..options.idpp_max_iter {
        let grad = idpp_gradient(&x, &target, &weight, n);
        let gmax = grad
            .iter()
            .flat_map(|g| g.iter())
            .fold(0.0f64, |m, &c| m.max(c.abs()));
        if gmax < options.idpp_tol {
            break;
        }
        let mut accepted = false;
        for _ in 0..10 {
            let trial: Vec<[f64; 3]> = x
                .iter()
                .zip(&grad)
                .map(|(xi, gi)| {
                    [
                        xi[0] - step * gi[0],
                        xi[1] - step * gi[1],
                        xi[2] - step * gi[2],
                    ]
                })
                .collect();
            let obj = idpp_objective(&trial, &target, &weight, n);
            if obj < objective {
                x = trial;
                objective = obj;
                step *= 1.2;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    x
}

fn idpp_objective(x: &[[f64; 3]], target: &[f64], weight: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = distance(x[i], x[j]);
            let diff = r - target[i * n + j];
            s += weight[i * n + j] * diff * diff;
        }
    }
    s
}

fn idpp_gradient(x: &[[f64; 3]], target: &[f64], weight: &[f64], n: usize) -> Vec<[f64; 3]> {
    let mut g = vec![[0.0; 3]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
            for c in d.iter_mut() {
                *c /= r;
            }
            let coeff = 2.0 * weight[i * n + j] * (r - target[i * n + j]);
            for c in 0..3 {
                g[i][c] += coeff * d[c];
                g[j][c] -= coeff * d[c];
            }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::GuessOptions;
    use super::*;

    /// Minimum pairwise distance over a point set.
    fn min_pair_distance(pts: &[[f64; 3]]) -> f64 {
        let mut m = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                m = m.min(distance(pts[i], pts[j]));
            }
        }
        m
    }

    #[test]
    fn idpp_avoids_a_clash_that_linear_interpolation_creates() {
        // A and B swap places; C is a spectator that breaks the symmetry. A small
        // transverse (y) offset keeps the swap from putting A and B at *exactly* the
        // same midpoint — coincident atoms have a vanishing IDPP gradient direction,
        // so they would never separate. With the offset they still pass through a
        // genuine clash (linear midpoint < 0.5 Bohr apart) but along a defined axis.
        let reactant = [[-3.0, 0.1, 0.0], [3.0, -0.1, 0.0], [0.0, 5.0, 0.0]];
        let product = [[3.0, 0.1, 0.0], [-3.0, -0.1, 0.0], [0.0, 5.0, 0.0]];

        // First confirm the straight Cartesian midpoint really collides A and B.
        let mid: Vec<[f64; 3]> = (0..reactant.len())
            .map(|i| {
                [
                    0.5 * reactant[i][0] + 0.5 * product[i][0],
                    0.5 * reactant[i][1] + 0.5 * product[i][1],
                    0.5 * reactant[i][2] + 0.5 * product[i][2],
                ]
            })
            .collect();
        assert!(
            min_pair_distance(&mid) < 0.5,
            "linear interpolation should clash, min distance {}",
            min_pair_distance(&mid)
        );

        // IDPP targets the (i,j) distance matrix; d(A,B)=6 at both endpoints, so the
        // relaxed image should keep A and B well separated.
        let opts = GuessOptions {
            interpolation: 0.5,
            ..Default::default()
        };
        let image = idpp_image(&reactant, &product, &opts);
        let min_d = min_pair_distance(&image);
        assert!(
            min_d > 1.0,
            "IDPP should avoid the clash, but min distance is {min_d}"
        );
    }
}
