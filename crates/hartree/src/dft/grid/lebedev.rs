#![allow(clippy::excessive_precision)]

mod tables;

pub const ORDERS: [usize; 17] = [
    6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590,
];

#[derive(Debug, Clone)]
pub struct LebedevGrid {
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
}

impl LebedevGrid {
    pub fn new(npts: usize) -> Option<Self> {
        let table = generators(npts)?;
        let mut points = Vec::with_capacity(npts);
        let mut weights = Vec::with_capacity(npts);
        for &entry in table {
            gen_oh(entry, &mut points, &mut weights);
        }
        debug_assert_eq!(points.len(), npts, "Lebedev {npts}: wrong point count");
        Some(Self { points, weights })
    }
}

pub fn degree(npts: usize) -> Option<usize> {
    Some(match npts {
        6 => 3,
        14 => 5,
        26 => 7,
        38 => 9,
        50 => 11,
        74 => 13,
        86 => 15,
        110 => 17,
        146 => 19,
        170 => 21,
        194 => 23,
        230 => 25,
        266 => 27,
        302 => 29,
        350 => 31,
        434 => 35,
        590 => 41,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct Gen {
    code: u8,
    a: f64,
    b: f64,
    v: f64,
}

const fn g(code: u8, a: f64, b: f64, v: f64) -> Gen {
    Gen { code, a, b, v }
}

fn gen_oh(entry: Gen, points: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    let Gen { code, a, b, v } = entry;
    let mut push = |p: [f64; 3]| {
        points.push(p);
        weights.push(v);
    };
    match code {
        0 => {
            let a = 1.0;
            push([a, 0.0, 0.0]);
            push([-a, 0.0, 0.0]);
            push([0.0, a, 0.0]);
            push([0.0, -a, 0.0]);
            push([0.0, 0.0, a]);
            push([0.0, 0.0, -a]);
        }
        1 => {
            let a = (0.5_f64).sqrt();
            push([0.0, a, a]);
            push([0.0, -a, a]);
            push([0.0, a, -a]);
            push([0.0, -a, -a]);
            push([a, 0.0, a]);
            push([-a, 0.0, a]);
            push([a, 0.0, -a]);
            push([-a, 0.0, -a]);
            push([a, a, 0.0]);
            push([-a, a, 0.0]);
            push([a, -a, 0.0]);
            push([-a, -a, 0.0]);
        }
        2 => {
            let a = (1.0_f64 / 3.0).sqrt();
            push([a, a, a]);
            push([-a, a, a]);
            push([a, -a, a]);
            push([-a, -a, a]);
            push([a, a, -a]);
            push([-a, a, -a]);
            push([a, -a, -a]);
            push([-a, -a, -a]);
        }
        3 => {
            let b = (1.0 - 2.0 * a * a).sqrt();
            push([a, a, b]);
            push([-a, a, b]);
            push([a, -a, b]);
            push([-a, -a, b]);
            push([a, a, -b]);
            push([-a, a, -b]);
            push([a, -a, -b]);
            push([-a, -a, -b]);
            push([a, b, a]);
            push([-a, b, a]);
            push([a, -b, a]);
            push([-a, -b, a]);
            push([a, b, -a]);
            push([-a, b, -a]);
            push([a, -b, -a]);
            push([-a, -b, -a]);
            push([b, a, a]);
            push([-b, a, a]);
            push([b, -a, a]);
            push([-b, -a, a]);
            push([b, a, -a]);
            push([-b, a, -a]);
            push([b, -a, -a]);
            push([-b, -a, -a]);
        }
        4 => {
            let b = (1.0 - a * a).sqrt();
            push([a, b, 0.0]);
            push([-a, b, 0.0]);
            push([a, -b, 0.0]);
            push([-a, -b, 0.0]);
            push([b, a, 0.0]);
            push([-b, a, 0.0]);
            push([b, -a, 0.0]);
            push([-b, -a, 0.0]);
            push([a, 0.0, b]);
            push([-a, 0.0, b]);
            push([a, 0.0, -b]);
            push([-a, 0.0, -b]);
            push([b, 0.0, a]);
            push([-b, 0.0, a]);
            push([b, 0.0, -a]);
            push([-b, 0.0, -a]);
            push([0.0, a, b]);
            push([0.0, -a, b]);
            push([0.0, a, -b]);
            push([0.0, -a, -b]);
            push([0.0, b, a]);
            push([0.0, -b, a]);
            push([0.0, b, -a]);
            push([0.0, -b, -a]);
        }
        5 => {
            let c = (1.0 - a * a - b * b).sqrt();
            push([a, b, c]);
            push([-a, b, c]);
            push([a, -b, c]);
            push([-a, -b, c]);
            push([a, b, -c]);
            push([-a, b, -c]);
            push([a, -b, -c]);
            push([-a, -b, -c]);
            push([a, c, b]);
            push([-a, c, b]);
            push([a, -c, b]);
            push([-a, -c, b]);
            push([a, c, -b]);
            push([-a, c, -b]);
            push([a, -c, -b]);
            push([-a, -c, -b]);
            push([b, a, c]);
            push([-b, a, c]);
            push([b, -a, c]);
            push([-b, -a, c]);
            push([b, a, -c]);
            push([-b, a, -c]);
            push([b, -a, -c]);
            push([-b, -a, -c]);
            push([b, c, a]);
            push([-b, c, a]);
            push([b, -c, a]);
            push([-b, -c, a]);
            push([b, c, -a]);
            push([-b, c, -a]);
            push([b, -c, -a]);
            push([-b, -c, -a]);
            push([c, a, b]);
            push([-c, a, b]);
            push([c, -a, b]);
            push([-c, -a, b]);
            push([c, a, -b]);
            push([-c, a, -b]);
            push([c, -a, -b]);
            push([-c, -a, -b]);
            push([c, b, a]);
            push([-c, b, a]);
            push([c, -b, a]);
            push([-c, -b, a]);
            push([c, b, -a]);
            push([-c, b, -a]);
            push([c, -b, -a]);
            push([-c, -b, -a]);
        }
        _ => unreachable!("invalid Lebedev gen_oh code {code}"),
    }
}

fn generators(npts: usize) -> Option<&'static [Gen]> {
    use tables::*;
    Some(match npts {
        6 => LD0006,
        14 => LD0014,
        26 => LD0026,
        38 => LD0038,
        50 => LD0050,
        74 => LD0074,
        86 => LD0086,
        110 => LD0110,
        146 => LD0146,
        170 => LD0170,
        194 => LD0194,
        230 => LD0230,
        266 => LD0266,
        302 => LD0302,
        350 => LD0350,
        434 => LD0434,
        590 => LD0590,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_factorial_odd(n: i64) -> f64 {
        let mut r = 1.0;
        let mut k = n;
        while k > 1 {
            r *= k as f64;
            k -= 2;
        }
        r
    }

    fn sphere_average(a: u32, b: u32, c: u32) -> f64 {
        if a % 2 == 1 || b % 2 == 1 || c % 2 == 1 {
            return 0.0;
        }
        let (a, b, c) = (a as i64, b as i64, c as i64);
        double_factorial_odd(a - 1) * double_factorial_odd(b - 1) * double_factorial_odd(c - 1)
            / double_factorial_odd(a + b + c + 1)
    }

    #[test]
    fn point_counts_and_weight_sums() {
        // Orders 74, 230, 266 each carry a few small negative weights (a published
        // property of those Lebedev–Laikov rules), so positivity is asserted only for
        // the rest; every order must still be finite and sum to unity.
        const NEGATIVE_WEIGHT_ORDERS: [usize; 3] = [74, 230, 266];
        for &npts in &ORDERS {
            let grid = LebedevGrid::new(npts).unwrap();
            assert_eq!(grid.points.len(), npts, "npts={npts}: point count");
            assert_eq!(grid.weights.len(), npts, "npts={npts}: weight count");
            assert!(
                grid.weights.iter().all(|&w| w.is_finite()),
                "npts={npts}: non-finite weight"
            );
            let sum: f64 = grid.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "npts={npts}: Σw = {sum}");
            if !NEGATIVE_WEIGHT_ORDERS.contains(&npts) {
                assert!(
                    grid.weights.iter().all(|&w| w > 0.0),
                    "npts={npts}: expected all-positive weights"
                );
            }
        }
        assert!(LebedevGrid::new(7).is_none());
        assert!(LebedevGrid::new(75).is_none());
        assert!(degree(75).is_none());
    }

    #[test]
    fn points_are_unit_vectors() {
        for &npts in &ORDERS {
            let grid = LebedevGrid::new(npts).unwrap();
            for p in &grid.points {
                let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                assert!((r2 - 1.0).abs() < 1e-13, "npts={npts}: |r|² = {r2}");
            }
        }
    }

    #[test]
    fn exact_for_all_monomials_up_to_degree() {
        let mut global_max = 0.0_f64;
        for &npts in &ORDERS {
            let grid = LebedevGrid::new(npts).unwrap();
            let deg = degree(npts).unwrap();
            let n = grid.points.len();
            let stride = deg + 1;

            let mut px = vec![0.0; n * stride];
            let mut py = vec![0.0; n * stride];
            let mut pz = vec![0.0; n * stride];
            for (i, p) in grid.points.iter().enumerate() {
                let (mut ex, mut ey, mut ez) = (1.0, 1.0, 1.0);
                for k in 0..stride {
                    px[i * stride + k] = ex;
                    py[i * stride + k] = ey;
                    pz[i * stride + k] = ez;
                    ex *= p[0];
                    ey *= p[1];
                    ez *= p[2];
                }
            }

            let mut max_err = 0.0_f64;
            let (mut worst, mut worst_q, mut worst_a) = ((0, 0, 0), 0.0, 0.0);
            for a in 0..=deg {
                for b in 0..=(deg - a) {
                    for c in 0..=(deg - a - b) {
                        let analytic = sphere_average(a as u32, b as u32, c as u32);
                        let mut quad = 0.0;
                        for i in 0..n {
                            quad += grid.weights[i]
                                * px[i * stride + a]
                                * py[i * stride + b]
                                * pz[i * stride + c];
                        }
                        let err = (quad - analytic).abs();
                        if err > max_err {
                            max_err = err;
                            worst = (a, b, c);
                            worst_q = quad;
                            worst_a = analytic;
                        }
                    }
                }
            }
            assert!(
                max_err < 1e-12,
                "npts={npts} deg={deg}: max monomial error {max_err:e} at \
                 x^{}y^{}z^{} (quad={worst_q}, exact={worst_a})",
                worst.0,
                worst.1,
                worst.2
            );
            global_max = global_max.max(max_err);
        }
        println!("Lebedev max monomial error over all shipped orders: {global_max:e}");
    }
}
