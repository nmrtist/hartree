use std::f64::consts::PI;

pub fn n_cart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

pub fn n_func(l: usize, spherical: bool) -> usize {
    if spherical { 2 * l + 1 } else { n_cart(l) }
}

pub fn cart_components(l: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::with_capacity(n_cart(l));
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            out.push([lx, ly, l - lx - ly]);
        }
    }
    out
}

pub fn double_factorial(n: i64) -> f64 {
    let mut k = n;
    let mut acc = 1.0_f64;
    while k > 1 {
        acc *= k as f64;
        k -= 2;
    }
    acc
}

pub fn cart_norm(alpha: f64, lx: usize, ly: usize, lz: usize) -> f64 {
    let l = (lx + ly + lz) as i32;
    let df = double_factorial(2 * lx as i64 - 1)
        * double_factorial(2 * ly as i64 - 1)
        * double_factorial(2 * lz as i64 - 1);
    let two_alpha_over_pi = 2.0 * alpha / PI;
    two_alpha_over_pi.powf(0.75) * (4.0 * alpha).powi(l).sqrt() / df.sqrt()
}

pub fn shell_norm(alpha: f64, l: usize) -> f64 {
    cart_norm(alpha, l, 0, 0)
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

fn binom(n: i64, k: i64) -> f64 {
    if k < 0 || k > n || n < 0 {
        return 0.0;
    }
    factorial(n as usize) / (factorial(k as usize) * factorial((n - k) as usize))
}

fn cos_half_pi(n: i64) -> f64 {
    match n.rem_euclid(4) {
        0 => 1.0,
        2 => -1.0,
        _ => 0.0,
    }
}

fn sin_half_pi(n: i64) -> f64 {
    match n.rem_euclid(4) {
        1 => 1.0,
        3 => -1.0,
        _ => 0.0,
    }
}

fn racah_real_solid_harmonic_coeff(l: usize, m: i64, lx: usize, ly: usize, lz: usize) -> f64 {
    debug_assert_eq!(lx + ly + lz, l);
    let big_m = m.unsigned_abs() as usize;
    if big_m > l {
        return 0.0;
    }
    let cosine = m >= 0;
    let norm = if big_m == 0 {
        (factorial(l - big_m) / factorial(l + big_m)).sqrt()
    } else {
        (2.0 * factorial(l - big_m) / factorial(l + big_m)).sqrt()
    };

    let mut acc = 0.0_f64;
    let kmax = (l - big_m) / 2;
    for k in 0..=kmax {
        let zpow = l - 2 * k - big_m;
        let gamma = (if k % 2 == 0 { 1.0 } else { -1.0 })
            * 2f64.powi(-(l as i32))
            * binom(l as i64, k as i64)
            * binom(2 * l as i64 - 2 * k as i64, l as i64)
            * factorial(l - 2 * k)
            / factorial(l - 2 * k - big_m);
        if gamma == 0.0 {
            continue;
        }
        for a in 0..=k {
            for b in 0..=(k - a) {
                let c = k - a - b;
                for p in 0..=big_m {
                    let q = big_m - p;
                    let tx = 2 * a + p;
                    let ty = 2 * b + q;
                    let tz = 2 * c + zpow;
                    if tx != lx || ty != ly || tz != lz {
                        continue;
                    }
                    let trig = if cosine {
                        cos_half_pi(q as i64)
                    } else {
                        sin_half_pi(q as i64)
                    };
                    if trig == 0.0 {
                        continue;
                    }
                    let multinomial = factorial(k) / (factorial(a) * factorial(b) * factorial(c));
                    acc += gamma * multinomial * binom(big_m as i64, p as i64) * trig;
                }
            }
        }
    }
    norm * acc
}

pub fn m_order(l: usize) -> Vec<i64> {
    if l == 1 {
        return vec![1, -1, 0];
    }
    (-(l as i64)..=(l as i64)).collect()
}

pub fn monomial_to_raw_factor(l: usize) -> f64 {
    if l < 2 {
        1.0
    } else {
        (4.0 * PI / (2 * l + 1) as f64).sqrt()
    }
}

fn gaussian_moment_1d(n: usize, beta: f64) -> f64 {
    if n % 2 == 1 {
        return 0.0;
    }
    let dfac = double_factorial(n as i64 - 1);
    dfac / (2.0 * beta).powi(n as i32 / 2) * (PI / beta).sqrt()
}

fn raw_self_overlap(l: usize, ci: [usize; 3], cj: [usize; 3]) -> f64 {
    let beta = 2.0; // 2α with α = 1
    let bare = gaussian_moment_1d(ci[0] + cj[0], beta)
        * gaussian_moment_1d(ci[1] + cj[1], beta)
        * gaussian_moment_1d(ci[2] + cj[2], beta);
    let n_mono = cart_norm(1.0, l, 0, 0);
    let raw_shell = monomial_to_raw_factor(l).powi(2);
    raw_shell * n_mono * n_mono * bare
}

pub fn c2s_matrix(l: usize) -> Vec<f64> {
    let comps = cart_components(l);
    let ncart = n_cart(l);
    let ms = m_order(l);
    let mut mat = vec![0.0_f64; ms.len() * ncart];

    for (row, &m) in ms.iter().enumerate() {
        let mut coeffs = vec![0.0_f64; ncart];
        for (i, c) in comps.iter().enumerate() {
            coeffs[i] = racah_real_solid_harmonic_coeff(l, m, c[0], c[1], c[2]);
        }
        let mut q = 0.0_f64;
        for (i, ci) in comps.iter().enumerate() {
            if coeffs[i] == 0.0 {
                continue;
            }
            for (j, cj) in comps.iter().enumerate() {
                if coeffs[j] == 0.0 {
                    continue;
                }
                q += coeffs[i] * coeffs[j] * raw_self_overlap(l, *ci, *cj);
            }
        }
        let kappa = 1.0 / q.sqrt();
        for i in 0..ncart {
            mat[row * ncart + i] = kappa * coeffs[i];
        }
    }
    mat
}

pub fn shell_transform(l: usize) -> Vec<f64> {
    let ratio = monomial_to_raw_factor(l);
    c2s_matrix(l).iter().map(|&c| ratio * c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cart_ordering_p_and_d() {
        assert_eq!(cart_components(1), vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        assert_eq!(
            cart_components(2),
            vec![
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]
            ]
        );
    }

    #[test]
    fn double_factorial_small() {
        assert_eq!(double_factorial(-1), 1.0);
        assert_eq!(double_factorial(5), 15.0);
        assert_eq!(double_factorial(7), 105.0);
    }

    #[test]
    fn cart_norm_unit_self_overlap() {
        for &alpha in &[0.1, 0.8, 3.3, 25.0] {
            for l in 0..=3 {
                let n = cart_norm(alpha, l, 0, 0);
                let raw = double_factorial(2 * l as i64 - 1) / (4.0 * alpha).powi(l as i32)
                    * (PI / (2.0 * alpha)).powf(1.5);
                assert!((n * n * raw - 1.0).abs() < 1e-13, "alpha={alpha} l={l}");
            }
        }
    }

    #[test]
    fn monomial_to_raw_factor_values() {
        assert_eq!(monomial_to_raw_factor(0), 1.0);
        assert_eq!(monomial_to_raw_factor(1), 1.0);
        assert!((monomial_to_raw_factor(2) - (4.0 * PI / 5.0).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn dz2_racah_pattern() {
        let c = |lx, ly, lz| racah_real_solid_harmonic_coeff(2, 0, lx, ly, lz);
        assert!((c(0, 0, 2) - 1.0).abs() < 1e-14);
        assert!((c(2, 0, 0) + 0.5).abs() < 1e-14);
        assert!((c(0, 2, 0) + 0.5).abs() < 1e-14);
        assert!(c(1, 1, 0).abs() < 1e-14);
    }

    #[test]
    fn c2s_orthonormal() {
        for l in 0..=4 {
            let c = c2s_matrix(l);
            let comps = cart_components(l);
            let ncart = n_cart(l);
            let nsph = 2 * l + 1;
            let mut s = vec![0.0; ncart * ncart];
            for (i, ci) in comps.iter().enumerate() {
                for (j, cj) in comps.iter().enumerate() {
                    s[i * ncart + j] = raw_self_overlap(l, *ci, *cj);
                }
            }
            for p in 0..nsph {
                for qq in 0..nsph {
                    let mut g = 0.0;
                    for i in 0..ncart {
                        let mut si = 0.0;
                        for j in 0..ncart {
                            si += s[i * ncart + j] * c[qq * ncart + j];
                        }
                        g += c[p * ncart + i] * si;
                    }
                    let expect = if p == qq { 1.0 } else { 0.0 };
                    assert!((g - expect).abs() < 1e-12, "l={l} G[{p},{qq}]={g}");
                }
            }
        }
    }
}
