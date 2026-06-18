pub(crate) const OCC_CUTOFF: f64 = 1e-14;

const SUM_TOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Smearing {
    Fermi { temperature_k: f64 },
}

fn fermi(eps: f64, mu: f64, kt: f64) -> f64 {
    let x = (eps - mu) / kt;
    if x > 700.0 {
        0.0
    } else if x < -700.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

pub(crate) fn fermi_occupations(eps: &[f64], n_target: f64, kt: f64) -> Vec<f64> {
    let m = eps.len();
    if n_target <= 0.0 {
        return vec![0.0; m];
    }
    if n_target >= m as f64 {
        return vec![1.0; m];
    }
    let sum_at = |mu: f64| eps.iter().map(|&e| fermi(e, mu, kt)).sum::<f64>();
    let mut lo = eps[0] - 40.0 * kt - 1.0;
    let mut hi = eps[m - 1] + 40.0 * kt + 1.0;
    let mut mu = 0.5 * (lo + hi);
    for _ in 0..200 {
        mu = 0.5 * (lo + hi);
        let s = sum_at(mu);
        if (s - n_target).abs() < SUM_TOL {
            break;
        }
        if s < n_target {
            lo = mu;
        } else {
            hi = mu;
        }
    }
    eps.iter().map(|&e| fermi(e, mu, kt)).collect()
}

pub(crate) fn entropy_sum(occ: &[f64]) -> f64 {
    occ.iter()
        .map(|&f| {
            if f <= 0.0 || f >= 1.0 {
                0.0
            } else {
                -(f * f.ln() + (1.0 - f) * (1.0 - f).ln())
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupations_conserve_electron_count() {
        let eps = [-1.2, -0.8, -0.5, -0.1, 0.3, 0.9];
        for kt in [1e-6, 1e-3, 0.05] {
            let f = fermi_occupations(&eps, 3.0, kt);
            let total: f64 = f.iter().sum();
            assert!((total - 3.0).abs() < 1e-10, "kT={kt}: Σf = {total}");
            assert!(f.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }

    #[test]
    fn low_temperature_limit_is_integer() {
        let eps = [-1.0, -0.5, 0.5, 1.0];
        let f = fermi_occupations(&eps, 2.0, 1e-8);
        assert!((f[0] - 1.0).abs() < 1e-12);
        assert!((f[1] - 1.0).abs() < 1e-12);
        assert!(f[2] < 1e-12);
        assert!(f[3] < 1e-12);
        assert!(entropy_sum(&f) < 1e-10);
    }

    #[test]
    fn degenerate_levels_share_occupation() {
        let eps = [-1.0, 0.0, 0.0];
        let f = fermi_occupations(&eps, 2.0, 0.01);
        assert!((f[1] - 0.5).abs() < 1e-10);
        assert!((f[2] - 0.5).abs() < 1e-10);
        assert!((entropy_sum(&f) - 2.0 * std::f64::consts::LN_2).abs() < 1e-8);
    }

    #[test]
    fn empty_and_full_channels() {
        let eps = [-1.0, 0.0];
        assert_eq!(fermi_occupations(&eps, 0.0, 0.01), vec![0.0, 0.0]);
        assert_eq!(fermi_occupations(&eps, 2.0, 0.01), vec![1.0, 1.0]);
    }
}
