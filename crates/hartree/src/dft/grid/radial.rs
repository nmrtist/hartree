use std::f64::consts::{LN_2, PI};

const TA_XI: [f64; 18] = [
    0.8, 0.9, // H, He
    1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9, // Li..Ne
    1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0, // Na..Ar
];

fn ta_xi(z: u32) -> f64 {
    TA_XI[(z - 1) as usize]
}

pub(crate) fn treutler_ahlrichs(z: u32, n: usize) -> (Vec<f64>, Vec<f64>) {
    let xi = ta_xi(z);
    let step = PI / (n as f64 + 1.0);
    let scale = xi / LN_2;

    let mut radii = vec![0.0; n];
    let mut weights = vec![0.0; n]; // holds dr until the 4π r² factor is folded in
    for i in 0..n {
        let theta = (i + 1) as f64 * step;
        let x = theta.cos();
        let onepx = 1.0 + x;
        let log_term = ((1.0 - x) / 2.0).ln(); // = ln((1-x)/2) < 0
        let pow = onepx.powf(0.6);
        radii[i] = -scale * pow * log_term;
        weights[i] = step * theta.sin() * scale * pow * (-0.6 / onepx * log_term + 1.0 / (1.0 - x));
    }

    radii.reverse();
    weights.reverse();

    for i in 0..n {
        weights[i] *= 4.0 * PI * radii[i] * radii[i];
    }
    (radii, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_integral(alpha: f64) -> f64 {
        (PI / alpha).powf(1.5)
    }

    #[test]
    fn integrates_radial_gaussian() {
        let n = 50;
        for &z in &[1u32, 2, 6, 8, 17] {
            let (radii, weights) = treutler_ahlrichs(z, n);
            assert_eq!(radii.len(), n);
            assert!(
                weights.iter().all(|&w| w > 0.0),
                "z={z}: non-positive weight"
            );
            assert!(radii.iter().all(|&r| r > 0.0), "z={z}: non-positive radius");
            assert!(
                radii.windows(2).all(|w| w[1] > w[0]),
                "z={z}: radii not ascending"
            );
            for &alpha in &[0.5_f64, 1.0, 2.0] {
                let quad: f64 = radii
                    .iter()
                    .zip(&weights)
                    .map(|(&r, &w)| w * (-alpha * r * r).exp())
                    .sum();
                let exact = gaussian_integral(alpha);
                let rel = (quad - exact).abs() / exact;
                assert!(rel < 1e-10, "z={z} alpha={alpha}: rel err {rel:e}");
            }
        }
    }

    #[test]
    fn xi_table_matches_treutler_ahlrichs() {
        assert_eq!(ta_xi(1), 0.8); // H
        assert_eq!(ta_xi(2), 0.9); // He
        assert_eq!(ta_xi(3), 1.8); // Li
        assert_eq!(ta_xi(6), 1.1); // C
        assert_eq!(ta_xi(8), 0.9); // O
        assert_eq!(ta_xi(18), 1.0); // Ar
    }
}
