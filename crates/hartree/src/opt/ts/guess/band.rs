//! Per-image IDPP band for a chain-of-states (NEB) search.
//!
//! [`build_ts_guess`](super::build_ts_guess) relaxes a *single* IDPP image at one
//! interpolation fraction; a nudged-elastic-band search instead needs a whole band
//! of interior images between two endpoints. This builds that band by relaxing one
//! [`idpp_image`](super::idpp::idpp_image) per interior knot, at evenly spaced
//! interpolation fractions, so the band starts on a clash-free, distance-interpolated
//! path rather than a straight Cartesian line that would drive atoms through one
//! another.

use super::GuessOptions;
use super::idpp::idpp_image;

/// Relax `n_images` interior IDPP images between `reactant` and `product`, which
/// must already share atom order and a common frame (the caller aligns them).
///
/// Interior image `k` (`1..=n_images`) targets the interatomic-distance matrix
/// interpolated at `λ = k / (n_images + 1)`, so the band spans the *open* interval
/// between the endpoints — image 1 sits nearest the reactant, image `n_images`
/// nearest the product, and the two fixed endpoints are deliberately excluded.
/// `relax` supplies the IDPP relaxation knobs (`idpp_max_iter`, `idpp_tol`, …); its
/// `interpolation` field is overridden per image. Returns `n_images` geometries in
/// the shared atom order (empty when `n_images == 0`).
pub(in crate::opt::ts) fn interpolate_band(
    reactant: &[[f64; 3]],
    product: &[[f64; 3]],
    n_images: usize,
    relax: &GuessOptions,
) -> Vec<Vec<[f64; 3]>> {
    (1..=n_images)
        .map(|k| {
            let lambda = k as f64 / (n_images as f64 + 1.0);
            let opts = GuessOptions {
                interpolation: lambda,
                ..relax.clone()
            };
            idpp_image(reactant, product, &opts)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// A band of `n` interior images excludes the endpoints, preserves atom count,
    /// and is ordered so consecutive images march monotonically from reactant to
    /// product (here, a single atom translating along x while a spectator stays put).
    #[test]
    fn band_excludes_endpoints_and_is_ordered() {
        // Two atoms: atom 0 slides from x=0 (reactant) to x=6 (product); atom 1 is a
        // fixed spectator far away so the pair distance never collapses.
        let reactant = [[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]];
        let product = [[6.0, 0.0, 0.0], [0.0, 8.0, 0.0]];
        let n = 5;
        let band = interpolate_band(&reactant, &product, n, &GuessOptions::default());
        assert_eq!(band.len(), n);
        for image in &band {
            assert_eq!(image.len(), 2);
        }
        // The moving atom's x-coordinate increases monotonically across the band and
        // stays strictly inside the open endpoint interval (0, 6).
        let xs: Vec<f64> = band.iter().map(|im| im[0][0]).collect();
        assert!(
            xs.first().unwrap() > &0.05 && xs.last().unwrap() < &5.95,
            "xs={xs:?}"
        );
        assert!(
            xs.windows(2).all(|w| w[1] > w[0]),
            "moving atom not monotonic across band: {xs:?}"
        );
    }

    /// IDPP keeps a swapping pair apart where straight Cartesian interpolation would
    /// collapse them at the midpoint, image by image across the whole band.
    #[test]
    fn band_images_avoid_clashes() {
        let reactant = [[-3.0, 0.1, 0.0], [3.0, -0.1, 0.0], [0.0, 5.0, 0.0]];
        let product = [[3.0, 0.1, 0.0], [-3.0, -0.1, 0.0], [0.0, 5.0, 0.0]];
        let band = interpolate_band(&reactant, &product, 4, &GuessOptions::default());
        for image in &band {
            let mut min_d = f64::INFINITY;
            for i in 0..image.len() {
                for j in (i + 1)..image.len() {
                    min_d = min_d.min(dist(image[i], image[j]));
                }
            }
            assert!(min_d > 1.0, "band image clashes, min distance {min_d}");
        }
    }

    #[test]
    fn empty_band_for_zero_images() {
        let reactant = [[0.0, 0.0, 0.0]];
        let product = [[1.0, 0.0, 0.0]];
        assert!(interpolate_band(&reactant, &product, 0, &GuessOptions::default()).is_empty());
    }
}
