//! Energy-peaked scan of the IDPP path: a cheap "poor man's path method".
//!
//! [`build_ts_guess`](super::build_ts_guess) places the guess at a fixed interpolation
//! fraction; this instead evaluates a surface's energy at a handful of IDPP images
//! spanning the path, parabola-fits the energy maximum, and returns the image at the
//! fitted peak plus the path tangent there. Both are a strictly better single-point
//! transition-state guess than a geometric midpoint when a surface is affordable, and
//! the tangent is the reaction-coordinate seed the local refiner wants.
//!
//! All images are built by [`idpp_image`](super::idpp::idpp_image) from the *same*
//! co-framed straight-line interpolation between the assembled endpoints, so a central
//! difference of two neighbouring images is the minimum-energy-path tangent without a
//! rigid-body realignment between them.

use super::GuessOptions;
use super::idpp::idpp_image;
use crate::opt::{OptError, Surface};

/// The result of [`scan_peak`]: the guess geometry at the fitted energy maximum and the
/// (unit) minimum-energy-path tangent there, one Cartesian vector per atom.
#[derive(Debug)]
pub(super) struct Peak {
    pub geometry: Vec<[f64; 3]>,
    pub tangent: Vec<[f64; 3]>,
}

/// Scan the IDPP path between two assembled endpoints, returning the geometry at the
/// parabola-fitted energy maximum and the path tangent there.
///
/// `n_points` interior images at `λ = k/(n_points+1)` (`k = 1..=n_points`) are each
/// IDPP-relaxed and energy-evaluated; the discrete maximum is refined to a sub-grid `λ*`
/// by a three-point parabola fit (falling back to the discrete peak at a boundary or a
/// non-concave triple). The tangent is a central difference of the IDPP images a half
/// grid-step either side of `λ*`. The caller guarantees `n_points >= 3`.
///
/// # Errors
/// [`OptError`] if any surface energy evaluation fails.
pub(super) fn scan_peak<S: Surface>(
    reactant_endpoint: &[[f64; 3]],
    prod_in_r: &[[f64; 3]],
    options: &GuessOptions,
    n_points: usize,
    surface: &mut S,
) -> Result<Peak, OptError> {
    let image_at = |lambda: f64| -> Vec<[f64; 3]> {
        let opts = GuessOptions {
            interpolation: lambda,
            ..options.clone()
        };
        idpp_image(reactant_endpoint, prod_in_r, &opts)
    };

    // Even interior grid on the open interval (0, 1); the two endpoints are minima and
    // are excluded. `step` is the spacing between adjacent λ.
    let step = 1.0 / (n_points as f64 + 1.0);
    let lambdas: Vec<f64> = (1..=n_points).map(|k| k as f64 * step).collect();

    let mut energies = Vec::with_capacity(n_points);
    for &lambda in &lambdas {
        let e = surface.energy(&image_at(lambda))?;
        // A non-finite energy that the surface returned as `Ok` (e.g. an SCF that
        // converged to garbage without erroring) would win the `total_cmp` argmax below
        // — a positive NaN orders above every finite value — and then flow through the
        // parabola fit and IDPP interpolation, silently corrupting the guess. Surface it
        // as an error instead, mirroring the non-finite guards on the saddle-search path.
        if !e.is_finite() {
            return Err(OptError::Evaluation(format!(
                "non-finite surface energy {e} at interpolation fraction {lambda}"
            )));
        }
        energies.push(e);
    }

    // The discrete peak, refined to a sub-grid λ* by a parabola through it and its two
    // neighbours (when it has both).
    let peak = (0..n_points)
        .max_by(|&a, &b| energies[a].total_cmp(&energies[b]))
        .ok_or_else(|| OptError::Evaluation("energy scan has no grid points".to_string()))?;
    let lambda_star = if peak == 0 || peak == n_points - 1 {
        lambdas[peak]
    } else {
        parabola_vertex(
            lambdas[peak],
            step,
            energies[peak - 1],
            energies[peak],
            energies[peak + 1],
        )
    };

    let geometry = image_at(lambda_star);

    // Path tangent: central difference of the images a half grid-step either side of the
    // peak, clamped into [0, 1]. The shared interpolation frame makes this the MEP
    // tangent; if it degenerates (coincident clamps), fall back to the overall path
    // direction reactant → product.
    let half = 0.5 * step;
    let lo = image_at((lambda_star - half).max(0.0));
    let hi = image_at((lambda_star + half).min(1.0));
    let mut tangent: Vec<[f64; 3]> = lo
        .iter()
        .zip(&hi)
        .map(|(a, b)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]])
        .collect();
    if !normalize(&mut tangent) {
        tangent = reactant_endpoint
            .iter()
            .zip(prod_in_r)
            .map(|(a, b)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]])
            .collect();
        normalize(&mut tangent);
    }

    Ok(Peak { geometry, tangent })
}

/// Vertex (in `x`) of the parabola through `(x1 - h, y0)`, `(x1, y1)`, `(x1 + h, y2)`,
/// clamped to the bracket `[x1 - h, x1 + h]`. Returns `x1` for a non-concave or flat
/// triple (no distinct maximum to interpolate). For a genuine bracketed maximum
/// (`y1 ≥ y0, y2`) the unclamped vertex already lies within `±h/2`; the clamp only
/// guards a near-degenerate denominator.
fn parabola_vertex(x1: f64, h: f64, y0: f64, y1: f64, y2: f64) -> f64 {
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-15 {
        return x1;
    }
    let offset = h * (y0 - y2) / (2.0 * denom);
    (x1 + offset).clamp(x1 - h, x1 + h)
}

/// Normalize a per-atom vector field in place; returns `false` (leaving it unchanged) if
/// its norm is below a small threshold.
fn normalize(v: &mut [[f64; 3]]) -> bool {
    let norm = v.iter().flatten().map(|c| c * c).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return false;
    }
    for a in v.iter_mut() {
        for c in a.iter_mut() {
            *c /= norm;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::distance;
    use super::*;

    /// A surface whose energy is a downward parabola in the atom-0/atom-1 distance,
    /// peaking at `target`.
    struct PairPeak {
        target: f64,
    }
    impl Surface for PairPeak {
        fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
            Ok(-(distance(x[0], x[1]) - self.target).powi(2))
        }
        fn analytic_gradient(
            &mut self,
            _x: &[[f64; 3]],
        ) -> Option<Result<Vec<[f64; 3]>, OptError>> {
            None
        }
    }

    #[test]
    fn parabola_vertex_recovers_a_known_maximum() {
        // y = -(x - 0.32)^2 sampled symmetrically about x1 = 0.30 (h = 0.05).
        let f = |x: f64| -(x - 0.32f64).powi(2);
        let v = parabola_vertex(0.30, 0.05, f(0.25), f(0.30), f(0.35));
        assert!((v - 0.32).abs() < 1e-9, "vertex {v} != 0.32");
    }

    #[test]
    fn parabola_vertex_handles_a_flat_triple() {
        // A perfectly flat triple has no distinct vertex: return the middle point.
        assert_eq!(parabola_vertex(0.5, 0.1, 1.0, 1.0, 1.0), 0.5);
    }

    #[test]
    fn parabola_vertex_clamps_a_near_degenerate_denominator() {
        // A denominator just above the flat threshold would throw the unclamped vertex
        // far outside the bracket; the clamp keeps it in [x1 - h, x1 + h].
        let v = parabola_vertex(0.5, 0.1, 1.0, 1.0, 1.0 + 1e-13);
        assert!((0.4..=0.6).contains(&v), "vertex {v} escaped the bracket");
    }

    #[test]
    fn scan_finds_the_midpoint_peak_and_axial_tangent() {
        // Atom 0 at x=0 (reactant) → x=2 (product); atom 1 at x=6 → x=4. The H–H
        // distance interpolates 6 → 2, so it equals 4 at λ = 0.5 (the path midpoint),
        // where the surface peaks.
        let reactant = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let product = [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let mut surface = PairPeak { target: 4.0 };
        let peak = scan_peak(
            &reactant,
            &product,
            &GuessOptions::default(),
            11,
            &mut surface,
        )
        .expect("scan");

        let d = distance(peak.geometry[0], peak.geometry[1]);
        assert!((d - 4.0).abs() < 0.05, "peak distance {d} not at 4.0");

        // Tangent: unit, axial, atoms antiparallel along x.
        let norm: f64 = peak
            .tangent
            .iter()
            .flatten()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "tangent not normalized");
        assert!(
            peak.tangent[0][0] * peak.tangent[1][0] < 0.0,
            "atoms not antiparallel: {:?}",
            peak.tangent
        );
        assert!(
            peak.tangent[0][1].abs() < 1e-6 && peak.tangent[0][2].abs() < 1e-6,
            "off-axis tangent leak: {:?}",
            peak.tangent
        );
    }

    /// A surface that returns a finite energy everywhere except `Ok(NaN)` at one image.
    struct NanAt {
        bad_distance_below: f64,
    }
    impl Surface for NanAt {
        fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
            let d = distance(x[0], x[1]);
            if d < self.bad_distance_below {
                Ok(f64::NAN)
            } else {
                Ok(-d)
            }
        }
        fn analytic_gradient(
            &mut self,
            _x: &[[f64; 3]],
        ) -> Option<Result<Vec<[f64; 3]>, OptError>> {
            None
        }
    }

    #[test]
    fn non_finite_energy_is_surfaced_as_an_error() {
        // A positive NaN would win the total_cmp argmax and corrupt the guess; the scan
        // must reject it rather than return a NaN geometry.
        let reactant = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let product = [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]; // H–H closes 6 → 2 along the path
        let mut surface = NanAt {
            bad_distance_below: 3.0,
        };
        let err = scan_peak(
            &reactant,
            &product,
            &GuessOptions::default(),
            11,
            &mut surface,
        )
        .expect_err("a NaN energy must surface as an error");
        assert!(matches!(err, OptError::Evaluation(_)), "got {err:?}");
    }

    #[test]
    fn scan_handles_a_peak_at_the_path_boundary() {
        // Monotonic energy (peaks as the pair gets closest, at the product end): the
        // discrete maximum is the last interior point, with no right neighbour, so the
        // fit falls back to it rather than extrapolating off the grid.
        let reactant = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let product = [[2.5, 0.0, 0.0], [3.5, 0.0, 0.0]];
        let mut surface = PairPeak { target: 0.0 }; // closer ⇒ higher energy
        let peak = scan_peak(
            &reactant,
            &product,
            &GuessOptions::default(),
            9,
            &mut surface,
        )
        .expect("scan");
        // The guess sits near the product end (smallest separation), not off the path.
        let d = distance(peak.geometry[0], peak.geometry[1]);
        assert!(
            d < 1.6,
            "boundary peak not near the product separation: d={d}"
        );
    }
}
