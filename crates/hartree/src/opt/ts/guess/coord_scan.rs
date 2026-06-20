//! Distinguished-coordinate scan: a relaxed-surface scan along one chosen internal
//! coordinate, returning the energy-peaked geometry as a transition-state guess.
//!
//! Where [`scan_peak`](super::scan::scan_peak) walks an IDPP interpolation between two
//! endpoints, this drives a single *internal* coordinate (a bond length, valence angle,
//! or torsion) across a value range. At each grid value the rest of the molecule is
//! relaxed with that coordinate held fixed
//! ([`optimize_constrained`](crate::opt::constrained::optimize_constrained)), giving the
//! minimum-energy profile along the distinguished coordinate. The energy maximum of that
//! profile — refined to a sub-grid value by the same three-point parabola fit the IDPP
//! scan uses — is the transition-state estimate, with the reaction-coordinate tangent
//! taken as a finite difference of the two relaxed geometries bracketing the peak.
//!
//! This is the classic distinguished-reaction-coordinate / coordinate-driving method
//! (Rothman–Lohr): cheap and robust when the reaction is dominated by one identifiable
//! coordinate, at the cost of one constrained minimization per grid point.

use super::scan::{normalize, parabola_vertex};
use crate::core::Molecule;
use crate::opt::constrained::{Constraint, optimize_constrained};
use crate::opt::internals::{self, Internal};
use crate::opt::{OptError, OptOptions, Surface};

/// The result of [`coord_scan_peak`]: the relaxed guess geometry at the fitted energy
/// maximum along the distinguished coordinate, and the (unit) reaction-coordinate tangent
/// there, one Cartesian vector per atom.
#[derive(Debug)]
pub struct CoordPeak {
    pub geometry: Vec<[f64; 3]>,
    pub tangent: Vec<[f64; 3]>,
    /// The distinguished-coordinate value at the fitted energy maximum.
    pub coordinate_value: f64,
}

/// Controls for [`coord_scan_peak`]; construct via [`CoordScanOptions::default`] and set
/// the coordinate and range. `#[non_exhaustive]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CoordScanOptions {
    /// The internal coordinate to drive. Matched against the molecule's redundant set by
    /// atom indices and appended if absent, so any well-defined coordinate is drivable.
    pub coordinate: Internal,
    /// Start of the driven coordinate's value range (Bohr / radians).
    pub start: f64,
    /// End of the driven coordinate's value range (Bohr / radians).
    pub end: f64,
    /// Number of grid points across `[start, end]` inclusive (must be ≥ 3 so the peak can
    /// be parabola-bracketed).
    pub n_points: usize,
    /// The relaxation controls for each constrained minimization at a grid value.
    pub opt: OptOptions,
}

impl CoordScanOptions {
    /// A scan of `coordinate` over `[start, end]` with `n_points` grid points and default
    /// relaxation controls.
    pub fn new(coordinate: Internal, start: f64, end: f64, n_points: usize) -> Self {
        Self {
            coordinate,
            start,
            end,
            n_points,
            opt: OptOptions::default(),
        }
    }
}

/// A failure of the distinguished-coordinate scan. `#[non_exhaustive]`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CoordScanError {
    /// A constrained relaxation or surface evaluation failed (see [`OptError`]).
    #[error(transparent)]
    Surface(#[from] OptError),
    /// Fewer than three grid points were requested, so the peak cannot be bracketed.
    #[error("distinguished-coordinate scan needs at least 3 grid points, got {0}")]
    TooFewPoints(usize),
}

/// Scan the distinguished coordinate across its range, relaxing the rest of the molecule
/// at each grid value, and return the relaxed geometry at the parabola-fitted energy
/// maximum together with the reaction-coordinate tangent there.
///
/// Each grid value runs an [`optimize_constrained`] holding the driven coordinate at that
/// value (started from the previous grid point's relaxed geometry so neighbouring points
/// are continuous, a "chained" scan). The discrete energy maximum is refined to a sub-grid
/// coordinate value `c*` by a three-point parabola fit (falling back to the discrete peak
/// at a boundary or a non-concave triple). The tangent is a central difference of the two
/// relaxed geometries adjacent to the peak.
///
/// # Errors
/// [`CoordScanError::TooFewPoints`] if `n_points < 3`; [`CoordScanError::Surface`] if any
/// constrained relaxation fails.
pub fn coord_scan_peak<S: Surface>(
    molecule: &Molecule,
    options: &CoordScanOptions,
    surface: &mut S,
) -> Result<CoordPeak, CoordScanError> {
    let n = options.n_points;
    if n < 3 {
        return Err(CoordScanError::TooFewPoints(n));
    }

    let step = (options.end - options.start) / (n as f64 - 1.0);
    let values: Vec<f64> = (0..n).map(|k| options.start + k as f64 * step).collect();

    // Relax each grid value, chaining from the previous relaxed geometry so the profile
    // is continuous and each minimization starts close to its answer.
    let mut current = molecule.clone();
    let mut energies = Vec::with_capacity(n);
    let mut geometries: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n);
    for &value in &values {
        let constraints = [Constraint {
            coordinate: options.coordinate,
            target: value,
        }];
        let res = optimize_constrained(&current, surface, &constraints, &options.opt)?;
        if !res.energy.is_finite() {
            return Err(CoordScanError::Surface(OptError::Evaluation(format!(
                "non-finite relaxed energy {} at coordinate value {value}",
                res.energy
            ))));
        }
        current = with_positions(molecule, &res.positions);
        energies.push(res.energy);
        geometries.push(res.positions);
    }

    // Discrete peak, refined to a sub-grid value by a parabola through it and its two
    // neighbours (when it has both).
    let peak = (0..n)
        .max_by(|&a, &b| energies[a].total_cmp(&energies[b]))
        .ok_or_else(|| {
            CoordScanError::Surface(OptError::Evaluation(
                "coordinate scan has no grid points".to_string(),
            ))
        })?;
    let coordinate_value = if peak == 0 || peak == n - 1 {
        values[peak]
    } else {
        // `parabola_vertex` expects the neighbours in coordinate order (`y0` at the
        // smaller value, `y2` at the larger). The grid descends when `start > end`
        // (`step < 0`), which flips which index is the lower neighbour, so order the
        // pair by the sign of `step` rather than by index; `h = |step|` keeps the
        // vertex clamp's bracket well-ordered either way.
        let (y0, y2) = if step >= 0.0 {
            (energies[peak - 1], energies[peak + 1])
        } else {
            (energies[peak + 1], energies[peak - 1])
        };
        parabola_vertex(values[peak], step.abs(), y0, energies[peak], y2)
    };

    // Geometry at the peak: the relaxed geometry at the discrete maximum. (A further
    // constrained relaxation at the interpolated `coordinate_value` would refine it, but
    // the discrete relaxed geometry is already a minimum off the driven coordinate and is
    // handed to the saddle refiner, which finishes the climb.)
    let geometry = geometries[peak].clone();

    // Reaction-coordinate tangent: the central difference of the relaxed geometries on
    // either side of the peak (the direction the geometry moves as the coordinate is
    // driven through the barrier top). At a boundary peak, use the one-sided difference;
    // fall back to the displacement the driven coordinate's own Wilson-B row induces.
    let (lo, hi) = bracket(&geometries, peak, n);
    let mut tangent: Vec<[f64; 3]> = lo
        .iter()
        .zip(hi)
        .map(|(a, b)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]])
        .collect();
    if !normalize(&mut tangent) {
        tangent = coordinate_direction(&options.coordinate, &geometry);
    }

    Ok(CoordPeak {
        geometry,
        tangent,
        coordinate_value,
    })
}

/// The pair of relaxed geometries bracketing the discrete peak: `(peak-1, peak+1)` in
/// the interior, or the peak paired with its single neighbour at a boundary.
fn bracket(geometries: &[Vec<[f64; 3]>], peak: usize, n: usize) -> (&[[f64; 3]], &[[f64; 3]]) {
    let lo = if peak == 0 { peak } else { peak - 1 };
    let hi = if peak == n - 1 { peak } else { peak + 1 };
    (&geometries[lo], &geometries[hi])
}

/// The Cartesian direction the driven internal coordinate moves the geometry — its row of
/// the Wilson B-matrix, reshaped per atom and used as a tangent fallback when the
/// bracketing relaxed geometries coincide.
fn coordinate_direction(coordinate: &Internal, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let defs = [*coordinate];
    let b = internals::wilson_b(&defs, x); // single row, length 3·natom
    let mut dir = vec![[0.0f64; 3]; x.len()];
    for (a, slot) in dir.iter_mut().enumerate() {
        for (c, slot_c) in slot.iter_mut().enumerate() {
            *slot_c = b[3 * a + c];
        }
    }
    normalize(&mut dir);
    dir
}

/// Place `positions` onto a copy of `template`'s atoms (same elements/charge/multiplicity).
fn with_positions(template: &Molecule, positions: &[[f64; 3]]) -> Molecule {
    let mut mol = template.clone();
    for (atom, p) in mol.atoms.iter_mut().zip(positions) {
        atom.position = *p;
    }
    mol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// A symmetric three-atom H-transfer / SN2 model: atom 1 (the migrating atom) sits on
    /// the axis between fixed-ish atoms 0 and 2. The energy is an inverted double well in
    /// the 0–1 distance that *peaks* when atom 1 is at the symmetric midpoint between 0 and
    /// 2, plus harmonic restraints keeping 0 and 2 near a reference separation. Driving the
    /// 0–1 bond therefore peaks at the symmetric crossing — exactly where a real H-transfer
    /// saddle sits.
    struct SymmetricTransfer {
        /// Total 0–2 separation the barrier top is symmetric about; peak of the 0–1 drive
        /// is at `half` = `r02 / 2`.
        half: f64,
        k_axis: f64,
        k_keep: f64,
        r02: f64,
    }
    impl Surface for SymmetricTransfer {
        fn energy(&mut self, x: &[[f64; 3]]) -> Result<f64, OptError> {
            let r01 = dist(x[0], x[1]);
            let r02 = dist(x[0], x[2]);
            // Inverted parabola in r01 about the midpoint (a maximum at r01 = half), plus
            // a restraint pinning the 0–2 separation, plus a weak restraint keeping atom 1
            // on the 0–2 line via the 1–2 distance complementing r01.
            let r12 = dist(x[1], x[2]);
            Ok(-self.k_axis * (r01 - self.half).powi(2)
                + 0.5 * self.k_keep * (r02 - self.r02).powi(2)
                + 0.5 * self.k_keep * (r01 + r12 - self.r02).powi(2))
        }
        fn analytic_gradient(
            &mut self,
            _x: &[[f64; 3]],
        ) -> Option<Result<Vec<[f64; 3]>, OptError>> {
            None
        }
    }

    #[test]
    fn scan_finds_the_symmetric_midpoint_peak() {
        // Linear A–H–B with A at x=0, B at x=4 (r02 = 4); the migrating atom 1 starts near
        // A. Driving the 0–1 distance from 1.2 to 2.8 must peak at the symmetric midpoint
        // 2.0, where the inverted parabola maxes.
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [1.3, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [4.0, 0.0, 0.0]),
            ],
            0,
            1,
        );
        let mut surface = SymmetricTransfer {
            half: 2.0,
            k_axis: 0.4,
            k_keep: 0.6,
            r02: 4.0,
        };
        let opts = CoordScanOptions::new(Internal::Bond(0, 1), 1.2, 2.8, 17);
        let peak = coord_scan_peak(&mol, &opts, &mut surface).expect("scan");

        // The fitted peak coordinate value lands at the symmetric midpoint 2.0.
        assert!(
            (peak.coordinate_value - 2.0).abs() < 0.06,
            "peak coordinate {} not at the symmetric midpoint 2.0",
            peak.coordinate_value
        );
        // The relaxed geometry holds the driven 0–1 distance near the peak value.
        let d01 = dist(peak.geometry[0], peak.geometry[1]);
        assert!((d01 - 2.0).abs() < 0.2, "peak geometry 0-1 distance {d01}");
        // The tangent is a unit vector.
        let nrm: f64 = peak
            .tangent
            .iter()
            .flatten()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt();
        assert!((nrm - 1.0).abs() < 1e-9, "tangent not normalized: {nrm}");
    }

    #[test]
    fn scan_finds_the_peak_on_a_descending_range() {
        // Same surface and peak as above, but driving the 0–1 distance from 2.8 down to
        // 1.2 (a descending range, step < 0). The sub-grid fit must still place the peak
        // at the symmetric midpoint 2.0 — i.e. the parabola neighbours are ordered by
        // coordinate, not grid index.
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [1.3, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [4.0, 0.0, 0.0]),
            ],
            0,
            1,
        );
        let mut surface = SymmetricTransfer {
            half: 2.0,
            k_axis: 0.4,
            k_keep: 0.6,
            r02: 4.0,
        };
        let opts = CoordScanOptions::new(Internal::Bond(0, 1), 2.8, 1.2, 17);
        let peak = coord_scan_peak(&mol, &opts, &mut surface).expect("scan");
        assert!(
            (peak.coordinate_value - 2.0).abs() < 0.06,
            "descending-range peak coordinate {} not at the symmetric midpoint 2.0",
            peak.coordinate_value
        );
    }

    #[test]
    fn rejects_too_few_points() {
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [1.3, 0.0, 0.0]),
            ],
            0,
            1,
        );
        let mut surface = SymmetricTransfer {
            half: 2.0,
            k_axis: 0.4,
            k_keep: 0.6,
            r02: 4.0,
        };
        let opts = CoordScanOptions::new(Internal::Bond(0, 1), 1.0, 2.0, 2);
        let err = coord_scan_peak(&mol, &opts, &mut surface).expect_err("too few points");
        assert!(
            matches!(err, CoordScanError::TooFewPoints(2)),
            "got {err:?}"
        );
    }
}
