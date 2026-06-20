//! The working coordinate frame a partitioned-RFO climb steps in.
//!
//! The climb in [`super::climb`] keeps its quadratic model in Cartesian
//! coordinates throughout — the maintained Hessian, its Bofill update, the
//! finite-difference recompute, the force/step convergence test, and the recovery
//! re-seeding all stay Cartesian. A [`Frame`] abstracts only the three places the
//! *step direction* depends on the coordinate system: the gradient the step is
//! built from, the Hessian spectrum whose mode it follows, and the Cartesian
//! displacement a frame-space step produces.
//!
//! [`CartesianFrame`] reproduces the historical mass-weighted,
//! translation/rotation-projected behaviour exactly, so the default search path is
//! byte-for-byte unchanged. An internal-coordinate frame works in redundant
//! internal coordinates, where the Wilson B-matrix removes rigid-body motion
//! intrinsically and conditions a soft (e.g. symmetric-stretch) reaction coordinate
//! that a Cartesian step sizes poorly.

use super::numerics::{MwSpectrum, mass_weight_grad, mw_projected_hessian, unmass_weight_step};

/// The coordinate frame a climb takes its steps in. The climb owns one `&dyn Frame`
/// and routes every coordinate-system-dependent operation through it; everything
/// else (Hessian maintenance, convergence, recovery) is frame-independent Cartesian.
pub(super) trait Frame {
    /// The working dimension: the length of a working-space gradient/step and the
    /// order of the spectrum (`3·natoms` for Cartesian, the number of internal
    /// coordinates for the internal frame). Constant over a climb.
    fn dim(&self) -> usize;

    /// The working-space gradient from the Cartesian gradient `gx` at geometry `x`.
    /// Its length is [`dim`](Self::dim).
    fn gradient(&self, gx: &[[f64; 3]], x: &[[f64; 3]]) -> Vec<f64>;

    /// The working-frame Hessian spectrum from the maintained Cartesian Hessian
    /// `hess_cart` at geometry `x`. The eigenvectors are columns in working
    /// coordinates; the climb follows one of their modes. An `Err` (a non-finite or
    /// non-converging eigenproblem) is handled by the climb's one self-healing
    /// finite-difference rebuild, exactly as before.
    fn spectrum(&self, x: &[[f64; 3]], hess_cart: &[f64]) -> Result<MwSpectrum, String>;

    /// The Cartesian displacement produced by a working-space step `dw` taken from
    /// geometry `x` (`dw` has the working dimension; the result is `natoms`
    /// Cartesian vectors).
    fn to_cartesian(&self, dw: &[f64], x: &[[f64; 3]]) -> Vec<[f64; 3]>;

    /// The reaction-coordinate seed expressed in working coordinates at `x`, when one
    /// was supplied — used to anchor the first climbed mode by maximum overlap.
    /// `None` when no seed is available.
    fn seed(&self, x: &[[f64; 3]]) -> Option<Vec<f64>>;
}

/// The historical frame: mass-weighted Cartesian coordinates with the rigid-body
/// translation/rotation modes projected out — the
/// [`crate::props::frequencies`] frame the saddle criterion lives in. Every method
/// delegates to the same `numerics` routine the climb called inline before the
/// [`Frame`] split, so a Cartesian search is unchanged to the bit.
pub(super) struct CartesianFrame {
    masses: Vec<f64>,
    /// The mass-weighted, normalized reaction-coordinate seed (length `3·natoms`),
    /// constant over the climb. `None` when the search carries no seed.
    seed_mw: Option<Vec<f64>>,
}

impl CartesianFrame {
    pub(super) fn new(masses: Vec<f64>, seed_mw: Option<Vec<f64>>) -> Self {
        Self { masses, seed_mw }
    }
}

impl Frame for CartesianFrame {
    fn dim(&self) -> usize {
        3 * self.masses.len()
    }

    fn gradient(&self, gx: &[[f64; 3]], _x: &[[f64; 3]]) -> Vec<f64> {
        mass_weight_grad(gx, &self.masses)
    }

    fn spectrum(&self, x: &[[f64; 3]], hess_cart: &[f64]) -> Result<MwSpectrum, String> {
        mw_projected_hessian(x, &self.masses, hess_cart)
    }

    fn to_cartesian(&self, dw: &[f64], _x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        unmass_weight_step(dw, &self.masses)
    }

    fn seed(&self, _x: &[[f64; 3]]) -> Option<Vec<f64>> {
        self.seed_mw.clone()
    }
}
