//! The redundant-internal-coordinate [`Frame`] for a partitioned-RFO climb.
//!
//! Where [`CartesianFrame`](super::frame::CartesianFrame) steps in mass-weighted
//! Cartesian coordinates, this frame steps in the molecule's redundant internal
//! coordinates (bonds and valence angles, bridging disconnected fragments so a
//! forming/breaking bond is represented). Two properties make this a clean swap
//! that needs no change to the climb's Hessian bookkeeping:
//!
//! * The Wilson B-matrix is invariant under rigid-body translation/rotation, so
//!   `B Hₓ Bᵀ` annihilates the six (or five) rigid modes automatically — the
//!   internal frame needs no explicit translation/rotation projection.
//! * The transform `H_q = G⁻ B Hₓ Bᵀ G⁻` sends the redundant null space to
//!   exactly-zero eigenvalues, which the climb's existing non-null filter already
//!   drops, and the internal gradient has no component there — so a redundant
//!   coordinate set contributes no spurious step.
//!
//! The maintained Hessian stays Cartesian and is transformed afresh each step, so
//! the climb's finite-difference build, Bofill update, and recovery are unchanged;
//! only the step *direction and size* are taken in the better-conditioned internal
//! metric. The step is mapped back to Cartesian by the iterative B-matrix
//! back-transformation, and the post-convergence saddle verification still runs in
//! the exact mass-weighted Cartesian frame.

use super::frame::Frame;
use super::numerics::{MwSpectrum, flatten};
use crate::linalg::{mat_from_row_major, mat_to_row_major, symmetric_eigh_checked};
use crate::opt::internals::{self, Internal};

pub(super) struct InternalFrame {
    defs: Vec<Internal>,
    ndof: usize,
    /// The reaction-coordinate seed as a flat, normalized Cartesian direction
    /// (length `ndof`); mapped into the internal tangent at each geometry by
    /// [`seed`](Frame::seed). `None` when the search carries no seed.
    seed_cart: Option<Vec<f64>>,
}

impl InternalFrame {
    /// Build an internal-coordinate frame from a primitive set `defs` generated for
    /// the molecule, or `None` when the set does not span the molecule's internal
    /// space at `x0` (rank `G < n_internal_dof`) — in which case the driver falls
    /// back to the Cartesian frame rather than run a search that cannot move along a
    /// missing coordinate. `n_internal_dof` is `3N` minus the rigid-body mode count
    /// (`6`, or `5` for a linear molecule).
    pub(super) fn new(
        x0: &[[f64; 3]],
        defs: Vec<Internal>,
        seed_cart: Option<Vec<f64>>,
        n_internal_dof: usize,
    ) -> Option<Self> {
        if defs.is_empty() || internals::internal_rank(&defs, x0) < n_internal_dof {
            return None;
        }
        Some(Self {
            defs,
            ndof: 3 * x0.len(),
            seed_cart,
        })
    }
}

impl Frame for InternalFrame {
    fn dim(&self) -> usize {
        self.defs.len()
    }

    fn gradient(&self, gx: &[[f64; 3]], x: &[[f64; 3]]) -> Vec<f64> {
        let b = internals::wilson_b(&self.defs, x);
        internals::internal_gradient(&b, &flatten(gx), self.defs.len(), self.ndof)
    }

    fn spectrum(&self, x: &[[f64; 3]], hess_cart: &[f64]) -> Result<MwSpectrum, String> {
        let nq = self.defs.len();
        let hq = internals::internal_hessian(&self.defs, x, hess_cart);
        let eig = symmetric_eigh_checked(&mat_from_row_major(nq, &hq))?;
        Ok(MwSpectrum {
            eigenvalues: eig.values,
            eigenvectors: mat_to_row_major(&eig.vectors),
        })
    }

    fn to_cartesian(&self, dw: &[f64], x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let x_new = internals::back_transform(&self.defs, x, dw);
        x.iter()
            .zip(&x_new)
            .map(|(a, b)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]])
            .collect()
    }

    fn seed(&self, x: &[[f64; 3]]) -> Option<Vec<f64>> {
        let seed_cart = self.seed_cart.as_ref()?;
        // The internal tangent of the Cartesian seed direction: δq = B δx, normalized.
        // It lands in the range of B (where the non-null Hessian eigenvectors live), so
        // its overlap with them anchors the first climbed mode.
        let b = internals::wilson_b(&self.defs, x);
        let nq = self.defs.len();
        let mut sq = vec![0.0; nq];
        for (i, slot) in sq.iter_mut().enumerate() {
            let row = i * self.ndof;
            *slot = (0..self.ndof).map(|j| b[row + j] * seed_cart[j]).sum();
        }
        let n: f64 = sq.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Reject a non-finite norm (a NaN leaking from a degenerate geometry's B-matrix)
        // as well as a vanishing one, so the seed drops cleanly to `None` rather than
        // returning a NaN-normalized direction that would corrupt mode tracking.
        if !n.is_finite() || n < 1e-12 {
            return None;
        }
        for v in &mut sq {
            *v /= n;
        }
        Some(sq)
    }
}
