//! Saddle-point search on real self-consistent-field surfaces.
//!
//! Complements the analytic-surface unit tests in `opt::ts::tests` with a
//! molecular case driven by an actual `HfSurface`, so the driver is exercised
//! end to end against a real energy/gradient path rather than a closed-form
//! model. The pinned numbers below are baselines from this code at the stated
//! level of theory; an intentional change to the search that moves them should
//! re-pin with review, not loosen the tolerances.

use hartree::HfSurface;
use hartree::core::{Atom, Element, Molecule};
use hartree::opt::OptError;
use hartree::opt::Surface;
use hartree::opt::ts::{TsOptions, TsStatus, find_transition_state};
use hartree::scf::Reference;

fn mol(atoms: &[(u32, [f64; 3])], charge: i32, multiplicity: u32) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|&(z, p)| Atom::new(Element::from_z(z).unwrap(), p))
            .collect(),
        charge,
        multiplicity,
    )
}

/// A bridging guess for the H–C≡N / H–N≡C isomerization saddle, where the
/// hydrogen sits off the C–N axis roughly equidistant from both heavy atoms
/// (positions in Bohr).
fn hcn_hnc_guess() -> Molecule {
    mol(
        &[
            (6, [0.0, 0.0, 0.0]),   // C
            (7, [2.23, 0.0, 0.0]),  // N
            (1, [1.10, 1.90, 0.0]), // H, bridging
        ],
        0,
        1,
    )
}

/// A `Surface` decorator that tallies how often each evaluation path is taken,
/// so a test can assert on the number of energy / gradient / Hessian calls a
/// search makes — the measurement hook for work-reduction changes to the driver.
struct CountingSurface<S: Surface> {
    inner: S,
    energy_calls: usize,
    gradient_calls: usize,
    hessian_calls: usize,
}

impl<S: Surface> CountingSurface<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            energy_calls: 0,
            gradient_calls: 0,
            hessian_calls: 0,
        }
    }
}

impl<S: Surface> Surface for CountingSurface<S> {
    fn energy(&mut self, positions: &[[f64; 3]]) -> Result<f64, OptError> {
        self.energy_calls += 1;
        self.inner.energy(positions)
    }

    fn analytic_gradient(
        &mut self,
        positions: &[[f64; 3]],
    ) -> Option<Result<Vec<[f64; 3]>, OptError>> {
        self.gradient_calls += 1;
        self.inner.analytic_gradient(positions)
    }

    fn fd_hessian(
        &mut self,
        positions: &[[f64; 3]],
        fd_step: f64,
    ) -> Option<Result<Vec<f64>, OptError>> {
        self.hessian_calls += 1;
        self.inner.fd_hessian(positions, fd_step)
    }
}

/// Distance between two atoms (Bohr) — a rotation/translation-invariant handle
/// on the geometry, unlike the absolute Cartesian frame the optimizer leaves
/// arbitrary.
fn bond(positions: &[[f64; 3]], i: usize, j: usize) -> f64 {
    let a = positions[i];
    let b = positions[j];
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The same SCF settings the `Job`/`transition_state` saddle path applies, so a
/// search driven through a wrapped surface reproduces that path's result.
fn prepare(surface: &mut HfSurface) {
    surface.set_scf_max_iter(400);
    surface.set_scf_level_shift(0.3);
}

#[test]
fn hcn_hnc_saddle_rhf_sto3g() {
    let molecule = hcn_hnc_guess();
    let mut surface =
        CountingSurface::new(HfSurface::new(&molecule, "sto-3g", Reference::Rhf).unwrap());
    prepare(&mut surface.inner);

    let mut opts = TsOptions::default();
    opts.confirm_irc = true;

    let result =
        find_transition_state(&molecule, &mut surface, &opts, None).expect("surface evaluation");

    // Converged to a genuine first-order saddle.
    assert_eq!(
        result.status,
        TsStatus::Converged,
        "status: {:?}",
        result.status
    );
    let v = result.verification.as_ref().expect("verification present");
    assert!(
        v.is_first_order_saddle(),
        "expected exactly one negative mode, got {:?}",
        v.negative_eigenvalues
    );
    let imag = v.imaginary_frequency_cm1.expect("imaginary frequency");
    assert!(
        imag < 0.0,
        "reaction mode should be imaginary, got {imag} cm⁻¹"
    );

    // Pinned baselines (RHF/STO-3G). Energy and the three internal coordinates
    // identify the saddle independent of its arbitrary Cartesian orientation.
    assert!(
        (result.energy - (-91.5648510302)).abs() < 1e-4,
        "saddle energy {:.8} Ha drifted from baseline",
        result.energy
    );
    let r_cn = bond(&result.positions, 0, 1);
    let r_ch = bond(&result.positions, 0, 2);
    let r_nh = bond(&result.positions, 1, 2);
    assert!((r_cn - 2.3080).abs() < 2e-2, "r(C–N) = {r_cn:.4} Bohr");
    assert!((r_ch - 2.2714).abs() < 2e-2, "r(C–H) = {r_ch:.4} Bohr");
    assert!((r_nh - 2.7168).abs() < 2e-2, "r(N–H) = {r_nh:.4} Bohr");
    assert!(
        (imag - (-1248.5)).abs() < 30.0,
        "imaginary frequency {imag:.1} cm⁻¹ drifted from baseline"
    );

    // The mass-weighted IRC integrator traces the path off the saddle and reaches
    // the two distinct isomer minima (the defining property of a transition state).
    let irc = result.irc.as_ref().expect("irc endpoints");
    assert!(
        irc.forward_converged && irc.reverse_converged,
        "an IRC endpoint did not reach a minimum (fwd conv={} steps={}, rev conv={} steps={})",
        irc.forward_converged,
        irc.forward_steps,
        irc.reverse_converged,
        irc.reverse_steps,
    );
    // Both endpoints relaxed well below the saddle into their basins.
    assert!(
        irc.forward_energy < result.energy - 0.05,
        "forward endpoint {:.6} not well below saddle {:.6}",
        irc.forward_energy,
        result.energy
    );
    assert!(
        irc.reverse_energy < result.energy - 0.05,
        "reverse endpoint {:.6} not well below saddle {:.6}",
        irc.reverse_energy,
        result.energy
    );
    // The two endpoints are the two distinct isomers: in one the hydrogen sits on
    // carbon (HCN: short C–H, long N–H), in the other on nitrogen (HNC: the reverse).
    let fwd_h_on_c = bond(&irc.forward, 0, 2) < bond(&irc.forward, 1, 2);
    let rev_h_on_c = bond(&irc.reverse, 0, 2) < bond(&irc.reverse, 1, 2);
    assert!(
        fwd_h_on_c != rev_h_on_c,
        "endpoints are not distinct isomers (both H-on-{})",
        if fwd_h_on_c { "C" } else { "N" }
    );
    // HCN (H on carbon) lies below HNC, the established ordering of the two isomers.
    let (e_hcn, e_hnc) = if fwd_h_on_c {
        (irc.forward_energy, irc.reverse_energy)
    } else {
        (irc.reverse_energy, irc.forward_energy)
    };
    assert!(
        e_hcn < e_hnc,
        "HCN should lie below HNC, got HCN {e_hcn:.6} vs HNC {e_hnc:.6}"
    );

    // The instrumentation observed real work on every path the search uses.
    assert!(surface.energy_calls > 0, "no energy evaluations recorded");
    assert!(
        surface.gradient_calls > 0,
        "no gradient evaluations recorded"
    );
    assert!(
        result.iterations < opts.max_iter,
        "search hit the iteration cap ({} iters)",
        result.iterations
    );
}
