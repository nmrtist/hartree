//! Mass-weighting, the translation/rotation-projected Hessian spectrum, the
//! finite-difference Hessian, and saddle verification, shared by
//! [`verify_saddle`](super::verify_saddle) and the [`super::prfo`] driver.

use serde::{Deserialize, Serialize};

use crate::core::Molecule;
use crate::core::units::FREQ_CONV_CM1;
use crate::linalg::{mat_from_row_major, mat_to_row_major, matmul, symmetric_eigh_checked};
use crate::opt::{OptError, Surface};

/// Negative-mode spectrum of the mass-weighted, translation/rotation-projected
/// Cartesian Hessian at a geometry — the output of the shared
/// [`verify_saddle`](super::verify_saddle) step. Carries the evidence that
/// classifies a point as a first-order saddle. `#[non_exhaustive]` so e.g. the full
/// eigenvalue spectrum or normal modes can be added later.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SaddleVerification {
    /// Eigenvalues counted as negative under
    /// [`TsOptions::negative_mode_tol`](super::TsOptions::negative_mode_tol)
    /// (atomic units, ascending). Exactly one entry for a first-order saddle.
    pub negative_eigenvalues: Vec<f64>,
    /// The reaction-mode eigenvector: the normalized displacement of the negative
    /// mode (Cartesian, length = natoms, input atom order), i.e. the direction the
    /// IRC is traced along. `Some` exactly for a first-order saddle (one negative
    /// mode) — so `reaction_mode.is_some()` agrees with
    /// [`is_first_order_saddle`](Self::is_first_order_saddle) — and `None` for a
    /// minimum or a higher-order saddle, whose reaction coordinate is ambiguous.
    /// This is what an agent inspects to identify *which* reaction the saddle
    /// describes, and what an IRC or TS thermochemistry step seeds from.
    pub reaction_mode: Option<Vec<[f64; 3]>>,
    /// The imaginary frequency in cm⁻¹ (reported negative, the
    /// `-√(-λ)·FREQ_CONV_CM1` convention of [`crate::props::frequencies`]) of the
    /// negative mode, for chemistry-meaningful reporting and RRHO thermochemistry.
    /// `Some` under the same first-order-saddle condition as
    /// [`reaction_mode`](Self::reaction_mode); `None` otherwise.
    pub imaginary_frequency_cm1: Option<f64>,
    /// The **full** mass-weighted, translation/rotation-projected eigenvalue spectrum
    /// (atomic units, ascending) — every mode, not just the negative ones. It comes
    /// free from the eigendecomposition the verification already runs, so a converged
    /// TS search carries its complete harmonic spectrum (hence its vibrational
    /// frequencies, via [`frequencies_cm1`](Self::frequencies_cm1)) without a second
    /// Hessian. The leading ≈5–6 near-zero entries are the projected-out
    /// translation/rotation residue. `#[serde(default)]`: an empty vector for records
    /// serialized before this field existed.
    #[serde(default)]
    pub eigenvalues: Vec<f64>,
}

impl SaddleVerification {
    /// `true` iff there is exactly one negative mode (a first-order saddle).
    pub fn is_first_order_saddle(&self) -> bool {
        self.negative_eigenvalues.len() == 1
    }

    /// The harmonic frequencies in cm⁻¹ of the physical modes, derived from
    /// [`eigenvalues`](Self::eigenvalues) at no extra cost — the same
    /// `±√|λ|·FREQ_CONV_CM1` convention as [`crate::props::frequencies`] (negative
    /// for an imaginary mode). The near-zero translation/rotation modes are dropped,
    /// so this is the `3N−6` (or `3N−5`) vibrational spectrum a frequency job would
    /// report at the same geometry. Empty for a record deserialized before the full
    /// spectrum was stored (see [`eigenvalues`](Self::eigenvalues)).
    pub fn frequencies_cm1(&self) -> Vec<f64> {
        self.eigenvalues
            .iter()
            .filter(|&&l| l.abs() > NULL_EPS)
            .map(|&l| l.signum() * l.abs().sqrt() * FREQ_CONV_CM1)
            .collect()
    }
}

pub(super) struct MwSpectrum {
    /// Ascending eigenvalues (atomic units).
    pub(super) eigenvalues: Vec<f64>,
    /// Row-major `ndof x ndof`; column `k` is mode `k`.
    pub(super) eigenvectors: Vec<f64>,
}

/// Eigenvalue magnitude below which a mode is the projected-out translation /
/// rotation residue (driven to rounding by the exact projection) rather than a
/// physical mode.
const NULL_EPS: f64 = 1e-6;

pub(super) fn flatten(g: &[[f64; 3]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(g.len() * 3);
    for v in g {
        out.extend_from_slice(v);
    }
    out
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

pub(super) fn force_norms(gx: &[[f64; 3]]) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut sum_sq = 0.0;
    let mut count = 0;
    for v in gx {
        for &c in v {
            max = max.max(c.abs());
            sum_sq += c * c;
            count += 1;
        }
    }
    let rms = if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    };
    (max, rms)
}

pub(super) fn disp_norms(x: &[[f64; 3]], x_prev: &[[f64; 3]]) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut sum_sq = 0.0;
    let mut count = 0;
    for (a, b) in x.iter().zip(x_prev) {
        for k in 0..3 {
            let d = a[k] - b[k];
            max = max.max(d.abs());
            sum_sq += d * d;
            count += 1;
        }
    }
    let rms = if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    };
    (max, rms)
}

/// Remove the rigid-body (translation/rotation) component of a Cartesian gradient
/// in the mass-weighted frame the Hessian spectrum uses, returning the residual in
/// Cartesian units. Where the gradient carries no rigid-body component this is the
/// identity, so the force thresholds keep their meaning; it strips only a net
/// force/torque (e.g. finite-difference residue, or drift accumulated over the
/// climb) that would otherwise inflate the convergence metric at a true saddle.
pub(super) fn project_trans_rot(g: &[[f64; 3]], masses: &[f64], x: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let basis = gram_schmidt(&trans_rot_vectors(x, masses));
    let mut v = mass_weight_grad(g, masses);
    for b in &basis {
        let p: f64 = v.iter().zip(b).map(|(vi, bi)| vi * bi).sum();
        for (vi, &bi) in v.iter_mut().zip(b) {
            *vi -= p * bi;
        }
    }
    (0..masses.len())
        .map(|a| {
            let s = masses[a].sqrt();
            [v[3 * a] * s, v[3 * a + 1] * s, v[3 * a + 2] * s]
        })
        .collect()
}

/// `force_norms` of the gradient with rigid-body contamination projected out — the
/// quantity the saddle search applies its force thresholds to (the step is taken in
/// the same projected frame, so the convergence test and the step now agree).
pub(super) fn projected_force_norms(g: &[[f64; 3]], masses: &[f64], x: &[[f64; 3]]) -> (f64, f64) {
    force_norms(&project_trans_rot(g, masses, x))
}

pub(super) fn masses_of(molecule: &Molecule) -> Vec<f64> {
    molecule.atoms.iter().map(|a| a.element.mass()).collect()
}

pub(super) fn positions_of(molecule: &Molecule) -> Vec<[f64; 3]> {
    molecule.atoms.iter().map(|a| a.position).collect()
}

pub(super) fn with_positions(molecule: &Molecule, x: &[[f64; 3]]) -> Molecule {
    let mut atoms = molecule.atoms.clone();
    for (atom, p) in atoms.iter_mut().zip(x) {
        atom.position = *p;
    }
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}

pub(super) fn gradient<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    fd_step: f64,
) -> Result<Vec<[f64; 3]>, OptError> {
    match surface.analytic_gradient(x) {
        Some(result) => result,
        None => crate::opt::fd::central_difference(surface, x, fd_step),
    }
}

/// Cartesian Hessian by central finite difference of the gradient. Uses the
/// surface's own (e.g. parallel) Hessian when it offers one, else `2*ndof`
/// serial gradient evaluations.
pub(super) fn fd_hessian<S: Surface>(
    surface: &mut S,
    x: &[[f64; 3]],
    fd_step: f64,
) -> Result<Vec<f64>, OptError> {
    if let Some(result) = surface.fd_hessian(x, fd_step) {
        return result;
    }

    let natom = x.len();
    let ndof = 3 * natom;
    let mut h = vec![0.0f64; ndof * ndof];
    for dof in 0..ndof {
        let (atom, axis) = (dof / 3, dof % 3);
        let mut xp = x.to_vec();
        xp[atom][axis] += fd_step;
        let mut xm = x.to_vec();
        xm[atom][axis] -= fd_step;
        let gp = flatten(&gradient(surface, &xp, fd_step)?);
        let gm = flatten(&gradient(surface, &xm, fd_step)?);
        for j in 0..ndof {
            h[dof * ndof + j] = (gp[j] - gm[j]) / (2.0 * fd_step);
        }
    }
    for i in 0..ndof {
        for j in (i + 1)..ndof {
            let avg = 0.5 * (h[i * ndof + j] + h[j * ndof + i]);
            h[i * ndof + j] = avg;
            h[j * ndof + i] = avg;
        }
    }
    Ok(h)
}

pub(super) fn mass_weight_grad(g: &[[f64; 3]], masses: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; 3 * g.len()];
    for (a, gi) in g.iter().enumerate() {
        let s = masses[a].sqrt();
        for c in 0..3 {
            out[3 * a + c] = gi[c] / s;
        }
    }
    out
}

pub(super) fn unmass_weight_step(dxi: &[f64], masses: &[f64]) -> Vec<[f64; 3]> {
    (0..masses.len())
        .map(|a| {
            let s = masses[a].sqrt();
            [dxi[3 * a] / s, dxi[3 * a + 1] / s, dxi[3 * a + 2] / s]
        })
        .collect()
}

pub(super) fn add_step(x: &[[f64; 3]], dx: &[[f64; 3]]) -> Vec<[f64; 3]> {
    x.iter()
        .zip(dx)
        .map(|(a, d)| [a[0] + d[0], a[1] + d[1], a[2] + d[2]])
        .collect()
}

pub(super) fn predicted_change_cart(g: &[[f64; 3]], hess: &[f64], dx: &[[f64; 3]]) -> f64 {
    let gf = flatten(g);
    let df = flatten(dx);
    let n = df.len();
    let mut p = dot(&gf, &df);
    for i in 0..n {
        let mut hd = 0.0;
        for j in 0..n {
            hd += hess[i * n + j] * df[j];
        }
        p += 0.5 * df[i] * hd;
    }
    p
}

pub(super) fn matvec(h: &[f64], s: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| h[i * n + j] * s[j]).sum())
        .collect()
}

/// Diagonalize the mass-weighted, translation/rotation-projected Cartesian
/// Hessian. Routed through the [non-panicking eigensolver](symmetric_eigh_checked)
/// because the input is a finite-difference (or quasi-Newton-maintained) Hessian
/// that can drift non-finite or ill-conditioned; the `Err` lets the caller rebuild
/// and retry rather than abort. The error string is wrapped into
/// [`TsError::Numerical`](super::TsError::Numerical) at the driver boundary.
pub(super) fn mw_projected_hessian(
    x: &[[f64; 3]],
    masses: &[f64],
    hess_cart: &[f64],
) -> Result<MwSpectrum, String> {
    let natom = x.len();
    let ndof = 3 * natom;

    let mut mw = hess_cart.to_vec();
    for i in 0..natom {
        for ki in 0..3 {
            let row = 3 * i + ki;
            for j in 0..natom {
                for kj in 0..3 {
                    let col = 3 * j + kj;
                    mw[row * ndof + col] /= (masses[i] * masses[j]).sqrt();
                }
            }
        }
    }

    let orth = gram_schmidt(&trans_rot_vectors(x, masses));
    let mut proj = vec![0.0f64; ndof * ndof];
    for i in 0..ndof {
        proj[i * ndof + i] = 1.0;
    }
    for v in &orth {
        for i in 0..ndof {
            for j in 0..ndof {
                proj[i * ndof + j] -= v[i] * v[j];
            }
        }
    }

    let p = mat_from_row_major(ndof, &proj);
    let f = mat_from_row_major(ndof, &mw);
    let fp = matmul(&matmul(&p, &f), &p);
    let eigh = symmetric_eigh_checked(&fp)?;
    Ok(MwSpectrum {
        eigenvalues: eigh.values,
        eigenvectors: mat_to_row_major(&eigh.vectors),
    })
}

/// The mass-weighted translation (3) and rotation (3, or 2 if linear) vectors,
/// matching `crate::props::frequencies`.
pub(super) fn trans_rot_vectors(x: &[[f64; 3]], masses: &[f64]) -> Vec<Vec<f64>> {
    let natom = x.len();
    let ndof = 3 * natom;
    let total: f64 = masses.iter().sum();
    let mut com = [0.0f64; 3];
    for (i, xi) in x.iter().enumerate() {
        for k in 0..3 {
            com[k] += masses[i] * xi[k];
        }
    }
    for c in &mut com {
        *c /= total;
    }

    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(6);
    for k in 0..3 {
        let mut v = vec![0.0f64; ndof];
        for i in 0..natom {
            v[3 * i + k] = masses[i].sqrt();
        }
        vecs.push(v);
    }
    let r: Vec<[f64; 3]> = x
        .iter()
        .map(|xi| [xi[0] - com[0], xi[1] - com[1], xi[2] - com[2]])
        .collect();
    type RotDisp = fn(&[f64; 3]) -> [f64; 3];
    let rot: [RotDisp; 3] = [
        |r| [0.0, -r[2], r[1]],
        |r| [r[2], 0.0, -r[0]],
        |r| [-r[1], r[0], 0.0],
    ];
    for disp in &rot {
        let mut v = vec![0.0f64; ndof];
        for i in 0..natom {
            let d = disp(&r[i]);
            for k in 0..3 {
                v[3 * i + k] = masses[i].sqrt() * d[k];
            }
        }
        vecs.push(v);
    }
    vecs
}

/// Orthonormalize, dropping vectors that collapse to ~0 (a linear molecule's
/// redundant rotation).
pub(super) fn gram_schmidt(vecs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut orth: Vec<Vec<f64>> = Vec::new();
    for v in vecs {
        let mut u = v.clone();
        for prev in &orth {
            let proj: f64 = u.iter().zip(prev).map(|(a, b)| a * b).sum();
            for (ui, &pi) in u.iter_mut().zip(prev.iter()) {
                *ui -= proj * pi;
            }
        }
        let n: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-10 {
            for x in &mut u {
                *x /= n;
            }
            orth.push(u);
        }
    }
    orth
}

pub(super) fn non_null_modes(spec: &MwSpectrum) -> Vec<usize> {
    (0..spec.eigenvalues.len())
        .filter(|&k| spec.eigenvalues[k].abs() > NULL_EPS)
        .collect()
}

pub(super) fn column(eigenvectors: &[f64], ndof: usize, k: usize) -> Vec<f64> {
    (0..ndof).map(|i| eigenvectors[i * ndof + k]).collect()
}

pub(super) fn overlap(spec: &MwSpectrum, ndof: usize, k: usize, reference: &[f64]) -> f64 {
    (0..ndof)
        .map(|i| spec.eigenvectors[i * ndof + k] * reference[i])
        .sum::<f64>()
        .abs()
}

/// Whether an (approximate, maintained) Hessian's spectrum has a physical mode close
/// enough to the negative-mode threshold that the approximation could misclassify it —
/// the cue for the [`Auto`](super::VerifyHessian::Auto) verification to recompute a
/// fresh finite-difference Hessian instead. A mode is suspect when its eigenvalue lies
/// in `(-2·tol, +tol)`: straddling the `−tol` cut, or small-positive enough that
/// Hessian error could push it past the cut. The trans/rot null modes (`|λ| ≤ NULL_EPS`)
/// are exempt — they are never counted. With no mode in that band the negative count is
/// robust to the maintained Hessian's error, so it can be trusted without a fresh build.
pub(super) fn spectrum_ambiguous(eigenvalues: &[f64], tol: f64) -> bool {
    eigenvalues
        .iter()
        .any(|&l| l.abs() > NULL_EPS && l > -2.0 * tol && l < tol)
}

pub(super) fn saddle_from_hessian(
    molecule: &Molecule,
    hessian: &[f64],
    tol: f64,
) -> Result<SaddleVerification, String> {
    let natom = molecule.len();
    let ndof = 3 * natom;
    let masses = masses_of(molecule);
    let spec = mw_projected_hessian(&positions_of(molecule), &masses, hessian)?;

    let negative_eigenvalues: Vec<f64> = spec
        .eigenvalues
        .iter()
        .copied()
        .filter(|&l| l < -tol)
        .collect();

    // The reaction mode and imaginary frequency are only well defined for a
    // first-order saddle (exactly one negative mode); a higher-order saddle has no
    // single reaction coordinate. Gate them on the same count as
    // `is_first_order_saddle`, so `reaction_mode.is_some()` agrees with it. The one
    // negative mode is necessarily the lowest (eigenvalues are ascending).
    let (reaction_mode, imaginary_frequency_cm1) = if negative_eigenvalues.len() == 1 {
        let lambda = spec.eigenvalues[0];
        // Un-mass-weight the lowest mode to a Cartesian displacement, then normalize.
        let q = column(&spec.eigenvectors, ndof, 0);
        let mut mode: Vec<[f64; 3]> = (0..natom)
            .map(|a| {
                let s = masses[a].sqrt();
                [q[3 * a] / s, q[3 * a + 1] / s, q[3 * a + 2] / s]
            })
            .collect();
        let nrm = norm(&flatten(&mode));
        if nrm > 0.0 {
            for m in &mut mode {
                for c in m.iter_mut() {
                    *c /= nrm;
                }
            }
        }
        (Some(mode), Some(-(-lambda).sqrt() * FREQ_CONV_CM1))
    } else {
        (None, None)
    };

    Ok(SaddleVerification {
        negative_eigenvalues,
        reaction_mode,
        imaginary_frequency_cm1,
        eigenvalues: spec.eigenvalues,
    })
}
