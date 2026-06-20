//! Pre-flight peak-memory estimation for a [`Job`].
//!
//! [`estimate_memory`] derives the dominant dense allocations a job will make —
//! the in-core ERI tensor, the density-fitted `B` tensor, the post-HF amplitude
//! blocks, the DFT grid, and — for a transition-state (`--ts`) search — the dense
//! Hessian and the concurrent SCF working sets of its parallel finite-difference
//! Hessian build — **without** running the SCF or allocating any of
//! them. It builds only the cheap pre-SCF objects (the AO/auxiliary basis and,
//! for DFT, the integration grid) that [`Job::run`] would build anyway, reads
//! their sizes, and applies the same per-backend scaling the run path uses.
//!
//! The numbers estimate the *dominant dense allocations* of the chosen backend
//! and method, summed (terms that are not strictly simultaneous are still added,
//! so the modeled terms err high). They are not an exact RSS, and some secondary
//! costs are deliberately not modeled: the implicit-solvation (C-PCM/SMD) cavity
//! matrices, per-block XC grid scratch, and allocator/fragmentation overhead. So
//! the result is a *budgeting signal, not a guaranteed ceiling* — apply a safety
//! margin before declaring a job "fits", especially for solvated or small-basis
//! jobs where the unmodeled terms are relatively larger.
//!
//! The backend the estimate assumes mirrors [`Job::run`]'s dispatch exactly, so
//! it stays correct only as long as the two are kept in sync; that is why the
//! estimator lives beside the driver and cites the allocation sites it models.

use serde::{Deserialize, Serialize};

use crate::basis::BasisSet;
use crate::cc::frozen_core_orbitals;
use crate::core::Molecule;
use crate::dft::{FunctionalSpec, GridXc};
use crate::job::alpha_beta_electrons;
use crate::{Job, Method};

/// The integral backend an estimate assumed, matching [`Job::run`]'s dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum EstimateBackend {
    /// Conventional in-core ERI tensor (also: double hybrids, optimization,
    /// transition-state search, and frequencies).
    Conventional,
    /// Integral-direct: ERIs recomputed on the fly, O(nao²) resident.
    Direct,
    /// Density fitting (RI-JK): the fitted three-index `B` tensor is resident.
    Ri,
}

impl std::fmt::Display for EstimateBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Conventional => "conventional",
            Self::Direct => "direct",
            Self::Ri => "ri",
        })
    }
}

/// One itemized contribution to a [`MemoryEstimate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryTerm {
    /// Stable identifier for the allocation, e.g. `"eri_in_core"`.
    pub label: String,
    /// Estimated size of this allocation, in bytes.
    pub bytes: u64,
}

/// A pre-flight estimate of a job's peak memory, returned by [`estimate_memory`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryEstimate {
    /// Estimated peak resident working set, in bytes: the saturating sum of
    /// [`Self::breakdown`]. It models the dominant dense allocations and errs
    /// high for those, but does not include every secondary cost (see the module
    /// docs), so treat it as a budgeting signal and apply a safety margin rather
    /// than an exact RSS.
    pub peak_bytes: u64,
    /// Which integral backend the estimate assumed.
    pub backend: EstimateBackend,
    /// The itemized allocations, largest first.
    pub breakdown: Vec<MemoryTerm>,
}

/// Estimate the peak memory [`Job::run`] would use, without running it.
///
/// Builds the AO basis (and, for DFT, the auxiliary basis and integration grid)
/// — all cheap relative to the SCF — then models the dominant dense
/// allocations of the backend the job would dispatch to. Returns an error for
/// the same up-front reasons `run` would reject the job (unknown basis, an
/// element the basis does not cover, a charge exceeding the nuclear charge, or
/// a missing RI auxiliary set).
pub fn estimate_memory(job: &Job) -> Result<MemoryEstimate, String> {
    let mol = &job.molecule;
    let opts = &job.options;

    let basis_set = BasisSet::load(&job.basis).map_err(|e| e.to_string())?;
    let ao = basis_set.build(mol).map_err(|e| e.to_string())?;
    let nao = ao.n_ao();
    let nshell = ao.shells().len();

    let ecp_core = basis_set.ecp_core_electrons(mol) as i64;
    let (n_alpha, _n_beta) = alpha_beta_electrons(mol, ecp_core)?;
    // Use the majority-spin occupied count n_alpha. This maximizes the occupied
    // count o (bounding the o-heavy blocks from above) at the cost of the
    // smallest v = nao − n_alpha. It is exact for the closed-shell references the
    // post-HF backends actually use — Ccsd/CcsdT force RHF, and conventional MP2
    // has n_alpha == n_beta — and an approximation only for the open-shell
    // UHF-MP2 path, whose spatial-orbital formula assumes a single occ/virt split.
    let n_occ = n_alpha;
    let n_frozen = if opts.all_electron {
        0
    } else {
        frozen_core_orbitals(mol)
    };

    let n = nao as u128;
    let o = (n_occ.saturating_sub(n_frozen)) as u128;
    let v = (nao.saturating_sub(n_occ)) as u128;
    let npair = n * (n + 1) / 2;

    let dft_spec: Option<&FunctionalSpec> = match &job.method {
        Method::Dft(spec) => Some(spec),
        _ => None,
    };

    // Backend dispatch — identical priority order to `Job::run_inner`: `--ri`
    // wins over `--direct`, and everything else (including double hybrids,
    // geometry optimization, transition-state search, and frequencies) runs on
    // the conventional in-core backend.
    let backend = if opts.ri {
        EstimateBackend::Ri
    } else if opts.direct {
        EstimateBackend::Direct
    } else {
        EstimateBackend::Conventional
    };

    let mut terms: Vec<MemoryTerm> = Vec::new();

    match backend {
        EstimateBackend::Conventional => {
            // Dense rank-4 ERI: nao⁴ doubles (integrals::build_eri_parallel).
            terms.extend(term("eri_in_core", n * n * n * n));
            // Range-separated (CAM) functionals build a second nao⁴ erf-attenuated
            // tensor for long-range exchange (ConventionalProvider::eri_lr_tensor).
            if dft_spec.is_some_and(|s| s.cam().is_some()) {
                terms.extend(term("eri_long_range", n * n * n * n));
            }
            terms.extend(term("scf_matrices", 6 * n * n));
        }
        EstimateBackend::Direct => {
            // Integral-direct keeps only O(nshell²) Schwarz bounds and a handful
            // of O(nao²) matrices; it never stores the ERI (integrals DirectProvider).
            terms.extend(term("schwarz_table", (nshell as u128) * (nshell as u128)));
            terms.extend(term("scf_matrices", 8 * n * n));
        }
        EstimateBackend::Ri => {
            let naux = aux_naux("def2-universal-jkfit", mol)?;
            // Peak build transient: the full nao²·naux three-index tensor before
            // it is packed to the triangular B (DfProvider::with_screening).
            terms.extend(term("df_3c_scratch", n * n * naux));
            // Persistent fitted tensor B: nao(nao+1)/2 · naux.
            terms.extend(term("df_b_tensor", npair * naux));
            terms.extend(term("scf_matrices", 6 * n * n));
        }
    }

    if let Some(spec) = dft_spec {
        // Persistent molecular grid: point coordinates + weights (~4 doubles per
        // point). Per-block AO scratch is bounded by BLOCK_SIZE and omitted as a
        // second-order, transient cost.
        let grid = GridXc::new(mol, &ao, spec, opts.grid_level).map_err(|e| e.to_string())?;
        terms.extend(term("dft_grid", (grid.n_points() as u128) * 4));
    }

    // Post-HF / PT2 working set, keyed exactly as `run_inner` dispatches it.
    let double_hybrid = dft_spec.is_some_and(|s| s.double_hybrid().is_some());
    match &job.method {
        Method::Mp2 => add_mp2_terms(&mut terms, opts.ri_mp2, n, o, v, &job.basis, mol)?,
        Method::Ccsd => add_ccsd_terms(&mut terms, n, o, v),
        Method::CcsdT => {
            add_ccsd_terms(&mut terms, n, o, v);
            // CCSD(T) triples blocks (cc::ccsd::triples): vvov + 2·vvoo + vooo.
            terms.extend(term(
                "ccsdt_triples_blocks",
                v * v * o * v + 2 * v * v * o * o + v * o * o * o,
            ));
        }
        // A double hybrid runs a conventional SCF then the same MP2/RI-MP2 PT2.
        Method::Dft(_) if double_hybrid => {
            add_mp2_terms(&mut terms, opts.ri_mp2, n, o, v, &job.basis, mol)?
        }
        _ => {}
    }

    // A transition-state search (`--ts`) carries dense terms on top of the single SCF
    // modeled above — and during its parallel finite-difference Hessian holds several
    // SCF working sets at once, so the Hessian phase, not the SCF, is usually the peak.
    if opts.transition_state {
        add_transition_state_terms(&mut terms, mol);
    }

    terms.sort_by_key(|t| std::cmp::Reverse(t.bytes));
    let peak_bytes = terms
        .iter()
        .map(|t| t.bytes)
        .fold(0u64, u64::saturating_add);

    Ok(MemoryEstimate {
        peak_bytes,
        backend,
        breakdown: terms,
    })
}

/// MP2 / RI-MP2 correlation working set (cc::mp2, cc::ri_mp2, cc::transform).
fn add_mp2_terms(
    terms: &mut Vec<MemoryTerm>,
    ri: bool,
    n: u128,
    o: u128,
    v: u128,
    basis: &str,
    mol: &Molecule,
) -> Result<(), String> {
    if ri {
        let naux_c = aux_naux(&format!("{}/c", basis.to_ascii_lowercase()), mol)?;
        // Full nao²·naux AO three-centre tensor plus the half- and fully
        // transformed o·nao·naux / o·v·naux MO tensors (cc::ri_mp2).
        terms.extend(term("rimp2_3c_scratch", n * n * naux_c));
        terms.extend(term("rimp2_mo_integrals", o * n * naux_c + o * v * naux_c));
    } else {
        // Conventional MP2 clones the full nao⁴ ERI (transform_block) and forms
        // the (ia|jb) MO block.
        terms.extend(term("mp2_transform_scratch", n * n * n * n));
        terms.extend(term("mp2_mo_integrals", o * v * o * v));
    }
    Ok(())
}

/// Spin-adapted CCSD MO-integral and amplitude working set (cc::ccsd::spin_adapted,
/// cc::ccsd::diis).
fn add_ccsd_terms(terms: &mut Vec<MemoryTerm>, n: u128, o: u128, v: u128) {
    // Stored MO integral blocks: vvvv dominates, plus ovvv, oovv/ovov, ovoo, oooo.
    terms.extend(term(
        "ccsd_mo_integrals",
        v * v * v * v + o * v * v * v + 2 * o * o * v * v + o * o * o * v + o * o * o * o,
    ));
    // The per-iteration W_vvvv intermediate is a SECOND full v⁴ tensor, resident
    // alongside the stored vvvv block during the residual build.
    terms.extend(term("ccsd_vvvv_intermediate", v * v * v * v));
    // A full nao⁴ ERI clone per transform_block call.
    terms.extend(term("ccsd_transform_scratch", n * n * n * n));
    // t1, t2, same-shape iteration intermediates, plus the DIIS history: up to
    // `diis_dim` amplitude AND error vectors, each (o·o·v·v + o·v) doubles.
    // `diis_dim` defaults to 8 (CcsdOptions::default in cc::ccsd).
    const DIIS_DIM: u128 = 8;
    terms.extend(term(
        "ccsd_amplitudes",
        3 * o * o * v * v + o * v + 2 * DIIS_DIM * (o * o * v * v + o * v),
    ));
}

/// Transition-state (`--ts`) memory terms, added on top of the single-SCF estimate. A
/// saddle search maintains a dense Cartesian Hessian and diagonalizes its
/// mass-weighted, translation/rotation-projected form each step, and builds the
/// Hessian by finite difference — `2·ndof` displaced SCF+gradient evaluations run
/// concurrently ([`crate::HfSurface`]'s parallel Hessian), each holding its own
/// backend working set. `terms` must already hold the per-evaluation SCF (and any
/// DFT-grid) terms; the concurrent-Hessian cost reuses their sum, captured *before*
/// the dense Hessian terms are appended so it reflects one gradient evaluation.
fn add_transition_state_terms(terms: &mut Vec<MemoryTerm>, mol: &Molecule) {
    let ndof = 3 * mol.atoms.len() as u128;

    // Per-evaluation SCF/DFT working set — everything modeled so far (a saddle search
    // runs HF/DFT only, so no post-HF terms are present).
    let scf_working_set = terms
        .iter()
        .map(|t| t.bytes)
        .fold(0u64, u64::saturating_add);

    // Dense Cartesian Hessian carried through the search (the maintained quasi-Newton
    // Hessian and a freshly finite-differenced one are the same size): 9·natom² doubles.
    terms.extend(term("ts_hessian", ndof * ndof));
    // Projected-Hessian eigendecomposition scratch (numerics::mw_projected_hessian +
    // linalg::symmetric_eigh): the mass-weighted copy, the trans/rot projector, their
    // product, and the eigenvector matrix — a few ndof² dense matrices.
    terms.extend(term("ts_eigensolver_scratch", 4 * ndof * ndof));

    // Parallel finite-difference Hessian: up to `2·ndof` displaced evaluations, each
    // rebuilding the backend working set; the number live at once is bounded by the
    // rayon thread pool, so model min(2·ndof, threads) concurrent copies — usually the
    // peak of a `--ts` run.
    let threads = rayon::current_num_threads().max(1) as u128;
    let concurrency = (2 * ndof).min(threads);
    let concurrent = (scf_working_set as u128)
        .saturating_mul(concurrency)
        .min(u64::MAX as u128) as u64;
    if concurrent > 0 {
        terms.push(MemoryTerm {
            label: "ts_fd_hessian_concurrency".to_string(),
            bytes: concurrent,
        });
    }
}

/// Build an auxiliary basis for `mol` and return its function count as `u128`.
fn aux_naux(name: &str, mol: &Molecule) -> Result<u128, String> {
    Ok(BasisSet::load_aux(name)
        .map_err(|e| e.to_string())?
        .build(mol)
        .map_err(|e| e.to_string())?
        .n_ao() as u128)
}

/// An f64 element count converted to bytes. The element-count products feeding
/// this are computed in `u128`, which cannot overflow for any basis a machine
/// could build (nao would need to exceed ~4e9 for nao⁴ to wrap u128); the final
/// ×8 saturates to `u64::MAX` rather than wrapping as a last line of defense.
fn doubles(elems: u128) -> u64 {
    elems
        .saturating_mul(std::mem::size_of::<f64>() as u128)
        .min(u64::MAX as u128) as u64
}

/// A breakdown term, or `None` when the element count is zero (e.g. no virtuals).
fn term(label: &str, elems: u128) -> Option<MemoryTerm> {
    (elems > 0).then(|| MemoryTerm {
        label: label.to_string(),
        bytes: doubles(elems),
    })
}

/// Format a byte count with a binary unit suffix, for human-facing messages.
pub(crate) fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
#[path = "estimate_tests.rs"]
mod tests;
