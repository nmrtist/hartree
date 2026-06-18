//! Self-consistent-field methods: RHF/UHF/ROHF and the method-agnostic SCF driver.

mod diis;
mod scf_math;
mod smearing;
mod solvent;
pub mod x2c;
mod xc;

use crate::core::units::BOLTZMANN_HT;
use crate::integrals::IntegralProvider;
use crate::linalg::{mat_from_row_major, mat_to_row_major};
use thiserror::Error;

use diis::Diis;
use scf_math::{
    ao_from_orth, canonical_orthogonalizer, commutator, eigh, max_abs, mul, orth_frac_density,
    orth_occ_density, transpose, vtav, xtax,
};
use smearing::{entropy_sum, fermi_occupations};

pub use smearing::Smearing;
pub use solvent::{SolventContribution, SolventModel};
pub use xc::{RangeSeparation, XcContribution, XcContributor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    Rhf,
    Uhf,
    Rohf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guess {
    Core,
    Gwh,
}

#[derive(Debug, Error)]
pub enum ScfError {
    #[error("basis has no functions")]
    EmptyBasis,

    #[error("RHF requires n_alpha == n_beta, got {n_alpha} and {n_beta}")]
    NotClosedShell { n_alpha: usize, n_beta: usize },

    #[error("{n_alpha} occupied orbitals requested but only {n_orbitals} are independent")]
    Overfilled { n_alpha: usize, n_orbitals: usize },

    #[error("restricted open-shell Kohn–Sham (ROKS) is not supported; use RKS or UKS")]
    RohfKohnSham,

    #[error("fractional-occupation smearing is not supported for ROHF; use RHF or UHF")]
    RohfSmearing,

    #[error("smearing temperature must be positive, got {temperature_k} K")]
    NonPositiveTemperature { temperature_k: f64 },

    #[error(
        "range-separated hybrid requires erf-attenuated exchange, which this integral \
         backend does not provide; use the conventional in-core backend"
    )]
    RangeSeparatedUnsupported,
}

#[derive(Debug, Clone)]
pub struct ScfOptions {
    pub max_iter: usize,
    pub energy_tol: f64,
    pub error_tol: f64,
    pub lindep_thresh: f64,
    pub diis_dim: usize,
    pub guess: Guess,
    pub level_shift: f64,
    pub incremental_fock: bool,
    pub fock_rebuild_period: usize,
    pub smearing: Option<Smearing>,
    pub hcore_override: Option<Vec<f64>>,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iter: 128,
            energy_tol: 1e-10,
            error_tol: 1e-8,
            lindep_thresh: 1e-6,
            diis_dim: 8,
            guess: Guess::Gwh,
            level_shift: 0.0,
            incremental_fock: false,
            fock_rebuild_period: 10,
            smearing: None,
            hcore_override: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScfIteration {
    pub iteration: usize,
    pub energy: f64,
    pub error_norm: f64,
}

#[derive(Debug, Clone)]
pub struct ScfResult {
    pub reference: Reference,
    pub energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion: f64,
    pub converged: bool,
    pub iterations: usize,
    pub n_basis: usize,
    pub n_orbitals: usize,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub density: Vec<f64>,
    pub density_alpha: Vec<f64>,
    pub density_beta: Vec<f64>,
    pub orbital_energies_alpha: Vec<f64>,
    pub orbital_energies_beta: Vec<f64>,
    pub mo_coeff_alpha: Vec<f64>,
    pub mo_coeff_beta: Vec<f64>,
    pub spin_squared: f64,
    pub history: Vec<ScfIteration>,
    pub xc_energy: Option<f64>,
    pub n_elec_grid: Option<f64>,
    pub solvation_energy: Option<f64>,
    pub occupations: Option<(Vec<f64>, Vec<f64>)>,
    pub electronic_entropy: Option<f64>,
    pub free_energy: Option<f64>,
}

impl ScfResult {
    pub fn homo_lumo_gap(&self) -> (Option<f64>, Option<f64>) {
        let gap = |eps: &[f64], n_occ: usize| {
            (n_occ > 0 && n_occ < eps.len()).then(|| eps[n_occ] - eps[n_occ - 1])
        };
        (
            gap(&self.orbital_energies_alpha, self.n_alpha),
            gap(&self.orbital_energies_beta, self.n_beta),
        )
    }
}

pub fn run_rhf<P: IntegralProvider>(
    provider: &P,
    n_electrons: usize,
    nuclear_repulsion: f64,
    options: &ScfOptions,
) -> Result<ScfResult, ScfError> {
    let half = n_electrons / 2;
    run_scf(
        provider,
        half,
        n_electrons - half,
        Reference::Rhf,
        nuclear_repulsion,
        options,
    )
}

pub fn run_scf<P: IntegralProvider>(
    provider: &P,
    n_alpha: usize,
    n_beta: usize,
    reference: Reference,
    nuclear_repulsion: f64,
    options: &ScfOptions,
) -> Result<ScfResult, ScfError> {
    run_scf_with_xc(
        provider,
        n_alpha,
        n_beta,
        reference,
        nuclear_repulsion,
        options,
        None,
    )
}

pub fn run_scf_with_xc<P: IntegralProvider>(
    provider: &P,
    n_alpha: usize,
    n_beta: usize,
    reference: Reference,
    nuclear_repulsion: f64,
    options: &ScfOptions,
    xc: Option<&dyn XcContributor>,
) -> Result<ScfResult, ScfError> {
    run_scf_with_env(
        provider,
        n_alpha,
        n_beta,
        reference,
        nuclear_repulsion,
        options,
        xc,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_scf_with_env<P: IntegralProvider>(
    provider: &P,
    n_alpha: usize,
    n_beta: usize,
    reference: Reference,
    nuclear_repulsion: f64,
    options: &ScfOptions,
    xc: Option<&dyn XcContributor>,
    solvent: Option<&dyn SolventModel>,
) -> Result<ScfResult, ScfError> {
    let n = provider.n_basis();
    if n == 0 {
        return Err(ScfError::EmptyBasis);
    }
    if reference == Reference::Rhf && n_alpha != n_beta {
        return Err(ScfError::NotClosedShell { n_alpha, n_beta });
    }
    if xc.is_some() && reference == Reference::Rohf {
        return Err(ScfError::RohfKohnSham);
    }
    let smear_t_kt = match options.smearing {
        Some(Smearing::Fermi { temperature_k }) => {
            if reference == Reference::Rohf {
                return Err(ScfError::RohfSmearing);
            }
            if temperature_k.is_nan() || temperature_k <= 0.0 {
                return Err(ScfError::NonPositiveTemperature { temperature_k });
            }
            Some((temperature_k, temperature_k * BOLTZMANN_HT))
        }
        None => None,
    };
    let rs = xc.and_then(|x| x.range_separation());
    let c_x = match &rs {
        Some(_) => 1.0,
        None => xc.map_or(1.0, |x| x.exx_fraction()),
    };
    let incremental_fock = options.incremental_fock && rs.is_none();
    let xc_restricted = reference != Reference::Uhf;

    let s = mat_to_row_major(&provider.overlap());
    let hcore = match &options.hcore_override {
        Some(h) => {
            assert_eq!(
                h.len(),
                n * n,
                "hcore_override must be n_basis² = {} elements, got {}",
                n * n,
                h.len()
            );
            h.clone()
        }
        None => mat_to_row_major(&provider.core_hamiltonian()),
    };
    let (x, m) = canonical_orthogonalizer(&s, n, options.lindep_thresh);
    if n_alpha > m {
        return Err(ScfError::Overfilled {
            n_alpha,
            n_orbitals: m,
        });
    }

    let n_sets = if reference == Reference::Uhf { 2 } else { 1 };
    let nc = n_beta; // closed (doubly occupied) for ROHF
    let no = n_alpha - n_beta; // open (singly occupied α) for ROHF

    let guess_ao = match options.guess {
        Guess::Core => hcore.clone(),
        Guess::Gwh => gwh_matrix(&hcore, &s, n),
    };
    let (eps0, v0) = eigh(&xtax(&guess_ao, &x, n, m), m);
    let mut orbitals: Vec<Vec<f64>> = vec![v0; n_sets];
    let mut orb_eps: Vec<Vec<f64>> = vec![eps0; n_sets];

    let mut diis = Diis::new(options.diis_dim);
    let mut energy;
    let mut previous = 0.0;
    let mut previous_free = 0.0;
    let mut last_occ: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut last_ts: Option<f64> = None;
    let mut iterations = 0;
    let mut converged = false;
    let mut history = Vec::new();

    let mut da_ao;
    let mut db_ao;
    let mut eps_a;
    let mut eps_b;
    let mut last_exc = 0.0;
    let mut last_nelec = 0.0;
    let mut last_esolv = 0.0;

    let mut ja = vec![0.0; n * n];
    let mut jb = vec![0.0; n * n];
    let mut ka = vec![0.0; n * n];
    let mut kb = vec![0.0; n * n];
    let mut da_prev = vec![0.0; n * n];
    let mut db_prev = vec![0.0; n * n];
    let mut iters_since_rebuild = 0usize;
    let rebuild_period = options.fock_rebuild_period.max(1);

    loop {
        iterations += 1;

        let (va, vb): (&[f64], &[f64]) = match reference {
            Reference::Uhf => (&orbitals[0], &orbitals[1]),
            _ => (&orbitals[0], &orbitals[0]),
        };
        if let Some((t, kt)) = smear_t_kt {
            let fa = fermi_occupations(&orb_eps[0], n_alpha as f64, kt);
            let fb = if n_sets == 2 {
                fermi_occupations(&orb_eps[1], n_beta as f64, kt)
            } else {
                fa.clone()
            };
            last_ts = Some(t * BOLTZMANN_HT * (entropy_sum(&fa) + entropy_sum(&fb)));
            last_occ = Some((fa, fb));
        }
        let (da_orth, db_orth) = match &last_occ {
            Some((fa, fb)) => (orth_frac_density(va, m, fa), orth_frac_density(vb, m, fb)),
            None => (
                orth_occ_density(va, m, n_alpha),
                orth_occ_density(vb, m, n_beta),
            ),
        };
        da_ao = ao_from_orth(&x, &da_orth, n, m);
        db_ao = ao_from_orth(&x, &db_orth, n, m);

        if !incremental_fock {
            let jk =
                provider.build_jk(&[mat_from_row_major(n, &da_ao), mat_from_row_major(n, &db_ao)]);
            ja = mat_to_row_major(&jk.coulomb[0]);
            jb = mat_to_row_major(&jk.coulomb[1]);
            ka = mat_to_row_major(&jk.exchange[0]);
            kb = mat_to_row_major(&jk.exchange[1]);
        } else if iterations == 1 || iters_since_rebuild >= rebuild_period {
            let jk = provider
                .build_jk_screened(&[mat_from_row_major(n, &da_ao), mat_from_row_major(n, &db_ao)]);
            ja = mat_to_row_major(&jk.coulomb[0]);
            jb = mat_to_row_major(&jk.coulomb[1]);
            ka = mat_to_row_major(&jk.exchange[0]);
            kb = mat_to_row_major(&jk.exchange[1]);
            iters_since_rebuild = 0;
        } else {
            let dda: Vec<f64> = (0..n * n).map(|i| da_ao[i] - da_prev[i]).collect();
            let ddb: Vec<f64> = (0..n * n).map(|i| db_ao[i] - db_prev[i]).collect();
            let jk = provider
                .build_jk_screened(&[mat_from_row_major(n, &dda), mat_from_row_major(n, &ddb)]);
            let dja = mat_to_row_major(&jk.coulomb[0]);
            let djb = mat_to_row_major(&jk.coulomb[1]);
            let dka = mat_to_row_major(&jk.exchange[0]);
            let dkb = mat_to_row_major(&jk.exchange[1]);
            for i in 0..n * n {
                ja[i] += dja[i];
                jb[i] += djb[i];
                ka[i] += dka[i];
                kb[i] += dkb[i];
            }
            iters_since_rebuild += 1;
        }
        if incremental_fock {
            da_prev.copy_from_slice(&da_ao);
            db_prev.copy_from_slice(&db_ao);
        }
        if let Some(rs) = &rs {
            let klr = provider
                .build_k_erf(
                    &[mat_from_row_major(n, &da_ao), mat_from_row_major(n, &db_ao)],
                    rs.omega,
                )
                .ok_or(ScfError::RangeSeparatedUnsupported)?;
            let klr_a = mat_to_row_major(&klr[0]);
            let klr_b = mat_to_row_major(&klr[1]);
            for i in 0..n * n {
                ka[i] = rs.alpha * ka[i] + rs.beta * klr_a[i];
                kb[i] = rs.alpha * kb[i] + rs.beta * klr_b[i];
            }
        }
        let xc_contrib = xc.map(|x| x.eval(&da_ao, &db_ao, n, xc_restricted));
        if let Some(c) = &xc_contrib {
            last_exc = c.exc;
            last_nelec = c.n_elec_grid;
        }

        let mut fa = vec![0.0; n * n];
        let mut fb = vec![0.0; n * n];
        for i in 0..n * n {
            let j_total = ja[i] + jb[i];
            fa[i] = hcore[i] + j_total - c_x * ka[i];
            fb[i] = hcore[i] + j_total - c_x * kb[i];
        }
        if let Some(c) = &xc_contrib {
            for i in 0..n * n {
                fa[i] += c.vxc_alpha[i];
                fb[i] += c.vxc_beta[i];
            }
        }

        let solv_contrib = solvent.map(|s| {
            let d_total: Vec<f64> = (0..n * n).map(|i| da_ao[i] + db_ao[i]).collect();
            s.eval(&d_total, n)
        });
        if let Some(c) = &solv_contrib {
            last_esolv = c.e_solv;
            for i in 0..n * n {
                fa[i] += c.v_solv[i];
                fb[i] += c.v_solv[i];
            }
        }

        let electronic = match (&xc_contrib, &solv_contrib) {
            (None, None) => {
                let mut e = 0.0;
                for i in 0..n * n {
                    e += 0.5
                        * ((da_ao[i] + db_ao[i]) * hcore[i] + da_ao[i] * fa[i] + db_ao[i] * fb[i]);
                }
                e
            }
            (xc_c, solv_c) => {
                let exc = xc_c.as_ref().map_or(0.0, |c| c.exc);
                let esolv = solv_c.as_ref().map_or(0.0, |c| c.e_solv);
                ks_electronic_energy(
                    &da_ao,
                    &db_ao,
                    &hcore,
                    &ja,
                    &jb,
                    &ka,
                    &kb,
                    c_x,
                    exc + esolv,
                    n,
                )
            }
        };
        energy = electronic + nuclear_repulsion;

        let fa_orth = xtax(&fa, &x, n, m);
        let fb_orth = xtax(&fb, &x, n, m);
        let (effective, errors): (Vec<Vec<f64>>, Vec<Vec<f64>>) = match reference {
            Reference::Rhf => (
                vec![fa_orth.clone()],
                vec![commutator(&fa_orth, &da_orth, m)],
            ),
            Reference::Uhf => (
                vec![fa_orth.clone(), fb_orth.clone()],
                vec![
                    commutator(&fa_orth, &da_orth, m),
                    commutator(&fb_orth, &db_orth, m),
                ],
            ),
            Reference::Rohf => {
                let reff = rohf_effective_fock(&fa_orth, &fb_orth, &orbitals[0], m, nc, no);
                let mut d_total = da_orth.clone();
                for i in 0..m * m {
                    d_total[i] += db_orth[i];
                }
                (vec![reff.clone()], vec![commutator(&reff, &d_total, m)])
            }
        };

        let error_cat: Vec<f64> = errors.concat();
        let error_norm = max_abs(&error_cat);
        history.push(ScfIteration {
            iteration: iterations,
            energy,
            error_norm,
        });

        eps_a = eigh(&effective[0], m).0;
        eps_b = if reference == Reference::Uhf {
            eigh(&effective[1], m).0
        } else {
            eps_a.clone()
        };

        let free = energy - last_ts.unwrap_or(0.0);
        if iterations > 1
            && (energy - previous).abs() < options.energy_tol
            && (free - previous_free).abs() < options.energy_tol
            && error_norm < options.error_tol
        {
            converged = true;
            break;
        }
        previous = energy;
        previous_free = free;
        if iterations >= options.max_iter {
            break;
        }

        diis.push(effective.concat(), error_cat);
        let extrapolated = diis.extrapolate();
        for set in 0..n_sets {
            let block = &extrapolated[set * m * m..(set + 1) * m * m];
            let n_occ = if set == 0 { n_alpha } else { n_beta };
            let shifted = apply_level_shift(block, &orbitals[set], m, n_occ, options.level_shift);
            let (eps_new, v_new) = eigh(&shifted, m);
            orbitals[set] = v_new;
            orb_eps[set] = eps_new;
        }
    }

    if incremental_fock && iters_since_rebuild > 0 {
        let jk = provider
            .build_jk_screened(&[mat_from_row_major(n, &da_ao), mat_from_row_major(n, &db_ao)]);
        ja = mat_to_row_major(&jk.coulomb[0]);
        jb = mat_to_row_major(&jk.coulomb[1]);
        ka = mat_to_row_major(&jk.exchange[0]);
        kb = mat_to_row_major(&jk.exchange[1]);
        let electronic = if xc.is_some() || solvent.is_some() {
            let env = if xc.is_some() { last_exc } else { 0.0 }
                + if solvent.is_some() { last_esolv } else { 0.0 };
            ks_electronic_energy(&da_ao, &db_ao, &hcore, &ja, &jb, &ka, &kb, c_x, env, n)
        } else {
            let mut e = 0.0;
            for i in 0..n * n {
                let j_total = ja[i] + jb[i];
                let fa_i = hcore[i] + j_total - ka[i];
                let fb_i = hcore[i] + j_total - kb[i];
                e += 0.5 * ((da_ao[i] + db_ao[i]) * hcore[i] + da_ao[i] * fa_i + db_ao[i] * fb_i);
            }
            e
        };
        energy = electronic + nuclear_repulsion;
    }

    let (va, vb): (&[f64], &[f64]) = match reference {
        Reference::Uhf => (&orbitals[0], &orbitals[1]),
        _ => (&orbitals[0], &orbitals[0]),
    };
    let spin_squared = spin_squared(va, n_alpha, vb, n_beta, m);

    let mut density = vec![0.0; n * n];
    for i in 0..n * n {
        density[i] = da_ao[i] + db_ao[i];
    }

    let mo_coeff_alpha = mul(&x, &orbitals[0], n, m, m);
    let mo_coeff_beta = if reference == Reference::Uhf {
        mul(&x, &orbitals[1], n, m, m)
    } else {
        mo_coeff_alpha.clone()
    };

    Ok(ScfResult {
        reference,
        energy,
        electronic_energy: energy - nuclear_repulsion,
        nuclear_repulsion,
        converged,
        iterations,
        n_basis: n,
        n_orbitals: m,
        n_alpha,
        n_beta,
        density,
        density_alpha: da_ao,
        density_beta: db_ao,
        orbital_energies_alpha: eps_a,
        orbital_energies_beta: eps_b,
        mo_coeff_alpha,
        mo_coeff_beta,
        spin_squared,
        history,
        xc_energy: xc.map(|_| last_exc),
        n_elec_grid: xc.map(|_| last_nelec),
        solvation_energy: solvent.map(|_| last_esolv),
        free_energy: last_ts.map(|ts| energy - ts),
        electronic_entropy: last_ts,
        occupations: last_occ,
    })
}

#[allow(clippy::too_many_arguments)]
fn ks_electronic_energy(
    da: &[f64],
    db: &[f64],
    hcore: &[f64],
    ja: &[f64],
    jb: &[f64],
    ka: &[f64],
    kb: &[f64],
    c_x: f64,
    exc: f64,
    n: usize,
) -> f64 {
    let mut e = exc;
    for i in 0..n * n {
        let dt = da[i] + db[i];
        e += dt * hcore[i] + 0.5 * dt * (ja[i] + jb[i])
            - 0.5 * c_x * (da[i] * ka[i] + db[i] * kb[i]);
    }
    e
}

fn gwh_matrix(hcore: &[f64], s: &[f64], n: usize) -> Vec<f64> {
    const K: f64 = 1.75;
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = if i == j {
                hcore[i * n + i]
            } else {
                0.5 * K * s[i * n + j] * (hcore[i * n + i] + hcore[j * n + j])
            };
        }
    }
    g
}

fn rohf_effective_fock(
    fa_orth: &[f64],
    fb_orth: &[f64],
    v: &[f64],
    m: usize,
    nc: usize,
    no: usize,
) -> Vec<f64> {
    let fa_mo = vtav(fa_orth, v, m); // Vᵀ Fα V
    let fb_mo = vtav(fb_orth, v, m);
    let n_alpha = nc + no;

    let space = |i: usize| -> u8 {
        if i < nc {
            0
        } else if i < n_alpha {
            1
        } else {
            2
        }
    };

    let mut reff_mo = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let idx = i * m + j;
            reff_mo[idx] = match (space(i), space(j)) {
                (0, 1) | (1, 0) => fb_mo[idx],
                (1, 2) | (2, 1) => fa_mo[idx],
                _ => 0.5 * (fa_mo[idx] + fb_mo[idx]),
            };
        }
    }

    let vt = transpose(v, m, m);
    let vr = mul(v, &reff_mo, m, m, m);
    mul(&vr, &vt, m, m, m)
}

fn apply_level_shift(f: &[f64], v: &[f64], m: usize, n_occ: usize, shift: f64) -> Vec<f64> {
    if shift == 0.0 {
        return f.to_vec();
    }
    let d_occ = orth_occ_density(v, m, n_occ);
    let mut out = f.to_vec();
    for i in 0..m {
        for j in 0..m {
            let identity = if i == j { 1.0 } else { 0.0 };
            out[i * m + j] += shift * (identity - d_occ[i * m + j]);
        }
    }
    out
}

fn spin_squared(va: &[f64], n_alpha: usize, vb: &[f64], n_beta: usize, m: usize) -> f64 {
    let sz = (n_alpha as f64 - n_beta as f64) / 2.0;
    let mut cross = 0.0;
    for i in 0..n_alpha {
        for j in 0..n_beta {
            let mut overlap = 0.0;
            for k in 0..m {
                overlap += va[k * m + i] * vb[k * m + j];
            }
            cross += overlap * overlap;
        }
    }
    sz * (sz + 1.0) + n_beta as f64 - cross
}
