use crate::integrals::integral;
use crate::linalg::{gemm, mat_from_row_major, symmetric_eigh};
use crate::scf::{Reference, ScfResult};
use rayon::prelude::*;
use thiserror::Error;

pub const METRIC_EIG_CUTOFF: f64 = 1e-10;

#[derive(Debug, Error)]
pub enum RiMp2Error {
    #[error("RI-MP2 requires a closed-shell RHF reference (got {0:?})")]
    OpenShellReference(Reference),
    #[error("UHF RI-MP2 requires a UHF reference (got {0:?})")]
    NotUhfReference(Reference),
    #[error("RI-MP2 auxiliary metric (P|Q) is numerically singular")]
    MetricSingular,
}

#[derive(Debug, Clone, Copy)]
pub struct RiMp2Result {
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub opposite_spin: f64,
    pub same_spin: f64,
    pub n_frozen: usize,
    pub naux: usize,
}

#[derive(Debug, Clone)]
pub struct RiMp2B {
    pub b: Vec<f64>,
    pub naux: usize,
    pub n_act: usize,
    pub n_virt: usize,
}

impl RiMp2B {
    pub fn iajb(&self, i: usize, a: usize, j: usize, b: usize) -> f64 {
        let row =
            |occ: usize, vir: usize| &self.b[(occ * self.n_virt + vir) * self.naux..][..self.naux];
        row(i, a).iter().zip(row(j, b)).map(|(x, y)| x * y).sum()
    }
}

pub fn rhf_ri_mp2_b(
    basis: &integral::Basis,
    aux: &integral::Basis,
    scf: &ScfResult,
    n_frozen: usize,
) -> Result<RiMp2B, RiMp2Error> {
    if scf.reference != Reference::Rhf {
        return Err(RiMp2Error::OpenShellReference(scf.reference));
    }
    let n_occ = scf.n_alpha; // doubly occupied (RHF)
    assert!(n_frozen <= n_occ, "more frozen orbitals than occupied");
    let naux = aux.nao();

    let jhalf = metric_inverse_sqrt(&aux.eri_2c(), naux)?;
    let a3c = build_3c(basis, aux);
    Ok(spin_b(
        &scf.mo_coeff_alpha,
        scf.n_basis,
        scf.n_orbitals,
        n_frozen,
        n_occ,
        &a3c,
        naux,
        &jhalf,
    ))
}

pub fn uhf_ri_mp2_b(
    basis: &integral::Basis,
    aux: &integral::Basis,
    scf: &ScfResult,
    n_frozen: usize,
) -> Result<(RiMp2B, RiMp2B), RiMp2Error> {
    if scf.reference != Reference::Uhf {
        return Err(RiMp2Error::NotUhfReference(scf.reference));
    }
    assert!(
        n_frozen <= scf.n_alpha.min(scf.n_beta),
        "more frozen orbitals than occupied in a spin channel"
    );
    let naux = aux.nao();
    let jhalf = metric_inverse_sqrt(&aux.eri_2c(), naux)?;
    let a3c = build_3c(basis, aux);
    let (n, m) = (scf.n_basis, scf.n_orbitals);
    let ba = spin_b(
        &scf.mo_coeff_alpha,
        n,
        m,
        n_frozen,
        scf.n_alpha,
        &a3c,
        naux,
        &jhalf,
    );
    let bb = spin_b(
        &scf.mo_coeff_beta,
        n,
        m,
        n_frozen,
        scf.n_beta,
        &a3c,
        naux,
        &jhalf,
    );
    Ok((ba, bb))
}

fn build_3c(basis: &integral::Basis, aux: &integral::Basis) -> Vec<f64> {
    let builder = basis.eri_3c_builder(aux);
    let mut a3c = vec![0.0; builder.output_len()];
    {
        let mut tasks = builder.partition(&mut a3c);
        let shells = basis.shells();
        tasks.sort_unstable_by(|a, b| {
            bra_pair_cost(shells, b.bra()).cmp(&bra_pair_cost(shells, a.bra()))
        });
        tasks.par_iter_mut().for_each(|task| builder.fill(task));
    }
    a3c
}

#[allow(clippy::too_many_arguments)]
fn spin_b(
    c: &[f64],
    n: usize,
    m: usize,
    n_frozen: usize,
    n_occ: usize,
    a3c: &[f64],
    naux: usize,
    jhalf: &[f64],
) -> RiMp2B {
    let n_act = n_occ - n_frozen;
    let n_virt = m - n_occ;

    let c_occ_t: Vec<f64> = (0..n_act)
        .flat_map(|k| (0..n).map(move |mu| c[mu * m + n_frozen + k]))
        .collect();
    let c_virt_t: Vec<f64> = (0..n_virt)
        .flat_map(|k| (0..n).map(move |mu| c[mu * m + n_occ + k]))
        .collect();

    let t1 = gemm(&c_occ_t, n_act, n, a3c, n * naux);

    let mut a_mo = vec![0.0; n_act * n_virt * naux];
    a_mo.par_chunks_mut(n_virt * naux)
        .zip(t1.par_chunks(n * naux))
        .for_each(|(dst, src)| {
            dst.copy_from_slice(&gemm(&c_virt_t, n_virt, n, src, naux));
        });
    drop(t1);

    let b = gemm(&a_mo, n_act * n_virt, naux, jhalf, naux);

    RiMp2B {
        b,
        naux,
        n_act,
        n_virt,
    }
}

pub fn rhf_ri_mp2(
    basis: &integral::Basis,
    aux: &integral::Basis,
    scf: &ScfResult,
    n_frozen: usize,
) -> Result<RiMp2Result, RiMp2Error> {
    let bt = rhf_ri_mp2_b(basis, aux, scf, n_frozen)?;
    let (n_act, n_virt, naux) = (bt.n_act, bt.n_virt, bt.naux);
    let n_occ = scf.n_alpha;
    let eps = &scf.orbital_energies_alpha;

    let bt_t = transpose_pair_blocks(&bt);

    let partials: Vec<(f64, f64)> = (0..n_act)
        .into_par_iter()
        .map(|i| {
            let b_i = &bt.b[i * n_virt * naux..(i + 1) * n_virt * naux];
            let eps_i = eps[n_frozen + i];
            let mut e_os = 0.0;
            let mut e_ss = 0.0;
            for j in 0..n_act {
                let bt_j = &bt_t[j * n_virt * naux..(j + 1) * n_virt * naux];
                let g = gemm(b_i, n_virt, naux, bt_j, n_virt);
                let eps_ij = eps_i + eps[n_frozen + j];
                for a in 0..n_virt {
                    let eps_ija = eps_ij - eps[n_occ + a];
                    for vb in 0..n_virt {
                        let denom = eps_ija - eps[n_occ + vb];
                        let iajb = g[a * n_virt + vb];
                        let ibja = g[vb * n_virt + a];
                        e_os += iajb * iajb / denom;
                        e_ss += iajb * (iajb - ibja) / denom;
                    }
                }
            }
            (e_os, e_ss)
        })
        .collect();
    let (e_os, e_ss) = partials
        .iter()
        .fold((0.0, 0.0), |(os, ss), &(po, ps)| (os + po, ss + ps));

    let correlation_energy = e_os + e_ss;
    Ok(RiMp2Result {
        correlation_energy,
        total_energy: scf.energy + correlation_energy,
        opposite_spin: e_os,
        same_spin: e_ss,
        n_frozen,
        naux,
    })
}

pub fn uhf_ri_mp2(
    basis: &integral::Basis,
    aux: &integral::Basis,
    scf: &ScfResult,
    n_frozen: usize,
) -> Result<RiMp2Result, RiMp2Error> {
    let (ba, bb) = uhf_ri_mp2_b(basis, aux, scf, n_frozen)?;
    let naux = ba.naux;
    let eps_a = &scf.orbital_energies_alpha;
    let eps_b = &scf.orbital_energies_beta;

    let e_aa = same_spin_pair_energy(&ba, eps_a, n_frozen, scf.n_alpha);
    let e_bb = same_spin_pair_energy(&bb, eps_b, n_frozen, scf.n_beta);

    let bt_b = transpose_pair_blocks(&bb);
    let (nva, nvb) = (ba.n_virt, bb.n_virt);
    let partials: Vec<f64> = (0..ba.n_act)
        .into_par_iter()
        .map(|i| {
            let b_i = &ba.b[i * nva * naux..(i + 1) * nva * naux];
            let eps_i = eps_a[n_frozen + i];
            let mut e_os = 0.0;
            for j in 0..bb.n_act {
                let bt_j = &bt_b[j * nvb * naux..(j + 1) * nvb * naux];
                let g = gemm(b_i, nva, naux, bt_j, nvb);
                let eps_ij = eps_i + eps_b[n_frozen + j];
                for a in 0..nva {
                    let eps_ija = eps_ij - eps_a[scf.n_alpha + a];
                    for vb in 0..nvb {
                        let denom = eps_ija - eps_b[scf.n_beta + vb];
                        let v = g[a * nvb + vb];
                        e_os += v * v / denom;
                    }
                }
            }
            e_os
        })
        .collect();
    let e_os: f64 = partials.iter().sum();

    let e_ss = e_aa + e_bb;
    let correlation_energy = e_os + e_ss;
    Ok(RiMp2Result {
        correlation_energy,
        total_energy: scf.energy + correlation_energy,
        opposite_spin: e_os,
        same_spin: e_ss,
        n_frozen,
        naux,
    })
}

fn same_spin_pair_energy(bt: &RiMp2B, eps: &[f64], n_frozen: usize, n_occ: usize) -> f64 {
    let (n_act, n_virt, naux) = (bt.n_act, bt.n_virt, bt.naux);
    let bt_t = transpose_pair_blocks(bt);
    let partials: Vec<f64> = (0..n_act)
        .into_par_iter()
        .map(|i| {
            let b_i = &bt.b[i * n_virt * naux..(i + 1) * n_virt * naux];
            let eps_i = eps[n_frozen + i];
            let mut e = 0.0;
            for j in 0..n_act {
                let bt_j = &bt_t[j * n_virt * naux..(j + 1) * n_virt * naux];
                let g = gemm(b_i, n_virt, naux, bt_j, n_virt);
                let eps_ij = eps_i + eps[n_frozen + j];
                for a in 0..n_virt {
                    let eps_ija = eps_ij - eps[n_occ + a];
                    for vb in 0..n_virt {
                        let denom = eps_ija - eps[n_occ + vb];
                        let anti = g[a * n_virt + vb] - g[vb * n_virt + a];
                        e += 0.25 * anti * anti / denom;
                    }
                }
            }
            e
        })
        .collect();
    partials.iter().sum()
}

fn transpose_pair_blocks(bt: &RiMp2B) -> Vec<f64> {
    let (n_act, n_virt, naux) = (bt.n_act, bt.n_virt, bt.naux);
    let mut out = vec![0.0; n_act * n_virt * naux];
    for j in 0..n_act {
        let src = &bt.b[j * n_virt * naux..(j + 1) * n_virt * naux];
        let dst = &mut out[j * n_virt * naux..(j + 1) * n_virt * naux];
        for vb in 0..n_virt {
            for q in 0..naux {
                dst[q * n_virt + vb] = src[vb * naux + q];
            }
        }
    }
    out
}

fn metric_inverse_sqrt(metric: &[f64], naux: usize) -> Result<Vec<f64>, RiMp2Error> {
    let eig = symmetric_eigh(&mat_from_row_major(naux, metric));
    let lambda_max = eig.values.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = METRIC_EIG_CUTOFF * lambda_max;
    let kept: Vec<usize> = (0..naux)
        .filter(|&k| eig.values[k] > cutoff && eig.values[k] > 0.0)
        .collect();
    if kept.is_empty() {
        return Err(RiMp2Error::MetricSingular);
    }
    let nk = kept.len();
    let mut w = vec![0.0; naux * nk];
    let mut wt = vec![0.0; nk * naux];
    for (col, &k) in kept.iter().enumerate() {
        let s = eig.values[k].powf(-0.25);
        for p in 0..naux {
            let v = eig.vectors[(p, k)] * s;
            w[p * nk + col] = v;
            wt[col * naux + p] = v;
        }
    }
    Ok(gemm(&w, naux, nk, &wt, naux))
}

fn bra_pair_cost(shells: &[integral::Shell], (i, j): (usize, usize)) -> u64 {
    fn weight(s: &integral::Shell) -> u64 {
        let l = s.l() as u64;
        let n_cart = (l + 1) * (l + 2) / 2;
        s.n_prim() as u64 * n_cart * (l + 1)
    }
    weight(&shells[i]) * weight(&shells[j])
}
