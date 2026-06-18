use crate::integrals::InCoreEri;
use crate::scf::ScfResult;

use crate::cc::transform::{column_block, transform_block};

#[derive(Debug, Clone, Copy)]
pub struct Mp2Result {
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub opposite_spin: f64,
    pub same_spin: f64,
    pub n_frozen: usize,
}

pub fn rhf_mp2<P: InCoreEri>(provider: &P, scf: &ScfResult, n_frozen: usize) -> Mp2Result {
    let n = scf.n_basis;
    let m = scf.n_orbitals;
    let n_occ = scf.n_alpha; // doubly occupied (RHF)
    assert!(n_frozen <= n_occ, "more frozen orbitals than occupied");
    let n_act = n_occ - n_frozen;
    let n_virt = m - n_occ;
    let eps = &scf.orbital_energies_alpha;
    let c = &scf.mo_coeff_alpha;

    let c_occ = column_block(c, n, m, n_frozen, n_act);
    let c_virt = column_block(c, n, m, n_occ, n_virt);

    let ovov = transform_block(provider.ao_eri(), n, [&c_occ, &c_virt, &c_occ, &c_virt]);
    let g = ovov.data();
    let idx = |i: usize, a: usize, j: usize, b: usize| ((i * n_virt + a) * n_act + j) * n_virt + b;

    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for i in 0..n_act {
        let eps_i = eps[n_frozen + i];
        for j in 0..n_act {
            let eps_ij = eps_i + eps[n_frozen + j];
            for a in 0..n_virt {
                let eps_ija = eps_ij - eps[n_occ + a];
                for b in 0..n_virt {
                    let denom = eps_ija - eps[n_occ + b];
                    let iajb = g[idx(i, a, j, b)];
                    let ibja = g[idx(i, b, j, a)];
                    e_os += iajb * iajb / denom;
                    e_ss += iajb * (iajb - ibja) / denom;
                }
            }
        }
    }

    let correlation_energy = e_os + e_ss;
    Mp2Result {
        correlation_energy,
        total_energy: scf.energy + correlation_energy,
        opposite_spin: e_os,
        same_spin: e_ss,
        n_frozen,
    }
}

pub fn uhf_mp2<P: InCoreEri>(provider: &P, scf: &ScfResult, n_frozen: usize) -> Mp2Result {
    let n = scf.n_basis;
    let m = scf.n_orbitals;
    assert!(
        n_frozen <= scf.n_beta.min(scf.n_alpha),
        "more frozen orbitals than occupied in a spin channel"
    );
    let spin = |n_occ: usize, c: &[f64]| {
        let n_act = n_occ - n_frozen;
        let c_occ = column_block(c, n, m, n_frozen, n_act);
        let c_virt = column_block(c, n, m, n_occ, m - n_occ);
        (n_act, m - n_occ, c_occ, c_virt)
    };
    let (na_act, na_virt, ca_occ, ca_virt) = spin(scf.n_alpha, &scf.mo_coeff_alpha);
    let (nb_act, nb_virt, cb_occ, cb_virt) = spin(scf.n_beta, &scf.mo_coeff_beta);
    let eps_a = &scf.orbital_energies_alpha;
    let eps_b = &scf.orbital_energies_beta;

    let same_spin_energy = |n_act: usize, n_virt: usize, g: &[f64], eps: &[f64], n_occ: usize| {
        let idx =
            |i: usize, a: usize, j: usize, b: usize| ((i * n_virt + a) * n_act + j) * n_virt + b;
        let mut e = 0.0;
        for i in 0..n_act {
            for j in 0..n_act {
                let eps_ij = eps[n_frozen + i] + eps[n_frozen + j];
                for a in 0..n_virt {
                    let eps_ija = eps_ij - eps[n_occ + a];
                    for b in 0..n_virt {
                        let denom = eps_ija - eps[n_occ + b];
                        let anti = g[idx(i, a, j, b)] - g[idx(i, b, j, a)];
                        e += 0.25 * anti * anti / denom;
                    }
                }
            }
        }
        e
    };

    let ao = provider.ao_eri();
    let g_aa = transform_block(ao, n, [&ca_occ, &ca_virt, &ca_occ, &ca_virt]);
    let e_aa = same_spin_energy(na_act, na_virt, g_aa.data(), eps_a, scf.n_alpha);
    drop(g_aa);
    let g_bb = transform_block(ao, n, [&cb_occ, &cb_virt, &cb_occ, &cb_virt]);
    let e_bb = same_spin_energy(nb_act, nb_virt, g_bb.data(), eps_b, scf.n_beta);
    drop(g_bb);

    let g_ab = transform_block(ao, n, [&ca_occ, &ca_virt, &cb_occ, &cb_virt]);
    let g = g_ab.data();
    let mut e_os = 0.0;
    for i in 0..na_act {
        for a in 0..na_virt {
            let eps_ia = eps_a[n_frozen + i] - eps_a[scf.n_alpha + a];
            for j in 0..nb_act {
                let eps_iaj = eps_ia + eps_b[n_frozen + j];
                for b in 0..nb_virt {
                    let denom = eps_iaj - eps_b[scf.n_beta + b];
                    let v = g[((i * na_virt + a) * nb_act + j) * nb_virt + b];
                    e_os += v * v / denom;
                }
            }
        }
    }

    let e_ss = e_aa + e_bb;
    let correlation_energy = e_os + e_ss;
    Mp2Result {
        correlation_energy,
        total_energy: scf.energy + correlation_energy,
        opposite_spin: e_os,
        same_spin: e_ss,
        n_frozen,
    }
}
