//! Grimme-type dispersion and basis-set corrections: D3(BJ), D4, gCP, and SRB.

use crate::core::Molecule;

mod d4;
pub mod d4data;
pub mod data;
mod gcp;
mod srb;

pub use d4::{D4Params, coordination_numbers_d4, d4_energy, d4_energy_gradient, eeq_charges};
pub use gcp::{
    GcpParams, gcp_energy, gcp_energy_gradient, gcp_r2scan3c_energy, gcp_r2scan3c_energy_gradient,
};
pub use srb::{SrbParams, srb_energy, srb_energy_gradient};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dispersion {
    D3(D3Params),
    D4(D4Params),
}

impl Dispersion {
    pub fn for_method(d4: bool, method: &str) -> Option<Dispersion> {
        if d4 {
            D4Params::for_method(method).map(Dispersion::D4)
        } else {
            D3Params::for_method(method).map(Dispersion::D3)
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Dispersion::D3(_) => "D3(BJ)",
            Dispersion::D4(_) => "D4",
        }
    }

    pub fn energy(&self, mol: &Molecule) -> f64 {
        match self {
            Dispersion::D3(p) => d3bj_energy(mol, p),
            Dispersion::D4(p) => d4_energy(mol, p),
        }
    }

    pub fn energy_gradient(&self, mol: &Molecule) -> (f64, Vec<[f64; 3]>) {
        match self {
            Dispersion::D3(p) => d3bj_energy_gradient(mol, p),
            Dispersion::D4(p) => d4_energy_gradient(mol, p),
        }
    }
}

pub(crate) fn without_ghosts(mol: &Molecule) -> Option<(Molecule, Vec<usize>)> {
    if !mol.has_ghosts() {
        return None;
    }
    let mut map = Vec::new();
    let atoms = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.ghost)
        .map(|(i, a)| {
            map.push(i);
            *a
        })
        .collect();
    Some((Molecule::new(atoms, mol.charge, mol.multiplicity), map))
}

pub(crate) fn scatter_gradient(
    grad: Vec<[f64; 3]>,
    map: &[usize],
    n_total: usize,
) -> Vec<[f64; 3]> {
    let mut full = vec![[0.0; 3]; n_total];
    for (g, &i) in grad.into_iter().zip(map) {
        full[i] = g;
    }
    full
}

const K1: f64 = 16.0;
const K3: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D3Params {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub a1: f64,
    pub a2: f64,
}

impl D3Params {
    pub const B3LYP_3C: D3Params = D3Params {
        s6: 1.0,
        s8: 1.9889,
        s9: 1.0,
        a1: 0.3981,
        a2: 4.4211,
    };

    pub const B97_3C: D3Params = D3Params {
        s6: 1.0,
        s8: 1.50,
        s9: 1.0,
        a1: 0.37,
        a2: 4.10,
    };

    pub const PBEH_3C: D3Params = D3Params {
        s6: 1.0,
        s8: 0.0,
        s9: 1.0,
        a1: 0.4860,
        a2: 4.5000,
    };

    pub fn for_method(name: &str) -> Option<D3Params> {
        let lower = name.to_ascii_lowercase();
        let key = if lower == "b3lyp5" {
            "b3lyp"
        } else {
            lower.as_str()
        };
        data::BJ_PARAMS
            .iter()
            .find(|(m, _, _, _)| *m == key)
            .map(|&(_, s8, a1, a2)| D3Params {
                s6: 1.0,
                s8,
                s9: 0.0,
                a1,
                a2,
            })
    }
}

struct AtomData {
    zi: Vec<usize>,
    rcov: Vec<f64>,
    r4r2: Vec<f64>,
    xyz: Vec<[f64; 3]>,
}

fn atom_data(mol: &Molecule) -> AtomData {
    let zi: Vec<usize> = mol
        .atoms
        .iter()
        .map(|a| {
            let z = a.element.z() as usize;
            assert!(
                (1..=data::MAX_Z).contains(&z),
                "D3(BJ) supports H-Ar only (got Z = {z})"
            );
            z - 1
        })
        .collect();
    AtomData {
        rcov: zi.iter().map(|&i| data::RCOV_D3[i]).collect(),
        r4r2: zi.iter().map(|&i| data::SQRT_Z_R4R2[i]).collect(),
        xyz: mol.atoms.iter().map(|a| a.position).collect(),
        zi,
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

pub fn coordination_numbers(mol: &Molecule) -> Vec<f64> {
    let ad = atom_data(mol);
    cn_impl(&ad)
}

fn cn_impl(ad: &AtomData) -> Vec<f64> {
    let nat = ad.zi.len();
    let mut cn = vec![0.0; nat];
    for i in 0..nat {
        for j in 0..i {
            let r = distance(ad.xyz[i], ad.xyz[j]);
            let f = 1.0 / (1.0 + (-K1 * ((ad.rcov[i] + ad.rcov[j]) / r - 1.0)).exp());
            cn[i] += f;
            cn[j] += f;
        }
    }
    cn
}

fn weights(zi: usize, cn: f64, dw: Option<&mut [f64; data::MAX_REF]>) -> [f64; data::MAX_REF] {
    let nref = data::N_REF[zi];
    let refs = &data::REF_CN[zi];
    let mut gw = [0.0; data::MAX_REF];
    let mut norm = 0.0;
    let mut dnorm = 0.0;
    for (iref, w) in gw.iter_mut().enumerate().take(nref) {
        *w = (-K3 * (cn - refs[iref]) * (cn - refs[iref])).exp();
        norm += *w;
        dnorm += 2.0 * K3 * (refs[iref] - cn) * *w;
    }
    let inv = 1.0 / norm;
    let cn_max = refs[..nref]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0.0; data::MAX_REF];
    let mut dout = [0.0; data::MAX_REF];
    for iref in 0..nref {
        let mut gwk = gw[iref] * inv;
        if !gwk.is_finite() {
            gwk = if refs[iref] == cn_max { 1.0 } else { 0.0 };
        }
        out[iref] = gwk;
        let expd = 2.0 * K3 * (refs[iref] - cn) * gw[iref];
        let mut dgwk = expd * inv - gw[iref] * dnorm * inv * inv;
        if !dgwk.is_finite() {
            dgwk = 0.0;
        }
        dout[iref] = dgwk;
    }
    if let Some(d) = dw {
        *d = dout;
    }
    out
}

fn ref_c6(zi: usize, zj: usize, iref: usize, jref: usize) -> f64 {
    let (hi, lo, ir, jr) = if zi >= zj {
        (zi, zj, iref, jref)
    } else {
        (zj, zi, jref, iref)
    };
    let ic = lo + hi * (hi + 1) / 2;
    data::C6AB[ic * data::MAX_REF * data::MAX_REF + ir * data::MAX_REF + jr]
}

fn pair_c6(
    zi: usize,
    zj: usize,
    wi: &[f64; data::MAX_REF],
    wj: &[f64; data::MAX_REF],
    dwi: Option<(&[f64; data::MAX_REF], &[f64; data::MAX_REF])>,
) -> (f64, f64, f64) {
    let mut c6 = 0.0;
    let (mut dci, mut dcj) = (0.0, 0.0);
    for iref in 0..data::N_REF[zi] {
        for jref in 0..data::N_REF[zj] {
            let rc6 = ref_c6(zi, zj, iref, jref);
            c6 += wi[iref] * wj[jref] * rc6;
            if let Some((di, dj)) = dwi {
                dci += di[iref] * wj[jref] * rc6;
                dcj += wi[iref] * dj[jref] * rc6;
            }
        }
    }
    (c6, dci, dcj)
}

pub fn d3bj_energy(mol: &Molecule, params: &D3Params) -> f64 {
    if let Some((real, _)) = without_ghosts(mol) {
        return d3bj_energy(&real, params);
    }
    d3_impl(mol, params, false).0
}

pub fn d3bj_energy_gradient(mol: &Molecule, params: &D3Params) -> (f64, Vec<[f64; 3]>) {
    if let Some((real, map)) = without_ghosts(mol) {
        let (e, g) = d3bj_energy_gradient(&real, params);
        return (e, scatter_gradient(g, &map, mol.len()));
    }
    let (e, g) = d3_impl(mol, params, true);
    (e, g.expect("gradient requested"))
}

fn d3_impl(mol: &Molecule, params: &D3Params, want_grad: bool) -> (f64, Option<Vec<[f64; 3]>>) {
    let ad = atom_data(mol);
    let nat = ad.zi.len();
    let cn = cn_impl(&ad);

    let mut w = Vec::with_capacity(nat);
    let mut dw = Vec::with_capacity(nat);
    for (&zi, &cni) in ad.zi.iter().zip(&cn) {
        let mut d = [0.0; data::MAX_REF];
        w.push(weights(zi, cni, Some(&mut d)));
        dw.push(d);
    }

    let mut energy = 0.0;
    let mut gradient = vec![[0.0; 3]; nat];
    let mut dedcn = vec![0.0; nat];

    for i in 0..nat {
        for j in 0..i {
            let (c6, dc6dcni, dc6dcnj) =
                pair_c6(ad.zi[i], ad.zi[j], &w[i], &w[j], Some((&dw[i], &dw[j])));
            let rrij = 3.0 * ad.r4r2[i] * ad.r4r2[j];
            let r0 = params.a1 * rrij.sqrt() + params.a2;
            let vec = [
                ad.xyz[i][0] - ad.xyz[j][0],
                ad.xyz[i][1] - ad.xyz[j][1],
                ad.xyz[i][2] - ad.xyz[j][2],
            ];
            let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
            let r = r2.sqrt();
            let r6 = r2 * r2 * r2;
            let r8 = r6 * r2;
            let r0_2 = r0 * r0;
            let r0_6 = r0_2 * r0_2 * r0_2;
            let r0_8 = r0_6 * r0_2;
            let t6 = 1.0 / (r6 + r0_6);
            let t8 = 1.0 / (r8 + r0_8);
            let damp = params.s6 * t6 + params.s8 * rrij * t8;
            energy -= c6 * damp;
            dedcn[i] -= dc6dcni * damp;
            dedcn[j] -= dc6dcnj * damp;
            let ddampdr = -(6.0 * params.s6 * r2 * r2 * r * t6 * t6
                + 8.0 * params.s8 * rrij * r6 * r * t8 * t8);
            let de_dr = -c6 * ddampdr;
            for (k, v) in vec.iter().enumerate() {
                let g = de_dr * v / r;
                gradient[i][k] += g;
                gradient[j][k] -= g;
            }
        }
    }

    if params.s9 != 0.0 && nat >= 3 {
        let mut c6t = vec![0.0; nat * nat];
        let mut dc6_i = vec![0.0; nat * nat];
        for i in 0..nat {
            for j in 0..i {
                let (c6, dci, dcj) =
                    pair_c6(ad.zi[i], ad.zi[j], &w[i], &w[j], Some((&dw[i], &dw[j])));
                c6t[i * nat + j] = c6;
                c6t[j * nat + i] = c6;
                dc6_i[i * nat + j] = dci;
                dc6_i[j * nat + i] = dcj;
            }
        }
        let rs9 = 4.0 / 3.0;
        let r0pair =
            |i: usize, j: usize| -> f64 { rs9 * data::r0ab_bohr(ad.zi[i] + 1, ad.zi[j] + 1) };
        let beta = 16.0 / 3.0; // alp/3 with the three-body alp = 14 + 2 = 16
        for i in 0..nat {
            for j in 0..i {
                for k in 0..j {
                    let c6ij = c6t[i * nat + j];
                    let c6jk = c6t[j * nat + k];
                    let c6ik = c6t[i * nat + k];
                    let c9 = (c6ij * c6jk * c6ik).abs().sqrt();

                    let vij = [
                        ad.xyz[i][0] - ad.xyz[j][0],
                        ad.xyz[i][1] - ad.xyz[j][1],
                        ad.xyz[i][2] - ad.xyz[j][2],
                    ];
                    let vjk = [
                        ad.xyz[j][0] - ad.xyz[k][0],
                        ad.xyz[j][1] - ad.xyz[k][1],
                        ad.xyz[j][2] - ad.xyz[k][2],
                    ];
                    let vik = [
                        ad.xyz[i][0] - ad.xyz[k][0],
                        ad.xyz[i][1] - ad.xyz[k][1],
                        ad.xyz[i][2] - ad.xyz[k][2],
                    ];
                    let a = vij[0] * vij[0] + vij[1] * vij[1] + vij[2] * vij[2];
                    let b = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
                    let c = vik[0] * vik[0] + vik[1] * vik[1] + vik[2] * vik[2];

                    let r2 = a * b * c;
                    let r1 = r2.sqrt();
                    let r3 = r1 * r2;
                    let r5 = r2 * r3;
                    let s = (a + b - c) * (a - b + c) * (-a + b + c);
                    let ang = 0.375 * s / r5 + 1.0 / r3;

                    let r0 = r0pair(i, j) * r0pair(j, k) * r0pair(i, k);
                    let t = r0 / r1;
                    let tb = t.powf(beta);
                    let fdamp = 1.0 / (1.0 + 6.0 * tb);

                    let pre = params.s9 * c9;
                    energy += pre * ang * fdamp;
                    if !want_grad {
                        continue;
                    }

                    let de_dc9 = params.s9 * ang * fdamp;
                    let f_ij = de_dc9 * c9 / (2.0 * c6ij);
                    let f_jk = de_dc9 * c9 / (2.0 * c6jk);
                    let f_ik = de_dc9 * c9 / (2.0 * c6ik);
                    dedcn[i] += f_ij * dc6_i[i * nat + j] + f_ik * dc6_i[i * nat + k];
                    dedcn[j] += f_ij * dc6_i[j * nat + i] + f_jk * dc6_i[j * nat + k];
                    dedcn[k] += f_jk * dc6_i[k * nat + j] + f_ik * dc6_i[k * nat + i];

                    let dfd = 3.0 * beta * tb * fdamp * fdamp; // · (1/x) below
                    let dang = |ds_dx: f64, x: f64| -> f64 {
                        0.375 * (ds_dx - 2.5 * s / x) / r5 - 1.5 / (r3 * x)
                    };
                    let ds_da = (a - b + c) * (-a + b + c) + (a + b - c) * (-a + b + c)
                        - (a + b - c) * (a - b + c);
                    let ds_db = (a - b + c) * (-a + b + c) - (a + b - c) * (-a + b + c)
                        + (a + b - c) * (a - b + c);
                    let ds_dc = -(a - b + c) * (-a + b + c)
                        + (a + b - c) * (-a + b + c)
                        + (a + b - c) * (a - b + c);
                    let de_da = pre * (dang(ds_da, a) * fdamp + ang * dfd / a);
                    let de_db = pre * (dang(ds_db, b) * fdamp + ang * dfd / b);
                    let de_dc = pre * (dang(ds_dc, c) * fdamp + ang * dfd / c);
                    for x in 0..3 {
                        let gij = 2.0 * de_da * vij[x];
                        let gjk = 2.0 * de_db * vjk[x];
                        let gik = 2.0 * de_dc * vik[x];
                        gradient[i][x] += gij + gik;
                        gradient[j][x] += gjk - gij;
                        gradient[k][x] -= gjk + gik;
                    }
                }
            }
        }
    }

    if !want_grad {
        return (energy, None);
    }

    for i in 0..nat {
        for j in 0..i {
            let vec = [
                ad.xyz[i][0] - ad.xyz[j][0],
                ad.xyz[i][1] - ad.xyz[j][1],
                ad.xyz[i][2] - ad.xyz[j][2],
            ];
            let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
            let r = r2.sqrt();
            let rc = ad.rcov[i] + ad.rcov[j];
            let e = (-K1 * (rc / r - 1.0)).exp();
            let f = 1.0 / (1.0 + e);
            let dfdr = -K1 * rc / r2 * f * f * e;
            let scale = (dedcn[i] + dedcn[j]) * dfdr;
            for (k, v) in vec.iter().enumerate() {
                let g = scale * v / r;
                gradient[i][k] += g;
                gradient[j][k] -= g;
            }
        }
    }

    (energy, Some(gradient))
}
