use crate::core::Molecule;
use libm::erf;

use crate::disp::d4data as dat;

const ALP: f64 = 16.0;

const SQRT_PI: f64 = 1.7724538509055159;
const SQRT_2_OVER_PI: f64 = 0.7978845608028654;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D4Params {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub a1: f64,
    pub a2: f64,
}

impl D4Params {
    pub const R2SCAN_3C: D4Params = D4Params {
        s6: 1.0,
        s8: 0.0,
        s9: 2.0,
        a1: 0.42,
        a2: 5.65,
    };

    pub fn for_method(name: &str) -> Option<D4Params> {
        let lower = name.to_ascii_lowercase();
        let key = if lower == "b3lyp5" {
            "b3lyp"
        } else {
            lower.as_str()
        };
        dat::BJ_PARAMS
            .iter()
            .find(|(m, _, _, _)| *m == key)
            .map(|&(_, s8, a1, a2)| D4Params {
                s6: 1.0,
                s8,
                s9: 1.0,
                a1,
                a2,
            })
            .or_else(|| {
                dat::BJ_PARAMS_DH
                    .iter()
                    .find(|(m, _, _, _, _)| *m == key)
                    .map(|&(_, s6, s8, a1, a2)| D4Params {
                        s6,
                        s8,
                        s9: 1.0,
                        a1,
                        a2,
                    })
            })
    }
}

struct AtomData {
    zi: Vec<usize>,
    xyz: Vec<[f64; 3]>,
}

fn atom_data(mol: &Molecule) -> AtomData {
    let zi: Vec<usize> = mol
        .atoms
        .iter()
        .map(|a| {
            let z = a.element.z() as usize;
            assert!(
                (1..=dat::MAX_Z).contains(&z),
                "D4 supports H-Ar only (got Z = {z})"
            );
            z - 1
        })
        .collect();
    AtomData {
        zi,
        xyz: mol.atoms.iter().map(|a| a.position).collect(),
    }
}

fn dist_vec(a: [f64; 3], b: [f64; 3]) -> ([f64; 3], f64) {
    let v = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (v, (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
}

fn erf_count(r: f64, r0: f64) -> (f64, f64) {
    let f = 0.5 * (1.0 + erf(-dat::KCN * (r / r0 - 1.0)));
    let x = dat::KCN * (r - r0) / r0;
    let df = -dat::KCN / SQRT_PI / r0 * (-x * x).exp();
    (f, df)
}

pub fn coordination_numbers_d4(mol: &Molecule) -> Vec<f64> {
    let ad = atom_data(mol);
    cn_d4(&ad)
}

fn cn_d4(ad: &AtomData) -> Vec<f64> {
    let nat = ad.zi.len();
    let mut cn = vec![0.0; nat];
    for i in 0..nat {
        for j in 0..i {
            let (_, r) = dist_vec(ad.xyz[i], ad.xyz[j]);
            let f = en_weight(ad.zi[i], ad.zi[j])
                * erf_count(r, dat::RCOV_D3[ad.zi[i]] + dat::RCOV_D3[ad.zi[j]]).0;
            cn[i] += f;
            cn[j] += f;
        }
    }
    cn
}

fn en_weight(zi: usize, zj: usize) -> f64 {
    let endiff = (dat::EN_PAULING[zi] - dat::EN_PAULING[zj]).abs();
    dat::K4 * (-(endiff + dat::K5) * (endiff + dat::K5) / dat::K6).exp()
}

fn cn_eeq(ad: &AtomData) -> (Vec<f64>, Vec<f64>) {
    let nat = ad.zi.len();
    let mut cn = vec![0.0; nat];
    for i in 0..nat {
        for j in 0..i {
            let (_, r) = dist_vec(ad.xyz[i], ad.xyz[j]);
            let f = erf_count(r, dat::RCOV_D3[ad.zi[i]] + dat::RCOV_D3[ad.zi[j]]).0;
            cn[i] += f;
            cn[j] += f;
        }
    }
    let m = dat::EEQ_CN_MAX;
    let cut: Vec<f64> = cn
        .iter()
        .map(|&c| (1.0 + m.exp()).ln() - (1.0 + (m - c).exp()).ln())
        .collect();
    let dcut: Vec<f64> = cn.iter().map(|&c| 1.0 / (1.0 + (c - m).exp())).collect();
    (cut, dcut)
}

fn lu_solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if a[i * n + k].abs() > a[p * n + k].abs() {
                p = i;
            }
        }
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
            b.swap(k, p);
        }
        let piv = a[k * n + k];
        assert!(piv.abs() > 1e-300, "EEQ system is singular");
        for i in k + 1..n {
            let f = a[i * n + k] / piv;
            if f == 0.0 {
                continue;
            }
            for c in k + 1..n {
                a[i * n + c] -= f * a[k * n + c];
            }
            b[i] -= f * b[k];
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in i + 1..n {
            s -= a[i * n + c] * b[c];
        }
        b[i] = s / a[i * n + i];
    }
    b
}

struct Eeq {
    a: Vec<f64>,
    x: Vec<f64>,
    dcut: Vec<f64>,
    dbdcn: Vec<f64>,
}

fn eeq_solve(ad: &AtomData, charge: f64) -> Eeq {
    let nat = ad.zi.len();
    let dim = nat + 1;
    let (cncut, dcut) = cn_eeq(ad);

    let mut a = vec![0.0; dim * dim];
    let mut b = vec![0.0; dim];
    for i in 0..nat {
        let radi = dat::EEQ_RAD[ad.zi[i]];
        a[i * dim + i] = dat::EEQ_ETA[ad.zi[i]] + SQRT_2_OVER_PI / radi;
        for j in 0..i {
            let radj = dat::EEQ_RAD[ad.zi[j]];
            let gamma = 1.0 / (radi * radi + radj * radj).sqrt();
            let (_, r) = dist_vec(ad.xyz[i], ad.xyz[j]);
            let aij = erf(gamma * r) / r;
            a[i * dim + j] = aij;
            a[j * dim + i] = aij;
        }
        a[i * dim + nat] = 1.0;
        a[nat * dim + i] = 1.0;
        b[i] = -dat::EEQ_CHI[ad.zi[i]] + dat::EEQ_KCN[ad.zi[i]] * cncut[i].sqrt();
    }
    b[nat] = charge;

    let x = lu_solve(a.clone(), b);
    let dbdcn = (0..nat)
        .map(|i| {
            if cncut[i] > 1e-300 {
                dat::EEQ_KCN[ad.zi[i]] * 0.5 / cncut[i].sqrt()
            } else {
                0.0
            }
        })
        .collect();
    Eeq { a, x, dcut, dbdcn }
}

pub fn eeq_charges(mol: &Molecule) -> Vec<f64> {
    let ad = atom_data(mol);
    let mut x = eeq_solve(&ad, mol.charge as f64).x;
    x.truncate(ad.zi.len());
    x
}

fn zeta(gam: f64, qref: f64, qmod: f64) -> (f64, f64) {
    if qmod > 0.0 {
        let scale = (gam * (1.0 - qref / qmod)).exp();
        let z = (dat::GA * (1.0 - scale)).exp();
        let dz = -dat::GA * gam * scale * z * qref / (qmod * qmod);
        (z, dz)
    } else {
        (dat::GA.exp(), 0.0)
    }
}

struct Weights {
    w: [f64; dat::MAX_REF],
    dwdcn: [f64; dat::MAX_REF],
    dwdq: [f64; dat::MAX_REF],
}

fn weights(zi: usize, cn: f64, q: f64) -> Weights {
    let nref = dat::N_REF[zi];
    let refs = &dat::REF_CN[zi];
    let mut expw = [0.0; dat::MAX_REF];
    let mut dexpw = [0.0; dat::MAX_REF];
    let mut norm = 0.0;
    let mut dnorm = 0.0;
    for iref in 0..nref {
        let d = cn - refs[iref];
        for k in 1..=dat::REF_IGW[zi][iref] {
            let g = (-(k as f64) * dat::WF * d * d).exp();
            expw[iref] += g;
            dexpw[iref] += -2.0 * (k as f64) * dat::WF * d * g;
        }
        norm += expw[iref];
        dnorm += dexpw[iref];
    }
    let cn_max = refs[..nref]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut out = Weights {
        w: [0.0; dat::MAX_REF],
        dwdcn: [0.0; dat::MAX_REF],
        dwdq: [0.0; dat::MAX_REF],
    };
    let zeffi = dat::ZEFF[zi];
    let gam = dat::GAM[zi] * dat::GC;
    for iref in 0..nref {
        let mut gw = expw[iref] / norm;
        let mut dgw = (dexpw[iref] - expw[iref] * dnorm / norm) / norm;
        if !gw.is_finite() {
            gw = if refs[iref] == cn_max { 1.0 } else { 0.0 };
            dgw = 0.0;
        }
        if !dgw.is_finite() {
            dgw = 0.0;
        }
        let (z, dz) = zeta(gam, dat::REF_Q[zi][iref] + zeffi, q + zeffi);
        out.w[iref] = z * gw;
        out.dwdcn[iref] = z * dgw;
        out.dwdq[iref] = dz * gw;
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
    dat::C6AB[ic * dat::MAX_REF * dat::MAX_REF + ir * dat::MAX_REF + jr]
}

struct PairC6 {
    c6: f64,
    dcni: f64,
    dcnj: f64,
    dqi: f64,
    dqj: f64,
}

fn pair_c6(zi: usize, zj: usize, wi: &Weights, wj: &Weights) -> PairC6 {
    let mut p = PairC6 {
        c6: 0.0,
        dcni: 0.0,
        dcnj: 0.0,
        dqi: 0.0,
        dqj: 0.0,
    };
    for iref in 0..dat::N_REF[zi] {
        for jref in 0..dat::N_REF[zj] {
            let rc6 = ref_c6(zi, zj, iref, jref);
            p.c6 += wi.w[iref] * wj.w[jref] * rc6;
            p.dcni += wi.dwdcn[iref] * wj.w[jref] * rc6;
            p.dcnj += wi.w[iref] * wj.dwdcn[jref] * rc6;
            p.dqi += wi.dwdq[iref] * wj.w[jref] * rc6;
            p.dqj += wi.w[iref] * wj.dwdq[jref] * rc6;
        }
    }
    p
}

pub fn d4_energy(mol: &Molecule, params: &D4Params) -> f64 {
    if let Some((real, _)) = crate::disp::without_ghosts(mol) {
        return d4_energy(&real, params);
    }
    d4_impl(mol, params, false).0
}

pub fn d4_energy_gradient(mol: &Molecule, params: &D4Params) -> (f64, Vec<[f64; 3]>) {
    if let Some((real, map)) = crate::disp::without_ghosts(mol) {
        let (e, g) = d4_energy_gradient(&real, params);
        return (e, crate::disp::scatter_gradient(g, &map, mol.len()));
    }
    let (e, g) = d4_impl(mol, params, true);
    (e, g.expect("gradient requested"))
}

fn d4_impl(mol: &Molecule, p: &D4Params, want_grad: bool) -> (f64, Option<Vec<[f64; 3]>>) {
    let ad = atom_data(mol);
    let nat = ad.zi.len();

    let cn = cn_d4(&ad);
    let eeq = eeq_solve(&ad, mol.charge as f64);
    let q = &eeq.x[..nat];

    let w: Vec<Weights> = (0..nat).map(|i| weights(ad.zi[i], cn[i], q[i])).collect();
    let w0: Vec<Weights> = (0..nat).map(|i| weights(ad.zi[i], cn[i], 0.0)).collect();

    let mut energy = 0.0;
    let mut gradient = vec![[0.0; 3]; nat];
    let mut dedcn = vec![0.0; nat];
    let mut dedq = vec![0.0; nat];

    for i in 0..nat {
        for j in 0..i {
            let pc = pair_c6(ad.zi[i], ad.zi[j], &w[i], &w[j]);
            let rrij = 3.0 * dat::SQRT_Z_R4R2[ad.zi[i]] * dat::SQRT_Z_R4R2[ad.zi[j]];
            let r0 = p.a1 * rrij.sqrt() + p.a2;
            let (vec, r) = dist_vec(ad.xyz[i], ad.xyz[j]);
            let r2 = r * r;
            let r6 = r2 * r2 * r2;
            let r8 = r6 * r2;
            let r0_2 = r0 * r0;
            let r0_6 = r0_2 * r0_2 * r0_2;
            let r0_8 = r0_6 * r0_2;
            let t6 = 1.0 / (r6 + r0_6);
            let t8 = 1.0 / (r8 + r0_8);
            let damp = p.s6 * t6 + p.s8 * rrij * t8;
            energy -= pc.c6 * damp;
            if !want_grad {
                continue;
            }
            dedcn[i] -= pc.dcni * damp;
            dedcn[j] -= pc.dcnj * damp;
            dedq[i] -= pc.dqi * damp;
            dedq[j] -= pc.dqj * damp;
            let ddampdr =
                -(6.0 * p.s6 * r2 * r2 * r * t6 * t6 + 8.0 * p.s8 * rrij * r6 * r * t8 * t8);
            let de_dr = -pc.c6 * ddampdr;
            for (k, v) in vec.iter().enumerate() {
                let g = de_dr * v / r;
                gradient[i][k] += g;
                gradient[j][k] -= g;
            }
        }
    }

    if p.s9 != 0.0 && nat >= 3 {
        let mut c60 = vec![0.0; nat * nat];
        let mut dc60_i = vec![0.0; nat * nat]; // d c6_ij / d cn_i
        for i in 0..nat {
            for j in 0..i {
                let pc = pair_c6(ad.zi[i], ad.zi[j], &w0[i], &w0[j]);
                c60[i * nat + j] = pc.c6;
                c60[j * nat + i] = pc.c6;
                dc60_i[i * nat + j] = pc.dcni; // wrt cn_i
                dc60_i[j * nat + i] = pc.dcnj; // wrt cn_j
            }
        }
        let r0pair = |i: usize, j: usize| -> f64 {
            let rr = 3.0 * dat::SQRT_Z_R4R2[ad.zi[i]] * dat::SQRT_Z_R4R2[ad.zi[j]];
            p.a1 * rr.sqrt() + p.a2
        };
        let beta = ALP / 3.0;
        for i in 0..nat {
            for j in 0..i {
                for k in 0..j {
                    let c6ij = c60[i * nat + j];
                    let c6jk = c60[j * nat + k];
                    let c6ik = c60[i * nat + k];
                    let c9 = (c6ij * c6jk * c6ik).abs().sqrt();

                    let (vij, rij) = dist_vec(ad.xyz[i], ad.xyz[j]);
                    let (vjk, rjk) = dist_vec(ad.xyz[j], ad.xyz[k]);
                    let (vik, rik) = dist_vec(ad.xyz[i], ad.xyz[k]);
                    let (a, b, c) = (rij * rij, rjk * rjk, rik * rik);

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

                    let pre = p.s9 * c9;
                    energy += pre * ang * fdamp;
                    if !want_grad {
                        continue;
                    }

                    let de_dc9 = p.s9 * ang * fdamp;
                    let f_ij = de_dc9 * c9 / (2.0 * c6ij);
                    let f_jk = de_dc9 * c9 / (2.0 * c6jk);
                    let f_ik = de_dc9 * c9 / (2.0 * c6ik);
                    dedcn[i] += f_ij * dc60_i[i * nat + j] + f_ik * dc60_i[i * nat + k];
                    dedcn[j] += f_ij * dc60_i[j * nat + i] + f_jk * dc60_i[j * nat + k];
                    dedcn[k] += f_jk * dc60_i[k * nat + j] + f_ik * dc60_i[k * nat + i];

                    let dfd = 3.0 * beta * tb * fdamp * fdamp; // * (1/x) below
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
            let (vec, r) = dist_vec(ad.xyz[i], ad.xyz[j]);
            let (_, df) = erf_count(r, dat::RCOV_D3[ad.zi[i]] + dat::RCOV_D3[ad.zi[j]]);
            let scale = (dedcn[i] + dedcn[j]) * en_weight(ad.zi[i], ad.zi[j]) * df;
            for (k, v) in vec.iter().enumerate() {
                let g = scale * v / r;
                gradient[i][k] += g;
                gradient[j][k] -= g;
            }
        }
    }

    let dim = nat + 1;
    let mut g_rhs = vec![0.0; dim];
    g_rhs[..nat].copy_from_slice(&dedq);
    let u = lu_solve(eeq.a.clone(), g_rhs);

    for i in 0..nat {
        for j in 0..i {
            let (vec, r) = dist_vec(ad.xyz[i], ad.xyz[j]);
            let radi = dat::EEQ_RAD[ad.zi[i]];
            let radj = dat::EEQ_RAD[ad.zi[j]];
            let gamma = 1.0 / (radi * radi + radj * radj).sqrt();
            let gr = gamma * r;
            let dadr = (2.0 * gamma / SQRT_PI * (-gr * gr).exp() - erf(gr) / r) / r;
            let mut scale = -(u[i] * eeq.x[j] + u[j] * eeq.x[i]) * dadr;
            let (_, df) = erf_count(r, dat::RCOV_D3[ad.zi[i]] + dat::RCOV_D3[ad.zi[j]]);
            scale += (u[i] * eeq.dbdcn[i] * eeq.dcut[i] + u[j] * eeq.dbdcn[j] * eeq.dcut[j]) * df;
            for (k, v) in vec.iter().enumerate() {
                let g = scale * v / r;
                gradient[i][k] += g;
                gradient[j][k] -= g;
            }
        }
    }

    (energy, Some(gradient))
}
