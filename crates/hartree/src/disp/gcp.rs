use crate::core::Molecule;

const THR_R: f64 = 60.0;
const DMP_SCAL: f64 = 4.0;
const DMP_EXP: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq)]
enum NvirtModel {
    R2scan3c,
    BasisMinusOcc(&'static [f32; 18]),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GcpParams {
    sigma: f64,
    eta: f64,
    alpha: f64,
    beta: f64,
    eta_spec: f64,
    damp: bool,
    emiss: &'static [f32; 18],
    nvirt: NvirtModel,
    pub label: &'static str,
}

impl GcpParams {
    pub const R2SCAN_3C: GcpParams = GcpParams {
        sigma: 1.0,
        eta: 1.315,
        alpha: 0.9410,
        beta: 1.4636,
        eta_spec: 1.15,
        damp: true,
        emiss: &EMISS_MTZVPP,
        nvirt: NvirtModel::R2scan3c,
        label: "def2-mTZVPP",
    };

    pub const B3LYP_3C: GcpParams = GcpParams {
        sigma: 0.2424,
        eta: 1.2371,
        alpha: 0.6076,
        beta: 1.4078,
        eta_spec: 1.0,
        damp: false,
        emiss: &EMISS_SVP_DFT,
        nvirt: NvirtModel::BasisMinusOcc(&NBAS_SVP),
        label: "DFT/SV(P)",
    };

    pub const PBEH_3C: GcpParams = GcpParams {
        sigma: 1.0,
        eta: 1.32492,
        alpha: 0.27649,
        beta: 1.95600,
        eta_spec: 1.0,
        damp: true,
        emiss: &EMISS_MSVP,
        nvirt: NvirtModel::BasisMinusOcc(&NBAS_MSVP),
        label: "PBEh-3c (def2-mSVP)",
    };

    pub fn by_keyword(name: &str) -> Option<GcpParams> {
        match name.to_ascii_lowercase().as_str() {
            "r2scan-3c" | "def2-mtzvpp" => Some(Self::R2SCAN_3C),
            "b3lyp-3c" | "dft/sv(p)" | "sv(p)" => Some(Self::B3LYP_3C),
            "pbeh-3c" | "def2-msvp" => Some(Self::PBEH_3C),
            _ => None,
        }
    }
}

const EMISS_MTZVPP: [f32; 18] = [
    0.027, 0.0, // H, He
    0.0, 0.0, 0.2, 0.02, 0.18, 0.08, 0.07, 0.065, // Li–Ne
    0.0, 0.0, 0.0, 0.2, 0.6, 0.6, 0.6, 0.3, // Na–Ar
];

const EMISS_SVP_DFT: [f32; 18] = [
    0.009037, 0.008045, // H (from SV), He
    0.113583, 0.028371, 0.049369, 0.055376, 0.072785, 0.100310, 0.133273, 0.173600, // Li–Ne
    0.181140, 0.125558, 0.167188, 0.149843, 0.145396, 0.164308, 0.182990, 0.205668, // Na–Ar
];

const EMISS_MSVP: [f32; 18] = [
    0.0, 0.0, // H, He (zero by construction)
    0.10775, 0.02, 0.02685, 0.02174, 0.02725, 0.03993, 0.03, 0.0, // Li–Ne
    0.15329, 0.1623, 0.1027, 0.07314, 0.05622, 0.06133, 0.06504, 0.0, // Na–Ar
];

const NBAS_MSVP: [f32; 18] = [
    2.0, 2.0, // H, He
    9.0, 9.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, // Li–Ne
    15.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, // Na–Ar
];

const NBAS_SVP: [f32; 18] = [
    2.0, 5.0, // H (SV, no p), He
    9.0, 9.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, // Li–Ne
    15.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, // Na–Ar
];

const ZS: [f32; 18] = [
    1.2000, 1.6469, 0.6534, 1.0365, 1.3990, 1.7210, 2.0348, 2.2399, 2.5644, 2.8812, 0.8675, 1.1935,
    1.5143, 1.7580, 1.9860, 2.1362, 2.3617, 2.5796,
];

const ZP: [f32; 18] = [
    0.0000, 0.0000, 0.5305, 0.8994, 1.2685, 1.6105, 1.9398, 2.0477, 2.4022, 2.7421, 0.6148, 0.8809,
    1.1660, 1.4337, 1.6755, 1.7721, 2.0176, 2.2501,
];

fn slater_exponent(z: usize, p: &GcpParams) -> f64 {
    let i = z - 1;
    let base = if z <= 2 {
        ZS[i] as f64
    } else {
        (ZS[i] as f64 + ZP[i] as f64) / 2.0
    };
    let spec = if z >= 11 { base * p.eta_spec } else { base };
    spec * p.eta
}

fn srow(z: usize) -> u32 {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        _ => 3,
    }
}

fn nvirt(z: usize, model: NvirtModel) -> f64 {
    match model {
        NvirtModel::R2scan3c => match z {
            6 => 3.0,
            7 | 8 => 0.5,
            _ => 1.0,
        },
        NvirtModel::BasisMinusOcc(nbas) => nbas[z - 1] as f64 - 0.5 * z as f64,
    }
}

fn aux_a(x: f64) -> [f64; 8] {
    let e = (-x).exp();
    let mut a = [0.0; 8];
    a[0] = e / x;
    for n in 1..8 {
        a[n] = (e + n as f64 * a[n - 1]) / x;
    }
    a
}

fn aux_b(x: f64) -> [f64; 8] {
    let ep = x.exp();
    let em = (-x).exp();
    let mut b = [0.0; 8];
    b[0] = (ep - em) / x;
    for n in 1..8 {
        let sign = if n % 2 == 0 { ep } else { -ep };
        b[n] = (sign - em + n as f64 * b[n - 1]) / x;
    }
    b
}

const FACT: [f64; 13] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
];

fn bint(x: f64, k: u32) -> (f64, f64) {
    let k = k as i32;
    if x.abs() < 1e-6 {
        let v = if k % 2 == 0 {
            2.0 / (k as f64 + 1.0)
        } else {
            0.0
        };
        let d = if k % 2 == 1 {
            -2.0 / (k as f64 + 2.0)
        } else {
            0.0
        };
        return (v, d);
    }
    let mut v = 0.0;
    let mut d = 0.0;
    for i in 0..=12 {
        let xx = 1.0 - (-1.0f64).powi(k + i + 1);
        let yy = FACT[i as usize] * (k + i + 1) as f64;
        let c = xx / yy;
        v += c * (-x).powi(i);
        if i >= 1 {
            d -= c * i as f64 * (-x).powi(i - 1);
        }
    }
    (v, d)
}

fn slater_overlap(r: f64, na: u32, nb: u32, za_in: f64, zb_in: f64) -> (f64, f64) {
    let (za, zb) = if na <= nb {
        (za_in, zb_in)
    } else {
        (zb_in, za_in)
    };
    let fa = za + zb;
    let fb = zb - za;
    let ax = fa * r * 0.5;
    let bx = fb * r * 0.5;
    let same = za_in == zb_in || (za_in - zb_in).abs() < 0.1;

    let a = aux_a(ax);
    let mut b = [0.0; 8];
    let mut db = [0.0; 8];
    if same {
        for n in 0..8 {
            let (v, d) = bint(bx, n as u32);
            b[n] = v;
            db[n] = d;
        }
    } else {
        b = aux_b(bx);
        for n in 0..7 {
            db[n] = -b[n + 1];
        }
    }

    let (lo, hi) = if na <= nb { (na, nb) } else { (nb, na) };
    let (norm, p, g, g_ax, g_bx): (f64, f64, f64, f64, f64) = match (lo, hi) {
        (1, 1) => {
            let norm = 0.25 * ((za * zb * r * r).powi(3)).sqrt();
            (
                norm,
                3.0,
                a[2] * b[0] - b[2] * a[0],
                -a[3] * b[0] + b[2] * a[1],
                a[2] * db[0] - db[2] * a[0],
            )
        }
        (1, 2) => {
            let norm = (1.0f64 / 3.0).sqrt() * (za.powi(3) * zb.powi(5)).sqrt() * r.powi(4) * 0.125;
            (
                norm,
                4.0,
                a[3] * b[0] - b[3] * a[0] + a[2] * b[1] - b[2] * a[1],
                -a[4] * b[0] + b[3] * a[1] - a[3] * b[1] + b[2] * a[2],
                a[3] * db[0] - db[3] * a[0] + a[2] * db[1] - db[2] * a[1],
            )
        }
        (2, 2) => {
            let norm = ((za * zb).powi(5)).sqrt() * r.powi(5) * 0.0625 / 3.0;
            (
                norm,
                5.0,
                a[4] * b[0] + b[4] * a[0] - 2.0 * a[2] * b[2],
                -a[5] * b[0] - b[4] * a[1] + 2.0 * a[3] * b[2],
                a[4] * db[0] + db[4] * a[0] - 2.0 * a[2] * db[2],
            )
        }
        (1, 3) => {
            let norm =
                (za.powi(3) * zb.powi(7) / 7.5).sqrt() * r.powi(5) * 0.0625 / (3.0f64).sqrt();
            (
                norm,
                5.0,
                a[4] * b[0] - b[4] * a[0] + 2.0 * (a[3] * b[1] - b[3] * a[1]),
                -a[5] * b[0] + b[4] * a[1] + 2.0 * (-a[4] * b[1] + b[3] * a[2]),
                a[4] * db[0] - db[4] * a[0] + 2.0 * (a[3] * db[1] - db[3] * a[1]),
            )
        }
        (2, 3) => {
            let norm = (za.powi(5) * zb.powi(7) / 7.5).sqrt() * r.powi(6) * 0.03125 / 3.0;
            (
                norm,
                6.0,
                a[5] * b[0] + a[4] * b[1] - 2.0 * (a[3] * b[2] + a[2] * b[3])
                    + a[1] * b[4]
                    + a[0] * b[5],
                -a[6] * b[0] - a[5] * b[1] + 2.0 * (a[4] * b[2] + a[3] * b[3])
                    - a[2] * b[4]
                    - a[1] * b[5],
                a[5] * db[0] + a[4] * db[1] - 2.0 * (a[3] * db[2] + a[2] * db[3])
                    + a[1] * db[4]
                    + a[0] * db[5],
            )
        }
        (3, 3) => {
            let norm = ((za * zb * r * r).powi(7)).sqrt() / 1440.0;
            (
                norm,
                7.0,
                a[6] * b[0] - 3.0 * (a[4] * b[2] - a[2] * b[4]) - a[0] * b[6],
                -a[7] * b[0] + 3.0 * (a[5] * b[2] - a[3] * b[4]) + a[1] * b[6],
                a[6] * db[0] - 3.0 * (a[4] * db[2] - a[2] * db[4]) - a[0] * db[6],
            )
        }
        _ => unreachable!("principal quantum numbers are 1..=3"),
    };

    let s = norm * g;
    let ds = (p / r) * s + norm * (g_ax * fa * 0.5 + g_bx * fb * 0.5);
    (s, ds)
}

fn atom_numbers(mol: &Molecule) -> Vec<usize> {
    mol.atoms
        .iter()
        .map(|a| {
            let z = a.element.z() as usize;
            assert!(
                (1..=18).contains(&z),
                "gCP supports H-Ar only (got Z = {z})"
            );
            z
        })
        .collect()
}

pub fn gcp_energy(mol: &Molecule, params: &GcpParams) -> f64 {
    if let Some((real, _)) = crate::disp::without_ghosts(mol) {
        return gcp_energy(&real, params);
    }
    gcp_impl(mol, params, false).0
}

pub fn gcp_energy_gradient(mol: &Molecule, params: &GcpParams) -> (f64, Vec<[f64; 3]>) {
    if let Some((real, map)) = crate::disp::without_ghosts(mol) {
        let (e, g) = gcp_energy_gradient(&real, params);
        return (e, crate::disp::scatter_gradient(g, &map, mol.len()));
    }
    gcp_impl(mol, params, true)
}

pub fn gcp_r2scan3c_energy(mol: &Molecule) -> f64 {
    gcp_energy(mol, &GcpParams::R2SCAN_3C)
}

pub fn gcp_r2scan3c_energy_gradient(mol: &Molecule) -> (f64, Vec<[f64; 3]>) {
    gcp_energy_gradient(mol, &GcpParams::R2SCAN_3C)
}

fn gcp_impl(mol: &Molecule, p: &GcpParams, grad: bool) -> (f64, Vec<[f64; 3]>) {
    let z = atom_numbers(mol);
    let xyz: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let nat = z.len();
    let zeta: Vec<f64> = z.iter().map(|&zi| slater_exponent(zi, p)).collect();
    let thr_e = f64::EPSILON;

    let mut energy = 0.0;
    let mut gradient = vec![[0.0; 3]; nat];

    for i in 0..nat {
        let emiss_i = p.emiss[z[i] - 1] as f64;
        for j in 0..nat {
            if i == j {
                continue;
            }
            let vb = nvirt(z[j], p.nvirt);
            if vb < 0.5 {
                continue;
            }
            let vec = [
                xyz[i][0] - xyz[j][0],
                xyz[i][1] - xyz[j][1],
                xyz[i][2] - xyz[j][2],
            ];
            let r = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
            if r > THR_R {
                continue;
            }
            let (sab, dsab) = slater_overlap(r, srow(z[i]), srow(z[j]), zeta[i], zeta[j]);
            if sab.abs().sqrt() < thr_e {
                continue;
            }
            let e_num = (-p.alpha * r.powf(p.beta)).exp();
            let den = (vb * sab).sqrt();
            let e_old = e_num / den;
            if e_old.abs() < thr_e {
                continue;
            }
            let (damp, ddamp) = if p.damp {
                let r0 = crate::disp::data::r0ab_bohr(z[i], z[j]);
                let rscal = r / r0;
                let rse = rscal.powf(DMP_EXP);
                let dval = 1.0 - 1.0 / (1.0 + DMP_SCAL * rse);
                let dder = DMP_SCAL * DMP_EXP * rscal.powf(DMP_EXP - 1.0)
                    / r0
                    / ((DMP_SCAL * rse + 1.0) * (DMP_SCAL * rse + 1.0));
                (dval, dder)
            } else {
                (1.0, 0.0)
            };
            energy += emiss_i * e_old * damp;

            if grad {
                let de_old =
                    e_old * (-p.alpha * p.beta * r.powf(p.beta - 1.0) - dsab / (2.0 * sab));
                let de_dr = emiss_i * (de_old * damp + e_old * ddamp);
                for (k, v) in vec.iter().enumerate() {
                    let g = de_dr * v / r;
                    gradient[i][k] += g;
                    gradient[j][k] -= g;
                }
            }
        }
    }
    energy *= p.sigma;
    if grad {
        for g in &mut gradient {
            for c in g.iter_mut() {
                *c *= p.sigma;
            }
        }
    }
    (energy, gradient)
}
