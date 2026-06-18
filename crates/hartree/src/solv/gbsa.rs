use crate::dft::grid::lebedev::LebedevGrid;
use crate::solv::SolvError;
use crate::solv::surface::BOHR;

mod params {
    include!("gbsa_params.rs");
}
pub use params::GBSA_PARAMS;

const AATOAU: f64 = 1.0 / BOHR;
const KCAL_TO_AU: f64 = 1.0 / 627.509_474_277_3;
const ALPB_ALPHA: f64 = 0.571_412;
const ZETA_P16_O16: f64 = 1.028 / 16.0;
const OBC_ALP: f64 = 1.0;
const OBC_BET: f64 = 0.8;
const OBC_GAM: f64 = 4.85;
const SURFACE_TENSION_UNIT: f64 = 1.0e-5;
const TOLSESP: f64 = 1.0e-6;
pub const DEFAULT_GBSA_GRID: usize = 194;

const VDW_RAD_D3_AA: [f64; 94] = [
    1.09155, 0.86735, 1.74780, 1.54910, 1.60800, 1.45515, 1.31125, 1.24085, 1.14980, 1.06870,
    1.85410, 1.74195, 2.00530, 1.89585, 1.75085, 1.65535, 1.55230, 1.45740, 2.12055, 2.05175,
    1.94515, 1.88210, 1.86055, 1.72070, 1.77310, 1.72105, 1.71635, 1.67310, 1.65040, 1.61545,
    1.97895, 1.93095, 1.83125, 1.76340, 1.68310, 1.60480, 2.30880, 2.23820, 2.10980, 2.02985,
    1.92980, 1.87715, 1.78450, 1.73115, 1.69875, 1.67625, 1.66540, 1.73100, 2.13115, 2.09370,
    2.00750, 1.94505, 1.86900, 1.79445, 2.52835, 2.59070, 2.31305, 2.31005, 2.28510, 2.26355,
    2.24480, 2.22575, 2.21170, 2.06215, 2.12135, 2.07705, 2.13970, 2.12250, 2.11040, 2.09930,
    2.00650, 2.12250, 2.04900, 1.99275, 1.94775, 1.87450, 1.72280, 1.67625, 1.62820, 1.67995,
    2.15635, 2.13820, 2.05875, 2.00270, 1.93220, 1.86080, 2.53980, 2.46470, 2.35215, 2.21260,
    2.22970, 2.19785, 2.17695, 2.21705,
];

#[derive(Debug, Clone)]
pub struct GbsaParams {
    pub name: &'static str,
    pub alpb: bool,
    pub epsv: f64,
    pub smass: f64,
    pub rhos: f64,
    pub c1: f64,
    pub rprobe: f64,
    pub gshift: f64,
    pub soset: f64,
    pub alpha: f64,
    pub gamscale: [f64; 94],
    pub sx: [f64; 94],
    pub tmp: [f64; 94],
}

pub fn alpb_solvent(name: &str) -> Option<&'static GbsaParams> {
    let lower = name.to_ascii_lowercase();
    GBSA_PARAMS.iter().find(|p| p.alpb && p.name == lower)
}

pub fn gbsa_solvent(name: &str) -> Option<&'static GbsaParams> {
    let lower = name.to_ascii_lowercase();
    GBSA_PARAMS.iter().find(|p| !p.alpb && p.name == lower)
}

pub fn alpb_solvent_names() -> Vec<&'static str> {
    GBSA_PARAMS
        .iter()
        .filter(|p| p.alpb)
        .map(|p| p.name)
        .collect()
}

pub fn gbsa_solvent_names() -> Vec<&'static str> {
    GBSA_PARAMS
        .iter()
        .filter(|p| !p.alpb)
        .map(|p| p.name)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GbsaBreakdown {
    pub g_born: f64,
    pub g_hb: f64,
    pub g_sasa: f64,
    pub g_shift: f64,
    pub g_solv: f64,
}

pub fn gbsa_energy(
    params: &GbsaParams,
    zs: &[usize],
    coords: &[[f64; 3]],
    qat: &[f64],
    ng: usize,
) -> Result<GbsaBreakdown, SolvError> {
    let n = zs.len();
    assert_eq!(coords.len(), n);
    assert_eq!(qat.len(), n);

    let mut vdwr = vec![0.0; n];
    let mut rho = vec![0.0; n];
    let mut svdw = vec![0.0; n];
    let mut vdwsa = vec![0.0; n];
    let born_offset = params.soset * 0.1 * AATOAU;
    let probe = params.rprobe * AATOAU;
    for i in 0..n {
        let z = zs[i];
        let r = *VDW_RAD_D3_AA
            .get(z.wrapping_sub(1))
            .ok_or(SolvError::NoRadius(z))?
            * AATOAU;
        vdwr[i] = r;
        rho[i] = r * params.sx[z - 1];
        svdw[i] = r - born_offset;
        vdwsa[i] = r + probe;
    }

    let dist = |i: usize, j: usize| -> f64 {
        let (p, q) = (coords[i], coords[j]);
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };

    let brad = born_radii(n, &dist, &vdwr, &rho, &svdw, params.c1);

    let sasa = compute_sasa(n, coords, &vdwsa, ng)?;
    let gamsasa: Vec<f64> = zs
        .iter()
        .map(|&z| params.gamscale[z - 1] * 4.0 * std::f64::consts::PI * SURFACE_TENSION_UNIT)
        .collect();
    let g_sasa: f64 = sasa.iter().zip(&gamsasa).map(|(&s, &g)| s * g).sum();

    let alpbet = if params.alpb {
        ALPB_ALPHA / params.epsv
    } else {
        0.0
    };
    let keps = (1.0 / params.epsv - 1.0) / (1.0 + alpbet);

    let has_hb = params.tmp.iter().any(|&t| t.abs() > 1.0e-3);
    let mut hbw = vec![0.0; n];
    if has_hb {
        for i in 0..n {
            let z = zs[i];
            let hbmag = -(params.tmp[z - 1].powi(2)) * KCAL_TO_AU;
            hbw[i] = hbmag * sasa[i] / (vdwsa[i] * vdwsa[i]);
        }
    }

    let mut amat = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            let r = dist(i, j);
            let v = if params.alpb {
                let ab = (brad[i] * brad[j]).sqrt();
                let mut arg = ab / (ab + ZETA_P16_O16 * r);
                arg = arg * arg; // ²
                arg = arg * arg; // ⁴
                arg = arg * arg; // ⁸
                arg = arg * arg; // ¹⁶
                keps / (r + ab * arg)
            } else {
                let aa = brad[i] * brad[j];
                let dd = 0.25 * r * r / aa;
                let fgb2 = r * r + aa * (-dd).exp();
                keps / fgb2.sqrt()
            };
            amat[i * n + j] += v;
            amat[j * n + i] += v;
        }
        amat[i * n + i] += keps / brad[i];
    }
    if has_hb {
        for i in 0..n {
            amat[i * n + i] += 2.0 * hbw[i];
        }
    }
    if alpbet > 0.0 {
        let adet = shape_descriptor(n, coords, &vdwr);
        let shift = keps * alpbet / adet;
        for x in &mut amat {
            *x += shift;
        }
    }

    let mut quad = 0.0;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            row += amat[i * n + j] * qat[j];
        }
        quad += qat[i] * row;
    }
    quad *= 0.5;
    let g_hb: f64 = if has_hb {
        (0..n).map(|i| hbw[i] * qat[i] * qat[i]).sum()
    } else {
        0.0
    };
    let g_born = quad - g_hb;
    let g_shift = params.gshift * KCAL_TO_AU;
    let g_solv = g_born + g_hb + g_sasa + g_shift;

    Ok(GbsaBreakdown {
        g_born,
        g_hb,
        g_sasa,
        g_shift,
        g_solv,
    })
}

fn born_radii(
    n: usize,
    dist: &impl Fn(usize, usize) -> f64,
    vdwr: &[f64],
    rho: &[f64],
    svdw: &[f64],
    c1: f64,
) -> Vec<f64> {
    let contrib = |r: f64, rho_j: f64, rvdw_i: f64| -> f64 {
        if r >= rvdw_i + rho_j {
            let ap = r + rho_j;
            let am = r - rho_j;
            let ab = ap * am;
            rho_j / ab + 0.5 * (am / ap).ln() / r
        } else if r + rho_j > rvdw_i {
            let r12 = 0.5 / r;
            let ap = r + rho_j;
            let am = r - rho_j;
            let rh1 = 1.0 / rvdw_i;
            let rhr1 = 1.0 / ap;
            let aprh1 = ap * rh1;
            rh1 - rhr1 + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - aprh1.ln())
        } else {
            0.0
        }
    };

    let mut psi = vec![0.0; n];
    for i in 0..n {
        for j in 0..i {
            let r = dist(i, j);
            psi[i] += contrib(r, rho[j], vdwr[i]);
            psi[j] += contrib(r, rho[i], vdwr[j]);
        }
    }

    let mut brad = vec![0.0; n];
    for i in 0..n {
        let svdwi = svdw[i];
        let vdwri = vdwr[i];
        let s1 = 1.0 / svdwi;
        let v1 = 1.0 / vdwri;
        let s2 = 0.5 * svdwi;
        let mut br = psi[i] * s2;
        let arg2 = br * (OBC_GAM * br - OBC_BET);
        let arg = br * (OBC_ALP + arg2);
        let th = arg.tanh();
        br = 1.0 / (s1 - v1 * th);
        brad[i] = c1 * br;
    }
    brad
}

fn shape_descriptor(n: usize, coords: &[[f64; 3]], rad: &[f64]) -> f64 {
    const TOF: f64 = 2.0 / 5.0;
    let mut tot_rad3 = 0.0;
    let mut center = [0.0; 3];
    for i in 0..n {
        let rad3 = rad[i].powi(3);
        tot_rad3 += rad3;
        for k in 0..3 {
            center[k] += coords[i][k] * rad3;
        }
    }
    for c in &mut center {
        *c /= tot_rad3;
    }
    let mut inertia = [[0.0; 3]; 3];
    for i in 0..n {
        let rad2 = rad[i] * rad[i];
        let rad3 = rad2 * rad[i];
        let vec = [
            coords[i][0] - center[0],
            coords[i][1] - center[1],
            coords[i][2] - center[2],
        ];
        let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
        let diag = rad3 * (r2 + TOF * rad2);
        for (a, row) in inertia.iter_mut().enumerate() {
            row[a] += diag;
            for (b, cell) in row.iter_mut().enumerate() {
                *cell -= rad3 * vec[a] * vec[b];
            }
        }
    }
    let det = inertia[0][0] * (inertia[1][1] * inertia[2][2] - inertia[1][2] * inertia[2][1])
        - inertia[0][1] * (inertia[1][0] * inertia[2][2] - inertia[1][2] * inertia[2][0])
        + inertia[0][2] * (inertia[1][0] * inertia[2][1] - inertia[1][1] * inertia[2][0]);
    (det.powf(1.0 / 3.0) / (TOF * tot_rad3)).sqrt()
}

fn compute_sasa(
    n: usize,
    coords: &[[f64; 3]],
    vdwsa: &[f64],
    ng: usize,
) -> Result<Vec<f64>, SolvError> {
    let w = 0.3 * AATOAU;
    let ah0 = 0.5;
    let ah1 = 3.0 / (4.0 * w);
    let ah3 = -1.0 / (4.0 * w * w * w);
    let grid = LebedevGrid::new(ng).ok_or(SolvError::BadGridSize(ng))?;

    let wrp: Vec<f64> = vdwsa
        .iter()
        .map(|&rsa| {
            let f = |r: f64| {
                (0.25 / w + 3.0 * ah3 * (0.2 * r * r - 0.5 * r * rsa + rsa * rsa / 3.0)) * r * r * r
            };
            f(rsa + w) - f(rsa - w)
        })
        .collect();
    let trj2: Vec<(f64, f64)> = vdwsa
        .iter()
        .map(|&r| ((r - w).powi(2), (r + w).powi(2)))
        .collect();

    let mut sasa = vec![0.0; n];
    for iat in 0..n {
        let rsas = vdwsa[iat];
        let mut sasai = 0.0;
        for (u, &uw) in grid.points.iter().zip(&grid.weights) {
            let p = [
                coords[iat][0] + rsas * u[0],
                coords[iat][1] + rsas * u[1],
                coords[iat][2] + rsas * u[2],
            ];
            let mut sasap = 1.0;
            for ja in 0..n {
                if ja == iat {
                    continue;
                }
                let tj = [
                    p[0] - coords[ja][0],
                    p[1] - coords[ja][1],
                    p[2] - coords[ja][2],
                ];
                let tj2 = tj[0] * tj[0] + tj[1] * tj[1] + tj[2] * tj[2];
                if tj2 < trj2[ja].1 {
                    if tj2 <= trj2[ja].0 {
                        sasap = 0.0;
                        break;
                    }
                    let uj = tj2.sqrt() - vdwsa[ja];
                    let ah3uj2 = ah3 * uj * uj;
                    let mut sasaij = ah0 + (ah1 + ah3uj2) * uj;
                    sasaij = sasaij.clamp(0.0, 1.0);
                    if sasaij < f64::MIN_POSITIVE {
                        sasap = 0.0;
                        break;
                    }
                    sasap *= sasaij;
                }
            }
            if sasap > TOLSESP {
                sasai += uw * wrp[iat] * sasap;
            }
        }
        sasa[iat] = sasai;
    }
    Ok(sasa)
}
