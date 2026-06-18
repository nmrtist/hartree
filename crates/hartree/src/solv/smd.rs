use crate::solv::SolvError;
use crate::solv::surface::{BOHR, build_surface};

pub const SASA_PROBE_ANGSTROM: f64 = 0.4;

pub const DEFAULT_SASA_GRID: usize = 590;

const HARTREE2KCAL: f64 = 627.509_451;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmdSolvent {
    pub name: &'static str,
    pub n: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub phi: f64,
    pub psi: f64,
}

impl SmdSolvent {
    pub fn is_water(&self) -> bool {
        self.name == "water"
    }
}

macro_rules! solvent {
    ($name:literal, $n:expr, $a:expr, $b:expr, $g:expr, $e:expr, $phi:expr, $psi:expr) => {
        SmdSolvent {
            name: $name,
            n: $n,
            alpha: $a,
            beta: $b,
            gamma: $g,
            epsilon: $e,
            phi: $phi,
            psi: $psi,
        }
    };
}

pub const SMD_SOLVENTS: [SmdSolvent; 20] = [
    solvent!("water", 1.3328, 0.82, 0.35, 71.99, 78.355, 0.0, 0.0),
    solvent!("acetonitrile", 1.3442, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0),
    solvent!("methanol", 1.3288, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0),
    solvent!("ethanol", 1.3611, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0),
    solvent!("dmso", 1.4783, 0.00, 0.88, 61.78, 46.826, 0.0, 0.0),
    solvent!("acetone", 1.3588, 0.04, 0.49, 33.77, 20.493, 0.0, 0.0),
    solvent!("chloroform", 1.4459, 0.15, 0.02, 38.39, 4.7113, 0.0, 0.75),
    solvent!(
        "dichloromethane",
        1.4242,
        0.10,
        0.05,
        39.15,
        8.93,
        0.0,
        0.667
    ),
    solvent!("toluene", 1.4961, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0),
    solvent!("benzene", 1.5011, 0.0, 0.14, 40.62, 2.2706, 1.0, 0.0),
    solvent!("thf", 1.4050, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0),
    solvent!("diethylether", 1.3526, 0.00, 0.41, 23.96, 4.2400, 0.0, 0.0),
    solvent!("dmf", 1.4305, 0.00, 0.74, 49.56, 37.219, 0.0, 0.0),
    solvent!("1-octanol", 1.4295, 0.37, 0.48, 39.01, 9.8629, 0.0, 0.0),
    solvent!("n-hexane", 1.3749, 0.00, 0.00, 25.75, 1.8819, 0.0, 0.0),
    solvent!("cyclohexane", 1.4266, 0.00, 0.00, 35.48, 2.0165, 0.0, 0.0),
    solvent!(
        "carbon tetrachloride",
        1.4601,
        0.00,
        0.00,
        38.04,
        2.2280,
        0.0,
        0.8
    ),
    solvent!(
        "chlorobenzene",
        1.5241,
        0.0,
        0.07,
        47.48,
        5.6968,
        0.857,
        0.143
    ),
    solvent!("nitromethane", 1.3817, 0.06, 0.31, 52.58, 36.562, 0.0, 0.0),
    solvent!("1,4-dioxane", 1.4224, 0.00, 0.64, 47.14, 2.2099, 0.0, 0.0),
];

pub fn smd_solvent(name: &str) -> Option<&'static SmdSolvent> {
    let lower = name.to_ascii_lowercase();
    SMD_SOLVENTS.iter().find(|s| s.name == lower)
}

const BONDI: [f64; 18] = [
    1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.10, 1.80, 1.80,
    1.75, 1.88,
];

fn bondi_radius(z: usize) -> Result<f64, SolvError> {
    BONDI
        .get(z.wrapping_sub(1))
        .copied()
        .ok_or(SolvError::NoRadius(z))
}

pub fn smd_coulomb_radius(z: usize, alpha: f64) -> Result<f64, SolvError> {
    Ok(match z {
        1 => 1.20,
        6 => 1.85,
        7 => 1.89,
        8 => {
            if alpha >= 0.43 {
                1.52
            } else {
                1.52 + 1.8 * (0.43 - alpha)
            }
        }
        9 => 1.73,
        14 => 2.47,
        15 => 2.12,
        16 => 2.49,
        17 => 2.38,
        other => bondi_radius(other)?,
    })
}

pub fn smd_coulomb_radii(zs: &[usize], alpha: f64) -> Result<Vec<f64>, SolvError> {
    zs.iter()
        .map(|&z| smd_coulomb_radius(z, alpha).map(|r| r / BOHR))
        .collect()
}

pub fn sasa(centers: &[[f64; 3]], radii: &[f64], ng: usize) -> Result<Vec<f64>, SolvError> {
    let surface = build_surface(centers, radii, ng)?;
    let mut out = vec![0.0; centers.len()];
    for (&a, &ia) in surface.area.iter().zip(&surface.atom) {
        out[ia] += a;
    }
    Ok(out)
}

fn cot(r: f64, rbar: f64, dr: f64) -> f64 {
    if r < rbar + dr {
        (dr / (r - rbar - dr)).exp()
    } else {
        0.0
    }
}

fn rbar_xc(z: usize) -> Option<f64> {
    Some(match z {
        1 => 1.55,
        6..=9 => 1.84,
        15 | 16 => 2.20,
        17 => 2.10,
        35 => 2.30,
        53 => 2.60,
        _ => return None,
    })
}

struct Tensions {
    h: f64,
    c: f64,
    n: f64,
    o: f64,
    f: f64,
    si: f64,
    s: f64,
    cl: f64,
    br: f64,
    hc: f64,
    ho: f64,
    cc: f64,
    cn: f64,
    nc: f64,
    nc3: f64,
    oc: f64,
    on: f64,
    oo: f64,
    op: f64,
    molecular: f64,
}

fn tensions(s: &SmdSolvent) -> Tensions {
    if s.is_water() {
        Tensions {
            h: 48.69,
            c: 129.74,
            n: 0.0,
            o: 0.0,
            f: 38.18,
            si: 0.0,
            s: -9.10,
            cl: 9.82,
            br: -8.72,
            hc: -60.77,
            ho: 0.0,
            cc: -72.95,
            cn: 0.0,
            nc: -48.22,
            nc3: 84.10,
            oc: 68.69,
            on: 121.98,
            oo: 0.0,
            op: 68.85,
            molecular: 0.0,
        }
    } else {
        let (n, a, b) = (s.n, s.alpha, s.beta);
        Tensions {
            h: 0.0,
            c: 58.10 * n + 48.10 * a + 32.87 * b,
            n: 32.62 * n,
            o: -17.56 * n + 193.06 * a - 43.79 * b,
            f: 0.0,
            si: -18.04 * n,
            s: -33.17 * n,
            cl: -24.31 * n,
            br: -35.42 * n,
            hc: -36.37 * n,
            ho: -19.39 * n,
            cc: -62.05 * n,
            cn: -99.76 * n + 152.20 * a,
            nc: -41.00 * a,
            nc3: 0.0,
            oc: -15.70 * n + 95.99 * a,
            on: 79.13 * b,
            oo: -128.16 * b,
            op: 0.0,
            molecular: 0.35 * s.gamma - 4.19 * s.phi * s.phi - 6.68 * s.psi * s.psi + 0.0 * b * b,
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn atomic_surface_tensions(
    zs: &[usize],
    coords: &[[f64; 3]],
    solvent: &SmdSolvent,
) -> Vec<f64> {
    let t = tensions(solvent);
    let nat = zs.len();
    let dist = |i: usize, j: usize| -> f64 {
        let (p, q) = (coords[i], coords[j]);
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() * BOHR
    };
    (0..nat)
        .map(|i| match zs[i] {
            1 => {
                let mut s = t.h;
                for j in 0..nat {
                    if zs[j] == 6 {
                        s += t.hc * cot(dist(i, j), 1.55, 0.3);
                    } else if zs[j] == 8 {
                        s += t.ho * cot(dist(i, j), 1.55, 0.3);
                    }
                }
                s
            }
            6 => {
                let mut s = t.c;
                let mut tn = 0.0;
                for j in 0..nat {
                    if j != i && zs[j] == 6 {
                        s += t.cc * cot(dist(i, j), 1.84, 0.3);
                    } else if zs[j] == 7 {
                        tn += cot(dist(i, j), 1.84, 0.3);
                    }
                }
                s + t.cn * tn * tn
            }
            7 => {
                let mut s = t.n;
                let mut tnc = 0.0;
                let mut tnc3 = 0.0;
                for j in 0..nat {
                    if zs[j] != 6 {
                        continue;
                    }
                    let mut tk = 0.0;
                    for k in 0..nat {
                        if k != i
                            && k != j
                            && let Some(rbar) = rbar_xc(zs[k])
                        {
                            tk += cot(dist(j, k), rbar, 0.3);
                        }
                    }
                    tnc += cot(dist(i, j), 1.84, 0.3) * tk * tk;
                    tnc3 += cot(dist(i, j), 1.225, 0.065);
                }
                s += t.nc * tnc.powf(1.3);
                s + t.nc3 * tnc3
            }
            8 => {
                let mut s = t.o;
                for j in 0..nat {
                    match zs[j] {
                        6 => s += t.oc * cot(dist(i, j), 1.33, 0.1),
                        7 => s += t.on * cot(dist(i, j), 1.50, 0.3),
                        8 if j != i => s += t.oo * cot(dist(i, j), 1.80, 0.3),
                        15 => s += t.op * cot(dist(i, j), 2.10, 0.3),
                        _ => {}
                    }
                }
                s
            }
            9 => t.f,
            14 => t.si,
            16 => t.s,
            17 => t.cl,
            35 => t.br,
            _ => 0.0,
        })
        .collect()
}

pub fn cds_energy(
    zs: &[usize],
    coords: &[[f64; 3]],
    solvent: &SmdSolvent,
    ng: usize,
) -> Result<f64, SolvError> {
    assert_eq!(zs.len(), coords.len());
    let radii: Vec<f64> = zs
        .iter()
        .map(|&z| bondi_radius(z).map(|r| (r + SASA_PROBE_ANGSTROM) / BOHR))
        .collect::<Result<_, _>>()?;
    let areas = sasa(coords, &radii, ng)?; // bohr²
    let sigmas = atomic_surface_tensions(zs, coords, solvent);
    let sig_m = tensions(solvent).molecular;
    let cal_per_mol: f64 = areas
        .iter()
        .zip(&sigmas)
        .map(|(&a, &s)| a * BOHR * BOHR * (s + sig_m))
        .sum();
    Ok(cal_per_mol / 1000.0 / HARTREE2KCAL)
}
