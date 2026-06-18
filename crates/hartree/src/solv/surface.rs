use crate::dft::grid::lebedev::LebedevGrid;

use crate::solv::SolvError;

pub(crate) const BOHR: f64 = 0.529_177_210_92;

const VDW_SCALE: f64 = 1.2;

const MODIFIED_BONDI: [f64; 18] = [
    1.10, // H (modified)
    1.40, // He
    1.82, // Li
    1.53, // Be
    1.92, // B
    1.70, // C
    1.55, // N
    1.52, // O
    1.47, // F
    1.54, // Ne
    2.27, // Na
    1.73, // Mg
    1.84, // Al
    2.10, // Si
    1.80, // P
    1.80, // S
    1.75, // Cl
    1.88, // Ar
];

pub fn cavity_radius(z: usize) -> Result<f64, SolvError> {
    MODIFIED_BONDI
        .get(z.wrapping_sub(1))
        .map(|&r| VDW_SCALE * r / BOHR)
        .ok_or(SolvError::NoRadius(z))
}

fn xi_constant(ng: usize) -> Option<f64> {
    Some(match ng {
        6 => 4.84566077868,
        14 => 4.86458714334,
        26 => 4.85478226219,
        38 => 4.90105812685,
        50 => 4.89250673295,
        86 => 4.89741372580,
        110 => 4.90101060987,
        146 => 4.89825187392,
        170 => 4.90685517725,
        194 => 4.90337644248,
        302 => 4.90498088169,
        434 => 4.90567349080,
        590 => 4.90624071359,
        _ => return None,
    })
}

fn switch_h(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
    }
}

#[derive(Debug, Clone)]
pub struct CavitySurface {
    pub points: Vec<[f64; 3]>,
    pub zeta: Vec<f64>,
    pub switch_f: Vec<f64>,
    pub area: Vec<f64>,
    pub atom: Vec<usize>,
}

pub fn build_surface(
    centers: &[[f64; 3]],
    radii: &[f64],
    ng: usize,
) -> Result<CavitySurface, SolvError> {
    assert_eq!(centers.len(), radii.len());
    let xi = xi_constant(ng).ok_or(SolvError::BadGridSize(ng))?;
    let unit = LebedevGrid::new(ng).ok_or(SolvError::BadGridSize(ng))?;

    let natm = centers.len();
    let r_sw: Vec<f64> = radii
        .iter()
        .map(|r| r * (14.0 / ng as f64).sqrt())
        .collect();
    let r_in: Vec<f64> = radii
        .iter()
        .zip(&r_sw)
        .map(|(r, sw)| {
            let ratio = r / sw;
            let alpha = 0.5 + ratio - (ratio * ratio - 1.0 / 28.0).sqrt();
            r - alpha * sw
        })
        .collect();

    let mut points = Vec::new();
    let mut zeta = Vec::new();
    let mut switch_f = Vec::new();
    let mut area = Vec::new();
    let mut atom = Vec::new();

    for ia in 0..natm {
        let r_vdw = radii[ia];
        for (u, &uw) in unit.points.iter().zip(&unit.weights) {
            let p = [
                centers[ia][0] + r_vdw * u[0],
                centers[ia][1] + r_vdw * u[1],
                centers[ia][2] + r_vdw * u[2],
            ];
            let w = uw * 4.0 * std::f64::consts::PI;
            let mut swf = 1.0;
            for j in 0..natm {
                if j == ia {
                    continue;
                }
                let c = centers[j];
                let rij =
                    ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
                let mut d = (rij - r_in[j]) / r_sw[j];
                if d < 1e-8 {
                    d = 0.0;
                }
                swf *= switch_h(d);
            }
            if w * swf > 1e-16 {
                points.push(p);
                zeta.push(xi / (r_vdw * w.sqrt()));
                switch_f.push(swf);
                area.push(w * swf * r_vdw * r_vdw);
                atom.push(ia);
            }
        }
    }

    if points.is_empty() {
        return Err(SolvError::EmptySurface);
    }
    Ok(CavitySurface {
        points,
        zeta,
        switch_f,
        area,
        atom,
    })
}
