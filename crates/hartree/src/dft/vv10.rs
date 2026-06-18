use crate::basis::ShellData;
use crate::core::Molecule;
use rayon::prelude::*;

use crate::dft::ao::par_blocks_fold;
use crate::dft::density::batch_density;
use crate::dft::error::Result;
use crate::dft::grid::MolecularGrid;

pub const NL_GRID_LEVEL: usize = 1;

const RHO_CUTOFF: f64 = 1e-10;

pub fn vv10_beta(b: f64) -> f64 {
    (3.0 / (b * b)).powf(0.75) / 32.0
}

pub fn vv10_energy(
    mol: &Molecule,
    shells: &[ShellData],
    nao: usize,
    d_tot: &[f64],
    b: f64,
    c: f64,
) -> Result<f64> {
    let grid = MolecularGrid::build(mol, NL_GRID_LEVEL)?;

    let mut segments = par_blocks_fold(
        shells,
        nao,
        &grid.points,
        true,
        Vec::new,
        |mut acc: Vec<(usize, Vec<f64>, Vec<[f64; 3]>)>, batch, start| {
            let bd = batch_density(batch, d_tot, true);
            acc.push((start, bd.rho, bd.grad));
            acc
        },
        |mut a, mut b| {
            a.append(&mut b);
            a
        },
    )?;
    segments.sort_unstable_by_key(|(start, _, _)| *start);

    let n = grid.points.len();
    let mut rho = Vec::with_capacity(n);
    let mut grad = Vec::with_capacity(n);
    for (_, r, g) in segments {
        rho.extend(r);
        grad.extend(g);
    }

    Ok(vv10_energy_on_points(
        &grid.points,
        &grid.weights,
        &rho,
        &grad,
        b,
        c,
    ))
}

pub fn vv10_energy_on_points(
    points: &[[f64; 3]],
    weights: &[f64],
    rho: &[f64],
    grad: &[[f64; 3]],
    b: f64,
    c: f64,
) -> f64 {
    assert_eq!(points.len(), weights.len());
    assert_eq!(points.len(), rho.len());
    assert_eq!(points.len(), grad.len());

    let kappa_pref = 3.0 * std::f64::consts::PI * b / (9.0 * std::f64::consts::PI).powf(1.0 / 6.0);
    let mut r = Vec::new(); // position
    let mut wn = Vec::new(); // w·n
    let mut w0 = Vec::new(); // ω₀
    let mut kap = Vec::new(); // κ
    for i in 0..points.len() {
        let n = rho[i];
        if n <= RHO_CUTOFF || weights[i] == 0.0 {
            continue;
        }
        let g = grad[i];
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        let wg2 = c * (g2 / (n * n)).powi(2); // C·|∇n|⁴/n⁴
        let wp2 = 4.0 * std::f64::consts::PI * n / 3.0; // ω_p²/3
        r.push(points[i]);
        wn.push(weights[i] * n);
        w0.push((wg2 + wp2).sqrt());
        kap.push(kappa_pref * n.powf(1.0 / 6.0));
    }

    let beta = vv10_beta(b);
    let n_pts = r.len();

    let idx: Vec<usize> = (0..n_pts).collect();
    let partials: Vec<f64> = idx
        .par_chunks(64)
        .map(|chunk| {
            let mut e = 0.0;
            for &i in chunk {
                let ri = r[i];
                let (w0i, ki) = (w0[i], kap[i]);
                let mut kernel = 0.0;
                for j in 0..n_pts {
                    let d = [r[j][0] - ri[0], r[j][1] - ri[1], r[j][2] - ri[2]];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    let g = w0i * r2 + ki;
                    let gp = w0[j] * r2 + kap[j];
                    kernel += wn[j] * (-1.5) / (g * gp * (g + gp));
                }
                e += wn[i] * (beta + 0.5 * kernel);
            }
            e
        })
        .collect();
    partials.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_symmetry_two_points() {
        let pts = [[0.0, 0.0, 0.0], [1.3, -0.2, 0.7]];
        let w = [0.8, 1.1];
        let rho = [0.3, 0.05];
        let grad = [[0.1, 0.0, -0.2], [0.02, 0.03, 0.0]];
        let (b, c) = (6.0, 0.01);
        let e = vv10_energy_on_points(&pts, &w, &rho, &grad, b, c);
        let e_swapped = vv10_energy_on_points(
            &[pts[1], pts[0]],
            &[w[1], w[0]],
            &[rho[1], rho[0]],
            &[grad[1], grad[0]],
            b,
            c,
        );
        assert!(
            (e - e_swapped).abs() < 1e-15,
            "kernel not symmetric: {e} vs {e_swapped}"
        );
    }

    #[test]
    fn beta_n_low_density_limit() {
        let pts: Vec<[f64; 3]> = (0..20).map(|i| [i as f64 * 2.0, 0.0, 0.0]).collect();
        let w = vec![1.0; 20];
        let rho = vec![1e-6; 20];
        let grad = vec![[0.0; 3]; 20];
        let (b, c) = (6.0, 0.01);
        let e = vv10_energy_on_points(&pts, &w, &rho, &grad, b, c);
        let n: f64 = w.iter().zip(&rho).map(|(w, r)| w * r).sum();
        let beta_n = vv10_beta(b) * n;
        assert!(beta_n > 0.0);
        assert!(
            ((e - beta_n) / beta_n).abs() < 1e-3,
            "E_nl = {e} vs β·N = {beta_n}"
        );
    }

    #[test]
    fn beta_value() {
        assert!((vv10_beta(6.0) - 0.004846900307823436).abs() < 1e-15);
        assert!((vv10_beta(5.9) - 0.00497).abs() < 1e-4);
    }
}
