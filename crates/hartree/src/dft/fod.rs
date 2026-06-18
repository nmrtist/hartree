use std::io::Write as _;
use std::path::Path;

use crate::basis::ShellData;
use crate::core::Molecule;
use crate::scf::ScfResult;

use crate::dft::ao::{self, eval_ao_batch};
use crate::dft::density::batch_density;
use crate::dft::error::{DftError, Result};
use crate::dft::grid::MolecularGrid;

#[derive(Debug, Clone)]
pub struct FodResult {
    pub n_fod: f64,
    pub n_fod_alpha: f64,
    pub n_fod_beta: f64,
    pub temperature_k: f64,
}

pub fn fod_default_temperature(exx_fraction: f64) -> f64 {
    5000.0 + 20000.0 * exx_fraction
}

pub fn fod_weights(occ: &[f64], n_occ: usize) -> Vec<f64> {
    occ.iter()
        .enumerate()
        .map(|(i, &f)| if i < n_occ { 1.0 - f } else { f })
        .collect()
}

pub fn fod_analysis(scf: &ScfResult, temperature_k: f64) -> Result<FodResult> {
    let (fa, fb) = scf
        .occupations
        .as_ref()
        .ok_or(DftError::NoFractionalOccupations)?;
    let n_fod_alpha: f64 = fod_weights(fa, scf.n_alpha).iter().sum();
    let n_fod_beta: f64 = fod_weights(fb, scf.n_beta).iter().sum();
    Ok(FodResult {
        n_fod: n_fod_alpha + n_fod_beta,
        n_fod_alpha,
        n_fod_beta,
        temperature_k,
    })
}

pub fn fod_density_matrices(scf: &ScfResult) -> Result<(Vec<f64>, Vec<f64>)> {
    let (fa, fb) = scf
        .occupations
        .as_ref()
        .ok_or(DftError::NoFractionalOccupations)?;
    let n = scf.n_basis;
    let m = scf.n_orbitals;
    let build = |c: &[f64], w: &[f64]| -> Vec<f64> {
        let mut d = vec![0.0; n * n];
        for j in 0..m {
            let wj = w[j];
            if wj == 0.0 {
                continue;
            }
            for mu in 0..n {
                let cw = wj * c[mu * m + j];
                if cw == 0.0 {
                    continue;
                }
                for nu in 0..n {
                    d[mu * n + nu] += cw * c[nu * m + j];
                }
            }
        }
        d
    };
    let da = build(&scf.mo_coeff_alpha, &fod_weights(fa, scf.n_alpha));
    let db = build(&scf.mo_coeff_beta, &fod_weights(fb, scf.n_beta));
    Ok((da, db))
}

pub fn fod_grid_integral(
    shells: &[ShellData],
    nao: usize,
    grid: &MolecularGrid,
    d_fod: &[f64],
) -> Result<f64> {
    let weights = &grid.weights;
    ao::par_blocks_fold(
        shells,
        nao,
        &grid.points,
        false,
        || 0.0,
        |acc, batch, start| {
            let bd = batch_density(batch, d_fod, false);
            acc + weights[start..start + batch.npts]
                .iter()
                .zip(&bd.rho)
                .map(|(&w, &r)| w * r)
                .sum::<f64>()
        },
        |a, b| a + b,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct CubeParams {
    pub margin: f64,
    pub spacing: f64,
}

impl Default for CubeParams {
    fn default() -> Self {
        Self {
            margin: 4.0,
            spacing: 0.2,
        }
    }
}

pub fn write_fod_cube(
    path: &Path,
    mol: &Molecule,
    shells: &[ShellData],
    nao: usize,
    d_fod_total: &[f64],
    params: &CubeParams,
) -> std::io::Result<()> {
    ao::ensure_supported(shells)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let (origin, npts) = cube_lattice(mol, params);
    let [nx, ny, nz] = npts;
    let h = params.spacing;

    let file = std::fs::File::create(path)?;
    let mut out = std::io::BufWriter::new(file);
    writeln!(out, "hartree FOD density rho_FOD (alpha+beta), bohr")?;
    writeln!(out, "Grimme fractional occupation number weighted density")?;
    writeln!(
        out,
        "{:5} {:12.6} {:12.6} {:12.6}",
        mol.atoms.len(),
        origin[0],
        origin[1],
        origin[2]
    )?;
    writeln!(out, "{:5} {:12.6} {:12.6} {:12.6}", nx, h, 0.0, 0.0)?;
    writeln!(out, "{:5} {:12.6} {:12.6} {:12.6}", ny, 0.0, h, 0.0)?;
    writeln!(out, "{:5} {:12.6} {:12.6} {:12.6}", nz, 0.0, 0.0, h)?;
    for atom in &mol.atoms {
        let z = atom.element.z();
        writeln!(
            out,
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}",
            z, z as f64, atom.position[0], atom.position[1], atom.position[2]
        )?;
    }

    let total = nx * ny * nz;
    let mut written = 0usize; // values on the current line
    let mut block = Vec::with_capacity(ao::BLOCK_SIZE);
    let mut flush = |out: &mut std::io::BufWriter<std::fs::File>,
                     block: &mut Vec<[f64; 3]>|
     -> std::io::Result<()> {
        if block.is_empty() {
            return Ok(());
        }
        let batch = eval_ao_batch(shells, nao, block, false);
        let bd = batch_density(&batch, d_fod_total, false);
        for &v in &bd.rho {
            write!(out, " {:12.5E}", v)?;
            written += 1;
            if written.is_multiple_of(6) || written == total {
                writeln!(out)?;
            }
        }
        block.clear();
        Ok(())
    };
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                block.push([
                    origin[0] + ix as f64 * h,
                    origin[1] + iy as f64 * h,
                    origin[2] + iz as f64 * h,
                ]);
                if block.len() == ao::BLOCK_SIZE {
                    flush(&mut out, &mut block)?;
                }
            }
        }
    }
    flush(&mut out, &mut block)?;
    out.flush()
}

fn cube_lattice(mol: &Molecule, params: &CubeParams) -> ([f64; 3], [usize; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for atom in &mol.atoms {
        for k in 0..3 {
            lo[k] = lo[k].min(atom.position[k]);
            hi[k] = hi[k].max(atom.position[k]);
        }
    }
    let origin = [
        lo[0] - params.margin,
        lo[1] - params.margin,
        lo[2] - params.margin,
    ];
    let mut npts = [0usize; 3];
    for k in 0..3 {
        let extent = hi[k] + params.margin - origin[k];
        npts[k] = (extent / params.spacing).ceil() as usize + 1;
    }
    (origin, npts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_temperature_formula() {
        assert_eq!(fod_default_temperature(0.0), 5000.0); // TPSS and all pure (m)GGAs
        assert_eq!(fod_default_temperature(0.25), 10000.0); // PBE0
        assert_eq!(fod_default_temperature(0.20), 9000.0); // B3LYP
        assert_eq!(fod_default_temperature(1.0), 25000.0); // Hartree–Fock
    }

    #[test]
    fn weights_split_at_fermi_level() {
        let occ = [1.0, 0.9, 0.1, 0.0];
        let w = fod_weights(&occ, 2);
        assert_eq!(w, vec![0.0, 0.09999999999999998, 0.1, 0.0]);
        assert!((w.iter().sum::<f64>() - 0.2).abs() < 1e-14);
        let w0 = fod_weights(&[1.0, 1.0, 0.0], 2);
        assert!(w0.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cube_lattice_covers_box() {
        use crate::core::{Atom, Element};
        let mol = Molecule::new(
            vec![
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 0.0]),
                Atom::new(Element::from_z(1).unwrap(), [0.0, 0.0, 2.0]),
            ],
            0,
            1,
        );
        let params = CubeParams {
            margin: 4.0,
            spacing: 0.2,
        };
        let (origin, n) = cube_lattice(&mol, &params);
        assert_eq!(origin, [-4.0, -4.0, -4.0]);
        assert_eq!(n, [41, 41, 51]);
        for k in 0..3 {
            let last = origin[k] + (n[k] - 1) as f64 * params.spacing;
            assert!(last >= [4.0, 4.0, 6.0][k] - 1e-12);
        }
    }
}
