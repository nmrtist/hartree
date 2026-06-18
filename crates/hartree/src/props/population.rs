use crate::core::Molecule;
use crate::integrals::IntegralProvider;
use crate::linalg::{mat_from_row_major, mat_to_row_major, matmul, symmetric_eigh};

#[derive(Debug, Clone)]
pub struct PopulationAnalysis {
    pub mulliken_charges: Vec<f64>,
    pub lowdin_charges: Vec<f64>,
    pub mayer_bond_orders: Vec<Vec<f64>>,
}

pub fn population_analysis<P: IntegralProvider>(
    provider: &P,
    molecule: &Molecule,
    density_alpha: &[f64],
    density_beta: &[f64],
) -> PopulationAnalysis {
    let n = provider.n_basis();
    let natom = molecule.len();

    let s_vec = mat_to_row_major(&provider.overlap());
    let ao_atom = provider.ao_atom_indices();

    let density: Vec<f64> = density_alpha
        .iter()
        .zip(density_beta.iter())
        .map(|(a, b)| a + b)
        .collect();

    let d_mat = mat_from_row_major(n, &density);
    let s_mat = mat_from_row_major(n, &s_vec);

    let ds = mat_to_row_major(&matmul(&d_mat, &s_mat));

    let mut mulliken_pop = vec![0.0f64; natom];
    for mu in 0..n {
        mulliken_pop[ao_atom[mu]] += ds[mu * n + mu];
    }
    let mulliken_charges: Vec<f64> = molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| a.element.z() as f64 - mulliken_pop[i])
        .collect();

    let eigh = symmetric_eigh(&s_mat);
    let m = eigh.values.len();
    let mut s_half_data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..m {
                let lam = eigh.values[k];
                if lam > 0.0 {
                    val += eigh.vectors[(i, k)] * lam.sqrt() * eigh.vectors[(j, k)];
                }
            }
            s_half_data[i * n + j] = val;
        }
    }
    let s_half = mat_from_row_major(n, &s_half_data);
    let sds = mat_to_row_major(&matmul(&matmul(&s_half, &d_mat), &s_half));

    let mut lowdin_pop = vec![0.0f64; natom];
    for mu in 0..n {
        lowdin_pop[ao_atom[mu]] += sds[mu * n + mu];
    }
    let lowdin_charges: Vec<f64> = molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| a.element.z() as f64 - lowdin_pop[i])
        .collect();

    let mut mayer = vec![vec![0.0f64; natom]; natom];
    for mu in 0..n {
        let a = ao_atom[mu];
        for nu in 0..n {
            let b = ao_atom[nu];
            mayer[a][b] += ds[mu * n + nu] * ds[nu * n + mu];
        }
    }

    PopulationAnalysis {
        mulliken_charges,
        lowdin_charges,
        mayer_bond_orders: mayer,
    }
}
