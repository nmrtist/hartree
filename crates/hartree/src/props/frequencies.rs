use crate::core::{Molecule, units::FREQ_CONV_CM1};
use crate::linalg::{mat_from_row_major, mat_to_row_major, matmul, symmetric_eigh};

#[derive(Debug, Clone)]
pub struct FrequencyResult {
    pub n_atoms: usize,
    pub hessian: Vec<f64>,
    pub frequencies_cm1: Vec<f64>,
    pub normal_modes: Vec<f64>,
    pub n_imaginary: usize,
}

pub fn harmonic_frequencies(molecule: &Molecule, hessian: &[f64]) -> FrequencyResult {
    harmonic_frequencies_projected(molecule, hessian, &[])
}

pub fn harmonic_frequencies_projected(
    molecule: &Molecule,
    hessian: &[f64],
    extra_mw_vectors: &[Vec<f64>],
) -> FrequencyResult {
    let natom = molecule.len();
    let ndof = 3 * natom;

    let masses: Vec<f64> = molecule.atoms.iter().map(|a| a.element.mass()).collect();

    let mut mw = hessian.to_vec();
    for i in 0..natom {
        for ki in 0..3 {
            let row = i * 3 + ki;
            for j in 0..natom {
                for kj in 0..3 {
                    let col = j * 3 + kj;
                    mw[row * ndof + col] /= (masses[i] * masses[j]).sqrt();
                }
            }
        }
    }

    let total_mass: f64 = masses.iter().sum();
    let mut com = [0.0f64; 3];
    for (i, atom) in molecule.atoms.iter().enumerate() {
        for (k, c) in com.iter_mut().enumerate() {
            *c += masses[i] * atom.position[k];
        }
    }
    for c in &mut com {
        *c /= total_mass;
    }

    let mut raw_vecs: Vec<Vec<f64>> = Vec::with_capacity(6);

    for k in 0..3 {
        let mut v = vec![0.0f64; ndof];
        for i in 0..natom {
            v[3 * i + k] = masses[i].sqrt();
        }
        raw_vecs.push(v);
    }

    let r: Vec<[f64; 3]> = molecule
        .atoms
        .iter()
        .map(|a| {
            [
                a.position[0] - com[0],
                a.position[1] - com[1],
                a.position[2] - com[2],
            ]
        })
        .collect();

    type RotDisp = fn(&[f64; 3]) -> [f64; 3];
    let rot_disps: [RotDisp; 3] = [
        |r| [0.0, -r[2], r[1]],
        |r| [r[2], 0.0, -r[0]],
        |r| [-r[1], r[0], 0.0],
    ];
    for disp_fn in &rot_disps {
        let mut v = vec![0.0f64; ndof];
        for i in 0..natom {
            let d = disp_fn(&r[i]);
            for k in 0..3 {
                v[3 * i + k] = masses[i].sqrt() * d[k];
            }
        }
        raw_vecs.push(v);
    }

    for v in extra_mw_vectors {
        debug_assert_eq!(v.len(), ndof, "extra projection vector must be length 3N");
        raw_vecs.push(v.clone());
    }

    let orth = gram_schmidt(&raw_vecs);

    let mut proj = vec![0.0f64; ndof * ndof];
    for i in 0..ndof {
        proj[i * ndof + i] = 1.0;
    }
    for v in &orth {
        for i in 0..ndof {
            for j in 0..ndof {
                proj[i * ndof + j] -= v[i] * v[j];
            }
        }
    }

    let p = mat_from_row_major(ndof, &proj);
    let f = mat_from_row_major(ndof, &mw);
    let fp_mat = matmul(&matmul(&p, &f), &p);

    let eigh = symmetric_eigh(&fp_mat);

    let frequencies_cm1: Vec<f64> = eigh
        .values
        .iter()
        .map(|&lam| {
            if lam >= 0.0 {
                lam.sqrt() * FREQ_CONV_CM1
            } else {
                -(-lam).sqrt() * FREQ_CONV_CM1
            }
        })
        .collect();

    let n_imaginary = frequencies_cm1.iter().filter(|&&f| f < -1.0).count();
    let normal_modes = mat_to_row_major(&eigh.vectors);

    FrequencyResult {
        n_atoms: natom,
        hessian: hessian.to_vec(),
        frequencies_cm1,
        normal_modes,
        n_imaginary,
    }
}

fn gram_schmidt(vecs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut orth: Vec<Vec<f64>> = Vec::new();
    for v in vecs {
        let mut u = v.clone();
        for prev in &orth {
            let proj: f64 = u.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for (ui, &pi) in u.iter_mut().zip(prev.iter()) {
                *ui -= proj * pi;
            }
        }
        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut u {
                *x /= norm;
            }
            orth.push(u);
        }
    }
    orth
}
