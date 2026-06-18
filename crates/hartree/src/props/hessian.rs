use crate::core::Molecule;
use rayon::prelude::*;

pub fn numerical_hessian<F>(molecule: &Molecule, step_size: f64, gradient_fn: F) -> Vec<f64>
where
    F: Fn(&Molecule) -> Vec<f64> + Sync,
{
    let natom = molecule.len();
    let ndof = 3 * natom;

    let evals: Vec<(usize, f64)> = (0..ndof)
        .flat_map(|dof| [(dof, step_size), (dof, -step_size)])
        .collect();
    let grads: Vec<Vec<f64>> = evals
        .par_iter()
        .map(|&(dof, delta)| gradient_fn(&displaced(molecule, dof / 3, dof % 3, delta)))
        .collect();

    let mut h = vec![0.0f64; ndof * ndof];
    for dof in 0..ndof {
        let gf = &grads[2 * dof];
        let gb = &grads[2 * dof + 1];
        for j in 0..ndof {
            h[dof * ndof + j] = (gf[j] - gb[j]) / (2.0 * step_size);
        }
    }

    for i in 0..ndof {
        for j in i + 1..ndof {
            let avg = (h[i * ndof + j] + h[j * ndof + i]) / 2.0;
            h[i * ndof + j] = avg;
            h[j * ndof + i] = avg;
        }
    }

    h
}

pub fn displaced(molecule: &Molecule, idx: usize, axis: usize, delta: f64) -> Molecule {
    let mut atoms = molecule.atoms.clone();
    atoms[idx].position[axis] += delta;
    Molecule::new(atoms, molecule.charge, molecule.multiplicity)
}
