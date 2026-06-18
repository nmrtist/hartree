use crate::core::{Atom, Molecule};

use crate::{Job, Method};

#[derive(Debug, Clone)]
pub struct CpFragments {
    pub fragment_a: Vec<usize>,
    pub charge_a: i32,
    pub multiplicity_a: u32,
    pub charge_b: i32,
    pub multiplicity_b: u32,
}

impl CpFragments {
    pub fn new(fragment_a: Vec<usize>) -> Self {
        Self {
            fragment_a,
            charge_a: 0,
            multiplicity_a: 1,
            charge_b: 0,
            multiplicity_b: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpResult {
    pub e_complex: f64,
    pub e_a_in_dimer_basis: f64,
    pub e_b_in_dimer_basis: f64,
    pub e_a: f64,
    pub e_b: f64,
}

impl CpResult {
    pub fn interaction_cp(&self) -> f64 {
        self.e_complex - self.e_a_in_dimer_basis - self.e_b_in_dimer_basis
    }

    pub fn interaction_uncorrected(&self) -> f64 {
        self.e_complex - self.e_a - self.e_b
    }

    pub fn bsse(&self) -> f64 {
        (self.e_a_in_dimer_basis - self.e_a) + (self.e_b_in_dimer_basis - self.e_b)
    }
}

pub fn counterpoise(job: &Job, frags: &CpFragments) -> Result<CpResult, String> {
    let mol = &job.molecule;
    let n = mol.len();

    if mol.has_ghosts() {
        return Err("counterpoise input must not already contain ghost atoms".into());
    }
    if frags.fragment_a.is_empty() {
        return Err("counterpoise fragment A is empty".into());
    }
    let mut in_a = vec![false; n];
    for &i in &frags.fragment_a {
        if i >= n {
            return Err(format!(
                "counterpoise fragment A names atom index {i}, but the complex has {n} atoms \
                 (indices are 0-based)"
            ));
        }
        if in_a[i] {
            return Err(format!(
                "counterpoise fragment A lists atom index {i} twice"
            ));
        }
        in_a[i] = true;
    }
    if frags.fragment_a.len() == n {
        return Err("counterpoise fragment B is empty (fragment A covers every atom)".into());
    }
    if frags.charge_a + frags.charge_b != mol.charge {
        return Err(format!(
            "counterpoise fragment charges are inconsistent: q_A ({}) + q_B ({}) != complex \
             charge ({})",
            frags.charge_a, frags.charge_b, mol.charge
        ));
    }

    let opts = &job.options;
    if opts.optimize_geometry {
        return Err("counterpoise is a single-point protocol; --opt is not supported".into());
    }
    if opts.compute_frequencies {
        return Err("counterpoise is a single-point protocol; --freq is not supported".into());
    }
    if opts.compute_properties {
        return Err("counterpoise does not run properties; drop --properties".into());
    }
    if opts.fod {
        return Err("counterpoise does not run the FOD diagnostic; drop --fod".into());
    }
    if opts.solvent_eps.is_some() {
        return Err(
            "counterpoise in solvent is not supported (ghost-atom C-PCM cavities \
             are unvalidated); run in gas phase"
                .into(),
        );
    }

    let ghosted = |a_real: bool| -> Vec<Atom> {
        mol.atoms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if in_a[i] == a_real {
                    *a
                } else {
                    Atom::new_ghost(a.element, a.position)
                }
            })
            .collect()
    };
    let only = |want_a: bool| -> Vec<Atom> {
        mol.atoms
            .iter()
            .enumerate()
            .filter(|(i, _)| in_a[*i] == want_a)
            .map(|(_, a)| *a)
            .collect()
    };

    let jobs: [(&str, Molecule); 5] = [
        ("E_AB^{AB} (complex)", mol.clone()),
        (
            "E_A^{AB} (A + ghost-B)",
            Molecule::new(ghosted(true), frags.charge_a, frags.multiplicity_a),
        ),
        (
            "E_B^{AB} (ghost-A + B)",
            Molecule::new(ghosted(false), frags.charge_b, frags.multiplicity_b),
        ),
        (
            "E_A^{A} (A alone)",
            Molecule::new(only(true), frags.charge_a, frags.multiplicity_a),
        ),
        (
            "E_B^{B} (B alone)",
            Molecule::new(only(false), frags.charge_b, frags.multiplicity_b),
        ),
    ];

    let mut energies = [0.0; 5];
    for (slot, (label, sub_mol)) in energies.iter_mut().zip(jobs) {
        sub_mol.validate().map_err(|e| {
            format!("counterpoise sub-job {label}: inconsistent charge/multiplicity: {e}")
        })?;
        let method = match (&job.method, sub_mol.multiplicity > 1) {
            (Method::Rhf, true) => Method::Uhf,
            (m, _) => m.clone(),
        };
        let result = Job {
            molecule: sub_mol,
            basis: job.basis.clone(),
            method,
            options: opts.clone(),
        }
        .run()
        .map_err(|e| format!("counterpoise sub-job {label}: {e}"))?;
        if !result.converged() {
            return Err(format!("counterpoise sub-job {label} did not converge"));
        }
        *slot = result.best_energy();
    }

    Ok(CpResult {
        e_complex: energies[0],
        e_a_in_dimer_basis: energies[1],
        e_b_in_dimer_basis: energies[2],
        e_a: energies[3],
        e_b: energies[4],
    })
}
