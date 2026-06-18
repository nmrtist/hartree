use crate::core::Molecule;
use crate::integrals::IntegralProvider;

pub fn dipole_moment<P: IntegralProvider>(
    provider: &P,
    molecule: &Molecule,
    density: &[f64],
    origin: [f64; 3],
) -> [f64; 3] {
    let n = provider.n_basis();
    let [dx, dy, dz] = provider.dipole_integrals(origin);

    let mut mu = [0.0f64; 3];

    for ao_mu in 0..n {
        for ao_nu in 0..n {
            let d = density[ao_mu * n + ao_nu];
            mu[0] -= d * dx[ao_mu * n + ao_nu];
            mu[1] -= d * dy[ao_mu * n + ao_nu];
            mu[2] -= d * dz[ao_mu * n + ao_nu];
        }
    }

    for atom in &molecule.atoms {
        let z = atom.element.z() as f64;
        let r = atom.position;
        mu[0] += z * (r[0] - origin[0]);
        mu[1] += z * (r[1] - origin[1]);
        mu[2] += z * (r[2] - origin[2]);
    }

    mu
}

pub fn center_of_mass(molecule: &Molecule) -> [f64; 3] {
    let mut com = [0.0f64; 3];
    let mut total_mass = 0.0f64;
    for atom in &molecule.atoms {
        let m = atom.element.mass();
        total_mass += m;
        for (k, c) in com.iter_mut().enumerate() {
            *c += m * atom.position[k];
        }
    }
    for c in &mut com {
        *c /= total_mass;
    }
    com
}
