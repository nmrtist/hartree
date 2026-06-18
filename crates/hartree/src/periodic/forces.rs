use crate::integrals::integral::Basis;
use crate::integrals::integral::periodic::{
    LatticeCollocator, ProjectorChannel, bloch_kinetic_grad_contract, bloch_overlap_grad_contract,
    bloch_projector_overlaps_grad,
};
use crate::linalg::C64;
use latx::{Cell, KPoint};

use crate::periodic::PeriodicError;
use crate::periodic::converged::{ConvergedState, assert_basis_atoms_align};
use crate::periodic::pseudo::{core_charge_forces, local_sr_forces, overlap_forces};
use crate::periodic::scf::{PeriodicAtom, PeriodicScfOptions};
use crate::periodic::xc::GridXc;

pub fn periodic_forces(
    basis: &Basis,
    cell: &Cell,
    kpoints: &[KPoint],
    n_elec: usize,
    atoms: &[PeriodicAtom],
    xc: &GridXc,
    options: &PeriodicScfOptions,
) -> Result<Vec<[f64; 3]>, PeriodicError> {
    let natom = atoms.len();
    assert_basis_atoms_align(basis, atoms);

    let state = ConvergedState::converge(basis, cell, kpoints, n_elec, atoms, xc, options)?;

    let collocator = LatticeCollocator::new(basis, &state.grid);
    let phases = collocator.bloch_phases(&state.kfracs, &state.weights);
    let ao_atom = ao_atom_map(basis);

    let mut forces = vec![[0.0_f64; 3]; natom];

    let g_kin = bloch_kinetic_grad_contract(
        basis,
        cell,
        &state.kfracs,
        &state.weights,
        &state.p_k,
        state.rmax,
    );
    let g_ovlp = bloch_overlap_grad_contract(
        basis,
        cell,
        &state.kfracs,
        &state.weights,
        &state.w_k,
        state.rmax,
    );
    for i in 0..natom {
        for ax in 0..3 {
            forces[i][ax] += -g_kin[i][ax] + g_ovlp[i][ax];
        }
    }

    let f_core = core_charge_forces(&state.grid, &state.local_atoms, &state.v_h);
    let f_locsr = local_sr_forces(&state.grid, &state.local_atoms, &state.n_r);
    for i in 0..natom {
        for ax in 0..3 {
            forces[i][ax] += f_core[i][ax] + f_locsr[i][ax];
        }
    }

    let f_coll = collocator.collocation_pulay_forces(
        &state.grid,
        &state.p_k,
        &state.v_loc_grid,
        &phases,
        &ao_atom,
        natom,
    );
    let f_nl = projector_forces(
        basis,
        cell,
        atoms,
        &state.kfracs,
        &state.weights,
        &state.p_k,
        state.rmax,
        &ao_atom,
    );
    for i in 0..natom {
        for ax in 0..3 {
            forces[i][ax] += f_coll[i][ax] + f_nl[i][ax];
        }
    }

    let f_ovrl = overlap_forces(cell, &state.local_atoms);
    for i in 0..natom {
        for ax in 0..3 {
            forces[i][ax] += f_ovrl[i][ax];
        }
    }

    Ok(forces)
}

fn ao_atom_map(basis: &Basis) -> Vec<usize> {
    let atoms = basis.atoms();
    let mut map = Vec::with_capacity(basis.nao());
    for sh in basis.shells() {
        let c = sh.center();
        let atom = atoms
            .iter()
            .position(|a| (0..3).map(|x| (a[x] - c[x]).powi(2)).sum::<f64>() < 1e-18)
            .expect("shell center must be a basis atom");
        for _ in 0..sh.n_func() {
            map.push(atom);
        }
    }
    map
}

#[allow(clippy::too_many_arguments)]
fn projector_forces(
    basis: &Basis,
    cell: &Cell,
    atoms: &[PeriodicAtom],
    kfracs: &[[f64; 3]],
    weights: &[f64],
    p_k: &[Vec<C64>],
    rmax: f64,
    ao_atom: &[usize],
) -> Vec<[f64; 3]> {
    let n = basis.nao();
    let mut forces = vec![[0.0_f64; 3]; atoms.len()];
    for (jatom, a) in atoms.iter().enumerate() {
        for ch in &a.channels {
            let n_proj = ch.h.len();
            if n_proj == 0 {
                continue;
            }
            let nlm = 2 * ch.l + 1;
            let ncol = n_proj * nlm;
            let chan = ProjectorChannel {
                center: a.center,
                l: ch.l,
                n_proj,
                r_l: ch.r_l,
            };
            for (ik, (&kf, &wk)) in kfracs.iter().zip(weights).enumerate() {
                let (b, db) = bloch_projector_overlaps_grad(basis, cell, &chan, kf, rmax);
                let pk = &p_k[ik];
                let mut pb = vec![C64::new(0.0, 0.0); n * ncol];
                for mu in 0..n {
                    let prow = &pk[mu * n..mu * n + n];
                    for col in 0..ncol {
                        let mut acc = C64::new(0.0, 0.0);
                        for nu in 0..n {
                            acc += prow[nu] * b[nu * ncol + col];
                        }
                        pb[mu * ncol + col] = acc;
                    }
                }
                for (mu, &amu) in ao_atom.iter().enumerate() {
                    for p in 0..n_proj {
                        for m in 0..nlm {
                            let mut phi = C64::new(0.0, 0.0);
                            for (q, &hpq) in ch.h[p].iter().enumerate() {
                                if hpq == 0.0 {
                                    continue;
                                }
                                phi += C64::new(hpq, 0.0) * pb[mu * ncol + q * nlm + m].conj();
                            }
                            let col_p = p * nlm + m;
                            for axis in 0..3 {
                                let c = wk * 2.0 * (db[axis][mu * ncol + col_p] * phi).re;
                                forces[amu][axis] -= c;
                                forces[jatom][axis] += c;
                            }
                        }
                    }
                }
            }
        }
    }
    forces
}

pub fn finite_difference_forces<F>(
    positions: &[[f64; 3]],
    h: f64,
    mut energy_at: F,
) -> Result<Vec<[f64; 3]>, PeriodicError>
where
    F: FnMut(&[[f64; 3]]) -> Result<f64, PeriodicError>,
{
    let mut forces = vec![[0.0; 3]; positions.len()];
    let mut work = positions.to_vec();
    for i in 0..positions.len() {
        for axis in 0..3 {
            let r0 = positions[i][axis];
            work[i][axis] = r0 + h;
            let e_plus = energy_at(&work)?;
            work[i][axis] = r0 - h;
            let e_minus = energy_at(&work)?;
            work[i][axis] = r0; // restore
            forces[i][axis] = -(e_plus - e_minus) / (2.0 * h);
        }
    }
    Ok(forces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_forces_match_quadratic_gradient() {
        let k = 2.0;
        let centers = [[0.0, 0.0, 0.0], [1.0, 0.5, -0.3]];
        let positions = [[0.1, -0.2, 0.05], [1.2, 0.4, -0.3]];
        let energy_at = |p: &[[f64; 3]]| -> Result<f64, PeriodicError> {
            let mut e = 0.0;
            for (pi, ci) in p.iter().zip(&centers) {
                for a in 0..3 {
                    e += 0.5 * k * (pi[a] - ci[a]).powi(2);
                }
            }
            Ok(e)
        };
        let f = finite_difference_forces(&positions, 1e-4, energy_at).unwrap();
        for (i, fi) in f.iter().enumerate() {
            for a in 0..3 {
                let expect = -k * (positions[i][a] - centers[i][a]);
                assert!(
                    (fi[a] - expect).abs() < 1e-6,
                    "atom {i} axis {a}: {} vs {expect}",
                    fi[a]
                );
            }
        }
    }

    use crate::basis::GthSet;
    use crate::integrals::integral::Shell;
    use crate::periodic::NonlocalChannel;
    use crate::periodic::scf::run_scf_periodic;

    fn si_szv_shells(center: [f64; 3]) -> Vec<Shell> {
        let exps = vec![1.203_240_36, 0.468_838_597, 0.167_985_391, 0.057_561_689];
        let s = vec![
            0.329_035_675_9,
            -0.253_316_261_6,
            -0.787_093_651_7,
            -0.190_987_019_3,
        ];
        let p = vec![
            0.047_453_643_9,
            -0.259_449_546_2,
            -0.544_093_223_5,
            -0.362_398_465_2,
        ];
        vec![
            Shell::new_spherical(0, center, exps.clone(), s).unwrap(),
            Shell::new_spherical(1, center, exps, p).unwrap(),
        ]
    }

    fn si_atom(center: [f64; 3]) -> PeriodicAtom {
        let set = GthSet::load_pade().unwrap();
        let si = set.get(14).unwrap();
        let channels = si
            .nonlocal
            .iter()
            .filter(|nl| !nl.h.is_empty())
            .map(|nl| NonlocalChannel {
                l: nl.l,
                r_l: nl.r,
                h: nl.h.clone(),
            })
            .collect();
        PeriodicAtom {
            center,
            z_ion: si.z_ion,
            r_loc: si.local.r_loc,
            c: si.local.c.clone(),
            channels,
        }
    }

    #[test]
    #[ignore = "force validation: run manually (minutes, ~13 SCFs on the 2-atom cell)"]
    fn periodic_forces_match_finite_difference_si() {
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1_eq = cell.frac_to_cart([0.25, 0.25, 0.25]);
        let r1 = [r1_eq[0] + 0.18, r1_eq[1] - 0.12, r1_eq[2] + 0.09];
        let positions = [r0, r1];

        let xc = GridXc::pade();
        let kpoints = [KPoint::gamma()];
        let options = PeriodicScfOptions {
            e_cut: 60.0,
            max_iter: 300,
            energy_tol: 1e-9,
            density_tol: 1e-8,
            mixing: 0.3,
            bloch_rmax: None,
            cache_chi: true,
        };

        let build = |p: &[[f64; 3]]| -> (Basis, Vec<PeriodicAtom>) {
            let mut shells = si_szv_shells(p[0]);
            shells.extend(si_szv_shells(p[1]));
            (Basis::new(shells), vec![si_atom(p[0]), si_atom(p[1])])
        };

        let (basis, atoms) = build(&positions);
        let analytic = periodic_forces(&basis, &cell, &kpoints, 8, &atoms, &xc, &options).unwrap();

        let fd = finite_difference_forces(&positions, 1e-3, |p| {
            let (b, at) = build(p);
            Ok(run_scf_periodic(&b, &cell, &kpoints, 8, &at, &xc, &options)?.energy)
        })
        .unwrap();

        eprintln!("[Si-forces] analytic {analytic:?}\n           FD       {fd:?}");
        for k in 0..2 {
            for ax in 0..3 {
                assert!(
                    (analytic[k][ax] - fd[k][ax]).abs() < 5e-4,
                    "atom {k} axis {ax}: analytic {} vs FD {}",
                    analytic[k][ax],
                    fd[k][ax]
                );
            }
        }
        for (ax, (&f0, &f1)) in analytic[0].iter().zip(&analytic[1]).enumerate() {
            let net = f0 + f1;
            assert!(net.abs() < 2e-3, "net force axis {ax} = {net}");
        }
    }
}
