use crate::integrals::integral::Basis;
use crate::integrals::integral::periodic::{
    LatticeCollocator, ProjectorChannel, bloch_kinetic_stress_contract,
    bloch_overlap_stress_contract, bloch_projector_overlaps_strain, hartree_reciprocal_stress,
};
use crate::linalg::C64;
use latx::{Cell, KPoint};

use crate::periodic::PeriodicError;
use crate::periodic::converged::{ConvergedState, assert_basis_atoms_align};
use crate::periodic::pseudo::{core_charge_stress, local_sr_stress, overlap_stress};
use crate::periodic::scf::{PeriodicAtom, PeriodicScfOptions};
use crate::periodic::xc::GridXc;

fn deform(m: &[[f64; 3]; 3], lambda: f64, v: [f64; 3]) -> [f64; 3] {
    let mut out = v;
    for (a, oa) in out.iter_mut().enumerate() {
        for (b, &vb) in v.iter().enumerate() {
            *oa += lambda * m[a][b] * vb;
        }
    }
    out
}

pub fn apply_strain(
    cell: &Cell,
    positions: &[[f64; 3]],
    m: &[[f64; 3]; 3],
    lambda: f64,
) -> Result<(Cell, Vec<[f64; 3]>), PeriodicError> {
    let (a1, a2, a3) = cell.vectors();
    let strained = Cell::from_vectors(
        deform(m, lambda, a1),
        deform(m, lambda, a2),
        deform(m, lambda, a3),
    )
    .map_err(|e| PeriodicError::Dimension(format!("singular strained cell: {e}")))?;
    let pos = positions.iter().map(|&p| deform(m, lambda, p)).collect();
    Ok((strained, pos))
}

pub fn finite_difference_strain_energy<F>(
    cell: &Cell,
    positions: &[[f64; 3]],
    m: &[[f64; 3]; 3],
    h: f64,
    mut energy_at: F,
) -> Result<f64, PeriodicError>
where
    F: FnMut(&Cell, &[[f64; 3]]) -> Result<f64, PeriodicError>,
{
    let (cp, pp) = apply_strain(cell, positions, m, h)?;
    let (cm, pm) = apply_strain(cell, positions, m, -h)?;
    let e_plus = energy_at(&cp, &pp)?;
    let e_minus = energy_at(&cm, &pm)?;
    Ok((e_plus - e_minus) / (2.0 * h))
}

#[allow(clippy::too_many_lines)]
pub fn periodic_stress(
    basis: &Basis,
    cell: &Cell,
    kpoints: &[KPoint],
    n_elec: usize,
    atoms: &[PeriodicAtom],
    xc: &GridXc,
    options: &PeriodicScfOptions,
) -> Result<[[f64; 3]; 3], PeriodicError> {
    assert_basis_atoms_align(basis, atoms);

    let state = ConvergedState::converge(basis, cell, kpoints, n_elec, atoms, xc, options)?;
    let collocator = LatticeCollocator::new(basis, &state.grid);
    let phases = collocator.bloch_phases(&state.kfracs, &state.weights);

    let mut tau = [[0.0_f64; 3]; 3];
    let add = |tau: &mut [[f64; 3]; 3], t: [[f64; 3]; 3], s: f64| {
        for (ta, sa) in tau.iter_mut().zip(&t) {
            for (x, &y) in ta.iter_mut().zip(sa) {
                *x += s * y;
            }
        }
    };
    add(
        &mut tau,
        bloch_kinetic_stress_contract(
            basis,
            cell,
            &state.kfracs,
            &state.weights,
            &state.p_k,
            state.rmax,
        ),
        1.0,
    );
    add(
        &mut tau,
        bloch_overlap_stress_contract(
            basis,
            cell,
            &state.kfracs,
            &state.weights,
            &state.w_k,
            state.rmax,
        ),
        -1.0,
    );
    add(
        &mut tau,
        collocator.collocation_pulay_stress(&state.grid, &state.p_k, &state.v_loc_grid, &phases),
        1.0,
    );
    add(
        &mut tau,
        core_charge_stress(&state.grid, &state.local_atoms, &state.v_h),
        1.0,
    );
    add(
        &mut tau,
        local_sr_stress(&state.grid, &state.local_atoms, &state.n_r),
        1.0,
    );
    add(
        &mut tau,
        projector_stress(
            basis,
            cell,
            atoms,
            &state.kfracs,
            &state.weights,
            &state.p_k,
            state.rmax,
        ),
        1.0,
    );
    add(&mut tau, overlap_stress(cell, &state.local_atoms), 1.0);
    add(
        &mut tau,
        hartree_reciprocal_stress(&state.rho_tot, &state.grid),
        1.0,
    );
    let metric = state.components.e_hartree + state.components.e_xc + state.components.e_local_sr;
    for (a, ta) in tau.iter_mut().enumerate() {
        ta[a] += metric;
    }

    let omega = cell.volume();
    let mut sigma = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            sigma[a][b] = (tau[a][b] + tau[b][a]) / (2.0 * omega);
        }
    }
    Ok(sigma)
}

#[must_use]
pub fn projector_stress(
    basis: &Basis,
    cell: &Cell,
    atoms: &[PeriodicAtom],
    kfracs: &[[f64; 3]],
    weights: &[f64],
    p_k: &[Vec<C64>],
    rmax: f64,
) -> [[f64; 3]; 3] {
    let n = basis.nao();
    let mut tau = [[0.0_f64; 3]; 3];
    for a in atoms {
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
                let (b, db_eps) = bloch_projector_overlaps_strain(basis, cell, &chan, kf, rmax);
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
                for mu in 0..n {
                    for p in 0..n_proj {
                        for m in 0..nlm {
                            let mut phi = C64::new(0.0, 0.0);
                            for (q, &hpq) in ch.h[p].iter().enumerate() {
                                if hpq == 0.0 {
                                    continue;
                                }
                                phi += C64::new(hpq, 0.0) * pb[mu * ncol + q * nlm + m].conj();
                            }
                            let idx = mu * ncol + (p * nlm + m);
                            for (alpha, ta) in tau.iter_mut().enumerate() {
                                for (beta, tab) in ta.iter_mut().enumerate() {
                                    *tab += wk * 2.0 * (db_eps[alpha][beta][idx] * phi).re;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    tau
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_strain_matches_quadratic() {
        let cell = Cell::cubic(6.0).unwrap();
        let positions = [[1.0, 0.5, -0.3], [2.0, -1.0, 0.7]];
        let m = [[0.3, 0.1, -0.2], [0.1, -0.4, 0.05], [-0.2, 0.05, 0.6]];
        let energy = |_c: &Cell, p: &[[f64; 3]]| -> Result<f64, PeriodicError> {
            Ok(p.iter()
                .map(|r| r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
                .sum())
        };
        let fd = finite_difference_strain_energy(&cell, &positions, &m, 1e-5, energy).unwrap();
        let mut analytic = 0.0;
        for r in &positions {
            for a in 0..3 {
                for b in 0..3 {
                    analytic += 2.0 * r[a] * m[a][b] * r[b];
                }
            }
        }
        assert!(
            (fd - analytic).abs() < 1e-6,
            "fd {fd} vs analytic {analytic}"
        );
    }

    use crate::basis::GthSet;
    use crate::integrals::integral::Shell;
    use crate::periodic::NonlocalChannel;
    use crate::periodic::scf::{build_vnl_k, trace_re};

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
    fn projector_stress_matches_finite_difference() {
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let r0 = cell.frac_to_cart([0.05, 0.02, -0.03]);
        let r1 = cell.frac_to_cart([0.27, 0.24, 0.26]);
        let positions = [r0, r1];
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.5, 0.5];
        let rmax = 9.0;

        let build = |_c: &Cell, p: &[[f64; 3]]| {
            let mut shells = si_szv_shells(p[0]);
            shells.extend(si_szv_shells(p[1]));
            (Basis::new(shells), vec![si_atom(p[0]), si_atom(p[1])])
        };
        let (basis, atoms) = build(&cell, &positions);
        let n = basis.nao();

        let mut p = vec![C64::new(0.0, 0.0); n * n];
        for a in 0..n {
            for b in 0..n {
                let v = 0.03 * ((a + 1) as f64) + 0.02 * ((b + 1) as f64) + 0.01 * (a * b) as f64;
                p[a * n + b] = C64::new(v, 0.0);
            }
        }
        for a in 0..n {
            for b in (a + 1)..n {
                let s = 0.5 * (p[a * n + b] + p[b * n + a]);
                p[a * n + b] = s;
                p[b * n + a] = s;
            }
        }
        let p_k = vec![p.clone(), p];

        let tau = projector_stress(&basis, &cell, &atoms, &kfracs, &weights, &p_k, rmax);

        let e_nl = |c: &Cell, pos: &[[f64; 3]]| -> Result<f64, PeriodicError> {
            let (b, at) = build(c, pos);
            let mut e = 0.0;
            for (ik, (&kf, &wk)) in kfracs.iter().zip(&weights).enumerate() {
                let vnl = build_vnl_k(&b, c, &at, kf, rmax);
                e += wk * trace_re(&vnl, &p_k[ik], n);
            }
            Ok(e)
        };
        let dirs = [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.2, -0.1], [0.2, -0.3, 0.15], [-0.1, 0.15, 0.5]],
        ];
        for m in dirs {
            let fd = finite_difference_strain_energy(&cell, &positions, &m, 1e-4, e_nl).unwrap();
            let analytic: f64 = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| tau[a][b] * m[a][b])
                .sum();
            assert!(
                (analytic - fd).abs() < 1e-6,
                "strain {m:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    use crate::periodic::{GridXc, PeriodicScfOptions, run_scf_periodic};
    use latx::KPoint;

    #[test]
    #[ignore = "stress validation: run manually (minutes; FD over 6 strains = 12 SCFs)"]
    fn periodic_stress_matches_finite_difference_si() {
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.27, 0.24, 0.26]); // off the ideal ¼¼¼ site
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
        let build = |p: &[[f64; 3]]| {
            let mut shells = si_szv_shells(p[0]);
            shells.extend(si_szv_shells(p[1]));
            (Basis::new(shells), vec![si_atom(p[0]), si_atom(p[1])])
        };
        let (basis, atoms) = build(&positions);
        let sigma = periodic_stress(&basis, &cell, &kpoints, 8, &atoms, &xc, &options).unwrap();

        let omega = cell.volume();
        for m in [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]],
        ] {
            let fd = finite_difference_strain_energy(&cell, &positions, &m, 1e-3, |c, p| {
                let (b, at) = build(p);
                Ok(run_scf_periodic(&b, c, &kpoints, 8, &at, &xc, &options)?.energy)
            })
            .unwrap();
            let analytic = omega * contract_sym(&sigma, &m);
            eprintln!("[Si-stress] strain {m:?}: analytic {analytic:.8} vs FD {fd:.8}");
            assert!(
                (analytic - fd).abs() < 1e-3,
                "strain {m:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    fn contract_sym(s: &[[f64; 3]; 3], m: &[[f64; 3]; 3]) -> f64 {
        (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| s[a][b] * m[a][b])
            .sum()
    }
}
