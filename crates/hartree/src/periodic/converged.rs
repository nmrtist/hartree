use crate::integrals::integral::Basis;
use crate::integrals::integral::periodic::{
    LatticeCollocator, RealSpaceGrid, bloch_kinetic, bloch_overlap, hartree,
};
use crate::linalg::{C64, cmat_from_row_major, hermitian_geneig};
use latx::{Cell, KPoint};

use crate::periodic::PeriodicError;
use crate::periodic::pseudo::{GthLocalAtom, core_charge_density, local_pp_short_range};
use crate::periodic::scf::{
    EnergyComponents, PeriodicAtom, PeriodicScfOptions, build_vnl_k, default_bloch_rmax,
    run_scf_periodic,
};
use crate::periodic::xc::GridXc;

pub(crate) fn assert_basis_atoms_align(basis: &Basis, atoms: &[PeriodicAtom]) {
    let bat = basis.atoms();
    assert_eq!(
        bat.len(),
        atoms.len(),
        "basis atoms vs pseudo atoms count mismatch"
    );
    for (i, a) in atoms.iter().enumerate() {
        let d = (0..3)
            .map(|x| (bat[i][x] - a.center[x]).powi(2))
            .sum::<f64>();
        assert!(
            d < 1e-12,
            "basis atom {i} does not match pseudo-atom center"
        );
    }
}

pub(crate) struct ConvergedState {
    pub grid: RealSpaceGrid,
    pub rmax: f64,
    pub kfracs: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub local_atoms: Vec<GthLocalAtom>,
    pub n_r: Vec<f64>,
    pub rho_tot: Vec<f64>,
    pub v_h: Vec<f64>,
    pub v_loc_grid: Vec<f64>,
    pub p_k: Vec<Vec<C64>>,
    pub w_k: Vec<Vec<C64>>,
    pub components: EnergyComponents,
}

impl ConvergedState {
    pub(crate) fn converge(
        basis: &Basis,
        cell: &Cell,
        kpoints: &[KPoint],
        n_elec: usize,
        atoms: &[PeriodicAtom],
        xc: &GridXc,
        options: &PeriodicScfOptions,
    ) -> Result<Self, PeriodicError> {
        assert!(
            n_elec.is_multiple_of(2),
            "spin-restricted SCF needs an even electron count"
        );
        let n = basis.nao();
        assert!(n > 0, "empty basis");
        let n_occ = n_elec / 2;

        let res = run_scf_periodic(basis, cell, kpoints, n_elec, atoms, xc, options)?;
        let n_r = res.density;

        let rmax = options
            .bloch_rmax
            .unwrap_or_else(|| default_bloch_rmax(basis));
        let grid = RealSpaceGrid::from_cutoff(*cell, options.e_cut);
        let collocator = LatticeCollocator::new(basis, &grid);
        let dv = grid.dv();

        let local_atoms: Vec<GthLocalAtom> = atoms
            .iter()
            .map(|a| GthLocalAtom {
                center: a.center,
                z_ion: a.z_ion,
                r_loc: a.r_loc,
                c: a.c.clone(),
            })
            .collect();
        let rho_core = core_charge_density(&grid, &local_atoms);
        let v_loc_sr = local_pp_short_range(&grid, &local_atoms);

        let kfracs: Vec<[f64; 3]> = kpoints.iter().map(|k| k.frac).collect();
        let weights: Vec<f64> = kpoints.iter().map(|k| k.weight).collect();
        let phases = collocator.bloch_phases(&kfracs, &weights);

        let s_k: Vec<Vec<C64>> = kpoints
            .iter()
            .map(|k| bloch_overlap(basis, cell, k.frac, rmax))
            .collect();
        let t_k: Vec<Vec<C64>> = kpoints
            .iter()
            .map(|k| bloch_kinetic(basis, cell, k.frac, rmax))
            .collect();
        let vnl_k: Vec<Vec<C64>> = kpoints
            .iter()
            .map(|k| build_vnl_k(basis, cell, atoms, k.frac, rmax))
            .collect();

        let rho_tot: Vec<f64> = n_r.iter().zip(&rho_core).map(|(&a, &b)| a + b).collect();
        let (v_h, _e_h) = hartree(&rho_tot, &grid);
        let (_e_xc, v_xc) = xc.energy_potential(&n_r, dv)?;
        let v_loc_grid: Vec<f64> = (0..v_h.len())
            .map(|g| v_h[g] + v_xc[g] + v_loc_sr[g])
            .collect();
        let v_loc_k = collocator.integrate_k(&grid, &v_loc_grid, &phases);

        let mut p_k: Vec<Vec<C64>> = Vec::with_capacity(kpoints.len());
        let mut w_k: Vec<Vec<C64>> = Vec::with_capacity(kpoints.len());
        for ik in 0..kpoints.len() {
            let h: Vec<C64> = (0..n * n)
                .map(|i| t_k[ik][i] + vnl_k[ik][i] + v_loc_k[ik][i])
                .collect();
            let eig = hermitian_geneig(
                &cmat_from_row_major(n, &h),
                &cmat_from_row_major(n, &s_k[ik]),
            );
            let mut pk = vec![C64::new(0.0, 0.0); n * n];
            let mut wk = vec![C64::new(0.0, 0.0); n * n];
            for i in 0..n_occ {
                let eps = eig.values[i];
                for mu in 0..n {
                    let cmu = eig.vectors[(mu, i)];
                    if cmu == C64::new(0.0, 0.0) {
                        continue;
                    }
                    let wp = C64::new(2.0, 0.0) * cmu;
                    let we = C64::new(2.0 * eps, 0.0) * cmu;
                    let (prow, wrow) = (&mut pk[mu * n..mu * n + n], &mut wk[mu * n..mu * n + n]);
                    for nu in 0..n {
                        let cnu = eig.vectors[(nu, i)].conj();
                        prow[nu] += wp * cnu;
                        wrow[nu] += we * cnu;
                    }
                }
            }
            p_k.push(pk);
            w_k.push(wk);
        }

        Ok(Self {
            grid,
            rmax,
            kfracs,
            weights,
            local_atoms,
            n_r,
            rho_tot,
            v_h,
            v_loc_grid,
            p_k,
            w_k,
            components: res.components,
        })
    }
}
