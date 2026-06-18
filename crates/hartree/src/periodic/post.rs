use crate::integrals::integral::Basis;
use crate::integrals::integral::periodic::{LatticeCollocator, bloch_kinetic, bloch_overlap};
use crate::linalg::{C64, cmat_from_row_major, hermitian_geneig};
use latx::{Cell, KPoint};

use crate::periodic::PeriodicError;
use crate::periodic::converged::ConvergedState;
use crate::periodic::scf::{PeriodicAtom, PeriodicScfOptions, build_vnl_k};
use crate::periodic::xc::GridXc;

fn eigenvalues_at(
    basis: &Basis,
    cell: &Cell,
    atoms: &[PeriodicAtom],
    state: &ConvergedState,
    kpts: &[KPoint],
) -> Vec<Vec<f64>> {
    let n = basis.nao();
    let collocator = LatticeCollocator::new(basis, &state.grid);
    let kfracs: Vec<[f64; 3]> = kpts.iter().map(|k| k.frac).collect();
    let weights: Vec<f64> = kpts.iter().map(|k| k.weight).collect();
    let phases = collocator.bloch_phases(&kfracs, &weights);
    let v_loc_k = collocator.integrate_k(&state.grid, &state.v_loc_grid, &phases);

    kpts.iter()
        .enumerate()
        .map(|(ik, k)| {
            let t = bloch_kinetic(basis, cell, k.frac, state.rmax);
            let vnl = build_vnl_k(basis, cell, atoms, k.frac, state.rmax);
            let s = bloch_overlap(basis, cell, k.frac, state.rmax);
            let h: Vec<C64> = (0..n * n).map(|i| t[i] + vnl[i] + v_loc_k[ik][i]).collect();
            let eig = hermitian_geneig(&cmat_from_row_major(n, &h), &cmat_from_row_major(n, &s));
            eig.values
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct BandStructure {
    pub kpoints: Vec<KPoint>,
    pub bands: Vec<Vec<f64>>,
    pub n_occ: usize,
}

impl BandStructure {
    #[must_use]
    pub fn vbm(&self) -> Option<f64> {
        if self.n_occ == 0 {
            return None;
        }
        self.bands
            .iter()
            .map(|b| b[self.n_occ - 1])
            .reduce(f64::max)
    }

    #[must_use]
    pub fn cbm(&self) -> Option<f64> {
        self.bands
            .iter()
            .filter_map(|b| b.get(self.n_occ).copied())
            .reduce(f64::min)
    }

    #[must_use]
    pub fn gap(&self) -> Option<f64> {
        Some(self.cbm()? - self.vbm()?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn band_structure(
    basis: &Basis,
    cell: &Cell,
    scf_kpoints: &[KPoint],
    n_elec: usize,
    atoms: &[PeriodicAtom],
    xc: &GridXc,
    options: &PeriodicScfOptions,
    path: &[KPoint],
) -> Result<BandStructure, PeriodicError> {
    let state = ConvergedState::converge(basis, cell, scf_kpoints, n_elec, atoms, xc, options)?;
    let bands = eigenvalues_at(basis, cell, atoms, &state, path);
    Ok(BandStructure {
        kpoints: path.to_vec(),
        bands,
        n_occ: n_elec / 2,
    })
}

#[derive(Debug, Clone)]
pub struct Dos {
    pub energies: Vec<f64>,
    pub dos: Vec<f64>,
    pub n_elec: usize,
}

impl Dos {
    #[must_use]
    pub fn integral(&self) -> f64 {
        if self.energies.len() < 2 {
            return 0.0;
        }
        self.energies
            .windows(2)
            .zip(self.dos.windows(2))
            .map(|(e, d)| 0.5 * (e[1] - e[0]) * (d[0] + d[1]))
            .sum()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn density_of_states(
    basis: &Basis,
    cell: &Cell,
    scf_kpoints: &[KPoint],
    n_elec: usize,
    atoms: &[PeriodicAtom],
    xc: &GridXc,
    options: &PeriodicScfOptions,
    eval_kpoints: &[KPoint],
    sigma: f64,
    n_points: usize,
) -> Result<Dos, PeriodicError> {
    assert!(sigma > 0.0, "DOS broadening sigma must be positive");
    assert!(n_points >= 2, "DOS needs at least 2 energy points");
    assert!(
        n_elec.is_multiple_of(2),
        "spin-restricted DOS needs an even electron count"
    );
    let state = ConvergedState::converge(basis, cell, scf_kpoints, n_elec, atoms, xc, options)?;
    let bands = eigenvalues_at(basis, cell, atoms, &state, eval_kpoints);
    let n_occ = n_elec / 2;

    let mut emin = f64::INFINITY;
    let mut emax = f64::NEG_INFINITY;
    for b in &bands {
        for &e in b.iter().take(n_occ) {
            emin = emin.min(e);
            emax = emax.max(e);
        }
    }
    assert!(
        emin.is_finite() && emax.is_finite(),
        "DOS needs at least one occupied band"
    );
    let pad = 5.0 * sigma;
    let (lo, hi) = (emin - pad, emax + pad);
    let de = (hi - lo) / (n_points - 1) as f64;
    let energies: Vec<f64> = (0..n_points).map(|i| lo + de * i as f64).collect();

    let norm = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let mut dos = vec![0.0; n_points];
    for (ik, b) in bands.iter().enumerate() {
        let wk = eval_kpoints[ik].weight;
        for &e in b.iter().take(n_occ) {
            for (j, &eg) in energies.iter().enumerate() {
                let x = (eg - e) / sigma;
                dos[j] += 2.0 * wk * norm * (-0.5 * x * x).exp();
            }
        }
    }
    Ok(Dos {
        energies,
        dos,
        n_elec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{GthBasisSet, GthSet};
    use crate::core::{Atom, Element, Molecule};
    use crate::periodic::run_scf_periodic;
    use crate::periodic::system::PeriodicSystem;

    fn si2() -> (Basis, Vec<PeriodicAtom>, Cell) {
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let si = Element::from_symbol("Si").unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.25, 0.25, 0.25]);
        let mol = Molecule::new(vec![Atom::new(si, r0), Atom::new(si, r1)], 0, 1);
        let bset = GthBasisSet::load_pade().unwrap();
        let pset = GthSet::load_pade().unwrap();
        let sys = PeriodicSystem::build(&mol, "SZV-GTH", &bset, &pset).unwrap();
        (sys.basis, sys.atoms, cell)
    }

    fn opts() -> PeriodicScfOptions {
        PeriodicScfOptions {
            e_cut: 80.0,
            max_iter: 100,
            energy_tol: 1e-8,
            density_tol: 1e-7,
            ..Default::default()
        }
    }

    #[test]
    fn band_gamma_matches_scf() {
        let (basis, atoms, cell) = si2();
        let xc = GridXc::pade();
        let o = opts();
        let gamma = [KPoint::gamma()];
        let scf = run_scf_periodic(&basis, &cell, &gamma, 8, &atoms, &xc, &o).unwrap();
        let bs = band_structure(&basis, &cell, &gamma, 8, &atoms, &xc, &o, &gamma).unwrap();
        assert_eq!(bs.n_occ, 4);
        assert_eq!(bs.bands[0].len(), basis.nao());
        for (a, b) in bs.bands[0].iter().zip(&scf.band_energies[0]) {
            assert!((a - b).abs() < 1e-5, "band {a} vs SCF {b}");
        }
    }

    #[test]
    fn band_path_runs() {
        let (basis, atoms, cell) = si2();
        let xc = GridXc::pade();
        let o = opts();
        let path = [
            KPoint::new([0.0, 0.0, 0.0], 1.0),
            KPoint::new([0.25, 0.0, 0.0], 1.0),
            KPoint::new([0.5, 0.0, 0.0], 1.0),
        ];
        let bs =
            band_structure(&basis, &cell, &[KPoint::gamma()], 8, &atoms, &xc, &o, &path).unwrap();
        assert_eq!(bs.bands.len(), 3);
        for b in &bs.bands {
            assert!(
                b.windows(2).all(|w| w[0] <= w[1] + 1e-9),
                "eigenvalues ascending"
            );
        }
        assert!(bs.gap().unwrap().is_finite());
    }

    #[test]
    fn dos_integrates_to_electron_count() {
        let (basis, atoms, cell) = si2();
        let xc = GridXc::pade();
        let o = opts();
        let gamma = [KPoint::gamma()];
        let dos = density_of_states(
            &basis, &cell, &gamma, 8, &atoms, &xc, &o, &gamma, 0.01, 4000,
        )
        .unwrap();
        let integral = dos.integral();
        let expected = 8.0;
        assert!(
            (integral - expected).abs() < 0.01 * expected,
            "∫DOS = {integral} vs {expected}"
        );
        assert!(dos.dos.iter().all(|&d| d >= 0.0));
        assert_eq!(dos.n_elec, 8);
    }
}
