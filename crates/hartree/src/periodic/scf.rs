use crate::integrals::integral::Basis;
use crate::integrals::integral::periodic::{
    ChiCache, LatticeCollocator, ProjectorChannel, RealSpaceGrid, bloch_kinetic, bloch_overlap,
    bloch_projector_overlaps, hartree,
};
use crate::linalg::{C64, cmat_from_row_major, hermitian_geneig};
use latx::{Cell, KPoint};

use crate::periodic::PeriodicError;
use crate::periodic::pseudo::{
    GthLocalAtom, core_charge_density, local_pp_short_range, overlap_energy, self_energy,
};
use crate::periodic::xc::GridXc;

#[derive(Debug, Clone)]
pub struct NonlocalChannel {
    pub l: usize,
    pub r_l: f64,
    pub h: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct PeriodicAtom {
    pub center: [f64; 3],
    pub z_ion: f64,
    pub r_loc: f64,
    pub c: Vec<f64>,
    pub channels: Vec<NonlocalChannel>,
}

#[derive(Debug, Clone)]
pub struct PeriodicScfOptions {
    pub e_cut: f64,
    pub max_iter: usize,
    pub energy_tol: f64,
    pub density_tol: f64,
    pub mixing: f64,
    pub bloch_rmax: Option<f64>,
    pub cache_chi: bool,
}

pub const CHI_CACHE_MAX_BYTES: usize = 6 << 30;

impl Default for PeriodicScfOptions {
    fn default() -> Self {
        Self {
            e_cut: 280.0,
            max_iter: 100,
            energy_tol: 1e-7,
            density_tol: 1e-6,
            mixing: 0.3,
            bloch_rmax: None,
            cache_chi: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EnergyComponents {
    pub e_kin: f64,
    pub e_hartree: f64,
    pub e_xc: f64,
    pub e_local_sr: f64,
    pub e_nonlocal: f64,
    pub e_self: f64,
    pub e_overlap: f64,
}

impl EnergyComponents {
    #[must_use]
    pub fn total(&self) -> f64 {
        self.e_kin
            + self.e_hartree
            + self.e_xc
            + self.e_local_sr
            + self.e_nonlocal
            + self.e_self
            + self.e_overlap
    }
}

#[derive(Debug, Clone)]
pub struct PeriodicScfResult {
    pub energy: f64,
    pub components: EnergyComponents,
    pub converged: bool,
    pub iterations: usize,
    pub n_elec_grid: f64,
    pub band_energies: Vec<Vec<f64>>,
    pub density: Vec<f64>,
}

pub(crate) fn default_bloch_rmax(basis: &Basis) -> f64 {
    let alpha_min = basis
        .shells()
        .iter()
        .flat_map(|s| s.exponents().iter().copied())
        .filter(|&a| a > 0.0)
        .fold(f64::INFINITY, f64::min);
    if alpha_min.is_finite() {
        (2.0 * 36.0 / alpha_min).sqrt()
    } else {
        12.0
    }
}

pub(crate) fn build_vnl_k(
    basis: &Basis,
    cell: &Cell,
    atoms: &[PeriodicAtom],
    k_frac: [f64; 3],
    rmax: f64,
) -> Vec<C64> {
    let n = basis.nao();
    let mut vnl = vec![C64::new(0.0, 0.0); n * n];
    for a in atoms {
        for ch in &a.channels {
            let n_proj = ch.h.len();
            if n_proj == 0 {
                continue;
            }
            let nlm = 2 * ch.l + 1;
            let chan = ProjectorChannel {
                center: a.center,
                l: ch.l,
                n_proj,
                r_l: ch.r_l,
            };
            let b = bloch_projector_overlaps(basis, cell, &chan, k_frac, rmax);
            let ncol = n_proj * nlm;
            for m in 0..nlm {
                for i in 0..n_proj {
                    let ci = i * nlm + m;
                    for j in 0..n_proj {
                        let hij = ch.h[i][j];
                        if hij == 0.0 {
                            continue;
                        }
                        let cj = j * nlm + m;
                        for mu in 0..n {
                            let bmu = b[mu * ncol + ci];
                            if bmu == C64::new(0.0, 0.0) {
                                continue;
                            }
                            let w = hij * bmu;
                            let row = &mut vnl[mu * n..mu * n + n];
                            for (nu, vij) in row.iter_mut().enumerate() {
                                *vij += w * b[nu * ncol + cj].conj();
                            }
                        }
                    }
                }
            }
        }
    }
    vnl
}

pub(crate) fn trace_re(a: &[C64], p: &[C64], n: usize) -> f64 {
    let mut acc = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            acc += (a[mu * n + nu] * p[nu * n + mu]).re;
        }
    }
    acc
}

#[allow(clippy::too_many_lines)]
pub fn run_scf_periodic(
    basis: &Basis,
    cell: &Cell,
    kpoints: &[KPoint],
    n_elec: usize,
    atoms: &[PeriodicAtom],
    xc: &GridXc,
    options: &PeriodicScfOptions,
) -> Result<PeriodicScfResult, PeriodicError> {
    assert!(
        n_elec.is_multiple_of(2),
        "spin-restricted SCF needs an even electron count"
    );
    let n = basis.nao();
    assert!(n > 0, "empty basis");
    let n_occ = n_elec / 2;
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
    let e_self = self_energy(&local_atoms);
    let e_overlap = overlap_energy(cell, &local_atoms);

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
    let h_fixed: Vec<Vec<C64>> = (0..kpoints.len())
        .map(|ik| {
            t_k[ik]
                .iter()
                .zip(&vnl_k[ik])
                .map(|(&a, &b)| a + b)
                .collect::<Vec<_>>()
        })
        .collect();

    let kfracs: Vec<[f64; 3]> = kpoints.iter().map(|k| k.frac).collect();
    let weights: Vec<f64> = kpoints.iter().map(|k| k.weight).collect();
    let phases = collocator.bloch_phases(&kfracs, &weights);
    let verbose = std::env::var("HARTREE_PERIODIC_VERBOSE").is_ok();
    if verbose {
        eprintln!(
            "[periodic] grid {:?} = {} pts, nao {}, k-points {}, bloch_rmax {:.1}",
            grid.n(),
            grid.n_points(),
            n,
            kpoints.len(),
            rmax
        );
    }

    let chi_bytes = grid
        .n_points()
        .saturating_mul(n)
        .saturating_mul(kpoints.len())
        * 16;
    let chi_cache: Option<ChiCache> = if options.cache_chi && chi_bytes <= CHI_CACHE_MAX_BYTES {
        Some(collocator.build_chi_cache(&grid, &phases))
    } else {
        None
    };
    if verbose {
        match &chi_cache {
            Some(_) => eprintln!("[periodic] χ-cache on (~{} MB)", chi_bytes >> 20),
            None if options.cache_chi => {
                eprintln!("[periodic] χ-cache off (~{} MB > cap)", chi_bytes >> 20);
            }
            None => eprintln!("[periodic] χ-cache off (disabled)"),
        }
    }

    let mut n_r = vec![0.0; grid.n_points()]; // density guess: bare cores (n = 0)
    let mut prev_energy = 0.0;
    let mut converged = false;
    let mut iterations = 0;
    let mut last_components = EnergyComponents {
        e_kin: 0.0,
        e_hartree: 0.0,
        e_xc: 0.0,
        e_local_sr: 0.0,
        e_nonlocal: 0.0,
        e_self,
        e_overlap,
    };
    let mut last_bands: Vec<Vec<f64>> = Vec::new();
    let mut last_nelec = 0.0;
    let mut last_density: Vec<f64> = Vec::new();

    for iter in 1..=options.max_iter {
        iterations = iter;
        let t_iter = std::time::Instant::now();

        let rho_tot: Vec<f64> = n_r.iter().zip(&rho_core).map(|(&a, &b)| a + b).collect();
        let (v_h, _e_h_in) = hartree(&rho_tot, &grid);
        let (_e_xc_in, v_xc) = xc.energy_potential(&n_r, dv)?;
        let v_loc_grid: Vec<f64> = (0..v_h.len())
            .map(|g| v_h[g] + v_xc[g] + v_loc_sr[g])
            .collect();
        let v_loc_k = match &chi_cache {
            Some(c) => collocator.integrate_k_cached(c, &grid, &v_loc_grid),
            None => collocator.integrate_k(&grid, &v_loc_grid, &phases),
        };

        let mut p_k: Vec<Vec<C64>> = Vec::with_capacity(kpoints.len());
        let mut bands: Vec<Vec<f64>> = Vec::with_capacity(kpoints.len());
        let mut e_kin = 0.0;
        let mut e_nonlocal = 0.0;
        for (ik, kp) in kpoints.iter().enumerate() {
            let h: Vec<C64> = (0..n * n)
                .map(|i| h_fixed[ik][i] + v_loc_k[ik][i])
                .collect();
            let hm = cmat_from_row_major(n, &h);
            let sm = cmat_from_row_major(n, &s_k[ik]);
            let eig = hermitian_geneig(&hm, &sm);
            let mut pk = vec![C64::new(0.0, 0.0); n * n];
            for i in 0..n_occ {
                for mu in 0..n {
                    let cmu = eig.vectors[(mu, i)];
                    if cmu == C64::new(0.0, 0.0) {
                        continue;
                    }
                    let w = C64::new(2.0, 0.0) * cmu;
                    let row = &mut pk[mu * n..mu * n + n];
                    for (nu, p) in row.iter_mut().enumerate() {
                        *p += w * eig.vectors[(nu, i)].conj();
                    }
                }
            }
            e_kin += kp.weight * trace_re(&t_k[ik], &pk, n);
            e_nonlocal += kp.weight * trace_re(&vnl_k[ik], &pk, n);
            bands.push(eig.values);
            p_k.push(pk);
        }

        let n_out = match &chi_cache {
            Some(c) => collocator.collocate_k_cached(c, &p_k),
            None => collocator.collocate_k(&grid, &p_k, &phases),
        };
        last_nelec = n_out.iter().sum::<f64>() * dv;
        last_density.clone_from(&n_out);

        let rho_tot_out: Vec<f64> = n_out.iter().zip(&rho_core).map(|(&a, &b)| a + b).collect();
        let (_v_h_out, e_hartree) = hartree(&rho_tot_out, &grid);
        let (e_xc, _v) = xc.energy_potential(&n_out, dv)?;
        let e_local_sr = dv
            * n_out
                .iter()
                .zip(&v_loc_sr)
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
        let components = EnergyComponents {
            e_kin,
            e_hartree,
            e_xc,
            e_local_sr,
            e_nonlocal,
            e_self,
            e_overlap,
        };
        let energy = components.total();

        let dn = n_out
            .iter()
            .zip(&n_r)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);
        let de = (energy - prev_energy).abs();
        if verbose {
            eprintln!(
                "[periodic] iter {iter:3}  E = {energy:.8}  dE = {de:.2e}  dn = {dn:.2e}  N = {last_nelec:.4}  ({:.1}s)",
                t_iter.elapsed().as_secs_f64()
            );
        }
        last_components = components;
        last_bands = bands;
        if iter > 1 && de < options.energy_tol && dn < options.density_tol {
            converged = true;
            prev_energy = energy;
            break;
        }
        prev_energy = energy;
        let a = options.mixing;
        for (ni, &no) in n_r.iter_mut().zip(&n_out) {
            *ni = (1.0 - a) * *ni + a * no;
        }
    }

    Ok(PeriodicScfResult {
        energy: prev_energy,
        components: last_components,
        converged,
        iterations,
        n_elec_grid: last_nelec,
        band_energies: last_bands,
        density: last_density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::GthSet;
    use crate::integrals::integral::Shell;
    use latx::KPoint;

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

    fn si_dzvp_shells(center: [f64; 3]) -> Vec<Shell> {
        let exps = vec![1.2032422345, 0.4688409786, 0.1679863234, 0.0575619526];
        let s = vec![0.3290350445, -0.2533118323, -0.7870946277, -0.1909898479];
        let p = vec![0.0474539126, -0.2594473573, -0.5440929303, -0.3624010364];
        vec![
            Shell::new_spherical(0, center, exps.clone(), s).unwrap(),
            Shell::new_spherical(0, center, vec![0.0575619526], vec![1.0]).unwrap(),
            Shell::new_spherical(1, center, exps, p).unwrap(),
            Shell::new_spherical(1, center, vec![0.0575619526], vec![1.0]).unwrap(),
            Shell::new_spherical(2, center, vec![0.45], vec![1.0]).unwrap(),
        ]
    }

    #[test]
    #[ignore = "bulk Si validation: run manually; default 150 Ry, 300 to match CP2K exactly"]
    fn bulk_si_8atom_dzvp_vs_cp2k() {
        let a = 5.430_697_5 * 1.889_726_988_6; // Å → bohr = 10.26254 bohr.
        let cell = Cell::cubic(a).unwrap();
        let frac = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ];
        let centers: Vec<[f64; 3]> = frac.iter().map(|&f| cell.frac_to_cart(f)).collect();
        let mut shells = Vec::new();
        let mut atoms = Vec::new();
        for &c in &centers {
            shells.extend(si_dzvp_shells(c));
            atoms.push(si_atom(c));
        }
        let basis = Basis::new(shells);
        let xc = GridXc::pade();
        let kpoints = [KPoint::gamma()];
        let e_cut = std::env::var("HARTREE_SI_CUTOFF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(150.0);
        let options = PeriodicScfOptions {
            e_cut,
            max_iter: 100,
            energy_tol: 1e-6,
            density_tol: 1e-5,
            mixing: 0.3,
            bloch_rmax: None,
            cache_chi: true,
        };
        let r = run_scf_periodic(&basis, &cell, &kpoints, 32, &atoms, &xc, &options).unwrap();
        eprintln!(
            "[Si-8atom] E = {:.6} Ha ({:.6}/atom), converged {}, iters {}, N {:.4}\n  {:?}",
            r.energy,
            r.energy / 8.0,
            r.converged,
            r.iterations,
            r.n_elec_grid,
            r.components
        );
        assert!(r.converged, "SCF did not converge");
        assert!((r.n_elec_grid - 32.0).abs() < 0.05, "N = {}", r.n_elec_grid);
        assert!(
            (r.energy - (-31.297820)).abs() < 1e-3,
            "E = {} Ha vs CP2K −31.297820 (±1 mHa)",
            r.energy
        );
    }

    #[test]
    #[ignore = "k-point validation: run manually (minutes, 8 complex k-points)"]
    fn bulk_si_primitive_monkhorst_pack() {
        use latx::MonkhorstPack;
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.25, 0.25, 0.25]);
        let mut shells = si_szv_shells(r0);
        shells.extend(si_szv_shells(r1));
        let basis = Basis::new(shells);
        let atoms = [si_atom(r0), si_atom(r1)];
        let xc = GridXc::pade();
        let options = PeriodicScfOptions {
            e_cut: 80.0,
            max_iter: 80,
            energy_tol: 1e-6,
            density_tol: 1e-5,
            mixing: 0.3,
            bloch_rmax: None,
            cache_chi: true,
        };

        let gamma =
            run_scf_periodic(&basis, &cell, &[KPoint::gamma()], 8, &atoms, &xc, &options).unwrap();
        let mp = MonkhorstPack::regular([2, 2, 2]).unwrap();
        assert!(!mp.iter().any(KPoint::is_gamma), "2×2×2 MP excludes Γ");
        let res = run_scf_periodic(&basis, &cell, &mp, 8, &atoms, &xc, &options).unwrap();
        eprintln!(
            "[Si-MP] Γ E/atom = {:.6}, 2×2×2 E/atom = {:.6} ({} k-pts), converged {}, N {:.4}",
            gamma.energy / 2.0,
            res.energy / 2.0,
            mp.len(),
            res.converged,
            res.n_elec_grid
        );
        assert!(res.converged, "MP SCF did not converge");
        assert!(res.energy.is_finite() && res.energy < 0.0);
        assert!(
            (res.n_elec_grid - 8.0).abs() < 1e-2,
            "N = {}",
            res.n_elec_grid
        );
        assert!(
            (res.energy - gamma.energy).abs() > 1e-4,
            "MP energy {} indistinguishable from Γ {}",
            res.energy,
            gamma.energy
        );
    }

    #[test]
    fn bulk_si_primitive_scf_converges() {
        let a = 10.263; // Si lattice constant (bohr), ≈ 5.431 Å.
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.25, 0.25, 0.25]);

        let mut shells = si_szv_shells(r0);
        shells.extend(si_szv_shells(r1));
        let basis = Basis::new(shells);
        let atoms = [si_atom(r0), si_atom(r1)];

        let xc = GridXc::pade();
        let kpoints = [KPoint::gamma()];
        let options = PeriodicScfOptions {
            e_cut: 100.0,
            max_iter: 80,
            energy_tol: 1e-6,
            density_tol: 1e-5,
            mixing: 0.3,
            bloch_rmax: None,
            cache_chi: true,
        };

        let r = run_scf_periodic(&basis, &cell, &kpoints, 8, &atoms, &xc, &options).unwrap();
        assert!(
            r.converged,
            "SCF did not converge in {} iters",
            r.iterations
        );
        assert!(
            (r.n_elec_grid - 8.0).abs() < 1e-2,
            "grid electron count {} (want 8)",
            r.n_elec_grid
        );
        assert!(
            r.energy.is_finite() && r.energy < 0.0,
            "energy = {}",
            r.energy
        );
    }

    #[test]
    #[ignore = "multigrid speedup benchmark: run manually with --nocapture"]
    fn multigrid_collocation_speedup() {
        use crate::integrals::integral::periodic::MultiGridCollocator;
        use std::time::Instant;

        let a = 5.430_697_5 * 1.889_726_988_6;
        let cell = Cell::cubic(a).unwrap();
        let frac = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ];
        let centers: Vec<[f64; 3]> = frac.iter().map(|&f| cell.frac_to_cart(f)).collect();
        let mut shells = Vec::new();
        for &c in &centers {
            shells.extend(si_dzvp_shells(c));
        }
        let basis = Basis::new(shells);
        let nao = basis.nao();
        let e_cut = std::env::var("HARTREE_SI_CUTOFF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(150.0);

        let mut pk = vec![C64::new(0.0, 0.0); nao * nao];
        for i in 0..nao {
            for j in 0..nao {
                let v = 0.1 * (((i * 7 + j) as f64) * 0.11).sin();
                pk[i * nao + j] = C64::new(v, 0.0);
            }
        }
        for i in 0..nao {
            for j in (i + 1)..nao {
                let s = (pk[i * nao + j] + pk[j * nao + i]) * C64::new(0.5, 0.0);
                pk[i * nao + j] = s;
                pk[j * nao + i] = s;
            }
        }
        let p_k = vec![pk];
        let kfracs = [[0.0, 0.0, 0.0]];
        let weights = [1.0];

        let grid = RealSpaceGrid::from_cutoff(cell, e_cut);
        let v: Vec<f64> = (0..grid.n_points())
            .map(|i| ((i as f64) * 0.001).sin())
            .collect();

        let t = Instant::now();
        let single = LatticeCollocator::new(&basis, &grid);
        let sph = single.bloch_phases(&kfracs, &weights);
        let build_single = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let n_s = single.collocate_k(&grid, &p_k, &sph);
        let coll_single = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _v_s = single.integrate_k(&grid, &v, &sph);
        let int_single = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let multi = MultiGridCollocator::new(&basis, &cell, e_cut);
        let mph = multi.bloch_phases(&kfracs, &weights);
        let build_multi = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let n_m = multi.collocate_k(&p_k, &mph);
        let coll_multi = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _v_m = multi.integrate_k(&v, &mph);
        let int_multi = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let cache = single.build_chi_cache(&grid, &sph);
        let build_cache = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let n_c = single.collocate_k_cached(&cache, &p_k);
        let coll_cached = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _v_c = single.integrate_k_cached(&cache, &grid, &v);
        let int_cached = t.elapsed().as_secs_f64();
        let dn_c = n_s
            .iter()
            .zip(&n_c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let per_iter_uncached = coll_single + int_single;
        let per_iter_cached = coll_cached + int_cached;
        eprintln!(
            "[bench] χ-cache: build {build_cache:.2}s ({} MB), per-iter coll+int {per_iter_cached:.2}s vs uncached {per_iter_uncached:.2}s ({:.1}×) | max|Δn| = {dn_c:.2e}",
            (cache.n_entries() * cache.nk() * 16) >> 20,
            per_iter_uncached / per_iter_cached.max(1e-9)
        );
        let n_iter = 20.0;
        let total_uncached = n_iter * per_iter_uncached;
        let total_cached = build_cache + n_iter * per_iter_cached;
        eprintln!(
            "[bench] 20-iter SCF (single-grid): uncached {total_uncached:.1}s vs χ-cached {total_cached:.1}s ({:.1}×)",
            total_uncached / total_cached.max(1e-9)
        );
        let maxn_c = n_s.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(
            dn_c < 1e-9 * maxn_c.max(1.0),
            "cached vs single Δn = {dn_c}"
        );

        let maxn = n_s.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let dn = n_s
            .iter()
            .zip(&n_m)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "[bench] grid {:?}, nao {nao}, multigrid levels {:?}",
            grid.n(),
            multi.level_dims()
        );
        eprintln!("[bench] build:     single {build_single:.2}s  multi {build_multi:.2}s");
        eprintln!(
            "[bench] collocate: single {coll_single:.2}s  multi {coll_multi:.2}s  ({:.1}x)",
            coll_single / coll_multi.max(1e-9)
        );
        eprintln!(
            "[bench] integrate: single {int_single:.2}s  multi {int_multi:.2}s  ({:.1}x)",
            int_single / int_multi.max(1e-9)
        );
        eprintln!(
            "[bench] total coll+int: single {:.2}s  multi {:.2}s  ({:.1}x)  | max|Δn| = {dn:.2e} (max|n| {maxn:.3})",
            coll_single + int_single,
            coll_multi + int_multi,
            (coll_single + int_single) / (coll_multi + int_multi).max(1e-9)
        );
        assert!(dn < 1e-6 * maxn.max(1.0), "multigrid vs single Δn = {dn}");
    }

    #[cfg(feature = "symmetry")]
    #[test]
    fn symmetry_reduces_si_mesh() {
        use latx::{MonkhorstPack, MoyoSymmetry, SymmetryProvider};
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]];
        let numbers = [14, 14];
        let sym = MoyoSymmetry::from_crystal(&cell, &positions, &numbers, 1e-4).unwrap();
        let full = MonkhorstPack::gamma_centered([4, 4, 4]).unwrap();
        let reduced = sym.irreducible_kpoints(&full);
        eprintln!(
            "[sym] 4×4×4 Si: {} → {} k-points",
            full.len(),
            reduced.len()
        );
        assert!(reduced.len() < full.len(), "no reduction");
        let wsum: f64 = reduced.iter().map(|k| k.weight).sum();
        assert!((wsum - 1.0).abs() < 1e-12, "weight sum {wsum}");
    }

    #[cfg(feature = "symmetry")]
    #[test]
    #[ignore = "symmetry SCF comparison: run manually (full vs reduced k-mesh)"]
    fn symmetry_reduced_scf_matches_full() {
        use latx::{MonkhorstPack, MoyoSymmetry, SymmetryProvider};
        let a = 10.263;
        let cell = Cell::from_vectors(
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        )
        .unwrap();
        let r0 = cell.frac_to_cart([0.0, 0.0, 0.0]);
        let r1 = cell.frac_to_cart([0.25, 0.25, 0.25]);
        let mut shells = si_szv_shells(r0);
        shells.extend(si_szv_shells(r1));
        let basis = Basis::new(shells);
        let atoms = [si_atom(r0), si_atom(r1)];
        let xc = GridXc::pade();
        let options = PeriodicScfOptions {
            e_cut: 80.0,
            max_iter: 100,
            energy_tol: 1e-7,
            density_tol: 1e-6,
            mixing: 0.3,
            bloch_rmax: None,
            cache_chi: true,
        };

        let full = MonkhorstPack::regular([2, 2, 2]).unwrap();
        let sym = MoyoSymmetry::from_crystal(
            &cell,
            &[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            &[14, 14],
            1e-4,
        )
        .unwrap();
        let reduced = sym.irreducible_kpoints(&full);

        let e_full = run_scf_periodic(&basis, &cell, &full, 8, &atoms, &xc, &options)
            .unwrap()
            .energy;
        let e_red = run_scf_periodic(&basis, &cell, &reduced, 8, &atoms, &xc, &options)
            .unwrap()
            .energy;
        eprintln!(
            "[sym-scf] full {} k → E = {e_full:.6}; reduced {} k → E = {e_red:.6}; Δ = {:.2e}",
            full.len(),
            reduced.len(),
            (e_full - e_red).abs()
        );
        assert!(reduced.len() < full.len(), "no reduction");
        assert!(e_red.is_finite() && e_red < 0.0);
        assert!(
            (e_full - e_red).abs() < 5e-3,
            "reduced {e_red} vs full {e_full}"
        );
    }
}
