use std::collections::HashMap;
use std::time::{Duration, Instant};

use hartree::basis::BasisSet;
use hartree::cc::{
    CcsdOptions, column_block, frozen_core_orbitals, rccsd_spin_adapted, rccsd_t_spin_adapted,
    rhf_mp2, transform_block,
};
use hartree::core::{Atom, Element, Molecule};
use hartree::dft::{FunctionalSpec, GridXc, MolecularGrid};
use hartree::integrals::{ConventionalProvider, InCoreEri};
use hartree::scf::{Reference, ScfOptions, XcContributor, run_scf, run_scf_with_xc};
use serde::Deserialize;

const GEOMETRIES_JSON: &str = include_str!("../../../tests/ref/geometries.json");

#[derive(Deserialize)]
struct Geometries {
    molecules: HashMap<String, GeomEntry>,
}

#[derive(Deserialize)]
struct GeomEntry {
    charge: i32,
    multiplicity: u32,
    atoms: Vec<(String, f64, f64, f64)>,
}

impl GeomEntry {
    fn molecule(&self) -> Molecule {
        let atoms = self
            .atoms
            .iter()
            .map(|(s, x, y, z)| Atom::new(Element::from_symbol(s).unwrap(), [*x, *y, *z]))
            .collect();
        Molecule::new(atoms, self.charge, self.multiplicity)
    }
}

fn time_reps<T>(reps: usize, mut f: impl FnMut() -> T) -> (Duration, Duration) {
    let mut ts: Vec<Duration> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let r = f();
        ts.push(t0.elapsed());
        std::hint::black_box(&r);
    }
    ts.sort();
    let med = ts[ts.len() / 2];
    let min = ts[0];
    (med, min)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Args {
    mol: String,
    basis: String,
    method: String,
    reps: usize,
    grid: usize,
    all_electron: bool,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut pos: Vec<String> = Vec::new();
    let mut reps = 3usize;
    let mut grid = 3usize;
    let mut all_electron = false;
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--reps" => {
                i += 1;
                reps = raw.get(i).and_then(|s| s.parse().ok()).unwrap_or(3);
            }
            "--grid" => {
                i += 1;
                grid = raw.get(i).and_then(|s| s.parse().ok()).unwrap_or(3);
            }
            "--all-electron" => all_electron = true,
            t if t.starts_with("--") => { /* ignore harness flag */ }
            t => pos.push(t.to_string()),
        }
        i += 1;
    }
    Args {
        mol: pos.first().cloned().unwrap_or_else(|| "water".into()),
        basis: pos.get(1).cloned().unwrap_or_else(|| "sto-3g".into()),
        method: pos
            .get(2)
            .cloned()
            .unwrap_or_else(|| "ccsdt".into())
            .to_ascii_lowercase(),
        reps,
        grid,
        all_electron,
    }
}

fn main() {
    let args = parse_args();
    let geoms: Geometries = serde_json::from_str(GEOMETRIES_JSON).expect("parse geometries.json");
    let mol = geoms
        .molecules
        .get(&args.mol)
        .unwrap_or_else(|| panic!("unknown molecule {:?}", args.mol))
        .molecule();

    let threads = rayon::current_num_threads();
    let n_elec = mol.n_electrons();
    let two_s = mol.multiplicity.saturating_sub(1) as i64;
    let n_alpha = ((n_elec + two_s) / 2) as usize;
    let n_beta = ((n_elec - two_s) / 2) as usize;
    let nuc = mol.nuclear_repulsion();

    let is_dft = !matches!(
        args.method.as_str(),
        "rhf" | "uhf" | "mp2" | "ccsd" | "ccsdt" | "ccsd-t" | "ccsd(t)"
    );

    let (t_basis, _) = time_reps(args.reps.max(1), || {
        BasisSet::load(&args.basis).unwrap().build(&mol).unwrap()
    });
    let ao = BasisSet::load(&args.basis).unwrap().build(&mol).unwrap();
    let n = ao.n_ao();

    println!("================================================================");
    println!(
        "hartree perf_pipeline  {}/{}/{}   ({} bf, {} e⁻, {}α/{}β)",
        args.mol, args.basis, args.method, n, n_elec, n_alpha, n_beta
    );
    println!(
        "rayon threads = {}   reps = {}   frozen-core = {}",
        threads,
        args.reps,
        if args.all_electron { "off" } else { "on" }
    );
    println!("----------------------------------------------------------------");
    println!("basis build           {:>10.3} ms", ms(t_basis));

    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();

    if is_dft {
        let spec = FunctionalSpec::parse(&args.method)
            .unwrap_or_else(|_| panic!("unknown method/functional {:?}", args.method));
        let reference = if mol.multiplicity > 1 {
            Reference::Uhf
        } else {
            Reference::Rhf
        };

        let (t_grid, _) = time_reps(args.reps.max(1), || {
            MolecularGrid::build(&mol, args.grid).unwrap()
        });
        let grid = MolecularGrid::build(&mol, args.grid).unwrap();
        let npts = grid.points.len();

        let ks_opts = ScfOptions {
            energy_tol: 1e-9,
            error_tol: 1e-6,
            ..ScfOptions::default()
        };
        let mut iters = 0usize;
        let mut e_ks = 0.0;
        let mut e_xc = 0.0;
        let (t_ks, _) = time_reps(args.reps, || {
            let gx = GridXc::new(&mol, &ao, &spec, args.grid).unwrap();
            let provider = ConventionalProvider::new(ao.clone().into_integral(), charges.clone());
            let scf = run_scf_with_xc(
                &provider,
                n_alpha,
                n_beta,
                reference,
                nuc,
                &ks_opts,
                Some(&gx as &dyn XcContributor),
            )
            .unwrap();
            iters = scf.iterations;
            e_ks = scf.energy;
            e_xc = scf.xc_energy.unwrap_or(0.0);
            scf.energy
        });

        println!(
            "DFT grid build        {:>10.3} ms   ({npts} points, level {})",
            ms(t_grid),
            args.grid
        );
        println!(
            "KS-SCF total          {:>10.3} ms   ({iters} iters, {:>7.3} ms/iter)",
            ms(t_ks),
            ms(t_ks) / iters.max(1) as f64
        );
        println!("----------------------------------------------------------------");
        println!("E(KS)   = {:.12} Eh", e_ks);
        println!("E_xc    = {:.12} Eh", e_xc);
        let _ = t_grid;
        println!(
            "RESULT\t{}\t{}\t{}\tdft\tbf={}\tthreads={}\tgrid_ms={:.3}\tksscf_ms={:.3}\titers={}\tE={:.12}",
            args.mol,
            args.basis,
            args.method,
            n,
            threads,
            ms(t_grid),
            ms(t_ks),
            iters,
            e_ks
        );
        return;
    }

    let reference = match args.method.as_str() {
        "uhf" => Reference::Uhf,
        _ => Reference::Rhf,
    };

    let (t_eri, _) = time_reps(args.reps, || {
        let p = ConventionalProvider::new(ao.clone().into_integral(), charges.clone());
        p.ao_eri();
        p.ao_eri().len()
    });

    let provider = ConventionalProvider::new(ao.clone().into_integral(), charges.clone());
    let _ = provider.ao_eri();

    let scf_opts = ScfOptions::default();
    let mut scf_iters = 0usize;
    let mut e_scf = 0.0;
    let (t_scf, _) = time_reps(args.reps, || {
        let scf = run_scf(&provider, n_alpha, n_beta, reference, nuc, &scf_opts).unwrap();
        scf_iters = scf.iterations;
        e_scf = scf.energy;
        scf.energy
    });
    let scf = run_scf(&provider, n_alpha, n_beta, reference, nuc, &scf_opts).unwrap();
    assert!(scf.converged, "SCF did not converge");

    println!("ERI build (nao⁴)      {:>10.3} ms", ms(t_eri));
    println!(
        "SCF total             {:>10.3} ms   ({scf_iters} iters, {:>7.3} ms/iter)",
        ms(t_scf),
        ms(t_scf) / scf_iters.max(1) as f64
    );

    let n_frozen = if args.all_electron {
        0
    } else {
        frozen_core_orbitals(&mol)
    };

    let mut summary = format!(
        "RESULT\t{}\t{}\t{}\tbf={}\tthreads={}\teri_ms={:.3}\tscf_ms={:.3}\tscf_iters={}",
        args.mol,
        args.basis,
        args.method,
        n,
        threads,
        ms(t_eri),
        ms(t_scf),
        scf_iters
    );
    println!("----------------------------------------------------------------");
    println!("E(SCF)  = {:.12} Eh", e_scf);

    let want_post = matches!(
        args.method.as_str(),
        "mp2" | "ccsd" | "ccsdt" | "ccsd-t" | "ccsd(t)"
    );
    if want_post && reference == Reference::Rhf {
        let m = scf.n_orbitals;
        let n_occ = scf.n_alpha;
        let n_act = n_occ - n_frozen;
        let n_virt = m - n_occ;
        let c = &scf.mo_coeff_alpha;

        let (t_tr, _) = time_reps(args.reps, || {
            let c_occ = column_block(c, n, m, n_frozen, n_act);
            let c_virt = column_block(c, n, m, n_occ, n_virt);
            transform_block(provider.ao_eri(), n, [&c_occ, &c_virt, &c_occ, &c_virt])
        });
        println!(
            "AO→MO (ov|ov)         {:>10.3} ms   (o={}, v={}, frozen={})",
            ms(t_tr),
            n_act,
            n_virt,
            n_frozen
        );
        summary += &format!("\ttransform_ovov_ms={:.3}", ms(t_tr));

        let mut e_mp2 = 0.0;
        let (t_mp2, _) = time_reps(args.reps, || {
            let r = rhf_mp2(&provider, &scf, n_frozen);
            e_mp2 = r.correlation_energy;
            r.correlation_energy
        });
        println!("MP2 (post-SCF)        {:>10.3} ms", ms(t_mp2));
        println!("  E(corr,MP2) = {:.12} Eh", e_mp2);
        summary += &format!("\tmp2_ms={:.3}\tEcorr_mp2={:.12}", ms(t_mp2), e_mp2);

        if matches!(
            args.method.as_str(),
            "ccsd" | "ccsdt" | "ccsd-t" | "ccsd(t)"
        ) {
            let cc_opts = CcsdOptions::default();
            let mut cc_iters = 0usize;
            let mut e_ccsd = 0.0;
            let (t_ccsd, _) = time_reps(args.reps, || {
                let r = rccsd_spin_adapted(&provider, &scf, n_frozen, &cc_opts);
                cc_iters = r.iterations;
                e_ccsd = r.correlation_energy;
                r.correlation_energy
            });
            println!(
                "CCSD (post-SCF)       {:>10.3} ms   ({cc_iters} iters, {:>7.3} ms/iter)",
                ms(t_ccsd),
                ms(t_ccsd) / cc_iters.max(1) as f64
            );
            println!("  E(corr,CCSD) = {:.12} Eh", e_ccsd);
            summary += &format!(
                "\tccsd_ms={:.3}\tccsd_iters={}\tEcorr_ccsd={:.12}",
                ms(t_ccsd),
                cc_iters,
                e_ccsd
            );

            if matches!(args.method.as_str(), "ccsdt" | "ccsd-t" | "ccsd(t)") {
                let mut e_t = 0.0;
                let (t_ccsdt, _) = time_reps(args.reps, || {
                    let r = rccsd_t_spin_adapted(&provider, &scf, n_frozen, &cc_opts);
                    e_t = r.triples_energy;
                    r.total_energy
                });
                let t_triples = t_ccsdt.saturating_sub(t_ccsd);
                println!(
                    "(T) triples           {:>10.3} ms   [= CCSD(T) {:.3} − CCSD {:.3}]",
                    ms(t_triples),
                    ms(t_ccsdt),
                    ms(t_ccsd)
                );
                println!("  E(T) = {:.12} Eh", e_t);
                summary += &format!(
                    "\ttriples_ms={:.3}\tccsdt_total_ms={:.3}\tE_T={:.12}",
                    ms(t_triples),
                    ms(t_ccsdt),
                    e_t
                );
            }
        }
    }

    println!("----------------------------------------------------------------");
    println!("{summary}");
}
