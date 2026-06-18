use std::collections::HashMap;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::{
    ConventionalProvider, DfProvider, DirectProvider, IntegralProvider, integral::Basis,
};
use hartree::linalg::{Mat, mat_from_row_major};
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

fn geometries() -> Geometries {
    serde_json::from_str(GEOMETRIES_JSON).expect("parse geometries.json")
}

fn basis_and_charges(mol: &Molecule, basis: &str) -> (Basis, Vec<([f64; 3], f64)>) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    (ao.into_integral(), charges)
}

fn probe_density(n: usize) -> Mat {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = (((i * 31 + j * 17 + 7) % 13) as f64 - 6.0) * 0.05;
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    mat_from_row_major(n, &d)
}

fn bench_build_jk(c: &mut Criterion) {
    let geoms = geometries();
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let cases: &[(&str, &str, bool)] = &[
        ("water", "cc-pvdz", true),
        ("benzene", "6-31g", true),
        ("benzene", "cc-pvdz", false),
    ];

    let mut group = c.benchmark_group("build_jk");
    group.sample_size(10);

    for &(name, basis, run_conv) in cases {
        let mol = geoms.molecules[name].molecule();
        let (ox, charges) = basis_and_charges(&mol, basis);
        let n = ox.nao();
        let dens = [probe_density(n)];
        let id = format!("{name}/{basis}");

        if run_conv {
            let conv = ConventionalProvider::new(ox.clone(), charges.clone());
            group.bench_function(format!("conv {id}"), |b| {
                b.iter(|| black_box(conv.build_jk(black_box(&dens))));
            });
        }

        let aux = BasisSet::load_aux("def2-universal-jkfit")
            .unwrap()
            .build(&mol)
            .unwrap()
            .into_integral();
        let df = DfProvider::new(ox.clone(), &aux, charges.clone()).unwrap();
        group.bench_function(format!("ri xN {id}"), |b| {
            b.iter(|| black_box(df.build_jk(black_box(&dens))));
        });
        drop(df);

        let direct = DirectProvider::new(ox, charges);
        group.bench_function(format!("direct x1 {id}"), |b| {
            b.iter(|| serial_pool.install(|| black_box(direct.build_jk(black_box(&dens)))));
        });
        group.bench_function(format!("direct xN {id}"), |b| {
            b.iter(|| black_box(direct.build_jk(black_box(&dens))));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_build_jk);
criterion_main!(benches);
