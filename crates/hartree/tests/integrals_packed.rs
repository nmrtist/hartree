use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::{ConventionalProvider, InCoreEri, IntegralProvider};
use hartree::linalg::{mat_from_row_major, mat_to_row_major};

fn provider() -> (ConventionalProvider, usize) {
    let atom = |s: &str, p: [f64; 3]| Atom::new(Element::from_symbol(s).unwrap(), p);
    let mol = Molecule::new(
        vec![
            atom("O", [0.0, -0.143225816552, 0.0]),
            atom("H", [1.638036840407, 1.136548822547, 0.0]),
            atom("H", [-1.638036840407, 1.136548822547, 0.3]),
        ],
        0,
        1,
    );
    let ao = BasisSet::load("cc-pvdz").unwrap().build(&mol).unwrap();
    let charges = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    let n = ao.n_ao();
    (ConventionalProvider::new(ao.into_integral(), charges), n)
}

fn dense_jk(eri: &[f64], d: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = vec![0.0; n * n];
    let mut k = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for e in 0..n {
                    let g = eri[((a * n + b) * n + c) * n + e];
                    j[a * n + b] += g * d[c * n + e];
                    k[a * n + c] += g * d[b * n + e];
                }
            }
        }
    }
    (j, k)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn packed_jk_matches_dense_contraction() {
    // The reference provider builds its dense tensor before any J/K call, so it
    // comes from the dense ERI builder rather than from the packed store.
    let (reference, n) = provider();
    let eri = reference.ao_eri();

    let symmetric: Vec<f64> = (0..n * n)
        .map(|p| {
            let (i, j) = (p / n, p % n);
            ((i * j + i + j) % 13) as f64 * 0.021 - 0.11
        })
        .collect();
    // A response-like density with no symmetry exercises the transposed half of K.
    let general: Vec<f64> = (0..n * n)
        .map(|p| ((p * 7 + 3) % 19) as f64 * 0.017 - 0.15)
        .collect();
    let densities = [symmetric, general];
    let mats: Vec<_> = densities.iter().map(|d| mat_from_row_major(n, d)).collect();

    let (packed, _) = provider();
    let jk = packed.build_jk(&mats);
    let j_only = packed.build_j(&mats);
    for (di, d) in densities.iter().enumerate() {
        let (j_ref, k_ref) = dense_jk(eri, d, n);
        let j = mat_to_row_major(&jk.coulomb[di]);
        let k = mat_to_row_major(&jk.exchange[di]);
        assert!(max_diff(&j, &j_ref) < 1e-10, "J[{di}]");
        assert!(max_diff(&k, &k_ref) < 1e-10, "K[{di}]");
        assert!(max_diff(&mat_to_row_major(&j_only[di]), &j_ref) < 1e-10);
    }

    // After an SCF-style J/K call the dense tensor is expanded from the packed store.
    assert!(max_diff(packed.ao_eri(), eri) < 1e-12);
}
