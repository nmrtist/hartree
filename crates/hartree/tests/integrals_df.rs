use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::integral::Basis as IntegralBasis;
use hartree::integrals::{ConventionalProvider, DfProvider, InCoreEri, IntegralProvider};
use hartree::linalg::{mat_from_row_major, mat_to_row_major};

fn basis_and_charges(mol: &Molecule, basis: &str) -> (IntegralBasis, Vec<([f64; 3], f64)>) {
    let ao = BasisSet::load(basis).unwrap().build(mol).unwrap();
    let charges: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .map(|a| (a.position, a.element.z() as f64))
        .collect();
    (ao.into_integral(), charges)
}

fn aux_basis(mol: &Molecule) -> IntegralBasis {
    BasisSet::load_aux("def2-universal-jkfit")
        .unwrap()
        .build(mol)
        .unwrap()
        .into_integral()
}

fn atom(sym: &str, position: [f64; 3]) -> Atom {
    Atom::new(Element::from_symbol(sym).unwrap(), position)
}

fn water() -> Molecule {
    Molecule::new(
        vec![
            atom("O", [0.0, -0.143225816552, 0.0]),
            atom("H", [1.638036840407, 1.136548822547, 0.0]),
            atom("H", [-1.638036840407, 1.136548822547, 0.0]),
        ],
        0,
        1,
    )
}

fn probe_density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = (((i * 31 + j * 17 + 7) % 13) as f64 - 6.0) * 0.05;
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn df_jk_is_symmetric() {
    for basis in ["sto-3g", "6-31g"] {
        let mol = water();
        let (ox, charges) = basis_and_charges(&mol, basis);
        let aux = aux_basis(&mol);
        let df = DfProvider::new(ox, &aux, charges).unwrap();
        let n = df.n_basis();
        let d = mat_from_row_major(n, &probe_density(n));
        let jk = df.build_jk(std::slice::from_ref(&d));
        let j = mat_to_row_major(&jk.coulomb[0]);
        let k = mat_to_row_major(&jk.exchange[0]);
        for mu in 0..n {
            for nu in 0..mu {
                assert_eq!(
                    j[mu * n + nu],
                    j[nu * n + mu],
                    "{basis}: J asymmetric at ({mu},{nu})"
                );
                let dk = (k[mu * n + nu] - k[nu * n + mu]).abs();
                assert!(
                    dk < 1e-12,
                    "{basis}: K asymmetric at ({mu},{nu}) by {dk:.2e}"
                );
            }
        }
    }
}

#[test]
fn df_jk_matches_conventional_to_fitting_error() {
    let cases = [("sto-3g", 1.5e-4), ("6-31g", 2.5e-3)];
    for (basis, gate) in cases {
        let mol = water();
        let (ox, charges) = basis_and_charges(&mol, basis);
        let aux = aux_basis(&mol);
        let conv = ConventionalProvider::new(ox.clone(), charges.clone());
        let df = DfProvider::new(ox, &aux, charges).unwrap();
        let n = conv.n_basis();
        let d = mat_from_row_major(n, &probe_density(n));

        let jk_conv = conv.build_jk(std::slice::from_ref(&d));
        let jk_df = df.build_jk(std::slice::from_ref(&d));
        let j_diff = max_abs_diff(
            &mat_to_row_major(&jk_conv.coulomb[0]),
            &mat_to_row_major(&jk_df.coulomb[0]),
        );
        let k_diff = max_abs_diff(
            &mat_to_row_major(&jk_conv.exchange[0]),
            &mat_to_row_major(&jk_df.exchange[0]),
        );
        eprintln!("water/{basis}: fitting error ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e}");
        assert!(
            j_diff < gate && k_diff < gate,
            "water/{basis}: ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e} exceeds fitting gate {gate:.0e}"
        );
        assert!(
            j_diff > 1e-10,
            "water/{basis}: ΔJ = {j_diff:.2e} is implausibly small for a fitted J"
        );
    }
}

#[test]
fn df_batched_matches_single() {
    let mol = water();
    let (ox, charges) = basis_and_charges(&mol, "6-31g");
    let aux = aux_basis(&mol);
    let df = DfProvider::new(ox, &aux, charges).unwrap();
    let n = df.n_basis();

    let d0 = mat_from_row_major(n, &probe_density(n));
    let scaled: Vec<f64> = probe_density(n).iter().map(|v| v * -0.5).collect();
    let d1 = mat_from_row_major(n, &scaled);

    let single0 = df.build_jk(std::slice::from_ref(&d0));
    let single1 = df.build_jk(std::slice::from_ref(&d1));
    let batch = df.build_jk(&[d0.clone(), d1, d0.clone()]);

    for (slot, single) in [(0, &single0), (1, &single1), (2, &single0)] {
        assert_eq!(
            mat_to_row_major(&batch.coulomb[slot]),
            mat_to_row_major(&single.coulomb[0]),
            "batched J[{slot}] differs from the single-density build"
        );
        assert_eq!(
            mat_to_row_major(&batch.exchange[slot]),
            mat_to_row_major(&single.exchange[0]),
            "batched K[{slot}] differs from the single-density build"
        );
    }
}

#[test]
fn df_diagonal_reconstruction_tracks_exact() {
    let mol = water();
    let (ox, charges) = basis_and_charges(&mol, "6-31g");
    let aux = aux_basis(&mol);
    let conv = ConventionalProvider::new(ox.clone(), charges.clone());
    let df = DfProvider::new(ox, &aux, charges).unwrap();
    let n = conv.n_basis();
    let eri = conv.ao_eri();

    let mut worst_rel = 0.0_f64;
    for nu in 0..n {
        let mut d = vec![0.0; n * n];
        d[nu * n + nu] = 1.0;
        let jk = df.build_jk(std::slice::from_ref(&mat_from_row_major(n, &d)));
        let k = mat_to_row_major(&jk.exchange[0]);
        for mu in 0..n {
            let exact = eri[((mu * n + nu) * n + mu) * n + nu];
            let fitted = k[mu * n + mu];
            if exact > 1e-3 {
                let rel = (fitted - exact).abs() / exact;
                worst_rel = worst_rel.max(rel);
            }
        }
    }
    eprintln!("water/6-31g fitted-diagonal worst relative deviation: {worst_rel:.2e}");
    assert!(
        worst_rel < 0.05,
        "fitted (μν|μν) deviates {worst_rel:.2e} (>5%) from exact — fit is broken"
    );
}
