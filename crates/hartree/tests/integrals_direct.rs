use hartree::basis::BasisSet;
use hartree::core::{Atom, Element, Molecule};
use hartree::integrals::integral::Basis as IntegralBasis;
use hartree::integrals::{ConventionalProvider, DirectProvider, IntegralProvider};
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

fn h2() -> Molecule {
    Molecule::new(
        vec![atom("H", [0.0, 0.0, 0.0]), atom("H", [0.0, 0.0, 1.4])],
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

type Case = (&'static str, fn() -> Molecule, &'static str);

#[test]
fn direct_matches_conventional_jk() {
    let cases: &[Case] = &[
        ("h2", h2, "sto-3g"),
        ("h2", h2, "6-31g"),
        ("water", water, "sto-3g"),
        ("water", water, "6-31g"),
        ("water", water, "cc-pvdz"),
    ];

    let mut worst = 0.0_f64;
    for (name, build, basis) in cases {
        let mol = build();
        let (ox, charges) = basis_and_charges(&mol, basis);
        let conv = ConventionalProvider::new(ox.clone(), charges.clone());
        let direct = DirectProvider::new(ox, charges);

        let n = conv.n_basis();
        assert_eq!(direct.n_basis(), n);

        let d = mat_from_row_major(n, &probe_density(n));
        let jk_conv = conv.build_jk(std::slice::from_ref(&d));
        let jk_direct = direct.build_jk(std::slice::from_ref(&d));

        let j_diff = max_abs_diff(
            &mat_to_row_major(&jk_conv.coulomb[0]),
            &mat_to_row_major(&jk_direct.coulomb[0]),
        );
        let k_diff = max_abs_diff(
            &mat_to_row_major(&jk_conv.exchange[0]),
            &mat_to_row_major(&jk_direct.exchange[0]),
        );
        worst = worst.max(j_diff).max(k_diff);

        assert!(
            j_diff < 1e-9 && k_diff < 1e-9,
            "{name}/{basis} (n={n}): ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e}"
        );
    }
    eprintln!("DirectProvider vs ConventionalProvider: worst ΔJK = {worst:.2e}");
}

#[test]
fn direct_matches_conventional_batched() {
    let mol = water();
    let (ox, charges) = basis_and_charges(&mol, "6-31g");
    let conv = ConventionalProvider::new(ox.clone(), charges.clone());
    let direct = DirectProvider::new(ox, charges);
    let n = conv.n_basis();

    let d0 = mat_from_row_major(n, &probe_density(n));
    let mut raw1 = probe_density(n);
    for v in raw1.iter_mut() {
        *v *= -0.5;
    }
    let d1 = mat_from_row_major(n, &raw1);
    let batch = [d0, d1];

    let jk_conv = conv.build_jk(&batch);
    let jk_direct = direct.build_jk(&batch);
    for s in 0..batch.len() {
        let j_diff = max_abs_diff(
            &mat_to_row_major(&jk_conv.coulomb[s]),
            &mat_to_row_major(&jk_direct.coulomb[s]),
        );
        let k_diff = max_abs_diff(
            &mat_to_row_major(&jk_conv.exchange[s]),
            &mat_to_row_major(&jk_direct.exchange[s]),
        );
        assert!(
            j_diff < 1e-9 && k_diff < 1e-9,
            "batch[{s}]: ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e}"
        );
    }
}

#[test]
fn screened_matches_full_build() {
    let cases: &[Case] = &[
        ("h2", h2, "6-31g"),
        ("water", water, "sto-3g"),
        ("water", water, "cc-pvdz"),
    ];
    for (name, build, basis) in cases {
        let mol = build();
        let (ox, charges) = basis_and_charges(&mol, basis);
        let direct = DirectProvider::new(ox, charges);
        let n = direct.n_basis();
        let d = mat_from_row_major(n, &probe_density(n));

        let full = direct.build_jk(std::slice::from_ref(&d));
        let screened = direct.build_jk_screened(std::slice::from_ref(&d));
        let j_diff = max_abs_diff(
            &mat_to_row_major(&full.coulomb[0]),
            &mat_to_row_major(&screened.coulomb[0]),
        );
        let k_diff = max_abs_diff(
            &mat_to_row_major(&full.exchange[0]),
            &mat_to_row_major(&screened.exchange[0]),
        );
        assert!(
            j_diff < 1e-9 && k_diff < 1e-9,
            "{name}/{basis}: screened vs full ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e}"
        );
    }
}

#[test]
fn incremental_delta_builds_sum_to_full() {
    let mol = water();
    let (ox, charges) = basis_and_charges(&mol, "cc-pvdz");
    let direct = DirectProvider::new(ox, charges);
    let n = direct.n_basis();

    let base = probe_density(n);
    let scale = |f: f64| -> Vec<f64> { base.iter().map(|v| v * f).collect() };
    let densities = [scale(0.6), scale(0.95), scale(1.0)];

    let mut j_acc = vec![0.0; n * n];
    let mut k_acc = vec![0.0; n * n];
    let mut prev = vec![0.0; n * n];
    for dens in &densities {
        let delta: Vec<f64> = dens.iter().zip(&prev).map(|(a, b)| a - b).collect();
        let jk = direct.build_jk_screened(std::slice::from_ref(&mat_from_row_major(n, &delta)));
        let dj = mat_to_row_major(&jk.coulomb[0]);
        let dk = mat_to_row_major(&jk.exchange[0]);
        for i in 0..n * n {
            j_acc[i] += dj[i];
            k_acc[i] += dk[i];
        }
        prev = dens.clone();
    }

    let full = direct.build_jk(std::slice::from_ref(&mat_from_row_major(n, &densities[2])));
    let j_diff = max_abs_diff(&j_acc, &mat_to_row_major(&full.coulomb[0]));
    let k_diff = max_abs_diff(&k_acc, &mat_to_row_major(&full.exchange[0]));
    eprintln!("incremental Δ-build vs full: ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e}");
    assert!(
        j_diff < 1e-9 && k_diff < 1e-9,
        "incremental Δ-build mismatch: ΔJ = {j_diff:.2e}, ΔK = {k_diff:.2e}"
    );
}

#[test]
fn tightening_screening_converges() {
    let mol = water();
    let (ox, charges) = basis_and_charges(&mol, "cc-pvdz");
    let conv = ConventionalProvider::new(ox.clone(), charges.clone());
    let n = conv.n_basis();
    let d = mat_from_row_major(n, &probe_density(n));
    let jk_conv = conv.build_jk(std::slice::from_ref(&d));
    let j_conv = mat_to_row_major(&jk_conv.coulomb[0]);
    let k_conv = mat_to_row_major(&jk_conv.exchange[0]);

    let gap = |tau: f64| {
        let direct = DirectProvider::with_screening(ox.clone(), charges.clone(), tau);
        let jk = direct.build_jk(std::slice::from_ref(&d));
        max_abs_diff(&mat_to_row_major(&jk.coulomb[0]), &j_conv)
            .max(max_abs_diff(&mat_to_row_major(&jk.exchange[0]), &k_conv))
    };

    let loose = gap(1e-3);
    let mid = gap(1e-7);
    let tight = gap(1e-14);

    eprintln!("screening gap: τ=1e-3 → {loose:.2e}, τ=1e-7 → {mid:.2e}, τ=1e-14 → {tight:.2e}");
    assert!(
        tight <= mid + 1e-15 && mid <= loose + 1e-15,
        "gap not monotone in τ: loose={loose:.2e}, mid={mid:.2e}, tight={tight:.2e}"
    );
    assert!(tight < 1e-9, "tight-screening gap {tight:.2e} exceeds 1e-9");
    assert!(
        loose > tight,
        "loose screening ({loose:.2e}) should differ more than tight ({tight:.2e})"
    );
}
