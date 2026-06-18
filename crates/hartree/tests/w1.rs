use hartree::core::Molecule;
use hartree::w1::{W1Options, extrapolate_corr_n3, extrapolate_hf_karton_martin, run_w1};

fn h2() -> Molecule {
    Molecule::from_xyz("2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n").unwrap()
}

fn water() -> Molecule {
    Molecule::from_xyz("3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.470\nH 0.0 -0.757 -0.470\n")
        .unwrap()
}

#[test]
fn h2_dz_tz_end_to_end_pinned() {
    let opts = W1Options {
        opt_level: "hf/sto-3g".into(),
        basis_small: "cc-pvdz".into(),
        basis_large: "cc-pvtz".into(),
        compute_frequencies: false,
        ..W1Options::default()
    };
    let res = run_w1(&h2(), &opts).unwrap();
    assert_eq!((res.cardinal_small, res.cardinal_large), (2, 3));
    assert_eq!(res.n_frozen, 0); // H has no noble-gas core
    assert!(res.thermo.is_none());

    let pins = [
        (res.e_hf_small, -1.127761652989, "E_HF(DZ)"),
        (res.e_hf_large, -1.132619728969, "E_HF(TZ)"),
        (res.e_ccsd_corr_small, -0.034190853250, "E_corr(DZ)"),
        (res.e_ccsd_corr_large, -0.039063010285, "E_corr(TZ)"),
        (res.e_t_small, 0.0, "E_(T)(DZ)"),
    ];
    for (got, want, label) in pins {
        assert!(
            (got - want).abs() < 1e-8,
            "{label} = {got:.12}, want {want:.12}"
        );
    }
    assert!(res.e_t_small.abs() < 1e-15, "E_(T) = {:e}", res.e_t_small);

    assert_eq!(
        res.e_hf_cbs,
        extrapolate_hf_karton_martin(res.e_hf_small, res.e_hf_large, 2, 3)
    );
    assert_eq!(
        res.e_ccsd_corr_cbs,
        extrapolate_corr_n3(res.e_ccsd_corr_small, res.e_ccsd_corr_large, 2, 3)
    );

    let sum = res.e_hf_cbs + res.e_ccsd_corr_cbs + res.e_t_small;
    assert_eq!(res.electronic_energy(), sum);
    assert!(
        (res.electronic_energy() - (-1.174135562977)).abs() < 1e-8,
        "E(hartree-W1, DZ/TZ test pair) = {:.12}",
        res.electronic_energy()
    );
    assert!(res.e_hf_cbs < res.e_hf_large && res.e_ccsd_corr_cbs < res.e_ccsd_corr_large);
}

#[test]
#[ignore = "slow (~40 s dev): full-default hartree-W1 with CCSD/cc-pVQZ stage"]
fn h2_full_default_pinned() {
    let opts = W1Options {
        symmetry_number: 2,
        ..W1Options::default()
    };
    let res = run_w1(&h2(), &opts).unwrap();
    assert_eq!((res.cardinal_small, res.cardinal_large), (3, 4));
    assert_eq!(res.opt_label, "b3lyp/cc-pvtz");

    let pins = [
        (res.e_hf_small, -1.132939043367, "E_HF(TZ)"),
        (res.e_hf_large, -1.133435695328, "E_HF(QZ)"),
        (res.e_hf_cbs, -1.133498396618, "E_HF/CBS"),
        (res.e_ccsd_corr_small, -0.039397589988, "E_corr(TZ)"),
        (res.e_ccsd_corr_large, -0.040360056348, "E_corr(QZ)"),
        (res.e_ccsd_corr_cbs, -0.041062396664, "E_corr/CBS"),
        (res.e_t_small, 0.0, "E_(T)(TZ)"),
        (res.electronic_energy(), -1.174560793282, "E(hartree-W1)"),
    ];
    for (got, want, label) in pins {
        assert!(
            (got - want).abs() < 1e-7,
            "{label} = {got:.12}, want {want:.12}"
        );
    }

    let t = res.thermo.as_ref().expect("frequencies on by default");
    assert!(
        (t.enthalpy - (-1.161188303457)).abs() < 1e-6,
        "H = {:.12}",
        t.enthalpy
    );
    assert!(
        (t.gibbs - (-1.175980941156)).abs() < 1e-6,
        "G = {:.12}",
        t.gibbs
    );
    assert_eq!(t.enthalpy, res.electronic_energy() + t.h_corr);
    assert_eq!(t.gibbs, res.electronic_energy() + t.g_corr);
    assert_eq!(t.gibbs_qrrho, res.electronic_energy() + t.g_corr_qrrho);
    assert_eq!(t.freq.frequencies.n_imaginary, 0);
}

#[test]
#[ignore = "slow (>30 s dev): CCSD(T)/cc-pVDZ + CCSD/cc-pVTZ on water"]
fn water_dz_tz_pinned_with_thermo() {
    let opts = W1Options {
        opt_level: "hf/cc-pvdz".into(),
        basis_small: "cc-pvdz".into(),
        basis_large: "cc-pvtz".into(),
        symmetry_number: 2,
        ..W1Options::default()
    };
    let res = run_w1(&water(), &opts).unwrap();
    assert_eq!(res.n_frozen, 1);
    assert!(
        res.e_t_small < -1e-4 && res.e_t_small > -0.02,
        "E_(T) = {:.12}",
        res.e_t_small
    );

    let pins = [
        (res.e_hf_small, -76.027053512765, "E_HF(DZ)"),
        (res.e_hf_large, -76.057662806301, "E_HF(TZ)"),
        (res.e_ccsd_corr_small, -0.210401652272, "E_corr(DZ)"),
        (res.e_ccsd_corr_large, -0.266661829617, "E_corr(TZ)"),
        (res.e_t_small, -0.002982974026, "E_(T)(DZ)"),
        (res.electronic_energy(), -76.353525139663, "E(hartree-W1)"),
    ];
    for (got, want, label) in pins {
        assert!(
            (got - want).abs() < 1e-7,
            "{label} = {got:.12}, want {want:.12}"
        );
    }

    assert_eq!(
        res.electronic_energy(),
        res.e_hf_cbs + res.e_ccsd_corr_cbs + res.e_t_small
    );
    let t = res.thermo.as_ref().expect("frequencies on by default");
    assert_eq!(t.enthalpy, res.electronic_energy() + t.h_corr);
    assert_eq!(t.gibbs, res.electronic_energy() + t.g_corr);
    assert!(t.h_corr > 0.0);
    assert_eq!(t.freq.frequencies.n_imaginary, 0);
}

#[test]
fn guards_open_shell_and_bad_bases() {
    let mut mol = h2();
    mol = mol.with_multiplicity(3);
    let err = run_w1(&mol, &W1Options::default()).unwrap_err();
    assert!(
        err.contains("closed-shell") && err.contains("multiplicity 3"),
        "unexpected error: {err}"
    );

    let opts = W1Options {
        basis_large: "cc-pv5z".into(),
        ..W1Options::default()
    };
    let err = run_w1(&h2(), &opts).unwrap_err();
    assert!(
        err.contains("unknown basis set") && err.contains("cc-pv5z"),
        "unexpected error: {err}"
    );

    let opts = W1Options {
        basis_small: "6-31g".into(),
        ..W1Options::default()
    };
    let err = run_w1(&h2(), &opts).unwrap_err();
    assert!(err.contains("cardinal number"), "unexpected error: {err}");

    let ag2 = Molecule::from_xyz("2\nag2\nAg 0.0 0.0 0.0\nAg 0.0 0.0 2.6\n").unwrap();
    let opts = W1Options {
        opt_level: "hf/cc-pvtz".into(),
        compute_frequencies: false,
        ..W1Options::default()
    };
    let err = run_w1(&ag2, &opts).unwrap_err();
    assert!(
        err.contains("Z=47") && err.contains("not defined"),
        "unexpected error: {err}"
    );

    let opts = W1Options {
        opt_level: "b3lyp".into(), // no basis
        ..W1Options::default()
    };
    let err = run_w1(&h2(), &opts).unwrap_err();
    assert!(
        err.contains("requires an explicit basis"),
        "unexpected error: {err}"
    );
}
