use hartree::core::Molecule;
use hartree::disp::{D3Params, D4Params, Dispersion, GcpParams, SrbParams};
use hartree::{Job, JobOptions, Method};

const WATER: &str = "3\nwater\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";

const WATER_GHOST_O: &str = "4\nwater + Gh(O)\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\nGh(O) 0 0 3.0\n";

pub const WATER_DIMER: &str = "6\nwater dimer\n\
O -1.551007 -0.114520  0.000000\n\
H -1.934259  0.762503  0.000000\n\
H -0.599677  0.040712  0.000000\n\
O  1.350625  0.111469  0.000000\n\
H  1.680398 -0.373741 -0.758561\n\
H  1.680398 -0.373741  0.758561\n";

fn rhf(mol: Molecule, basis: &str) -> hartree::JobResult {
    Job {
        molecule: mol,
        basis: basis.into(),
        method: Method::Rhf,
        options: JobOptions::default(),
    }
    .run()
    .unwrap()
}

#[test]
fn ghost_o_lowers_water_scf_variationally() {
    let bare = rhf(Molecule::from_xyz(WATER).unwrap(), "sto-3g");
    let ghosted = rhf(Molecule::from_xyz(WATER_GHOST_O).unwrap(), "sto-3g");
    assert!(bare.scf.converged && ghosted.scf.converged);
    assert_eq!(
        ghosted.scf.n_alpha + ghosted.scf.n_beta,
        bare.scf.n_alpha + bare.scf.n_beta,
        "ghost contributed electrons"
    );
    assert!(
        (ghosted.scf.nuclear_repulsion - bare.scf.nuclear_repulsion).abs() < 1e-12,
        "ghost contributed nuclear repulsion"
    );
    assert_eq!(ghosted.scf.n_basis, bare.scf.n_basis + 5); // STO-3G O: 1s,2s,2p
    assert!(
        ghosted.scf.energy < bare.scf.energy,
        "variational bound violated: {} !< {}",
        ghosted.scf.energy,
        bare.scf.energy
    );
}

#[test]
fn ghost_only_molecule_rejected() {
    let mol = Molecule::from_xyz("2\nghosts\nGh(O) 0 0 0\nGh(H) 0 0 1\n").unwrap();
    assert!(mol.validate().is_err());
    let err = Job {
        molecule: mol,
        basis: "sto-3g".into(),
        method: Method::Rhf,
        options: JobOptions::default(),
    }
    .run()
    .unwrap_err();
    assert!(err.contains("ghost-only"), "unexpected error: {err}");
}

#[test]
fn pairwise_corrections_exclude_ghosts() {
    let bare = Molecule::from_xyz(WATER).unwrap();
    let ghosted = Molecule::from_xyz(WATER_GHOST_O).unwrap();

    let d3 = Dispersion::D3(D3Params::B3LYP_3C);
    let d4 = Dispersion::D4(D4Params::for_method("pbe").unwrap());
    assert_eq!(d3.energy(&bare), d3.energy(&ghosted));
    assert_eq!(d4.energy(&bare), d4.energy(&ghosted));
    assert_eq!(
        hartree::disp::gcp_energy(&bare, &GcpParams::R2SCAN_3C),
        hartree::disp::gcp_energy(&ghosted, &GcpParams::R2SCAN_3C)
    );
    assert_eq!(
        hartree::disp::srb_energy(&bare, &SrbParams::B97_3C),
        hartree::disp::srb_energy(&ghosted, &SrbParams::B97_3C)
    );

    let (e_b, g_b) = d3.energy_gradient(&bare);
    let (e_g, g_g) = d3.energy_gradient(&ghosted);
    assert_eq!(e_b, e_g);
    assert_eq!(g_g.len(), 4);
    assert_eq!(&g_g[..3], &g_b[..]);
    assert_eq!(g_g[3], [0.0; 3]);
    let (_, gg) = hartree::disp::gcp_energy_gradient(&ghosted, &GcpParams::R2SCAN_3C);
    assert_eq!(gg[3], [0.0; 3]);
}

#[test]
fn monomer_in_dimer_basis_is_variationally_lower() {
    let dimer = Molecule::from_xyz(WATER_DIMER).unwrap();
    let a_alone = Molecule::new(dimer.atoms[..3].to_vec(), 0, 1);
    let mut a_in_ab = a_alone.clone();
    for atom in &dimer.atoms[3..] {
        a_in_ab
            .atoms
            .push(hartree::Atom::new_ghost(atom.element, atom.position));
    }
    let e_a = rhf(a_alone, "sto-3g").scf.energy;
    let e_a_ab = rhf(a_in_ab, "sto-3g").scf.energy;
    assert!(
        e_a_ab <= e_a,
        "variational bound violated: E_A^(AB) = {e_a_ab} > E_A^(A) = {e_a}"
    );
    assert!(e_a_ab < e_a - 1e-6, "ghost basis had no effect at all");
}

#[test]
fn mp2_frozen_core_ignores_ghosts() {
    let ghosted = Molecule::from_xyz(WATER_GHOST_O).unwrap();
    assert_eq!(hartree::cc::frozen_core_orbitals(&ghosted), 1); // O 1s only
    let result = Job {
        molecule: ghosted,
        basis: "sto-3g".into(),
        method: Method::Mp2,
        options: JobOptions::default(),
    }
    .run()
    .unwrap();
    let post = result.post_hf.unwrap();
    assert_eq!(post.n_frozen(), 1);
    assert!(post.total_energy() < result.scf.energy);
}

#[test]
fn def2_svpd_has_diffuse_functions_and_runs() {
    let water = Molecule::from_xyz(WATER).unwrap();
    let nao = |name: &str| {
        hartree::BasisSet::load(name)
            .unwrap()
            .build(&water)
            .unwrap()
            .n_ao()
    };
    let (svp, svpd) = (nao("def2-svp"), nao("def2-svpd"));
    assert!(
        svpd > svp,
        "def2-SVPD ({svpd} AOs) must exceed def2-SVP ({svp} AOs)"
    );
    let result = rhf(water, "def2-svpd");
    assert!(result.scf.converged);
    assert!(result.scf.energy < -75.9, "implausible SVPD water energy");
}

#[test]
fn ghost_guards() {
    let run = |f: fn(&mut JobOptions)| {
        let mut options = JobOptions::default();
        f(&mut options);
        Job {
            molecule: Molecule::from_xyz(WATER_GHOST_O).unwrap(),
            basis: "sto-3g".into(),
            method: Method::Rhf,
            options,
        }
        .run()
        .unwrap_err()
    };
    assert!(run(|o| o.optimize_geometry = true).contains("ghost"));
    assert!(run(|o| o.compute_frequencies = true).contains("ghost"));
    assert!(run(|o| o.compute_properties = true).contains("ghost"));
    assert!(run(|o| o.solvent_eps = Some(78.0)).contains("ghost"));
}
