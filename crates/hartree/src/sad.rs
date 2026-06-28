//! Superposition-of-atomic-densities (SAD) initial guess.
//!
//! The molecular guess density is the block diagonal of converged free-atom densities, one
//! per nucleus, with no inter-atomic density. Each atomic block is produced by a cheap
//! single-atom SCF in the *same* orbital basis and with the *same* functional as the target
//! calculation, so a block drops straight onto the corresponding atom's diagonal sub-block
//! of the molecular density with no projection, and — because same-center overlaps are
//! independent of the rest of the molecule — the assembled guess carries exactly the right
//! number of electrons. The molecular SCF then relaxes this superposition to the real
//! density; the guess only sets the starting point. See Van Lenthe, Zwaans, van Dam and
//! Guest, "Starting SCF calculations by superposition of atomic densities",
//! J. Comput. Chem. 27 (2006) 926.

use std::collections::HashMap;

use crate::basis::{AoBasis, BasisSet};
use crate::core::{Atom, Element, Molecule};
use crate::dft::{FunctionalSpec, GridXc};
use crate::integrals::ConventionalProvider;
use crate::job::{Method, alpha_beta_electrons, ecp_setup};
use crate::scf::{Guess, Reference, ScfOptions, Smearing, XcContributor, run_scf_with_xc};

/// Electronic temperature (kelvin) for the open-shell atomic SCFs. A partially filled atomic
/// shell is symmetry-degenerate, so Fermi occupations spread its electrons equally over the
/// degenerate components and recover a spherical density; an integer-occupation open-shell
/// atom would instead localize into oriented lobes and seed a poor, non-spherical guess. The
/// value only has to be warm enough to average the open shell while leaving the core/valence
/// gap (≫ kT) integer-occupied.
const OPEN_SHELL_SMEAR_K: f64 = 3000.0;

/// Assemble the molecular SAD guess density (total, AO basis, row-major `n_ao²`) for `mol`
/// in the working basis. Returns `None` — signalling the caller to fall back to the default
/// guess — if any constituent element is unsupported or its atomic SCF fails to converge, so
/// the guess can never turn a previously working calculation into a failing one.
pub(crate) fn sad_guess_density(
    mol: &Molecule,
    ao: &AoBasis,
    basis: &str,
    method: &Method,
    grid_level: usize,
) -> Option<Vec<f64>> {
    let n = ao.n_ao();
    // Build the atomic densities with the molecule's own functional (or plain Hartree–Fock
    // for the wavefunction methods) so the seed is consistent with the target Fock.
    let spec = match method {
        Method::Dft(spec) => Some(spec),
        _ => None,
    };
    let set = BasisSet::load(basis).ok()?;
    let (starts, sizes) = atom_ao_ranges(mol, ao);

    let mut cache: HashMap<u32, Vec<f64>> = HashMap::new();
    let mut density = vec![0.0; n * n];
    for (a, atom) in mol.atoms.iter().enumerate() {
        if atom.ghost {
            continue; // basis-only center: no nucleus, no electrons — its block stays zero
        }
        let z = atom.element.z();
        if !cache.contains_key(&z) {
            cache.insert(z, atomic_density(&set, z, spec, grid_level)?);
        }
        let block = &cache[&z];
        let sz = sizes[a];
        if block.len() != sz * sz {
            return None; // atomic and molecular AO block sizes disagree — bail to the fallback
        }
        let off = starts[a];
        for i in 0..sz {
            for j in 0..sz {
                density[(off + i) * n + (off + j)] = block[i * sz + j];
            }
        }
    }
    Some(density)
}

/// Converged total density of the neutral free atom of element `z` in `set`, or `None` if the
/// element has no ground-state assignment, no basis, or the atomic SCF does not converge.
fn atomic_density(
    set: &BasisSet,
    z: u32,
    spec: Option<&FunctionalSpec>,
    grid_level: usize,
) -> Option<Vec<f64>> {
    let multiplicity = ground_state_multiplicity(z)?;
    let element = Element::from_z(z).ok()?;
    let mol = Molecule::new(vec![Atom::new(element, [0.0; 3])], 0, multiplicity);
    mol.validate().ok()?;

    let ao = set.build(&mol).ok()?;
    let setup = ecp_setup(&mol, &ao);
    let (n_alpha, n_beta) = alpha_beta_electrons(&mol, ao.ecp_core_electrons() as i64).ok()?;
    // Closed-shell atom → restricted; open shell → unrestricted, the only reference that
    // admits the fractional occupations used below to sphericalize the open shell.
    let reference = if n_alpha == n_beta {
        Reference::Rhf
    } else {
        Reference::Uhf
    };
    let smearing = (n_alpha != n_beta).then_some(Smearing::Fermi {
        temperature_k: OPEN_SHELL_SMEAR_K,
    });

    let xc = match spec {
        Some(spec) => Some(GridXc::new(&mol, &ao, spec, grid_level).ok()?),
        None => None,
    };
    let xc_ref = xc.as_ref().map(|g| g as &dyn XcContributor);

    let options = ScfOptions {
        // Never `Sad` here: the atomic SCF is the base case that builds the guess itself.
        guess: Guess::Gwh,
        smearing,
        max_iter: 200,
        energy_tol: 1e-8,
        error_tol: 1e-6,
        ..ScfOptions::default()
    };
    let provider =
        ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
    // A single atom sits at the origin, so the nuclear repulsion is zero.
    let scf = run_scf_with_xc(&provider, n_alpha, n_beta, reference, 0.0, &options, xc_ref).ok()?;
    scf.converged.then_some(scf.density)
}

/// Per-atom AO block ranges in the molecular basis: `(starts, sizes)`, each indexed by atom.
/// AOs are laid out contiguously per atom in `mol.atoms` order, so a block is `[start, start +
/// size)`. `size` is summed from each shell's component count rather than differenced from the
/// next atom's offset, so atoms carrying no basis functions are handled correctly.
fn atom_ao_ranges(mol: &Molecule, ao: &AoBasis) -> (Vec<usize>, Vec<usize>) {
    let natoms = mol.atoms.len();
    let shell_atom = ao.shell_atom();
    let ao_offset = ao.ao_offset();
    let shells = ao.shells();
    let mut starts = vec![0usize; natoms];
    let mut sizes = vec![0usize; natoms];
    let mut seen = vec![false; natoms];
    for (si, &a) in shell_atom.iter().enumerate() {
        if !seen[a] {
            starts[a] = ao_offset[si];
            seen[a] = true;
        }
        sizes[a] += shell_nfunc(shells[si].l, shells[si].spherical);
    }
    (starts, sizes)
}

/// Number of basis functions in a shell of angular momentum `l`: `2l+1` spherical (real solid
/// harmonics) or `(l+1)(l+2)/2` Cartesian.
fn shell_nfunc(l: u32, spherical: bool) -> usize {
    let l = l as usize;
    if spherical {
        2 * l + 1
    } else {
        (l + 1) * (l + 2) / 2
    }
}

/// Ground-state spin multiplicity (2S+1) of the neutral atom for Z = 1..=86 (H..Rn), from the
/// experimental ground terms — Hund's rules with the standard d/f anomalies (e.g. Cr ⁷S,
/// Cu ²S, Nb ⁶D, Mo ⁷S, Ru ⁵F, Rh ⁴F, Pd ¹S, Eu ⁸S, Gd ⁹D, Pt ³D, Au ²S). `None` outside the
/// range steers the caller to the fallback guess. This only selects the open-shell occupation
/// of the atomic SCF; even an imperfect choice still yields a valid, electron-count-correct
/// seed, and the parity of (Z, multiplicity) is checked by `Molecule::validate`.
fn ground_state_multiplicity(z: u32) -> Option<u32> {
    // Indexed by Z (1-based); index 0 is an unused placeholder.
    const MULT: [u8; 87] = [
        0, // (unused)
        2, 1, // H  He
        2, 1, 2, 3, 4, 3, 2, 1, // Li Be B  C  N  O  F  Ne
        2, 1, 2, 3, 4, 3, 2, 1, // Na Mg Al Si P  S  Cl Ar
        2, 1, // K  Ca
        2, 3, 4, 7, 6, 5, 4, 3, 2, 1, // Sc Ti V  Cr Mn Fe Co Ni Cu Zn
        2, 3, 4, 3, 2, 1, // Ga Ge As Se Br Kr
        2, 1, // Rb Sr
        2, 3, 6, 7, 6, 5, 4, 1, 2, 1, // Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd
        2, 3, 4, 3, 2, 1, // In Sn Sb Te I  Xe
        2, 1, // Cs Ba
        2, // La
        3, 4, 5, 6, 7, 8, 9, 6, 5, 4, 3, 2, 1, 2, // Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
        3, 4, 5, 6, 5, 4, 3, 2, 1, // Hf Ta W  Re Os Ir Pt Au Hg
        2, 3, 4, 3, 2, 1, // Tl Pb Bi Po At Rn
    ];
    MULT.get(z as usize)
        .copied()
        .filter(|&m| m != 0)
        .map(u32::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Molecule;
    use crate::scf::run_scf_with_xc;

    // Energy-neutrality: the SAD density guess and the extended-Hückel guess must reach the
    // same SCF fixed point. Converged tightly (so the comparison is not limited by the
    // production termination threshold), their energies must agree to well under the oracle
    // tolerance — the guess moves only the starting point.
    fn same_fixed_point(xyz: &str, basis: &str, functional: &str, level: usize) -> f64 {
        let mol = Molecule::from_xyz(xyz).unwrap();
        let spec = FunctionalSpec::parse(functional).unwrap();
        let ao = BasisSet::load(basis).unwrap().build(&mol).unwrap();
        let xc = GridXc::new(&mol, &ao, &spec, level).unwrap();
        let xc_ref = Some(&xc as &dyn XcContributor);
        let setup = ecp_setup(&mol, &ao);
        let nr = setup.nuclear_repulsion;
        let (na, nb) = alpha_beta_electrons(&mol, ao.ecp_core_electrons() as i64).unwrap();
        let total = sad_guess_density(&mol, &ao, basis, &Method::Dft(spec), level).unwrap();
        let half: Vec<f64> = total.iter().map(|v| 0.5 * v).collect();
        let provider =
            ConventionalProvider::new(ao.into_integral(), setup.charges).with_ecps(setup.ecps);
        let tight = ScfOptions {
            energy_tol: 1e-12,
            error_tol: 1e-9,
            ..ScfOptions::default()
        };
        let run = |opts: &ScfOptions| {
            run_scf_with_xc(&provider, na, nb, Reference::Rhf, nr, opts, xc_ref)
                .unwrap()
                .energy
        };
        let e_gwh = run(&ScfOptions {
            guess: Guess::Gwh,
            ..tight.clone()
        });
        let e_sad = run(&ScfOptions {
            guess: Guess::Sad,
            initial_density: Some((half.clone(), half)),
            ..tight
        });
        e_gwh - e_sad
    }

    #[test]
    fn sad_and_gwh_share_the_fixed_point() {
        // A hybrid on a small basis (the b3lyp-3c fixture case) and a pure meta-GGA on a
        // larger one (the r2scan-3c case), where the guesses converge most differently.
        let water =
            "3\n\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";
        let d_b3lyp = same_fixed_point(water, "def2-msvp", "b3lyp5", 3);
        assert!(
            d_b3lyp.abs() < 1e-9,
            "b3lyp5/def2-msvp water: SAD and GWH fixed points differ by {d_b3lyp:.2e}"
        );
        let d_r2scan = same_fixed_point(water, "def2-mtzvpp", "r2scan", 3);
        assert!(
            d_r2scan.abs() < 1e-9,
            "r2scan/def2-mtzvpp water: SAD and GWH fixed points differ by {d_r2scan:.2e}"
        );
    }
}
