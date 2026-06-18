//! Moller-Plesset and coupled cluster: MO integral transform, MP2, CCSD, and CCSD(T).

pub mod ccsd;
pub mod mp2;
pub mod ri_mp2;
pub mod transform;

pub use ccsd::{
    CcsdOptions, CcsdResult, CcsdTResult, rccsd_spin_adapted, rccsd_spin_orbital,
    rccsd_t_spin_adapted,
};
pub use mp2::{Mp2Result, rhf_mp2, uhf_mp2};
pub use ri_mp2::{
    RiMp2B, RiMp2Error, RiMp2Result, rhf_ri_mp2, rhf_ri_mp2_b, uhf_ri_mp2, uhf_ri_mp2_b,
};
pub use transform::{column_block, core_hamiltonian_mo, transform_block};

use crate::core::Molecule;

pub fn frozen_core_orbitals(molecule: &Molecule) -> usize {
    molecule
        .atoms
        .iter()
        .map(|a| core_orbitals(a.z_eff()))
        .sum()
}

fn core_orbitals(z: u32) -> usize {
    match z {
        0..=2 => 0,    // (none)
        3..=10 => 1,   // He core: 1s
        11..=18 => 5,  // Ne core: 1s2s2p
        19..=36 => 9,  // Ar core: + 3s3p
        37..=54 => 18, // Kr core: + 3d4s4p
        55..=86 => 27, // Xe core: + 4d5s5p
        _ => 43,       // Rn core: + 4f5d6s6p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    fn mol(symbols: &[&str]) -> Molecule {
        let atoms = symbols
            .iter()
            .map(|s| Atom::new(Element::from_symbol(s).unwrap(), [0.0, 0.0, 0.0]))
            .collect();
        Molecule::new(atoms, 0, 1)
    }

    #[test]
    fn frozen_core_counts() {
        assert_eq!(frozen_core_orbitals(&mol(&["H", "H"])), 0);
        assert_eq!(frozen_core_orbitals(&mol(&["O", "H", "H"])), 1); // O 1s
        assert_eq!(frozen_core_orbitals(&mol(&["C", "O"])), 2); // C 1s + O 1s
        assert_eq!(frozen_core_orbitals(&mol(&["S", "H", "H"])), 5); // Ne core on S
    }

    #[test]
    fn ghosts_freeze_nothing() {
        let mut m = mol(&["O", "H", "H"]);
        m.atoms.push(Atom::new_ghost(
            Element::from_symbol("O").unwrap(),
            [0.0, 0.0, 3.0],
        ));
        assert_eq!(frozen_core_orbitals(&m), 1);
    }
}
