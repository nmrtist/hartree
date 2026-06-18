use crate::core::error::{HartreeError, Result};
use crate::core::units::ANGSTROM_TO_BOHR;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Element(u32);

pub const MAX_Z: u32 = 118;

impl Element {
    pub fn from_z(z: u32) -> Result<Self> {
        if (1..=MAX_Z).contains(&z) {
            Ok(Element(z))
        } else {
            Err(HartreeError::InvalidAtomicNumber(z))
        }
    }

    pub fn from_symbol(symbol: &str) -> Result<Self> {
        let trimmed = symbol.trim();
        SYMBOLS
            .iter()
            .position(|&candidate| candidate.eq_ignore_ascii_case(trimmed))
            .map(|index| Element(index as u32 + 1))
            .ok_or_else(|| HartreeError::UnknownElement(symbol.to_string()))
    }

    #[inline]
    pub fn z(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn symbol(self) -> &'static str {
        SYMBOLS[(self.0 - 1) as usize]
    }

    #[inline]
    pub fn mass(self) -> f64 {
        MASSES[(self.0 - 1) as usize]
    }

    pub fn covalent_radius(self) -> f64 {
        let z = self.0 as usize;
        let angstrom = if z <= COVALENT_RADII_ANGSTROM.len() {
            COVALENT_RADII_ANGSTROM[z - 1]
        } else {
            1.5
        };
        angstrom * ANGSTROM_TO_BOHR
    }
}

#[rustfmt::skip]
static COVALENT_RADII_ANGSTROM: [f64; 36] = [
    0.31, 0.28,
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
];

#[rustfmt::skip]
static SYMBOLS: [&str; 118] = [
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

#[rustfmt::skip]
static MASSES: [f64; 118] = [
    1.008,        4.002602,
    6.94,         9.0121831,    10.81,        12.011,       14.007,       15.999,       18.998403163, 20.1797,
    22.98976928,  24.305,       26.9815385,   28.085,       30.973761998, 32.06,        35.45,        39.948,
    39.0983,      40.078,       44.955908,    47.867,       50.9415,      51.9961,      54.938044,    55.845,       58.933194,    58.6934,      63.546,       65.38,
    69.723,       72.630,       74.921595,    78.971,       79.904,       83.798,
    85.4678,      87.62,        88.90584,     91.224,       92.90637,     95.95,        98.0,         101.07,       102.90550,    106.42,       107.8682,     112.414,
    114.818,      118.710,      121.760,      127.60,       126.90447,    131.293,
    132.90545196, 137.327,
    138.90547,    140.116,      140.90766,    144.242,      145.0,        150.36,       151.964,      157.25,       158.92535,    162.500,      164.93033,    167.259,      168.93422,    173.045,      174.9668,
    178.49,       180.94788,    183.84,       186.207,      190.23,       192.217,      195.084,      196.966569,   200.592,
    204.38,       207.2,        208.98040,    209.0,        210.0,        222.0,
    223.0,        226.0,
    227.0,        232.0377,     231.03588,    238.02891,    237.0,        244.0,        243.0,        247.0,        247.0,        251.0,        252.0,        257.0,        258.0,        259.0,        266.0,
    267.0,        268.0,        269.0,        270.0,        269.0,        278.0,        281.0,        282.0,        285.0,
    286.0,        289.0,        290.0,        293.0,        294.0,        294.0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_have_matching_length() {
        assert_eq!(SYMBOLS.len(), MAX_Z as usize);
        assert_eq!(MASSES.len(), MAX_Z as usize);
    }

    #[test]
    fn lookup_round_trips() {
        for z in 1..=MAX_Z {
            let element = Element::from_z(z).unwrap();
            let by_symbol = Element::from_symbol(element.symbol()).unwrap();
            assert_eq!(element, by_symbol, "mismatch at Z={z}");
        }
    }

    #[test]
    fn spot_check_known_elements() {
        assert_eq!(Element::from_symbol("H").unwrap().z(), 1);
        assert_eq!(Element::from_symbol("o").unwrap().z(), 8); // case-insensitive
        assert_eq!(Element::from_symbol("Fe").unwrap().z(), 26);
        assert_eq!(Element::from_z(6).unwrap().symbol(), "C");
        assert!((Element::from_symbol("C").unwrap().mass() - 12.011).abs() < 1e-6);
    }

    #[test]
    fn rejects_out_of_range() {
        assert!(Element::from_z(0).is_err());
        assert!(Element::from_z(119).is_err());
        assert!(Element::from_symbol("Xx").is_err());
    }

    #[test]
    fn covalent_radii_are_sane() {
        let h = Element::from_symbol("H").unwrap().covalent_radius();
        assert!((h - 0.31 * ANGSTROM_TO_BOHR).abs() < 1e-9);
        for z in 1..=36u32 {
            let r = Element::from_z(z).unwrap().covalent_radius();
            assert!((0.4..6.0).contains(&r), "Z={z} covalent radius {r} bohr");
        }
        let oh = Element::from_symbol("O").unwrap().covalent_radius()
            + Element::from_symbol("H").unwrap().covalent_radius();
        assert!(1.83 < 1.3 * oh, "O–H (~1.83 bohr) must register as bonded");
        let hh = 2.0 * Element::from_symbol("H").unwrap().covalent_radius();
        assert!(
            2.8 > 1.3 * hh,
            "H···H (~2.8 bohr) must register as non-bonded"
        );
    }
}
