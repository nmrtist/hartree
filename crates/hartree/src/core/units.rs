pub const ANGSTROM_TO_BOHR: f64 = 1.889_726_124_625_770_4;

pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;

pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;

pub const HARTREE_TO_KCAL_MOL: f64 = 627.509_474_063;

pub const HARTREE_TO_KJ_MOL: f64 = 2_625.499_639_5;

pub const HARTREE_TO_WAVENUMBER: f64 = 219_474.631_363_2;

pub const HARTREE_TO_KELVIN: f64 = 315_775.024_804;

pub const AU_DIPOLE_TO_DEBYE: f64 = 2.541_747_100;

pub const FREQ_CONV_CM1: f64 = 5_140.487;

pub const BOLTZMANN_HT: f64 = 3.166_811_563_e-6;

pub const STANDARD_PRESSURE_PA: f64 = 101_325.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_conversions_are_inverse() {
        let round_trip = ANGSTROM_TO_BOHR * BOHR_TO_ANGSTROM;
        assert!((round_trip - 1.0).abs() < 1e-9, "round trip = {round_trip}");
    }
}
