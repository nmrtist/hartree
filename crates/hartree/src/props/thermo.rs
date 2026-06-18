use std::f64::consts::PI;

use crate::core::{Molecule, units::BOLTZMANN_HT};
use crate::linalg::symmetric_eigh;

use crate::props::frequencies::FrequencyResult;

const KB_J: f64 = 1.380_649e-23; // J/K (exact)
const H_J: f64 = 6.626_070_15e-34; // J·s (exact)
const C_CM: f64 = 2.997_924_58e10; // cm/s (exact)
const AMU_KG: f64 = 1.660_539_066_60e-27; // kg/amu (CODATA 2018)
const BOHR_M: f64 = 5.291_772_109_03e-11; // m/bohr (CODATA 2018)
const EH_J: f64 = 4.359_744_722_207_1e-18; // J/Eh (CODATA 2018)
const STD_P_PA: f64 = 101_325.0; // Pa (1 atm)

pub const QRRHO_W0_DEFAULT_CM1: f64 = 100.0;

const B_AV_KGM2: f64 = 1.0e-44;

#[derive(Debug, Clone)]
pub struct ThermoResult {
    pub temperature: f64,
    pub symmetry_number: u32,
    pub zpe: f64,
    pub thermal_energy_corr: f64,
    pub enthalpy_corr: f64,
    pub enthalpy: f64,
    pub entropy: f64,
    pub gibbs: f64,
    pub vib_frequencies_cm1: Vec<f64>,
    pub moments_of_inertia: Vec<f64>,
    pub is_linear: bool,
    pub qrrho_w0_cm1: f64,
    pub entropy_qrrho: f64,
    pub gibbs_qrrho: f64,
}

pub fn qrrho_weight(nu_cm1: f64, w0_cm1: f64) -> f64 {
    1.0 / (1.0 + (w0_cm1 / nu_cm1).powi(4))
}

pub fn harmonic_mode_entropy(nu_cm1: f64, temperature: f64) -> f64 {
    let kt_j = KB_J * temperature;
    let x = H_J * nu_cm1 * C_CM / kt_j;
    KB_J * (x / (x.exp() - 1.0) - (1.0 - (-x).exp()).ln()) / EH_J
}

pub fn free_rotor_mode_entropy(nu_cm1: f64, temperature: f64) -> f64 {
    let mu = H_J / (8.0 * PI * PI * C_CM * nu_cm1); // kg·m²
    let mu_eff = mu * B_AV_KGM2 / (mu + B_AV_KGM2);
    let q2 = 8.0 * PI.powi(3) * mu_eff * KB_J * temperature / (H_J * H_J);
    KB_J * (0.5 + (q2.sqrt()).ln()) / EH_J
}

pub fn qrrho_mode_entropy(nu_cm1: f64, temperature: f64, w0_cm1: f64) -> f64 {
    let w = qrrho_weight(nu_cm1, w0_cm1);
    w * harmonic_mode_entropy(nu_cm1, temperature)
        + (1.0 - w) * free_rotor_mode_entropy(nu_cm1, temperature)
}

pub fn rrho_thermochemistry(
    molecule: &Molecule,
    freq_result: &FrequencyResult,
    electronic_energy: f64,
    temperature: f64,
    symmetry_number: u32,
    multiplicity: u32,
) -> ThermoResult {
    rrho_thermochemistry_w0(
        molecule,
        freq_result,
        electronic_energy,
        temperature,
        symmetry_number,
        multiplicity,
        QRRHO_W0_DEFAULT_CM1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rrho_thermochemistry_w0(
    molecule: &Molecule,
    freq_result: &FrequencyResult,
    electronic_energy: f64,
    temperature: f64,
    symmetry_number: u32,
    multiplicity: u32,
    qrrho_w0_cm1: f64,
) -> ThermoResult {
    let kt_j = KB_J * temperature; // k_B T in Joules
    let kt_eh = BOLTZMANN_HT * temperature; // k_B T in hartree

    let (moments, is_linear) = principal_moments(molecule);

    let total_mass_kg: f64 = molecule
        .atoms
        .iter()
        .map(|a| a.element.mass() * AMU_KG)
        .sum();

    let q_trans = (2.0 * PI * total_mass_kg * kt_j).powf(1.5) * kt_j / (H_J.powi(3) * STD_P_PA);
    let s_trans_j = KB_J * (2.5 + q_trans.ln()); // Sackur–Tetrode
    let u_trans_j = 1.5 * kt_j;

    let (_q_rot, s_rot_j, u_rot_j) = if is_linear {
        let i_si = moments[2] * AMU_KG * BOHR_M * BOHR_M; // kg·m²
        let q = 8.0 * PI * PI * i_si * kt_j / (H_J * H_J * symmetry_number as f64);
        let s = KB_J * (1.0 + q.ln());
        let u = kt_j;
        (q, s, u)
    } else {
        let i_si: Vec<f64> = moments
            .iter()
            .map(|m| m * AMU_KG * BOHR_M * BOHR_M)
            .collect();
        let prod_i = i_si[0] * i_si[1] * i_si[2];
        let q = PI.sqrt() / symmetry_number as f64
            * (8.0 * PI * PI * kt_j / (H_J * H_J)).powf(1.5)
            * prod_i.sqrt();
        let s = KB_J * (1.5 + q.ln());
        let u = 1.5 * kt_j;
        (q, s, u)
    };

    let n_trans_rot = if is_linear { 5 } else { 6 };
    let vib_cm1: Vec<f64> = freq_result
        .frequencies_cm1
        .iter()
        .skip(n_trans_rot)
        .copied()
        .filter(|&f| f > 0.0) // defensive: ignore any residual imaginary mode
        .collect();

    let mut zpe_j = 0.0f64;
    let mut u_vib_thermal_j = 0.0f64;
    let mut s_vib_j = 0.0f64;
    let mut s_vib_qrrho = 0.0f64; // hartree/K (the per-mode helpers convert)
    for &nu_cm1 in &vib_cm1 {
        let nu_hz = nu_cm1 * C_CM; // Hz
        let x = H_J * nu_hz / kt_j; // dimensionless h ν / k_B T
        zpe_j += 0.5 * H_J * nu_hz;
        u_vib_thermal_j += H_J * nu_hz / (x.exp() - 1.0);
        s_vib_j += KB_J * (x / (x.exp() - 1.0) - (1.0 - (-x).exp()).ln());
        s_vib_qrrho += qrrho_mode_entropy(nu_cm1, temperature, qrrho_w0_cm1);
    }

    let s_elec_j = KB_J * (multiplicity as f64).ln();

    let to_eh = |j: f64| j / EH_J;
    let zpe = to_eh(zpe_j);
    let u_vib_thermal = to_eh(u_vib_thermal_j);
    let u_rot = to_eh(u_rot_j);
    let u_trans = to_eh(u_trans_j);
    let s_vib = to_eh(s_vib_j);
    let s_rot = to_eh(s_rot_j);
    let s_trans = to_eh(s_trans_j);
    let s_elec = to_eh(s_elec_j);

    let thermal_energy_corr = zpe + u_vib_thermal + u_rot + u_trans;
    let enthalpy_corr = thermal_energy_corr + kt_eh; // PV = k_B T for ideal gas
    let enthalpy = electronic_energy + enthalpy_corr;
    let entropy = s_vib + s_rot + s_trans + s_elec;
    let gibbs = enthalpy - temperature * entropy;
    let entropy_qrrho = s_vib_qrrho + s_rot + s_trans + s_elec;
    let gibbs_qrrho = enthalpy - temperature * entropy_qrrho;

    ThermoResult {
        temperature,
        symmetry_number,
        zpe,
        thermal_energy_corr,
        enthalpy_corr,
        enthalpy,
        entropy,
        gibbs,
        vib_frequencies_cm1: vib_cm1,
        moments_of_inertia: moments,
        is_linear,
        qrrho_w0_cm1,
        entropy_qrrho,
        gibbs_qrrho,
    }
}

fn principal_moments(molecule: &Molecule) -> (Vec<f64>, bool) {
    let masses: Vec<f64> = molecule.atoms.iter().map(|a| a.element.mass()).collect();
    let total_mass: f64 = masses.iter().sum();

    let mut com = [0.0f64; 3];
    for (i, atom) in molecule.atoms.iter().enumerate() {
        for (k, c) in com.iter_mut().enumerate() {
            *c += masses[i] * atom.position[k];
        }
    }
    for c in &mut com {
        *c /= total_mass;
    }

    let mut imat = [[0.0f64; 3]; 3];
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let r = [
            atom.position[0] - com[0],
            atom.position[1] - com[1],
            atom.position[2] - com[2],
        ];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let m = masses[i];
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                imat[a][b] += m * (delta * r2 - r[a] * r[b]);
            }
        }
    }

    let flat: Vec<f64> = (0..3)
        .flat_map(|i| (0..3).map(move |j| imat[i][j]))
        .collect();
    let imat_fmat = crate::linalg::mat_from_row_major(3, &flat);
    let eigh = symmetric_eigh(&imat_fmat);

    let moments: Vec<f64> = eigh.values.iter().map(|&v| v.max(0.0)).collect();
    let is_linear = moments[0] < 1e-4;

    (moments, is_linear)
}

#[cfg(test)]
mod qrrho_tests {
    use super::*;

    const TOL_S: f64 = 0.01 * 4.184 / 6.022_140_76e23 / EH_J;
    const T: f64 = 298.15;
    const W0: f64 = 100.0;

    #[test]
    fn weight_function_exact_values() {
        assert!((qrrho_weight(W0, W0) - 0.5).abs() < 1e-14);
        assert!(qrrho_weight(1e4, W0) > 0.999_999);
        assert!(qrrho_weight(1.0, W0) < 1e-7);
    }

    #[test]
    fn high_frequency_limit_is_harmonic() {
        for nu in [1000.0, 2000.0, 4000.0] {
            let d = (qrrho_mode_entropy(nu, T, W0) - harmonic_mode_entropy(nu, T)).abs();
            assert!(d < TOL_S, "nu={nu}: |S_qRRHO - S_HO| = {d:.3e} Eh/K");
        }
    }

    #[test]
    fn low_frequency_limit_is_free_rotor() {
        for nu in [0.5, 1.0, 2.0] {
            let s_q = qrrho_mode_entropy(nu, T, W0);
            let s_fr = free_rotor_mode_entropy(nu, T);
            assert!(
                (s_q - s_fr).abs() < 10.0 * TOL_S,
                "nu={nu}: S_qRRHO={s_q:.3e} vs S_FR={s_fr:.3e}"
            );
            assert!(s_q.is_finite() && s_fr.is_finite());
        }
    }

    #[test]
    fn qrrho_entropy_monotone_and_continuous() {
        let mut prev = f64::INFINITY;
        let mut nu = 0.5;
        while nu < 3000.0 {
            let s = qrrho_mode_entropy(nu, T, W0);
            assert!(s.is_finite() && s > 0.0, "nu={nu}: S={s}");
            assert!(s < prev, "nu={nu}: S not decreasing ({s:.6e} ≥ {prev:.6e})");
            if prev.is_finite() {
                assert!(
                    prev - s < 0.12 * prev,
                    "nu={nu}: jump {:.3e} (prev {prev:.3e})",
                    prev - s
                );
            }
            prev = s;
            nu += 0.5;
        }
    }
}
