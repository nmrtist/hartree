//! Basis-set loading: Basis Set Exchange JSON, def2 ECPs, and GTH basis/pseudopotential data.

mod bse;
mod ecp;
mod error;
pub mod gth;
pub mod gth_basis;

use std::collections::HashMap;

use crate::core::Molecule;

pub use ecp::{EcpPrimitive, EcpSet, ElementEcp};
pub use error::{BasisError, Result};
pub use gth::{GthLocal, GthNonlocal, GthPotential, GthSet};
pub use gth_basis::{GthBasisSet, GthBasisShell, GthElementBasis};

#[derive(Debug, Clone, PartialEq)]
pub struct ContractedShell {
    pub l: u32,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShellData {
    pub l: u32,
    pub center: [f64; 3],
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub spherical: bool,
}

#[derive(Debug, Clone)]
pub struct BasisSet {
    pub name: String,
    pub version: String,
    pub spherical: bool,
    elements: HashMap<u32, Vec<ContractedShell>>,
    ecp: Option<EcpSet>,
}

macro_rules! bundled {
    ($name:literal) => {
        include_str!(concat!("data/basis/", $name, ".json"))
    };
}

impl BasisSet {
    pub fn load(name: &str) -> Result<Self> {
        let lower = name.to_ascii_lowercase();
        let json = match lower.as_str() {
            "sto-3g" => bundled!("sto-3g"),
            "6-31g" => bundled!("6-31g"),
            "6-31g(d)" | "6-31g*" => bundled!("6-31g(d)"),
            "6-311g" => bundled!("6-311g"),
            "6-311g(d,p)" => bundled!("6-311g(d,p)"),
            "6-311+g(d,p)" => bundled!("6-311+g(d,p)"),
            "6-311++g(d,p)" => bundled!("6-311++g(d,p)"),
            "cc-pvdz" => bundled!("cc-pvdz"),
            "cc-pvtz" => bundled!("cc-pvtz"),
            "def2-tzvp" => bundled!("def2-tzvp"),
            "aug-cc-pvtz" => bundled!("aug-cc-pvtz"),
            "cc-pvqz" => bundled!("cc-pvqz"),
            "def2-qzvp" => bundled!("def2-qzvp"),
            "def2-svp" => bundled!("def2-svp"),
            "def2-tzvpp" => bundled!("def2-tzvpp"),
            "def2-qzvpp" => bundled!("def2-qzvpp"),
            "def2-tzvpd" => bundled!("def2-tzvpd"),
            "def2-tzvppd" => bundled!("def2-tzvppd"),
            "def2-svpd" => bundled!("def2-svpd"),
            "ma-def2-svp" => {
                return Ok(Self::load("def2-svp")?.minimally_augmented("ma-def2-SVP"));
            }
            "ma-def2-tzvp" => {
                return Ok(Self::load("def2-tzvp")?.minimally_augmented("ma-def2-TZVP"));
            }
            "def2-mtzvpp" => bundled!("def2-mtzvpp"),
            "def2-mtzvp" | "mtzvp" => bundled!("def2-mtzvp"),
            "def2-msvp" => bundled!("def2-msvp"),
            "def2-universal-jkfit" | "def2-svp/c" | "def2-tzvp/c" => {
                return Err(BasisError::AuxiliaryAsOrbital(name.to_string()));
            }
            _ => return Err(BasisError::UnknownSet(name.to_string())),
        };
        let mut set = Self::from_bse_json(json)?;

        let heavy = match lower.as_str() {
            "def2-svp" => Some(bundled!("def2-svp.heavy")),
            "def2-tzvp" => Some(bundled!("def2-tzvp.heavy")),
            _ => None,
        };
        if let Some(heavy_json) = heavy {
            set.merge_heavy(heavy_json)?;
        }
        if lower.starts_with("def2-") {
            set.ecp = Some(ecp::parse(include_str!("data/ecp/def2-ecp.json"))?);
        }
        Ok(set)
    }

    fn merge_heavy(&mut self, json: &str) -> Result<()> {
        let heavy = Self::from_bse_json(json)?;
        for (z, shells) in heavy.elements {
            if self.elements.insert(z, shells).is_some() {
                return Err(BasisError::Schema(format!(
                    "heavy-element extension of {:?} redefines Z={z}",
                    self.name
                )));
            }
        }
        Ok(())
    }

    pub fn load_aux(name: &str) -> Result<Self> {
        let json = match name.to_ascii_lowercase().as_str() {
            "def2-universal-jkfit" => bundled!("def2-universal-jkfit"),
            "def2-svp/c" => bundled!("def2-svp-c"),
            "def2-tzvp/c" => bundled!("def2-tzvp-c"),
            _ => return Err(BasisError::UnknownAuxSet(name.to_string())),
        };
        Self::from_bse_json(json)
    }

    pub fn from_bse_json(json: &str) -> Result<Self> {
        bse::parse(json)
    }

    fn minimally_augmented(mut self, name: &str) -> Self {
        for (&z, shells) in self.elements.iter_mut() {
            if z == 1 {
                continue;
            }
            for l in [0u32, 1] {
                let min_exp = shells
                    .iter()
                    .filter(|s| s.l == l)
                    .flat_map(|s| s.exponents.iter().copied())
                    .fold(f64::INFINITY, f64::min);
                if min_exp.is_finite() {
                    shells.push(ContractedShell {
                        l,
                        exponents: vec![min_exp / 3.0],
                        coefficients: vec![1.0],
                    });
                }
            }
        }
        self.name = name.to_string();
        self
    }

    pub fn shells_for(&self, z: u32) -> Option<&[ContractedShell]> {
        self.elements.get(&z).map(Vec::as_slice)
    }

    pub fn ecp_for(&self, z: u32) -> Option<&ElementEcp> {
        self.ecp.as_ref().and_then(|set| set.get(z))
    }

    pub fn ecp_set(&self) -> Option<&EcpSet> {
        self.ecp.as_ref()
    }

    pub fn ecp_core_electrons(&self, molecule: &Molecule) -> u32 {
        molecule
            .atoms
            .iter()
            .filter(|a| !a.ghost)
            .filter_map(|a| self.ecp_for(a.element.z()))
            .map(|e| e.n_core)
            .sum()
    }

    pub fn build(&self, molecule: &Molecule) -> Result<AoBasis> {
        let mut shells = Vec::new();
        let mut shell_atom = Vec::new();
        let mut shell_data = Vec::new();
        let mut ecps: Vec<integral::Ecp> = Vec::new();
        let mut ecp_core = vec![0u32; molecule.len()];

        for (atom_index, atom) in molecule.atoms.iter().enumerate() {
            let z = atom.element.z();
            let defs = match self.shells_for(z) {
                Some(defs) => defs,
                // Z > 36 needs a small-core ECP plus a matching heavy orbital basis.
                // hartree vendors def2-ECP and the def2 heavy orbital split only for
                // Ag/Sn/I/Au, and the split is merged only into def2-SVP/def2-TZVP, so
                // give a heavy-element-aware message instead of the bare "not in set".
                None if z > 36 => {
                    let hint = if self.ecp_for(z).is_some() {
                        // def2 basis: the ECP is loaded but this set carries no heavy shells.
                        format!(
                            "hartree has a def2-ECP for Z={z}, but the heavy def2 orbital \
                             basis is bundled only with def2-SVP and def2-TZVP — rerun with \
                             --basis def2-svp or --basis def2-tzvp"
                        )
                    } else if (37..=86).contains(&z) {
                        // In the def2-ECP range, but this basis family has no ECP at all.
                        format!(
                            "Z={z} is in the def2-ECP range (Rb–Rn) but the {:?} basis is \
                             all-electron only — use --basis def2-svp or --basis def2-tzvp \
                             for small-core ECP support",
                            self.name
                        )
                    } else {
                        // Beyond the def2-ECP range (Z > 86).
                        format!(
                            "Z={z} is beyond hartree's def2-ECP range (Rb 37 – Rn 86); for \
                             all-electron scalar relativity use --x2c with an all-electron basis"
                        )
                    };
                    return Err(BasisError::UnsupportedHeavyElement {
                        z,
                        set: self.name.clone(),
                        hint,
                    });
                }
                None => {
                    return Err(BasisError::ElementNotInSet {
                        z,
                        set: self.name.clone(),
                    });
                }
            };

            let center = atom.position;
            for def in defs {
                let l = def.l as usize;
                let exps = def.exponents.clone();
                let coeffs = def.coefficients.clone();
                let shell = if self.spherical {
                    integral::Shell::new_spherical(l, center, exps.clone(), coeffs.clone())
                } else {
                    integral::Shell::new(l, center, exps.clone(), coeffs.clone())
                }?;
                shells.push(shell);
                shell_atom.push(atom_index);
                shell_data.push(ShellData {
                    l: def.l,
                    center,
                    exponents: exps,
                    coefficients: coeffs,
                    spherical: self.spherical,
                });
            }

            if atom.ghost {
                continue;
            }
            if let Some(e) = self.ecp_for(z) {
                ecps.push(integral::Ecp {
                    atom: atom_index,
                    n_core: e.n_core,
                    max_l: e.max_l,
                    local: convert_prims(&e.local),
                    semilocal: e.semilocal.iter().map(|c| convert_prims(c)).collect(),
                });
                ecp_core[atom_index] = e.n_core;
            }
        }

        if !ecps.is_empty()
            && let Some(bad) = shell_data.iter().find(|s| s.l > 4)
        {
            return Err(BasisError::Schema(format!(
                "basis set {:?} carries an l={} shell, but ECP integrals \
                 support AO angular momentum l <= 4",
                self.name, bad.l
            )));
        }

        let basis = integral::Basis::new(shells);

        let mut ao_offset = Vec::with_capacity(basis.shells().len());
        let mut acc = 0;
        for shell in basis.shells() {
            ao_offset.push(acc);
            acc += shell.n_func();
        }

        Ok(AoBasis {
            basis,
            shell_atom,
            ao_offset,
            n_ao: acc,
            shell_data,
            ecps,
            ecp_core,
        })
    }
}

fn convert_prims(prims: &[EcpPrimitive]) -> Vec<integral::EcpPrimitive> {
    prims
        .iter()
        .map(|p| integral::EcpPrimitive {
            n: p.n,
            zeta: p.zeta,
            coef: p.coef,
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct AoBasis {
    basis: integral::Basis,
    shell_atom: Vec<usize>,
    ao_offset: Vec<usize>,
    n_ao: usize,
    shell_data: Vec<ShellData>,
    ecps: Vec<integral::Ecp>,
    ecp_core: Vec<u32>,
}

impl AoBasis {
    pub fn integral(&self) -> &integral::Basis {
        &self.basis
    }

    pub fn into_integral(self) -> integral::Basis {
        self.basis
    }

    pub fn n_ao(&self) -> usize {
        self.n_ao
    }

    pub fn n_shells(&self) -> usize {
        self.shell_atom.len()
    }

    pub fn shell_atom(&self) -> &[usize] {
        &self.shell_atom
    }

    pub fn ao_offset(&self) -> &[usize] {
        &self.ao_offset
    }

    pub fn shells(&self) -> &[ShellData] {
        &self.shell_data
    }

    pub fn ecps(&self) -> &[integral::Ecp] {
        &self.ecps
    }

    pub fn ecp_core(&self) -> &[u32] {
        &self.ecp_core
    }

    pub fn ecp_core_electrons(&self) -> u32 {
        self.ecp_core.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        Molecule::from_xyz("3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.755 -0.471\nH 0.0 -0.755 -0.471\n")
            .unwrap()
    }

    fn h2() -> Molecule {
        Molecule::from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap()
    }

    const BUNDLED_SETS: [&str; 22] = [
        "sto-3g",
        "6-31g",
        "6-31g(d)",
        "6-311g",
        "6-311g(d,p)",
        "6-311+g(d,p)",
        "6-311++g(d,p)",
        "cc-pvdz",
        "cc-pvtz",
        "cc-pvqz",
        "def2-svp",
        "def2-svpd",
        "def2-tzvp",
        "def2-tzvpp",
        "def2-tzvpd",
        "def2-tzvppd",
        "def2-qzvp",
        "def2-qzvpp",
        "ma-def2-svp",
        "ma-def2-tzvp",
        "aug-cc-pvtz",
        "def2-mtzvp",
    ];

    const DEF2_SETS: [&str; 13] = [
        "def2-msvp",
        "def2-mtzvp",
        "def2-svp",
        "def2-svpd",
        "def2-tzvp",
        "def2-tzvpp",
        "def2-tzvpd",
        "def2-tzvppd",
        "def2-qzvp",
        "def2-qzvpp",
        "def2-mtzvpp",
        "ma-def2-svp",
        "ma-def2-tzvp",
    ];

    #[test]
    fn all_sets_load_with_provenance() {
        for name in BUNDLED_SETS {
            let set = BasisSet::load(name).unwrap();
            assert!(!set.version.is_empty(), "{name} missing version");
            assert!(set.shells_for(1).is_some(), "{name} missing H");
            assert!(set.shells_for(8).is_some(), "{name} missing O");
        }
        assert!(BasisSet::load("not-a-basis").is_err());
    }

    #[test]
    fn spherical_convention_is_per_family() {
        assert!(!BasisSet::load("sto-3g").unwrap().spherical);
        assert!(!BasisSet::load("6-31g").unwrap().spherical);
        // The polarized 6-31 set is spherical (5d), unlike its unpolarized parent; the
        // `*` alias resolves to the same spherical set.
        assert!(BasisSet::load("6-31g(d)").unwrap().spherical);
        assert!(BasisSet::load("6-31g*").unwrap().spherical);
        assert!(BasisSet::load("6-311g").unwrap().spherical);
        assert!(BasisSet::load("6-311g(d,p)").unwrap().spherical);
        assert!(BasisSet::load("6-311+g(d,p)").unwrap().spherical);
        assert!(BasisSet::load("6-311++g(d,p)").unwrap().spherical);
        assert!(BasisSet::load("cc-pvdz").unwrap().spherical);
        assert!(BasisSet::load("cc-pvtz").unwrap().spherical);
        assert!(BasisSet::load("cc-pvqz").unwrap().spherical);
        assert!(BasisSet::load("def2-svp").unwrap().spherical);
        assert!(BasisSet::load("def2-tzvp").unwrap().spherical);
        assert!(BasisSet::load("def2-qzvp").unwrap().spherical);
        assert!(BasisSet::load("aug-cc-pvtz").unwrap().spherical);
        assert!(BasisSet::load("def2-tzvpp").unwrap().spherical);
        assert!(BasisSet::load("def2-qzvpp").unwrap().spherical);
        assert!(BasisSet::load("def2-tzvpd").unwrap().spherical);
        assert!(BasisSet::load("def2-svpd").unwrap().spherical);
    }

    #[test]
    fn water_ao_counts() {
        let mol = water();
        assert_eq!(
            BasisSet::load("sto-3g")
                .unwrap()
                .build(&mol)
                .unwrap()
                .n_ao(),
            7
        );
        assert_eq!(
            BasisSet::load("6-31g").unwrap().build(&mol).unwrap().n_ao(),
            13
        );
        assert_eq!(
            BasisSet::load("cc-pvdz")
                .unwrap()
                .build(&mol)
                .unwrap()
                .n_ao(),
            24
        );
        assert_eq!(
            BasisSet::load("def2-svp")
                .unwrap()
                .build(&mol)
                .unwrap()
                .n_ao(),
            24
        );
    }

    #[test]
    fn h2_ao_counts() {
        let mol = h2();
        assert_eq!(
            BasisSet::load("sto-3g")
                .unwrap()
                .build(&mol)
                .unwrap()
                .n_ao(),
            2
        );
        assert_eq!(
            BasisSet::load("cc-pvdz")
                .unwrap()
                .build(&mol)
                .unwrap()
                .n_ao(),
            10
        );
    }

    #[test]
    fn shell_data_matches_bookkeeping() {
        let mol = water();
        for name in BUNDLED_SETS {
            let set = BasisSet::load(name).unwrap();
            let ao = set.build(&mol).unwrap();
            let shells = ao.shells();
            assert_eq!(shells.len(), ao.n_shells(), "{name} shell count");
            assert_eq!(shells.len(), ao.ao_offset().len(), "{name} offsets");

            let mut acc = 0;
            for (i, sd) in shells.iter().enumerate() {
                assert_eq!(sd.spherical, set.spherical, "{name} spherical flag");
                assert_eq!(sd.exponents.len(), sd.coefficients.len(), "{name} lengths");
                assert_eq!(ao.ao_offset()[i], acc, "{name} offset shell {i}");
                let l = sd.l as usize;
                let n_func = if sd.spherical {
                    2 * l + 1
                } else {
                    (l + 1) * (l + 2) / 2
                };
                acc += n_func;
            }
            assert_eq!(acc, ao.n_ao(), "{name} total AO count");
        }
    }

    #[test]
    fn aux_set_registry_is_separate() {
        let aux = BasisSet::load_aux("def2-universal-jkfit").unwrap();
        assert!(aux.spherical, "JK fitting set must be spherical");
        assert!(!aux.version.is_empty());
        for z in 1..=36 {
            assert!(aux.shells_for(z).is_some(), "aux set missing Z={z}");
        }
        let max_l = (1..=18)
            .flat_map(|z| aux.shells_for(z).unwrap())
            .map(|s| s.l)
            .max()
            .unwrap();
        assert_eq!(max_l, 4, "def2-universal-jkfit max l is g");

        assert!(matches!(
            BasisSet::load("def2-universal-jkfit"),
            Err(BasisError::AuxiliaryAsOrbital(_))
        ));
        assert!(matches!(
            BasisSet::load_aux("cc-pvdz"),
            Err(BasisError::UnknownAuxSet(_))
        ));
    }

    #[test]
    fn mp2_fit_registry() {
        for name in ["def2-svp/c", "def2-tzvp/c"] {
            let aux = BasisSet::load_aux(name).unwrap();
            assert!(aux.spherical, "{name} must be spherical");
            assert!(!aux.version.is_empty(), "{name} missing version");
            for z in 1..=36 {
                assert!(aux.shells_for(z).is_some(), "{name} missing Z={z}");
            }
            assert!(aux.shells_for(37).is_none(), "{name} should stop at Kr");
            assert!(matches!(
                BasisSet::load(name),
                Err(BasisError::AuxiliaryAsOrbital(_))
            ));
        }
        assert!(matches!(
            BasisSet::load_aux("cc-pvdz/c"),
            Err(BasisError::UnknownAuxSet(_))
        ));
    }

    #[test]
    fn mp2_fit_shell_structure() {
        let count_for = |name: &str, z: u32, l: u32| {
            BasisSet::load_aux(name)
                .unwrap()
                .shells_for(z)
                .unwrap()
                .iter()
                .filter(|s| s.l == l)
                .count()
        };
        for (l, expect) in [6, 5, 4, 1].iter().enumerate() {
            assert_eq!(
                count_for("def2-svp/c", 8, l as u32),
                *expect,
                "def2-SVP/C oxygen l={l}"
            );
        }
        for (l, expect) in [3, 2, 1].iter().enumerate() {
            assert_eq!(
                count_for("def2-svp/c", 1, l as u32),
                *expect,
                "def2-SVP/C hydrogen l={l}"
            );
        }
        for (l, expect) in [8, 6, 4, 3, 1].iter().enumerate() {
            assert_eq!(
                count_for("def2-tzvp/c", 8, l as u32),
                *expect,
                "def2-TZVP/C oxygen l={l}"
            );
        }
        for name in ["def2-svp/c", "def2-tzvp/c"] {
            for z in 1..=36u32 {
                for s in BasisSet::load_aux(name).unwrap().shells_for(z).unwrap() {
                    assert_eq!(s.exponents.len(), s.coefficients.len(), "{name} Z={z}");
                }
            }
        }
    }

    #[test]
    fn oxygen_ccpvdz_shell_structure() {
        let set = BasisSet::load("cc-pvdz").unwrap();
        let shells = set.shells_for(8).unwrap();
        let count = |l: u32| shells.iter().filter(|s| s.l == l).count();
        assert_eq!(count(0), 3, "oxygen s shells");
        assert_eq!(count(1), 2, "oxygen p shells");
        assert_eq!(count(2), 1, "oxygen d shells");
    }

    #[test]
    fn oxygen_ccpvtz_has_f_shell() {
        let count_for = |name: &str, l: u32| {
            BasisSet::load(name)
                .unwrap()
                .shells_for(8)
                .unwrap()
                .iter()
                .filter(|s| s.l == l)
                .count()
        };
        assert_eq!(count_for("cc-pvtz", 0), 4, "cc-pVTZ oxygen s shells");
        assert_eq!(count_for("cc-pvtz", 1), 3, "cc-pVTZ oxygen p shells");
        assert_eq!(count_for("cc-pvtz", 2), 2, "cc-pVTZ oxygen d shells");
        assert_eq!(count_for("cc-pvtz", 3), 1, "cc-pVTZ oxygen f shells");
        assert_eq!(
            count_for("aug-cc-pvtz", 3),
            2,
            "aug-cc-pVTZ oxygen f shells"
        );
        assert_eq!(count_for("def2-tzvp", 3), 1, "def2-TZVP oxygen f shells");
    }

    #[test]
    fn oxygen_qz_has_g_shell() {
        let count_for = |name: &str, l: u32| {
            BasisSet::load(name)
                .unwrap()
                .shells_for(8)
                .unwrap()
                .iter()
                .filter(|s| s.l == l)
                .count()
        };
        assert_eq!(count_for("cc-pvqz", 0), 5, "cc-pVQZ oxygen s shells");
        assert_eq!(count_for("cc-pvqz", 1), 4, "cc-pVQZ oxygen p shells");
        assert_eq!(count_for("cc-pvqz", 2), 3, "cc-pVQZ oxygen d shells");
        assert_eq!(count_for("cc-pvqz", 3), 2, "cc-pVQZ oxygen f shells");
        assert_eq!(count_for("cc-pvqz", 4), 1, "cc-pVQZ oxygen g shells");
        assert_eq!(count_for("def2-qzvp", 4), 1, "def2-QZVP oxygen g shells");
    }

    #[test]
    fn oxygen_def2_pp_and_d_shell_structure() {
        let count_for = |name: &str, l: u32| {
            BasisSet::load(name)
                .unwrap()
                .shells_for(8)
                .unwrap()
                .iter()
                .filter(|s| s.l == l)
                .count()
        };
        for (name, counts) in [
            ("def2-tzvpp", vec![5, 3, 2, 1]),
            ("def2-qzvpp", vec![7, 4, 3, 2, 1]),
            ("def2-tzvpd", vec![6, 4, 3, 1]),
            ("def2-svpd", vec![4, 3, 2]),
        ] {
            for (l, expect) in counts.iter().enumerate() {
                assert_eq!(
                    count_for(name, l as u32),
                    *expect,
                    "{name} oxygen l={l} shells"
                );
            }
        }
    }

    #[test]
    fn def2_sets_element_coverage() {
        // The def2-ECP heavy orbital split (Rb 37 - Rn 86) is merged into def2-SVP and
        // def2-TZVP, and so into the ma- sets derived from them; every other def2 orbital
        // set is all-electron H-Kr only.
        const HEAVY_SPLIT: [&str; 4] = ["def2-svp", "def2-tzvp", "ma-def2-svp", "ma-def2-tzvp"];
        for name in DEF2_SETS {
            let set = BasisSet::load(name).unwrap();
            for z in 1..=36 {
                assert!(set.shells_for(z).is_some(), "{name} missing Z={z}");
            }
            if HEAVY_SPLIT.contains(&name) {
                for z in 37..=86 {
                    assert!(set.shells_for(z).is_some(), "{name} missing heavy Z={z}");
                }
                assert!(set.shells_for(87).is_none(), "{name} should stop at Rn");
            } else {
                assert!(set.shells_for(37).is_none(), "{name} should stop at Kr");
            }
        }
        assert!(BasisSet::load("cc-pvdz").unwrap().shells_for(19).is_none());
    }

    #[test]
    fn heavy_element_ao_counts() {
        let atom = |sym: &str| Molecule::from_xyz(&format!("1\natom\n{sym} 0 0 0\n")).unwrap();
        for (name, sym, nao) in [
            ("def2-svp", "Br", 32),    // [5s4p3d]
            ("def2-svp", "Fe", 31),    // [5s3p2d1f]
            ("def2-tzvp", "Fe", 45),   // [6s4p4d1f]
            ("def2-tzvp", "Kr", 48),   // [6s5p4d1f]
            ("def2-qzvp", "Ca", 56),   // [11s6p4d1f]
            ("def2-tzvppd", "O", 40),  // [6s4p3d1f]
            ("def2-tzvppd", "H", 17),  // [3s3p1d]
            ("def2-tzvppd", "Br", 57), // [7s6p5d1f]
            ("def2-svpd", "Br", 41),   // [6s5p4d]
            ("def2-qzvpp", "Kr", 89),  // [11s7p4d4f1g]
            ("def2-mtzvpp", "Fe", 33), // [6s4p3d] (max l = d by construction)
            ("def2-mtzvp", "H", 3),    // [3s] (B97-3c mTZVP: no H polarization)
            ("def2-mtzvp", "C", 19),   // [5s3p1d]
            ("def2-mtzvp", "S", 22),   // [5s4p1d]
        ] {
            let ao = BasisSet::load(name).unwrap().build(&atom(sym)).unwrap();
            assert_eq!(ao.n_ao(), nao, "{name} {sym} AO count");
        }
    }

    #[test]
    fn heavy_ecp_elements_build_and_unsupported_ones_explain_why() {
        let atom = |sym: &str| Molecule::from_xyz(&format!("1\natom\n{sym} 0 0 0\n")).unwrap();

        // Representative vendored def2-ECP elements build on the heavy-split bases
        // (SVP/TZVP), each carrying an ECP with its documented core size -- including a
        // lanthanide (Yb), whose h local channel exercises the raised parser cap.
        for (name, sym, n_core) in [
            ("def2-svp", "Ag", 28u32), // 4d, f-local
            ("def2-tzvp", "I", 28),    // 5p, f-local
            ("def2-svp", "Au", 60),    // 5d, ECP60, f-local
            ("def2-tzvp", "Pb", 60),   // 6p, ECP60, f-local
            ("def2-svp", "Yb", 28),    // 4f lanthanide, ECP28, h-local (L=5)
        ] {
            let ao = BasisSet::load(name).unwrap().build(&atom(sym)).unwrap();
            assert_eq!(ao.ecps().len(), 1, "{name} {sym}: one ECP center");
            assert_eq!(ao.ecp_core_electrons(), n_core, "{name} {sym} core count");
            assert_eq!(ao.ecps()[0].n_core, n_core, "{name} {sym} ECP n_core");
        }

        // Hint branch 1: a vendored ECP element on a def2 basis WITHOUT the heavy orbital
        // split redirects to def2-SVP/def2-TZVP, not the bare "not in set".
        let err = BasisSet::load("def2-mtzvpp")
            .unwrap()
            .build(&atom("Ag"))
            .unwrap_err();
        match err {
            BasisError::UnsupportedHeavyElement { z, ref hint, .. } => {
                assert_eq!(z, 47);
                assert!(hint.contains("def2-svp"), "hint should redirect: {hint}");
            }
            other => panic!("expected UnsupportedHeavyElement, got {other:?}"),
        }

        // Hint branch 2: an in-range heavy element on an all-electron (non-def2) basis
        // points at the def2 ECP bases.
        let err = BasisSet::load("cc-pvdz")
            .unwrap()
            .build(&atom("I"))
            .unwrap_err();
        match err {
            BasisError::UnsupportedHeavyElement { z, ref hint, .. } => {
                assert_eq!(z, 53);
                assert!(
                    hint.contains("def2-svp"),
                    "hint should point at def2: {hint}"
                );
            }
            other => panic!("expected UnsupportedHeavyElement, got {other:?}"),
        }

        // Hint branch 3: beyond the def2-ECP range (Z > 86) explains the limit and --x2c.
        let err = BasisSet::load("def2-svp")
            .unwrap()
            .build(&atom("U"))
            .unwrap_err();
        match err {
            BasisError::UnsupportedHeavyElement { z, ref hint, .. } => {
                assert_eq!(z, 92);
                assert!(hint.contains("--x2c"), "hint should mention --x2c: {hint}");
            }
            other => panic!("expected UnsupportedHeavyElement, got {other:?}"),
        }
    }

    #[test]
    fn heavy_element_overlap_diagonal_is_unit() {
        let mol = Molecule::from_xyz(
            "2\nKBr\nK 0 0 0\nBr 0 0 2.9\n", // any geometry; diagonal is per-AO
        )
        .unwrap();
        for name in DEF2_SETS {
            let ao = BasisSet::load(name).unwrap().build(&mol).unwrap();
            let n = ao.n_ao();
            let s = ao.integral().overlap();
            for i in 0..n {
                let sii = s[i * n + i];
                assert!(
                    (sii - 1.0).abs() < 1e-9,
                    "{name}: S[{i},{i}] = {sii}, expected 1"
                );
            }
        }
    }

    #[test]
    fn ma_sets_are_parent_plus_minimal_diffuse() {
        for (ma_name, parent_name) in [("ma-def2-svp", "def2-svp"), ("ma-def2-tzvp", "def2-tzvp")] {
            let ma = BasisSet::load(ma_name).unwrap();
            let parent = BasisSet::load(parent_name).unwrap();
            assert!(ma.spherical);
            assert_eq!(ma.version, parent.version);

            assert_eq!(ma.shells_for(1).unwrap(), parent.shells_for(1).unwrap());

            for z in 2..=36u32 {
                let p = parent.shells_for(z).unwrap();
                let m = ma.shells_for(z).unwrap();
                assert_eq!(m.len(), p.len() + 2, "{ma_name} Z={z} adds s+p");
                assert_eq!(&m[..p.len()], p, "{ma_name} Z={z} parent unchanged");
                for (extra, l) in m[p.len()..].iter().zip([0u32, 1]) {
                    assert_eq!(extra.l, l);
                    assert_eq!(extra.coefficients, vec![1.0]);
                    let min_parent = p
                        .iter()
                        .filter(|s| s.l == l)
                        .flat_map(|s| s.exponents.iter().copied())
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(extra.exponents, vec![min_parent / 3.0]);
                }
            }
        }
        let ao = BasisSet::load("ma-def2-svp")
            .unwrap()
            .build(&water())
            .unwrap();
        assert_eq!(ao.n_ao(), 28);
    }

    #[test]
    fn pople_631gd_structure() {
        let set = BasisSet::load("6-31g(d)").unwrap();
        assert!(
            set.spherical,
            "6-31g(d) must be spherical (5d, not 6d Cartesian)"
        );
        assert!(!set.version.is_empty());

        // Carbon: 6-31G split valence (3 s, 2 p) plus one d polarization shell.
        let c = set.shells_for(6).unwrap();
        let count = |l: u32| c.iter().filter(|s| s.l == l).count();
        assert_eq!(count(0), 3, "carbon s shells");
        assert_eq!(count(1), 2, "carbon p shells");
        assert_eq!(count(2), 1, "carbon d shells");
        // Hydrogen carries no polarization in 6-31G(d): just the split 1s.
        let h = set.shells_for(1).unwrap();
        assert_eq!(
            h.iter().filter(|s| s.l == 0).count(),
            2,
            "hydrogen s shells"
        );
        assert_eq!(
            h.iter().filter(|s| s.l > 0).count(),
            0,
            "hydrogen has no polarization"
        );

        // AO counts with spherical d (5 per d shell): C atom = 3·1 + 2·3 + 1·5 = 14;
        // water (O + 2H) = 14 + 2·2 = 18. The `*` alias resolves identically.
        let c_atom = Molecule::from_xyz("1\nC\nC 0 0 0\n").unwrap();
        assert_eq!(
            set.build(&c_atom).unwrap().n_ao(),
            14,
            "carbon-atom AO count"
        );
        assert_eq!(set.build(&water()).unwrap().n_ao(), 18, "water AO count");
        assert_eq!(
            BasisSet::load("6-31g*")
                .unwrap()
                .build(&water())
                .unwrap()
                .n_ao(),
            18,
            "6-31g* alias AO count"
        );
    }

    #[test]
    fn pople_sp_shells_split() {
        let set = BasisSet::load("6-31g").unwrap();
        let shells = set.shells_for(8).unwrap();
        assert_eq!(shells.iter().filter(|s| s.l == 0).count(), 3);
        assert_eq!(shells.iter().filter(|s| s.l == 1).count(), 2);
        let any_p = shells.iter().find(|s| s.l == 1).unwrap();
        assert!(
            shells
                .iter()
                .any(|s| s.l == 0 && s.exponents == any_p.exponents),
            "expected an s contraction sharing the p exponents"
        );
    }

    #[test]
    fn overlap_diagonal_is_unit() {
        let mol = water();
        for name in BUNDLED_SETS {
            let ao = BasisSet::load(name).unwrap().build(&mol).unwrap();
            let n = ao.n_ao();
            let s = ao.integral().overlap();
            for i in 0..n {
                let sii = s[i * n + i];
                assert!(
                    (sii - 1.0).abs() < 1e-9,
                    "{name}: S[{i},{i}] = {sii}, expected 1 (renormalized contraction)"
                );
            }
            for i in 0..n {
                for j in 0..n {
                    assert!((s[i * n + j] - s[j * n + i]).abs() < 1e-12);
                }
            }
        }
    }
}
