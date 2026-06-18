use std::collections::HashMap;

use serde::Deserialize;

use crate::basis::error::{BasisError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct EcpPrimitive {
    pub n: i32,
    pub zeta: f64,
    pub coef: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementEcp {
    pub n_core: u32,
    pub max_l: usize,
    pub local: Vec<EcpPrimitive>,
    pub semilocal: Vec<Vec<EcpPrimitive>>,
}

#[derive(Debug, Clone)]
pub struct EcpSet {
    pub name: String,
    pub version: String,
    elements: HashMap<u32, ElementEcp>,
}

impl EcpSet {
    pub fn get(&self, z: u32) -> Option<&ElementEcp> {
        self.elements.get(&z)
    }

    pub fn elements(&self) -> Vec<u32> {
        let mut zs: Vec<u32> = self.elements.keys().copied().collect();
        zs.sort_unstable();
        zs
    }
}

#[derive(Deserialize)]
struct BseEcpFile {
    name: String,
    version: String,
    elements: HashMap<String, BseEcpElement>,
}

#[derive(Deserialize)]
struct BseEcpElement {
    #[serde(default)]
    ecp_electrons: u32,
    #[serde(default)]
    ecp_potentials: Vec<BseEcpPotential>,
}

#[derive(Deserialize)]
struct BseEcpPotential {
    ecp_type: String,
    angular_momentum: Vec<u32>,
    r_exponents: Vec<i32>,
    gaussian_exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

pub(crate) fn parse(json: &str) -> Result<EcpSet> {
    let file: BseEcpFile = serde_json::from_str(json)?;
    let mut elements = HashMap::new();
    for (z_str, element) in file.elements {
        let z: u32 = z_str
            .parse()
            .map_err(|_| BasisError::Schema(format!("non-integer element key {z_str:?}")))?;
        if element.ecp_potentials.is_empty() {
            continue;
        }
        elements.insert(z, parse_element(z, &element)?);
    }
    Ok(EcpSet {
        name: file.name,
        version: file.version,
        elements,
    })
}

fn parse_element(z: u32, element: &BseEcpElement) -> Result<ElementEcp> {
    if element.ecp_electrons == 0 {
        return Err(BasisError::Schema(format!(
            "Z={z}: ecp_potentials present but ecp_electrons is 0"
        )));
    }
    let mut channels: HashMap<usize, Vec<EcpPrimitive>> = HashMap::new();
    for pot in &element.ecp_potentials {
        if pot.ecp_type != "scalar_ecp" {
            return Err(BasisError::Schema(format!(
                "Z={z}: unsupported ECP block type {:?} (only scalar_ecp is \
                 supported; spin-orbit ECPs are out of scope)",
                pot.ecp_type
            )));
        }
        let [l] = pot.angular_momentum.as_slice() else {
            return Err(BasisError::Schema(format!(
                "Z={z}: ECP channel must carry exactly one angular momentum, \
                 got {:?}",
                pot.angular_momentum
            )));
        };
        let [coeff_row] = pot.coefficients.as_slice() else {
            return Err(BasisError::Schema(format!(
                "Z={z}: scalar ECP channel l={l} must have exactly one \
                 coefficient row, got {}",
                pot.coefficients.len()
            )));
        };
        if pot.r_exponents.len() != pot.gaussian_exponents.len()
            || coeff_row.len() != pot.gaussian_exponents.len()
        {
            return Err(BasisError::Schema(format!(
                "Z={z}: ECP channel l={l} has mismatched primitive counts \
                 ({} r-exponents, {} gaussian exponents, {} coefficients)",
                pot.r_exponents.len(),
                pot.gaussian_exponents.len(),
                coeff_row.len()
            )));
        }
        let mut prims = Vec::with_capacity(pot.r_exponents.len());
        for ((&n, zeta), coef) in pot
            .r_exponents
            .iter()
            .zip(&pot.gaussian_exponents)
            .zip(coeff_row)
        {
            prims.push(EcpPrimitive {
                n,
                zeta: parse_float(zeta)?,
                coef: parse_float(coef)?,
            });
        }
        if channels.insert(*l as usize, prims).is_some() {
            return Err(BasisError::Schema(format!(
                "Z={z}: duplicate ECP channel l={l}"
            )));
        }
    }
    let max_l = *channels
        .keys()
        .max()
        .ok_or_else(|| BasisError::Schema(format!("Z={z}: ECP has no channels")))?;
    if max_l > 4 {
        return Err(BasisError::Schema(format!(
            "Z={z}: ECP local channel l={max_l} exceeds the supported \
             projector range (l <= 4)"
        )));
    }
    let local = channels.remove(&max_l).expect("max_l key exists");
    let mut semilocal = Vec::with_capacity(max_l);
    for l in 0..max_l {
        semilocal.push(channels.remove(&l).ok_or_else(|| {
            BasisError::Schema(format!(
                "Z={z}: ECP is missing projector channel l={l} (channels must \
                 cover l = 0..{max_l})"
            ))
        })?);
    }
    Ok(ElementEcp {
        n_core: element.ecp_electrons,
        max_l,
        local,
        semilocal,
    })
}

fn parse_float(s: &str) -> Result<f64> {
    s.trim()
        .parse::<f64>()
        .map_err(|_| BasisError::Schema(format!("non-numeric ECP value {s:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def2_ecp() -> EcpSet {
        parse(include_str!("data/ecp/def2-ecp.json")).unwrap()
    }

    #[test]
    fn def2_ecp_shapes() {
        let set = def2_ecp();
        assert_eq!(set.name, "def2-ECP");
        assert_eq!(set.elements(), vec![47, 50, 53, 79]);
        for (z, n_core, counts) in [
            (47, 28, [2, 4, 4, 4]),  // Ag: ECP28MWB
            (50, 28, [2, 4, 6, 6]),  // Sn: ECP28MDF
            (53, 28, [4, 7, 8, 10]), // I:  ECP28MDF
            (79, 60, [2, 4, 4, 4]),  // Au: ECP60MDF
        ] {
            let e = set.get(z).unwrap();
            assert_eq!(e.n_core, n_core, "Z={z} n_core");
            assert_eq!(e.max_l, 3, "Z={z} local channel is f");
            assert_eq!(e.semilocal.len(), 3, "Z={z} projector channels");
            assert_eq!(e.local.len(), counts[0], "Z={z} local primitives");
            for l in 0..3 {
                assert_eq!(
                    e.semilocal[l].len(),
                    counts[l + 1],
                    "Z={z} channel l={l} primitives"
                );
            }
            for p in e.local.iter().chain(e.semilocal.iter().flatten()) {
                assert_eq!(p.n, 2, "Z={z}: def2-ECP primitives are r^0 (n = 2)");
                assert!(p.zeta > 0.0);
            }
        }
        assert!(set.get(36).is_none(), "no ECP below Rb");
    }

    #[test]
    fn silver_literal_values() {
        let set = def2_ecp();
        let ag = set.get(47).unwrap();
        assert_eq!(ag.n_core, 28);
        assert_eq!(
            ag.local,
            vec![
                EcpPrimitive {
                    n: 2,
                    zeta: 14.22,
                    coef: -33.68992012
                },
                EcpPrimitive {
                    n: 2,
                    zeta: 7.11,
                    coef: -5.53112021
                },
            ]
        );
        assert_eq!(
            ag.semilocal[0],
            vec![
                EcpPrimitive {
                    n: 2,
                    zeta: 13.13,
                    coef: 255.13936452
                },
                EcpPrimitive {
                    n: 2,
                    zeta: 6.51,
                    coef: 36.86612154
                },
                EcpPrimitive {
                    n: 2,
                    zeta: 14.22,
                    coef: 33.68992012
                },
                EcpPrimitive {
                    n: 2,
                    zeta: 7.11,
                    coef: 5.53112021
                },
            ]
        );
        assert_eq!(ag.semilocal[2][0].zeta, 10.21);
        assert_eq!(ag.semilocal[2][0].coef, 73.71926087);
        let au = set.get(79).unwrap();
        assert_eq!(au.local[0].coef, 30.49008890);
        assert_eq!(au.semilocal[0][2].coef, -30.49008890);
    }
}
