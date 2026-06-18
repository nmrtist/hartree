use std::collections::HashMap;

use crate::core::Element;

use crate::basis::error::{BasisError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GthBasisShell {
    pub l: usize,
    pub exps: Vec<f64>,
    pub coeffs: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GthElementBasis {
    pub shells: Vec<GthBasisShell>,
}

#[derive(Debug, Clone)]
pub struct GthBasisSet {
    pub name: String,
    bases: HashMap<String, HashMap<u32, GthElementBasis>>,
}

impl GthBasisSet {
    pub fn load_pade() -> Result<Self> {
        parse_basis("GTH-PADE", include_str!("data/gth/gth-basis-pade.gbs"))
    }

    pub fn from_text(name: &str, text: &str) -> Result<Self> {
        parse_basis(name, text)
    }

    #[must_use]
    pub fn get(&self, name: &str, z: u32) -> Option<&GthElementBasis> {
        self.bases
            .get(&name.to_ascii_uppercase())
            .and_then(|m| m.get(&z))
    }

    pub fn shells(&self, name: &str, z: u32, center: [f64; 3]) -> Result<Vec<integral::Shell>> {
        let eb = self
            .get(name, z)
            .ok_or_else(|| BasisError::ElementNotInSet {
                z,
                set: format!("{} basis {name}", self.name),
            })?;
        eb.shells
            .iter()
            .map(|s| {
                integral::Shell::new_spherical(s.l, center, s.exps.clone(), s.coeffs.clone())
                    .map_err(|e| {
                        BasisError::Schema(format!(
                            "GTH basis {name} Z={z}: bad shell l={}: {e}",
                            s.l
                        ))
                    })
            })
            .collect()
    }

    #[must_use]
    pub fn basis_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.bases.keys().cloned().collect();
        names.sort_unstable();
        names
    }
}

struct Block {
    symbol: String,
    names: Vec<String>,
    data: Vec<Vec<f64>>,
}

pub(crate) fn parse_basis(name: &str, text: &str) -> Result<GthBasisSet> {
    let blocks = split_blocks(text)?;
    let mut bases: HashMap<String, HashMap<u32, GthElementBasis>> = HashMap::new();
    for b in blocks {
        let z = Element::from_symbol(&b.symbol)
            .map_err(|e| {
                BasisError::Schema(format!("GTH basis: unknown element {:?}: {e}", b.symbol))
            })?
            .z();
        if b.names.is_empty() {
            return Err(BasisError::Schema(format!(
                "GTH basis: element {} header has no basis name",
                b.symbol
            )));
        }
        let eb = parse_block(&b)?;
        for bn in &b.names {
            bases
                .entry(bn.to_ascii_uppercase())
                .or_default()
                .insert(z, eb.clone());
        }
    }
    Ok(GthBasisSet {
        name: name.to_string(),
        bases,
    })
}

fn split_blocks(text: &str) -> Result<Vec<Block>> {
    let mut blocks: Vec<Block> = Vec::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut toks = line.split_whitespace();
        let first = toks.next().unwrap();
        if first.chars().next().is_some_and(char::is_alphabetic) {
            blocks.push(Block {
                symbol: first.to_string(),
                names: toks.map(str::to_string).collect(),
                data: Vec::new(),
            });
        } else {
            let nums = line
                .split_whitespace()
                .map(|t| {
                    t.parse::<f64>().map_err(|_| {
                        BasisError::Schema(format!("GTH basis: non-numeric token {t:?}"))
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            let block = blocks.last_mut().ok_or_else(|| {
                BasisError::Schema("GTH basis: data line before any element header".into())
            })?;
            block.data.push(nums);
        }
    }
    Ok(blocks)
}

fn parse_block(b: &Block) -> Result<GthElementBasis> {
    let schema = |m: &str| BasisError::Schema(format!("GTH basis {} {:?}: {m}", b.symbol, b.names));
    let mut shells = Vec::new();
    let mut idx = 0;
    while idx < b.data.len() {
        let header = &b.data[idx];
        if header.len() < 2 {
            return Err(schema("shell header needs `l nexp`"));
        }
        let l = header[0] as usize;
        let nexp = header[1] as usize;
        idx += 1;
        if nexp == 0 {
            return Err(schema("shell with zero primitives"));
        }
        let mut exps = Vec::with_capacity(nexp);
        let mut coeffs = Vec::with_capacity(nexp);
        for _ in 0..nexp {
            let row = b
                .data
                .get(idx)
                .ok_or_else(|| schema("truncated shell primitives"))?;
            if row.len() < 2 {
                return Err(schema("primitive line needs `exp coeff`"));
            }
            exps.push(row[0]);
            coeffs.push(row[1]);
            idx += 1;
        }
        shells.push(GthBasisShell { l, exps, coeffs });
    }
    if shells.is_empty() {
        return Err(schema("no shells"));
    }
    Ok(GthElementBasis { shells })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_si_szv() {
        let set = GthBasisSet::load_pade().unwrap();
        let si = set.get("SZV-GTH", 14).expect("Si SZV-GTH present");
        assert_eq!(si.shells.len(), 2, "SZV Si = one s + one p");
        let s = &si.shells[0];
        assert_eq!(s.l, 0);
        assert_eq!(s.exps.len(), 4);
        assert!((s.exps[0] - 1.2032403600).abs() < 1e-12);
        assert!((s.coeffs[0] - 0.3290356759).abs() < 1e-12);
        let p = &si.shells[1];
        assert_eq!(p.l, 1);
        assert!((p.coeffs[3] - (-0.3623984652)).abs() < 1e-12);
        assert_eq!(set.get("szv-gth-q4", 14), Some(si));
    }

    #[test]
    fn parses_si_dzvp_matches_validated_shells() {
        let set = GthBasisSet::load_pade().unwrap();
        let si = set.get("DZVP-GTH-PADE", 14).unwrap();
        let ls: Vec<(usize, usize)> = si.shells.iter().map(|s| (s.l, s.exps.len())).collect();
        assert_eq!(ls, vec![(0, 4), (0, 1), (1, 4), (1, 1), (2, 1)]);
        assert!((si.shells[1].exps[0] - 0.0575619526).abs() < 1e-12);
        assert!((si.shells[1].coeffs[0] - 1.0).abs() < 1e-12);
        assert!((si.shells[4].exps[0] - 0.45).abs() < 1e-12);
        let center = [0.0, 0.0, 0.0];
        let shells = set.shells("DZVP-GTH-PADE", 14, center).unwrap();
        let nao: usize = shells.iter().map(integral::Shell::n_func).sum();
        assert_eq!(nao, 13);
    }

    #[test]
    fn bundled_szv_elements_present() {
        let set = GthBasisSet::load_pade().unwrap();
        for z in [1, 3, 6, 8, 9, 11, 12, 14, 17] {
            assert!(set.get("SZV-GTH", z).is_some(), "SZV-GTH missing Z={z}");
        }
    }

    #[test]
    fn bundled_dzvp_elements_present() {
        let set = GthBasisSet::load_pade().unwrap();
        for (name, z) in [
            ("DZVP-GTH", 3),
            ("DZVP-GTH", 9),
            ("DZVP-GTH", 11),
            ("DZVP-GTH", 12),
            ("DZVP-GTH", 17),
            ("DZVP-MOLOPT-PBE-GTH-q13", 31),
            ("DZVP-MOLOPT-PBE-GTH-q4", 32),
            ("DZVP-MOLOPT-PBE-GTH-q5", 33),
            ("DZVP-MOLOPT-PBE-GTH", 31),
            ("DZVP-MOLOPT-PBE-GTH", 32),
            ("DZVP-MOLOPT-PBE-GTH", 33),
        ] {
            assert!(set.get(name, z).is_some(), "{name} missing Z={z}");
        }
        let ga = set.get("DZVP-MOLOPT-GGA-GTH-q13", 31).unwrap();
        assert_eq!(ga.shells.last().unwrap().l, 3, "Ga carries an f shell");
    }

    #[test]
    fn missing_pair_errors() {
        let set = GthBasisSet::load_pade().unwrap();
        assert!(set.get("DZVP-GTH-PADE", 1).is_none(), "no H DZVP bundled");
        assert!(set.shells("DZVP-GTH-PADE", 1, [0.0; 3]).is_err());
    }
}
