use std::collections::HashMap;

use crate::core::Element;

use crate::basis::error::{BasisError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GthLocal {
    pub r_loc: f64,
    pub c: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GthNonlocal {
    pub l: usize,
    pub r: f64,
    pub h: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GthPotential {
    pub n_elec: Vec<u32>,
    pub z_ion: f64,
    pub local: GthLocal,
    pub nonlocal: Vec<GthNonlocal>,
}

impl GthPotential {
    #[must_use]
    pub fn n_core(&self, z: u32) -> u32 {
        z - self.z_ion as u32
    }
}

#[derive(Debug, Clone)]
pub struct GthSet {
    pub name: String,
    elements: HashMap<u32, GthPotential>,
}

impl GthSet {
    pub fn load_pade() -> Result<Self> {
        parse("GTH-PADE", include_str!("data/gth/gth-pade.gth"))
    }

    pub fn from_text(name: &str, text: &str) -> Result<Self> {
        parse(name, text)
    }

    #[must_use]
    pub fn get(&self, z: u32) -> Option<&GthPotential> {
        self.elements.get(&z)
    }

    #[must_use]
    pub fn elements(&self) -> Vec<u32> {
        let mut zs: Vec<u32> = self.elements.keys().copied().collect();
        zs.sort_unstable();
        zs
    }
}

struct Block {
    symbol: String,
    data: Vec<Vec<f64>>,
}

pub(crate) fn parse(name: &str, text: &str) -> Result<GthSet> {
    let blocks = split_blocks(text)?;
    let mut elements = HashMap::new();
    for b in blocks {
        let z = Element::from_symbol(&b.symbol)
            .map_err(|e| BasisError::Schema(format!("GTH: unknown element {:?}: {e}", b.symbol)))?
            .z();
        elements.insert(z, parse_block(&b)?);
    }
    Ok(GthSet {
        name: name.to_string(),
        elements,
    })
}

fn split_blocks(text: &str) -> Result<Vec<Block>> {
    let mut blocks: Vec<Block> = Vec::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let first = line.split_whitespace().next().unwrap();
        if first.chars().next().is_some_and(char::is_alphabetic) {
            blocks.push(Block {
                symbol: first.to_string(),
                data: Vec::new(),
            });
        } else {
            let nums = line
                .split_whitespace()
                .map(|t| {
                    t.parse::<f64>()
                        .map_err(|_| BasisError::Schema(format!("GTH: non-numeric token {t:?}")))
                })
                .collect::<Result<Vec<f64>>>()?;
            let block = blocks.last_mut().ok_or_else(|| {
                BasisError::Schema("GTH: data line before any element header".into())
            })?;
            block.data.push(nums);
        }
    }
    Ok(blocks)
}

fn parse_block(b: &Block) -> Result<GthPotential> {
    let schema = |m: &str| BasisError::Schema(format!("GTH {}: {m}", b.symbol));
    let d = &b.data;
    if d.len() < 3 {
        return Err(schema("block too short (need n_elec, local, nprj lines)"));
    }
    let n_elec: Vec<u32> = d[0].iter().map(|&x| x as u32).collect();
    let z_ion: f64 = n_elec.iter().map(|&x| f64::from(x)).sum();

    let loc = &d[1];
    if loc.len() < 2 {
        return Err(schema("local line needs r_loc and nexp"));
    }
    let r_loc = loc[0];
    let nexp = loc[1] as usize;
    if loc.len() < 2 + nexp {
        return Err(schema("local line missing C coefficients"));
    }
    let c = loc[2..2 + nexp].to_vec();
    let local = GthLocal { r_loc, c };

    let nprj = d[2][0] as usize;
    let mut nonlocal = Vec::with_capacity(nprj);
    let mut idx = 3;
    for l in 0..nprj {
        let line0 = d
            .get(idx)
            .ok_or_else(|| schema("missing projector channel"))?;
        idx += 1;
        let r_l = line0[0];
        let nproj = line0[1] as usize;
        let mut h = vec![vec![0.0; nproj]; nproj];
        if line0.len() < 2 + nproj {
            return Err(schema("projector first row truncated"));
        }
        for j in 0..nproj {
            let val = line0[2 + j];
            h[0][j] = val;
            h[j][0] = val;
        }
        #[allow(clippy::needless_range_loop)]
        for i in 1..nproj {
            let cont = d
                .get(idx)
                .ok_or_else(|| schema("missing projector continuation row"))?;
            idx += 1;
            if cont.len() < nproj - i {
                return Err(schema("projector continuation row truncated"));
            }
            for (k, &val) in cont.iter().take(nproj - i).enumerate() {
                let j = i + k;
                h[i][j] = val;
                h[j][i] = val;
            }
        }
        nonlocal.push(GthNonlocal { l, r: r_l, h });
    }

    Ok(GthPotential {
        n_elec,
        z_ion,
        local,
        nonlocal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_silicon() {
        let set = GthSet::load_pade().unwrap();
        let si = set.get(14).expect("Si present");
        assert_eq!(si.n_elec, vec![2, 2]);
        assert!((si.z_ion - 4.0).abs() < 1e-12);
        assert_eq!(si.n_core(14), 10);
        assert!((si.local.r_loc - 0.44).abs() < 1e-12);
        assert_eq!(si.local.c, vec![-7.336_102_97]);
        assert_eq!(si.nonlocal.len(), 2);
        let s = &si.nonlocal[0];
        assert_eq!(s.l, 0);
        assert!((s.r - 0.422_738_13).abs() < 1e-12);
        assert!((s.h[0][0] - 5.906_928_31).abs() < 1e-9);
        assert!((s.h[0][1] - (-1.261_893_97)).abs() < 1e-9);
        assert_eq!(s.h[1][0], s.h[0][1], "h symmetric");
        assert!((s.h[1][1] - 3.258_196_22).abs() < 1e-9);
        let p = &si.nonlocal[1];
        assert_eq!(p.l, 1);
        assert!((p.h[0][0] - 2.727_013_46).abs() < 1e-9);
    }

    #[test]
    fn parses_hydrogen_no_projectors() {
        let set = GthSet::load_pade().unwrap();
        let h = set.get(1).unwrap();
        assert!((h.z_ion - 1.0).abs() < 1e-12);
        assert_eq!(h.local.c, vec![-4.180_236_80, 0.725_074_82]);
        assert!(
            h.nonlocal.is_empty(),
            "H GTH-PADE has no nonlocal projectors"
        );
    }

    #[test]
    fn all_bundled_elements_present() {
        let set = GthSet::load_pade().unwrap();
        assert_eq!(
            set.elements(),
            vec![1, 3, 6, 8, 9, 11, 12, 14, 17, 31, 32, 33]
        );
    }
}
