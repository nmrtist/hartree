use std::collections::HashMap;

use serde::Deserialize;

use crate::basis::error::{BasisError, Result};
use crate::basis::{BasisSet, ContractedShell};

#[derive(Deserialize)]
struct BseFile {
    name: String,
    version: String,
    elements: HashMap<String, BseElement>,
}

#[derive(Deserialize)]
struct BseElement {
    #[serde(default)]
    electron_shells: Vec<BseShell>,
}

#[derive(Deserialize)]
struct BseShell {
    angular_momentum: Vec<u32>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

pub(crate) fn parse(json: &str) -> Result<BasisSet> {
    let file: BseFile = serde_json::from_str(json)?;
    let spherical = default_spherical(&file.name);

    let mut elements = HashMap::with_capacity(file.elements.len());
    for (z_str, element) in file.elements {
        let z: u32 = z_str
            .parse()
            .map_err(|_| BasisError::Schema(format!("non-integer element key {z_str:?}")))?;
        let mut shells = Vec::new();
        for shell in &element.electron_shells {
            segment(shell, &mut shells)?;
        }
        elements.insert(z, shells);
    }

    Ok(BasisSet {
        name: file.name,
        version: file.version,
        spherical,
        elements,
        ecp: None,
    })
}

fn segment(shell: &BseShell, out: &mut Vec<ContractedShell>) -> Result<()> {
    let exponents = parse_floats(&shell.exponents)?;
    if exponents.is_empty() {
        return Err(BasisError::Schema("shell has no exponents".into()));
    }

    match shell.angular_momentum.as_slice() {
        [] => Err(BasisError::Schema(
            "shell has empty angular_momentum".into(),
        )),

        [l] => {
            for row in &shell.coefficients {
                out.push(make_shell(*l, &exponents, row)?);
            }
            Ok(())
        }

        ls => {
            if shell.coefficients.len() != ls.len() {
                return Err(BasisError::Schema(format!(
                    "fused shell has {} angular momenta but {} coefficient rows",
                    ls.len(),
                    shell.coefficients.len()
                )));
            }
            for (l, row) in ls.iter().zip(&shell.coefficients) {
                out.push(make_shell(*l, &exponents, row)?);
            }
            Ok(())
        }
    }
}

fn make_shell(l: u32, exponents: &[f64], row: &[String]) -> Result<ContractedShell> {
    let mut coefficients = parse_floats(row)?;
    if coefficients.len() != exponents.len() {
        return Err(BasisError::Schema(format!(
            "contraction has {} exponents but {} coefficients",
            exponents.len(),
            coefficients.len()
        )));
    }
    normalize_contraction(l, exponents, &mut coefficients);
    Ok(ContractedShell {
        l,
        exponents: exponents.to_vec(),
        coefficients,
    })
}

fn normalize_contraction(l: u32, exponents: &[f64], coefficients: &mut [f64]) {
    let power = l as f64 + 1.5;
    let mut norm_sq = 0.0;
    for (i, &a) in exponents.iter().enumerate() {
        for (j, &b) in exponents.iter().enumerate() {
            let primitive_overlap = (2.0 * (a * b).sqrt() / (a + b)).powf(power);
            norm_sq += coefficients[i] * coefficients[j] * primitive_overlap;
        }
    }
    if norm_sq > 0.0 {
        let inv_norm = norm_sq.sqrt().recip();
        for c in coefficients.iter_mut() {
            *c *= inv_norm;
        }
    }
}

fn parse_floats(values: &[String]) -> Result<Vec<f64>> {
    values
        .iter()
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| BasisError::Schema(format!("non-numeric value {s:?}")))
        })
        .collect()
}

fn default_spherical(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    let cartesian = n.starts_with("sto")
        || n.starts_with("3-21")
        || n.starts_with("4-31")
        || (n.starts_with("6-31") && !n.starts_with("6-311"));
    !cartesian
}
