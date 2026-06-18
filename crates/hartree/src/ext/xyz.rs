use crate::core::Molecule;
use crate::core::units::BOHR_TO_ANGSTROM;

use crate::ext::ExtError;

pub fn write_xyz(molecule: &Molecule, comment: &str) -> Result<String, ExtError> {
    if molecule.has_ghosts() {
        return Err(ExtError::ConfGen(
            "ghost atoms cannot be written to an external-program XYZ input".into(),
        ));
    }
    let mut out = String::new();
    out.push_str(&format!("{}\n{}\n", molecule.len(), comment));
    for atom in &molecule.atoms {
        out.push_str(&format!(
            "{:<2} {:>18.10} {:>18.10} {:>18.10}\n",
            atom.element.symbol(),
            atom.position[0] * BOHR_TO_ANGSTROM,
            atom.position[1] * BOHR_TO_ANGSTROM,
            atom.position[2] * BOHR_TO_ANGSTROM,
        ));
    }
    Ok(out)
}

pub fn parse_multi_xyz(input: &str) -> Result<Vec<(Molecule, String)>, ExtError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut frames = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        let count: usize = lines[i].trim().parse().map_err(|_| ExtError::Parse {
            what: "multi-frame XYZ",
            message: format!("frame header is not an atom count: {:?}", lines[i]),
        })?;
        if i + 2 + count > lines.len() {
            return Err(ExtError::Parse {
                what: "multi-frame XYZ",
                message: format!(
                    "truncated frame: expected {count} atoms after line {}, file ends early",
                    i + 1
                ),
            });
        }
        let comment = lines.get(i + 1).copied().unwrap_or("").to_string();
        let frame_text = lines[i..i + 2 + count].join("\n");
        let molecule = Molecule::from_xyz(&frame_text).map_err(|e| ExtError::Parse {
            what: "multi-frame XYZ",
            message: e.to_string(),
        })?;
        frames.push((molecule, comment));
        i += 2 + count;
    }
    if frames.is_empty() {
        return Err(ExtError::Parse {
            what: "multi-frame XYZ",
            message: "no frames found".into(),
        });
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xyz_round_trip() {
        let mol = Molecule::from_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let text = write_xyz(&mol, "h2").unwrap();
        let back = Molecule::from_xyz(&text).unwrap();
        assert_eq!(back.len(), 2);
        assert!((back.atoms[1].position[2] - mol.atoms[1].position[2]).abs() < 1e-9);
    }

    #[test]
    fn ghost_rejected() {
        let mol = Molecule::from_xyz("2\ng\nH 0 0 0\nGh(H) 0 0 1\n").unwrap();
        assert!(write_xyz(&mol, "x").is_err());
    }

    #[test]
    fn multi_xyz_two_frames() {
        let text = "2\n -1.0\nH 0 0 0\nH 0 0 0.74\n2\n -0.9\nH 0 0 0\nH 0 0 0.80\n";
        let frames = parse_multi_xyz(text).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].1.trim(), "-0.9");
    }

    #[test]
    fn multi_xyz_truncated_is_error() {
        assert!(parse_multi_xyz("2\nc\nH 0 0 0\n").is_err());
        assert!(parse_multi_xyz("").is_err());
    }
}
