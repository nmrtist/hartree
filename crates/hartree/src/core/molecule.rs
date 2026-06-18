use crate::core::element::Element;
use crate::core::error::{HartreeError, Result};
use crate::core::units::ANGSTROM_TO_BOHR;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    pub element: Element,
    pub position: [f64; 3],
    pub ghost: bool,
}

impl Atom {
    pub fn new(element: Element, position: [f64; 3]) -> Self {
        Self {
            element,
            position,
            ghost: false,
        }
    }

    pub fn new_ghost(element: Element, position: [f64; 3]) -> Self {
        Self {
            element,
            position,
            ghost: true,
        }
    }

    #[inline]
    pub fn z_eff(&self) -> u32 {
        if self.ghost { 0 } else { self.element.z() }
    }
}

#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: u32,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>, charge: i32, multiplicity: u32) -> Self {
        Self {
            atoms,
            charge,
            multiplicity,
        }
    }

    pub fn with_charge(mut self, charge: i32) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_multiplicity(mut self, multiplicity: u32) -> Self {
        self.multiplicity = multiplicity;
        self
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn total_nuclear_charge(&self) -> i64 {
        self.atoms.iter().map(|a| a.z_eff() as i64).sum()
    }

    pub fn has_ghosts(&self) -> bool {
        self.atoms.iter().any(|a| a.ghost)
    }

    pub fn n_real_atoms(&self) -> usize {
        self.atoms.iter().filter(|a| !a.ghost).count()
    }

    pub fn n_electrons(&self) -> i64 {
        self.total_nuclear_charge() - self.charge as i64
    }

    pub fn nuclear_repulsion(&self) -> f64 {
        let mut energy = 0.0;
        for (i, a) in self.atoms.iter().enumerate() {
            for b in &self.atoms[i + 1..] {
                let zi = a.z_eff() as f64;
                let zj = b.z_eff() as f64;
                if zi == 0.0 || zj == 0.0 {
                    continue; // ghost atoms carry no nuclear charge
                }
                energy += zi * zj / distance(a.position, b.position);
            }
        }
        energy
    }

    pub fn nuclear_repulsion_with(&self, charges: &[f64]) -> f64 {
        assert_eq!(
            charges.len(),
            self.len(),
            "nuclear_repulsion_with needs one charge per atom"
        );
        let mut energy = 0.0;
        for (i, a) in self.atoms.iter().enumerate() {
            for (j, b) in self.atoms.iter().enumerate().skip(i + 1) {
                energy += charges[i] * charges[j] / distance(a.position, b.position);
            }
        }
        energy
    }

    pub fn validate(&self) -> Result<()> {
        if !self.atoms.is_empty() && self.n_real_atoms() == 0 {
            return Err(HartreeError::Other(
                "ghost-only molecule: every atom is a ghost (basis functions only, no nuclei, \
                 no electrons); at least one real atom is required"
                    .into(),
            ));
        }
        let n = self.n_electrons();
        let two_s = self.multiplicity as i64 - 1;
        if two_s < 0 || two_s > n || (n - two_s) % 2 != 0 {
            return Err(HartreeError::InconsistentSpin {
                n_electrons: n,
                multiplicity: self.multiplicity,
            });
        }
        Ok(())
    }

    pub fn from_xyz(input: &str) -> Result<Self> {
        Ok(Self::from_xyz_with_lattice(input)?.0)
    }

    pub fn from_xyz_with_lattice(input: &str) -> Result<(Self, Option<[[f64; 3]; 3]>)> {
        let mut lines = input.lines();

        let count_line = lines
            .next()
            .ok_or_else(|| HartreeError::MalformedXyz("input is empty".into()))?;
        let count: usize = count_line.trim().parse().map_err(|_| {
            HartreeError::MalformedXyz(format!("first line is not an atom count: {count_line:?}"))
        })?;

        let comment = lines.next().unwrap_or("");
        let lattice = parse_lattice(comment)?;

        let mut atoms = Vec::with_capacity(count);
        for i in 0..count {
            let line = lines.next().ok_or_else(|| {
                HartreeError::MalformedXyz(format!("expected {count} atoms, found {i}"))
            })?;
            atoms.push(parse_atom_line(line)?);
        }

        Ok((Molecule::new(atoms, 0, 1), lattice))
    }
}

fn parse_lattice(comment: &str) -> Result<Option<[[f64; 3]; 3]>> {
    let Some(key_pos) = comment.to_ascii_lowercase().find("lattice=") else {
        return Ok(None);
    };
    let after = comment[key_pos + "lattice=".len()..].trim_start();
    let stripped = after.strip_prefix('"').ok_or_else(|| {
        HartreeError::MalformedXyz("extended-XYZ Lattice= value must be double-quoted".into())
    })?;
    let end = stripped.find('"').ok_or_else(|| {
        HartreeError::MalformedXyz("extended-XYZ Lattice= value missing its closing quote".into())
    })?;
    let vals: Vec<f64> = stripped[..end]
        .split_whitespace()
        .map(|t| {
            t.parse::<f64>().map_err(|_| {
                HartreeError::MalformedXyz(format!("non-numeric Lattice value: {t:?}"))
            })
        })
        .collect::<Result<_>>()?;
    if vals.len() != 9 {
        return Err(HartreeError::MalformedXyz(format!(
            "extended-XYZ Lattice= needs 9 numbers (three row vectors), got {}",
            vals.len()
        )));
    }
    let mut m = [[0.0_f64; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            *x = vals[r * 3 + c] * ANGSTROM_TO_BOHR;
        }
    }
    Ok(Some(m))
}

fn parse_atom_line(line: &str) -> Result<Atom> {
    let mut fields = line.split_whitespace();
    let symbol = fields
        .next()
        .ok_or_else(|| HartreeError::MalformedXyz(format!("empty atom line: {line:?}")))?;

    let (symbol, ghost) = parse_ghost_symbol(symbol)?;

    let element = match symbol.parse::<u32>() {
        Ok(z) => Element::from_z(z)?,
        Err(_) => Element::from_symbol(symbol)?,
    };

    let mut coord = [0.0_f64; 3];
    for (axis, slot) in coord.iter_mut().enumerate() {
        let token = fields.next().ok_or_else(|| {
            HartreeError::MalformedXyz(format!("missing coordinate {axis} in line: {line:?}"))
        })?;
        let value: f64 = token.parse().map_err(|_| {
            HartreeError::MalformedXyz(format!("non-numeric coordinate: {token:?}"))
        })?;
        *slot = value * ANGSTROM_TO_BOHR;
    }

    Ok(Atom {
        element,
        position: coord,
        ghost,
    })
}

fn parse_ghost_symbol(token: &str) -> Result<(&str, bool)> {
    if let Some(rest) = token.strip_prefix('@') {
        if rest.is_empty() {
            return Err(HartreeError::MalformedXyz(format!(
                "ghost marker {token:?} has no element symbol (expected e.g. @O)"
            )));
        }
        return Ok((rest, true));
    }
    let lower = token.to_ascii_lowercase();
    if lower.starts_with("gh(") {
        let inner = &token[3..];
        let inner = inner.strip_suffix(')').ok_or_else(|| {
            HartreeError::MalformedXyz(format!(
                "ghost marker {token:?} is missing the closing parenthesis (expected e.g. Gh(O))"
            ))
        })?;
        if inner.is_empty() {
            return Err(HartreeError::MalformedXyz(format!(
                "ghost marker {token:?} has no element symbol (expected e.g. Gh(O))"
            )));
        }
        return Ok((inner, true));
    }
    Ok((token, false))
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const WATER_XYZ: &str = "3
water, experimental geometry
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
";

    #[test]
    fn parses_extended_xyz_lattice() {
        let xyz = "1
Lattice=\"5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0\" Properties=species:S:1:pos:R:3
Si 0.0 0.0 0.0
";
        let (mol, lattice) = Molecule::from_xyz_with_lattice(xyz).unwrap();
        assert_eq!(mol.len(), 1);
        let l = lattice.expect("lattice present");
        assert!((l[0][0] - 5.0 * ANGSTROM_TO_BOHR).abs() < 1e-10);
        assert!((l[1][1] - 5.0 * ANGSTROM_TO_BOHR).abs() < 1e-10);
        assert!((l[2][2] - 5.0 * ANGSTROM_TO_BOHR).abs() < 1e-10);
        assert!(l[0][1].abs() < 1e-12 && l[2][0].abs() < 1e-12);
        assert!(
            Molecule::from_xyz_with_lattice(WATER_XYZ)
                .unwrap()
                .1
                .is_none()
        );
    }

    #[test]
    fn parses_water() {
        let mol = Molecule::from_xyz(WATER_XYZ).unwrap();
        assert_eq!(mol.len(), 3);
        assert_eq!(mol.atoms[0].element.symbol(), "O");
        assert_eq!(mol.n_electrons(), 10);
        mol.validate().unwrap();
    }

    #[test]
    fn h2_nuclear_repulsion() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::from_xyz(xyz).unwrap();
        let expected = 1.0 / (0.74 * ANGSTROM_TO_BOHR);
        assert!((mol.nuclear_repulsion() - expected).abs() < 1e-10);
    }

    #[test]
    fn accepts_atomic_number_column() {
        let mol = Molecule::from_xyz("1\nlone oxygen\n8 0 0 0\n").unwrap();
        assert_eq!(mol.atoms[0].element.z(), 8);
    }

    #[test]
    fn ghost_atoms_parse_and_count() {
        let xyz = "4\nwater + ghost O\nO 0 0 0.1178\nH 0 0.7555 -0.4712\nH 0 -0.7555 -0.4712\nGh(O) 0 0 3.0\n";
        let mol = Molecule::from_xyz(xyz).unwrap();
        assert!(mol.atoms[3].ghost && mol.has_ghosts());
        assert_eq!(mol.atoms[3].element.symbol(), "O");
        assert_eq!(mol.atoms[3].z_eff(), 0);
        assert_eq!(mol.n_real_atoms(), 3);
        assert_eq!(mol.total_nuclear_charge(), 10);
        assert_eq!(mol.n_electrons(), 10);
        mol.validate().unwrap();

        let short = Molecule::from_xyz("2\nshorthand\n@He 0 0 0\ngh(ne) 0 0 1\nH 0 0 2\n").unwrap();
        assert!(short.atoms[0].ghost && short.atoms[1].ghost);
        assert_eq!(short.atoms[1].element.symbol(), "Ne");
    }

    #[test]
    fn ghost_nuclear_repulsion_is_zero_with_ghost_partner() {
        let xyz = "2\nH + Gh(H)\nH 0 0 0\nGh(H) 0 0 0.74\n";
        let mol = Molecule::from_xyz(xyz).unwrap();
        assert_eq!(mol.nuclear_repulsion(), 0.0);
        assert_eq!(mol.n_electrons(), 1);
    }

    #[test]
    fn ghost_only_molecule_rejected() {
        let mol = Molecule::from_xyz("1\nghost only\nGh(O) 0 0 0\n").unwrap();
        assert!(mol.validate().is_err());
        assert!(Molecule::from_xyz("1\nbad\nGh(O 0 0 0\n").is_err());
        assert!(Molecule::from_xyz("1\nbad\n@ 0 0 0\n").is_err());
        assert!(Molecule::from_xyz("1\nbad\nGh() 0 0 0\n").is_err());
    }

    #[test]
    fn spin_validation() {
        let mol = Molecule::from_xyz(WATER_XYZ).unwrap();
        assert!(mol.clone().with_multiplicity(1).validate().is_ok());
        assert!(mol.clone().with_multiplicity(2).validate().is_err());

        let cation = mol.with_charge(1);
        assert_eq!(cation.n_electrons(), 9);
        assert!(cation.clone().with_multiplicity(2).validate().is_ok());
        assert!(cation.with_multiplicity(1).validate().is_err());
    }
}
