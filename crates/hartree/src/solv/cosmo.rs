use std::fmt::Write as _;

use crate::solv::surface::BOHR;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosmoSegment {
    pub atom: usize,
    pub position: [f64; 3],
    pub charge: f64,
    pub area: f64,
    pub potential: f64,
}

#[derive(Debug, Clone)]
pub struct CosmoAtom {
    pub symbol: String,
    pub position: [f64; 3],
    pub radius: f64,
}

#[derive(Debug, Clone)]
pub struct CosmoExport {
    pub epsilon: f64,
    pub total_energy: f64,
    pub dielectric_energy: f64,
    pub atoms: Vec<CosmoAtom>,
    pub segments: Vec<CosmoSegment>,
}

impl CosmoExport {
    pub fn fepsi(&self) -> f64 {
        if self.epsilon.is_infinite() {
            0.5
        } else {
            0.5 * (1.0 - 1.0 / self.epsilon)
        }
    }

    pub fn total_area(&self) -> f64 {
        self.segments.iter().map(|s| s.area).sum()
    }

    pub fn total_charge(&self) -> f64 {
        self.segments.iter().map(|s| s.charge).sum()
    }
}

pub fn write_cosmo(export: &CosmoExport) -> String {
    let mut s = String::new();
    let eps = export.epsilon;
    let eps_str = if eps.is_infinite() {
        "infinity".to_string()
    } else {
        format!("{eps}")
    };

    writeln!(s, "$info").unwrap();
    writeln!(s, "prog.: hartree").unwrap();
    writeln!(s, "$cosmo").unwrap();
    writeln!(s, "  epsilon={eps_str}").unwrap();
    writeln!(s, "$cosmo_data").unwrap();
    writeln!(s, "  fepsi={:.10}", export.fepsi()).unwrap();
    writeln!(s, "  area={:.6}", export.total_area()).unwrap();
    writeln!(
        s,
        "$coord_rad\n#atom   x                  y                  z             element  radius [A]"
    )
    .unwrap();
    for (i, a) in export.atoms.iter().enumerate() {
        writeln!(
            s,
            "{:4} {:19.14} {:19.14} {:19.14}  {:<4} {:9.5}",
            i + 1,
            a.position[0],
            a.position[1],
            a.position[2],
            a.symbol,
            a.radius
        )
        .unwrap();
    }
    writeln!(s, "$screening_charge").unwrap();
    writeln!(s, "  cosmo      = {:.6}", export.total_charge()).unwrap();
    writeln!(s, "  correction = {:.6}", 0.0).unwrap();
    writeln!(s, "  total      = {:.6}", export.total_charge()).unwrap();
    writeln!(s, "$cosmo_energy").unwrap();
    writeln!(
        s,
        "  Total energy [a.u.]            = {:21.10}",
        export.total_energy
    )
    .unwrap();
    writeln!(
        s,
        "  Dielectric energy [a.u.]       = {:21.10}",
        export.dielectric_energy
    )
    .unwrap();
    writeln!(s, "$segment_information").unwrap();
    writeln!(s, "# n             - segment number").unwrap();
    writeln!(s, "# atom          - atom associated with segment n").unwrap();
    writeln!(s, "# position      - segment coordinates [a.u.]").unwrap();
    writeln!(s, "# charge        - segment charge (corrected)").unwrap();
    writeln!(s, "# area          - segment area [A**2]").unwrap();
    writeln!(s, "# potential     - solute potential on segment [a.u.]").unwrap();
    writeln!(s, "#").unwrap();
    writeln!(
        s,
        "#  n   atom              position (X, Y, Z)                   charge         area        charge/area     potential"
    )
    .unwrap();
    writeln!(s, "#").unwrap();
    for (i, seg) in export.segments.iter().enumerate() {
        let ca = if seg.area.abs() > 0.0 {
            seg.charge / seg.area
        } else {
            0.0
        };
        writeln!(
            s,
            "{:5}{:5} {:14.9} {:14.9} {:14.9} {:14.9} {:14.9} {:14.9} {:14.9}",
            i + 1,
            seg.atom,
            seg.position[0],
            seg.position[1],
            seg.position[2],
            seg.charge,
            seg.area,
            ca,
            seg.potential
        )
        .unwrap();
    }
    s
}

#[derive(Debug, Clone)]
pub struct ParsedCosmo {
    pub epsilon_infinite: bool,
    pub fepsi: f64,
    pub area: f64,
    pub total_energy: f64,
    pub dielectric_energy: f64,
    pub n_atoms: usize,
    pub segment_charges: Vec<f64>,
    pub segment_areas: Vec<f64>,
    pub screening_charge_total: f64,
}

pub fn parse_cosmo(text: &str) -> ParsedCosmo {
    let mut epsilon_infinite = false;
    let mut fepsi = 0.0;
    let mut area = 0.0;
    let mut total_energy = 0.0;
    let mut dielectric_energy = 0.0;
    let mut n_atoms = 0;
    let mut screening_charge_total = 0.0;
    let mut segment_charges = Vec::new();
    let mut segment_areas = Vec::new();

    let mut section = "";
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('$') {
            section = trimmed;
            continue;
        }
        match section {
            "$cosmo" => {
                if let Some(v) = trimmed.strip_prefix("epsilon=") {
                    epsilon_infinite = v.trim() == "infinity";
                }
            }
            "$cosmo_data" => {
                if let Some(v) = trimmed.strip_prefix("fepsi=") {
                    fepsi = v.trim().parse().unwrap_or(0.0);
                } else if let Some(v) = trimmed.strip_prefix("area=") {
                    area = v.trim().parse().unwrap_or(0.0);
                }
            }
            "$coord_rad" if !trimmed.starts_with('#') && !trimmed.is_empty() => {
                n_atoms += 1;
            }
            "$screening_charge" => {
                if let Some(v) = trimmed.strip_prefix("total") {
                    screening_charge_total = v
                        .trim_start_matches([' ', '='])
                        .trim()
                        .parse()
                        .unwrap_or(0.0);
                }
            }
            "$cosmo_energy" => {
                if let Some(idx) = trimmed.find('=') {
                    let val: f64 = trimmed[idx + 1..].trim().parse().unwrap_or(0.0);
                    if trimmed.starts_with("Total energy") {
                        total_energy = val;
                    } else if trimmed.starts_with("Dielectric energy") {
                        dielectric_energy = val;
                    }
                }
            }
            "$segment_information" if !trimmed.starts_with('#') && !trimmed.is_empty() => {
                let cols: Vec<&str> = trimmed.split_whitespace().collect();
                if cols.len() >= 9 {
                    segment_charges.push(cols[5].parse().unwrap_or(0.0));
                    segment_areas.push(cols[6].parse().unwrap_or(0.0));
                }
            }
            _ => {}
        }
    }

    ParsedCosmo {
        epsilon_infinite,
        fepsi,
        area,
        total_energy,
        dielectric_energy,
        n_atoms,
        segment_charges,
        segment_areas,
        screening_charge_total,
    }
}

pub(crate) fn bohr2_to_aa2(area_bohr2: f64) -> f64 {
    area_bohr2 * BOHR * BOHR
}
