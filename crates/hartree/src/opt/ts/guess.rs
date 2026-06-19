//! Transition-state guess construction from reaction endpoints, for callers that
//! have a reactant + product rather than a near-saddle geometry.
//!
//! It is IDPP-first (image-dependent pair potential, Smidstrup *et al.*, J. Chem.
//! Phys. 140, 214106 (2014)): interpolate the interatomic-distance matrix between
//! the endpoints and relax an image to match, instead of linearly interpolating
//! Cartesians (which drives atoms through each other). The three stages are atom
//! mapping ([`mapping`]), reactant-endpoint assembly ([`assembly`]), and the IDPP
//! image ([`idpp`]). The forming/breaking bonds found along the way are returned
//! as the reaction coordinate.

mod assembly;
mod idpp;
mod mapping;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::Molecule;
use crate::core::units::ANGSTROM_TO_BOHR;

/// How a bond changes between reactant and product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BondChange {
    /// Absent in the reactant, present in the product.
    Forming,
    /// Present in the reactant, absent in the product.
    Breaking,
}

/// A bond that changes across the reaction. Indices are into the guess molecule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionBond {
    pub atoms: (usize, usize),
    /// Separation in the assembled reactant endpoint, Bohr.
    pub reactant_distance: f64,
    /// Separation in the product, Bohr.
    pub product_distance: f64,
    pub kind: BondChange,
}

/// Knobs for [`build_ts_guess`]; construct via [`GuessOptions::default`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GuessOptions {
    /// Interpolation fraction `λ ∈ [0, 1]` from reactant (0) to product (1) for the
    /// IDPP target distances.
    pub interpolation: f64,
    /// How far (Bohr) to pull fragments apart when assembling the reactant
    /// endpoint. Ignored for a single-fragment reactant.
    pub separation: f64,
    pub idpp_max_iter: usize,
    /// IDPP convergence: largest objective-gradient component.
    pub idpp_tol: f64,
    /// Covalent-radius multiplier for the bond cutoff.
    pub bond_factor: f64,
}

impl Default for GuessOptions {
    fn default() -> Self {
        Self {
            interpolation: 0.5,
            separation: 0.5 * ANGSTROM_TO_BOHR,
            idpp_max_iter: 400,
            idpp_tol: 1e-4,
            bond_factor: 1.3,
        }
    }
}

/// A transition-state guess plus the evidence used to build it.
#[derive(Debug, Clone)]
pub struct TsGuess {
    /// The combined guess geometry, in reactant atom order. Feed this to
    /// [`find_transition_state`](super::find_transition_state).
    pub molecule: Molecule,
    /// `atom_map[r]` is the product atom corresponding to reactant atom `r`.
    pub atom_map: Vec<usize>,
    pub reaction_coordinate: Vec<ReactionBond>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GuessError {
    #[error("no reactant fragments supplied")]
    Empty,
    #[error("reactant has {reactant} atoms but product has {product}")]
    AtomCountMismatch { reactant: usize, product: usize },
    #[error("reactant and product have different element compositions: {0}")]
    ElementMismatch(String),
}

/// Build a transition-state guess from reactant fragment(s) and a product. The
/// fragments are concatenated in input order; the product must hold the same
/// multiset of elements. The guess takes its charge from the summed fragment
/// charges and its spin multiplicity from `product`.
///
/// # Errors
/// [`GuessError`] if no fragments are given, or the reactant and product disagree
/// on atom count or element composition.
pub fn build_ts_guess(
    reactant_fragments: &[Molecule],
    product: &Molecule,
    options: &GuessOptions,
) -> Result<TsGuess, GuessError> {
    if reactant_fragments.is_empty() {
        return Err(GuessError::Empty);
    }

    let mut atoms = Vec::new();
    let mut fragment_id = Vec::new();
    let mut charge = 0;
    for (fid, frag) in reactant_fragments.iter().enumerate() {
        for atom in &frag.atoms {
            atoms.push(*atom);
            fragment_id.push(fid);
        }
        charge += frag.charge;
    }
    let reactant = Molecule::new(atoms, charge, product.multiplicity);

    let n = reactant.len();
    if n != product.len() {
        return Err(GuessError::AtomCountMismatch {
            reactant: n,
            product: product.len(),
        });
    }
    check_composition(&reactant, product)?;

    let z_r: Vec<u32> = reactant.atoms.iter().map(|a| a.element.z()).collect();
    let z_p: Vec<u32> = product.atoms.iter().map(|a| a.element.z()).collect();
    let pos_r: Vec<[f64; 3]> = reactant.atoms.iter().map(|a| a.position).collect();
    let pos_p: Vec<[f64; 3]> = product.atoms.iter().map(|a| a.position).collect();

    // Reactant adjacency is per fragment (no inter-fragment bonds; fragment
    // coordinates may even overlap in space).
    let adj_r = fragment_adjacency(&reactant, &fragment_id, options.bond_factor);
    let adj_p = adjacency(product, options.bond_factor);

    let map = mapping::atom_map(&z_r, &adj_r, &z_p, &adj_p);
    let prod_in_r: Vec<[f64; 3]> = (0..n).map(|i| pos_p[map[i]]).collect();

    let aligned = assembly::align_fragments(&pos_r, &prod_in_r, &fragment_id);
    let n_frag = reactant_fragments.len();
    let reactant_endpoint = if n_frag > 1 {
        assembly::separate_fragments(&aligned, &fragment_id, n_frag, options.separation)
    } else {
        aligned
    };

    let guess_pos = idpp::idpp_image(&reactant_endpoint, &prod_in_r, options);
    let mut guess = reactant;
    for (atom, p) in guess.atoms.iter_mut().zip(&guess_pos) {
        atom.position = *p;
    }

    let reaction_coordinate =
        assembly::reaction_bonds(&adj_r, &adj_p, &map, &reactant_endpoint, &prod_in_r);

    Ok(TsGuess {
        molecule: guess,
        atom_map: map,
        reaction_coordinate,
    })
}

fn check_composition(reactant: &Molecule, product: &Molecule) -> Result<(), GuessError> {
    let mut count = std::collections::BTreeMap::new();
    for a in &reactant.atoms {
        *count.entry(a.element.z()).or_insert(0i64) += 1;
    }
    for a in &product.atoms {
        *count.entry(a.element.z()).or_insert(0i64) -= 1;
    }
    if count.values().any(|&c| c != 0) {
        let detail: Vec<String> = count
            .iter()
            .filter(|&(_, &c)| c != 0)
            .map(|(z, c)| format!("Z={z}:{c:+}"))
            .collect();
        return Err(GuessError::ElementMismatch(detail.join(", ")));
    }
    Ok(())
}

pub(super) fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn adjacency(mol: &Molecule, bond_factor: f64) -> Vec<Vec<usize>> {
    let n = mol.len();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let cutoff = bond_factor
                * (mol.atoms[i].element.covalent_radius() + mol.atoms[j].element.covalent_radius());
            if distance(pos[i], pos[j]) < cutoff {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

fn fragment_adjacency(mol: &Molecule, fragment_id: &[usize], bond_factor: f64) -> Vec<Vec<usize>> {
    let n = mol.len();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if fragment_id[i] != fragment_id[j] {
                continue;
            }
            let cutoff = bond_factor
                * (mol.atoms[i].element.covalent_radius() + mol.atoms[j].element.covalent_radius());
            if distance(pos[i], pos[j]) < cutoff {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element};

    fn atom(z: u32, p: [f64; 3]) -> Atom {
        Atom::new(Element::from_z(z).unwrap(), p)
    }

    fn ethane() -> Molecule {
        let cc = 2.9;
        let ch = 2.06;
        Molecule::new(
            vec![
                atom(6, [0.0, 0.0, 0.0]),
                atom(6, [0.0, 0.0, cc]),
                atom(1, [ch, 0.0, -0.7]),
                atom(1, [-ch * 0.5, ch * 0.87, -0.7]),
                atom(1, [-ch * 0.5, -ch * 0.87, -0.7]),
                atom(1, [ch * 0.5, ch * 0.87, cc + 0.7]),
                atom(1, [ch * 0.5, -ch * 0.87, cc + 0.7]),
                atom(1, [-ch, 0.0, cc + 0.7]),
            ],
            0,
            1,
        )
    }

    fn methyl(center: [f64; 3]) -> Molecule {
        let ch = 2.06;
        Molecule::new(
            vec![
                atom(6, center),
                atom(1, [center[0] + ch, center[1], center[2]]),
                atom(1, [center[0] - ch * 0.5, center[1] + ch * 0.87, center[2]]),
                atom(1, [center[0] - ch * 0.5, center[1] - ch * 0.87, center[2]]),
            ],
            0,
            2,
        )
    }

    #[test]
    fn rejects_mismatched_composition() {
        let product = ethane();
        let frag = methyl([0.0, 0.0, 0.0]);
        let err = build_ts_guess(&[frag], &product, &GuessOptions::default()).unwrap_err();
        assert!(matches!(err, GuessError::AtomCountMismatch { .. }));
    }

    #[test]
    fn maps_atoms_and_finds_forming_bond() {
        let product = ethane();
        let frag_a = methyl([0.0, 0.0, 0.0]);
        let frag_b = methyl([10.0, 0.0, 0.0]);
        let guess = build_ts_guess(&[frag_a, frag_b], &product, &GuessOptions::default()).unwrap();

        let mut seen = vec![false; product.len()];
        for (r, &p) in guess.atom_map.iter().enumerate() {
            assert!(!seen[p], "product atom {p} mapped twice");
            seen[p] = true;
            assert_eq!(
                guess.molecule.atoms[r].element.z(),
                product.atoms[p].element.z()
            );
        }

        let forming: Vec<&ReactionBond> = guess
            .reaction_coordinate
            .iter()
            .filter(|b| b.kind == BondChange::Forming)
            .collect();
        assert_eq!(forming.len(), 1, "expected one forming bond");
        let (a, b) = forming[0].atoms;
        assert_eq!(guess.molecule.atoms[a].element.z(), 6);
        assert_eq!(guess.molecule.atoms[b].element.z(), 6);

        let cc = distance(
            guess.molecule.atoms[a].position,
            guess.molecule.atoms[b].position,
        );
        assert!(cc > 2.9, "forming C–C should be stretched, got {cc} bohr");
        let mut min_d = f64::INFINITY;
        for i in 0..guess.molecule.len() {
            for j in (i + 1)..guess.molecule.len() {
                min_d = min_d.min(distance(
                    guess.molecule.atoms[i].position,
                    guess.molecule.atoms[j].position,
                ));
            }
        }
        assert!(min_d > 1.0, "atoms collapsed (min distance {min_d} bohr)");
    }

    #[test]
    fn breaks_cc_bond_and_maps_atoms() {
        // Reactant is one ethane fragment (C–C bonded). The product is the same
        // atoms but with the two methyls pulled ~10 Bohr apart, breaking the C–C.
        let reactant = ethane();
        let mut product_atoms = methyl([0.0, 0.0, 0.0]).atoms;
        product_atoms.extend(methyl([10.0, 0.0, 0.0]).atoms);
        let product = Molecule::new(product_atoms, 0, 1);

        let guess = build_ts_guess(&[reactant], &product, &GuessOptions::default()).unwrap();

        // Map is total + injective + element-respecting.
        let mut seen = vec![false; product.len()];
        for (r, &p) in guess.atom_map.iter().enumerate() {
            assert!(p < product.len(), "reactant atom {r} unassigned (got {p})");
            assert!(!seen[p], "product atom {p} mapped twice");
            seen[p] = true;
            assert_eq!(
                guess.molecule.atoms[r].element.z(),
                product.atoms[p].element.z(),
                "element mismatch at reactant atom {r}"
            );
        }

        // Exactly one breaking bond, between the two carbons, and no forming bond.
        let breaking: Vec<&ReactionBond> = guess
            .reaction_coordinate
            .iter()
            .filter(|b| b.kind == BondChange::Breaking)
            .collect();
        assert_eq!(breaking.len(), 1, "expected one breaking bond");
        let forming = guess
            .reaction_coordinate
            .iter()
            .filter(|b| b.kind == BondChange::Forming)
            .count();
        assert_eq!(forming, 0, "expected no forming bond");
        let (a, b) = breaking[0].atoms;
        assert_eq!(guess.molecule.atoms[a].element.z(), 6);
        assert_eq!(guess.molecule.atoms[b].element.z(), 6);

        // No atoms collapsed onto each other in the assembled guess.
        let mut min_d = f64::INFINITY;
        for i in 0..guess.molecule.len() {
            for j in (i + 1)..guess.molecule.len() {
                min_d = min_d.min(distance(
                    guess.molecule.atoms[i].position,
                    guess.molecule.atoms[j].position,
                ));
            }
        }
        assert!(min_d > 1.0, "atoms collapsed (min distance {min_d} bohr)");
    }

    #[test]
    fn unimolecular_single_fragment_interpolates() {
        let reactant = methyl([0.0, 0.0, 0.0]);
        let mut product = methyl([0.0, 0.0, 0.0]);
        for a in &mut product.atoms {
            a.position[2] += 0.3;
        }
        let guess = build_ts_guess(&[reactant], &product, &GuessOptions::default()).unwrap();
        assert_eq!(guess.molecule.len(), 4);
        assert!(guess.reaction_coordinate.is_empty());
    }
}
