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
pub(in crate::opt::ts) mod band;
mod hungarian;
mod idpp;
mod mapping;
mod scan;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use mapping::MappingConfidence;

use crate::core::Molecule;
use crate::core::units::ANGSTROM_TO_BOHR;
use crate::opt::{OptError, Surface};

#[cfg(test)]
mod tests;

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
    /// The two atom indices (into the guess molecule) joined by this bond.
    pub atoms: (usize, usize),
    /// Separation in the assembled reactant endpoint, Bohr.
    pub reactant_distance: f64,
    /// Separation in the product, Bohr.
    pub product_distance: f64,
    /// Whether the bond forms or breaks across the reaction.
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
    /// Maximum IDPP relaxation iterations per image.
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

/// A transition-state guess plus the evidence used to build it. `#[non_exhaustive]`
/// because the energy-scanned builder ([`build_ts_guess_scanned`]) adds a path tangent
/// the plain [`build_ts_guess`] leaves unset.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TsGuess {
    /// The combined guess geometry, in reactant atom order. Feed this to
    /// [`find_transition_state`](super::find_transition_state).
    pub molecule: Molecule,
    /// `atom_map[r]` is the product atom corresponding to reactant atom `r`.
    pub atom_map: Vec<usize>,
    /// Confidence/ambiguity diagnostic for [`atom_map`](Self::atom_map): how uniquely the
    /// reactant→product correspondence was determined. See [`MappingConfidence`].
    pub mapping_confidence: MappingConfidence,
    pub reaction_coordinate: Vec<ReactionBond>,
    /// The minimum-energy-path tangent at the guess, one (unit) Cartesian vector per
    /// atom, set only by [`build_ts_guess_scanned`] (the energy-peaked scan). `None` for
    /// the geometric [`build_ts_guess`]. When present it is the preferred
    /// reaction-coordinate seed — it is the *actual* uphill direction on the surface,
    /// not the geometric forming/breaking-bond guess — and [`reaction_mode_seed`] returns
    /// it in place of the bond-vector sum.
    pub reaction_tangent: Option<Vec<[f64; 3]>>,
}

impl TsGuess {
    /// The forming/breaking-bond direction as a Cartesian reaction-coordinate seed
    /// for [`TsOptions::reaction_mode_seed`](super::TsOptions::reaction_mode_seed):
    /// each reaction bond contributes its unit bond axis to the two atoms it links —
    /// a forming bond drawing them together, a breaking bond pushing them apart — and
    /// the concerted, normalized sum (one `[f64; 3]` per atom, in the guess atom
    /// order) is returned. `None` when there is no reaction bond (e.g. a
    /// single-fragment interpolation) or the contributions cancel, in which case a
    /// caller leaves the seed unset and the saddle search falls back to `follow_mode`.
    pub fn reaction_mode_seed(&self) -> Option<Vec<[f64; 3]>> {
        // The energy-scanned builder supplies the true minimum-energy-path tangent;
        // prefer it over the geometric forming/breaking-bond direction.
        if let Some(tangent) = &self.reaction_tangent {
            return Some(tangent.clone());
        }
        if self.reaction_coordinate.is_empty() {
            return None;
        }
        let pos: Vec<[f64; 3]> = self.molecule.atoms.iter().map(|a| a.position).collect();
        let mut seed = vec![[0.0f64; 3]; pos.len()];
        for bond in &self.reaction_coordinate {
            let (i, j) = bond.atoms;
            let d = [
                pos[j][0] - pos[i][0],
                pos[j][1] - pos[i][1],
                pos[j][2] - pos[i][2],
            ];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if len < 1e-12 {
                continue;
            }
            // A forming bond moves i and j toward each other; a breaking bond moves
            // them apart. The overall sign is irrelevant (the saddle search compares
            // |overlap|), but the relative signs across a concerted set of bonds
            // define the coordinate.
            let s = match bond.kind {
                BondChange::Forming => 1.0,
                BondChange::Breaking => -1.0,
            };
            for c in 0..3 {
                let u = s * d[c] / len;
                seed[i][c] += u;
                seed[j][c] -= u;
            }
        }
        let nrm: f64 = seed.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        if nrm < 1e-12 {
            return None;
        }
        for v in &mut seed {
            for c in v.iter_mut() {
                *c /= nrm;
            }
        }
        Some(seed)
    }
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

/// The reactant-endpoint assembly shared by [`build_ts_guess`] and
/// [`build_ts_guess_scanned`]: everything up to (but not including) the IDPP image — atom
/// mapping, fragment alignment/separation, and the forming/breaking-bond reaction
/// coordinate. Splitting it out lets the energy scan reuse the exact same endpoints the
/// geometric guess interpolates between, so the two builders stay consistent.
struct Assembly {
    /// The concatenated reactant molecule (atoms in input order), positions still the
    /// raw fragment coordinates.
    reactant: Molecule,
    /// The assembled reactant-side endpoint (fragments rigidly aligned to the product
    /// image and, when multi-fragment, pulled apart), in reactant atom order.
    reactant_endpoint: Vec<[f64; 3]>,
    /// The product geometry permuted into reactant atom order (`prod_in_r[r]` is the
    /// product position of the atom mapped to reactant atom `r`).
    prod_in_r: Vec<[f64; 3]>,
    atom_map: Vec<usize>,
    mapping_confidence: MappingConfidence,
    reaction_coordinate: Vec<ReactionBond>,
}

fn assemble(
    reactant_fragments: &[Molecule],
    product: &Molecule,
    options: &GuessOptions,
) -> Result<Assembly, GuessError> {
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

    let (map, mapping_confidence) = mapping::atom_map(&z_r, &adj_r, &pos_r, &z_p, &adj_p, &pos_p);
    let prod_in_r: Vec<[f64; 3]> = (0..n).map(|i| pos_p[map[i]]).collect();

    let aligned = assembly::align_fragments(&pos_r, &prod_in_r, &fragment_id);
    let n_frag = reactant_fragments.len();
    let reactant_endpoint = if n_frag > 1 {
        assembly::separate_fragments(&aligned, &fragment_id, n_frag, options.separation)
    } else {
        aligned
    };

    let reaction_coordinate =
        assembly::reaction_bonds(&adj_r, &adj_p, &map, &reactant_endpoint, &prod_in_r);

    Ok(Assembly {
        reactant,
        reactant_endpoint,
        prod_in_r,
        atom_map: map,
        mapping_confidence,
        reaction_coordinate,
    })
}

/// Reorder `product`'s atoms into `reactant`'s order by mapping atoms across the
/// reaction (the same [`mapping::atom_map`] the guess builder uses), returning the
/// reordered product molecule (product atoms and positions permuted so atom `r` matches
/// reactant atom `r`). Lets a chain-of-states driver accept two endpoints whose atoms are
/// not already in a common order.
///
/// # Errors
/// [`GuessError`] if the two molecules disagree on atom count or element composition.
pub(in crate::opt::ts) fn reorder_product_onto_reactant(
    reactant: &Molecule,
    product: &Molecule,
    bond_factor: f64,
) -> Result<Molecule, GuessError> {
    let n = reactant.len();
    if n != product.len() {
        return Err(GuessError::AtomCountMismatch {
            reactant: n,
            product: product.len(),
        });
    }
    check_composition(reactant, product)?;
    let z_r: Vec<u32> = reactant.atoms.iter().map(|a| a.element.z()).collect();
    let z_p: Vec<u32> = product.atoms.iter().map(|a| a.element.z()).collect();
    let pos_r: Vec<[f64; 3]> = reactant.atoms.iter().map(|a| a.position).collect();
    let pos_p: Vec<[f64; 3]> = product.atoms.iter().map(|a| a.position).collect();
    let adj_r = adjacency(reactant, bond_factor);
    let adj_p = adjacency(product, bond_factor);
    let (map, _confidence) = mapping::atom_map(&z_r, &adj_r, &pos_r, &z_p, &adj_p, &pos_p);
    let atoms = (0..n).map(|r| product.atoms[map[r]]).collect();
    Ok(Molecule::new(atoms, product.charge, product.multiplicity))
}

/// Place `positions` onto a copy of `template`'s atoms (same elements/charge/multiplicity,
/// reactant atom order).
fn with_positions(template: &Molecule, positions: &[[f64; 3]]) -> Molecule {
    let mut mol = template.clone();
    for (atom, p) in mol.atoms.iter_mut().zip(positions) {
        atom.position = *p;
    }
    mol
}

/// Build a transition-state guess from reactant fragment(s) and a product. The
/// fragments are concatenated in input order; the product must hold the same
/// multiset of elements. The guess takes its charge from the summed fragment
/// charges and its spin multiplicity from `product`.
///
/// This is the geometric (surface-free) builder: it places the guess at a single IDPP
/// image at `options.interpolation`. For a guess placed at the *energy* maximum of the
/// interpolated path — a better single-point transition-state guess when a surface is
/// affordable — see [`build_ts_guess_scanned`].
///
/// # Errors
/// [`GuessError`] if no fragments are given, or the reactant and product disagree
/// on atom count or element composition.
pub fn build_ts_guess(
    reactant_fragments: &[Molecule],
    product: &Molecule,
    options: &GuessOptions,
) -> Result<TsGuess, GuessError> {
    let assembled = assemble(reactant_fragments, product, options)?;
    let guess_pos = idpp::idpp_image(&assembled.reactant_endpoint, &assembled.prod_in_r, options);
    Ok(TsGuess {
        molecule: with_positions(&assembled.reactant, &guess_pos),
        atom_map: assembled.atom_map,
        mapping_confidence: assembled.mapping_confidence,
        reaction_coordinate: assembled.reaction_coordinate,
        reaction_tangent: None,
    })
}

/// Knobs for the energy-peaked scan ([`build_ts_guess_scanned`]); construct via
/// [`ScanOptions::default`]. `#[non_exhaustive]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScanOptions {
    /// The IDPP guess-builder controls (atom-map bond cutoff, fragment separation, the
    /// per-image IDPP relaxation). The `interpolation` fraction is overridden per scan
    /// point.
    pub guess: GuessOptions,
    /// Number of interior path points to evaluate the surface at (must be ≥ 3 so the
    /// peak can be parabola-bracketed). More points resolve the barrier top at
    /// proportionally more single-point energies.
    pub n_points: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            guess: GuessOptions::default(),
            n_points: 11,
        }
    }
}

/// A failure of the energy-peaked scan ([`build_ts_guess_scanned`]). `#[non_exhaustive]`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ScanError {
    /// The reactant/product endpoints could not be assembled (see [`GuessError`]).
    #[error(transparent)]
    Guess(#[from] GuessError),
    /// A surface energy evaluation along the path failed (see [`OptError`]).
    #[error("transition-state guess scan: surface evaluation failed: {0}")]
    Surface(#[from] OptError),
    /// Fewer than three scan points were requested, so the peak cannot be bracketed.
    #[error("transition-state guess scan needs at least 3 path points, got {0}")]
    TooFewPoints(usize),
}

/// Build a transition-state guess at the **energy maximum** of the interpolated path —
/// a cheap "poor man's path method".
///
/// Like [`build_ts_guess`] it assembles the same reactant endpoint and maps atoms, but
/// instead of placing the guess at a fixed interpolation fraction it evaluates `surface`
/// at [`ScanOptions::n_points`] IDPP images spanning the path, parabola-fits the energy
/// peak, and returns the image at the fitted maximum. The returned [`TsGuess`] carries
/// the minimum-energy-path tangent at that peak in
/// [`reaction_tangent`](TsGuess::reaction_tangent), so
/// [`reaction_mode_seed`](TsGuess::reaction_mode_seed) hands the saddle search the true
/// uphill direction rather than the geometric bond-vector guess.
///
/// `surface` is queried for energies only (no gradient), one image at a time —
/// `n_points` single points plus two for the tangent finite difference.
///
/// # Errors
/// [`ScanError::Guess`] if the endpoints cannot be assembled, [`ScanError::Surface`] if a
/// path energy evaluation fails, or [`ScanError::TooFewPoints`] if `n_points < 3`.
pub fn build_ts_guess_scanned<S: Surface>(
    reactant_fragments: &[Molecule],
    product: &Molecule,
    surface: &mut S,
    options: &ScanOptions,
) -> Result<TsGuess, ScanError> {
    if options.n_points < 3 {
        return Err(ScanError::TooFewPoints(options.n_points));
    }
    let assembled = assemble(reactant_fragments, product, &options.guess)?;
    let peak = scan::scan_peak(
        &assembled.reactant_endpoint,
        &assembled.prod_in_r,
        &options.guess,
        options.n_points,
        surface,
    )?;
    Ok(TsGuess {
        molecule: with_positions(&assembled.reactant, &peak.geometry),
        atom_map: assembled.atom_map,
        mapping_confidence: assembled.mapping_confidence,
        reaction_coordinate: assembled.reaction_coordinate,
        reaction_tangent: Some(peak.tangent),
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
