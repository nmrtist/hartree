//! Reactant→product atom mapping. Tries an exact subgraph monomorphism (every
//! reactant bond preserved — correct for additions/cycloadditions, which break no
//! bonds); when a bond breaks and no embedding exists, falls back to a minimum-cost
//! (Hungarian) assignment over a connectivity-signature-plus-geometry cost.
//!
//! A connectivity-only embedding is not unique for a molecule with equivalent atoms (the
//! three hydrogens of a methyl, atoms related by a symmetry operation): several
//! edge-preserving maps exist, differing only in how they permute the interchangeable atoms.
//! So the monomorphism path enumerates the embeddings (bounded) and, after a rigid Kabsch
//! fit, picks the one minimizing the aligned atom-position discrepancy. The reported
//! confidence reflects how cleanly geometry separates that choice from the next-best,
//! dropping toward zero when the candidates are near-degenerate (interchangeable atoms).
//!
//! The Hungarian fallback bootstraps in two passes: a signatures-only assignment gives a
//! coarse correspondence used to Kabsch-align the reactant onto the product, then a
//! second assignment over `signature + aligned-distance` cost refines it. Element
//! mismatches are forbidden (a large finite penalty), and the geometry term is normalized
//! so its *total* over an assignment stays below one signature shell — geometry therefore
//! only breaks ties between connectivity-equivalent atoms and can never overturn a
//! connectivity-better correspondence (see [`build_cost`]). A confidence/ambiguity
//! diagnostic flags atoms that have an equal-cost alternative assignment (e.g. genuinely
//! symmetric fragments geometry cannot disambiguate).
//!
//! The Kabsch step is a single *global* rigid fit. For a multi-fragment reactant whose
//! fragments sit in arbitrary relative frames (they are assembled into a common frame only
//! *after* mapping), that global fit cannot co-align the fragments, so the geometric
//! tie-break is best-effort there; connectivity signatures still determine the
//! correspondence, and the diagnostic reports the residual ambiguity.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::hungarian;
use crate::ext::kabsch::optimal_rotation;

mod cycles;
mod symmetry;

const SIGNATURE_SHELLS: usize = 3;

/// Cost of pairing atoms of different elements: forbidden, but finite (the Hungarian
/// solver does arithmetic on costs, so `f64::INFINITY` is unusable). Far larger than any
/// realistic sum of signature + normalized-geometry costs, so a feasible same-element
/// matching — guaranteed to exist when the element multisets agree, as the caller checks
/// — never selects one.
const ELEMENT_PENALTY: f64 = 1.0e6;

/// Weight of one differing signature shell. The geometry term is normalized (in
/// [`build_cost`]) so the *total* geometric contribution of any assignment is below `1`,
/// strictly less than this; hence a single differing signature shell outranks every
/// accumulated geometric difference, and geometry only ever breaks ties between
/// assignments of equal total connectivity-signature cost.
const SIGNATURE_WEIGHT: f64 = 10.0;

/// Largest "no distinguishable alternative" cost gap: an off-assignment 2-swap whose
/// total cost is within this of the chosen assignment marks both atoms ambiguous.
const AMBIGUITY_TOL: f64 = 1.0e-6;

/// Upper bound on the number of edge-preserving embeddings enumerated when an exact
/// monomorphism exists. A highly symmetric molecule admits many automorphic embeddings;
/// the geometric tie-break only needs enough of them to separate the equivalent atoms, so
/// the search stops once this many have been collected (keeping the cost away from `N!`).
const MAX_MONOMORPHISMS: usize = 4096;

/// Confidence floor for the geometry-chosen embedding: a positive best-to-second separation
/// maps to a confidence in `[FLOOR, 1]` growing with the separation, while a vanishing
/// separation reports `0` (the equivalent atoms are genuinely interchangeable).
const GEOMETRIC_CONFIDENCE_FLOOR: f64 = 0.5;

/// Squared-distance separation between the best and second-best embedding above which the
/// geometric choice is treated as fully unambiguous (confidence `1.0`).
const GEOMETRIC_SEPARATION_FULL: f64 = 1.0e-2;

/// Confidence/ambiguity diagnostic for a reactant→product atom map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingConfidence {
    /// Fraction of atoms with no equal-cost alternative assignment, in `[0, 1]`: `1.0`
    /// means every atom's correspondence is uniquely determined (by element, then
    /// connectivity, then geometry); lower values mean some atoms are interchangeable.
    pub confidence: f64,
    /// Reactant atoms whose mapped product partner could be exchanged at no cost increase —
    /// a free pairwise swap or a longer cyclic reassignment of a symmetric orbit (e.g.
    /// symmetric hydrogens geometry does not separate). Empty for an unambiguous map.
    pub ambiguous: Vec<usize>,
}

/// Map reactant atoms onto product atoms, returning the map (`map[r]` is the product
/// atom for reactant atom `r`) and a [`MappingConfidence`] diagnostic. Prefers an exact
/// subgraph monomorphism; falls back to the geometry-refined Hungarian assignment when a
/// bond breaks.
pub(super) fn atom_map(
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    pos_r: &[[f64; 3]],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
    pos_p: &[[f64; 3]],
) -> (Vec<usize>, MappingConfidence) {
    let embeddings = enumerate_monomorphisms(z_r, adj_r, z_p, adj_p);
    if !embeddings.is_empty() {
        // An exact embedding exists, but it is not unique when the molecule has equivalent
        // atoms: choose the candidate whose Kabsch-aligned positions best match the
        // reactant, and report a confidence reflecting the geometric margin to the next.
        return choose_by_geometry(pos_r, pos_p, embeddings);
    }
    let sig_r = signatures(adj_r, z_r);
    let sig_p = signatures(adj_p, z_p);
    let map = hungarian_map(z_r, &sig_r, pos_r, z_p, &sig_p, pos_p);
    let confidence = diagnose(z_r, &sig_r, pos_r, z_p, &sig_p, pos_p, &map);
    (map, confidence)
}

/// Geometry-refined Hungarian assignment: a signatures-only pass seeds a Kabsch
/// alignment, then a `signature + aligned-distance` pass refines the correspondence.
fn hungarian_map(
    z_r: &[u32],
    sig_r: &[Vec<u64>],
    pos_r: &[[f64; 3]],
    z_p: &[u32],
    sig_p: &[Vec<u64>],
    pos_p: &[[f64; 3]],
) -> Vec<usize> {
    // Bootstrap: a connectivity-only assignment (no geometry).
    let cost0 = build_cost(z_r, sig_r, z_p, sig_p, None);
    let map0 = hungarian::solve(&cost0);

    // Refine: align the reactant onto the product under the bootstrap map, then re-solve
    // with the aligned-distance tie-break folded in.
    let aligned = align_by_map(pos_r, pos_p, &map0);
    let cost1 = build_cost(z_r, sig_r, z_p, sig_p, Some((&aligned, pos_p)));
    hungarian::solve(&cost1)
}

/// The `(aligned-reactant, product)` positions supplying the geometric tie-break term in
/// [`build_cost`]; `None` selects a signatures-only cost.
type Geometry<'a> = (&'a [[f64; 3]], &'a [[f64; 3]]);

/// The `n×n` assignment cost. Cross-element pairs cost [`ELEMENT_PENALTY`]; same-element
/// pairs cost `SIGNATURE_WEIGHT · (differing shells)` plus, when `geometry` is supplied, a
/// squared aligned-distance tie-break.
///
/// The geometry term is normalized by `(max same-element d² + 1) · n` so each cell is in
/// `[0, 1/n)` and the **whole** geometry sum an assignment can carry is `< 1` — strictly
/// less than one [`SIGNATURE_WEIGHT`]. The Hungarian solver minimizes the *total* cost, so
/// this per-`n` normalization (not a per-cell one) is what guarantees a single differing
/// signature shell outranks *any* accumulated geometric difference across all `n` atoms:
/// geometry can only ever break ties between assignments of equal total signature cost,
/// never overturn a connectivity-better correspondence.
fn build_cost(
    z_r: &[u32],
    sig_r: &[Vec<u64>],
    z_p: &[u32],
    sig_p: &[Vec<u64>],
    geometry: Option<Geometry>,
) -> Vec<Vec<f64>> {
    let n = z_r.len();
    // Normalizer for the geometry term: the largest same-element squared distance, plus
    // one (so a single cell is < 1), times n (so the SUM over a matching is < 1). The
    // total geometry contribution is therefore strictly below one SIGNATURE_WEIGHT.
    let norm = geometry
        .map(|(ar, pp)| {
            let mut max_d2 = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    if z_r[i] == z_p[j] {
                        max_d2 = max_d2.max(dist2(ar[i], pp[j]));
                    }
                }
            }
            (max_d2 + 1.0) * n.max(1) as f64
        })
        .unwrap_or(1.0);

    let mut cost = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            cost[i][j] = if z_r[i] != z_p[j] {
                ELEMENT_PENALTY
            } else {
                let sig = SIGNATURE_WEIGHT * signature_cost(&sig_r[i], &sig_p[j]) as f64;
                let geo = geometry.map_or(0.0, |(ar, pp)| dist2(ar[i], pp[j]) / norm);
                sig + geo
            };
        }
    }
    cost
}

/// Rigidly Kabsch-align the reactant positions onto the product positions permuted by
/// `map` (`target[i] = pos_p[map[i]]`), returning the aligned reactant in the product
/// frame. Used only to score a geometric tie-break, so a rough fit suffices.
fn align_by_map(pos_r: &[[f64; 3]], pos_p: &[[f64; 3]], map: &[usize]) -> Vec<[f64; 3]> {
    let n = pos_r.len();
    let target: Vec<[f64; 3]> = (0..n).map(|i| pos_p[map[i]]).collect();
    let cr = centroid(pos_r);
    let ct = centroid(&target);
    let pc: Vec<[f64; 3]> = pos_r.iter().map(|a| sub(*a, cr)).collect();
    let tc: Vec<[f64; 3]> = target.iter().map(|a| sub(*a, ct)).collect();
    let rot = optimal_rotation(&pc, &tc);
    pc.iter()
        .map(|a| {
            let r = matvec(&rot, *a);
            [r[0] + ct[0], r[1] + ct[1], r[2] + ct[2]]
        })
        .collect()
}

/// The confidence/ambiguity diagnostic for `map`. Two complementary tests flag an atom
/// whose correspondence is not uniquely determined: (1) a directed "free reassignment"
/// graph over the fixed alignment, where an atom on a cycle (SCC of size > 1) can be
/// cyclically reassigned at no cost — catching atoms left near-coincident by the alignment;
/// and (2) a per-group symmetry test that, within a set of equivalent atoms, marks a
/// symmetric orbit whose cyclic rotation re-aligns as well as the current map. The second
/// flags a 3-fold (or higher) orbit a pairwise-only check misses: a single swap of such an
/// orbit is an improper move the rigid fit cannot match, but the full rotation is proper.
fn diagnose(
    z_r: &[u32],
    sig_r: &[Vec<u64>],
    pos_r: &[[f64; 3]],
    z_p: &[u32],
    sig_p: &[Vec<u64>],
    pos_p: &[[f64; 3]],
    map: &[usize],
) -> MappingConfidence {
    let n = map.len();
    if n == 0 {
        return MappingConfidence {
            confidence: 1.0,
            ambiguous: Vec::new(),
        };
    }
    let aligned = align_by_map(pos_r, pos_p, map);
    let cost = build_cost(z_r, sig_r, z_p, sig_p, Some((&aligned, pos_p)));

    // Test 1: directed free-reassignment graph (see the doc comment).
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        let own = cost[i][map[i]];
        for k in 0..n {
            if k == i {
                continue;
            }
            let alt = cost[i][map[k]];
            if alt < ELEMENT_PENALTY && alt - own < AMBIGUITY_TOL {
                adj[i].push(k);
            }
        }
    }
    let mut ambiguous = cycles::atoms_on_cycles(&adj);

    // Test 2: cost-neutral cyclic rotation within each connectivity-equivalent group, scored
    // by a rigid re-alignment of the whole candidate map.
    let residual = |candidate: &[usize]| alignment_residual(pos_r, pos_p, candidate);
    symmetry::flag_symmetric_groups(z_r, sig_r, pos_r, map, residual, &mut ambiguous);

    let amb: Vec<usize> = (0..n).filter(|&i| ambiguous[i]).collect();
    MappingConfidence {
        confidence: (n - amb.len()) as f64 / n as f64,
        ambiguous: amb,
    }
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for p in points {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    let inv = 1.0 / points.len().max(1) as f64;
    [c[0] * inv, c[1] * inv, c[2] * inv]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn matvec(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// A single injective, element-respecting, edge-preserving map from the reactant graph
/// into the product graph, or `None` if no such embedding exists (a bond breaks). A thin
/// wrapper over [`enumerate_monomorphisms`] returning its first embedding.
#[cfg(test)]
fn map_monomorphism(
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
) -> Option<Vec<usize>> {
    enumerate_monomorphisms(z_r, adj_r, z_p, adj_p)
        .into_iter()
        .next()
}

/// Every injective, element-respecting, edge-preserving embedding of the reactant graph
/// into the product graph (empty if none exists — a bond breaks), up to
/// [`MAX_MONOMORPHISMS`]. Backtracking with element/degree/neighbour pruning, in a
/// connected visitation order so each atom (after the first of its fragment) has a mapped
/// neighbour. Distinct connectivity environments admit only one image, so the branching
/// that produces multiple embeddings happens only among equivalent atoms — exactly the
/// permutations a geometric tie-break must consider.
fn enumerate_monomorphisms(
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let order = connected_order(adj_r);
    let mut map = vec![usize::MAX; z_r.len()];
    // `used` is indexed by product atom, so it must span the product (which may be larger
    // than the reactant when the reactant embeds as a subgraph with spectator atoms).
    let mut used = vec![false; z_p.len()];
    let mut out = Vec::new();
    mono_backtrack(
        0, &order, z_r, adj_r, z_p, adj_p, &mut map, &mut used, &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn mono_backtrack(
    pos: usize,
    order: &[usize],
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
    map: &mut [usize],
    used: &mut [bool],
    out: &mut Vec<Vec<usize>>,
) {
    if out.len() >= MAX_MONOMORPHISMS {
        return;
    }
    if pos == order.len() {
        out.push(map.to_vec());
        return;
    }
    let i = order[pos];
    for j in 0..z_p.len() {
        if used[j] || z_p[j] != z_r[i] || adj_p[j].len() < adj_r[i].len() {
            continue;
        }
        let consistent = adj_r[i]
            .iter()
            .all(|&k| map[k] == usize::MAX || adj_p[j].contains(&map[k]));
        if !consistent {
            continue;
        }
        map[i] = j;
        used[j] = true;
        mono_backtrack(pos + 1, order, z_r, adj_r, z_p, adj_p, map, used, out);
        map[i] = usize::MAX;
        used[j] = false;
        if out.len() >= MAX_MONOMORPHISMS {
            return;
        }
    }
}

/// Pick the embedding whose Kabsch-aligned reactant best matches the product, and report a
/// confidence from the geometric margin to the next-best candidate. With a single embedding
/// the choice is unique and fully confident; with several, the residual after alignment
/// breaks the tie and a small best-to-second margin lowers the confidence toward zero (the
/// permuted atoms are geometrically interchangeable).
fn choose_by_geometry(
    pos_r: &[[f64; 3]],
    pos_p: &[[f64; 3]],
    embeddings: Vec<Vec<usize>>,
) -> (Vec<usize>, MappingConfidence) {
    let mut scored: Vec<(f64, Vec<usize>)> = embeddings
        .into_iter()
        .map(|m| (alignment_residual(pos_r, pos_p, &m), m))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let best = scored[0].1.clone();
    let confidence = if scored.len() < 2 {
        1.0
    } else {
        let separation = (scored[1].0 - scored[0].0).max(0.0);
        if separation <= AMBIGUITY_TOL {
            0.0
        } else {
            let frac = (separation / GEOMETRIC_SEPARATION_FULL).min(1.0);
            GEOMETRIC_CONFIDENCE_FLOOR + (1.0 - GEOMETRIC_CONFIDENCE_FLOOR) * frac
        }
    };
    // Flag the atoms whose image differs between the best and a near-degenerate runner-up:
    // those are the interchangeable ones geometry could not cleanly separate.
    let ambiguous = if confidence >= 1.0 {
        Vec::new()
    } else {
        let runner = &scored[1].1;
        (0..best.len()).filter(|&i| best[i] != runner[i]).collect()
    };
    (
        best,
        MappingConfidence {
            confidence,
            ambiguous,
        },
    )
}

/// Mean squared per-atom discrepancy after rigidly Kabsch-aligning the reactant onto the
/// product permuted by `map` — the geometric score of one candidate embedding.
fn alignment_residual(pos_r: &[[f64; 3]], pos_p: &[[f64; 3]], map: &[usize]) -> f64 {
    let n = pos_r.len();
    if n == 0 {
        return 0.0;
    }
    let aligned = align_by_map(pos_r, pos_p, map);
    let mut sum = 0.0;
    for i in 0..n {
        sum += dist2(aligned[i], pos_p[map[i]]);
    }
    sum / n as f64
}

/// Breadth-first order over all components, each started from its highest-degree atom.
fn connected_order(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    loop {
        let start = (0..n)
            .filter(|&i| !visited[i])
            .max_by_key(|&i| adj[i].len());
        let Some(start) = start else { break };
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &w in &adj[v] {
                if !visited[w] {
                    visited[w] = true;
                    queue.push_back(w);
                }
            }
        }
    }
    order
}

/// Per atom, a Morgan-style label per shell, folding in larger neighbourhoods.
fn signatures(adj: &[Vec<usize>], z: &[u32]) -> Vec<Vec<u64>> {
    let n = z.len();
    let mut labels: Vec<u64> = z.iter().map(|&zi| zi as u64).collect();
    let mut out: Vec<Vec<u64>> = labels.iter().map(|&l| vec![l]).collect();
    for _ in 1..=SIGNATURE_SHELLS {
        let mut next = vec![0u64; n];
        for i in 0..n {
            let mut neigh: Vec<u64> = adj[i].iter().map(|&k| labels[k]).collect();
            neigh.sort_unstable();
            next[i] = hash_label(labels[i], &neigh);
        }
        for i in 0..n {
            out[i].push(next[i]);
        }
        labels = next;
    }
    out
}

fn hash_label(own: u64, neigh: &[u64]) -> u64 {
    let mut h = 1469598103934665603u64;
    let mut mix = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
    };
    mix(own);
    for &v in neigh {
        mix(v);
    }
    h
}

fn signature_cost(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[cfg(test)]
mod tests;
