//! Reactant→product atom mapping. Tries an exact subgraph monomorphism (every
//! reactant bond preserved — correct for additions/cycloadditions, which break no
//! bonds); when a bond breaks and no embedding exists, falls back to a minimum-cost
//! (Hungarian) assignment over a connectivity-signature-plus-geometry cost.
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

use super::hungarian;
use crate::ext::kabsch::optimal_rotation;

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

/// Confidence/ambiguity diagnostic for a reactant→product atom map.
#[derive(Debug, Clone)]
pub struct MappingConfidence {
    /// Fraction of atoms with no equal-cost alternative assignment, in `[0, 1]`: `1.0`
    /// means every atom's correspondence is uniquely determined (by element, then
    /// connectivity, then geometry); lower values mean some atoms are interchangeable.
    /// A monomorphism (exact edge-preserving embedding) reports `1.0`.
    pub confidence: f64,
    /// Reactant atoms involved in a near-zero-cost feasible 2-swap — those whose mapped
    /// product partner could be exchanged with another atom's at no cost increase (e.g.
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
    if let Some(map) = map_monomorphism(z_r, adj_r, z_p, adj_p) {
        // An exact edge-preserving embedding is topologically determined; report full
        // confidence rather than a geometric swap diagnostic (which would second-guess a
        // correct embedding on automorphic fragments).
        return (
            map,
            MappingConfidence {
                confidence: 1.0,
                ambiguous: Vec::new(),
            },
        );
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

/// The confidence/ambiguity diagnostic for `map`: builds the final `signature + geometry`
/// cost (aligned under `map`) and flags every atom that has a feasible, no-cost-increase
/// 2-swap of its assignment.
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

    let mut ambiguous = vec![false; n];
    for i in 0..n {
        for k in (i + 1)..n {
            let (ji, jk) = (map[i], map[k]);
            let current = cost[i][ji] + cost[k][jk];
            let swapped = cost[i][jk] + cost[k][ji];
            // A feasible (same-element, finite) swap that does not raise the total cost
            // means atoms i and k are interchangeable under this map.
            if swapped < ELEMENT_PENALTY && swapped - current < AMBIGUITY_TOL {
                ambiguous[i] = true;
                ambiguous[k] = true;
            }
        }
    }
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

/// An injective, element-respecting, edge-preserving map from the reactant graph
/// into the product graph, or `None` if no such embedding exists (a bond breaks).
/// Backtracking with element/degree/neighbour pruning, in a connected visitation
/// order so each atom (after the first of its fragment) has a mapped neighbour.
fn map_monomorphism(
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
) -> Option<Vec<usize>> {
    let n = z_r.len();
    let order = connected_order(adj_r);
    let mut map = vec![usize::MAX; n];
    let mut used = vec![false; n];
    if mono_backtrack(0, &order, z_r, adj_r, z_p, adj_p, &mut map, &mut used) {
        Some(map)
    } else {
        None
    }
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
) -> bool {
    if pos == order.len() {
        return true;
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
        if mono_backtrack(pos + 1, order, z_r, adj_r, z_p, adj_p, map, used) {
            return true;
        }
        map[i] = usize::MAX;
        used[j] = false;
    }
    false
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
