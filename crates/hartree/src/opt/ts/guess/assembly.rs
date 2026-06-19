//! Reactant-endpoint assembly (rigid fragment alignment + separation) and the
//! forming/breaking-bond reaction-coordinate extraction.

use super::{BondChange, ReactionBond, distance};
use crate::ext::kabsch::optimal_rotation;

fn centroid(points: &[[f64; 3]], members: &[usize]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for &m in members {
        for k in 0..3 {
            c[k] += points[m][k];
        }
    }
    let inv = 1.0 / members.len() as f64;
    [c[0] * inv, c[1] * inv, c[2] * inv]
}

fn matvec3(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// Rigidly superimpose each reactant fragment onto its product image.
pub(super) fn align_fragments(
    pos_r: &[[f64; 3]],
    target: &[[f64; 3]],
    fragment_id: &[usize],
) -> Vec<[f64; 3]> {
    let n = pos_r.len();
    let mut out = pos_r.to_vec();
    let n_frag = fragment_id.iter().copied().max().map_or(0, |m| m + 1);
    for fid in 0..n_frag {
        let members: Vec<usize> = (0..n).filter(|&i| fragment_id[i] == fid).collect();
        if members.is_empty() {
            continue;
        }
        let c_src = centroid(pos_r, &members);
        let c_dst = centroid(target, &members);
        let src: Vec<[f64; 3]> = members
            .iter()
            .map(|&i| {
                [
                    pos_r[i][0] - c_src[0],
                    pos_r[i][1] - c_src[1],
                    pos_r[i][2] - c_src[2],
                ]
            })
            .collect();
        let dst: Vec<[f64; 3]> = members
            .iter()
            .map(|&i| {
                [
                    target[i][0] - c_dst[0],
                    target[i][1] - c_dst[1],
                    target[i][2] - c_dst[2],
                ]
            })
            .collect();
        let rot = optimal_rotation(&src, &dst);
        for (k, &i) in members.iter().enumerate() {
            let rotated = matvec3(&rot, src[k]);
            out[i] = [
                rotated[0] + c_dst[0],
                rotated[1] + c_dst[1],
                rotated[2] + c_dst[2],
            ];
        }
    }
    out
}

/// Push each fragment radially away from the overall centroid, stretching the
/// forming bonds into a reactant-side endpoint.
pub(super) fn separate_fragments(
    pos: &[[f64; 3]],
    fragment_id: &[usize],
    n_frag: usize,
    separation: f64,
) -> Vec<[f64; 3]> {
    let n = pos.len();
    let all: Vec<usize> = (0..n).collect();
    let global = centroid(pos, &all);
    let mut out = pos.to_vec();
    for fid in 0..n_frag {
        let members: Vec<usize> = (0..n).filter(|&i| fragment_id[i] == fid).collect();
        if members.is_empty() {
            continue;
        }
        let c = centroid(pos, &members);
        let mut dir = [c[0] - global[0], c[1] - global[1], c[2] - global[2]];
        let dnorm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if dnorm < 1e-8 {
            // Centroid coincides with the global centroid: pick a deterministic
            // axis so coincident fragments still separate.
            dir = [0.0; 3];
            dir[fid % 3] = 1.0;
        } else {
            for d in &mut dir {
                *d /= dnorm;
            }
        }
        for &i in &members {
            for k in 0..3 {
                out[i][k] += separation * dir[k];
            }
        }
    }
    out
}

/// Forming bonds (product-bonded, not reactant-bonded) and breaking bonds (the
/// reverse), in reactant-order indices.
pub(super) fn reaction_bonds(
    adj_r: &[Vec<usize>],
    adj_p: &[Vec<usize>],
    map: &[usize],
    reactant_endpoint: &[[f64; 3]],
    product: &[[f64; 3]],
) -> Vec<ReactionBond> {
    let n = map.len();
    let mut inv = vec![usize::MAX; n];
    for (r, &p) in map.iter().enumerate() {
        inv[p] = r;
    }
    let bonded_r = |a: usize, b: usize| adj_r[a].contains(&b);

    let mut bonds = Vec::new();
    for p in 0..n {
        for &q in &adj_p[p] {
            if p >= q {
                continue;
            }
            let (a, b) = (inv[p], inv[q]);
            if a == usize::MAX || b == usize::MAX || bonded_r(a, b) {
                continue;
            }
            bonds.push(ReactionBond {
                atoms: (a, b),
                reactant_distance: distance(reactant_endpoint[a], reactant_endpoint[b]),
                product_distance: distance(product[a], product[b]),
                kind: BondChange::Forming,
            });
        }
    }
    for a in 0..n {
        for &b in &adj_r[a] {
            if a >= b {
                continue;
            }
            if adj_p[map[a]].contains(&map[b]) {
                continue;
            }
            bonds.push(ReactionBond {
                atoms: (a, b),
                reactant_distance: distance(reactant_endpoint[a], reactant_endpoint[b]),
                product_distance: distance(product[a], product[b]),
                kind: BondChange::Breaking,
            });
        }
    }
    bonds
}
