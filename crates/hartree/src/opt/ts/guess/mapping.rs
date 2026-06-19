//! Reactant→product atom mapping. Tries an exact subgraph monomorphism (every
//! reactant bond preserved — correct for additions/cycloadditions, which break no
//! bonds), then falls back to a connectivity-signature heuristic.

use std::collections::VecDeque;

const SIGNATURE_SHELLS: usize = 3;

pub(super) fn atom_map(
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
) -> Vec<usize> {
    if let Some(m) = map_monomorphism(z_r, adj_r, z_p, adj_p) {
        return m;
    }
    atom_map_heuristic(z_r, adj_r, z_p, adj_p)
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

/// Fallback for bond-breaking reactions: match heavy atoms by connectivity
/// signature, then attach each hydrogen to a product H on its mapped heavy neighbour.
fn atom_map_heuristic(
    z_r: &[u32],
    adj_r: &[Vec<usize>],
    z_p: &[u32],
    adj_p: &[Vec<usize>],
) -> Vec<usize> {
    let n = z_r.len();
    let sig_r = signatures(adj_r, z_r);
    let sig_p = signatures(adj_p, z_p);

    let mut map = vec![usize::MAX; n];
    let mut used = vec![false; n];

    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
    for i in 0..n {
        if z_r[i] == 1 {
            continue;
        }
        for j in 0..n {
            if z_p[j] == z_r[i] {
                pairs.push((signature_cost(&sig_r[i], &sig_p[j]), i, j));
            }
        }
    }
    pairs.sort_by_key(|&(c, _, _)| c);
    for (_, i, j) in pairs {
        if map[i] == usize::MAX && !used[j] {
            map[i] = j;
            used[j] = true;
        }
    }

    for hp_center in 0..n {
        if z_r[hp_center] == 1 {
            continue;
        }
        let img = map[hp_center];
        if img == usize::MAX {
            continue;
        }
        let r_hydrogens: Vec<usize> = adj_r[hp_center]
            .iter()
            .copied()
            .filter(|&k| z_r[k] == 1 && map[k] == usize::MAX)
            .collect();
        let mut p_hydrogens: Vec<usize> = adj_p[img]
            .iter()
            .copied()
            .filter(|&k| z_p[k] == 1 && !used[k])
            .collect();
        for r_h in r_hydrogens {
            if let Some(p_h) = p_hydrogens.pop() {
                map[r_h] = p_h;
                used[p_h] = true;
            }
        }
    }

    for i in 0..n {
        if map[i] != usize::MAX {
            continue;
        }
        let best = (0..n)
            .filter(|&j| !used[j] && z_p[j] == z_r[i])
            .min_by_key(|&j| signature_cost(&sig_r[i], &sig_p[j]));
        if let Some(j) = best {
            map[i] = j;
            used[j] = true;
        }
    }
    map
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
mod tests {
    use super::*;

    /// Ethane reactant (C0–C1 bonded, each C carrying three H) in one atom order,
    /// versus a product of two separated methyls (no C–C bond) in a different order.
    // Returns the two `(z, adj)` graphs; a named type would obscure more than it clarifies.
    #[allow(clippy::type_complexity)]
    fn ethane_vs_two_methyls() -> (Vec<u32>, Vec<Vec<usize>>, Vec<u32>, Vec<Vec<usize>>) {
        let z_r = vec![6, 6, 1, 1, 1, 1, 1, 1];
        let adj_r = vec![
            vec![1, 2, 3, 4],
            vec![0, 5, 6, 7],
            vec![0],
            vec![0],
            vec![0],
            vec![1],
            vec![1],
            vec![1],
        ];
        let z_p = vec![6, 1, 1, 1, 6, 1, 1, 1];
        let adj_p = vec![
            vec![1, 2, 3],
            vec![0],
            vec![0],
            vec![0],
            vec![5, 6, 7],
            vec![4],
            vec![4],
            vec![4],
        ];
        (z_r, adj_r, z_p, adj_p)
    }

    #[test]
    fn map_monomorphism_fails_when_a_bond_breaks() {
        let (z_r, adj_r, z_p, adj_p) = ethane_vs_two_methyls();
        // The reactant C–C edge cannot embed into a product with no C–C bond.
        assert!(map_monomorphism(&z_r, &adj_r, &z_p, &adj_p).is_none());
    }

    #[test]
    fn atom_map_falls_back_to_total_injective_heuristic() {
        let (z_r, adj_r, z_p, adj_p) = ethane_vs_two_methyls();
        let map = atom_map(&z_r, &adj_r, &z_p, &adj_p);
        assert_eq!(map.len(), 8);

        let mut seen = [false; 8];
        for (r, &p) in map.iter().enumerate() {
            // TOTAL: every reactant atom got a real product slot.
            assert!(p < 8, "reactant atom {r} left unassigned (map[{r}]={p})");
            // INJECTIVE: no product atom is reused.
            assert!(!seen[p], "product atom {p} mapped from two reactant atoms");
            seen[p] = true;
            // element-respecting.
            assert_eq!(z_r[r], z_p[p], "element mismatch at reactant atom {r}");
        }
    }

    #[test]
    fn map_monomorphism_succeeds_without_bond_breaking() {
        // Reactant = a single methyl, embedded as a subgraph of a product methyl
        // (with one extra spectator atom) preserving every reactant bond.
        let z_r = vec![6, 1, 1, 1];
        let adj_r = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
        // Product: methyl on atoms 0..4 plus a detached spectator C at index 4.
        let z_p = vec![6, 1, 1, 1, 6];
        let adj_p = vec![vec![1, 2, 3], vec![0], vec![0], vec![0], vec![]];

        let map = map_monomorphism(&z_r, &adj_r, &z_p, &adj_p).expect("subgraph embeds");
        assert_eq!(map.len(), 4);

        let mut seen = [false; 5];
        for (r, &p) in map.iter().enumerate() {
            assert!(p < 5, "reactant atom {r} unassigned");
            assert!(!seen[p], "product atom {p} mapped twice");
            seen[p] = true;
            assert_eq!(z_r[r], z_p[p], "element mismatch at reactant atom {r}");
        }
        // Edge preservation: the reactant C maps to a product atom whose neighbours
        // include all three mapped H images.
        let c_img = map[0];
        for &h in &adj_r[0] {
            assert!(
                adj_p[c_img].contains(&map[h]),
                "reactant bond 0-{h} not preserved under the embedding"
            );
        }
    }
}
