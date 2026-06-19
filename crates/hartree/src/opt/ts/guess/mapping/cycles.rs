//! Cycle detection on the "free reassignment" graph used by the mapping diagnostic. A
//! directed cycle of length ≥ 2 marks a set of atoms whose product partners can be
//! cyclically permuted at no cost — an ambiguous correspondence (a pairwise free swap is
//! the length-2 case; a 3-fold symmetric set rotated among itself is a length-3 case).

/// Mark every vertex lying on a directed cycle of length ≥ 2 (equivalently, every vertex in
/// a strongly connected component of size > 1) of `adj`. Tarjan's SCC algorithm, iterative
/// to avoid deep recursion on large graphs.
pub(super) fn atoms_on_cycles(adj: &[Vec<usize>]) -> Vec<bool> {
    let n = adj.len();
    let mut index = vec![usize::MAX; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut on_cycle = vec![false; n];
    let mut next_index = 0usize;

    // (vertex, position in its adjacency list) frames of the explicit DFS stack.
    let mut work: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if index[start] != usize::MAX {
            continue;
        }
        work.push((start, 0));
        while let Some(&(v, ei)) = work.last() {
            if ei == 0 {
                index[v] = next_index;
                lowlink[v] = next_index;
                next_index += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if ei < adj[v].len() {
                work.last_mut().unwrap().1 += 1;
                let w = adj[v][ei];
                if index[w] == usize::MAX {
                    work.push((w, 0));
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(index[w]);
                }
            } else {
                // Done with v: if it roots an SCC, pop the component.
                if lowlink[v] == index[v] {
                    let mut component = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    if component.len() > 1 {
                        for &w in &component {
                            on_cycle[w] = true;
                        }
                    }
                }
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    lowlink[parent] = lowlink[parent].min(lowlink[v]);
                }
            }
        }
    }
    on_cycle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_edges_no_cycles() {
        let adj = vec![vec![], vec![], vec![]];
        assert_eq!(atoms_on_cycles(&adj), vec![false, false, false]);
    }

    #[test]
    fn pairwise_swap_is_a_two_cycle() {
        // 0 ↔ 1 form a 2-cycle; 2 is isolated.
        let adj = vec![vec![1], vec![0], vec![]];
        assert_eq!(atoms_on_cycles(&adj), vec![true, true, false]);
    }

    #[test]
    fn three_fold_ring_is_a_cycle() {
        // 0 → 1 → 2 → 0: a length-3 cycle a pairwise check would miss.
        let adj = vec![vec![1], vec![2], vec![0]];
        assert_eq!(atoms_on_cycles(&adj), vec![true, true, true]);
    }

    #[test]
    fn self_loops_alone_do_not_count() {
        // A self-loop is a length-1 "cycle"; it is not an interchange of two atoms.
        let adj = vec![vec![0], vec![]];
        assert_eq!(atoms_on_cycles(&adj), vec![false, false]);
    }

    #[test]
    fn separates_a_cycle_from_a_tail() {
        // 0 → 1 → 2 → 1: 1 and 2 form a 2-cycle; 0 only feeds into it.
        let adj = vec![vec![1], vec![2], vec![1]];
        assert_eq!(atoms_on_cycles(&adj), vec![false, true, true]);
    }
}
