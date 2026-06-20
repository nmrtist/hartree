//! Dense O(n³) Hungarian (Kuhn–Munkres) solver for the rectangular-free square linear
//! assignment problem: given an `n×n` cost matrix, find the permutation `σ` minimizing
//! `Σ_i cost[i][σ(i)]`.
//!
//! This is the standard shortest-augmenting-path formulation with dual potentials (the
//! "e-maxx" variant), specialized to `f64` costs. Atom mapping uses it over a
//! signature-plus-geometry cost with same-element assignments made cheap and
//! cross-element assignments made prohibitively (but finitely) expensive, so the optimum
//! is the best element-respecting correspondence.
//!
//! Costs must be finite. Forbidden pairings are encoded as a large finite sentinel by the
//! caller, never `f64::INFINITY` — the algorithm does arithmetic on the costs, and an
//! infinity would poison the dual updates. A feasible (finite-optimal) perfect matching
//! is assumed to exist; when the caller guarantees a same-element matching is possible
//! (equal element multisets) the optimum never selects a sentinel pairing.

/// Solve the square assignment problem for `cost` (minimization). Returns `assign` with
/// `assign[i]` the column matched to row `i`; the result is a permutation of `0..n`.
///
/// `cost` must be square and every entry finite. An empty matrix yields an empty
/// assignment.
pub(super) fn solve(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    debug_assert!(cost.iter().all(|row| row.len() == n), "cost must be square");

    // 1-indexed working arrays (index 0 is the sentinel/staging slot).
    let mut u = vec![0.0f64; n + 1]; // row potentials
    let mut v = vec![0.0f64; n + 1]; // column potentials
    let mut p = vec![0usize; n + 1]; // p[j] = row currently matched to column j (0 = free)
    let mut way = vec![0usize; n + 1]; // back-pointers along the augmenting path

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize; // current "free" column staging the new row
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            // Tighten potentials by `delta` along the visited set.
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Augment along the back-pointers.
        while j0 != 0 {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }

    let mut assign = vec![0usize; n];
    for j in 1..=n {
        // p[j] is the row matched to column j; an unmatched column (p[j] == 0) cannot
        // occur for a feasible square instance, but guard against it defensively.
        if p[j] != 0 {
            assign[p[j] - 1] = j - 1;
        }
    }
    assign
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force minimum-cost assignment over all permutations, for cross-checking on
    /// small instances.
    fn brute_force(cost: &[Vec<f64>]) -> f64 {
        let n = cost.len();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut best = f64::INFINITY;
        permute(&mut perm, 0, &mut |p| {
            let total: f64 = (0..n).map(|i| cost[i][p[i]]).sum();
            best = best.min(total);
        });
        best
    }

    fn permute(perm: &mut [usize], k: usize, f: &mut impl FnMut(&[usize])) {
        if k == perm.len() {
            f(perm);
            return;
        }
        for i in k..perm.len() {
            perm.swap(k, i);
            permute(perm, k + 1, f);
            perm.swap(k, i);
        }
    }

    fn cost_of(cost: &[Vec<f64>], assign: &[usize]) -> f64 {
        (0..assign.len()).map(|i| cost[i][assign[i]]).sum()
    }

    fn is_permutation(assign: &[usize]) -> bool {
        let mut seen = vec![false; assign.len()];
        for &j in assign {
            if j >= assign.len() || seen[j] {
                return false;
            }
            seen[j] = true;
        }
        true
    }

    #[test]
    fn empty_matrix() {
        assert!(solve(&[]).is_empty());
    }

    #[test]
    fn single_cell() {
        assert_eq!(solve(&[vec![3.0]]), vec![0]);
    }

    #[test]
    fn prefers_the_cheap_diagonal() {
        // Identity is uniquely optimal: the diagonal is far cheaper.
        let cost = vec![
            vec![0.0, 9.0, 9.0],
            vec![9.0, 0.0, 9.0],
            vec![9.0, 9.0, 0.0],
        ];
        assert_eq!(solve(&cost), vec![0, 1, 2]);
    }

    #[test]
    fn finds_a_nontrivial_optimum() {
        // The optimal assignment is the anti-diagonal (0->2, 1->1, 2->0), total 1+2+3=6,
        // beating the diagonal (7+6+9).
        let cost = vec![
            vec![7.0, 5.0, 1.0],
            vec![4.0, 2.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        let assign = solve(&cost);
        assert!(is_permutation(&assign));
        assert!((cost_of(&cost, &assign) - brute_force(&cost)).abs() < 1e-9);
    }

    #[test]
    fn avoids_forbidden_cells_when_feasible() {
        // A large finite sentinel on the diagonal forces the off-diagonal matching.
        let big = 1e9;
        let cost = vec![
            vec![big, 1.0, 2.0],
            vec![2.0, big, 1.0],
            vec![1.0, 2.0, big],
        ];
        let assign = solve(&cost);
        assert!(is_permutation(&assign));
        // No matched cell is a sentinel.
        for (i, &j) in assign.iter().enumerate() {
            assert!(cost[i][j] < big, "row {i} took a forbidden cell");
        }
        assert!((cost_of(&cost, &assign) - brute_force(&cost)).abs() < 1e-6);
    }

    #[test]
    fn matches_brute_force_on_assorted_matrices() {
        // A handful of deterministic non-symmetric matrices; the optimum total must equal
        // the brute-force optimum and the assignment must be a permutation.
        let matrices = [
            vec![
                vec![4.0, 1.0, 3.0],
                vec![2.0, 0.0, 5.0],
                vec![3.0, 2.0, 2.0],
            ],
            vec![
                vec![10.0, 19.0, 8.0, 15.0],
                vec![10.0, 18.0, 7.0, 17.0],
                vec![13.0, 16.0, 9.0, 14.0],
                vec![12.0, 19.0, 8.0, 18.0],
            ],
            vec![
                vec![0.5, 0.5, 0.5, 0.5],
                vec![0.1, 0.9, 0.2, 0.8],
                vec![0.7, 0.3, 0.6, 0.4],
                vec![0.2, 0.2, 0.9, 0.1],
            ],
        ];
        for cost in &matrices {
            let assign = solve(cost);
            assert!(is_permutation(&assign), "not a permutation: {assign:?}");
            assert!(
                (cost_of(cost, &assign) - brute_force(cost)).abs() < 1e-9,
                "suboptimal on {cost:?}: got {}, optimal {}",
                cost_of(cost, &assign),
                brute_force(cost)
            );
        }
    }
}
