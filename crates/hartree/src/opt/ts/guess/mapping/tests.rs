//! Tests for reactant→product atom mapping: the monomorphism fast path, the
//! geometry-refined Hungarian fallback, the cost matrix, and the ambiguity diagnostic.
//! Split out of `mapping.rs` to keep it under the module-size limit.

use super::*;

/// Ethane reactant (C0–C1 bonded, each C carrying three H) in one atom order, versus a
/// product of two separated methyls (no C–C bond) in a different order, with positions.
#[allow(clippy::type_complexity)]
fn ethane_vs_two_methyls() -> (
    Vec<u32>,
    Vec<Vec<usize>>,
    Vec<[f64; 3]>,
    Vec<u32>,
    Vec<Vec<usize>>,
    Vec<[f64; 3]>,
) {
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
    // C0 at origin with three H; C1 along +x with three H (a stretched ethane).
    let pos_r = vec![
        [0.0, 0.0, 0.0],
        [2.9, 0.0, 0.0],
        [-0.6, 1.0, 0.0],
        [-0.6, -0.5, 0.9],
        [-0.6, -0.5, -0.9],
        [3.5, 1.0, 0.0],
        [3.5, -0.5, 0.9],
        [3.5, -0.5, -0.9],
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
    // Methyl A near the reactant's C0; methyl B pulled far out along +x (bond broken).
    let pos_p = vec![
        [0.0, 0.0, 0.0],
        [-0.6, 1.0, 0.0],
        [-0.6, -0.5, 0.9],
        [-0.6, -0.5, -0.9],
        [12.0, 0.0, 0.0],
        [12.6, 1.0, 0.0],
        [12.6, -0.5, 0.9],
        [12.6, -0.5, -0.9],
    ];
    (z_r, adj_r, pos_r, z_p, adj_p, pos_p)
}

#[test]
fn map_monomorphism_fails_when_a_bond_breaks() {
    let (z_r, adj_r, _pr, z_p, adj_p, _pp) = ethane_vs_two_methyls();
    // The reactant C–C edge cannot embed into a product with no C–C bond.
    assert!(map_monomorphism(&z_r, &adj_r, &z_p, &adj_p).is_none());
}

#[test]
fn atom_map_falls_back_to_total_injective_hungarian() {
    let (z_r, adj_r, pos_r, z_p, adj_p, pos_p) = ethane_vs_two_methyls();
    let (map, conf) = atom_map(&z_r, &adj_r, &pos_r, &z_p, &adj_p, &pos_p);
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
    assert!(
        (0.0..=1.0).contains(&conf.confidence),
        "confidence {} out of range",
        conf.confidence
    );
}

#[test]
fn map_monomorphism_succeeds_without_bond_breaking() {
    // Reactant = a single methyl, embedded as a subgraph of a product methyl (with one
    // extra spectator atom) preserving every reactant bond. atom_map returns it with
    // full confidence (an exact embedding is topologically determined).
    let z_r = vec![6, 1, 1, 1];
    let adj_r = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
    let pos_r = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
    ];
    let z_p = vec![6, 1, 1, 1, 6];
    let adj_p = vec![vec![1, 2, 3], vec![0], vec![0], vec![0], vec![]];
    let pos_p = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
        [9.0, 0.0, 0.0],
    ];

    let (map, conf) = atom_map(&z_r, &adj_r, &pos_r, &z_p, &adj_p, &pos_p);
    assert_eq!(map.len(), 4);
    assert_eq!(
        conf.confidence, 1.0,
        "an exact embedding is fully confident"
    );
    assert!(conf.ambiguous.is_empty());

    let c_img = map[0];
    for &h in &adj_r[0] {
        assert!(
            adj_p[c_img].contains(&map[h]),
            "reactant bond 0-{h} not preserved under the embedding"
        );
    }
}

#[test]
fn atom_map_geometry_breaks_a_signature_tie() {
    // A bond-breaking case (monomorphism fails) with two connectivity-equivalent
    // hydrogens H1, H2 on C0. The off-axis C3 pins the molecular frame, so the
    // geometry-refined pass must pair each reactant H with the product H at its
    // position — even though the product hydrogens are listed in swapped order.
    let z_r = vec![6, 1, 1, 6];
    let adj_r = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
    let pos_r = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ];
    // Product: C0 keeps H1p, H2p; the C0–C3 bond is broken (C3 isolated). The hydrogens
    // are listed swapped: index 1 sits where reactant H2 is, index 2 where H1 is.
    let z_p = vec![6, 1, 1, 6];
    let adj_p = vec![vec![1, 2], vec![0], vec![0], vec![]];
    let pos_p = vec![
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ];

    assert!(
        map_monomorphism(&z_r, &adj_r, &z_p, &adj_p).is_none(),
        "the broken C–C bond should defeat the monomorphism"
    );
    let (map, _conf) = atom_map(&z_r, &adj_r, &pos_r, &z_p, &adj_p, &pos_p);
    // Geometry pairs reactant H1 ([1,0,0]) with product index 2 ([1,0,0]) and H2 with
    // index 1; the carbons map to themselves.
    assert_eq!(map, vec![0, 2, 1, 3], "geometry did not break the H tie");
}

#[test]
fn build_cost_gates_elements_and_ranks_signature_over_geometry() {
    // Two carbons and structure where one product carbon matches the signature and the
    // other does not; a hydrogen makes the cross-element gate observable.
    let z_r = vec![6, 6];
    let z_p = vec![6, 6];
    let sig_r = vec![vec![6, 100], vec![6, 200]];
    let sig_p = vec![vec![6, 100], vec![6, 999]];
    // Geometry: identical positions so the geometric term is 0 on the diagonal and
    // maximal off it — yet it must never overturn a signature match.
    let pos_r = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    let pos_p = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];

    let cost = build_cost(&z_r, &sig_r, &z_p, &sig_p, Some((&pos_r, &pos_p)));
    // Signature match (row 0 → col 0) costs less than one shell of mismatch (row 0 → col
    // 1), and the geometry term alone can never close a one-shell gap (it is < WEIGHT).
    assert!(
        cost[0][0] < SIGNATURE_WEIGHT,
        "matched signature too costly"
    );
    assert!(
        cost[0][1] >= SIGNATURE_WEIGHT,
        "a signature mismatch must cost at least one full weight"
    );

    // The cross-element gate: a C↔H pairing is forbidden (a large finite penalty).
    let zr2 = vec![6, 1];
    let zp2 = vec![6, 1];
    let sig2 = vec![vec![6], vec![1]];
    let pos2 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let gated = build_cost(&zr2, &sig2, &zp2, &sig2, Some((&pos2, &pos2)));
    assert_eq!(gated[0][1], ELEMENT_PENALTY, "C↔H must be forbidden");
    assert_eq!(gated[1][0], ELEMENT_PENALTY, "H↔C must be forbidden");
}

#[test]
fn geometry_total_cannot_outrank_one_signature_shell() {
    // The Hungarian solver minimizes the SUM of per-atom costs, so the geometry term must
    // be normalized by n: its total over a whole assignment must stay below one signature
    // shell, otherwise geometry could buy back a connectivity-shell mismatch on a large
    // molecule. Take many same-element atoms with large, varied distances; even the
    // upper-bounding "max geometry per row, summed" must stay under SIGNATURE_WEIGHT.
    let n = 20;
    let z: Vec<u32> = vec![1; n];
    let sig: Vec<Vec<u64>> = (0..n).map(|_| vec![1, 1, 1, 1]).collect(); // all identical
    let pos_r: Vec<[f64; 3]> = (0..n).map(|i| [i as f64 * 7.0, 0.0, 0.0]).collect();
    let pos_p: Vec<[f64; 3]> = (0..n).map(|i| [(n - i) as f64 * 7.0, 0.0, 0.0]).collect();

    let cost = build_cost(&z, &sig, &z, &sig, Some((&pos_r, &pos_p)));
    // Signatures are identical, so every cell is geometry-only; the largest geometry an
    // assignment can carry is bounded by the per-row maxima summed.
    let worst_total: f64 = (0..n)
        .map(|i| (0..n).map(|j| cost[i][j]).fold(0.0, f64::max))
        .sum();
    assert!(
        worst_total < SIGNATURE_WEIGHT,
        "total geometry {worst_total} can outrank one signature shell ({SIGNATURE_WEIGHT})"
    );
}

#[test]
fn diagnose_flags_interchangeable_atoms() {
    let z = vec![1, 1];
    let sig = vec![vec![1], vec![1]];
    let map = vec![0, 1];

    // Coincident positions ⇒ the two atoms are genuinely interchangeable: zero confidence
    // and both flagged.
    let coincident = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let amb = diagnose(&z, &sig, &coincident, &z, &sig, &coincident, &map);
    assert_eq!(amb.confidence, 0.0);
    assert_eq!(amb.ambiguous, vec![0, 1]);

    // Well-separated positions ⇒ geometry distinguishes them: full confidence, none
    // flagged.
    let distinct = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
    let clear = diagnose(&z, &sig, &distinct, &z, &sig, &distinct, &map);
    assert_eq!(clear.confidence, 1.0);
    assert!(clear.ambiguous.is_empty());
}
