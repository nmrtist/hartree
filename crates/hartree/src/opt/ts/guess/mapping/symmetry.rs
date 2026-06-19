//! Higher-order (orbit) ambiguity detection for a reactant→product atom map. Where the
//! fixed-alignment free-reassignment graph (see [`super::cycles`]) catches near-coincident
//! pairs, this catches symmetric orbits of three or more equivalent atoms: a pairwise swap
//! of such an orbit is an improper move a rigid fit cannot match, but a full cyclic rotation
//! is a proper symmetry the fit recovers, so the orbit's labeling is geometrically
//! undetermined.

/// Largest squared group spread treated as "clustered" when deciding whether an equivalent
/// set could be a symmetric orbit; a more separated set must additionally lie on a common
/// sphere (equal radii) to qualify.
const GROUP_AMBIGUITY_TOL: f64 = 1.0e-4;

/// Largest equivalent-atom group whose cyclic reassignments are searched. The search is
/// over the group's cyclic rotations (linear in the group size), so this only bounds
/// pathologically symmetric cases.
const GROUP_AMBIGUITY_MAX: usize = 8;

/// Flag atoms in any connectivity-equivalent, spatially clustered (or equal-radius) group of
/// three or more for which a non-trivial cyclic rotation of the group's product partners —
/// scored by `residual`, a rigid re-alignment of the *whole* candidate map — fits the
/// reactant as well as the current map. `residual(candidate)` returns the mean squared
/// per-atom discrepancy after aligning the reactant onto the product permuted by
/// `candidate`. Atoms outside the rotated group keep their partners, so the rest of the
/// molecule pins the frame: only a genuine orbit symmetry survives.
pub(super) fn flag_symmetric_groups(
    z_r: &[u32],
    sig_r: &[Vec<u64>],
    pos_r: &[[f64; 3]],
    map: &[usize],
    residual: impl Fn(&[usize]) -> f64,
    ambiguous: &mut [bool],
) {
    let n = map.len();
    let baseline = residual(map);

    let mut visited = vec![false; n];
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let group: Vec<usize> = (start..n)
            .filter(|&j| z_r[j] == z_r[start] && sig_r[j] == sig_r[start])
            .collect();
        for &g in &group {
            visited[g] = true;
        }
        // Pairwise (2-cycle) ambiguity is the province of the fixed-alignment graph; this
        // higher-order test handles orbits of three or more.
        if group.len() < 3 || group.len() > GROUP_AMBIGUITY_MAX {
            continue;
        }
        // Only a clustered or equal-radius (symmetric-orbit) group can be rotated onto
        // itself; a generic separated set is geometrically pinned even if equivalent.
        let radius2 = group_radius2(pos_r, &group);
        if radius2 > GROUP_AMBIGUITY_TOL && !group_is_symmetric_orbit(pos_r, &group) {
            continue;
        }
        // A non-trivial cyclic rotation of this group's partners that re-aligns as well as
        // the current map means the orbit's labeling is ambiguous.
        let m = group.len();
        for shift in 1..m {
            let mut candidate = map.to_vec();
            for (idx, &atom) in group.iter().enumerate() {
                candidate[atom] = map[group[(idx + shift) % m]];
            }
            if residual(&candidate) - baseline < GROUP_AMBIGUITY_TOL {
                for &g in &group {
                    ambiguous[g] = true;
                }
                break;
            }
        }
    }
}

/// Squared distance of the farthest group member from the group centroid — its spatial
/// spread.
fn group_radius2(pos: &[[f64; 3]], group: &[usize]) -> f64 {
    let pts: Vec<[f64; 3]> = group.iter().map(|&g| pos[g]).collect();
    let c = centroid(&pts);
    pts.iter().map(|&p| dist2(p, c)).fold(0.0, f64::max)
}

/// Whether the group's atoms lie on a common sphere about their centroid (equal radii) —
/// the geometric signature of a symmetric orbit (a regular polygon, an equilateral set),
/// for which a cyclic relabeling is a rigid symmetry rather than a distinct arrangement.
fn group_is_symmetric_orbit(pos: &[[f64; 3]], group: &[usize]) -> bool {
    let pts: Vec<[f64; 3]> = group.iter().map(|&g| pos[g]).collect();
    let c = centroid(&pts);
    let radii: Vec<f64> = pts.iter().map(|&p| dist2(p, c)).collect();
    let r0 = radii[0];
    radii.iter().all(|&r| (r - r0).abs() < 1.0e-3 * (r0 + 1.0))
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
