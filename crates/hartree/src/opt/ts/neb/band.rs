//! The chain-of-states force math: the improved (upwind) tangent, the nudged
//! perpendicular + spring force, the climbing-image inverted-parallel force, and the
//! rigid Kabsch alignment of one endpoint onto the other. Pure numerics — no surface,
//! no optimizer state — so each piece is unit-tested in isolation below.
//!
//! All vectors are flat `3·natoms` Cartesian (atomic units). The tangent and force
//! conventions follow Henkelman, Uberuaga & Jónsson, J. Chem. Phys. 113, 9901 (2000):
//! the tangent points toward the higher-energy neighbour, and the climbing image
//! inverts the true force along that tangent so it ascends to the saddle.

use crate::ext::kabsch::optimal_rotation;
use crate::opt::ts::numerics::{dot, flatten, norm};

/// The improved/upwind tangent at interior image `curr`, given its two neighbours
/// and the three energies. Returns a **unit** flat vector.
///
/// On a monotonic stretch of the band the tangent is simply the segment toward the
/// higher-energy neighbour (which keeps it from kinking, unlike the normalized
/// bisector of the two segments). At a local energy extremum along the band it is the
/// energy-weighted blend of both segments.
pub(super) fn improved_tangent(
    prev: &[[f64; 3]],
    curr: &[[f64; 3]],
    next: &[[f64; 3]],
    e_prev: f64,
    e_curr: f64,
    e_next: f64,
) -> Vec<f64> {
    let tau_plus = sub(next, curr); // R_{i+1} − R_i
    let tau_minus = sub(curr, prev); // R_i − R_{i−1}

    let mut tau = if e_next > e_curr && e_curr > e_prev {
        // Uphill toward i+1.
        tau_plus
    } else if e_next < e_curr && e_curr < e_prev {
        // Uphill toward i−1.
        tau_minus
    } else {
        // Local extremum: blend, weighting the segment toward the higher neighbour by
        // the larger energy difference.
        let d1 = (e_next - e_curr).abs();
        let d2 = (e_prev - e_curr).abs();
        let de_max = d1.max(d2);
        let de_min = d1.min(d2);
        if e_next > e_prev {
            combine(&tau_plus, de_max, &tau_minus, de_min)
        } else {
            combine(&tau_plus, de_min, &tau_minus, de_max)
        }
    };

    let n = norm(&tau);
    if n > 1e-12 {
        for t in &mut tau {
            *t /= n;
        }
    } else {
        // Coincident neighbours leave the blend degenerate; fall back to the forward
        // segment (and, failing that, a zero tangent the caller treats as no nudge).
        tau = sub(next, curr);
        let n2 = norm(&tau);
        if n2 > 1e-12 {
            for t in &mut tau {
                *t /= n2;
            }
        }
    }
    tau
}

/// Per-interior-image NEB force, concatenated into one `n_images · 3natoms` vector,
/// plus its max-component and RMS norms (the convergence metric).
///
/// `images`, `energies`, and `grads` span the **full** band (endpoints at index 0 and
/// `len−1`); only interior images `1..=n_images` are forced, and only `grads[i]` for
/// interior `i` is read. For a regular image the force is the perpendicular component
/// of the true force, `−(g − (g·τ)τ)`, plus the nudged spring `k(|R₊|−|R₋|)τ` along
/// the tangent. The climbing image (`ci_index`, a full-band index) instead drops its
/// springs and inverts its parallel true force, `−g + 2(g·τ)τ`, so it ascends the
/// band to the saddle.
pub(super) fn neb_forces(
    images: &[Vec<[f64; 3]>],
    energies: &[f64],
    grads: &[Vec<[f64; 3]>],
    spring_k: f64,
    ci_index: Option<usize>,
) -> (Vec<f64>, f64, f64) {
    let n_total = images.len();
    let n_images = n_total - 2;
    let ndof = images[0].len() * 3;
    let mut force = vec![0.0f64; n_images * ndof];
    let (mut max_c, mut sumsq, mut count) = (0.0f64, 0.0f64, 0usize);

    for i in 1..=n_images {
        let tau = improved_tangent(
            &images[i - 1],
            &images[i],
            &images[i + 1],
            energies[i - 1],
            energies[i],
            energies[i + 1],
        );
        let g = flatten(&grads[i]);
        let g_par = dot(&g, &tau);

        let f_i: Vec<f64> = if ci_index == Some(i) {
            // Climbing image: invert the parallel true force, no springs.
            (0..ndof).map(|d| -g[d] + 2.0 * g_par * tau[d]).collect()
        } else {
            // Perpendicular true force −(g − (g·τ)τ) = −g + (g·τ)τ, plus the nudged
            // (parallel-only) spring force.
            let d_plus = dist(&images[i + 1], &images[i]);
            let d_minus = dist(&images[i], &images[i - 1]);
            let spring = spring_k * (d_plus - d_minus);
            (0..ndof)
                .map(|d| -g[d] + g_par * tau[d] + spring * tau[d])
                .collect()
        };

        let base = (i - 1) * ndof;
        for d in 0..ndof {
            force[base + d] = f_i[d];
            max_c = max_c.max(f_i[d].abs());
            sumsq += f_i[d] * f_i[d];
            count += 1;
        }
    }

    let rms = if count > 0 {
        (sumsq / count as f64).sqrt()
    } else {
        0.0
    };
    (force, max_c, rms)
}

/// Rigidly superimpose `mobile` onto `reference` (Kabsch): translate to a common
/// centroid, apply the optimal rotation, and translate onto the reference centroid.
/// Preserves `mobile`'s internal geometry exactly (only the rigid frame changes).
pub(super) fn kabsch_align(mobile: &[[f64; 3]], reference: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let c_src = centroid(mobile);
    let c_dst = centroid(reference);
    let src: Vec<[f64; 3]> = mobile.iter().map(|p| sub3(*p, c_src)).collect();
    let dst: Vec<[f64; 3]> = reference.iter().map(|p| sub3(*p, c_dst)).collect();
    let rot = optimal_rotation(&src, &dst);
    src.iter()
        .map(|p| {
            let r = matvec3(&rot, *p);
            [r[0] + c_dst[0], r[1] + c_dst[1], r[2] + c_dst[2]]
        })
        .collect()
}

/// Flat vector → per-atom `[f64; 3]` (used to surface the peak tangent).
pub(super) fn reshape(flat: &[f64]) -> Vec<[f64; 3]> {
    (0..flat.len() / 3)
        .map(|a| [flat[3 * a], flat[3 * a + 1], flat[3 * a + 2]])
        .collect()
}

fn sub(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() * 3);
    for (p, q) in a.iter().zip(b) {
        for k in 0..3 {
            out.push(p[k] - q[k]);
        }
    }
    out
}

fn combine(u: &[f64], cu: f64, v: &[f64], cv: f64) -> Vec<f64> {
    u.iter().zip(v).map(|(a, b)| cu * a + cv * b).collect()
}

fn dist(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let mut s = 0.0;
    for (p, q) in a.iter().zip(b) {
        for k in 0..3 {
            let d = p[k] - q[k];
            s += d * d;
        }
    }
    s.sqrt()
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for p in points {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    let inv = 1.0 / points.len() as f64;
    [c[0] * inv, c[1] * inv, c[2] * inv]
}

fn matvec3(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // A single atom per "image" keeps the flat vectors one 3-vector long, so the
    // tangent/force algebra is easy to read; the math is identical at any atom count.
    fn img(p: [f64; 3]) -> Vec<[f64; 3]> {
        vec![p]
    }

    #[test]
    fn tangent_points_uphill_on_a_monotonic_band() {
        // Energies increase left→right, so the tangent is the forward segment (+x).
        let prev = img([0.0, 0.0, 0.0]);
        let curr = img([1.0, 0.0, 0.0]);
        let next = img([2.0, 0.0, 0.0]);
        let tau = improved_tangent(&prev, &curr, &next, 0.0, 1.0, 2.0);
        assert!((tau[0] - 1.0).abs() < 1e-12, "tau={tau:?}");
        // Falling energies → tangent is the backward segment, but still +x (points
        // toward the higher-energy i−1 neighbour, i.e. R_i − R_{i−1}).
        let tau_rev = improved_tangent(&prev, &curr, &next, 2.0, 1.0, 0.0);
        assert!((tau_rev[0] - 1.0).abs() < 1e-12, "tau_rev={tau_rev:?}");
        assert!(
            (norm(&tau) - 1.0).abs() < 1e-12,
            "tangent must be unit length"
        );
    }

    #[test]
    fn tangent_blends_at_a_band_maximum() {
        // curr is the band maximum (1.0 > both neighbours). With e_next > e_prev the
        // forward segment carries the larger weight; the blend stays along x here.
        let prev = img([-1.0, 0.0, 0.0]);
        let curr = img([0.0, 0.0, 0.0]);
        let next = img([1.0, 0.0, 0.0]);
        let tau = improved_tangent(&prev, &curr, &next, 0.2, 1.0, 0.5);
        assert!((norm(&tau) - 1.0).abs() < 1e-12);
        assert!(tau[0] > 0.99, "blend should stay along +x, tau={tau:?}");
    }

    #[test]
    fn climbing_image_inverts_the_parallel_force() {
        // Band along x; the middle image is the climbing image. Gradient points +x
        // (so the plain force −g points −x). The CI force must instead point +x
        // (uphill along the tangent): F = −g + 2(g·τ)τ with g·τ = +1 ⇒ F = +g.
        let images = vec![
            img([0.0, 0.0, 0.0]),
            img([1.0, 0.0, 0.0]),
            img([2.0, 0.0, 0.0]),
        ];
        let energies = [0.0, 1.0, 0.5];
        let grads = vec![img([0.0; 3]), img([1.0, 0.0, 0.0]), img([0.0; 3])];
        let (force, _max, _rms) = neb_forces(&images, &energies, &grads, 0.1, Some(1));
        // Interior slot 0 is the lone climbing image.
        assert!(
            (force[0] - 1.0).abs() < 1e-12,
            "CI force should be +x, got {force:?}"
        );
        assert!(force[1].abs() < 1e-12 && force[2].abs() < 1e-12);
    }

    #[test]
    fn regular_image_keeps_only_the_perpendicular_true_force() {
        // Tangent is +x (monotonic band). The gradient has an along-tangent part
        // (+x) and a perpendicular part (+y). The non-climbing force must drop the
        // along-tangent true force and keep −(perp) = −y (plus an equal-spacing
        // spring of zero here, since the band is evenly spaced).
        let images = vec![
            img([0.0, 0.0, 0.0]),
            img([1.0, 0.0, 0.0]),
            img([2.0, 0.0, 0.0]),
        ];
        let energies = [0.0, 1.0, 2.0];
        let grads = vec![img([0.0; 3]), img([0.7, 0.4, 0.0]), img([0.0; 3])];
        let (force, _max, _rms) = neb_forces(&images, &energies, &grads, 0.5, None);
        assert!(
            force[0].abs() < 1e-12,
            "along-tangent true force not removed: {force:?}"
        );
        assert!(
            (force[1] + 0.4).abs() < 1e-12,
            "perpendicular force wrong: {force:?}"
        );
    }

    #[test]
    fn spring_pushes_toward_the_more_distant_neighbour() {
        // Uneven spacing: i−1 at x=0, i at x=1, i+1 at x=3. |R₊|=2 > |R₋|=1, so the
        // spring k(|R₊|−|R₋|)τ is +x (toward the farther neighbour), restoring spacing.
        // Zero gradient isolates the spring term.
        let images = vec![
            img([0.0, 0.0, 0.0]),
            img([1.0, 0.0, 0.0]),
            img([3.0, 0.0, 0.0]),
        ];
        let energies = [0.0, 1.0, 2.0]; // monotonic ⇒ τ = +x
        let grads = vec![img([0.0; 3]), img([0.0; 3]), img([0.0; 3])];
        let k = 0.5;
        let (force, _max, _rms) = neb_forces(&images, &energies, &grads, k, None);
        assert!(
            (force[0] - k * (2.0 - 1.0)).abs() < 1e-12,
            "spring wrong: {force:?}"
        );
    }

    #[test]
    fn kabsch_align_recovers_a_rotated_translated_copy() {
        // A rigid 90°-about-z rotation + translation of a reference; aligning back
        // must reproduce the reference (zero residual) and preserve internal distances.
        let reference = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let mobile = [[5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [3.0, 1.0, 0.0]]; // R_z(90°)+t
        let aligned = kabsch_align(&mobile, &reference);
        for (a, r) in aligned.iter().zip(&reference) {
            for k in 0..3 {
                assert!((a[k] - r[k]).abs() < 1e-9, "aligned {a:?} vs ref {r:?}");
            }
        }
        // Internal geometry preserved: the mobile's 0–1 distance survives alignment.
        let d_mobile = dist(&[mobile[0]], &[mobile[1]]);
        let d_aligned = dist(&[aligned[0]], &[aligned[1]]);
        assert!((d_mobile - d_aligned).abs() < 1e-9);
    }
}
