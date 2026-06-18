use latx::Cell;
use std::f64::consts::PI;

const CONV: f64 = 5.3;

#[must_use]
pub fn ewald_energy(cell: &Cell, positions: &[[f64; 3]], charges: &[f64]) -> f64 {
    ewald_energy_eta(cell, positions, charges, default_eta(cell))
}

#[must_use]
pub fn ewald_energy_eta(cell: &Cell, positions: &[[f64; 3]], charges: &[f64], eta: f64) -> f64 {
    assert_eq!(
        positions.len(),
        charges.len(),
        "positions and charges must align"
    );
    let omega = cell.volume();
    let q_tot: f64 = charges.iter().sum();
    let q_sq: f64 = charges.iter().map(|q| q * q).sum();

    let r_cut = CONV / eta;
    let max_pair = max_pair_distance(positions);
    let images = cell.lattice_images(r_cut + max_pair);
    let mut e_real = 0.0;
    for (i, &ri) in positions.iter().enumerate() {
        for (j, &rj) in positions.iter().enumerate() {
            let qq = charges[i] * charges[j];
            if qq == 0.0 {
                continue;
            }
            let d0 = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            for &(triple, r) in &images {
                if i == j && triple == [0, 0, 0] {
                    continue;
                }
                let d = [d0[0] + r[0], d0[1] + r[1], d0[2] + r[2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if dist > r_cut || dist < 1e-12 {
                    continue;
                }
                e_real += qq * libm::erfc(eta * dist) / dist;
            }
        }
    }
    e_real *= 0.5;

    let g_cut = 2.0 * eta * CONV;
    let e_recip = reciprocal_sum(cell, positions, charges, eta, g_cut, omega);

    let e_self = -eta / PI.sqrt() * q_sq;
    let e_bg = -PI / (2.0 * omega * eta * eta) * q_tot * q_tot;

    e_real + e_recip + e_self + e_bg
}

fn reciprocal_sum(
    cell: &Cell,
    positions: &[[f64; 3]],
    charges: &[f64],
    eta: f64,
    g_cut: f64,
    omega: f64,
) -> f64 {
    let recip = cell.reciprocal();
    let (a0, a1, a2) = cell.vectors();
    let len = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let bound = |a: [f64; 3]| (g_cut * len(a) / (2.0 * PI)).floor() as i32 + 1;
    let (m0, m1, m2) = (bound(a0), bound(a1), bound(a2));

    let g2_cut = g_cut * g_cut;
    let inv_4eta2 = 1.0 / (4.0 * eta * eta);
    let mut sum = 0.0;
    for h in -m0..=m0 {
        for k in -m1..=m1 {
            for l in -m2..=m2 {
                if h == 0 && k == 0 && l == 0 {
                    continue;
                }
                let g = recip.g_vector([h, k, l]);
                let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                if g2 > g2_cut {
                    continue;
                }
                let mut sc = 0.0;
                let mut ss = 0.0;
                for (&q, &r) in charges.iter().zip(positions) {
                    let gr = g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
                    sc += q * gr.cos();
                    ss += q * gr.sin();
                }
                let s2 = sc * sc + ss * ss;
                sum += (-g2 * inv_4eta2).exp() / g2 * s2;
            }
        }
    }
    2.0 * PI / omega * sum
}

fn default_eta(cell: &Cell) -> f64 {
    let l = cell.volume().cbrt();
    3.2 / l
}

fn max_pair_distance(positions: &[[f64; 3]]) -> f64 {
    let mut m = 0.0_f64;
    for (i, a) in positions.iter().enumerate() {
        for b in &positions[i + 1..] {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            m = m.max(d);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nacl_madelung_constant() {
        let a = 1.0;
        let cell = Cell::cubic(a).unwrap();
        let na = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        let cl = [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ];
        let mut pos = Vec::new();
        let mut q = Vec::new();
        for r in na {
            pos.push([r[0] * a, r[1] * a, r[2] * a]);
            q.push(1.0);
        }
        for r in cl {
            pos.push([r[0] * a, r[1] * a, r[2] * a]);
            q.push(-1.0);
        }
        let e = ewald_energy(&cell, &pos, &q);
        let madelung = 1.747_564_594_633_2;
        let expect = -8.0 * madelung / a; // 4 pairs × (−α/(a/2))
        assert!((e - expect).abs() < 1e-5, "Ewald {e} vs −8α/a {expect}");
    }

    #[test]
    fn independent_of_eta() {
        let cell = Cell::cubic(1.0).unwrap();
        let pos = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
        let q = [1.0, -1.0];
        let e1 = ewald_energy_eta(&cell, &pos, &q, 2.5);
        let e2 = ewald_energy_eta(&cell, &pos, &q, 4.5);
        let e3 = ewald_energy_eta(&cell, &pos, &q, 6.0);
        assert!(
            (e1 - e2).abs() < 1e-8 && (e2 - e3).abs() < 1e-8,
            "η dependence: {e1} {e2} {e3}"
        );
    }

    #[test]
    fn point_charge_in_background() {
        let l = 2.0;
        let cell = Cell::cubic(l).unwrap();
        let e = ewald_energy(&cell, &[[0.0, 0.0, 0.0]], &[1.0]);
        let xi = 2.837_297;
        let expect = -xi / (2.0 * l);
        assert!((e - expect).abs() < 1e-5, "jellium {e} vs −ξ/2L {expect}");
    }
}
