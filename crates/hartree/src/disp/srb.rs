use crate::core::Molecule;

const THR_R: f64 = 30.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrbParams {
    pub gamma: f64,
    pub s: f64,
    pub label: &'static str,
}

impl SrbParams {
    pub const B97_3C: SrbParams = SrbParams {
        gamma: 10.0,
        s: 0.08,
        label: "b97-3c",
    };
}

fn atom_numbers(mol: &Molecule) -> Vec<usize> {
    mol.atoms
        .iter()
        .map(|a| {
            let z = a.element.z() as usize;
            assert!(
                (1..=18).contains(&z),
                "SRB supports H-Ar only (got Z = {z})"
            );
            z
        })
        .collect()
}

pub fn srb_energy(mol: &Molecule, params: &SrbParams) -> f64 {
    if let Some((real, _)) = crate::disp::without_ghosts(mol) {
        return srb_energy(&real, params);
    }
    srb_impl(mol, params, false).0
}

pub fn srb_energy_gradient(mol: &Molecule, params: &SrbParams) -> (f64, Vec<[f64; 3]>) {
    if let Some((real, map)) = crate::disp::without_ghosts(mol) {
        let (e, g) = srb_energy_gradient(&real, params);
        return (e, crate::disp::scatter_gradient(g, &map, mol.len()));
    }
    srb_impl(mol, params, true)
}

fn srb_impl(mol: &Molecule, p: &SrbParams, grad: bool) -> (f64, Vec<[f64; 3]>) {
    let z = atom_numbers(mol);
    let xyz: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    let nat = z.len();

    let mut energy = 0.0;
    let mut gradient = vec![[0.0; 3]; nat];

    for i in 0..nat {
        for j in 0..i {
            let vec = [
                xyz[i][0] - xyz[j][0],
                xyz[i][1] - xyz[j][1],
                xyz[i][2] - xyz[j][2],
            ];
            let r = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
            if r > THR_R {
                continue;
            }
            let r0 = p.gamma / crate::disp::data::r0ab_bohr(z[i], z[j]);
            let ff = -((z[i] * z[j]) as f64).sqrt();
            let e = p.s * ff * (-r0 * r).exp();
            energy += e;
            if grad {
                let de_dr = -r0 * e;
                for (k, v) in vec.iter().enumerate() {
                    let g = de_dr * v / r;
                    gradient[i][k] += g;
                    gradient[j][k] -= g;
                }
            }
        }
    }
    (energy, gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Element, Molecule};

    fn mol(atoms: &[(&str, [f64; 3])]) -> Molecule {
        Molecule::new(
            atoms
                .iter()
                .map(|(s, p)| Atom::new(Element::from_symbol(s).unwrap(), *p))
                .collect(),
            0,
            1,
        )
    }

    #[test]
    fn pair_energy_matches_closed_form() {
        let r = 1.8;
        let m = mol(&[("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, r])]);
        let p = SrbParams::B97_3C;
        let e = srb_energy(&m, &p);
        let r0 = crate::disp::data::r0ab_bohr(8, 1);
        let expect = -p.s * (8.0f64).sqrt() * (-p.gamma * r / r0).exp();
        assert!((e - expect).abs() < 1e-15, "{e} vs {expect}");
        assert!(e < 0.0, "SRB is attractive (negative) by construction");
    }

    #[test]
    fn decays_smoothly_to_zero_at_long_range() {
        let p = SrbParams::B97_3C;
        let mut last = f64::NEG_INFINITY;
        for r in [1.0, 2.0, 4.0, 8.0, 16.0] {
            let e = srb_energy(&mol(&[("C", [0.0; 3]), ("N", [r, 0.0, 0.0])]), &p);
            assert!(e < 0.0 && e > last, "monotone rise to 0 at r = {r}");
            last = e;
        }
        let far = srb_energy(&mol(&[("C", [0.0; 3]), ("N", [31.0, 0.0, 0.0])]), &p);
        assert_eq!(far, 0.0, "beyond the 30 bohr cutoff the term is zero");
    }

    #[test]
    fn size_consistent_over_separated_fragments() {
        let p = SrbParams::B97_3C;
        let a = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 1.8])];
        let b = [("N", [0.0, 0.0, 0.0]), ("H", [0.0, 1.9, 0.0])];
        let ea = srb_energy(&mol(&a), &p);
        let eb = srb_energy(&mol(&b), &p);
        let shifted: Vec<_> = b
            .iter()
            .map(|(s, q)| (*s, [q[0] + 100.0, q[1], q[2]]))
            .collect();
        let both = mol(&[&a[..], &shifted[..]].concat());
        let e = srb_energy(&both, &p);
        assert!((e - (ea + eb)).abs() < 1e-15, "{e} vs {}", ea + eb);
    }

    #[test]
    fn gradient_has_no_net_force() {
        let m = mol(&[
            ("O", [0.0, 0.1, 0.2]),
            ("H", [0.0, 1.5, -0.9]),
            ("H", [0.3, -1.4, -0.9]),
        ]);
        let (_, g) = srb_energy_gradient(&m, &SrbParams::B97_3C);
        for k in 0..3usize {
            let net: f64 = g.iter().map(|gi| gi[k]).sum();
            assert!(net.abs() < 1e-15, "net force component {k}: {net}");
        }
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let base = [
            ("O", [0.0, 0.12, 0.31]),
            ("H", [0.05, 1.52, -0.93]),
            ("C", [2.31, -1.42, -0.88]),
            ("H", [2.95, -1.40, 0.84]),
        ];
        let p = SrbParams::B97_3C;
        let (_, g) = srb_energy_gradient(&mol(&base), &p);
        let h = 1e-6;
        let mut worst = 0.0f64;
        for (i, gi) in g.iter().enumerate() {
            for (k, gik) in gi.iter().enumerate() {
                let mut plus = base;
                plus[i].1[k] += h;
                let mut minus = base;
                minus[i].1[k] -= h;
                let fd = (srb_energy(&mol(&plus), &p) - srb_energy(&mol(&minus), &p)) / (2.0 * h);
                worst = worst.max((gik - fd).abs());
            }
        }
        assert!(worst < 1e-10, "worst FD deviation {worst:.3e}");
    }
}
