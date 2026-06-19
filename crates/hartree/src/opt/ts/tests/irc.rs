use super::*;
use crate::core::{Atom, Element, Molecule};
use crate::opt::ts::{TsOptions, TsStatus, find_transition_state};

#[test]
fn irc_endpoints_separate_and_descend() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            start[a][c] += 0.05 * basis[0][3 * a + c];
        }
    }
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    opts.confirm_irc = true;
    let result = find_transition_state(&h3_molecule(&start), &mut surf, &opts, None).unwrap();
    assert_eq!(result.status, TsStatus::Converged);
    let irc = result.irc.expect("IRC requested");
    let mut sep = 0.0;
    for (f, r) in irc.forward.iter().zip(&irc.reverse) {
        for c in 0..3 {
            sep += (f[c] - r[c]).powi(2);
        }
    }
    assert!(sep.sqrt() > 0.1, "endpoints too close: {}", sep.sqrt());
    assert!(irc.forward_energy <= result.energy + 1e-9);
    assert!(irc.reverse_energy <= result.energy + 1e-9);
}

/// A heteronuclear (H, C, O) saddle with an explicit opposite-projection check:
/// the two IRC endpoints must land on opposite sides of the saddle along the
/// reaction mode, and both downhill of it.
#[test]
fn irc_endpoints_split_reaction_coordinate_heteronuclear() {
    let x0 = h3_positions();
    let basis = internal_basis(&x0);
    let h = hessian_from(&basis, &[-0.4, 0.6, 0.9]);
    let mut start = x0.clone();
    for a in 0..3 {
        for c in 0..3 {
            start[a][c] += 0.05 * basis[0][3 * a + c];
        }
    }
    // H, C, O on the bent geometry, started from the tiny displacement.
    let start_mol = Molecule::new(
        [1u32, 6, 8]
            .iter()
            .zip(&start)
            .map(|(&z, &p)| Atom::new(Element::from_z(z).unwrap(), p))
            .collect(),
        0,
        1,
    );
    let mut surf = Quadratic { x0: x0.clone(), h };
    let mut opts = TsOptions::default();
    opts.confirm_irc = true;
    let result = find_transition_state(&start_mol, &mut surf, &opts, None).unwrap();
    assert_eq!(result.status, TsStatus::Converged);
    let irc = result.irc.as_ref().expect("IRC requested");

    let mode = result
        .verification
        .as_ref()
        .unwrap()
        .reaction_mode
        .as_ref()
        .unwrap();
    let saddle = &result.positions;
    let fwd = &irc.forward;
    let rev = &irc.reverse;

    let project = |end: &[[f64; 3]]| -> f64 {
        end.iter()
            .zip(saddle)
            .zip(mode)
            .map(|((e, s), m)| (0..3).map(|c| (e[c] - s[c]) * m[c]).sum::<f64>())
            .sum()
    };
    let proj_fwd = project(fwd);
    let proj_rev = project(rev);
    assert!(
        proj_fwd * proj_rev < 0.0,
        "endpoints on same side: fwd {proj_fwd}, rev {proj_rev}"
    );

    assert!(irc.forward_energy <= result.energy + 1e-9);
    assert!(irc.reverse_energy <= result.energy + 1e-9);

    let mut sep = 0.0;
    for (f, r) in fwd.iter().zip(rev) {
        for c in 0..3 {
            sep += (f[c] - r[c]).powi(2);
        }
    }
    assert!(sep.sqrt() > 0.1, "endpoints too close: {}", sep.sqrt());
}
