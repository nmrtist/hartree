use hartree::solv::{CpcmSolver, build_surface, f_epsilon, interaction_energy};

fn born_cpcm(q: f64, r: f64, eps: f64, ng: usize) -> f64 {
    let surface = build_surface(&[[0.0, 0.0, 0.0]], &[r], ng).unwrap();
    let solver = CpcmSolver::new(&surface, eps).unwrap();
    let v: Vec<f64> = surface
        .zeta
        .iter()
        .map(|&z| q * libm::erf(z * r) / r)
        .collect();
    let qs = solver.charges(&v);
    interaction_energy(&qs, &v)
}

#[test]
fn born_ion_converges_to_closed_form() {
    let (q, r, eps) = (1.0, 3.0, 78.3553);
    let exact = -f_epsilon(eps) * q * q / (2.0 * r);

    let mut prev_err = f64::INFINITY;
    for ng in [110, 302, 590] {
        let e = born_cpcm(q, r, eps, ng);
        let err = (e - exact).abs();
        assert!(
            err < prev_err,
            "Born error must decrease with Lebedev order: ng={ng}, err={err:.3e} vs {prev_err:.3e}"
        );
        prev_err = err;
    }
    assert!(
        prev_err < 1e-6 * exact.abs(),
        "Born residual at ng=590: {prev_err:.3e}"
    );
}

#[test]
fn born_scales_with_charge_radius_and_epsilon() {
    let e1 = born_cpcm(1.0, 3.0, 78.3553, 302);
    let e2 = born_cpcm(2.0, 3.0, 78.3553, 302);
    assert!((e2 / e1 - 4.0).abs() < 1e-10);

    let e_half = born_cpcm(1.0, 6.0, 78.3553, 302);
    assert!((e1 / e_half - 2.0).abs() < 1e-6);

    let e_weak = born_cpcm(1.0, 3.0, 1.0 + 1e-9, 302);
    assert!(e_weak.abs() < 1e-9);
    assert!(born_cpcm(1.0, 3.0, 80.0, 302) < born_cpcm(1.0, 3.0, 2.0, 302));
    assert!(e1 < 0.0);
}
